"""
학습 스크립트 (로컬 GPU에서 실행 -> model/ 산출 -> 제출 zip에 그대로 포함).

실행:
  python train.py --data ./train.csv

산출물 (model/):
  - config.json, model.safetensors, tokenizer 파일들  (HF 표준)
  - label_maps.json        : label2id / id2label
  - serialize_config.json  : 직렬화 마커 (추론과 동일 보장)
  - thresholds.json        : (옵션) 클래스별 Macro-F1 최적 임계값
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional

import config as C
from common import serialize, serialize_config_from, build_label_maps


@dataclass
class HierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    coarse_logits: torch.FloatTensor = None


class HierClassifier(nn.Module):
    """계층(멀티태스크) 분류: 인코더 공유 + fine(14) 헤드 + coarse(4) 보조 헤드.
    coarse는 학습 시 regularizer, 추론은 fine 헤드만 사용."""
    def __init__(self, backbone_name, num_labels, n_coarse, id2label, label2id, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fine_head = nn.Linear(h, num_labels)
        self.coarse_head = nn.Linear(h, n_coarse)
        self.num_labels = num_labels
        self.config = self.backbone.config
        self.config.num_labels = num_labels
        self.config.id2label = id2label
        self.config.label2id = label2id

    def _pool(self, H, mask):
        m = mask.unsqueeze(-1).float()
        return (H * m).sum(1) / m.sum(1).clamp(min=1e-6)   # masked mean

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, **kw):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and getattr(self.config, "type_vocab_size", 0):
            kwargs["token_type_ids"] = token_type_ids
        H = self.backbone(**kwargs).last_hidden_state
        pooled = self.dropout(self._pool(H, attention_mask))
        fine = self.fine_head(pooled)
        if self.training:
            return HierOutput(logits=fine, coarse_logits=self.coarse_head(pooled))
        return SequenceClassifierOutput(logits=fine)


# ---------------------------------------------------------------------------
# Attention Pooling 분류기 (--pool attn): [CLS] 대신 학습형 attention 가중합
# ---------------------------------------------------------------------------
class AttnPoolClassifier(nn.Module):
    """backbone hidden states 에 학습 query로 softmax attention -> 가중합 -> 분류.
    모델이 발화·직전 action 등 정보량 큰 토큰에 스스로 가중치를 주도록 유도."""
    def __init__(self, backbone_name, num_labels, id2label, label2id, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        h = self.backbone.config.hidden_size
        self.attn = nn.Linear(h, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h, num_labels)
        self.num_labels = num_labels
        # HF 호환용 config (id2label 등 저장/로드 편의)
        self.config = self.backbone.config
        self.config.num_labels = num_labels
        self.config.id2label = id2label
        self.config.label2id = label2id

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, **kw):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and getattr(self.config, "type_vocab_size", 0):
            kwargs["token_type_ids"] = token_type_ids
        out = self.backbone(**kwargs)
        H = out.last_hidden_state                         # (B,T,h)
        scores = self.attn(H).squeeze(-1)                 # (B,T)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)    # (B,T,1)
        pooled = (H * w).sum(dim=1)                        # (B,h)
        logits = self.classifier(self.dropout(pooled))
        return SequenceClassifierOutput(logits=logits)


# ---------------------------------------------------------------------------
# 데이터 로딩 (PLACEHOLDER: 내일 실제 포맷에 맞게 조정)
# ---------------------------------------------------------------------------
def _maybe_json(x):
    """CSV 셀 안에 JSON 문자열로 들어온 history/meta를 파싱."""
    if isinstance(x, str):
        s = x.strip()
        if s and s[0] in "[{":
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


def load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    elif path.endswith(".json"):
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)
    for col in (C.COLS.history, C.COLS.session_meta):
        if col in df.columns:
            df[col] = df[col].map(_maybe_json)
    return df


def to_text(df: pd.DataFrame, include_history: bool = True) -> list:
    ser_cfg = serialize_config_from(C.SER)
    out = []
    for _, r in df.iterrows():
        hist = r.get(C.COLS.history) if include_history else None  # prompt-중심 실험용
        out.append(
            serialize(
                r.get(C.COLS.current_prompt),
                hist,
                r.get(C.COLS.session_meta),
                ser_cfg,
            )
        )
    return out


# ---------------------------------------------------------------------------
# 클래스 불균형 대응 Trainer
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


@dataclass
class SupConOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    proj: torch.FloatTensor = None


class SupConClassifier(nn.Module):
    """인코더 공유 + 분류헤드 + projection헤드. SupCon으로 같은 클래스 표현을 뭉치게."""
    def __init__(self, backbone_name, num_labels, id2label, label2id, proj_dim=128, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h, num_labels)
        self.proj = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, proj_dim))
        self.num_labels = num_labels
        self.config = self.backbone.config
        self.config.num_labels = num_labels
        self.config.id2label = id2label
        self.config.label2id = label2id

    def _pool(self, H, mask):
        m = mask.unsqueeze(-1).float()
        return (H * m).sum(1) / m.sum(1).clamp(min=1e-6)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kw):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and getattr(self.config, "type_vocab_size", 0):
            kwargs["token_type_ids"] = token_type_ids
        pooled = self._pool(self.backbone(**kwargs).last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        if self.training:
            z = nn.functional.normalize(self.proj(pooled), dim=1)
            return SupConOutput(logits=logits, proj=z)
        return SequenceClassifierOutput(logits=logits)


def supcon_loss(feats, labels, temp=0.1):
    """Supervised Contrastive: 같은 클래스 당기고 다른 클래스 밀어냄."""
    device = feats.device
    B = feats.shape[0]
    sim = (feats @ feats.T) / temp
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()   # 안정화
    self_mask = torch.eye(B, device=device)
    exp_sim = torch.exp(sim) * (1 - self_mask)
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * (1 - self_mask)
    pos_cnt = pos.sum(1)
    loss = -(pos * log_prob).sum(1) / pos_cnt.clamp(min=1)
    loss = loss[pos_cnt > 0]
    return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=device)


class FGM:
    """Fast Gradient Method — 워드 임베딩에 grad 방향 perturbation을 줘 적대학습."""
    def __init__(self, model, emb_name="word_embeddings", eps=1.0):
        self.model = model
        self.emb_name = emb_name
        self.eps = eps
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.eps * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class AWP:
    """Adversarial Weight Perturbation — 가중치를 grad 방향으로 eps-ball 내 교란해 적대학습.
    FGM(입력 교란)보다 강함. 저신뢰 경계 일반화에 효과."""
    def __init__(self, model, adv_param="weight", adv_lr=1e-4, adv_eps=1e-2):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def perturb(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None and self.adv_param in n:
                if n not in self.backup:
                    self.backup[n] = p.data.clone()
                    eps = self.adv_eps * p.abs().detach()
                    self.backup_eps[n] = (self.backup[n] - eps, self.backup[n] + eps)
                norm_g = torch.norm(p.grad)
                norm_d = torch.norm(p.detach())
                if norm_g != 0 and not torch.isnan(norm_g):
                    r = self.adv_lr * p.grad / (norm_g + 1e-12) * (norm_d + 1e-12)
                    p.data.add_(r)
                    p.data = torch.min(torch.max(p.data, self.backup_eps[n][0]),
                                       self.backup_eps[n][1])

    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data = self.backup[n]
        self.backup = {}
        self.backup_eps = {}


class ImbalancedTrainer(Trainer):
    def __init__(self, *args, loss_type="weighted_ce", class_weights=None, focal_gamma=2.0,
                 use_fgm=False, fgm_eps=1.0, kd_alpha=0.0, kd_temp=2.0,
                 hier_lambda=0.0, fine2coarse=None, supcon_w=0.0, supcon_temp=0.1,
                 use_awp=False, awp_lr=1e-4, awp_eps=1e-2, awp_start_epoch=1,
                 label_smoothing=0.0, **kw):
        super().__init__(*args, **kw)
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self._cw = class_weights
        self.use_fgm = use_fgm
        self.fgm = FGM(self.model, eps=fgm_eps) if use_fgm else None
        self.use_awp = use_awp
        self.awp = AWP(self.model, adv_lr=awp_lr, adv_eps=awp_eps) if use_awp else None
        self.awp_start_epoch = awp_start_epoch
        self.label_smoothing = label_smoothing
        self.kd_alpha = kd_alpha      # 증류 가중치 (0=순수 CE)
        self.kd_temp = kd_temp        # temperature
        self.hier_lambda = hier_lambda            # 계층 coarse 보조손실 가중치
        self.fine2coarse = fine2coarse            # (14,) fine id -> coarse id
        self.supcon_w = supcon_w                  # SupCon 가중치
        self.supcon_temp = supcon_temp

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        # non-destructive: FGM이 같은 배치로 두 번 호출하므로 inputs를 변형하지 않음
        labels = inputs["labels"]
        teacher = inputs.get("teacher_logits")
        model_inputs = {k: v for k, v in inputs.items()
                        if k not in ("labels", "teacher_logits")}
        outputs = model(**model_inputs)
        logits = outputs.logits
        w = self._cw.to(logits.device) if self._cw is not None else None
        ls = self.label_smoothing
        if self.loss_type == "focal":
            ce = FocalLoss(self.focal_gamma, weight=w)(logits, labels)
        elif self.loss_type == "weighted_ce":
            ce = nn.functional.cross_entropy(logits, labels, weight=w, label_smoothing=ls)
        else:
            ce = nn.functional.cross_entropy(logits, labels, label_smoothing=ls)
        # Knowledge Distillation: KL(student ‖ teacher) at temperature T
        if self.kd_alpha > 0 and teacher is not None:
            T = self.kd_temp
            kd = nn.functional.kl_div(
                nn.functional.log_softmax(logits / T, dim=1),
                nn.functional.softmax(teacher.to(logits.device) / T, dim=1),
                reduction="batchmean") * (T * T)
            loss = self.kd_alpha * kd + (1.0 - self.kd_alpha) * ce
        else:
            loss = ce
        # 계층 보조손실: coarse 라벨은 fine 라벨에서 결정론적으로 유도
        coarse_logits = getattr(outputs, "coarse_logits", None)
        if self.hier_lambda > 0 and coarse_logits is not None and self.fine2coarse is not None:
            coarse_labels = self.fine2coarse.to(labels.device)[labels]
            loss = loss + self.hier_lambda * nn.functional.cross_entropy(coarse_logits, coarse_labels)
        # SupCon: projection 임베딩으로 같은 클래스 뭉치기
        proj = getattr(outputs, "proj", None)
        if self.supcon_w > 0 and proj is not None:
            loss = loss + self.supcon_w * supcon_loss(proj, labels, self.supcon_temp)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not (self.use_fgm or self.use_awp):
            return super().training_step(model, inputs, num_items_in_batch)
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)          # 정상 grad
        # AWP는 warmup 이후(awp_start_epoch)부터, 그 전엔 FGM warmup
        epoch = self.state.epoch or 0.0
        if self.use_awp and epoch >= self.awp_start_epoch:
            self.awp.perturb()                    # 가중치 교란
            with self.compute_loss_context_manager():
                loss_adv = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss_adv = loss_adv.mean()
            self.accelerator.backward(loss_adv)
            self.awp.restore()
        elif self.use_fgm:
            self.fgm.attack()                     # 임베딩 교란(warmup)
            with self.compute_loss_context_manager():
                loss_adv = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss_adv = loss_adv.mean()
            self.accelerator.backward(loss_adv)
            self.fgm.restore()
        return loss.detach()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


# ---------------------------------------------------------------------------
# 클래스별 threshold 튜닝 (Macro-F1 최대화)
# ---------------------------------------------------------------------------
def tune_thresholds(probs: np.ndarray, y_true: np.ndarray, n_classes: int):
    """
    각 클래스 c의 logit/확률에 가산 bias를 줘서 소수 클래스 recall을 끌어올림.
    간단한 좌표하강: 클래스별로 bias 후보를 스윕하며 Macro-F1 개선되면 채택.
    """
    bias = np.zeros(n_classes)

    def macro(b):
        preds = np.argmax(probs + b, axis=1)
        return f1_score(y_true, preds, average="macro")

    best = macro(bias)
    for _ in range(3):  # 몇 번 반복
        for c in range(n_classes):
            for cand in np.linspace(-0.3, 0.3, 13):
                trial = bias.copy()
                trial[c] = cand
                s = macro(trial)
                if s > best:
                    best, bias = s, trial
    return bias.tolist(), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="학습 데이터 경로 (csv/parquet/jsonl)")
    # 빠른 실험용 override (없으면 config.py 값 사용)
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="학습 샘플 수 제한 (스모크 테스트)")
    ap.add_argument("--tag", default=None, help="모델 태그 -> runs/<tag>/ 에 저장(앙상블용)")
    ap.add_argument("--pool", choices=["cls", "attn"], default="cls",
                    help="cls=기본 헤드, attn=attention pooling 헤드")
    ap.add_argument("--split", choices=["group", "random"], default="group",
                    help="group=세션 단위 분리(누수 방지, 기본), random=단순 stratified")
    ap.add_argument("--fgm", action="store_true", help="FGM 적대학습 on")
    ap.add_argument("--fgm_eps", type=float, default=1.0)
    ap.add_argument("--teacher_logits", default=None, help="교사 logits npz (증류)")
    ap.add_argument("--kd_alpha", type=float, default=0.5, help="증류 가중치")
    ap.add_argument("--kd_temp", type=float, default=2.0, help="증류 temperature")
    ap.add_argument("--hierarchical", action="store_true", help="계층(멀티태스크) 분류 on")
    ap.add_argument("--hier_lambda", type=float, default=0.3, help="coarse 보조손실 가중치")
    ap.add_argument("--no_history", action="store_true", help="history 제외(current_prompt+meta만)")
    ap.add_argument("--supcon", action="store_true", help="SupCon contrastive 보조손실 on")
    ap.add_argument("--supcon_weight", type=float, default=0.3)
    ap.add_argument("--supcon_temp", type=float, default=0.1)
    # 입력 실험 (블록별 ablation — 동시에 다 켜지 말고 하나씩)
    ap.add_argument("--add_cues", action="store_true", help="판별 피처(path/sym/dir/glob) CUE")
    ap.add_argument("--add_openfiles", action="store_true", help="meta에 open_files 경로")
    ap.add_argument("--add_langmix", action="store_true", help="meta에 language_mix")
    ap.add_argument("--bf16", action="store_true",
                    help="bf16 학습 (bf16으로 로드되는 모델용, granite 등 — fp16 GradScaler 충돌 회피)")
    ap.add_argument("--awp", action="store_true", help="AWP 적대학습(가중치 교란). FGM warmup 후 발동")
    ap.add_argument("--awp_lr", type=float, default=1e-4)
    ap.add_argument("--awp_eps", type=float, default=1e-2)
    ap.add_argument("--awp_start_epoch", type=int, default=1, help="이 epoch부터 AWP(그 전엔 FGM)")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="과신 교정(예: 0.1)")
    ap.add_argument("--seed", type=int, default=None,
                    help="모델 seed(weight init/학습) 변경. split은 42 고정 유지 -> 앙상블 정렬 보존")
    ap.add_argument("--lora", action="store_true", help="LoRA 파인튜닝(디코더용)")
    ap.add_argument("--load_4bit", action="store_true", help="4bit 양자화 로드(QLoRA)")
    ap.add_argument("--grad_ckpt", action="store_true", help="grad checkpointing(대형용)")
    args = ap.parse_args()
    MODEL_SEED = args.seed if args.seed is not None else C.TRAIN.seed

    if args.add_cues:      C.SER.add_symbol_cues = True; print("[input] +symbol_cues(path/sym/dir/glob)")
    if args.add_openfiles: C.SER.add_open_files = True;  print("[input] +open_files")
    if args.add_langmix:   C.SER.add_langmix = True;     print("[input] +langmix")
    if args.model:      C.MODEL_NAME = args.model
    if args.max_length: C.MAX_LENGTH = args.max_length
    if args.epochs:     C.TRAIN.epochs = args.epochs
    if args.batch_size: C.TRAIN.train_batch_size = args.batch_size
    if args.tag:        C.TRAIN.output_dir = os.path.join("runs", args.tag)
    # CPU 환경에서는 fp16/bf16 불가 -> 자동 비활성화
    if not torch.cuda.is_available():
        C.TRAIN.fp16 = False
        args.bf16 = False
        print("[info] no CUDA -> fp16 disabled (CPU run)")

    os.makedirs(C.TRAIN.output_dir, exist_ok=True)
    torch.manual_seed(MODEL_SEED)   # 모델 seed만 변경 (split은 C.TRAIN.seed=42 고정)
    np.random.seed(C.TRAIN.seed)    # 데이터/split 재현성은 고정 seed 유지

    df = load_dataframe(args.data)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=C.TRAIN.seed).reset_index(drop=True)
    print(f"[load] {len(df)} rows, columns={list(df.columns)}")

    texts = to_text(df, include_history=not args.no_history)
    if args.no_history:
        print("[input] history 제외 (current_prompt + meta 만)")
    raw_labels = df[C.COLS.label].astype(str).tolist()
    label2id, id2label = build_label_maps(raw_labels, C.CLASS_NAMES)
    n_classes = len(label2id)
    y = np.array([label2id[l] for l in raw_labels])
    print(f"[labels] {n_classes} classes")
    print(pd.Series(raw_labels).value_counts())

    # 증류: 교사 logits 로드 -> df 순서로 정렬
    teacher_arr = None
    if args.teacher_logits:
        z = np.load(args.teacher_logits, allow_pickle=True)
        tmap = {str(i): v for i, v in zip(z["ids"], z["logits"])}
        cur_ids = df[C.COLS.id].astype(str).values
        teacher_arr = np.stack([tmap[i] for i in cur_ids]).astype(np.float32)
        print(f"[kd] teacher logits {teacher_arr.shape}, alpha={args.kd_alpha}, T={args.kd_temp}")

    # 세션 단위 그룹 분리(기본): 같은 세션(id=sess_..._step_XX)이 train/val에 갈리는 누수 방지
    if args.split == "group":
        from sklearn.model_selection import StratifiedGroupKFold
        groups = np.array([re.sub(r"-step_\d+$", "", str(i))
                           for i in df[C.COLS.id].astype(str).values])
        n_splits = max(2, round(1.0 / C.TRAIN.val_size))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=C.TRAIN.seed)
        tr_idx, va_idx = next(sgkf.split(np.arange(len(texts)), y, groups))
        overlap = set(groups[tr_idx]) & set(groups[va_idx])
        print(f"[split] session-grouped: val {len(va_idx)} samples / "
              f"{len(set(groups[va_idx]))} sessions, overlap sessions={len(overlap)}")
    else:
        can_stratify = np.bincount(y, minlength=n_classes).min() >= 2
        tr_idx, va_idx = train_test_split(
            np.arange(len(texts)), test_size=C.TRAIN.val_size,
            stratify=(y if can_stratify else None), random_state=C.TRAIN.seed,
        )
        print("[split] random stratified"
              + ("" if can_stratify else " (some class <2 -> no stratify)"))

    tok = AutoTokenizer.from_pretrained(C.MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(idx, with_teacher=False):
        enc = tok([texts[i] for i in idx], truncation=True,
                  max_length=C.MAX_LENGTH, padding=False)
        enc["labels"] = [int(y[i]) for i in idx]
        if with_teacher and teacher_arr is not None:
            enc["teacher_logits"] = [teacher_arr[i] for i in idx]  # train에만
        return enc

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc
        def __len__(self):
            return len(self.enc["labels"])
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}

    train_ds, val_ds = DS(encode(tr_idx, with_teacher=True)), DS(encode(va_idx))

    # 클래스 가중치 = inverse frequency
    counts = np.bincount(y[tr_idx], minlength=n_classes).astype(float)
    cw = torch.tensor((counts.sum() / (n_classes * np.maximum(counts, 1))), dtype=torch.float)

    # 계층 분류용 fine->coarse 매핑
    fine2coarse = None
    if args.hierarchical:
        f2c = {}
        for cid, (_, members) in enumerate(C.COARSE_GROUPS):
            for name in members:
                f2c[label2id[name]] = cid
        fine2coarse = torch.tensor([f2c[i] for i in range(n_classes)], dtype=torch.long)
        print(f"[hier] coarse groups={len(C.COARSE_GROUPS)}, lambda={args.hier_lambda}")

    is_custom = args.hierarchical or args.pool == "attn" or args.supcon or args.lora   # 커스텀/LoRA
    if args.supcon:
        model = SupConClassifier(C.MODEL_NAME, n_classes, id2label, label2id)
    elif args.hierarchical:
        model = HierClassifier(C.MODEL_NAME, n_classes, len(C.COARSE_GROUPS), id2label, label2id)
    elif args.pool == "attn":
        model = AttnPoolClassifier(C.MODEL_NAME, n_classes, id2label, label2id)
    else:
        _kw = dict(num_labels=n_classes, id2label=id2label, label2id=label2id, trust_remote_code=True)
        if args.load_4bit:
            from transformers import BitsAndBytesConfig
            _kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            _kw["device_map"] = {"": 0}
        model = AutoModelForSequenceClassification.from_pretrained(C.MODEL_NAME, **_kw)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tok.pad_token_id
        if args.lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if args.load_4bit:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            _lc = LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                modules_to_save=["score"])
            model = get_peft_model(model, _lc); model.print_trainable_parameters()

    from transformers import DataCollatorWithPadding
    _base_collator = DataCollatorWithPadding(tokenizer=tok)

    def collator(features):
        # teacher_logits(고정 14차원)를 패딩 대상에서 빼고 따로 stack
        teacher = None
        if "teacher_logits" in features[0]:
            teacher = [f.pop("teacher_logits") for f in features]
        batch = _base_collator(features)
        if teacher is not None:
            batch["teacher_logits"] = torch.tensor(np.array(teacher), dtype=torch.float)
        return batch

    targs = TrainingArguments(
        output_dir=os.path.join(C.TRAIN.output_dir, "_hf_ckpt"),
        num_train_epochs=C.TRAIN.epochs,
        per_device_train_batch_size=C.TRAIN.train_batch_size,
        per_device_eval_batch_size=C.TRAIN.eval_batch_size,
        learning_rate=C.TRAIN.lr,
        weight_decay=C.TRAIN.weight_decay,
        warmup_ratio=C.TRAIN.warmup_ratio,
        fp16=(C.TRAIN.fp16 and not args.bf16),
        bf16=args.bf16,
        eval_strategy="epoch",
        # 커스텀 nn.Module(attn/hier)은 Trainer 체크포인트 reload가 까다로워 save/best 끔
        save_strategy=("epoch" if not is_custom else "no"),
        load_best_model_at_end=(not is_custom),
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=MODEL_SEED,
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_ckpt else None,
    )

    trainer = ImbalancedTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok, data_collator=collator,
        compute_metrics=compute_metrics,
        loss_type=C.TRAIN.loss_type, class_weights=cw, focal_gamma=C.TRAIN.focal_gamma,
        use_fgm=args.fgm, fgm_eps=args.fgm_eps,
        use_awp=args.awp, awp_lr=args.awp_lr, awp_eps=args.awp_eps,
        awp_start_epoch=args.awp_start_epoch, label_smoothing=args.label_smoothing,
        kd_alpha=(args.kd_alpha if teacher_arr is not None else 0.0), kd_temp=args.kd_temp,
        hier_lambda=(args.hier_lambda if args.hierarchical else 0.0), fine2coarse=fine2coarse,
        supcon_w=(args.supcon_weight if args.supcon else 0.0), supcon_temp=args.supcon_temp,
    )
    if args.supcon:
        print(f"[supcon] contrastive ON (w={args.supcon_weight}, temp={args.supcon_temp})")
    if args.fgm:
        print(f"[fgm] adversarial training ON (eps={args.fgm_eps})")
    trainer.train()

    # 검증 리포트 + threshold 튜닝
    pred = trainer.predict(val_ds)
    probs = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()
    y_val = y[va_idx]
    print(classification_report(y_val, probs.argmax(1),
                                labels=list(range(n_classes)),
                                target_names=[id2label[i] for i in range(n_classes)],
                                zero_division=0))

    bias = [0.0] * n_classes
    if C.TRAIN.tune_thresholds:
        bias, best = tune_thresholds(probs, y_val, n_classes)
        print(f"[threshold] tuned macro_f1 = {best:.4f} "
              f"(base = {f1_score(y_val, probs.argmax(1), average='macro'):.4f})")

    # ---- 최종 산출물 저장 (model/) ----
    out = C.TRAIN.output_dir
    os.makedirs(out, exist_ok=True)
    if is_custom:
        kind = "supcon" if args.supcon else ("hier" if args.hierarchical else "attn")
        torch.save(model.state_dict(), os.path.join(out, f"{kind}_state.pt"))
        model.config.save_pretrained(out)          # backbone arch/config
        json.dump({"pool": kind, "backbone": C.MODEL_NAME}, open(os.path.join(out, "pool.json"), "w"))
    else:
        trainer.save_model(out)
    tok.save_pretrained(out)
    json.dump({"label2id": label2id, "id2label": id2label},
              open(os.path.join(out, "label_maps.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(serialize_config_from(C.SER),
              open(os.path.join(out, "serialize_config.json"), "w"), ensure_ascii=False, indent=2)
    json.dump({"bias": bias, "max_length": C.MAX_LENGTH},
              open(os.path.join(out, "thresholds.json"), "w"), ensure_ascii=False, indent=2)

    # ---- 앙상블용: 동일 holdout에 대한 val 확률 저장 (id 정렬 -> 스태킹) ----
    val_ids = df.iloc[va_idx][C.COLS.id].astype(str).values
    np.savez(os.path.join(out, "val_probs.npz"),
             ids=val_ids, probs=probs, y=y_val,
             classes=np.array([id2label[i] for i in range(n_classes)]))
    print(f"[done] saved to {out}/  (+ val_probs.npz for stacking)")


if __name__ == "__main__":
    main()
