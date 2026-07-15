#!/usr/bin/env python
"""교사 1개 구성(모델×시드) 5-fold 학습 → OOF/holdout 확률만 npz 저장(~8MB).

가중치는 저장 안 함(증류에는 확률만 필요) → 다운로드 초경량, 60분 세션에 최적.
env: AD_MODEL, AD_SEED, AD_VERSION, AD_MAXLEN, AD_EPOCHS, AD_LR, AD_BATCH,
     AD_FP16(1/0), AD_LLRD, AD_FGM, AD_FOLD_LO, AD_FOLD_HI(대형모델 분할용), AD_TAG.
출력: /content/teacher_<TAG>.npz + DONE_<TAG>
"""
import os, sys, subprocess, time, zipfile, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
WORK = os.environ.get("AD_WORK", "/content" if os.path.isdir("/content") else os.getcwd())
os.chdir(WORK)
# [재현 패치] 런타임 pip install 비활성화 — intern_aichallenge(4.53.3, mmbert용) 다운그레이드 방지.
#   xlm-roberta-large는 4.53.3에서 동일 동작. 원본: transformers==4.46.3 accelerate==1.9.0 sentencepiece==0.1.99
pass
for z in ["open.zip", "ad_common.zip"]:
    if os.path.exists(z):
        with zipfile.ZipFile(z) as f:
            f.extractall(".")
sys.path.insert(0, WORK)

import numpy as np, torch
# 외부 GPU 침입 대응(07-08): 시작 즉시 VRAM 선점 예약 — 캐싱 할당자에 보존되어 이후 학습이 그 안에서 돎
_PRE = float(os.environ.get("AD_PREALLOC_GB", "0"))
if _PRE > 0 and torch.cuda.is_available():
    _t = torch.empty(int(_PRE * 1e9 // 2), dtype=torch.float16, device="cuda")
    del _t
    print(f"[prealloc] {_PRE}GB 예약 완료", flush=True)
from common.io_utils import load_train, CLASSES, NUM_CLASSES, set_seed
from common.cv import make_splits
from common.metrics import macro_f1
from common import ad_lib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler

MODEL = os.environ.get("AD_MODEL", "xlm-roberta-base")
SEED = int(os.environ.get("AD_SEED", "1234"))
VERSION = os.environ.get("AD_VERSION", "v5")
MAX_LEN = int(os.environ.get("AD_MAXLEN", "320"))
EPOCHS = int(os.environ.get("AD_EPOCHS", "4"))
LR = float(os.environ.get("AD_LR", "3e-5"))
BATCH = int(os.environ.get("AD_BATCH", "128"))
FP16 = os.environ.get("AD_FP16", "1") == "1"
LLRD = os.environ.get("AD_LLRD", "1") == "1"
FGM_ON = os.environ.get("AD_FGM", "0") == "1"
FGM_EMB = os.environ.get("AD_FGM_EMB", "word_embeddings")  # [적응] ModernBERT=tok_embeddings, XLM-R=word_embeddings(기본)
AWP_ON = os.environ.get("AD_AWP", "0") == "1"  # 적대적 가중치 섭동(상혁 검증). FGM과 병용 가능
FOLD_LO = int(os.environ.get("AD_FOLD_LO", "0"))
FOLD_HI = int(os.environ.get("AD_FOLD_HI", "5"))
EXCLUDE_AU = os.environ.get("AD_EXCLUDE_AU", "0") == "1"   # sim-only 학습 프로브
SELECT_SIM = os.environ.get("AD_SELECT_SIM", "0") == "1"   # sim-only 나침반용 epoch 선택
RDROP_A = float(os.environ.get("AD_RDROP", "0"))   # >0: R-Drop 2-forward 대칭KL (Liang'21). FGM과 가산적(R-AT). 학습 ~1.7배
EMA_D = float(os.environ.get("AD_EMA", "0"))       # >0(예 0.999): 스텝 EMA 패시브 계측 — 학습역학 불변, npz에 *_ema 병행 저장
TAG = os.environ.get("AD_TAG", f"{MODEL.split('/')[-1]}_s{SEED}_f{FOLD_LO}{FOLD_HI}")
HEAD_SEED = 1234
device = "cuda"; assert torch.cuda.is_available()
print(f"[teacher] {TAG} model={MODEL} v={VERSION} len={MAX_LEN} ep={EPOCHS} lr={LR} "
      f"b={BATCH} fp16={FP16} llrd={LLRD} fgm={FGM_ON} exclude_au={EXCLUDE_AU} "
      f"select_sim={SELECT_SIM} rdrop={RDROP_A} ema={EMA_D} folds=[{FOLD_LO},{FOLD_HI})", flush=True)

set_seed(SEED)
samples, y, ids = load_train()
y = np.array(y); groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
dev_idx, hold_idx = sp["dev_idx"], sp["holdout_idx"]
folds = sp["folds"]
cnt = np.bincount(y[dev_idx], minlength=NUM_CLASSES)
cw = len(dev_idx) / (NUM_CLASSES * np.maximum(cnt, 1)); cw /= cw.mean()

tok = AutoTokenizer.from_pretrained(MODEL); tok.truncation_side = "left"
texts = [ad_lib.serialize(s, VERSION, int(os.environ.get("AD_MHT", "8"))) for s in samples]  # [적응] AD_MHT=12로 history 확장
_PM = os.environ.get("AD_ACTIONS_PREAMBLE", "0")
if _PM == "1":   # 용욱 안: 14개 액션 전체 설명(레이블 grounding)
    _PRE = ("[AVAILABLE ACTIONS] read_file: read an existing file. grep_search: search text or pattern in files. "
            "list_directory: list files in a directory. glob_pattern: find files by a glob pattern. "
            "edit_file: make a targeted edit to an existing file. write_file: create or replace a complete file. "
            "apply_patch: modify files using a structured patch. run_bash: run a shell command. "
            "run_tests: run the test suite. lint_or_typecheck: run linter or type checker. "
            "ask_user: ask for missing information or clarification. plan_task: plan or outline the task. "
            "web_search: search the web. respond_only: respond without taking an action. ")
    texts = [_PRE + t for t in texts]; print("[preamble] full-14 추가", flush=True)
elif _PM == "2":   # 건우 refine: 헷갈리는 4개(list/read/grep/glob)만 설명
    _PRE = ("[HINTS] read_file: read the contents of one specific file. grep_search: search for text/pattern across files. "
            "list_directory: list the files inside a directory. glob_pattern: match files by a wildcard pattern. ")
    texts = [_PRE + t for t in texts]; print("[preamble] confusing-4만 추가", flush=True)
elif _PM == "3":   # 용욱 원안 faithful: 14설명 + [SELECTION TASK] 지시문(긴 버전)
    _PRE = ("[AVAILABLE ACTIONS] read_file: read an existing file. grep_search: search text or pattern in files. "
            "list_directory: list files in a directory. glob_pattern: find files by a glob pattern. "
            "edit_file: make a targeted edit to an existing file. write_file: create or replace a complete file. "
            "apply_patch: modify files using a structured patch. run_bash: run a shell command. "
            "run_tests: run the test suite. lint_or_typecheck: run linter or type checker. "
            "ask_user: ask for missing information or clarification. plan_task: plan or outline the task. "
            "web_search: search the web. respond_only: respond without taking an action. "
            "[SELECTION TASK] Choose exactly one action from the list above that should be taken next. "
            "Use the current query, session metadata, and conversation history as evidence. ")
    texts = [_PRE + t for t in texts]; print("[preamble] 용욱원안(full-14 + SELECTION TASK) 추가", flush=True)
elif _PM == "4":   # 건우: 압축 용욱 — 14개 짧게 + 짧은 task (@512서 덜 잘리게)
    _PRE = ("[ACTIONS] read_file: read a file. grep_search: search text. list_directory: list a folder. "
            "glob_pattern: match files by pattern. edit_file: edit a file. write_file: create a file. "
            "apply_patch: patch files. run_bash: run shell. run_tests: run tests. lint_or_typecheck: lint/typecheck. "
            "ask_user: ask clarification. plan_task: plan. web_search: search web. respond_only: just respond. "
            "[TASK] choose one action to take next. ")
    texts = [_PRE + t for t in texts]; print("[preamble] 압축용욱(14짧게+task) 추가", flush=True)
t0 = time.time()
enc_all = tok(texts, truncation=True, max_length=MAX_LEN, padding=False)
INPUT_IDS = enc_all["input_ids"]
print(f"[tok] {len(texts)} in {time.time()-t0:.0f}s", flush=True)


def build():
    torch.manual_seed(HEAD_SEED)
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=NUM_CLASSES,
        id2label={i: c for i, c in enumerate(CLASSES)},
        label2id={c: i for i, c in enumerate(CLASSES)}).to(device)


def pad_batch(idx_list):
    return tok.pad({"input_ids": [INPUT_IDS[j] for j in idx_list]}, return_tensors="pt")


def infer_probs(model, idx):
    model.eval(); bs = 192
    order = sorted(range(len(idx)), key=lambda k: len(INPUT_IDS[int(idx[k])]))
    out = np.zeros((len(idx), NUM_CLASSES), np.float32)
    with torch.no_grad():
        for b in range(0, len(order), bs):
            ks = order[b:b + bs]; sub = [int(idx[k]) for k in ks]
            enc = pad_batch(sub).to(device)
            if FP16:
                with autocast("cuda", dtype=torch.float16):
                    lg = model(**enc).logits.float()
            else:
                lg = model(**enc).logits.float()
            p = torch.softmax(lg, 1).cpu().numpy()
            for m, k in enumerate(ks):
                out[k] = p[m]
    return out


class FGM:
    def __init__(self, model, eps=1.0):
        self.model, self.eps, self.backup = model, eps, {}
    def attack(self, emb_name="word_embeddings"):
        for n, p in self.model.named_parameters():
            if p.requires_grad and emb_name in n and p.grad is not None:
                self.backup[n] = p.data.clone()
                norm = torch.norm(p.grad)
                if norm and not torch.isnan(norm):
                    p.data.add_(self.eps * p.grad / norm)
    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data = self.backup[n]
        self.backup = {}


class AWP:
    """적대적 가중치 섭동(상혁 검증). 2D weight를 grad방향(크기비례) 섭동 후 eps-ball 클립."""
    def __init__(self, model, adv_lr=1.0, adv_eps=0.01):
        self.model, self.adv_lr, self.adv_eps = model, adv_lr, adv_eps
        self.backup, self.backup_eps = {}, {}
    def attack(self):
        e = 1e-6
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None and "weight" in n and p.dim() >= 2:
                if n not in self.backup:
                    self.backup[n] = p.data.clone()
                    ge = self.adv_eps * p.abs().detach()
                    self.backup_eps[n] = (self.backup[n] - ge, self.backup[n] + ge)
                norm = torch.norm(p.grad)
                if norm != 0 and not torch.isnan(norm):
                    r = self.adv_lr * p.grad / (norm + e) * p.abs().detach()
                    p.data.add_(r)
                    p.data = torch.min(torch.max(p.data, self.backup_eps[n][0]), self.backup_eps[n][1])
    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data = self.backup[n]
        self.backup, self.backup_eps = {}, {}


def make_opt(model):
    if not LLRD:
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    base, decay = LR, 0.9
    nl = model.config.num_hidden_layers
    groups, seen = [], set()
    def add(ps, lr):
        ps = [p for p in ps if id(p) not in seen and p.requires_grad]
        for p in ps: seen.add(id(p))
        if ps: groups.append({"params": ps, "lr": lr})
    add([p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n], base * 1.5)
    for i in range(nl - 1, -1, -1):
        add([p for n, p in model.named_parameters() if (f"encoder.layer.{i}." in n or f".layers.{i}." in n)], base * (decay ** (nl - 1 - i)))
    add([p for n, p in model.named_parameters() if "embeddings" in n], base * (decay ** nl))
    add([p for _, p in model.named_parameters()], base)
    return torch.optim.AdamW(groups, lr=base, weight_decay=0.01)


oof = np.zeros((len(samples), NUM_CLASSES), np.float32)
hold_sum = np.zeros((len(hold_idx), NUM_CLASSES), np.float32)
scores = []
oof_ema = np.zeros((len(samples), NUM_CLASSES), np.float32) if EMA_D > 0 else None
hold_sum_ema = np.zeros((len(hold_idx), NUM_CLASSES), np.float32) if EMA_D > 0 else None
scores_ema = []

def _ema_extra():
    """EMA 계측 켜졌을 때만 npz에 *_ema 키 추가 (off면 기존 스키마 그대로)."""
    if EMA_D <= 0:
        return {}
    return {"oof_ema": oof_ema, "hold_ema": hold_sum_ema / max(len(scores_ema), 1),
            "scores_ema": np.array(scores_ema), "ema_decay": EMA_D}
t0 = time.time()
GEN = np.array([s["gen"] for s in samples])
for fi in range(FOLD_LO, FOLD_HI):
    tr, va = folds[fi]
    print(f"=== fold {fi} ===", flush=True)
    model = build(); opt = make_opt(model)
    tr = np.asarray(tr)
    if EXCLUDE_AU:
        n0 = len(tr); tr = tr[GEN[tr] == "sim"]
        print(f"    [exclude_au] train {n0} -> {len(tr)}", flush=True)

    class DS(torch.utils.data.Dataset):
        def __len__(s): return len(tr)
        def __getitem__(s, i): return int(tr[i])

    def coll(b):
        return pad_batch(b), torch.tensor([y[j] for j in b])

    dl = torch.utils.data.DataLoader(DS(), batch_size=BATCH, shuffle=True, collate_fn=coll,
                                     num_workers=4, pin_memory=True, persistent_workers=True)
    tot = len(dl) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(tot * 0.06), tot)
    scaler = GradScaler("cuda", enabled=FP16)
    lossfn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))
    fgm = FGM(model) if FGM_ON else None
    awp = AWP(model) if AWP_ON else None
    ema = ({n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad}
           if EMA_D > 0 else None)
    best, bva, bho = -1, None, None
    best_e, bva_e, bho_e = -1, None, None
    for ep in range(EPOCHS):
        model.train()
        cek = [0.0, 0.0, 0]   # RDROP시 CE/KL 성분 추적 (R26 가드: α사고 vs 축기각 구분)
        for enc, lb in dl:
            enc = {k: v.to(device) for k, v in enc.items()}; lb = lb.to(device); opt.zero_grad()
            def _base_loss():
                if RDROP_A > 0:   # R-Drop: 드롭아웃 2회 forward + 대칭 KL 일치성
                    l1 = model(**enc).logits
                    l2 = model(**enc).logits
                    ce = 0.5 * (lossfn(l1, lb) + lossfn(l2, lb))
                    lp1 = torch.log_softmax(l1.float(), 1)
                    lp2 = torch.log_softmax(l2.float(), 1)
                    kl = 0.5 * (torch.nn.functional.kl_div(lp2, lp1.exp(), reduction="batchmean")
                                + torch.nn.functional.kl_div(lp1, lp2.exp(), reduction="batchmean"))
                    cek[0] += float(ce.detach()); cek[1] += float(kl.detach()); cek[2] += 1
                    return ce + RDROP_A * kl
                return lossfn(model(**enc).logits, lb)
            if FP16:
                with autocast("cuda", dtype=torch.float16):
                    loss = _base_loss()
            else:
                loss = _base_loss()
            scaler.scale(loss).backward()
            if fgm is not None:
                fgm.attack(FGM_EMB)
                if FP16:
                    with autocast("cuda", dtype=torch.float16):
                        aloss = lossfn(model(**enc).logits, lb)
                else:
                    aloss = lossfn(model(**enc).logits, lb)
                scaler.scale(aloss).backward()
                fgm.restore()
            if awp is not None:
                awp.attack()
                if FP16:
                    with autocast("cuda", dtype=torch.float16):
                        aloss2 = lossfn(model(**enc).logits, lb)
                else:
                    aloss2 = lossfn(model(**enc).logits, lb)
                scaler.scale(aloss2).backward()
                awp.restore()
            scaler.step(opt); scaler.update(); sch.step()
            if ema is not None:
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in ema:
                            ema[n].mul_(EMA_D).add_(p.detach().float(), alpha=1 - EMA_D)
        pv = infer_probs(model, va)
        mf1, _ = macro_f1(y[va], pv.argmax(1))
        sim_mask = GEN[np.asarray(va)] == "sim"
        smf1, _ = macro_f1(y[np.asarray(va)[sim_mask]], pv[sim_mask].argmax(1))
        print(f"    epoch {ep+1}: val={mf1:.4f} sim={smf1:.4f} @{(time.time()-t0)/60:.1f}min", flush=True)
        if RDROP_A > 0 and cek[2]:
            print(f"      [rdrop] CE={cek[0]/cek[2]:.4f} KL={cek[1]/cek[2]:.4f} αKL={RDROP_A*cek[1]/cek[2]:.4f}", flush=True)
        pick = smf1 if SELECT_SIM else mf1
        if pick > best:
            best = pick; bva = pv; bho = infer_probs(model, hold_idx)
        if ema is not None:   # EMA 가중치로도 평가 (패시브 — 학습에 무영향)
            bak = {n: p.detach().clone() for n, p in model.named_parameters() if n in ema}
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in ema: p.data.copy_(ema[n].to(p.dtype))
            pv_e = infer_probs(model, va)
            mf1_e, _ = macro_f1(y[va], pv_e.argmax(1))
            smf1_e, _ = macro_f1(y[np.asarray(va)[sim_mask]], pv_e[sim_mask].argmax(1))
            print(f"      [ema] val={mf1_e:.4f} sim={smf1_e:.4f}", flush=True)
            pick_e = smf1_e if SELECT_SIM else mf1_e
            if pick_e > best_e:
                best_e = pick_e; bva_e = pv_e; bho_e = infer_probs(model, hold_idx)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in bak: p.data.copy_(bak[n])
    oof[va] = bva; hold_sum += bho; scores.append(best)
    if ema is not None:
        oof_ema[va] = bva_e; hold_sum_ema += bho_e; scores_ema.append(best_e)
    if os.environ.get("AD_SAVE_MODEL", "0") == "1":   # [측정용] fold 모델 저장 (gen_rescue 실측)
        _md = os.path.join(WORK, f"foldmodel_{TAG}_f{fi}")
        model.save_pretrained(_md); tok.save_pretrained(_md)
        print(f"[save] fold{fi} model -> {_md}", flush=True)
    del model; torch.cuda.empty_cache()
    # 증분 저장: 세션이 죽어도 완료 fold까지 보존 (fold_hi=현재까지)
    np.savez_compressed(os.path.join(WORK, f"teacher_{TAG}.npz"),
                        oof=oof, hold=hold_sum / max(len(scores), 1),
                        scores=np.array(scores), fold_lo=FOLD_LO, fold_hi=fi + 1,
                        model=MODEL, version=VERSION, max_len=MAX_LEN, **_ema_extra())
    print(f"    [incremental npz saved: folds {FOLD_LO}..{fi}]", flush=True)

nf = FOLD_HI - FOLD_LO
np.savez_compressed(os.path.join(WORK, f"teacher_{TAG}.npz"),
                    oof=oof, hold=hold_sum / max(nf, 1),
                    scores=np.array(scores), fold_lo=FOLD_LO, fold_hi=FOLD_HI,
                    model=MODEL, version=VERSION, max_len=MAX_LEN, **_ema_extra())
cov = np.concatenate([folds[i][1] for i in range(FOLD_LO, FOLD_HI)])
pmf1, _ = macro_f1(y[cov], oof[cov].argmax(1))
print(f"[teacher {TAG}] fold_scores={[round(s,4) for s in scores]} covered-OOF={pmf1:.4f} "
      f"time={(time.time()-t0)/60:.1f}min", flush=True)
open(os.path.join(WORK, f"DONE_{TAG}"), "w").write(f"oof={pmf1:.4f} scores={scores}")
print("=== DONE ===", flush=True)
