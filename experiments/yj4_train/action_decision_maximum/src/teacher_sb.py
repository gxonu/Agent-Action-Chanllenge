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
pass  # [패치] pip 무력화
for z in ["open.zip", "ad_common.zip"]:
    if os.path.exists(z):
        with zipfile.ZipFile(z) as f:
            f.extractall(".")
sys.path.insert(0, WORK)

import numpy as np, torch
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
FOLD_LO = int(os.environ.get("AD_FOLD_LO", "0"))
FOLD_HI = int(os.environ.get("AD_FOLD_HI", "5"))
EXCLUDE_AU = os.environ.get("AD_EXCLUDE_AU", "0") == "1"   # sim-only 학습 프로브
# session-balanced 학습 (R26): ""=row-uniform / "weight"=1/session_len 가중 / "sample"=epoch당 세션당 1step
SESS_BAL = os.environ.get("AD_SESSION_BALANCED", "")
INIT_FROM = os.environ.get("AD_INIT_FROM", "")             # 체크포인트 디렉터리에서 FT 시작 (모델+토크나이저)
SAVE_WEIGHTS = os.environ.get("AD_SAVE_WEIGHTS", "0") == "1"  # fold별 best-epoch 가중치 저장(fp16)
# R55 T3: rescue-정합 학습 — 헤더보존 절단(배포 gen_rescue 와 동일 함수) + [HIST] 아이템 수.
# 학습·검증·holdout 모두 같은 INPUT_IDS 를 쓰므로 자동으로 훈련-서빙 정합. 기본 off = byte 동일.
GEN_RESCUE = os.environ.get("AD_GEN_RESCUE", "0") == "1"
MHT = int(os.environ.get("AD_MHT", "8"))                   # serialize max_hist_turns (기존 기본 8)
TAG = os.environ.get("AD_TAG", f"{MODEL.split('/')[-1]}_s{SEED}_f{FOLD_LO}{FOLD_HI}")
HEAD_SEED = 1234
device = "cuda"; assert torch.cuda.is_available()
print(f"[teacher] {TAG} model={MODEL} v={VERSION} len={MAX_LEN} ep={EPOCHS} lr={LR} "
      f"b={BATCH} fp16={FP16} llrd={LLRD} fgm={FGM_ON} folds=[{FOLD_LO},{FOLD_HI})"
      f" sess_bal={SESS_BAL or 'off'} init_from={INIT_FROM or 'hub'} save_w={SAVE_WEIGHTS}"
      f" gen_rescue={GEN_RESCUE} mht={MHT}", flush=True)
if SESS_BAL not in ("", "weight", "sample"):
    raise ValueError(f"AD_SESSION_BALANCED must be '', 'weight', or 'sample' (got {SESS_BAL!r})")

set_seed(SEED)
samples, y, ids = load_train()
y = np.array(y); groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
dev_idx, hold_idx = sp["dev_idx"], sp["holdout_idx"]
folds = sp["folds"]
cnt = np.bincount(y[dev_idx], minlength=NUM_CLASSES)
cw = len(dev_idx) / (NUM_CLASSES * np.maximum(cnt, 1)); cw /= cw.mean()

SRC = INIT_FROM if INIT_FROM else MODEL   # INIT_FROM이면 토크나이저도 체크포인트 것(프루닝 정합)
tok = AutoTokenizer.from_pretrained(SRC); tok.truncation_side = "left"
texts = [ad_lib.serialize(s, VERSION, MHT) for s in samples]
t0 = time.time()
enc_all = tok(texts, truncation=True, max_length=MAX_LEN, padding=False)
INPUT_IDS = enc_all["input_ids"]
print(f"[tok] {len(texts)} in {time.time()-t0:.0f}s", flush=True)
if GEN_RESCUE:
    # 배포와 동일 함수로 [GEN] 삭제 절단 행만 헤더보존 재구성 — 학습·검증·holdout 공통 적용
    _resc = ad_lib._gen_rescue_ids(tok, texts, MAX_LEN)
    for _i, _ids in _resc.items():
        INPUT_IDS[_i] = _ids
    print(f"[gen_rescue] {len(_resc)}/{len(texts)} rows header-preserved (mht={MHT})", flush=True)


def build():
    torch.manual_seed(HEAD_SEED)
    return AutoModelForSequenceClassification.from_pretrained(
        SRC, num_labels=NUM_CLASSES, torch_dtype=torch.float32,  # fp16 저장 ckpt도 fp32 업캐스트
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
        add([p for n, p in model.named_parameters() if f"encoder.layer.{i}." in n], base * (decay ** (nl - 1 - i)))
    add([p for n, p in model.named_parameters() if "embeddings" in n], base * (decay ** nl))
    add([p for _, p in model.named_parameters()], base)
    return torch.optim.AdamW(groups, lr=base, weight_decay=0.01)


oof = np.zeros((len(samples), NUM_CLASSES), np.float32)
hold_sum = np.zeros((len(hold_idx), NUM_CLASSES), np.float32)
scores = []
t0 = time.time()
GEN = np.array([s["gen"] for s in samples])
from collections import Counter
_slen = Counter(groups.tolist())
ROW_W_ALL = np.array([1.0 / _slen[g] for g in groups], np.float32)  # 세션균등 가중(그룹split이라 fold내 세션 온전)
for fi in range(FOLD_LO, FOLD_HI):
    tr, va = folds[fi]
    print(f"=== fold {fi} ===", flush=True)
    model = build(); opt = make_opt(model)
    tr = np.asarray(tr)
    if EXCLUDE_AU:
        n0 = len(tr); tr = tr[GEN[tr] == "sim"]
        print(f"    [exclude_au] train {n0} -> {len(tr)}", flush=True)

    if SESS_BAL == "sample":
        sess2rows = {}
        for j in tr:
            sess2rows.setdefault(groups[j], []).append(int(j))
        sess_rows = [np.array(v) for v in sess2rows.values()]
        rng = np.random.RandomState(SEED + 1000 + fi)
        steps_per_ep = (len(sess_rows) + BATCH - 1) // BATCH
        dl = None
        print(f"    [sess_bal=sample] sessions={len(sess_rows)} steps/ep={steps_per_ep}", flush=True)
    else:
        class DS(torch.utils.data.Dataset):
            def __len__(s): return len(tr)
            def __getitem__(s, i): return int(tr[i])

        def coll(b):
            return pad_batch(b), torch.tensor([y[j] for j in b]), torch.tensor(b)

        dl = torch.utils.data.DataLoader(DS(), batch_size=BATCH, shuffle=True, collate_fn=coll,
                                         num_workers=4, pin_memory=True, persistent_workers=True)
        steps_per_ep = len(dl)
    W = None
    if SESS_BAL == "weight":
        w = ROW_W_ALL[tr].astype(np.float64); w *= len(w) / w.sum()   # mean 1 정규화
        W = np.zeros(len(samples)); W[tr] = w
        print(f"    [sess_bal=weight] min={w.min():.3f} max={w.max():.3f}", flush=True)
    tot = steps_per_ep * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(tot * 0.06), tot)
    scaler = GradScaler("cuda", enabled=FP16)
    cw_t = torch.tensor(cw, dtype=torch.float, device=device)
    lossfn = torch.nn.CrossEntropyLoss(weight=cw_t)
    lossfn_none = torch.nn.CrossEntropyLoss(weight=cw_t, reduction="none")

    def batch_loss(logits, lb, bi):
        if W is None:
            return lossfn(logits, lb)
        l = lossfn_none(logits, lb)                      # cw_y·CE per row
        rw = torch.tensor(W[bi.numpy()], dtype=torch.float, device=device)
        return (l * rw).sum() / (rw * cw_t[lb]).sum().clamp_min(1e-8)  # torch weighted-mean의 세션가중 일반화

    fgm = FGM(model) if FGM_ON else None
    best, bva, bho, bsd = -1, None, None, None
    for ep in range(EPOCHS):
        model.train()
        if SESS_BAL == "sample":
            picks = np.array([r[rng.randint(len(r))] for r in sess_rows])
            rng.shuffle(picks)
            it = ((pad_batch([int(j) for j in picks[b:b + BATCH]]),
                   torch.tensor([y[j] for j in picks[b:b + BATCH]]),
                   torch.tensor(picks[b:b + BATCH]))
                  for b in range(0, len(picks), BATCH))
        else:
            it = dl
        for enc, lb, bi in it:
            enc = {k: v.to(device) for k, v in enc.items()}; lb = lb.to(device); opt.zero_grad()
            if FP16:
                with autocast("cuda", dtype=torch.float16):
                    loss = batch_loss(model(**enc).logits, lb, bi)
            else:
                loss = batch_loss(model(**enc).logits, lb, bi)
            scaler.scale(loss).backward()
            if fgm is not None:
                fgm.attack()
                if FP16:
                    with autocast("cuda", dtype=torch.float16):
                        aloss = batch_loss(model(**enc).logits, lb, bi)
                else:
                    aloss = batch_loss(model(**enc).logits, lb, bi)
                scaler.scale(aloss).backward()
                fgm.restore()
            scaler.step(opt); scaler.update(); sch.step()
        pv = infer_probs(model, va)
        mf1, _ = macro_f1(y[va], pv.argmax(1))
        sim_mask = GEN[np.asarray(va)] == "sim"
        smf1, _ = macro_f1(y[np.asarray(va)[sim_mask]], pv[sim_mask].argmax(1))
        print(f"    epoch {ep+1}: val={mf1:.4f} sim={smf1:.4f} @{(time.time()-t0)/60:.1f}min", flush=True)
        if mf1 > best:
            best = mf1; bva = pv; bho = infer_probs(model, hold_idx)
            if SAVE_WEIGHTS:
                bsd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    oof[va] = bva; hold_sum += bho; scores.append(best)
    if SAVE_WEIGHTS and bsd is not None:
        model.load_state_dict(bsd)
        ck = os.path.join(WORK, f"foldckpt_{TAG}_f{fi}")
        model.half().save_pretrained(ck, safe_serialization=True)
        tok.save_pretrained(ck)
        print(f"    [ckpt saved: {ck} val={best:.4f}]", flush=True)
    del model; torch.cuda.empty_cache()
    # 증분 저장: 세션이 죽어도 완료 fold까지 보존 (fold_hi=현재까지)
    np.savez_compressed(os.path.join(WORK, f"teacher_{TAG}.npz"),
                        oof=oof, hold=hold_sum / max(len(scores), 1),
                        scores=np.array(scores), fold_lo=FOLD_LO, fold_hi=fi + 1,
                        model=MODEL, version=VERSION, max_len=MAX_LEN)
    print(f"    [incremental npz saved: folds {FOLD_LO}..{fi}]", flush=True)

nf = FOLD_HI - FOLD_LO
np.savez_compressed(os.path.join(WORK, f"teacher_{TAG}.npz"),
                    oof=oof, hold=hold_sum / max(nf, 1),
                    scores=np.array(scores), fold_lo=FOLD_LO, fold_hi=FOLD_HI,
                    model=MODEL, version=VERSION, max_len=MAX_LEN)
cov = np.concatenate([folds[i][1] for i in range(FOLD_LO, FOLD_HI)])
pmf1, _ = macro_f1(y[cov], oof[cov].argmax(1))
print(f"[teacher {TAG}] fold_scores={[round(s,4) for s in scores]} covered-OOF={pmf1:.4f} "
      f"time={(time.time()-t0)/60:.1f}min", flush=True)
open(os.path.join(WORK, f"DONE_{TAG}"), "w").write(f"oof={pmf1:.4f} scores={scores}")
print("=== DONE ===", flush=True)
