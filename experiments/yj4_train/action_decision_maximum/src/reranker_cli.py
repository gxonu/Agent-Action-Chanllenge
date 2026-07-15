#!/usr/bin/env python
"""E4(탐색4클래스) 전용 reranker 교사 — R40 A안 (DEBATE R43~R48 사전등록 사양).

- 학습: fold0-train ∩ label∈E4 (v6 직렬화 동일입력, 단독변수 = reranker 함수족)
- 추론: fold0-val 전행에 4-class 확률 저장 (트리거/α 평가는 별도 CPU 스크립트)
- 판정(3분기): hard FAIL(Δ<+0.0005) / soft positive(+0.0005~14 → B 1회) / PASS(≥+0.0015)

env: AD_MODEL(xlm-roberta-base), AD_EPOCHS(5), AD_LR(3e-5), AD_BATCH(64), AD_SEED(1234),
     AD_MAXLEN(320), AD_TAG. FGM/RDROP 없음(단독변수 유지).
출력: work/reranker_<TAG>.npz  {probs4: (n_val,4), val_idx, e4_classes, scores}
"""
import os, sys, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, torch
from common.io_utils import load_train, CLASSES, set_seed
from common.cv import make_splits
from common.metrics import macro_f1
from common import ad_lib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler

MODEL = os.environ.get("AD_MODEL", "xlm-roberta-base")
VERSION = os.environ.get("AD_VERSION", "v6")
MAX_LEN = int(os.environ.get("AD_MAXLEN", "320"))
EPOCHS = int(os.environ.get("AD_EPOCHS", "5"))
LR = float(os.environ.get("AD_LR", "3e-5"))
BATCH = int(os.environ.get("AD_BATCH", "64"))
SEED = int(os.environ.get("AD_SEED", "1234"))
TAG = os.environ.get("AD_TAG", "cc_reranker_a_f0")
WORK = os.environ.get("AD_WORK", "/workspace/work")
FULL = os.environ.get("AD_FULL", "0") == "1"   # 1: 전 E4행 학습(배포용) — 고정 에폭·최종가중치 저장·프루닝
E4 = ["read_file", "grep_search", "list_directory", "glob_pattern"]
E4_IDX = [CLASSES.index(c) for c in E4]
device = "cuda"; assert torch.cuda.is_available()
print(f"[reranker] {TAG} model={MODEL} v={VERSION} len={MAX_LEN} ep={EPOCHS} lr={LR} b={BATCH}", flush=True)

set_seed(SEED)
samples, y, ids = load_train()
y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups)
tr_all, va = sp["folds"][0]
tr_all, va = np.asarray(tr_all), np.asarray(va)
in_e4 = np.isin(y, E4_IDX)
if FULL:
    tr = np.where(in_e4)[0]                      # 배포용: 전 E4행(70k 중 28.8k)
else:
    tr = tr_all[in_e4[tr_all]]                   # fold0-train ∩ E4 라벨
y4 = np.full(len(samples), -1, np.int64)
for j, c in enumerate(E4_IDX):
    y4[y == c] = j
sim = np.array([not i.startswith("sess_au_") for i in ids])
print(f"[data] train(E4)={len(tr)}  val(all)={len(va)}  val∩E4∩sim={((in_e4 & sim)[va]).sum()}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL); tok.truncation_side = "left"
texts = [ad_lib.serialize(s, VERSION) for s in samples]
enc_all = tok(texts, truncation=True, max_length=MAX_LEN, padding=False)
INPUT_IDS = enc_all["input_ids"]
cnt = np.bincount(y4[tr], minlength=4)
cw = len(tr) / (4 * np.maximum(cnt, 1)); cw /= cw.mean()
print(f"[cw] {dict(zip(E4, np.round(cw,3)))}", flush=True)

torch.manual_seed(1234)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=4,
    id2label={i: c for i, c in enumerate(E4)}, label2id={c: i for i, c in enumerate(E4)}).to(device)

def pad_batch(idx_list):
    return tok.pad({"input_ids": [INPUT_IDS[j] for j in idx_list]}, return_tensors="pt")

def infer4(idx):
    model.eval(); bs = 256
    order = sorted(range(len(idx)), key=lambda k: len(INPUT_IDS[int(idx[k])]))
    out = np.zeros((len(idx), 4), np.float32)
    with torch.no_grad():
        for b in range(0, len(order), bs):
            ks = order[b:b + bs]; sub = [int(idx[k]) for k in ks]
            enc = pad_batch(sub).to(device)
            with autocast("cuda", dtype=torch.float16):
                lg = model(**enc).logits.float()
            p = torch.softmax(lg, 1).cpu().numpy()
            for m, k in enumerate(ks):
                out[k] = p[m]
    return out

class DS(torch.utils.data.Dataset):
    def __len__(s): return len(tr)
    def __getitem__(s, i): return int(tr[i])

def coll(b):
    return pad_batch(b), torch.tensor([y4[j] for j in b])

dl = torch.utils.data.DataLoader(DS(), batch_size=BATCH, shuffle=True, collate_fn=coll,
                                 num_workers=2, pin_memory=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sch = get_linear_schedule_with_warmup(opt, int(len(dl) * EPOCHS * 0.06), len(dl) * EPOCHS)
scaler = GradScaler("cuda")
lossfn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))

va_e4_sim = va[(in_e4 & sim)[va]]
best, bp = -1, None
scores = []
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    for enc, lb in dl:
        enc = {k: v.to(device) for k, v in enc.items()}; lb = lb.to(device); opt.zero_grad()
        with autocast("cuda", dtype=torch.float16):
            loss = lossfn(model(**enc).logits, lb)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sch.step()
    pv = infer4(va_e4_sim)
    mf, _ = macro_f1(y4[va_e4_sim], pv.argmax(1), n_classes=4)
    scores.append(mf)
    print(f"    epoch {ep+1}: e4sim_macro4={mf:.4f} @{(time.time()-t0)/60:.1f}min", flush=True)
    if not FULL and mf > best:
        best = mf
        bp = infer4(va)          # best epoch에서 val 전행 확률 저장
if FULL:
    # 배포용: 최종 가중치 저장 → 프루닝 → work/rr_<TAG>/
    import shutil
    from common.vocab_prune import prune_model_dir
    raw = os.path.join(WORK, f"rr_raw_{TAG}"); out = os.path.join(WORK, f"rr_{TAG}")
    shutil.rmtree(raw, ignore_errors=True); shutil.rmtree(out, ignore_errors=True)
    model.half().save_pretrained(raw); tok.save_pretrained(raw)
    K, _ = prune_model_dir(raw, out, tok, texts, max_len=MAX_LEN)
    shutil.rmtree(raw, ignore_errors=True)
    sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)
    print(f"[full-rr] pruned vocab={K}  dir={out}  size={sz/1e6:.0f}MB", flush=True)
else:
    np.savez_compressed(os.path.join(WORK, f"reranker_{TAG}.npz"),
                        probs4=bp, val_idx=va, e4_classes=np.array(E4_IDX),
                        scores=np.array(scores), model=MODEL, version=VERSION)
print(f"[reranker {TAG}] best_macro4={best:.4f}  saved", flush=True)
print("=== DONE ===", flush=True)
