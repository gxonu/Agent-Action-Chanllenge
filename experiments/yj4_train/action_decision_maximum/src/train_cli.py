#!/usr/bin/env python
"""Track B(정확도 최대) CLI 학습 — Colab VM에서 실행.

멀티시드(옵션 멀티아키텍처) 5-fold 앙상블 → OOF soft-label 지식증류 → 단일 xlm-r student.
입력: open.zip, ad_common.zip (또는 이미 풀린 data/, common/, splits/).
출력: submit_maximum.zip + DONE.
환경변수: AD_SEEDS(콤마), AD_VERSION, AD_MAXLEN, AD_EPOCHS, AD_BATCH, AD_DISTILL_EPOCHS, AD_MDEBERTA(1이면 추가).
"""
import os, sys, subprocess, time, zipfile, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
WORK = os.environ.get("AD_WORK", "/content" if os.path.isdir("/content") else os.getcwd())
os.chdir(WORK)
print(f"[trainB] WORK={WORK}", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.46.3", "accelerate==1.9.0", "sentencepiece==0.1.99"], check=False)
for z in ["open.zip", "ad_common.zip"]:
    if os.path.exists(z):
        with zipfile.ZipFile(z) as f:
            f.extractall(".")
sys.path.insert(0, WORK)

import numpy as np, torch
from common.io_utils import load_train, CLASSES, NUM_CLASSES, set_seed
from common.cv import make_splits
from common.metrics import macro_f1, print_report
from common import ad_lib, postproc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler

VERSION = os.environ.get("AD_VERSION", "v4")
MAX_LEN = int(os.environ.get("AD_MAXLEN", "256"))
EPOCHS = int(os.environ.get("AD_EPOCHS", "2"))
BATCH = int(os.environ.get("AD_BATCH", "48"))
DISTILL_EPOCHS = int(os.environ.get("AD_DISTILL_EPOCHS", "3"))
# TEACHERS: AD_TEACHERS="name:fp16:seed,..." (fp16=1 fp16학습, 0 fp32). 미지정 시 기본.
_spec = os.environ.get("AD_TEACHERS", "xlm-roberta-base:1:1234,xlm-roberta-large:1:1234")
TEACHERS = []
for tok_spec in _spec.split(","):
    parts = tok_spec.split(":")
    name = parts[0]
    fp16 = (len(parts) < 2 or parts[1] == "1")
    seed = int(parts[2]) if len(parts) > 2 else 1234
    TEACHERS.append((name, fp16, seed))
STUDENT = "xlm-roberta-base"; ALPHA = 0.7; LR = 2e-5; WD = 0.01; WARMUP = 0.06; HEAD_SEED = 1234
OUTDIR = os.environ.get("AD_OUT", "submit_maximum"); MODELDIR = OUTDIR + "/model"
device = "cuda"; assert torch.cuda.is_available()
print(f"[cfgB] version={VERSION} max_len={MAX_LEN} epochs={EPOCHS} batch={BATCH} teachers={TEACHERS}", flush=True)

set_seed(42)
samples, y, ids = load_train()
y = np.array(y); groups = np.array([s["session"] for s in samples]); gen = np.array([s["gen"] for s in samples])
sp = make_splits(ids, y, groups); dev_idx, hold_idx, folds = sp["dev_idx"], sp["holdout_idx"], sp["folds"]
texts = [ad_lib.serialize(s, VERSION) for s in samples]
cnt = np.bincount(y[dev_idx], minlength=NUM_CLASSES); cw = len(dev_idx) / (NUM_CLASSES * np.maximum(cnt, 1)); cw /= cw.mean()
_TOK = {}
def get_tok(name):
    if name not in _TOK:
        t = AutoTokenizer.from_pretrained(name); t.truncation_side = "left"; _TOK[name] = t
    return _TOK[name]

def build(name):
    torch.manual_seed(HEAD_SEED)
    return AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=NUM_CLASSES, id2label={i: c for i, c in enumerate(CLASSES)},
        label2id={c: i for i, c in enumerate(CLASSES)}).to(device)

def infer_probs(model, tok, idx, fp16=True):
    model.eval(); bs = 96
    order = sorted(range(len(idx)), key=lambda k: len(texts[int(idx[k])]))
    out = np.zeros((len(idx), NUM_CLASSES), np.float32)
    with torch.no_grad():
        for b in range(0, len(order), bs):
            ks = order[b:b+bs]; sub = [int(idx[k]) for k in ks]
            enc = tok([texts[j] for j in sub], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
            if fp16:
                with autocast("cuda", dtype=torch.float16): lg = model(**enc).logits.float()
            else:
                lg = model(**enc).logits.float()
            p = torch.softmax(lg, 1).cpu().numpy()
            for m, k in enumerate(ks): out[k] = p[m]
    return out

def train_one(name, seed, tr, va, fp16=True):
    set_seed(seed); tok = get_tok(name); model = build(name)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    class DS(torch.utils.data.Dataset):
        def __len__(s): return len(tr)
        def __getitem__(s, i): j = int(tr[i]); return texts[j], int(y[j])
    def coll(b):
        enc = tok([x[0] for x in b], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return enc, torch.tensor([x[1] for x in b])
    dl = torch.utils.data.DataLoader(DS(), batch_size=BATCH, shuffle=True, collate_fn=coll, num_workers=2)
    tot = len(dl) * EPOCHS; sch = get_linear_schedule_with_warmup(opt, int(tot*WARMUP), tot)
    scaler = GradScaler("cuda", enabled=fp16); wt = torch.tensor(cw, dtype=torch.float, device=device)
    ce = torch.nn.CrossEntropyLoss(weight=wt); best = -1; bva = None; bho = None
    for ep in range(EPOCHS):
        model.train()
        for enc, lb in dl:
            enc = {k: v.to(device) for k, v in enc.items()}; lb = lb.to(device); opt.zero_grad()
            if fp16:
                with autocast("cuda", dtype=torch.float16): loss = ce(model(**enc).logits, lb)
            else:
                loss = ce(model(**enc).logits, lb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        pv = infer_probs(model, tok, va, fp16); mf1, _ = macro_f1(y[va], pv.argmax(1))
        if mf1 > best: best = mf1; bva = pv; bho = infer_probs(model, tok, hold_idx, fp16)
    print(f"    [{name} seed{seed}] best val={best:.4f}", flush=True)
    del model; torch.cuda.empty_cache()
    return bva, bho

oof_sum = np.zeros((len(samples), NUM_CLASSES), np.float32); oof_cnt = np.zeros(len(samples), np.float32)
hold_sum = np.zeros((len(hold_idx), NUM_CLASSES), np.float32); hold_n = 0
t0 = time.time()
for (name, fp16, seed) in TEACHERS:
    print(f"=== teacher {name} seed{seed} fp16={fp16} ===", flush=True)
    for fi, (tr, va) in enumerate(folds):
        pv, ph = train_one(name, seed, tr, va, fp16)
        oof_sum[va] += pv; oof_cnt[va] += 1; hold_sum += ph; hold_n += 1
teacher_oof = np.zeros_like(oof_sum); m = oof_cnt > 0; teacher_oof[m] = oof_sum[m] / oof_cnt[m, None]
teacher_hold = hold_sum / max(hold_n, 1)
# 재사용(증류/후처리/에러분석 반복)을 위해 교사 OOF 저장 → 다운로드 가능
np.savez(os.path.join(WORK, "teacher_oof.npz"), oof=teacher_oof, hold=teacher_hold,
         dev_idx=dev_idx, hold_idx=hold_idx, y=y)
print(f"[teachers] {(time.time()-t0)/60:.1f} min | saved teacher_oof.npz", flush=True)

ens_mf1, _ = macro_f1(y[dev_idx], teacher_oof[dev_idx].argmax(1))
sim_dev = dev_idx[gen[dev_idx] == "sim"]
print(f"[ENSEMBLE] pooled-OOF={ens_mf1:.4f} sim-only={macro_f1(y[sim_dev], teacher_oof[sim_dev].argmax(1))[0]:.4f} holdout={macro_f1(y[hold_idx], teacher_hold.argmax(1))[0]:.4f}", flush=True)
print_report(y[dev_idx], teacher_oof[dev_idx].argmax(1), "ENSEMBLE OOF")

# 증류
tok = get_tok(STUDENT); soft = torch.tensor(teacher_oof[dev_idx], dtype=torch.float)
class KDS(torch.utils.data.Dataset):
    def __len__(s): return len(dev_idx)
    def __getitem__(s, i): j = int(dev_idx[i]); return texts[j], soft[i], int(y[j])
def kcoll(b):
    enc = tok([x[0] for x in b], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return enc, torch.stack([x[1] for x in b]), torch.tensor([x[2] for x in b])
dl = torch.utils.data.DataLoader(KDS(), batch_size=BATCH, shuffle=True, collate_fn=kcoll, num_workers=2)
student = build(STUDENT); opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WD)
tot = len(dl) * DISTILL_EPOCHS; sch = get_linear_schedule_with_warmup(opt, int(tot*WARMUP), tot)
scaler = GradScaler("cuda"); wt = torch.tensor(cw, dtype=torch.float, device=device); ce = torch.nn.CrossEntropyLoss(weight=wt)
best = -1; best_state = None
for ep in range(DISTILL_EPOCHS):
    student.train()
    for enc, sp_t, hy in dl:
        enc = {k: v.to(device) for k, v in enc.items()}; sp_t = sp_t.to(device); hy = hy.to(device); opt.zero_grad()
        with autocast("cuda", dtype=torch.float16):
            z = student(**enc).logits; logp = torch.log_softmax(z, 1)
            loss = ALPHA * (-(sp_t * logp).sum(1).mean()) + (1 - ALPHA) * ce(z, hy)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    ph = infer_probs(student, tok, hold_idx); mf1, _ = macro_f1(y[hold_idx], ph.argmax(1))
    print(f"  distill epoch {ep+1}: holdout={mf1:.4f}", flush=True)
    if mf1 > best: best = mf1; best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
student.load_state_dict(best_state); print(f"[student] best holdout={best:.4f}", flush=True)

# 후처리 + 저장
bias, fit = postproc.fit_bias(np.log(teacher_oof[dev_idx] + 1e-9), y[dev_idx])
hlog = infer_probs(student, tok, hold_idx)
h_nob = macro_f1(y[hold_idx], hlog.argmax(1))[0]
h_b = macro_f1(y[hold_idx], (np.log(hlog + 1e-9) + bias).argmax(1))[0]
USE_BIAS = bool(h_b >= h_nob - 1e-4)
print(f"[postproc] fit={fit:.4f} student holdout no-bias={h_nob:.4f} +bias={h_b:.4f} use={USE_BIAS}", flush=True)

import shutil
os.makedirs(MODELDIR, exist_ok=True)
student.half().save_pretrained(MODELDIR, safe_serialization=True); tok.save_pretrained(MODELDIR)
shutil.copy("common/ad_lib.py", os.path.join(MODELDIR, "ad_lib.py"))
postproc.save(os.path.join(MODELDIR, "postproc.json"), bias if USE_BIAS else np.zeros(NUM_CLASSES),
              {"use_bias": USE_BIAS, "ensemble_oof": float(ens_mf1), "student_holdout": float(max(h_b, h_nob))})
json.dump({"version": VERSION, "max_len": MAX_LEN, "batch_size": 128}, open(os.path.join(MODELDIR, "run_meta.json"), "w"))
shutil.copy("common/server_script.py", OUTDIR + "/script.py"); open(OUTDIR + "/requirements.txt", "w").write("")
mb = sum(os.path.getsize(os.path.join(MODELDIR, f)) for f in os.listdir(MODELDIR)) / 1e6
print(f"[save] model/ size={mb:.1f}MB", flush=True)

bidx = np.random.RandomState(0).choice(dev_idx, size=min(30000, len(dev_idx)), replace=False)
bench = [samples[int(i)] for i in bidx]; t = time.time()
_ = ad_lib.predict(MODELDIR, bench, version=VERSION, max_len=MAX_LEN, batch_size=128, postproc_path=MODELDIR + "/postproc.json")
dt = time.time() - t; print(f"[timing] 30k {dt:.1f}s {'OK' if dt<600 else 'SLOW'}", flush=True)

def zipdir(src, zf):
    for root, _, fs in os.walk(src):
        for fn in fs: zf.write(os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), src))
zp = os.path.join(WORK, OUTDIR + ".zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z: zipdir(OUTDIR, z)
print(f"[zip] {zp} {os.path.getsize(zp)/1e6:.0f}MB", flush=True)
open(os.path.join(WORK, "DONE"), "w").write(f"ensemble_oof={ens_mf1:.4f} student_holdout={max(h_b,h_nob):.4f} timing={dt:.0f}s")
print("=== DONE ===", flush=True)
