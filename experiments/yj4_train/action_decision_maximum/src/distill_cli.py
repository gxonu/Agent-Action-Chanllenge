#!/usr/bin/env python
"""교사 npz들(teacher_*.npz) 평균 → 단일 xlm-r-base student 증류 → submit zip.

env: AD_VERSION(v5), AD_MAXLEN(320), AD_EPOCHS(3), AD_LR(3e-5), AD_BATCH(96),
     AD_ALPHA(0.7 soft비중), AD_LLRD(1), AD_FULL(0: dev만 학습+holdout검증 / 1: 70k 전체 학습),
     AD_BIAS_JSON(재사용할 bias json 경로, 없으면 OOF로 적합), AD_OUT(submit_maximum).
입력: /content/teacher_*.npz (여러 개), open.zip, ad_common.zip.
"""
import os, sys, subprocess, time, zipfile, json, glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"
WORK = os.environ.get("AD_WORK", "/content" if os.path.isdir("/content") else os.getcwd())
os.chdir(WORK)
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

STUDENT = os.environ.get("AD_STUDENT", "xlm-roberta-base")
VERSION = os.environ.get("AD_VERSION", "v5")
MAX_LEN = int(os.environ.get("AD_MAXLEN", "320"))
EPOCHS = int(os.environ.get("AD_EPOCHS", "3"))
LR = float(os.environ.get("AD_LR", "3e-5"))
BATCH = int(os.environ.get("AD_BATCH", "96"))
ALPHA = float(os.environ.get("AD_ALPHA", "0.7"))
LLRD = os.environ.get("AD_LLRD", "1") == "1"
FULL = os.environ.get("AD_FULL", "0") == "1"
BIAS_JSON = os.environ.get("AD_BIAS_JSON", "")
OUTDIR = os.environ.get("AD_OUT", "submit_maximum"); MODELDIR = OUTDIR + "/model"
device = "cuda"; assert torch.cuda.is_available()

set_seed(42)
samples, y, ids = load_train()
y = np.array(y); groups = np.array([s["session"] for s in samples]); gen = np.array([s["gen"] for s in samples])
sp = make_splits(ids, y, groups)
dev_idx, hold_idx, folds = sp["dev_idx"], sp["holdout_idx"], sp["folds"]

# ---- 교사 확률 평균 ----
tfiles = sorted(glob.glob(os.path.join(WORK, "teacher_*.npz")))
assert tfiles, "teacher_*.npz 없음"
oof_sum = np.zeros((len(samples), NUM_CLASSES), np.float32)
oof_cnt = np.zeros(len(samples), np.float32)
hold_sum = np.zeros((len(hold_idx), NUM_CLASSES), np.float32); hold_w = 0.0
for tf in tfiles:
    d = np.load(tf, allow_pickle=True)
    o = d["oof"]; lo, hi = int(d["fold_lo"]), int(d["fold_hi"])
    cov = np.concatenate([folds[i][1] for i in range(lo, hi)])
    oof_sum[cov] += o[cov]; oof_cnt[cov] += 1
    w = (hi - lo) / 5.0
    hold_sum += d["hold"] * w; hold_w += w
    print(f"[load] {os.path.basename(tf)} folds=[{lo},{hi}) scores={list(np.round(d['scores'],4))}", flush=True)
teacher_oof = np.zeros_like(oof_sum); m = oof_cnt > 0
teacher_oof[m] = oof_sum[m] / oof_cnt[m, None]
teacher_hold = hold_sum / max(hold_w, 1e-9)
cov_all = m & np.isin(np.arange(len(samples)), dev_idx)
ens_mf1, _ = macro_f1(y[cov_all], teacher_oof[cov_all].argmax(1))
hold_mf1, _ = macro_f1(y[hold_idx], teacher_hold.argmax(1))
print(f"[ENSEMBLE] pooled-OOF={ens_mf1:.4f} holdout={hold_mf1:.4f} (teachers={len(tfiles)})", flush=True)
print_report(y[cov_all], teacher_oof[cov_all].argmax(1), "ENSEMBLE OOF")

# ---- soft label 구성 ----
if FULL:
    train_idx = np.concatenate([dev_idx, hold_idx])
    soft_np = np.zeros((len(samples), NUM_CLASSES), np.float32)
    soft_np[dev_idx] = teacher_oof[dev_idx]
    soft_np[hold_idx] = teacher_hold          # holdout은 교사(전부 dev 학습)의 클린 예측
else:
    train_idx = dev_idx
    soft_np = teacher_oof

# ---- student 학습 ----
tok = AutoTokenizer.from_pretrained(STUDENT); tok.truncation_side = "left"
texts = [ad_lib.serialize(s, VERSION) for s in samples]
enc_all = tok(texts, truncation=True, max_length=MAX_LEN, padding=False)
INPUT_IDS = enc_all["input_ids"]

def pad_batch(idx_list):
    return tok.pad({"input_ids": [INPUT_IDS[j] for j in idx_list]}, return_tensors="pt")

def build():
    torch.manual_seed(1234)
    return AutoModelForSequenceClassification.from_pretrained(
        STUDENT, num_labels=NUM_CLASSES,
        id2label={i: c for i, c in enumerate(CLASSES)},
        label2id={c: i for i, c in enumerate(CLASSES)}).to(device)

def infer_logits(model, idx):
    model.eval(); bs = 192
    order = sorted(range(len(idx)), key=lambda k: len(INPUT_IDS[int(idx[k])]))
    out = np.zeros((len(idx), NUM_CLASSES), np.float32)
    with torch.no_grad():
        for b in range(0, len(order), bs):
            ks = order[b:b + bs]; sub = [int(idx[k]) for k in ks]
            enc = pad_batch(sub).to(device)
            with autocast("cuda", dtype=torch.float16):
                lg = model(**enc).logits.float().cpu().numpy()
            for mm, k in enumerate(ks):
                out[k] = lg[mm]
    return out

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

cnt = np.bincount(y[train_idx], minlength=NUM_CLASSES)
cw = len(train_idx) / (NUM_CLASSES * np.maximum(cnt, 1)); cw /= cw.mean()
soft_t = torch.tensor(soft_np, dtype=torch.float)
student = build(); opt = make_opt(student)

class KDS(torch.utils.data.Dataset):
    def __len__(s): return len(train_idx)
    def __getitem__(s, i): return int(train_idx[i])

def kcoll(b):
    return pad_batch(b), soft_t[b], torch.tensor([y[j] for j in b])

dl = torch.utils.data.DataLoader(KDS(), batch_size=BATCH, shuffle=True, collate_fn=kcoll,
                                 num_workers=4, pin_memory=True, persistent_workers=True)
tot = len(dl) * EPOCHS
sch = get_linear_schedule_with_warmup(opt, int(tot * 0.06), tot)
scaler = GradScaler("cuda")
ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))
t0 = time.time()
best, best_state = -1, None
for ep in range(EPOCHS):
    student.train()
    for enc, sp_t, hy in dl:
        enc = {k: v.to(device) for k, v in enc.items()}
        sp_t = sp_t.to(device); hy = hy.to(device); opt.zero_grad()
        with autocast("cuda", dtype=torch.float16):
            z = student(**enc).logits
            logp = torch.log_softmax(z, 1)
            loss = ALPHA * (-(sp_t * logp).sum(1).mean()) + (1 - ALPHA) * ce(z, hy)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    if FULL:
        print(f"  distill(FULL) epoch {ep+1} @{(time.time()-t0)/60:.1f}min", flush=True)
        best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    else:
        hlog = infer_logits(student, hold_idx)
        mf1, _ = macro_f1(y[hold_idx], hlog.argmax(1))
        print(f"  distill epoch {ep+1}: holdout={mf1:.4f} @{(time.time()-t0)/60:.1f}min", flush=True)
        if mf1 > best:
            best = mf1; best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
student.load_state_dict(best_state)

# ---- bias ----
if BIAS_JSON and os.path.exists(BIAS_JSON):
    bias = np.array(json.load(open(BIAS_JSON))["bias"])
    print(f"[postproc] bias 재사용: {BIAS_JSON}", flush=True)
    USE_BIAS = True
else:
    bias, fit = postproc.fit_bias(np.log(teacher_oof[cov_all] + 1e-9), y[cov_all])
    if FULL:
        USE_BIAS = True
        print(f"[postproc] teacher-OOF bias fit={fit:.4f} (FULL: holdout 검증 생략, OOF 기준 채택)", flush=True)
    else:
        hlog = infer_logits(student, hold_idx)
        h_nob = macro_f1(y[hold_idx], hlog.argmax(1))[0]
        h_b = macro_f1(y[hold_idx], (ad_lib._to_logprobs_np(hlog, np) + bias).argmax(1))[0]
        USE_BIAS = bool(h_b >= h_nob - 1e-4)
        print(f"[postproc] fit={fit:.4f} holdout no-bias={h_nob:.4f} +bias={h_b:.4f} use={USE_BIAS}", flush=True)

# ---- 저장/패키징 ----
import shutil
os.makedirs(MODELDIR, exist_ok=True)
student.half().save_pretrained(MODELDIR, safe_serialization=True)
tok.save_pretrained(MODELDIR)
shutil.copy("common/ad_lib.py", os.path.join(MODELDIR, "ad_lib.py"))
postproc.save(os.path.join(MODELDIR, "postproc.json"), bias if USE_BIAS else np.zeros(NUM_CLASSES),
              {"use_bias": bool(USE_BIAS), "ensemble_oof": float(ens_mf1),
               "student_holdout": float(best if best > 0 else -1), "full": FULL})
json.dump({"version": VERSION, "max_len": MAX_LEN, "batch_size": 128},
          open(os.path.join(MODELDIR, "run_meta.json"), "w"))
shutil.copy("common/server_script.py", OUTDIR + "/script.py")
open(OUTDIR + "/requirements.txt", "w").write("")

bidx = np.random.RandomState(0).choice(len(samples), size=30000, replace=False)
bench = [samples[int(i)] for i in bidx]; t = time.time()
_ = ad_lib.predict(MODELDIR, bench, version=VERSION, max_len=MAX_LEN, batch_size=128,
                   postproc_path=MODELDIR + "/postproc.json")
dt = time.time() - t
print(f"[timing] 30k {dt:.1f}s {'OK' if dt < 600 else 'SLOW'}", flush=True)

def zipdir(src, zf):
    for root, _, fs in os.walk(src):
        for fn in fs:
            zf.write(os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), src))
zp = os.path.join(WORK, OUTDIR + ".zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    zipdir(OUTDIR, z)
tops = sorted(set(n.split("/")[0] for n in zipfile.ZipFile(zp).namelist()))
print(f"[zip] {zp} {os.path.getsize(zp)/1e6:.0f}MB tops={tops}", flush=True)
open(os.path.join(WORK, "DONE_DISTILL"), "w").write(
    f"ens_oof={ens_mf1:.4f} student_holdout={best:.4f} timing={dt:.0f}s full={FULL}")
print("=== DONE ===", flush=True)
