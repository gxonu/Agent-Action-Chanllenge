"""배포용 n-gram 멤버 — HashingVectorizer(무상태) + LogReg. sklearn pickle 회피.

산출:
  - teacher_ngram.npz (5fold OOF, 스태커 학습용)
  - ngram_model/ : coef.npy(K×14), intercept.npy(14), meta.json(hashing 파라미터)
배포: 서버 sklearn 1.8.0으로 HashingVectorizer 재생성(파라미터만, pickle無) → X @ coef.T + intercept → softmax.
검증: klue+large 스택에 넣어 +lift 재확인.
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.io_utils import load_train, CLASSES, NUM_CLASSES
from common.cv import make_splits
from common.metrics import macro_f1
from common.postproc import fit_bias
from common import ad_lib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import lightgbm as lgb

EXPD = "action_decision_maximum/experiments"
OUT = os.environ.get("AD_NGRAM_OUT", "artifacts/ngram_model")
NW = int(os.environ.get("AD_NW", "18"))   # word hash bits (2^NW)
NC = int(os.environ.get("AD_NC", "20"))   # char hash bits
C = float(os.environ.get("AD_C", "3.0"))
os.makedirs(OUT, exist_ok=True)

samples, y, ids = load_train(); y = np.array(y)
groups = np.array([s["session"] for s in samples])
sp = make_splits(ids, y, groups); folds = sp["folds"]; hold = sp["holdout_idx"]
cov = np.concatenate([f[1] for f in folds])
texts = [ad_lib.serialize(s, "v6") for s in samples]

WPAR = dict(analyzer="word", ngram_range=(1, 2), n_features=2 ** NW, alternate_sign=False, norm="l2")
CPAR = dict(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** NC, alternate_sign=False, norm="l2")
hvw = HashingVectorizer(**WPAR); hvc = HashingVectorizer(**CPAR)

def feat(idx):
    return hstack([hvw.transform([texts[i] for i in idx]), hvc.transform([texts[i] for i in idx])]).tocsr()

oof = np.zeros((len(samples), NUM_CLASSES), np.float32)
hold_sum = np.zeros((len(hold), NUM_CLASSES), np.float32)
for fi, (tr, va) in enumerate(folds):
    clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", n_jobs=8)
    clf.fit(feat(tr), y[tr])
    oof[va] = clf.predict_proba(feat(va))
    hold_sum += clf.predict_proba(feat(hold))
    print(f"  fold{fi}: {macro_f1(y[va], oof[va].argmax(1))[0]:.4f}", flush=True)
hold_oof = hold_sum / len(folds)
pm = macro_f1(y[cov], oof[cov].argmax(1))[0]
print(f"[ngram-hash] standalone pooled-OOF={pm:.4f}", flush=True)
np.savez_compressed(f"{EXPD}/teacher_ngram.npz", oof=oof, hold=hold_oof,
                    scores=np.array([pm]), fold_lo=0, fold_hi=5, model="hash-logreg-v6", version="v6", max_len=0)

# 배포 모델: dev 전체(cov)로 재학습 → coef/intercept 저장
clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", n_jobs=8)
clf.fit(feat(cov), y[cov])
# LogReg는 클래스 순서가 clf.classes_ → CLASSES 인덱스와 정렬
order = np.argsort(clf.classes_)   # classes_는 0..13 정렬돼있음(정수라벨) → 항등
coef = clf.coef_.astype(np.float32)          # (14, K)
intercept = clf.intercept_.astype(np.float32)
np.save(f"{OUT}/coef.npy", coef); np.save(f"{OUT}/intercept.npy", intercept)
json.dump({"wpar": WPAR, "cpar": CPAR, "classes": [int(c) for c in clf.classes_], "version": "v6"},
          open(f"{OUT}/meta.json", "w"))
sz = sum(os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT)) / 1e6
print(f"[deploy] {OUT} coef={coef.shape} size={sz:.1f}MB", flush=True)

# 스택 이득 재확인 (klue + large + ngram)
def load(k):
    pats = {"klue": ["teacher_klue_s1234.npz"],
            "large": ["teacher_largev6A_a*.npz", "teacher_largev6B_a*.npz"]}
    o = np.zeros((len(samples), NUM_CLASSES), np.float32); h = np.zeros((len(hold), NUM_CLASSES), np.float32)
    cs, w = set(), 0
    for pat in pats[k]:
        for p in ([f"{EXPD}/{pat}"] if "*" not in pat else sorted(glob.glob(f"{EXPD}/{pat}"))):
            if not os.path.exists(p): continue
            z = np.load(p, allow_pickle=True); n = 0
            for f in range(int(z["fold_lo"]), int(z["fold_hi"])):
                if f in cs: continue
                o[folds[f][1]] = z["oof"][folds[f][1]]; cs.add(f); n += 1
            if n: h += z["hold"] * n; w += n
    return o, h / max(w, 1)

To = {"klue": load("klue"), "large": load("large"), "ngram": (oof, hold_oof)}
F = np.array([ad_lib.stack_features(s) for s in samples], dtype=np.float32); Fh = F[hold]
params = dict(objective="multiclass", num_class=14, learning_rate=0.05, num_leaves=63,
             min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
             verbose=-1, num_threads=8, seed=1234)
def stack(keys):
    P = np.concatenate([To[k][0] for k in keys] + [F], axis=1); Ph = np.concatenate([To[k][1] for k in keys] + [Fh], axis=1)
    om = np.zeros((len(samples), NUM_CLASSES), np.float32)
    for tr, va in folds:
        om[va] = lgb.train(params, lgb.Dataset(P[tr], label=y[tr]), num_boost_round=400).predict(P[va])
    b, _ = fit_bias(np.log(om[cov] + 1e-9), y[cov])
    hp = lgb.train(params, lgb.Dataset(P[cov], label=y[cov]), num_boost_round=400).predict(Ph)
    hb = macro_f1(y[hold], (np.log(hp + 1e-9) + np.array(b)).argmax(1))[0]
    print(f"[{'+'.join(keys)}] holdout+bias={hb:.4f}"); return hb
b2 = stack(["klue", "large"]); b3 = stack(["klue", "large", "ngram"])
print(f"\n판정: hash-ngram {'채택(+' + format(b3-b2,'.4f') + ')' if b3 > b2 + 0.002 else '기각'}")
