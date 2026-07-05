"""
Baseline (TF-IDF + LogReg) 10-fold OOF 학습·평가.

베이스라인 pkl 파라미터 재현 (`baseline/model/tfidf_logreg.pkl`에서 확인한 값):
- TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2, max_features=80000,
                  sublinear_tf=True, use_idf=True, norm='l2', token_pattern=r'(?u)\\b\\w\\w+\\b')
- LogisticRegression(C=2.0, class_weight='balanced', solver='lbfgs', max_iter=500)

입력: current_prompt 문자열만 (베이스라인과 동일)

산출:
- data/processed/baseline_oof.csv (id, fold, action_true, action_pred)
- data/processed/baseline_oof_probs.npz (ids, probs, classes)

실행:
    python -m src.baseline_oof
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.cv import CV_DIR, PROCESSED_DIR, ROOT, load_fold_assignments

RAW_DIR = ROOT / "data" / "raw"
OOF_CSV = PROCESSED_DIR / "baseline_oof.csv"
OOF_PROBS = PROCESSED_DIR / "baseline_oof_probs.npz"


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=80000,
            sublinear_tf=True,
            use_idf=True,
            norm="l2",
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b",
        )),
        ("clf", LogisticRegression(
            C=2.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=500,
        )),
    ])


def load_prompts_by_id() -> dict[str, str]:
    """train.jsonl에서 id → current_prompt 매핑."""
    d: dict[str, str] = {}
    with open(RAW_DIR / "train.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            d[obj["id"]] = obj.get("current_prompt", "") or ""
    return d


def run_oof() -> None:
    print("### baseline OOF 학습 시작")
    print(f"CV: fold_assignments.csv 로드")
    rows = load_fold_assignments()
    print(f"  총 {len(rows):,}개 샘플, fold 10개")

    print("Prompt 로드")
    prompts = load_prompts_by_id()

    # 정렬된 id 배열 (fold_assignments.csv 순서 그대로)
    ids = [r["id"] for r in rows]
    y_true = np.array([r["action"] for r in rows])
    folds = np.array([r["fold"] for r in rows])
    texts = np.array([prompts[i] for i in ids])

    # 클래스 라벨 통일 (알파벳 순, sklearn 기본)
    all_classes = sorted(set(y_true.tolist()))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    print(f"  클래스 {len(all_classes)}개: {all_classes[:3]}... (알파벳순)")

    n = len(ids)
    n_cls = len(all_classes)
    oof_pred = np.empty(n, dtype=object)
    oof_probs = np.zeros((n, n_cls), dtype=np.float32)

    fold_scores: list[dict] = []
    from sklearn.metrics import f1_score

    for k in range(10):
        t0 = time.time()
        val_mask = folds == k
        tr_mask = ~val_mask
        print(f"\n[fold {k}] train {tr_mask.sum():,} / val {val_mask.sum():,}")
        pipe = make_pipeline()
        pipe.fit(texts[tr_mask], y_true[tr_mask])
        pred = pipe.predict(texts[val_mask])
        probs = pipe.predict_proba(texts[val_mask])
        # pipe.classes_와 우리 all_classes 순서 맞추기 (알파벳 순이라 대개 동일)
        pipe_classes = list(pipe.classes_)
        col_map = [pipe_classes.index(c) for c in all_classes]
        probs_aligned = probs[:, col_map]

        oof_pred[val_mask] = pred
        oof_probs[val_mask] = probs_aligned

        macro = f1_score(y_true[val_mask], pred, labels=all_classes, average="macro", zero_division=0)
        dt = time.time() - t0
        print(f"  fold {k} val Macro-F1 = {macro:.4f}  ({dt:.1f}s)")
        fold_scores.append({"fold": k, "macro_f1": float(macro), "n_val": int(val_mask.sum()), "sec": dt})

    # 저장
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n저장: {OOF_CSV}")
    with open(OOF_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "fold", "action_true", "action_pred"])
        w.writeheader()
        for i in range(n):
            w.writerow({
                "id": ids[i],
                "fold": int(folds[i]),
                "action_true": y_true[i],
                "action_pred": oof_pred[i],
            })

    print(f"저장: {OOF_PROBS}")
    np.savez(
        OOF_PROBS,
        ids=np.array(ids),
        probs=oof_probs,
        classes=np.array(all_classes),
        folds=folds,
    )

    # fold별 스코어 요약
    print("\n### fold별 Macro-F1")
    for s in fold_scores:
        print(f"  fold {s['fold']}: {s['macro_f1']:.4f} (n={s['n_val']:,}, {s['sec']:.1f}s)")
    m = np.mean([s["macro_f1"] for s in fold_scores])
    sd = np.std([s["macro_f1"] for s in fold_scores])
    print(f"평균: {m:.4f} ± {sd:.4f}")


if __name__ == "__main__":
    run_oof()
