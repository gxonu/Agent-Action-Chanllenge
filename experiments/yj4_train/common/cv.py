"""CV 설계 — 세션단위 프로즌 홀드아웃 + StratifiedGroupKFold.

- 그룹키 = session_id (같은 세션 step 누수 방지; 테스트=미지 세션 가정)
- 전 트랙 공유 인덱스를 splits/ 에 캐시(재현·스태킹 정합)
- fold별 클래스 카운트 표(0셀 감시)
"""
from __future__ import annotations
import os, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(os.path.dirname(_HERE), "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)


def make_splits(ids, y, groups, holdout_frac=0.08, n_splits=5, seed=42, force=False):
    """세션단위 홀드아웃 + dev셋 StratifiedGroupKFold.

    반환 dict: {holdout_idx, dev_idx, folds:[(tr,va),...], n_splits}
    (인덱스는 전체 배열 기준 절대 인덱스)
    캐시: splits/splits.npz
    """
    cache = os.path.join(SPLITS_DIR, "splits.npz")
    meta = os.path.join(SPLITS_DIR, "splits_meta.json")
    if os.path.exists(cache) and not force:
        d = np.load(cache, allow_pickle=True)
        folds = [(d[f"tr{i}"], d[f"va{i}"]) for i in range(int(d["n_splits"]))]
        return {"holdout_idx": d["holdout_idx"], "dev_idx": d["dev_idx"],
                "folds": folds, "n_splits": int(d["n_splits"])}

    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    y = np.asarray(y)
    groups = np.asarray(groups)
    idx_all = np.arange(len(y))

    # 1) 세션단위 홀드아웃
    gss = GroupShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
    dev_rel, hold_rel = next(gss.split(idx_all, y, groups))
    dev_idx = idx_all[dev_rel]
    holdout_idx = idx_all[hold_rel]

    # 2) dev셋에서 StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_rel, va_rel in sgkf.split(dev_idx, y[dev_idx], groups[dev_idx]):
        folds.append((dev_idx[tr_rel], dev_idx[va_rel]))

    # 저장
    save = {"holdout_idx": holdout_idx, "dev_idx": dev_idx, "n_splits": n_splits}
    for i, (tr, va) in enumerate(folds):
        save[f"tr{i}"] = tr; save[f"va{i}"] = va
    np.savez(cache, **save)
    # 홀드아웃 세션과 dev 세션 disjoint 검증
    assert set(groups[holdout_idx]).isdisjoint(set(groups[dev_idx])), "홀드아웃 세션 누수!"
    with open(meta, "w", encoding="utf-8") as f:
        json.dump({"holdout_frac": holdout_frac, "n_splits": n_splits, "seed": seed,
                   "n_holdout": int(len(holdout_idx)), "n_dev": int(len(dev_idx))}, f, indent=2)
    return {"holdout_idx": holdout_idx, "dev_idx": dev_idx, "folds": folds, "n_splits": n_splits}


def fold_class_counts(y, folds, n_classes=14):
    """fold별 validation 클래스 카운트 표 (0셀 감시)."""
    y = np.asarray(y)
    table = np.zeros((len(folds), n_classes), dtype=int)
    for i, (_, va) in enumerate(folds):
        table[i] = np.bincount(y[va], minlength=n_classes)
    return table


def leakage_gap_report(ids, y, groups, n_splits=5, seed=42):
    """GroupKFold vs 일반 StratifiedKFold의 낙관 gap 정량화용 헬퍼(설명 목적)."""
    return {"note": "run a model under both splitters; compare pooled-OOF macro-F1"}
