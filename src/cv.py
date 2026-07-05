"""
팀 공유 CV split 유틸.

상혁 v01(LB 0.77585) 코드의 split 로직과 완전히 정렬:
- StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
- Groups = session_id (id에서 -step_NN 잘라낸 것)
- Stratify = train_labels.csv의 action 클래스
- fold 0 = val 기본값 (상혁 방식과 동일)

산출:
- data/processed/cv_splits/fold_assignments.csv (id, session, fold, action)
- 팀원 누구든 이 CSV 읽어 fold==k인 row를 val로, 나머지를 train으로 사용

사용 예:
    python -m src.cv                          # fold 계산 + 저장
    from src.cv import load_fold_assignments  # 다른 스크립트에서 로드
    df = load_fold_assignments()
    val_ids = df[df.fold == 0].id.tolist()
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# 프로젝트 루트 (src/cv.py 기준 한 단계 위)
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CV_DIR = PROCESSED_DIR / "cv_splits"

# 상혁 정렬 상수 (config.py와 동일)
SEED = 42
N_SPLITS = 10
DEFAULT_VAL_FOLD = 0


def _session_id_of(sample_id: str) -> str:
    """sess_XXX_YYYYYY-step_NN → sess_XXX_YYYYYY"""
    return re.sub(r"-step_\d+$", "", sample_id)


def compute_fold_assignments(
    train_jsonl: Path = RAW_DIR / "train.jsonl",
    train_labels_csv: Path = RAW_DIR / "train_labels.csv",
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> list[dict]:
    """
    train.jsonl 순서대로 id를 뽑고, 라벨/세션과 조인해 fold 할당.

    상혁 코드와 동일 로직:
      groups = session_id 배열 (같은 세션은 같은 fold로만)
      y = 14 클래스 라벨 (각 fold의 클래스 분포 유사)
      sgkf.split()의 순서: fold_0 val이 나오도록 next()로 첫 fold부터

    Returns:
        [{id, session, action, fold}, ...] — len == 샘플 수
    """
    # 라벨 로드
    labels: dict[str, str] = {}
    with open(train_labels_csv) as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    # jsonl 순서대로 id 목록
    ids: list[str] = []
    with open(train_jsonl) as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])

    # 세션·라벨 벡터
    sessions = np.array([_session_id_of(i) for i in ids])
    y_str = np.array([labels[i] for i in ids])
    # StratifiedGroupKFold는 정수 라벨을 좋아함
    classes = sorted(set(y_str.tolist()))
    label2id = {c: i for i, c in enumerate(classes)}
    y = np.array([label2id[c] for c in y_str])

    # 상혁과 동일한 split 파라미터
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    fold_of = np.full(len(ids), -1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(np.arange(len(ids)), y, sessions)):
        fold_of[val_idx] = fold_idx

    assert (fold_of >= 0).all(), "일부 샘플에 fold가 할당 안 됨"

    rows = [
        {"id": ids[i], "session": sessions[i], "action": y_str[i], "fold": int(fold_of[i])}
        for i in range(len(ids))
    ]
    return rows


def save_fold_assignments(rows: list[dict], out_path: Optional[Path] = None) -> Path:
    """fold 할당을 CSV로 저장."""
    if out_path is None:
        CV_DIR.mkdir(parents=True, exist_ok=True)
        out_path = CV_DIR / "fold_assignments.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "session", "action", "fold"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_path


def load_fold_assignments(path: Optional[Path] = None) -> list[dict]:
    """fold 할당 CSV 로드. 없으면 파일부터 만들어서 로드."""
    if path is None:
        path = CV_DIR / "fold_assignments.csv"
    if not path.exists():
        rows = compute_fold_assignments()
        save_fold_assignments(rows, path)
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["fold"] = int(row["fold"])
            out.append(row)
    return out


def get_val_ids(fold: int = DEFAULT_VAL_FOLD) -> list[str]:
    """지정 fold의 id 목록 (val 용)."""
    rows = load_fold_assignments()
    return [r["id"] for r in rows if r["fold"] == fold]


def get_train_ids(fold: int = DEFAULT_VAL_FOLD) -> list[str]:
    """지정 fold 외 모든 id (train 용)."""
    rows = load_fold_assignments()
    return [r["id"] for r in rows if r["fold"] != fold]


if __name__ == "__main__":
    print(f"CV split 계산 (n_splits={N_SPLITS}, seed={SEED})...")
    rows = compute_fold_assignments()
    out_path = save_fold_assignments(rows)
    print(f"→ saved: {out_path}")
    print(f"총 샘플: {len(rows):,}")
    from collections import Counter
    c = Counter(r["fold"] for r in rows)
    for k in sorted(c.keys()):
        print(f"  fold_{k}: {c[k]:,}건")
