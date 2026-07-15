"""공용 I/O 유틸 — Dacon 236694 (AI Agent Action Decision).

- 14 클래스 정규 순서/매핑
- jsonl / labels 로더 (id 정합성 검증)
- 세션키/제너레이터/스텝 추출 (두 id 포맷 sess_sim_* / sess_au_* 모두 지원)
- seed 고정

로컬(Python 3.10)·Colab(3.11) 양쪽에서 import 가능하도록 표준 라이브러리만 사용.
"""
from __future__ import annotations
import json
import os
import random
from typing import Iterator

# 14 클래스 — 대회 규정 문자열, 정규 순서(빈도 내림차순은 아님; 고정 인덱스가 핵심)
CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
assert NUM_CLASSES == 14

# 기본 경로 (프로젝트 루트 기준). script.py(제출)는 자체 상대경로를 쓰므로 여기 미사용.
_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_THIS)
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_JSONL = os.path.join(DATA_DIR, "train.jsonl")
TRAIN_LABELS = os.path.join(DATA_DIR, "train_labels.csv")
TEST_JSONL = os.path.join(DATA_DIR, "test.jsonl")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")


# ----------------------------- id 파싱 -----------------------------
def session_id(sample_id: str) -> str:
    """세션키 = '-step_' 앞부분. 두 포맷(sess_sim_*, sess_au_*) 모두 안전.

    예) 'sess_sim_20260522_024730-step_08' -> 'sess_sim_20260522_024730'
        'sess_au_050092_004-step_04'       -> 'sess_au_050092_004'
    """
    return sample_id.rsplit("-step_", 1)[0]


def generator(sample_id: str) -> str:
    """제너레이터 태그: 'au' (sess_au_*) / 'sim' (그 외)."""
    return "au" if sample_id.startswith("sess_au_") else "sim"


def step_num(sample_id: str) -> int:
    """스텝 번호(정수). 파싱 실패 시 -1."""
    tail = sample_id.rsplit("-step_", 1)
    if len(tail) != 2:
        return -1
    try:
        return int(tail[1])
    except ValueError:
        return -1


# ----------------------------- 로더 -----------------------------
def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 파싱 실패: {e}")


def load_jsonl(path: str) -> list[dict]:
    return list(iter_jsonl(path))


def load_labels(path: str = TRAIN_LABELS) -> dict[str, str]:
    """id -> action 딕셔너리. 헤더 (id, action) 검증."""
    import csv
    out: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if header[:2] != ["id", "action"]:
            raise ValueError(f"labels 헤더가 (id, action) 아님: {header}")
        for row in r:
            out[row[0]] = row[1]
    return out


def load_train(jsonl: str = TRAIN_JSONL, labels: str = TRAIN_LABELS):
    """(samples, y, ids) 반환. 각 sample dict에 'label'/'y'/'session'/'gen'/'step' 주입.

    id 정합성(누락/불일치)을 강하게 검증한다.
    """
    lab = load_labels(labels)
    samples = load_jsonl(jsonl)
    ids = [s["id"] for s in samples]
    missing = [i for i in ids if i not in lab]
    if missing:
        raise ValueError(f"labels에 없는 id {len(missing)}건 (예: {missing[:3]})")
    y = []
    for s in samples:
        act = lab[s["id"]]
        if act not in CLASS_TO_IDX:
            raise ValueError(f"미지 클래스 '{act}' (id={s['id']})")
        s["label"] = act
        s["y"] = CLASS_TO_IDX[act]
        s["session"] = session_id(s["id"])
        s["gen"] = generator(s["id"])
        s["step"] = step_num(s["id"])
        y.append(CLASS_TO_IDX[act])
    return samples, y, ids


# ----------------------------- seed -----------------------------
def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


if __name__ == "__main__":
    s, y, ids = load_train()
    print(f"loaded {len(s)} samples, {len(set(ids))} unique ids")
    import collections
    print("gen:", collections.Counter(x["gen"] for x in s))
    print("classes ok:", set(CLASSES) == set(load_labels().values()))
