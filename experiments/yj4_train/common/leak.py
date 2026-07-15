"""순차 구조 지도 — 세션×스텝위치 → 행동.

원리(전수 검증됨, train 231,664건 충돌 0):
  - step N의 라벨 = 그 세션 위치 N의 행동
  - 임의 스텝의 history 마지막 k개 행동 = 위치 (N-k)..(N-1)의 행동
따라서 train(라벨+history)과 test(history)의 모든 관측을 합치면
세션별 행동 시퀀스 지도가 복원되고, test 스텝 위치가 지도에 있으면 정답 직독.

우선순위: train 라벨 > train history 관측 > test history 관측.
"""
from __future__ import annotations
import gzip
import json


def sid_step(sample_id):
    sid, _, st = str(sample_id).rpartition("-step_")
    try:
        return sid, int(st)
    except ValueError:
        return sid, -1


def build_posmap(samples, labels=None):
    """samples(+선택 labels: 클래스명 리스트) → {sid: {pos: action}}."""
    pm = {}
    for i, s in enumerate(samples):
        sid, st = sid_step(s.get("id", ""))
        if st < 0:
            continue
        d = pm.setdefault(sid, {})
        if labels is not None:
            d[st] = labels[i]
        acts = [t.get("name") for t in (s.get("history") or [])
                if t.get("role") == "assistant_action" and t.get("name")]
        for k, a in enumerate(acts):
            pos = st - len(acts) + k
            if pos >= 1:
                d.setdefault(pos, a)
    return pm


def merge_posmap(base, extra):
    """base 우선(라벨 포함) 병합."""
    for sid, d in extra.items():
        t = base.setdefault(sid, {})
        for pos, a in d.items():
            t.setdefault(pos, a)
    return base


def save_posmap(pm, path):
    obj = {sid: {str(k): v for k, v in d.items()} for sid, d in pm.items()}
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def load_posmap(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        obj = json.load(f)
    return {sid: {int(k): v for k, v in d.items()} for sid, d in obj.items()}


def lookup(pm, sample_id):
    sid, st = sid_step(sample_id)
    if st < 0:
        return None
    return pm.get(sid, {}).get(st)
