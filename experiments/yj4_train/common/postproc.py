"""Macro-F1 후처리 — per-class additive bias 좌표상승 최적화 (HR-2: pooled-OOF 기준).

argmax(logits + bias) 의 macro-F1 을 최대화하는 bias(14,) 를 coarse 좌표상승으로 탐색.
학습 재실행 불필요. 희귀클래스 재현율↑ → macro-F1↑ (흔히 +1~3p).
저장/로드는 순수 json (버전 안전).
"""
from __future__ import annotations
import json
import numpy as np
from .metrics import macro_f1
from .io_utils import CLASSES, NUM_CLASSES


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def to_logprobs(logits):
    """logits(N,C) → log-prob. bias는 log-prob 공간에서 더한다(스케일 안정)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
    return z - lse


def fit_bias(logits, y, n_classes=NUM_CLASSES, rounds=8, seed=42):
    """coarse→fine 좌표상승으로 per-class bias 탐색.

    반환: (bias(np.array C,), best_macro_f1)
    """
    lp = to_logprobs(logits)
    y = np.asarray(y)
    bias = np.zeros(n_classes, dtype=np.float64)

    def score(b):
        return macro_f1(y, (lp + b).argmax(1), n_classes)[0]

    best = score(bias)
    grids = [np.linspace(-2.0, 2.0, 21), np.linspace(-1.0, 1.0, 21),
             np.linspace(-0.4, 0.4, 17)]
    rng = np.random.default_rng(seed)
    for gi, grid in enumerate(grids):
        improved = True
        it = 0
        while improved and it < rounds:
            improved = False
            it += 1
            order = rng.permutation(n_classes)
            for c in order:
                cur = bias[c]
                best_v, best_s = cur, best
                for delta in grid:
                    bias[c] = cur + delta
                    s = score(bias)
                    if s > best_s + 1e-6:
                        best_s, best_v = s, bias[c]
                bias[c] = best_v
                if best_s > best + 1e-9:
                    best = best_s
                    improved = True
    return bias, best


def apply_bias(logits, bias):
    return (to_logprobs(logits) + np.asarray(bias)).argmax(1)


def save(path, bias, meta=None):
    obj = {"bias": list(map(float, np.asarray(bias).ravel())),
           "classes": CLASSES}
    if meta:
        obj["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load(path):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if obj.get("classes") != CLASSES:
        raise ValueError("postproc.json 클래스 순서 불일치")
    return np.array(obj["bias"], dtype=np.float64)
