"""평가 지표 — pooled-OOF macro-F1 (주지표) + per-class 리포트.

HR-2: fold별 F1 평균이 아니라, 5-fold OOF 예측을 이어붙여 macro-F1 1회 계산.
"""
from __future__ import annotations
import numpy as np
from .io_utils import CLASSES, NUM_CLASSES


def macro_f1(y_true, y_pred, n_classes=NUM_CLASSES):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    f1s = np.zeros(n_classes)
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s[c] = 2 * p * r / (p + r) if (p + r) else 0.0
    return float(f1s.mean()), f1s


def per_class_report(y_true, y_pred, n_classes=NUM_CLASSES):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    _, f1s = macro_f1(y_true, y_pred, n_classes)
    rows = []
    for c in range(n_classes):
        support = int(np.sum(y_true == c))
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append((CLASSES[c], support, round(p, 4), round(r, 4), round(f1s[c], 4)))
    return rows


def print_report(y_true, y_pred, title="report"):
    mf1, _ = macro_f1(y_true, y_pred)
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    print(f"\n=== {title} === pooled macro-F1={mf1:.4f}  acc={acc:.4f}")
    print(f"{'class':20} {'sup':>6} {'prec':>7} {'rec':>7} {'f1':>7}")
    for name, sup, p, r, f1 in sorted(per_class_report(y_true, y_pred), key=lambda x: x[4]):
        print(f"{name:20} {sup:6d} {p:7.4f} {r:7.4f} {f1:7.4f}")
    return mf1, acc
