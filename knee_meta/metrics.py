"""Macro AUC for the competition metric and per-target percentile ranks."""
import numpy as np
from sklearn.metrics import roc_auc_score

from .config import TARGETS


def macro_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    aucs = []
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if col.min() == col.max():
            continue
        aucs.append(roc_auc_score(col, y_score[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def per_label_auc(y_true, y_score):
    out = {}
    for i, lab in enumerate(TARGETS):
        col = np.asarray(y_true)[:, i]
        if col.min() == col.max():
            out[lab] = float("nan")
        else:
            out[lab] = float(roc_auc_score(col, np.asarray(y_score)[:, i]))
    return out


def percentile_rank(scores):
    scores = np.asarray(scores, dtype=np.float64)
    out = np.empty_like(scores)
    n = scores.shape[0]
    order = np.argsort(scores, axis=0, kind="mergesort")
    ranks = np.empty((n, scores.shape[1]), dtype=np.float64)
    for c in range(scores.shape[1]):
        col = order[:, c]
        r = np.empty(n, dtype=np.float64)
        r[col] = np.arange(n, dtype=np.float64)
        vals = scores[col, c]
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[j + 1] == vals[i]:
                j += 1
            if j > i:
                r[col[i:j + 1]] = 0.5 * (i + j)
            i = j + 1
        ranks[:, c] = r
    out = (ranks + 0.5) / n
    return out
