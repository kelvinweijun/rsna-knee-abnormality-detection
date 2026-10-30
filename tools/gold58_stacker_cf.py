# -*- coding: utf-8 -*-
"""Cross-fitted stack evaluator: honest gold58+gold74 gates + deployment weights.

Unlike gold58_stacker.py (grid fit and eval on the same 58 studies), this fits the
per-label simplex weights on K-1 folds and scores the held-out fold, so the reported
macro-AUC is an honest generalization estimate. Deployment weights are refit on all
studies. Bootstrap resamples studies on the fixed cross-fitted predictions.

Usage:
  python gold58_stacker_cf.py --arm-dir DIR [--gold74 CSV] [--train CSV] [--out report.txt]
"""
import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SHRINK = 0.5
STEPS = 7
KFOLD = 5
BOOT = 2000
SEED = 0


def simplex_grid(m, steps=STEPS):
    out = []
    def rec(prefix, left, k):
        if k == 1:
            out.append(tuple(prefix + [left]))
            return
        for v in range(left + 1):
            rec(prefix + [v], left - v, k - 1)
    rec([], steps, m)
    return np.array(out, dtype=np.float64) / steps


def rank01(x):
    """Average-tie rank percentiles in [0,1]."""
    order = argsort_stable(x)
    r = np.empty(len(x)); r[order] = np.arange(len(x))
    # tie-average
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
    return sums[inv] / cnt[inv] / max(len(x) - 1, 1)


def argsort_stable(x):
    return np.argsort(x, kind="stable")


def fit_label(preds, y, grid, uni):
    """Grid-search weights for one label. preds (n,a) rank01, y binary."""
    if len(np.unique(y)) < 2:
        return uni.copy(), np.nan
    scores = preds @ grid.T                      # (n, g)
    # AUC for every grid column via vectorized rank-sum
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    order = np.argsort(scores, axis=0, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.arange(len(y))[:, None].repeat(scores.shape[1], 1), axis=0)
    rsum = ranks[pos].sum(axis=0)
    auc = (rsum - n1 * (n1 + 1) / 2) / (n1 * n0)
    w = SHRINK * grid[int(np.argmax(auc))] + (1 - SHRINK) * uni
    return w, float(auc.max())


def eval_stack(P, y, W):
    """P (n,a,12) preds, W (12,a) per-label weights, y (n,12)."""
    out = np.einsum("naj,ja->nj", P, W)
    return float(np.mean([roc_auc_score(y[:, j], out[:, j]) for j in range(12)]))


def load_truth(gold74, train_csv):
    if gold74:
        g = pd.read_csv(gold74, dtype={"StudyInstanceUID": str})
        y = g.set_index("StudyInstanceUID")[LAB].astype(float).dropna()
        return y, "gold74"
    t = pd.read_csv(train_csv, dtype={"StudyInstanceUID": str})
    y = t.set_index("StudyInstanceUID")[LAB].dropna()
    return y, "gold58"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--gold74", default="")
    ap.add_argument("--train", default=r"C:\Users\kelvi\Downloads\RSNA Detection\train.csv")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.arm_dir, "arm_*_58.csv")))
    paths = [p for p in paths if "arm_multi_rank" not in p]
    names, A = [], []
    for p in paths:
        tag = os.path.basename(p).replace("arm_", "").split("_58")[0]
        df = pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        names.append(tag); A.append(df)
    assert names, f"no arm csvs under {args.arm_dir}"
    ytab, tname = load_truth(args.gold74 or None, args.train)
    common = ytab.index
    for df in A:
        common = common.intersection(df.index)
    ytab = ytab.loc[common].values
    ids = list(common)
    n, a = len(ids), len(names)
    P = np.stack([df.loc[ids, LAB].values for df in A], 1)   # (n, a, 12)
    print(f"loaded {a} arms x {n} studies ({tname})", flush=True)

    grid = simplex_grid(a)
    uni = np.full(a, 1.0 / a)
    print(f"simplex grid: {len(grid)} points x {a} arms", flush=True)

    gates = [float(np.mean([roc_auc_score(ytab[:, j], P[:, i, j]) for j in range(12)]))
             for i in range(a)]
    for nm, g in sorted(zip(names, gates), key=lambda kv: -kv[1]):
        print(f"  {nm:16s} {g:.4f}", flush=True)

    # ---- cross-fit honest gate ----
    rng = np.random.RandomState(SEED)
    folds = np.arange(n) % KFOLD
    rng.shuffle(folds)
    CF = np.zeros((n, 12))
    Wfit = np.zeros((12, a))
    for f in range(KFOLD):
        tr, va = folds != f, folds == f
        for j in range(12):
            wj, _ = fit_label(P[tr][:, :, j], ytab[tr, j], grid, uni)
            Wfit[j] = wj
            CF[va, j] = P[va][:, :, j] @ wj
    cf_gate = float(np.mean([roc_auc_score(ytab[:, j], CF[:, j]) for j in range(12)]))
    ctrl = float(np.mean([roc_auc_score(ytab[:, j], P[:, names.index("nartaa_r384"), j])
                          for j in range(12)])) if "nartaa_r384" in names else np.nan
    uni_gate = eval_stack(P, ytab, np.tile(uni, (12, 1)))
    print(f"\ncross-fitted shrunk stack: {cf_gate:.4f} | uniform: {uni_gate:.4f} | "
          f"nartaa_r384 control: {ctrl:.4f}", flush=True)

    # ---- bootstrap of cross-fitted vs control ----
    if not np.isnan(ctrl):
        diffs = []
        for _ in range(BOOT):
            idx = rng.randint(0, n, n)
            try:
                d = (float(np.mean([roc_auc_score(ytab[idx, j], CF[idx, j]) for j in range(12)]))
                     - float(np.mean([roc_auc_score(ytab[idx, j], P[idx, names.index('nartaa_r384'), j])
                                      for j in range(12)])))
                diffs.append(d)
            except ValueError:
                pass
        diffs = np.array(diffs)
        print(f"bootstrap cf-stack - control: mean {diffs.mean():+.4f} | "
              f"p10 {np.percentile(diffs, 10):+.4f} | p90 {np.percentile(diffs, 90):+.4f} | "
              f"P(>0) {(diffs > 0).mean():.3f}", flush=True)

    # ---- deployment weights on all studies ----
    Wdep = np.zeros((12, a))
    for j in range(12):
        Wdep[j], _ = fit_label(P[:, :, j], ytab[:, j], grid, uni)
    dep_gate = eval_stack(P, ytab, Wdep)
    print(f"\ndeployment (all-data fit, in-sample): {dep_gate:.4f}", flush=True)
    print("deployment weights:", flush=True)
    for j, lab in enumerate(LAB):
        ws = " ".join(f"{nm}={Wdep[j, i]:.2f}" for i, nm in enumerate(names) if Wdep[j, i] > 0.01)
        print(f"  {lab:18s} {ws}", flush=True)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(f"arms: {names}\ntruth: {tname} n={n}\n")
            fh.write(f"per-arm gates: {dict(zip(names, np.round(gates, 4)))}\n")
            fh.write(f"cross-fitted: {cf_gate:.4f} uniform: {uni_gate:.4f} control: {ctrl:.4f}\n")
            fh.write(f"deployment in-sample: {dep_gate:.4f}\n")
            fh.write("Wdep=\n" + np.array2string(Wdep, precision=2) + "\n")
        print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
