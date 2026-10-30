"""Multi-arm additive fit on gold74: which extra arms, mixed into the h957
baseline b0, raise gold74 macro-AUC (and by how much)?

Same b0 replication as fit_w_arm2.py. For every candidate arm CSV in
%TEMP%\\arm_g74 (excluding raptor384/r224 which are structurally in b0),
grids  b(w) = (1 - w) * b0 + w * rankpct(arm)  for w in {0.05..0.30} and
prints standalone AUC, spearman vs b0, and best delta. Then, for the top
additive arms, grids pairwise combos  b = (1 - w1 - w2) * b0 + w1*a1 + w2*a2.

Reads:
  %TEMP%\\r7gold_out\\arm_r7_74.csv, %TEMP%\\arm_g74\\arm_*_58.csv,
  %TEMP%\\jev_labels\\gold74_eval_set.csv
"""
import os
import sys
import glob
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T = os.environ.get("TEMP", "/tmp")
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
A224 = 0.35
W_RAPTOR = 0.5
W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
IN_BLEND = {"arm_nartaa_r384", "arm_nartaa_r224"}          # inside b0 by construction
SKIP = {"arm_arm3cnv"}                     # arm3 tested separately: no gain (redundant with arm2)
GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
COMBO_GRID = [0.05, 0.10, 0.15, 0.20]


def load(path):
    d = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    return d.set_index("StudyInstanceUID")[LAB]


def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method="average", pct=True).values


def rankpct(d, uid, name):
    arr = d.loc[uid, LAB].values.astype(np.float64)
    if (arr == 0.5).all(axis=1).any():
        raise RuntimeError(f"{name}: all-0.5 rows")
    return pd.DataFrame(np.stack([pct(arr[:, j]) for j in range(len(LAB))], axis=1),
                        index=uid, columns=LAB)


def macro_auc(scores, labels_df):
    out = []
    s = scores.loc[labels_df.index]
    for c in LAB:
        y_raw = labels_df[c].values
        mask = ~pd.isna(y_raw)
        y = y_raw[mask].astype(int)
        if len(np.unique(y)) < 2:
            continue
        out.append(roc_auc_score(y, s[c].values[mask]))
    return float(np.mean(out))


def main():
    gold = pd.read_csv(os.path.join(T, "jev_labels", "gold74_eval_set.csv"),
                       dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    uid = gold.index

    r7 = load(os.path.join(T, "r7gold_out", "arm_r7_74.csv"))
    r384 = load(os.path.join(T, "arm_g74", "arm_nartaa_r384_58.csv"))
    r224 = load(os.path.join(T, "arm_g74", "arm_nartaa_r224_58.csv"))
    for nm, d in (("r7", r7), ("r384", r384), ("r224", r224)):
        assert sorted(d.index) == sorted(uid), f"{nm}: uid mismatch"

    R384, R224, RO = (rankpct(d, uid, n) for d, n in
                      ((r384, "r384"), (r224, "r224"), (r7, "r7")))
    rr = (1.0 - A224) * R384 + A224 * R224
    b0 = pd.DataFrame({c: W_RAPTOR_C.get(c, W_RAPTOR) * rr[c]
                       + (1.0 - W_RAPTOR_C.get(c, W_RAPTOR)) * RO[c] for c in LAB},
                      index=uid)
    auc0 = macro_auc(b0, gold)
    print(f"h957 baseline gold74: {auc0:.4f}")

    results = []
    for p in sorted(glob.glob(os.path.join(T, "arm_g74", "arm_*_58.csv"))):
        name = os.path.basename(p)[:-len("_58.csv")]
        if name in IN_BLEND or name in SKIP:
            continue
        try:
            d = load(p)
            if sorted(d.index) != sorted(uid):
                print(f"{name}: uid mismatch ({len(d)} rows) - SKIP")
                continue
            A = rankpct(d, uid, name)
        except Exception as e:
            print(f"{name}: load failed ({e}) - SKIP")
            continue
        solo = macro_auc(A, gold)
        rho = float(np.mean([pd.Series(A[c].values).corr(pd.Series(b0[c].values), method="spearman")
                             for c in LAB]))
        best = (None, -9)
        for w in GRID:
            b = (1.0 - w) * b0 + w * A
            a = macro_auc(b, gold)
            if a > best[1]:
                best = (w, a)
        results.append((name, solo, rho, best[0], best[1] - auc0))
        print(f"{name:>22}  solo {solo:.4f}  rho_vs_b0 {rho:.3f}  "
              f"best w={best[0]:.2f}  delta {best[1] - auc0:+.4f}")

    print("\n== top additive arms (delta >= +0.0010) ==")
    good = sorted([r for r in results if r[4] >= 0.0010], key=lambda r: -r[4])
    if not good:
        print("none - no single arm adds anything on gold74")
        return 0
    for r in good:
        print(f"  {r[0]:>22}  delta {r[4]:+.4f} at w={r[3]:.2f}")

    print("\n== pairwise combos on top of b0 ==")
    arms = {}
    for name, *_ in good[:6]:
        arms[name] = rankpct(load(os.path.join(T, "arm_g74", f"{name}_58.csv")), uid, name)
    top = {"auc": auc0, "desc": "baseline b0"}
    for (n1, a1), (n2, a2) in itertools.combinations(arms.items(), 2):
        for w1 in COMBO_GRID:
            for w2 in COMBO_GRID:
                if w1 + w2 > 0.40:
                    continue
                b = (1.0 - w1 - w2) * b0 + w1 * a1 + w2 * a2
                a = macro_auc(b, gold)
                if a > top["auc"]:
                    top = {"auc": a, "desc": f"{n1}@{w1:.2f} + {n2}@{w2:.2f}",
                           "delta": a - auc0}
                    print(f"  new best {a:.4f}: {top['desc']} ({a - auc0:+.4f})")
    print(f"best combo: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
