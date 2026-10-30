"""Fit W_ARM2: h957's gold74 blend score with the new arm mixed in at weight w2.

Replicates the h957 stage-3 formula on gold74 (Raptor half 0.5 + R7 half 0.5,
A224 = 0.35, per-finding Raptor weights from W_RAPTOR_C = reweight5, heliosli
decision 10-16) and grids b(w2) = (1 - w2) * b0 + w2 * rankpct(arm2) per finding.

Inputs (local mirrors of the kernels' outputs):
  %TEMP%\\r7gold_out\\arm_r7_74.csv          compact R7 on gold74   (stage 2 "ours")
  %TEMP%\\arm_g74\\arm_nartaa_r384_58.csv   raptor 384            (stage 1)
  %TEMP%\\arm_g74\\arm_nartaa_r224_58.csv   raptor 224            (stage 1b)
  %TEMP%\\arm_g74\\arm_arm2hyb_58.csv       our new arm           (arm2_gold kernel)
  %TEMP%\\jev_labels\\gold74_eval_set.csv   gold74 labels (source col splits official58)

Without the arm2 file this prints the h957 baseline on gold74 and exits 0
(preflight check). With it, prints the w2 grid; W_ARM2 should be taken from the
interior of a plateau, then re-checked with the shrinkage stacker before the
scored run.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T = os.environ.get("TEMP", "/tmp")
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
A224 = 0.35     # must match h957 cell 8
W_RAPTOR = 0.5
W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}   # reweight5: per-finding Raptor weight (ours = 1 - it)
GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def load(path):
    d = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    d = d.set_index("StudyInstanceUID")[LAB]
    return d


def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method="average", pct=True).values


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
    return float(np.mean(out)), out


def main():
    gold = pd.read_csv(os.path.join(T, "jev_labels", "gold74_eval_set.csv"),
                       dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    gold = gold[LAB]
    print("gold rows:", len(gold), "| source counts:",
          pd.read_csv(os.path.join(T, "jev_labels", "gold74_eval_set.csv"))["source"].value_counts().to_dict())

    r7 = load(os.path.join(T, "r7gold_out", "arm_r7_74.csv"))
    r384 = load(os.path.join(T, "arm_g74", "arm_nartaa_r384_58.csv"))
    r224 = load(os.path.join(T, "arm_g74", "arm_nartaa_r224_58.csv"))
    uid = gold.index
    for nm, d in (("r7", r7), ("r384", r384), ("r224", r224)):
        assert sorted(d.index) == sorted(uid), f"{nm}: uid mismatch"

    # ---- h957 stage-3 formula on gold ----
    def rankpct_all(d):
        arr = d.loc[uid, LAB].values.astype(np.float64)
        fail = (arr == 0.5).all(axis=1)
        if fail.any():
            raise RuntimeError(f"gold arm has {int(fail.sum())} all-0.5 rows")
        return pd.DataFrame(np.stack([pct(arr[:, j]) for j in range(len(LAB))], axis=1),
                            index=uid, columns=LAB), fail

    R384, f384 = rankpct_all(r384)
    R224, f224 = rankpct_all(r224)
    RO, fo = rankpct_all(r7)
    rr = (1.0 - A224) * R384 + A224 * R224          # raptor half
    b0 = pd.DataFrame({c: W_RAPTOR_C.get(c, W_RAPTOR) * rr[c]
                       + (1.0 - W_RAPTOR_C.get(c, W_RAPTOR)) * RO[c] for c in LAB},
                      index=uid)                    # h957 blend (reweight5 per-finding weights)
    auc0, per0 = macro_auc(b0, gold)
    print(f"h957 baseline on gold74: {auc0:.4f}")
    print("per-label baseline:", {c: round(a, 4) for c, a in zip(LAB, per0)})

    # ---- arm2 (optional until the gold kernel has run) ----
    cands = sorted(glob.glob(os.path.join(T, "arm_g74", "arm_*_58.csv")))
    a2path = [p for p in cands if "arm2" in os.path.basename(p)]
    a2path += [p for p in cands if "arm2hyb" in os.path.basename(p)]
    if not a2path:
        print("arm2 gold preds not found yet (arm_g74/arm_arm2hyb_58.csv) - baseline only")
        return 0
    A2, fa2 = rankpct_all(load(a2path[0]))
    print("arm2 file:", a2path[0])

    # diversity vs the arms already in the blend
    for nm, R in (("raptor384", R384), ("r7", RO)):
        rhos = []
        for c in LAB:
            rhos.append(pd.Series(A2[c].values).corr(pd.Series(R[c].values), method="spearman"))
        print(f"spearman arm2 vs {nm}: mean {np.mean(rhos):.3f} per-label {np.round(rhos, 3).tolist()}")

    print(f"{'w2':>5} {'gold74':>8} {'delta74':>8}")
    rows = []
    for w2 in GRID:
        b = (1.0 - w2) * b0 + w2 * A2
        a74, _ = macro_auc(b, gold)
        rows.append((w2, a74))
        print(f"{w2:5.2f} {a74:8.4f} {a74 - auc0:+8.4f}")
    best = max(rows, key=lambda r: r[1])
    print(f"best grid w2={best[0]:.2f} ({best[1]:.4f}) vs baseline {auc0:.4f} | "
          f"pick interior of plateau, conservative")
    return 0


if __name__ == "__main__":
    sys.exit(main())
