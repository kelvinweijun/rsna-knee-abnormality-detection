"""Fit W_ARM3 on top of the arm2blend (h957 + arm2 at w2=0.1) on gold74.

Same baseline replication as fit_w_arm2.py (stage-3 formula, A224=0.35,
reweight5 per-finding Raptor weights), then:
    b(w2, w3) = (1 - w2 - w3) * b0 + w2 * rankpct(arm2) + w3 * rankpct(arm3)
Grids w2 in {0.05, 0.10, 0.15} x w3 in {0.0, 0.05, 0.10, 0.15, 0.20}
(w2 + w3 <= 0.35). Reports spearman correlation of arm3 vs every arm
already in the blend (diversity check) and the grid; pick interior of
plateau, conservative. If arm_arm3cnv_58.csv is missing, prints baseline
and exits 0.

Inputs (local mirrors of kernel outputs):
  %TEMP%\\r7gold_out\\arm_r7_74.csv
  %TEMP%\\arm_g74\\arm_nartaa_r384_58.csv
  %TEMP%\\arm_g74\\arm_nartaa_r224_58.csv
  %TEMP%\\arm_g74\\arm_arm2hyb_58.csv
  %TEMP%\\arm_g74\\arm_arm3cnv_58.csv        (arm3-gold-eval kernel)
  %TEMP%\\jev_labels\\gold74_eval_set.csv
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
A224 = 0.35
W_RAPTOR = 0.5
W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
W2_FIXED = 0.10          # current submitted arm2blend setting
W2_GRID = [0.05, 0.10, 0.15]
W3_GRID = [0.0, 0.05, 0.10, 0.15, 0.20]


def load(path):
    d = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    return d.set_index("StudyInstanceUID")[LAB]


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


def rankpct_all(d, uid, name):
    arr = d.loc[uid, LAB].values.astype(np.float64)
    fail = (arr == 0.5).all(axis=1)
    if fail.any():
        raise RuntimeError(f"{name}: {int(fail.sum())} all-0.5 rows")
    return pd.DataFrame(np.stack([pct(arr[:, j]) for j in range(len(LAB))], axis=1),
                        index=uid, columns=LAB)


def main():
    gold = pd.read_csv(os.path.join(T, "jev_labels", "gold74_eval_set.csv"),
                       dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    uid = gold.index

    r7 = load(os.path.join(T, "r7gold_out", "arm_r7_74.csv"))
    r384 = load(os.path.join(T, "arm_g74", "arm_nartaa_r384_58.csv"))
    r224 = load(os.path.join(T, "arm_g74", "arm_nartaa_r224_58.csv"))
    a2p = os.path.join(T, "arm_g74", "arm_arm2hyb_58.csv")
    for nm, d in (("r7", r7), ("r384", r384), ("r224", r224), ("arm2", load(a2p))):
        assert sorted(d.index) == sorted(uid), f"{nm}: uid mismatch"

    R384 = rankpct_all(r384, uid, "r384")
    R224 = rankpct_all(r224, uid, "r224")
    RO = rankpct_all(r7, uid, "r7")
    A2 = rankpct_all(load(a2p), uid, "arm2")

    rr = (1.0 - A224) * R384 + A224 * R224
    b0 = pd.DataFrame({c: W_RAPTOR_C.get(c, W_RAPTOR) * rr[c]
                       + (1.0 - W_RAPTOR_C.get(c, W_RAPTOR)) * RO[c] for c in LAB},
                      index=uid)
    auc0, _ = macro_auc(b0, gold)
    b_arm2 = (1.0 - W2_FIXED) * b0 + W2_FIXED * A2
    auc_arm2, _ = macro_auc(b_arm2, gold)
    print(f"h957 baseline gold74: {auc0:.4f} | +arm2@{W2_FIXED}: {auc_arm2:.4f}")

    cands = sorted(glob.glob(os.path.join(T, "arm_g74", "arm_*_58.csv")))
    a3path = [p for p in cands if "arm3" in os.path.basename(p)]
    if not a3path:
        print("arm3 gold preds not yet present (arm_g74/arm_arm3cnv_58.csv)")
        return 0
    A3 = rankpct_all(load(a3path[0]), uid, "arm3")
    print("arm3 file:", a3path[0])
    auc3, per3 = macro_auc(A3, gold)
    print(f"arm3 standalone gold74 macro-AUC: {auc3:.4f}")
    print("arm3 per-label:", {c: round(a, 4) for c, a in zip(LAB, per3)})

    for nm, R in (("raptor384", R384), ("r7", RO), ("arm2", A2)):
        rhos = [pd.Series(A3[c].values).corr(pd.Series(R[c].values), method="spearman")
                for c in LAB]
        print(f"spearman arm3 vs {nm}: mean {np.mean(rhos):.3f} per-label {np.round(rhos, 3).tolist()}")

    print(f"{'w2':>5} {'w3':>5} {'gold74':>8} {'vs_b0':>8} {'vs_arm2b':>9}")
    rows = []
    for w2 in W2_GRID:
        for w3 in W3_GRID:
            if w2 + w3 > 0.35:
                continue
            b = (1.0 - w2 - w3) * b0 + w2 * A2 + w3 * A3
            a, _ = macro_auc(b, gold)
            rows.append((w2, w3, a))
            print(f"{w2:5.2f} {w3:5.2f} {a:8.4f} {a - auc0:+8.4f} {a - auc_arm2:+9.4f}")
    best = max(rows, key=lambda r: r[2])
    print(f"best: w2={best[0]:.2f} w3={best[1]:.2f} ({best[2]:.4f}) | "
          f"current submitted (w2=0.1,w3=0): {auc_arm2:.4f} | pick interior of plateau")
    return 0


if __name__ == "__main__":
    sys.exit(main())
