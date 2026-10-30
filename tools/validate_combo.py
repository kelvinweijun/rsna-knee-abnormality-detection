"""Validate the winning multi-arm combo on gold74: neighborhood stability +
bootstrap P(gain>0) + arm2 interaction 3-way check.

b = (1 - wa - wc - we) * b0 + wa*arm2 + wc*coatnetx + we*effv2l   (rank-pct space)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T = os.environ.get("TEMP", "/tmp")
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
A224 = 0.35
W_RAPTOR = 0.5
W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}
BOOT = 2000


def load(path):
    d = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    return d.set_index("StudyInstanceUID")[LAB]


def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method="average", pct=True).values


def rankpct(d, uid):
    arr = d.loc[uid, LAB].values.astype(np.float64)
    return pd.DataFrame(np.stack([pct(arr[:, j]) for j in range(len(LAB))], axis=1),
                        index=uid, columns=LAB)


def per_label_auc_df(scores, labels_df):
    out = []
    s = scores.loc[labels_df.index]
    for c in LAB:
        y_raw = labels_df[c].values
        mask = ~pd.isna(y_raw)
        y = y_raw[mask].astype(int)
        if len(np.unique(y)) < 2:
            continue
        out.append(roc_auc_score(y, s[c].values[mask]))
    return np.array(out)


def main():
    gold = pd.read_csv(os.path.join(T, "jev_labels", "gold74_eval_set.csv"),
                       dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    uid = gold.index

    r7 = load(os.path.join(T, "r7gold_out", "arm_r7_74.csv"))
    r384 = load(os.path.join(T, "arm_g74", "arm_nartaa_r384_58.csv"))
    r224 = load(os.path.join(T, "arm_g74", "arm_nartaa_r224_58.csv"))
    R384, R224, RO = (rankpct(d, uid) for d in (r384, r224, r7))
    rr = (1.0 - A224) * R384 + A224 * R224
    b0 = pd.DataFrame({c: W_RAPTOR_C.get(c, W_RAPTOR) * rr[c]
                       + (1.0 - W_RAPTOR_C.get(c, W_RAPTOR)) * RO[c] for c in LAB}, index=uid)

    A2 = rankpct(load(os.path.join(T, "arm_g74", "arm_arm2hyb_58.csv")), uid)
    CX = rankpct(load(os.path.join(T, "arm_g74", "arm_dread_coatnetx_58.csv")), uid)
    EF = rankpct(load(os.path.join(T, "arm_g74", "arm_dread_effv2l_58.csv")), uid)

    def blend(wa, wc, we):
        return (1.0 - wa - wc - we) * b0 + wa * A2 + wc * CX + we * EF

    auc0 = per_label_auc_df(b0, gold).mean()
    print(f"b0 {auc0:.4f}")

    print("\n== neighborhood around (arm2, coatnetx, effv2l) ==")
    best = (auc0, 0, 0, 0)
    for wa in (0.0, 0.05, 0.10):
        for wc in (0.10, 0.15, 0.20):
            for we in (0.0, 0.05, 0.10):
                a = per_label_auc_df(blend(wa, wc, we), gold).mean()
                mark = " *" if a > best[0] else ""
                print(f"  arm2={wa:.2f} cx={wc:.2f} eff={we:.2f} -> {a:.4f} ({a - auc0:+.4f}){mark}")
                if a > best[0]:
                    best = (a, wa, wc, we)
    print(f"best: arm2={best[1]:.2f} cx={best[2]:.2f} eff={best[3]:.2f} auc {best[0]:.4f} ({best[0]-auc0:+.4f})")

    # bootstrap the best vs b0
    rng = np.random.default_rng(0)
    fb = blend(best[1], best[2], best[3]).to_numpy()
    f0 = b0.to_numpy()
    y = gold.to_numpy(dtype=np.float64)
    n = len(uid)
    deltas = []
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        try:
            a1 = per_label_auc_df(pd.DataFrame(fb[idx], columns=LAB), pd.DataFrame(y[idx], columns=LAB)).mean()
            a0 = per_label_auc_df(pd.DataFrame(f0[idx], columns=LAB), pd.DataFrame(y[idx], columns=LAB)).mean()
        except ValueError:
            continue
        deltas.append(a1 - a0)
    deltas = np.array(deltas)
    print(f"\n== bootstrap({len(deltas)}) best-vs-b0: mean {deltas.mean():+.4f} "
          f"p10 {np.percentile(deltas,10):+.4f} p90 {np.percentile(deltas,90):+.4f} "
          f"P(>0) {(deltas>0).mean():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
