"""gold58 stacking harness: score arms, measure complementarity, fit shrunk per-label weights.

Usage:
  python tools/gold58_stacker.py --arm-dir <dir with arm_*_58.csv> [--out report.md]

Every arm CSV must have StudyInstanceUID + the 12 target columns (probabilities or ranks;
values are percentile-ranked per label before stacking, so raw probs and rank-pct inputs
are interchangeable). Gold labels come from train.csv (58 labelled studies).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
GRID = np.round(np.arange(0.0, 1.001, 0.1), 3)
SHRINK = 0.5          # pull per-label weights halfway to uniform (58-sample protection)
BOOT = 2000


def rank_pct(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in TARGETS:
        x = df[c].to_numpy(dtype=np.float64)
        ok = np.isfinite(x)
        r = np.full(len(x), 0.5)
        r[ok] = rankdata(x[ok]) / (ok.sum() + 1e-9)
        out[c] = np.clip(r, 0.001, 0.999)
    return out


def load_arms(arm_dir: str) -> dict[str, pd.DataFrame]:
    arms = {}
    paths = sorted(glob.glob(os.path.join(arm_dir, "**", "arm_*_58.csv"), recursive=True))
    # prefer *_raw over the rank-pct sibling of the same arm (monotone duplicates)
    raws = [p for p in paths if "_raw_" in os.path.basename(p)]
    if raws:
        keep = []
        for p in paths:
            if "_rank_" in os.path.basename(p):
                stem = os.path.basename(p).replace("_rank_", "_raw_")
                if any(os.path.basename(r) == stem for r in raws):
                    continue
            keep.append(p)
        paths = keep
    for p in paths:
        name = os.path.basename(p).replace("arm_", "").split("_58")[0]
        df = pd.read_csv(p)
        df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
        df = df.set_index("StudyInstanceUID")
        missing = [t for t in TARGETS if t not in df.columns]
        if missing:
            raise SystemExit(f"{p}: missing columns {missing}")
        arms[name] = rank_pct(df[TARGETS])
        print(f"loaded {name}: {df.shape}")
    if not arms:
        raise SystemExit(f"no arm_*_58.csv under {arm_dir}")
    return arms


def per_label_auc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array([roc_auc_score(y[:, j], p[:, j]) for j in range(len(TARGETS))])


def simplex_grid(m: int, steps: int = 10) -> np.ndarray:
    """All non-negative weight vectors of length m summing to 1 in 1/steps increments."""
    out = []

    def rec(prefix, left, parts):
        if parts == 1:
            out.append(tuple(v / steps for v in prefix + [left]))
            return
        for i in range(left + 1):
            rec(prefix + [i], left - i, parts - 1)

    rec([], steps, m)
    return np.array(out)


def stack_auc(y: np.ndarray, weights: dict[str, float],
              arms: dict[str, pd.DataFrame], idx: np.ndarray) -> tuple[np.ndarray, float]:
    total = np.zeros((len(idx), len(TARGETS)))
    for name, w in weights.items():
        if w <= 0:
            continue
        total += w * arms[name].to_numpy()[idx]
    auc = per_label_auc(y[idx], total)
    return auc, float(auc.mean())


def fuse(arms: dict[str, pd.DataFrame], names: list[str],
         w_mat: np.ndarray) -> np.ndarray:
    """w_mat: (labels, arms) weights -> (studies, labels) fused scores."""
    out = np.zeros((len(arms[names[0]]), len(TARGETS)))
    for k, name in enumerate(names):
        out += arms[name].to_numpy() * w_mat[:, k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--train", default=r"C:\Users\kelvi\Downloads\RSNA Detection\train.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = load_arms(args.arm_dir)
    names = list(arms)
    common = set(arms[names[0]].index)
    for n in names[1:]:
        common &= set(arms[n].index)
    gold = pd.read_csv(args.train, dtype={"StudyInstanceUID": str})
    gold = gold[gold["ACL"].notna()].set_index("StudyInstanceUID")
    uids = [u for u in gold.index if u in common]
    if len(uids) < 50:
        raise SystemExit(f"only {len(uids)} gold studies with all-arm predictions")
    for n in names:
        arms[n] = arms[n].loc[uids]
    y = gold.loc[uids, TARGETS].to_numpy(dtype=np.float64)
    n = len(uids)
    lines = []
    say = lambda s: (print(s), lines.append(s))

    say(f"gold studies with all arms: {n}")
    say("")
    say("== per-arm gold58 AUC ==")
    solo = {}
    for name in names:
        auc = per_label_auc(y, arms[name].to_numpy())
        solo[name] = auc
        say(f"{name:>10}: macro {auc.mean():.4f} | "
            + " ".join(f"{t.split()[0][:4]}={a:.3f}" for t, a in zip(TARGETS, auc)))

    say("")
    say("== per-label Spearman between arms (rank space) ==")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = [np.corrcoef(rankdata(arms[a].to_numpy()[:, j]),
                               rankdata(arms[b].to_numpy()[:, j]))[0, 1]
                   for j in range(len(TARGETS))]
            say(f"{a} vs {b}: mean {np.mean(rho):.3f} | "
                + " ".join(f"{t.split()[0][:4]}={r:.2f}" for t, r in zip(TARGETS, rho)))

    say("")
    say("== uniform rank-mean stacks ==")
    uniform = {name: 1.0 / len(names) for name in names}
    u_auc, u_macro = stack_auc(y, uniform, arms, np.arange(n))
    say(f"uniform ({'+'.join(names)}): macro {u_macro:.4f}")

    # pairwise uniforms for reference
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa_, pm = stack_auc(y, {a: 0.5, b: 0.5}, arms, np.arange(n))
            say(f"uniform {a}+{b}: macro {pm:.4f}")

    if len(names) < 2:
        say("")
        say("(need >=2 arms for per-label weights)")
        report = "\n".join(str(x) for x in lines)
        if args.out:
            open(args.out, "w", encoding="utf-8").write(report)
        return

    # ---- per-label weights over the simplex for ALL arms ----
    say("")
    say(f"== per-label weights: simplex grid on {names} (shrink {SHRINK} toward uniform) ==")
    p_arr = {name: arms[name].to_numpy() for name in names}
    grid = simplex_grid(len(names), steps=10)
    best_w = np.zeros((len(TARGETS), len(names)))
    best_auc = np.zeros(len(TARGETS))
    for j in range(len(TARGETS)):
        scores = np.empty(len(grid))
        for gi, wv in enumerate(grid):
            p = np.zeros(n)
            for k_, name in enumerate(names):
                p += wv[k_] * p_arr[name][:, j]
            scores[gi] = roc_auc_score(y[:, j], p)
        jb = int(np.argmax(scores))
        best_w[j] = grid[jb]
        best_auc[j] = scores[jb]
    say("label          " + "".join(f"{nm[:6]:>8}" for nm in names) + "  auc_grid  auc_unif   delta")
    for j, t in enumerate(TARGETS):
        delta = best_auc[j] - u_auc[j]
        say(f"{t:<14}" + "".join(f"{best_w[j, k_]:8.2f}" for k_ in range(len(names)))
            + f"  {best_auc[j]:7.4f}  {u_auc[j]:7.4f}  {delta:+.4f}")
    uniform_vec = np.full(len(names), 1.0 / len(names))
    shrunk = SHRINK * best_w + (1 - SHRINK) * uniform_vec
    s_stack = np.zeros((n, len(TARGETS)))
    for k_, name in enumerate(names):
        s_stack += p_arr[name] * shrunk[:, k_]
    s_auc = per_label_auc(y, s_stack)
    say(f"shrunk weights macro: {s_auc.mean():.4f} (uniform {u_auc.mean():.4f}, "
        f"grid-in-sample {best_auc.mean():.4f})")

    say("")
    say(f"== bootstrap ({BOOT} resamples of the 58): shrunk-stack macro - uniform macro ==")
    rng = np.random.default_rng(0)
    deltas = []
    f_shr = fuse(arms, names, shrunk)
    f_uni = fuse(arms, names, np.tile(uniform_vec, (len(TARGETS), 1)))
    first_err = None
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        try:
            m_shr = float(per_label_auc(y[idx], f_shr[idx]).mean())
            m_uni = float(per_label_auc(y[idx], f_uni[idx]).mean())
        except ValueError as e:
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"
            continue
        deltas.append(m_shr - m_uni)
    if first_err and not deltas:
        raise SystemExit(f"bootstrap failed on every draw; first error -> {first_err}")
    deltas = np.array(deltas)
    say(f"mean {deltas.mean():+.4f} | p10 {np.percentile(deltas, 10):+.4f} | "
        f"p90 {np.percentile(deltas, 90):+.4f} | P(>0) {(deltas > 0).mean():.2f}")

    say("")
    say("== recommendation ==")
    if s_auc.mean() - u_auc.mean() > 0.004 and (deltas > 0).mean() > 0.7:
        say("per-label shrunk weights look real on this sample — submit the shrunk stack")
    else:
        say("per-label weights not stable on 58 samples — use uniform stack or shrink harder")
    say(f"shrunk per-label weights (columns {names}):")
    say(np.round(shrunk, 2).tolist())

    report = "\n".join(str(x) for x in lines)
    if args.out:
        open(args.out, "w", encoding="utf-8", newline="\n").write(report + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
