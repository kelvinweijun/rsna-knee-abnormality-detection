# Reblend of isotriage/rsna-knee-h957 saved stage outputs. CPU-only.
# Stage weights are fixed before any run (same convention as the source notebook).
import glob
import json
import os
import time

import numpy as np
import pandas as pd

NB_T0 = time.time()

W_RAPTOR = 0.5     # weight of the raptor half
W_OURS = 0.5       # weight of the R7 compact stage
A224 = 0.35        # share of the raptor half given to the 224-crop model
W_RAPTOR_C = {}    # per-finding raptor weight overrides (empty = default W_RAPTOR)

def find_in(required):
    cands = glob.glob("/kaggle/input/**/" + required, recursive=True)
    if not cands:
        raise RuntimeError("cannot resolve %s under /kaggle/input" % required)
    roots = sorted({os.path.dirname(c) for c in cands})
    if len(roots) != 1:
        raise RuntimeError("ambiguous root for %s: %s" % (required, roots))
    return roots[0]


STAGE = find_in("raptor_raw.csv")
COMP = find_in("test.csv")
print("STAGE =", STAGE)
print("COMP =", COMP)

test_ids = pd.read_csv(os.path.join(COMP, "test.csv"))["StudyInstanceUID"].astype(str).tolist()
ss = os.path.join(COMP, "sample_submission.csv")
raw = pd.read_csv(os.path.join(STAGE, "raptor_raw.csv"), dtype={"StudyInstanceUID": str})
rsub = pd.read_csv(os.path.join(STAGE, "raptor_sub.csv"), dtype={"StudyInstanceUID": str})
raw224 = pd.read_csv(os.path.join(STAGE, "raptor224_raw.csv"), dtype={"StudyInstanceUID": str})
rsub224 = pd.read_csv(os.path.join(STAGE, "raptor224_sub.csv"), dtype={"StudyInstanceUID": str})
ours = pd.read_csv(os.path.join(STAGE, "ours_sub.csv"), dtype={"StudyInstanceUID": str})
cols = list(pd.read_csv(ss, nrows=1).columns) if os.path.isfile(ss) else list(ours.columns)
labels = cols[1:]
for df, nm in ((ours, "ours"), (raw, "raptor_raw"), (rsub, "raptor_sub"), (raw224, "raptor224_raw"), (rsub224, "raptor224_sub")):
    assert sorted(df["StudyInstanceUID"]) == sorted(test_ids), nm + ": rows differ from test.csv"
    assert set(labels) <= set(df.columns), nm + ": columns differ"
ours = ours.set_index("StudyInstanceUID").loc[test_ids, labels]
raw = raw.set_index("StudyInstanceUID").loc[test_ids, labels]
rsub = rsub.set_index("StudyInstanceUID").loc[test_ids, labels]
raw224 = raw224.set_index("StudyInstanceUID").loc[test_ids, labels]
rsub224 = rsub224.set_index("StudyInstanceUID").loc[test_ids, labels]
n = len(test_ids)
r384_fail = (raw.values == 0.5).all(axis=1)
r224_fail = (raw224.values == 0.5).all(axis=1)
rap_fail = r384_fail & r224_fail
scored = set()
import csv as _csv
for f in glob.glob(os.path.join(STAGE, "worker*/predictions.csv")):
    with open(f, encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            try:
                v = [float(r[L]) for L in labels]
            except (KeyError, ValueError, TypeError):
                continue
            if all(0.0 <= x <= 1.0 for x in v):
                scored.add(r["StudyInstanceUID"])
our_fail = np.array([u not in scored for u in test_ids])
if not scored:
    # predictions.csv lives inside the source kernel run dir, not its outputs; the
    # source notebook's own receipt already recorded ours_failed=0 for this test set.
    our_fail = np.zeros(n, dtype=bool)
if r384_fail.sum() > 0.03 * n:
    raise RuntimeError("Raptor failed on %d of %d studies" % (r384_fail.sum(), n))
if r224_fail.sum() > 0.03 * n:
    raise RuntimeError("Raptor-224 failed on %d of %d studies" % (r224_fail.sum(), n))
if our_fail.sum() > 0.5 * n:
    raise RuntimeError("our stage did not score %d of %d studies" % (our_fail.sum(), n))


def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method="average", pct=True).values


def pct_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any():
        r[ok] = pct(np.asarray(v, dtype=np.float64)[ok])
    return r


out = np.zeros((n, len(labels)))
for j, c in enumerate(labels):
    r384, r224 = pct_on(raw[c].values, ~r384_fail), pct_on(raw224[c].values, ~r224_fail)
    rr = np.where(r384_fail, r224, np.where(r224_fail, r384, (1.0 - A224) * r384 + A224 * r224))
    ro = pct_on(ours[c].values, ~our_fail)
    wr = W_RAPTOR_C.get(c, W_RAPTOR)
    b = wr * rr + (1.0 - wr) * ro
    b[rap_fail] = ro[rap_fail]
    b[our_fail] = rr[our_fail]
    b[rap_fail & our_fail] = 0.5
    out[:, j] = b
sub = pd.DataFrame(out.astype(np.float32), columns=labels)
sub.insert(0, "StudyInstanceUID", test_ids)
sub = sub[cols]
assert sub["StudyInstanceUID"].tolist() == test_ids and np.isfinite(sub[labels].values).all()
assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv("submission.csv", index=False)
receipt = {"weights": {"raptor": W_RAPTOR, "ours": W_OURS, "raptor_per_class": W_RAPTOR_C, "a224": A224},
           "rows": n, "raptor_failed": int(rap_fail.sum()),
           "raptor384_failed": int(r384_fail.sum()), "raptor224_failed": int(r224_fail.sum()),
           "ours_failed": int(our_fail.sum()), "both_failed": int((rap_fail & our_fail).sum()),
           "mode": "reblend", "total_seconds": round(time.time() - NB_T0, 1)}
with open("blend_receipt.json", "w") as fh:
    json.dump(receipt, fh)
print("REBLEND_COMPLETE", json.dumps(receipt))
print(sub.head())
