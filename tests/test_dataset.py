import numpy as np
import pandas as pd
import pytest

from knee_meta.config import TARGETS
from knee_meta.dataset import StudyDataset


def _table(studies, encode):
    rows = []
    for s in studies:
        row = {"StudyInstanceUID": s, "is_gold": 0}
        for lab in TARGETS:
            row[lab] = 0.0
            row[lab + "__conf"] = 1.0
        row["ACL"] = encode(s)
        rows.append(row)
    return pd.DataFrame(rows)


def test_study_ids_alignment():
    n = 8
    cache = np.zeros((n, 6, 3, 16, 16), dtype=np.uint8)
    for i in range(n):
        cache[i] = i
    mask = np.ones((n, 6), dtype=np.uint8)
    studies = [f"s{i}" for i in range(n)]
    encode = lambda s: int(s[1:]) / 10.0  # noqa: E731
    table = _table(studies, encode)

    perm = [5, 2, 7, 0, 3, 1, 6, 4]
    fold = [studies[p] for p in perm]
    ds = StudyDataset(cache, mask, studies, table, train=False, study_ids=fold)

    assert ds.uids == fold
    for i, p in enumerate(perm):
        item = ds[i]
        assert ds.uids[i] == studies[p]
        assert float(item["target"][0]) == pytest.approx(p / 10.0)
        assert float(item["imgs"].max()) == pytest.approx(p / 255.0)
        assert float(item["mask"].max()) == 1.0


def test_no_study_ids_keeps_cache_order():
    n = 6
    cache = np.zeros((n, 6, 3, 8, 8), dtype=np.uint8)
    for i in range(n):
        cache[i] = i
    mask = np.ones((n, 6), dtype=np.uint8)
    studies = [f"s{i}" for i in range(n)]
    table = _table(studies, lambda s: int(s[1:]) / 10.0)
    table = table.iloc[::-1].reset_index(drop=True)

    ds = StudyDataset(cache, mask, studies, table, train=False)
    for i in range(n):
        item = ds[i]
        assert ds.uids[i] == studies[i]
        assert float(item["target"][0]) == pytest.approx(i / 10.0)
        assert float(item["imgs"].max()) == pytest.approx(i / 255.0)
