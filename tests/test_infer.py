from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from knee_meta import infer
from knee_meta.config import TARGETS
from knee_meta.dataset import StudyDataset


def _touch(directory, names):
    Path(directory).mkdir(parents=True, exist_ok=True)
    for n in names:
        (Path(directory) / n).write_bytes(b"x")


def test_resolve_prefers_env(monkeypatch, tmp_path):
    _touch(tmp_path / "envdir", ["fold0.pt"])
    _touch(tmp_path / "default", ["fold0.pt"])
    monkeypatch.setenv("RSNA_MODELS_DIR", str(tmp_path / "envdir"))
    assert infer.resolve_models_dir(str(tmp_path / "default")) == str(
        tmp_path / "envdir")


def test_resolve_uses_preferred_then_candidates(tmp_path, monkeypatch):
    monkeypatch.delenv("RSNA_MODELS_DIR", raising=False)
    _touch(tmp_path / "later", ["fold1.pt", "fold0.pt"])
    got = infer.resolve_models_dir(str(tmp_path / "missing"),
                                   candidates=[str(tmp_path / "later")])
    assert got == str(tmp_path / "later")


def test_resolve_raises_when_nothing_found(tmp_path, monkeypatch):
    monkeypatch.delenv("RSNA_MODELS_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="no fold"):
        infer.resolve_models_dir(str(tmp_path / "a"),
                                 candidates=[str(tmp_path / "b")])
    assert not infer.find_checkpoints(tmp_path / "a")


def test_resolve_skips_dirs_without_checkpoints(tmp_path, monkeypatch):
    monkeypatch.delenv("RSNA_MODELS_DIR", raising=False)
    _touch(tmp_path / "empty", ["other.pt"])
    _touch(tmp_path / "good", ["fold0.pt"])
    got = infer.resolve_models_dir(str(tmp_path / "empty"),
                                   candidates=[str(tmp_path / "good")])
    assert got == str(tmp_path / "good")
    with pytest.raises(FileNotFoundError):
        infer.resolve_models_dir(str(tmp_path / "empty"),
                                 candidates=[str(tmp_path / "nope")])


def _template(uids):
    sub = pd.DataFrame({"StudyInstanceUID": uids})
    for lab in TARGETS:
        sub[lab] = 0.5
    return sub


def test_load_template_prefers_test_csv(tmp_path):
    _template(["a", "b"]).to_csv(tmp_path / "sample_submission.csv",
                                 index=False)
    _template(["c"]).to_csv(tmp_path / "submission.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["x", "y"]}).to_csv(
        tmp_path / "test.csv", index=False)
    got = infer.load_template(tmp_path)
    assert got.StudyInstanceUID.tolist() == ["x", "y"]


def test_load_template_sample_without_test_csv(tmp_path):
    _template(["a", "b"]).to_csv(tmp_path / "sample_submission.csv",
                                 index=False)
    _template(["c"]).to_csv(tmp_path / "submission.csv", index=False)
    got = infer.load_template(tmp_path)
    assert got.StudyInstanceUID.tolist() == ["a", "b"]


def test_load_template_falls_back_to_submission_then_test_csv(tmp_path):
    _template(["c"]).to_csv(tmp_path / "submission.csv", index=False)
    assert infer.load_template(tmp_path).StudyInstanceUID.tolist() == ["c"]
    (tmp_path / "submission.csv").unlink()
    pd.DataFrame({"StudyInstanceUID": ["d", "e"]}).to_csv(
        tmp_path / "test.csv", index=False)
    got = infer.load_template(tmp_path)
    assert got.StudyInstanceUID.tolist() == ["d", "e"]
    assert (got[TARGETS] == 0.5).all().all()
    with pytest.raises(FileNotFoundError):
        infer.load_template(tmp_path / "nowhere")


def test_fill_submission_keeps_05_for_studies_without_cache():
    sub = _template(["a", "b", "c"])
    scores = np.full((2, len(TARGETS)), 0.5)
    scores[0, 0] = 0.1
    scores[1, 0] = 0.9
    got = infer.fill_submission(sub, scores, ["b", "c"])
    assert got.loc[got.StudyInstanceUID == "a", "ACL"].iloc[0] == 0.5
    assert got.loc[got.StudyInstanceUID == "b", "ACL"].iloc[0] == 0.1
    assert got.loc[got.StudyInstanceUID == "c", "ACL"].iloc[0] == 0.9
    with pytest.raises(ValueError, match="does not match"):
        infer.fill_submission(sub, scores[:, :2], ["b", "c"])


def test_check_coverage_returns_missing_for_small_templates():
    sub = _template(["a", "b", "c"])
    assert infer.check_coverage(sub, ["b", "c"]) == 1
    assert infer.check_coverage(sub, ["a", "b", "c"]) == 0


def test_check_coverage_raises_on_bulk_missing():
    uids = [f"s{i}" for i in range(50)]
    sub = _template(uids)
    with pytest.raises(ValueError, match="refusing to submit"):
        infer.check_coverage(sub, uids[:10])


def test_fill_submission_raises_when_nothing_predicted():
    uids = [f"s{i}" for i in range(50)]
    sub = _template(uids)
    scores = np.full((50, len(TARGETS)), 0.5)
    with pytest.raises(ValueError, match="constant 0.5"):
        infer.fill_submission(sub, scores, uids)


def test_fill_submission_raises_on_bulk_missing():
    uids = [f"s{i}" for i in range(50)]
    sub = _template(uids)
    scores = np.linspace(0, 1, 10 * len(TARGETS)).reshape(10, len(TARGETS))
    with pytest.raises(ValueError, match="refusing to submit"):
        infer.fill_submission(sub, scores, uids[:10])


def test_check_slots_passes_when_decoded():
    mask = np.ones((20, 6), dtype=np.uint8)
    fill, empty = infer.check_slots(mask, [f"s{i}" for i in range(20)])
    assert fill == 1.0
    assert empty == 0


def test_check_slots_raises_when_most_studies_empty():
    mask = np.zeros((50, 6), dtype=np.uint8)
    mask[:5] = 1
    with pytest.raises(ValueError, match="slot decode failed"):
        infer.check_slots(mask, [f"s{i}" for i in range(50)])


def test_check_slots_ignores_small_example_runs():
    mask = np.zeros((3, 6), dtype=np.uint8)
    fill, empty = infer.check_slots(mask, ["a", "b", "c"])
    assert empty == 3
    assert fill == 0.0


class _Recorder:
    def __init__(self):
        self.seen = []

    def __call__(self, imgs, mask, img_size=None):
        self.seen.append(imgs.detach().cpu().clone())
        return torch.zeros(imgs.shape[0], len(TARGETS))


def _table_for(studies):
    rows = []
    for s in studies:
        row = {"StudyInstanceUID": s, "is_gold": 0}
        for lab in TARGETS:
            row[lab] = 0.0
            row[lab + "__conf"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_predict_feeds_model_same_range_as_dataset():
    rng = np.random.RandomState(0)
    n, slots = 4, 6
    cache = rng.randint(0, 256, (n, slots, 3, 8, 8), dtype=np.uint8)
    mask = np.ones((n, slots), dtype=np.uint8)
    studies = [f"s{i}" for i in range(n)]

    recorder = _Recorder()
    infer.predict([("rec", recorder, None)], cache, mask, torch.device("cpu"),
                  batch=n, groups=[0])
    assert len(recorder.seen) == 1
    pred_in = recorder.seen[0]

    ds = StudyDataset(cache, mask, studies, _table_for(studies), train=False,
                      groups=0)
    ds_in = torch.stack([ds[i]["imgs"] for i in range(n)])

    assert pred_in.dtype == torch.float32
    assert float(pred_in.max()) <= 1.0 + 1e-6
    torch.testing.assert_close(pred_in, ds_in)
