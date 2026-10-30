from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from knee_meta import infer, labels, train
from knee_meta.config import N_SLOT, PREPROC, TARGETS
from knee_meta.imaging import save_cache

REPO = Path(__file__).resolve().parents[1]
N_STUDIES = 16
IMG = 64


def test_array_stats_matches_numpy_small():
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, (16, 6, 3, 64, 64), dtype=np.uint8)
    mean, std = train.array_stats(arr)
    assert mean == pytest.approx(float(arr.mean()), abs=1e-6)
    assert std == pytest.approx(float(arr.std()), abs=1e-6)


def test_array_stats_is_sampled_on_large_input():
    arr = np.arange(60_000_000, dtype=np.uint8).reshape(1000, 1000, 60)
    mean, std = train.array_stats(arr, target=1_000_000)
    assert abs(mean - 127.5) < 2.0
    assert 70.0 < std < 75.0


def test_average_states_means_floats_keeps_last_buffer():
    a = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor([5])}
    b = {"w": torch.tensor([5.0, 7.0]), "n": torch.tensor([9])}
    out = train.average_states([a, b])
    assert torch.equal(out["w"], torch.tensor([3.0, 5.0]))
    assert torch.equal(out["n"], torch.tensor([9]))
    assert out["w"].dtype == a["w"].dtype


def _write_corpus(root, rng):
    (root / "test_series").mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(REPO / "train.csv").head(N_STUDIES)
    train.to_csv(root / "train.csv", index=False)
    sub = pd.read_csv(REPO / "sample_submission.csv")
    sub.to_csv(root / "submission.csv", index=False)
    mask = np.zeros((N_STUDIES, 6), dtype=np.uint8)
    mask[:, :4] = 1
    cache = rng.randint(0, 255, (N_STUDIES, 6, 3, IMG, IMG), dtype=np.uint8)
    save_cache(str(root / "cache_train.npy"), cache, mask,
               train.StudyInstanceUID.to_numpy())
    return sub


def test_train_then_infer(tmp_path):
    rng = np.random.RandomState(0)
    root = tmp_path / "comp"
    sub = _write_corpus(root, rng)

    results = train.main([
        "--root", str(root),
        "--cache", str(root / "cache_train.npy"),
        "--labels", str(tmp_path / "report_labels.csv"),
        "--out", str(tmp_path / "models"),
        "--folds", "2", "--epochs", "1", "--tiny",
        "--img-size", str(IMG), "--batch-size", "4", "--eval-batch", "4",
        "--budget", "600", "--seed", "7",
    ])
    assert len(results) == 2
    assert all(np.isfinite(r["val_auc"]) for r in results)
    assert (tmp_path / "models" / "fold0.pt").is_file()
    assert (tmp_path / "models" / "fold1.pt").is_file()
    blob = torch.load(tmp_path / "models" / "fold0.pt", map_location="cpu",
                      weights_only=False)
    assert blob["preproc"] == PREPROC

    uids = sub.StudyInstanceUID.tolist()
    test_cache = rng.randint(0, 255, (len(uids), 6, 3, IMG, IMG), dtype=np.uint8)
    save_cache(str(root / "cache_test.npy"), test_cache,
               np.ones((len(uids), 6), dtype=np.uint8),
               np.array(uids, dtype=object))

    out = infer.main([
        "--root", str(root),
        "--cache", str(root / "cache_test.npy"),
        "--models", str(tmp_path / "models"),
        "--out", str(tmp_path / "submission.csv"),
        "--img-size", str(IMG), "--tiny", "--batch", "4",
    ])
    assert list(out.columns) == ["StudyInstanceUID"] + TARGETS
    assert len(out) == len(uids)
    values = out[TARGETS].to_numpy()
    assert np.isfinite(values).all()
    assert (values > 0).all() and (values <= 1).all()
    submitted = pd.read_csv(tmp_path / "submission.csv")
    assert len(submitted) == len(uids)


def test_only_fold_and_swa(tmp_path):
    rng = np.random.RandomState(0)
    root = tmp_path / "comp"
    _write_corpus(root, rng)

    results = train.main([
        "--root", str(root),
        "--cache", str(root / "cache_train.npy"),
        "--labels", str(tmp_path / "report_labels.csv"),
        "--out", str(tmp_path / "models"),
        "--folds", "3", "--epochs", "2", "--tiny",
        "--img-size", str(IMG), "--batch-size", "4", "--eval-batch", "4",
        "--budget", "600", "--seed", "7",
        "--only-fold", "1", "--swa-from", "1",
    ])
    assert [r["fold"] for r in results] == [1]
    assert not (tmp_path / "models" / "fold0.pt").exists()
    assert not (tmp_path / "models" / "fold2.pt").exists()
    blob = torch.load(tmp_path / "models" / "fold1.pt", map_location="cpu",
                      weights_only=False)
    assert blob["swa_from"] == 1
    assert blob["swa_members"] == 1
    assert 0.0 <= blob["swa_val_auc"] <= 1.0
    assert blob["val_auc"] == pytest.approx(max(
        blob["swa_val_auc"], max(h["val_auc"] for h in blob["history"])))


def test_only_fold_list_trains_selected_folds(tmp_path):
    assert train.parse_fold_spec(None) is None
    assert train.parse_fold_spec("") is None
    assert train.parse_fold_spec("0") == {0}
    assert train.parse_fold_spec("0,1,2") == {0, 1, 2}
    assert train.parse_fold_spec(" 1 , 3 ") == {1, 3}

    rng = np.random.RandomState(0)
    root = tmp_path / "comp"
    _write_corpus(root, rng)
    results = train.main([
        "--root", str(root),
        "--cache", str(root / "cache_train.npy"),
        "--labels", str(tmp_path / "report_labels.csv"),
        "--out", str(tmp_path / "models"),
        "--folds", "3", "--epochs", "1", "--tiny",
        "--img-size", str(IMG), "--batch-size", "4", "--eval-batch", "4",
        "--budget", "600", "--seed", "7",
        "--only-fold", "1,2",
    ])
    assert [r["fold"] for r in results] == [1, 2]
    assert not (tmp_path / "models" / "fold0.pt").exists()
    assert (tmp_path / "models" / "fold1.pt").is_file()
    assert (tmp_path / "models" / "fold2.pt").is_file()


def test_determinism_check_passes_on_cpu():
    model = train.build_model(tiny=True, img_size=IMG)
    batch = {
        "imgs": torch.rand(2, N_SLOT, 3, IMG, IMG),
        "mask": torch.ones(2, N_SLOT),
        "target": torch.rand(2, len(TARGETS)),
        "conf": torch.rand(2, len(TARGETS)),
        "gold": torch.zeros(2),
    }
    ok, why, strict_unusable = train.determinism_check(
        model, batch, torch.device("cpu"), IMG)
    assert ok, why
    assert not strict_unusable


def test_deterministic_mode_trains_single_fold(tmp_path):
    rng = np.random.RandomState(0)
    root = tmp_path / "comp"
    _write_corpus(root, rng)
    try:
        results = train.main([
            "--root", str(root),
            "--cache", str(root / "cache_train.npy"),
            "--labels", str(tmp_path / "report_labels.csv"),
            "--out", str(tmp_path / "models"),
            "--folds", "2", "--epochs", "1", "--tiny",
            "--img-size", str(IMG), "--batch-size", "4", "--eval-batch", "4",
            "--budget", "600", "--seed", "7",
            "--only-fold", "0", "--deterministic",
        ])
    finally:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
    assert len(results) == 1
    assert np.isfinite(results[0]["val_auc"])
    assert (tmp_path / "models" / "fold0.pt").is_file()


def test_consensus_training_runs(tmp_path):
    rng = np.random.RandomState(0)
    root = tmp_path / "comp"
    _write_corpus(root, rng)

    table = labels.build_label_table(str(root / "train.csv"))
    partner = table[["StudyInstanceUID"] + TARGETS].copy()
    nongold = table.index[table.is_gold == 0]
    i = nongold[0] if len(nongold) else table.index[0]
    partner.loc[i, "ACL"] = 1.0 - float(partner.loc[i, "ACL"])
    partner_path = tmp_path / "partner.csv"
    partner.to_csv(partner_path, index=False)

    results = train.main([
        "--root", str(root),
        "--cache", str(root / "cache_train.npy"),
        "--labels", str(tmp_path / "report_labels.csv"),
        "--out", str(tmp_path / "models"),
        "--folds", "2", "--epochs", "1", "--tiny",
        "--img-size", str(IMG), "--batch-size", "4", "--eval-batch", "4",
        "--budget", "600", "--seed", "7",
        "--consensus", str(partner_path),
    ])
    assert len(results) == 2
    assert all(np.isfinite(r["val_auc"]) for r in results)
