import json
import re
from pathlib import Path

from tools import build_notebooks as bn

REPO = Path(__file__).resolve().parents[1]


def test_writefile_cells_match_sources():
    for cell in bn.writefile_cells():
        first, *body = cell["source"]
        name = re.match(r"%%writefile (\S+)\n", first).group(1)
        src = (REPO / name).read_text(encoding="utf-8")
        assert "".join(body) == src, name


def test_run_train_src_no_extra_is_unchanged():
    assert bn.run_train_src("") == bn.RUN_TRAIN


def test_run_train_src_appends_extra_args():
    src = bn.run_train_src("--only-fold 1 --swa-from 5")
    assert '"--only-fold",\n    "1",\n    "--swa-from",\n    "5",' in src
    assert src.startswith(bn.RUN_TRAIN.split("])")[0])
    assert src.rstrip().endswith("results")


def test_run_train_src_reads_env(monkeypatch):
    monkeypatch.setenv("RN_TRAIN_EXTRA", "--only-fold 0")
    assert '"--only-fold",\n    "0",' in bn.run_train_src()


def test_write_kernel_extra_datasets_and_id(tmp_path, monkeypatch):
    monkeypatch.setattr(bn, "REPO", tmp_path)
    monkeypatch.setenv("RN_TRAIN_DATASETS",
                       "stevenleehans/rsna-knee-llm-report-labels")
    monkeypatch.setenv("RN_TRAIN_KERNEL_ID", "tensorhydra/rsna-knee-screen-b")
    monkeypatch.delenv("RN_TRAIN_KERNEL_TITLE", raising=False)
    nb = tmp_path / "nb.ipynb"
    nb.write_text("{}", encoding="utf-8")
    bn.write_kernel("01_train.ipynb", nb)
    meta = json.loads((tmp_path / "kernels" / "train" /
                       "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["dataset_sources"] == [
        "stevenleehans/rsna-knee-llm-report-labels"]
    assert meta["id"] == "tensorhydra/rsna-knee-screen-b"
    assert bn.KERNELS["01_train.ipynb"].get("dataset_sources", []) == []
    assert bn.KERNELS["01_train.ipynb"]["id"] == "tensorhydra/rsna-knee-meta-train"


def test_infer_kernel_attaches_folds_dataset():
    meta = json.loads((REPO / "kernels" / "infer" / "kernel-metadata.json")
                      .read_text(encoding="utf-8"))
    assert "tensorhydra/rsna-knee-meta-folds-v2" in meta["dataset_sources"]
    assert bn.FOLDS_SLUG in meta["dataset_sources"]
    assert "tensorhydra/rsna-knee-meta-folds" not in meta["dataset_sources"]
    train_meta = json.loads((REPO / "kernels" / "train" /
                             "kernel-metadata.json").read_text(encoding="utf-8"))
    assert train_meta["dataset_sources"] == []
