"""Generate the two Kaggle notebooks from the knee_meta sources."""
import json
import os
import shlex
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "knee_meta"
OUT = REPO / "notebooks"

MODULES = ["__init__.py", "config.py", "labels.py", "imaging.py", "model.py",
           "dataset.py", "metrics.py", "train.py", "infer.py"]


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def writefile_cells():
    cells = []
    for name in MODULES:
        body = (SRC / name).read_text(encoding="utf-8")
        cells.append(code(f"%%writefile knee_meta/{name}\n{body}"))
    return cells


MKDIR = '''from pathlib import Path
Path("knee_meta").mkdir(exist_ok=True)'''

SETUP = '''import json, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.getcwd())
import torch
from knee_meta.config import (CACHE_IMG, EPOCHS, N_FOLDS, TIME_BUDGET_S,
                              TRAIN_IMG_SIZE, find_root, find_backbone)
root = find_root()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("root    :", root)
print("device  :", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
print("backbone:", find_backbone())
print("cache   :", CACHE_IMG, "px | folds", N_FOLDS, "| epochs", EPOCHS,
      "| train img", TRAIN_IMG_SIZE, "| budget", TIME_BUDGET_S / 3600, "h")'''

LABELS = '''from knee_meta.labels import audit, load_label_table, save_label_table
label_path = Path("report_labels.csv")
if label_path.is_file():
    label_table = load_label_table(label_path)
else:
    label_table = save_label_table(root / "train.csv", label_path)
print(json.dumps(audit(label_table, root / "train.csv"), indent=2))
label_table.head()'''

CACHE_TRAIN = '''from knee_meta import imaging
cache_path = Path("cache_train.npy")
if cache_path.is_file() and Path(str(cache_path) + ".mask.npz").is_file():
    print("train cache present:", cache_path)
else:
    t0 = time.time()
    cache, mask, studies = imaging.build_cache(str(root), "train", img_size=CACHE_IMG)
    imaging.save_cache(str(cache_path), cache, mask, studies)
    print("built", tuple(cache.shape), "in %.1f s" % (time.time() - t0))'''

RUN_TRAIN = '''from knee_meta import train
results = train.main([
    "--root", str(root),
    "--cache", "cache_train.npy",
    "--labels", "report_labels.csv",
    "--out", "models",
    "--folds", str(N_FOLDS),
    "--epochs", str(EPOCHS),
    "--img-size", str(TRAIN_IMG_SIZE),
    "--budget", str(TIME_BUDGET_S),
])
results'''


def run_train_src(extra=None):
    """RUN_TRAIN, optionally with extra CLI args (RN_TRAIN_EXTRA env, build time)."""
    if extra is None:
        extra = os.environ.get("RN_TRAIN_EXTRA", "")
    items = "".join(f'    {json.dumps(tok)},\n' for tok in shlex.split(extra))
    if not items:
        return RUN_TRAIN
    return RUN_TRAIN.replace('    "--budget", str(TIME_BUDGET_S),\n',
                             '    "--budget", str(TIME_BUDGET_S),\n' + items)

CACHE_TEST = '''from knee_meta import imaging
test_cache = Path(f"cache_test_{EVAL_IMG}.npy")
if test_cache.is_file() and Path(str(test_cache) + ".mask.npz").is_file():
    print("test cache present:", test_cache)
else:
    t0 = time.time()
    cache, mask, studies = imaging.build_cache(str(root), "test", img_size=EVAL_IMG)
    imaging.save_cache(str(test_cache), cache, mask, studies)
    print("built", tuple(cache.shape), "in %.1f s" % (time.time() - t0))'''

RUN_INFER = '''from knee_meta import infer
submission = infer.main([
    "--root", str(root),
    "--cache", f"cache_test_{EVAL_IMG}.npy",
    "--models", "models",
    "--out", "/kaggle/working/submission.csv",
    "--img-size", str(EVAL_IMG),
])
submission.head()'''

META_MD = """# RSNA Knee Abnormality Detection - training notebook

Recipe taken from the current top of the public leaderboard and discussion tab:

- **Labels**: the twelve findings are weak labels extracted from the free-text
  report (multilingual rule extractor, per-finding confidence) with the 58 gold
  studies weighted `GOLD_WEIGHT` above every derived row. A report that never
  mentions a finding is an unknown at low confidence, not a negative.
- **Images**: 130 mm physical centre crop, canonical laterality from geometry,
  `CACHE_IMG` uint8 cache, three consecutive slices in the 0.20-0.80 band of
  each plane stacked as the three encoder channels, six slots
  (Sagittal/Coronal/Axial x fluid-sensitive / structural).
- **Model**: DINOv2 ViT-S/14, last `UNFREEZE_LAST` blocks open, slot-attention
  head with one query per finding and an anatomical slot prior.
- **Optimisation**: AdamW, head 1e-3 / backbone 8e-6, weight decay 0.02,
  confidence-weighted BCE plus a pairwise ranking term, `EPOCHS` per fold,
  macro-AUC on the held-out fold, best state saved.
- **Budget**: one notebook run stays inside `TIME_BUDGET_S` on a single GPU,
  internet off, competition data plus the attached DINOv2 weights only."""

SUB_MD = """# RSNA Knee Abnormality Detection - inference notebook

Loads every fold checkpoint written by the training notebook, averages logits
across folds and slice groups, converts the mean to per-target percentile ranks
(the metric only reads the order), and writes `submission.csv`.

Offline, single GPU, competition data plus attached weights only. Select
`submission.csv` in the Submit to Competition flow; this notebook does not
submit on its own."""


def build():
    OUT.mkdir(exist_ok=True)
    train_nb = {
        "cells": [md(META_MD), code(MKDIR)] + writefile_cells() + [
            code(SETUP), code(LABELS), code(CACHE_TRAIN), code(run_train_src())],
        "metadata": {"kaggle": {"accelerator": "gpu",
                                "isGpuEnabled": True, "isInternetEnabled": False,
                                "language": "python"},
                     "kernelspec": {"display_name": "Python 3",
                                    "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    infer_cells = [md(SUB_MD), code(MKDIR)] + writefile_cells() + [
        code(SETUP + "\nEVAL_IMG = CACHE_IMG"),
        code(CACHE_TEST), code(RUN_INFER)]
    infer_nb = {
        "cells": infer_cells,
        "metadata": {"kaggle": {"accelerator": "gpu",
                                "isGpuEnabled": True, "isInternetEnabled": False,
                                "language": "python"},
                     "kernelspec": {"display_name": "Python 3",
                                    "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    for name, nb in (("01_train.ipynb", train_nb), ("02_infer.ipynb", infer_nb)):
        for i, cell in enumerate(nb["cells"]):
            cell["id"] = f"{name[:2]}{i:02d}"
        path = OUT / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", path, len(nb["cells"]), "cells")
        write_kernel(name, path)


DINOV2 = "metaresearch/dinov2/pytorch/small/1"
COMP_SLUG = "rsna-knee-abnormality-detection"
FOLDS_SLUG = "tensorhydra/rsna-knee-meta-folds-v2"

KERNELS = {
    "01_train.ipynb": {
        "dir": "train",
        "id": "tensorhydra/rsna-knee-meta-train",
        "title": "RSNA Knee Meta Train",
    },
    "02_infer.ipynb": {
        "dir": "infer",
        "id": "tensorhydra/rsna-knee-meta-infer",
        "title": "RSNA Knee Meta Infer",
        "dataset_sources": [FOLDS_SLUG],
    },
}


def write_kernel(nb_name, nb_path):
    spec = dict(KERNELS[nb_name])
    if spec["dir"] == "train":
        spec["id"] = os.environ.get("RN_TRAIN_KERNEL_ID", spec["id"])
        spec["title"] = os.environ.get("RN_TRAIN_KERNEL_TITLE", spec["title"])
        extra_ds = [s for s in os.environ.get("RN_TRAIN_DATASETS", "").split(",")
                    if s]
        spec["dataset_sources"] = list(spec.get("dataset_sources", [])) + extra_ds
    kdir = REPO / "kernels" / spec["dir"]
    kdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(nb_path, kdir / nb_name)
    meta = {
        "id": spec["id"],
        "title": spec["title"],
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "",
        "dataset_sources": list(spec.get("dataset_sources", [])),
        "competition_sources": [COMP_SLUG],
        "kernel_sources": [],
        "model_sources": [DINOV2],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2),
                                                encoding="utf-8")
    print("wrote", kdir / "kernel-metadata.json")


if __name__ == "__main__":
    build()
