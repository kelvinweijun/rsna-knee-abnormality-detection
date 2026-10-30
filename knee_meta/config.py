"""Meta constants shared by every module and both Kaggle notebooks."""
import os
from pathlib import Path

SEED = 2026

PREPROC = "rgb01"

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
N_LABELS = len(TARGETS)

PLANES = ("Sagittal", "Coronal", "Axial")
SLOTS = [(p, fs) for p in PLANES for fs in (1, 0)]
SLOT_NAMES = ["{}_{}".format(p.upper(), "FLUID" if fs else "STRUCT") for p, fs in SLOTS]
N_SLOT = len(SLOTS)

CROP_MM = 130.0
CACHE_IMG = 336
IMG_SIZES = (336, 224)
TRAIN_IMG_SIZE = 224
EVAL_IMG_SIZE = 224
SLICES_PER_SLOT = 3
SLICE_BAND = (0.20, 0.80)
PLANE_BAND = {
    "Sagittal": (0.12, 0.88),
    "Coronal": (0.20, 0.80),
    "Axial": (0.12, 0.88),
}
LAT_MIN_OFFSET_MM = 20.0

POOL = "cls_mean_focal"
POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}
SLOT_PRIOR_TABLE = {
    "ACL": (0, 3, 5),
    "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4),
    "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5),
    "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5),
    "Effusion": (0, 2),
    "Synovitis": (0, 2),
    "Baker's": (0,),
    "Contusion": (0, 1, 2),
    "Fracture": (0, 1, 2, 4, 5),
}
SLOT_PRIOR_STRENGTH = 0.55

EPOCHS = 10
BATCH_STUDIES = 8
LR_HEAD = 1e-3
LR_BACKBONE = 8e-6
WEIGHT_DECAY = 0.02
UNFREEZE_LAST = 6
WARMUP_STEPS = 100
PAIRWISE_W = 0.10
GOLD_WEIGHT = 8.0
MISSING_CONF = 0.15
MENTION_CONF = 0.70
POSITIVE_CONF = 0.95

AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.10

N_FOLDS = 5
EVAL_BATCH = 8
TIME_BUDGET_S = 8.0 * 3600
HDR_THREADS = 16
PIX_THREADS = 12

DINOV2_CANDIDATES = (
    "/kaggle/input/models/metaresearch/dinov2/pytorch/small/v1",
    "/kaggle/input/models/metaresearch/dinov2/pytorch/small/1",
    "/kaggle/input/dinov2-small",
)

COMPETITION_SLUG = "rsna-knee-abnormality-detection"
LABEL_FILE = "report_labels.csv"


def competition_roots():
    return [
        Path("/kaggle/input/competitions") / COMPETITION_SLUG,
        Path("/kaggle/input") / COMPETITION_SLUG,
        Path("data"),
        Path("."),
    ]


def find_root():
    for cand in competition_roots():
        if (cand / "test.csv").is_file() and (cand / "test_series").is_dir():
            return cand
    base = Path("/kaggle/input")
    if base.is_dir():
        for d1 in sorted(p for p in base.iterdir() if p.is_dir()):
            for d2 in [d1] + sorted(p for p in d1.iterdir() if p.is_dir()):
                if (d2 / "test.csv").is_file() and (d2 / "test_series").is_dir():
                    return d2
    raise FileNotFoundError("competition mount not found")


def find_backbone():
    for cand in DINOV2_CANDIDATES:
        if Path(cand).is_dir() and (Path(cand) / "config.json").is_file():
            return cand
    base = Path("/kaggle/input")
    if base.is_dir():
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
            if "config.json" in files and "dinov2" in root.lower():
                return root
    return None
