"""Offline inference: fold and group average, then per-target rank blend."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import (EVAL_BATCH, EVAL_IMG_SIZE, PREPROC, TARGETS, find_root)
from .imaging import build_cache, load_cache, save_cache
from .metrics import percentile_rank
from .model import build_model

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

MODEL_CANDIDATES = (
    "models",
    "/kaggle/input/rsna-knee-meta-folds-v2",
    "/kaggle/input/datasets/tensorhydra/rsna-knee-meta-folds-v2",
    "/kaggle/working/models",
)


def find_checkpoints(directory):
    p = Path(directory)
    if not p.is_dir():
        return []
    return sorted(p.glob("fold*.pt"))


def resolve_models_dir(preferred=None, candidates=None):
    """Locate fold*.pt checkpoints: explicit dir, then known mount points."""
    env = os.environ.get("RSNA_MODELS_DIR")
    order = [d for d in (env, preferred) if d]
    order += [d for d in (candidates or MODEL_CANDIDATES) if d not in order]
    for d in order:
        if find_checkpoints(d):
            return str(Path(d))
    raise FileNotFoundError(
        "no fold*.pt checkpoints found; looked in: " + ", ".join(order))


def load_models(models_dir, device, backbone=None, unfreeze=None, tiny=False):
    from .config import UNFREEZE_LAST
    paths = sorted(Path(models_dir).glob("fold*.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints in {models_dir}")
    models = []
    for p in paths:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        if blob.get("preproc") != PREPROC:
            raise ValueError(
                f"{p.name}: checkpoint preproc {blob.get('preproc')!r} != "
                f"infer {PREPROC!r}; checkpoint/inference input contract "
                f"mismatch — use matching checkpoints")
        model = build_model(unfreeze_last=unfreeze or UNFREEZE_LAST,
                            source=backbone, img_size=blob.get("img_size"),
                            tiny=tiny)
        model.load_state_dict(blob["state_dict"])
        model.to(device).eval()
        models.append((p.name, model, blob.get("img_size")))
    return models


@torch.no_grad()
def predict(models, cache, mask, device, batch=EVAL_BATCH, groups=None):
    n = cache.shape[0]
    n_groups = max(1, cache.shape[2] // 3)
    groups = list(range(n_groups)) if groups is None else list(groups)
    total = np.zeros((n, len(TARGETS)), dtype=np.float64)
    count = 0
    for _, model, img_size in models:
        for g in groups:
            start = g * 3
            block = cache[:, :, start:start + 3]
            if block.shape[2] < 3:
                pad = np.repeat(block[:, :, -1:], 3 - block.shape[2], axis=2)
                block = np.concatenate([block, pad], axis=2)
            for i in range(0, n, batch):
                x = torch.from_numpy(block[i:i + batch]).float().div(255.0).to(device)
                m = torch.from_numpy(mask[i:i + batch]).float().to(device)
                logits = model(x, m, img_size=img_size)
                total[i:i + batch] += logits.float().cpu().numpy()
            count += 1
    return total / max(count, 1)


def ensure_test_cache(root, img_size, out=None):
    out = Path(out) if out else Path(root) / f"cache_test_{img_size}.npy"
    if out.is_file() and Path(str(out) + ".mask.npz").is_file():
        return load_cache(str(out))
    cache, mask, studies = build_cache(str(root), "test", img_size=img_size,
                                       progress=_progress("test"))
    save_cache(str(out), cache, mask, studies)
    return cache, mask, studies


def _progress(tag):
    if tqdm is None:
        return None
    bar = {"n": 0}

    def hook(done, total):
        if bar["n"] == 0:
            bar["n"] = 1
            bar["bar"] = tqdm(total=total, desc=tag)
        bar["bar"].n = done
        bar["bar"].refresh()
    return hook


def load_template(root):
    """Study order comes from test.csv (guaranteed replaced at scoring).

    sample_submission.csv is only cross-checked: the competition docs list
    test.csv / test_series.csv / test_series/ as the swapped files, so a
    template built from anything else can silently shrink to the 3-row
    example during scoring.
    """
    root = Path(root)
    frames = {}
    for name in ("test.csv", "sample_submission.csv", "submission.csv"):
        p = root / name
        if p.is_file():
            frames[name] = pd.read_csv(p, dtype={"StudyInstanceUID": str})
    if "test.csv" in frames:
        uids = frames["test.csv"].StudyInstanceUID
        source = "test.csv"
    elif "sample_submission.csv" in frames:
        uids = frames["sample_submission.csv"].StudyInstanceUID
        source = "sample_submission.csv"
    elif "submission.csv" in frames:
        uids = frames["submission.csv"].StudyInstanceUID
        source = "submission.csv"
    else:
        raise FileNotFoundError(f"no submission template or test.csv in {root}")
    sample = frames.get("sample_submission.csv")
    if source != "sample_submission.csv" and sample is not None:
        extra = set(uids) - set(sample.StudyInstanceUID)
        dropped = set(sample.StudyInstanceUID) - set(uids)
        if extra or dropped:
            print(f"template cross-check: {source} has {len(uids)} studies, "
                  f"sample_submission has {len(sample)} "
                  f"(+{len(extra)}/-{len(dropped)}); using {source}",
                  flush=True)
    print(f"template: {len(uids)} studies from {source}", flush=True)
    sub = pd.DataFrame({"StudyInstanceUID": list(uids)})
    for lab in TARGETS:
        sub[lab] = 0.5
    return sub


def check_coverage(sub, studies, max_frac=0.02):
    """Template vs cache coverage; raises when most studies would stay 0.5."""
    known = set(studies)
    missing = int(sum(1 for s in sub.StudyInstanceUID if s not in known))
    frac = missing / max(len(sub), 1)
    print(f"coverage: template {len(sub)}, cache {len(studies)}, "
          f"missing {missing} ({frac:.1%})", flush=True)
    if missing and len(sub) > 10 and frac > max_frac:
        raise ValueError(
            f"{missing}/{len(sub)} template studies missing from cache "
            f"({frac:.1%} > {max_frac:.0%}); refusing to submit a "
            f"0.5-filled file")
    if missing and len(sub) <= 10:
        print(f"warning: {missing} template studies missing from cache "
              f"(kept 0.5)", flush=True)
    return missing


def fill_submission(sub, scores, studies, max_frac=0.02):
    """Write scores into the template; small gaps keep 0.5, big gaps raise."""
    if scores.shape != (len(studies), len(TARGETS)):
        raise ValueError(f"scores {scores.shape} does not match "
                         f"{len(studies)} studies x {len(TARGETS)} targets")
    order = {s: i for i, s in enumerate(studies)}
    rows = np.array([order.get(s, -1) for s in sub.StudyInstanceUID])
    known = rows >= 0
    for j, lab in enumerate(TARGETS):
        col = np.full(len(sub), 0.5)
        col[known] = scores[rows[known], j]
        sub[lab] = col
    check_coverage(sub, studies, max_frac=max_frac)
    if len(sub) > 10:
        flat = [lab for lab in TARGETS if (sub[lab] == 0.5).all()]
        if flat:
            raise ValueError(
                f"constant 0.5 column(s) {flat}: predictions carry no "
                f"ranking, the fallback path fired")
    return sub


def check_slots(mask, studies, min_fill=0.5, max_empty_frac=0.02):
    """Decode health: every study enters the cache even when DICOM decode
    fails, so empty slots are the only signal that predictions would be
    constant ties."""
    fill = float(mask.mean()) if mask.size else 0.0
    empty = int((mask.sum(axis=1) == 0).sum()) if mask.size else 0
    n = len(studies)
    print(f"slots: fill {fill:.1%}, studies with no decoded slots "
          f"{empty}/{n}", flush=True)
    if n > 10 and (fill < min_fill or empty > max_empty_frac * n):
        raise ValueError(
            f"slot decode failed: fill {fill:.1%} (< {min_fill:.0%}) or "
            f"{empty}/{n} studies with no slots (> {max_empty_frac:.0%}) — "
            f"predictions would be mostly ties")
    return fill, empty


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--batch", type=int, default=EVAL_BATCH)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--rank", action="store_true", default=True)
    args = ap.parse_args(argv)

    root = Path(args.root) if args.root else find_root()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))
    models_dir = resolve_models_dir(args.models)
    print(f"checkpoints: {models_dir}", flush=True)
    models = load_models(models_dir, device, backbone=args.backbone,
                         tiny=args.tiny)
    img_size = args.img_size or models[0][2] or EVAL_IMG_SIZE

    if args.cache:
        cache, mask, studies = load_cache(args.cache)
    else:
        cache, mask, studies = ensure_test_cache(root, img_size)
    studies = [str(s) for s in studies]
    check_slots(mask, studies)
    sub = load_template(root)
    check_coverage(sub, studies)
    groups = None if args.groups is None else [int(g) for g in
                                                args.groups.split(",")]
    print(f"test cache {tuple(cache.shape)} {len(models)} models", flush=True)
    scores = predict(models, cache, mask, device, batch=args.batch, groups=groups)
    if args.rank:
        scores = percentile_rank(scores)

    fill_submission(sub, scores, studies)
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(sub)} studies, {len(TARGETS)} targets",
          flush=True)
    return sub


if __name__ == "__main__":
    main()
