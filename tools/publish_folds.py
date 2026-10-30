"""Publish fold checkpoints from the train kernel as a Kaggle dataset.

Downloads only the checkpoint and summary files from the completed run of
``tensorhydra/rsna-knee-meta-train``, stages them at the dataset root (so
``resolve_models_dir`` finds them at ``/kaggle/input/rsna-knee-meta-folds-v2``),
writes ``dataset-metadata.json`` and creates or versions the dataset.

Usage:
    python tools/publish_folds.py [--dry-run] [--force-version]
"""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KERNEL = "tensorhydra/rsna-knee-meta-train"
SLUG = "tensorhydra/rsna-knee-meta-folds-v2"
STAGE = REPO / "folds-data-v2"
PATTERN = r"(fold\d+\.pt|cv_summary\.json)"
METADATA = {
    "title": "RSNA Knee Meta Fold Checkpoints V2",
    "subtitle": "rgb01-preprocessing checkpoints from rsna-knee-meta-train",
    "description": ("Five-fold DINOv2 ViT-S checkpoints trained by the private "
                    "kernel tensorhydra/rsna-knee-meta-train on the RSNA knee "
                    "abnormality detection competition, saved with preproc "
                    "marker rgb01 (inputs float in [0,1]). V2 supersedes "
                    "rsna-knee-meta-folds; the v1 dataset is frozen to pair "
                    "with pre-fix infer kernels."),
    "id": SLUG,
    "licenses": [{"name": "CC0-1.0"}],
}


def sh(cmd, check=True):
    print("+", " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          cwd=str(REPO))
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip())
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {cmd[0]}")
    return proc


def download(stage, force):
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    args = ["kaggle", "kernels", "output", KERNEL, "-p", stage,
            "--file-pattern", PATTERN]
    if force:
        args.append("-o")
    proc = sh(args, check=False)
    if proc.returncode != 0:
        raise SystemExit("download failed - is the run complete?")
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    if not files:
        raise SystemExit("no checkpoint files downloaded - is the run complete?")
    return files


def stage_root(files, stage):
    """Flatten matches to the dataset root and drop everything else."""
    matches = [p for p in files if re.search(PATTERN, p.name)]
    keep = set()
    for p in matches:
        target = stage / p.name
        if p != target:
            shutil.move(str(p), str(target))
        keep.add(target.resolve())
    for p in sorted(stage.rglob("*"), reverse=True):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists() and p.resolve() not in keep:
            p.unlink()
    return [stage / p.name for p in matches]


def require_checkpoints(moved):
    if not any(p.name.startswith("fold") for p in moved):
        raise SystemExit("no fold*.pt in download - refusing to publish")
    return moved


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="download and stage only, do not upload")
    ap.add_argument("--force", action="store_true",
                    help="force re-download and new dataset version")
    args = ap.parse_args(argv)

    files = download(STAGE, args.force)
    moved = require_checkpoints(stage_root(files, STAGE))
    print("staged:", ", ".join(p.name for p in moved))
    (STAGE / "dataset-metadata.json").write_text(
        json.dumps(METADATA, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"dry run: {STAGE} ready, not uploaded")
        return

    exists = sh(["kaggle", "datasets", "list", "-s", "rsna-knee-meta-folds-v2"],
                check=False)
    if SLUG in exists.stdout and not args.force:
        sh(["kaggle", "datasets", "version", "-p", STAGE,
            "-m", "fold checkpoints"])
    else:
        proc = sh(["kaggle", "datasets", "create", "-p", STAGE], check=False)
        if proc.returncode != 0:
            sh(["kaggle", "datasets", "version", "-p", STAGE,
                "-m", "fold checkpoints"])
    print(f"published {SLUG} - attach as dataset_sources to the infer kernel")


if __name__ == "__main__":
    main()
