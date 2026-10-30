#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu GPU box (>=45 GB free disk, internet on).
# Downloads: raptor corpus (2 parts, 22 GB), OAI 3D-DESS (12.5 GB), fused labels (private).
# Usage: KAGGLE_API_TOKEN=<token> bash setup_box.sh
set -euo pipefail

DATA="${DATA:-$HOME/data}"

if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  echo "KAGGLE_API_TOKEN is not set - need it for the private labels dataset"; exit 1
fi

python -m pip install -q --upgrade kaggle

if ! python -c "import torch" 2>/dev/null; then
  echo "WARNING: torch not importable - use a PyTorch CUDA image (e.g. pytorch/pytorch:2.x-cuda12.x)"
fi
python -c "import timm" 2>/dev/null || python -m pip install -q timm
python -c "import pandas, numpy" 2>/dev/null || python -m pip install -q pandas numpy

FREE_GB=$(df -Pk "$HOME" | awk 'NR==2 {printf "%d", $4/1048576}')
if [ "$FREE_GB" -lt 45 ]; then
  echo "WARNING: only ${FREE_GB} GB free under $HOME; need ~40 GB"
fi

download() {  # download <slug> <dir>
  local slug="$1" dir="$DATA/$2"
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "== skip $slug (already present)"; return 0
  fi
  echo "== downloading $slug -> $dir"
  mkdir -p "$dir"
  python -m kaggle datasets download -d "$slug" -p "$dir" --unzip
}

download dreaddevelopment/knee-raptor-corpus     corpus
download dreaddevelopment/knee-raptor-corpus-ext corpus_ext
download mohamedberrimi/oaimri3ddess             oai
download sixmillionfold/rsna-knee-fused-labels   labels

echo "== verifying expected files"
for f in corpus/all_vols.npy corpus/all_masks.npy corpus/all_ids.npy \
         corpus_ext/extra_vols.npy corpus_ext/extra_masks.npy corpus_ext/extra_ids.npy \
         oai/normal-3DESS-128-64.npy oai/abnormal-3DESS-128-64.npy \
         labels/fused_labels.csv labels/fused_omer.csv; do
  if [ -f "$DATA/$f" ]; then echo "  OK  $f"; else echo "  MISS $f"; fi
done
du -sh "$DATA"/* 2>/dev/null || true
echo "== setup done. Next: DATA=$DATA bash run_train.sh"
