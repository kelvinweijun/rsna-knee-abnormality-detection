#!/usr/bin/env bash
# Train OAI-supervised Raptor folds on the box, then stage a Kaggle upload.
# Usage: DATA=$HOME/data FOLDS="0 1 2 3" bash run_train.sh
# Env knobs: ARCH RES EPOCHS BS SEED TAGPREFIX WARM(kaggle names, e.g. sixmillionfold) UPLOAD_OWNER
set -euo pipefail

DATA="${DATA:-$HOME/data}"
RUN="${RUN:-$HOME/run}"
ARCH="${ARCH:-coatnet_rmlp_2_rw_384}"
RES="${RES:-384}"
EPOCHS="${EPOCHS:-16}"
BS="${BS:-4}"
SEED="${SEED:-42}"
TAGPREFIX="${TAGPREFIX:-oai384}"
FOLDS="${FOLDS:-0 1 2 3}"
LABELS_FILE="${LABELS_FILE:-$DATA/labels/fused_labels.csv}"
WARM="${WARM:-}"                 # path to a .pt to warm-start from; empty = timm-pretrained

mkdir -p "$RUN"; cd "$RUN"
if [ ! -f train_jev_oai_box.py ]; then
  echo "train_jev_oai_box.py missing in $RUN - copy it there first"; exit 1
fi
if [ ! -f "$LABELS_FILE" ]; then
  echo "labels file missing: $LABELS_FILE (run setup_box.sh first)"; exit 1
fi

export RAPTOR_INPUT="$DATA"
export RAPTOR_INIT="$WARM"

for F in $FOLDS; do
  echo "===== fold $F / arch $ARCH res $RES epochs $EPOCHS bs $BS seed $SEED ====="
  python -u train_jev_oai_box.py --ckpt timm --arch "$ARCH" --res "$RES" \
    --k 12 --k_eval 24 --epochs "$EPOCHS" --bs "$BS" \
    --bb_lr 2e-5 --head_lr 1e-3 --wd 0.02 --workers 4 --topk 3 \
    --grad_ckpt --folds 5 --fold "$F" --seed "$SEED" \
    --labels "$LABELS_FILE" --tag "${TAGPREFIX}_f${F}" 2>&1 | tee "log_${TAGPREFIX}_f${F}.txt"
done

mkdir -p upload
cp -v raptor_ft_${TAGPREFIX}_f*_swa.pt upload/ 2>/dev/null || true
cp -v raptor_gold_${TAGPREFIX}_f*_swa.npz upload/ 2>/dev/null || true
cp -v raptor_ft_${TAGPREFIX}_f*.json upload/ 2>/dev/null || true
cp -v log_${TAGPREFIX}_f*.txt upload/ 2>/dev/null || true

OWNER="${UPLOAD_OWNER:-sixmillionfold}"
cat > upload/dataset-metadata.json <<EOF
{"title":"RSNA Knee OAI Raptor ${TAGPREFIX}","id":"${OWNER}/rsna-knee-oai-${TAGPREFIX}","licenses":[{"name":"CC0-1.0"}]}
EOF
echo "== staged $(ls upload | wc -l) files in $RUN/upload"
echo "== upload with:  python -m kaggle datasets create -p $RUN/upload"
