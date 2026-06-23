#!/usr/bin/env bash
# Full CS-WAE pipeline on a single GPU (sequential).
#
# Usage:
#   bash run_dataset_pipeline_single_gpu.sh cifar10 cuda:1
#   bash run_dataset_pipeline_single_gpu.sh cifar10 cuda:1 resnet18
#
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

DATASET="${1:?dataset required, e.g. cifar10}"
GPU="${2:-cuda:0}"
BACKBONE="${3:-${BACKBONE:-cnn}}"

if [[ "$DATASET" == "cifar10" ]] || [[ "$DATASET" == "svhn" ]]; then
  BACKBONE="${BACKBONE:-resnet18}"
fi

if [[ "$BACKBONE" != "cnn" ]]; then
  RUNS_DIR="runs/${DATASET}_${BACKBONE}"
else
  RUNS_DIR="runs/${DATASET}"
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR" "$RUNS_DIR"

EXTRA_VIZ=(--skip-viz --skip-advanced-viz)
BACKBONE_ARGS=(--backbone "$BACKBONE")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="$LOG_DIR/pipeline_${DATASET}_${BACKBONE}_single_${GPU//:/}_${TIMESTAMP}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"
}

log "=== CS-WAE Single-GPU Pipeline ==="
log "dataset=$DATASET  backbone=$BACKBONE  gpu=$GPU  runs=$RUNS_DIR"

for seed in 0 1 2; do
  log "Training seed $seed on $GPU"
  python train_cs_wae.py \
    --dataset "$DATASET" --seed "$seed" --device "$GPU" \
    --output-dir "$RUNS_DIR/seed_$seed" \
    "${BACKBONE_ARGS[@]}" "${EXTRA_VIZ[@]}"
  log "Done seed $seed"
done

ABLATION_DIR="$RUNS_DIR/ablation_${TIMESTAMP}"
mkdir -p "$ABLATION_DIR"
log "Ablation → $ABLATION_DIR"
python run_ablation_study.py \
  --dataset "$DATASET" --seed 0 --device "$GPU" \
  --variants baseline no_sup_mmd minimal euclidean vmf_prior \
  --results-dir "$ABLATION_DIR" \
  "${BACKBONE_ARGS[@]}"

BASELINE_DIR="$RUNS_DIR/baselines/seed_0"
mkdir -p "$BASELINE_DIR"
log "Baselines → $BASELINE_DIR"
python compare_baselines.py \
  --dataset "$DATASET" --seed 0 --device "$GPU" --epochs 50 \
  --output-dir "$BASELINE_DIR" \
  "${BACKBONE_ARGS[@]}"

log "Aggregate"
python aggregate_results.py --runs-dir "$RUNS_DIR"

log "=== Pipeline complete ==="
log "Runs:       $RUNS_DIR/seed_{0,1,2}/"
log "Ablation:   $ABLATION_DIR/"
log "Baselines:  $BASELINE_DIR/"
log "Aggregate:  $RUNS_DIR/aggregated_metrics.json"
log "Log:        $PIPELINE_LOG"
