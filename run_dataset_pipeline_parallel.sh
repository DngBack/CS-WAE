#!/usr/bin/env bash
# Full CS-WAE pipeline for any supported dataset (2 GPUs parallel).
#
# Usage:
#   bash run_dataset_pipeline_parallel.sh kmnist
#   bash run_dataset_pipeline_parallel.sh emnist_letters
#   DATASET=kmnist GPU0=cuda:0 GPU1=cuda:1 bash run_dataset_pipeline_parallel.sh
#
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

DATASET="${1:-${DATASET:-kmnist}}"
GPU0="${GPU0:-cuda:0}"
GPU1="${GPU1:-cuda:1}"
SEEDS=(0 1 2)
RUNS_DIR="runs/${DATASET}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" "$RUNS_DIR"

# Skip heavy viz for color / many-class datasets
EXTRA_VIZ=()
if [[ "$DATASET" == "emnist_letters" ]] || [[ "$DATASET" == "cifar10" ]] || [[ "$DATASET" == "svhn" ]]; then
  EXTRA_VIZ=(--skip-advanced-viz)
fi
if [[ "$DATASET" == "cifar10" ]] || [[ "$DATASET" == "svhn" ]]; then
  EXTRA_VIZ+=(--skip-viz)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="$LOG_DIR/pipeline_${DATASET}_${TIMESTAMP}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG" >&2
}

LAST_BG_PID=0
run_bg() {
  local name="$1"
  shift
  log "START [$name] $*"
  "$@" >> "$PIPELINE_LOG" 2>&1 &
  LAST_BG_PID=$!
}

log "=== CS-WAE Pipeline — dataset=$DATASET  GPU0=$GPU0  GPU1=$GPU1  seeds=${SEEDS[*]} ==="

# ── Phase 1: Multi-seed training ─────────────────────────────────────────────
run_bg "seed_0" python train_cs_wae.py \
  --dataset "$DATASET" --seed 0 --device "$GPU0" \
  --output-dir "$RUNS_DIR/seed_0" "${EXTRA_VIZ[@]}"
P1A=$LAST_BG_PID
run_bg "seed_1" python train_cs_wae.py \
  --dataset "$DATASET" --seed 1 --device "$GPU1" \
  --output-dir "$RUNS_DIR/seed_1" --skip-advanced-viz "${EXTRA_VIZ[@]}"
P1B=$LAST_BG_PID
wait "$P1A" "$P1B"
log "Done seeds 0 & 1"

log "Training seed 2 on $GPU0"
python train_cs_wae.py \
  --dataset "$DATASET" --seed 2 --device "$GPU0" \
  --output-dir "$RUNS_DIR/seed_2" --skip-advanced-viz "${EXTRA_VIZ[@]}" \
  2>&1 | tee -a "$PIPELINE_LOG"

# ── Phase 2: Ablation ────────────────────────────────────────────────────────
ABLATION_DIR="$RUNS_DIR/ablation_${TIMESTAMP}"
mkdir -p "$ABLATION_DIR"

run_bg "ablation_gpu0" python run_ablation_study.py \
  --dataset "$DATASET" --seed 0 --device "$GPU0" \
  --variants baseline no_sup_mmd minimal \
  --results-dir "$ABLATION_DIR" --skip-aggregate
P2A=$LAST_BG_PID
run_bg "ablation_gpu1" python run_ablation_study.py \
  --dataset "$DATASET" --seed 0 --device "$GPU1" \
  --variants euclidean vmf_prior \
  --results-dir "$ABLATION_DIR" --skip-aggregate
P2B=$LAST_BG_PID
wait "$P2A" "$P2B"
python run_ablation_study.py --summarize-only --results-dir "$ABLATION_DIR" \
  2>&1 | tee -a "$PIPELINE_LOG"

# ── Phase 3: Baselines ───────────────────────────────────────────────────────
BASELINE_DIR="$RUNS_DIR/baselines/seed_0"
mkdir -p "$BASELINE_DIR"

run_bg "baseline_gpu0" python compare_baselines.py \
  --dataset "$DATASET" --seed 0 --device "$GPU0" --epochs 50 \
  --output-dir "$BASELINE_DIR" --models VAE WAE-MMD --skip-summary
P3A=$LAST_BG_PID
run_bg "baseline_gpu1" python compare_baselines.py \
  --dataset "$DATASET" --seed 0 --device "$GPU1" --epochs 50 \
  --output-dir "$BASELINE_DIR" --models VaDE CS-WAE --skip-summary
P3B=$LAST_BG_PID
wait "$P3A" "$P3B"
python compare_baselines.py --dataset "$DATASET" --summarize-only --output-dir "$BASELINE_DIR" \
  2>&1 | tee -a "$PIPELINE_LOG"

# ── Phase 4: Aggregate ───────────────────────────────────────────────────────
python aggregate_results.py --runs-dir "$RUNS_DIR" 2>&1 | tee -a "$PIPELINE_LOG"

log "=== Pipeline complete ==="
log "Runs:       $RUNS_DIR/seed_{0,1,2}/"
log "Ablation:   $ABLATION_DIR/"
log "Baselines:  $BASELINE_DIR/"
log "Aggregate:  $RUNS_DIR/aggregated_metrics.json"
log "Log:        $PIPELINE_LOG"
