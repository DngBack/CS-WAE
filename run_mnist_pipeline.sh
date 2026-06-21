#!/usr/bin/env bash
# Full MNIST pipeline: multi-seed CS-WAE + ablation + baselines + aggregation
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

SEEDS=(0 1 2)
RUNS_DIR="runs/mnist"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" "$RUNS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"
}

log "=== CS-WAE MNIST Pipeline (seeds: ${SEEDS[*]}) ==="

# B1: Multi-seed main training
for SEED in "${SEEDS[@]}"; do
  OUT_DIR="$RUNS_DIR/seed_${SEED}"
  if [[ "$SEED" == "0" ]]; then
    log "Training CS-WAE seed=$SEED (full viz) -> $OUT_DIR"
    python train_cs_wae.py --seed "$SEED" --output-dir "$OUT_DIR" 2>&1 | tee -a "$PIPELINE_LOG"
  else
    log "Training CS-WAE seed=$SEED (skip advanced viz) -> $OUT_DIR"
    python train_cs_wae.py --seed "$SEED" --output-dir "$OUT_DIR" --skip-advanced-viz 2>&1 | tee -a "$PIPELINE_LOG"
  fi
done

# A2: Ablation study (single seed; use seed 0 for reproducibility)
ABLATION_DIR="$RUNS_DIR/ablation_${TIMESTAMP}"
log "Ablation study -> $ABLATION_DIR"
python run_ablation_study.py --seed 0 --results-dir "$ABLATION_DIR" 2>&1 | tee -a "$PIPELINE_LOG"

# A3: Fair baselines (50 epochs, VaDE enabled)
BASELINE_DIR="$RUNS_DIR/baselines/seed_0"
log "Baseline comparison -> $BASELINE_DIR"
python compare_baselines.py --seed 0 --epochs 50 --output-dir "$BASELINE_DIR" 2>&1 | tee -a "$PIPELINE_LOG"

# Aggregate multi-seed metrics
log "Aggregating multi-seed results"
python aggregate_results.py --runs-dir "$RUNS_DIR" 2>&1 | tee -a "$PIPELINE_LOG"

log "=== Pipeline complete ==="
log "Main runs:     $RUNS_DIR/seed_{0,1,2}/"
log "Ablation:      $ABLATION_DIR/"
log "Baselines:     $BASELINE_DIR/"
log "Aggregated:    $RUNS_DIR/aggregated_metrics.json"
log "Log:           $PIPELINE_LOG"
