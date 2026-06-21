#!/usr/bin/env bash
# Resume pipeline from seed_2 onward (skip completed seed_0/seed_1)
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

GPU0="${GPU0:-cuda:0}"
GPU1="${GPU1:-cuda:1}"
RUNS_DIR="runs/mnist"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" "$RUNS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="$LOG_DIR/pipeline_resume_${TIMESTAMP}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG" >&2
}

run_bg() {
  local name="$1"
  shift
  log "START [$name] $*"
  "$@" >> "$PIPELINE_LOG" 2>&1 &
  echo $!
}

log "=== Resume pipeline (seed_2 + ablation + baselines) ==="

# seed_2
if [[ -f "$RUNS_DIR/seed_2/metrics.json" ]]; then
  log "Skip seed_2 — already done"
else
  log "Training seed 2 on $GPU0"
  python train_cs_wae.py \
    --seed 2 --device "$GPU0" --output-dir "$RUNS_DIR/seed_2" --skip-advanced-viz \
    2>&1 | tee -a "$PIPELINE_LOG"
fi

# Ablation
ABLATION_DIR="$RUNS_DIR/ablation_${TIMESTAMP}"
mkdir -p "$ABLATION_DIR"

P2A=$(run_bg "ablation_gpu0" python run_ablation_study.py \
  --seed 0 --device "$GPU0" \
  --variants baseline no_sup_mmd minimal \
  --results-dir "$ABLATION_DIR" --skip-aggregate)
P2B=$(run_bg "ablation_gpu1" python run_ablation_study.py \
  --seed 0 --device "$GPU1" \
  --variants euclidean vmf_prior \
  --results-dir "$ABLATION_DIR" --skip-aggregate)
wait "$P2A" "$P2B"
log "Ablation workers done"
python run_ablation_study.py --summarize-only --results-dir "$ABLATION_DIR" \
  2>&1 | tee -a "$PIPELINE_LOG"

# Baselines
BASELINE_DIR="$RUNS_DIR/baselines/seed_0"
mkdir -p "$BASELINE_DIR"

P3A=$(run_bg "baseline_gpu0" python compare_baselines.py \
  --seed 0 --device "$GPU0" --epochs 50 \
  --output-dir "$BASELINE_DIR" --models VAE WAE-MMD --skip-summary)
P3B=$(run_bg "baseline_gpu1" python compare_baselines.py \
  --seed 0 --device "$GPU1" --epochs 50 \
  --output-dir "$BASELINE_DIR" --models VaDE CS-WAE --skip-summary)
wait "$P3A" "$P3B"
log "Baseline workers done"
python compare_baselines.py --summarize-only --output-dir "$BASELINE_DIR" \
  2>&1 | tee -a "$PIPELINE_LOG"

# Aggregate
log "Aggregating multi-seed results"
python aggregate_results.py --runs-dir "$RUNS_DIR" 2>&1 | tee -a "$PIPELINE_LOG"

log "=== Resume complete ==="
log "Seeds:      $RUNS_DIR/seed_{0,1,2}/"
log "Ablation:   $ABLATION_DIR/"
log "Baselines:  $BASELINE_DIR/"
log "Aggregated: $RUNS_DIR/aggregated_metrics.json"
