#!/usr/bin/env bash
# Full MNIST pipeline using 2 GPUs in parallel (cuda:0 + cuda:1)
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

GPU0="${GPU0:-cuda:0}"
GPU1="${GPU1:-cuda:1}"
SEEDS=(0 1 2)
RUNS_DIR="runs/mnist"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" "$RUNS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="$LOG_DIR/pipeline_parallel_${TIMESTAMP}.log"

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

log "=== CS-WAE MNIST Pipeline — 2 GPU parallel ==="
log "GPU0=$GPU0  GPU1=$GPU1  seeds=${SEEDS[*]}"

# ── Phase 1: Multi-seed training ─────────────────────────────────────────────
# Round 1: seed 0 + seed 1 song song
P1A=$(run_bg "seed_0" python train_cs_wae.py \
  --seed 0 --device "$GPU0" --output-dir "$RUNS_DIR/seed_0")
P1B=$(run_bg "seed_1" python train_cs_wae.py \
  --seed 1 --device "$GPU1" --output-dir "$RUNS_DIR/seed_1" --skip-advanced-viz)
wait "$P1A" "$P1B"
log "Done seeds 0 & 1"

# Round 2: seed 2 (dùng GPU0; GPU1 rảnh cho phase tiếp theo nếu muốn overlap)
log "Training seed 2 on $GPU0"
python train_cs_wae.py \
  --seed 2 --device "$GPU0" --output-dir "$RUNS_DIR/seed_2" --skip-advanced-viz \
  2>&1 | tee -a "$PIPELINE_LOG"

# ── Phase 2: Ablation — chia 5 variants trên 2 GPU ───────────────────────────
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
log "Ablation workers done — building summary table"
python run_ablation_study.py --summarize-only --results-dir "$ABLATION_DIR" \
  2>&1 | tee -a "$PIPELINE_LOG"

# ── Phase 3: Baselines — chia models trên 2 GPU ────────────────────────────────
BASELINE_DIR="$RUNS_DIR/baselines/seed_0"
mkdir -p "$BASELINE_DIR"

P3A=$(run_bg "baseline_gpu0" python compare_baselines.py \
  --seed 0 --device "$GPU0" --epochs 50 \
  --output-dir "$BASELINE_DIR" --models VAE WAE-MMD --skip-summary)
P3B=$(run_bg "baseline_gpu1" python compare_baselines.py \
  --seed 0 --device "$GPU1" --epochs 50 \
  --output-dir "$BASELINE_DIR" --models VaDE CS-WAE --skip-summary)
wait "$P3A" "$P3B"
log "Baseline workers done — building comparison CSV"
python compare_baselines.py --summarize-only --output-dir "$BASELINE_DIR" \
  2>&1 | tee -a "$PIPELINE_LOG"

# ── Phase 4: Aggregate multi-seed ─────────────────────────────────────────────
log "Aggregating multi-seed results"
python aggregate_results.py --runs-dir "$RUNS_DIR" 2>&1 | tee -a "$PIPELINE_LOG"

log "=== Pipeline complete ==="
log "Main runs:  $RUNS_DIR/seed_{0,1,2}/"
log "Ablation:   $ABLATION_DIR/"
log "Baselines:  $BASELINE_DIR/"
log "Aggregated: $RUNS_DIR/aggregated_metrics.json"
log "Log:        $PIPELINE_LOG"
