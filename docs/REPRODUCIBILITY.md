# CS-WAE Reproducibility Guide

This document describes how to reproduce the main paper experiments on **MNIST**, **Fashion-MNIST**, and **CIFAR-10** using Python only (no shell scripts).

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch with CUDA (recommended for full runs).

Optional sanity check:

```bash
python tests/test_ablation.py
```

## 2. Supported datasets

| Dataset | CLI key | Channels | Size | Classes | Paper |
|---------|---------|----------|------|---------|-------|
| MNIST | `mnist` | 1 | 28×28 | 10 | ✅ |
| Fashion-MNIST | `fashion_mnist` | 1 | 28×28 | 10 | ✅ |
| CIFAR-10 | `cifar10` | 3 | 32×32 | 10 | ✅ |
| KMNIST | `kmnist` | 1 | 28×28 | 10 | optional |
| EMNIST Letters | `emnist_letters` | 1 | 28×28 | 26 | optional |
| SVHN | `svhn` | 3 | 32×32 | 10 | optional |

Data is auto-downloaded to `./data/` on first use.

## 3. Entry points

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | **Recommended** — full or per-phase pipeline |
| `train_cs_wae.py` | Multi-seed CS-WAE training |
| `run_ablation_study.py` | 5 ablation variants |
| `compare_baselines.py` | VAE, WAE-MMD, VaDE, CS-WAE |
| `aggregate_results.py` | Mean ± std over seeds |
| `scripts/plot_paper_figures.py` | Export paper tables/figures |

## 4. Full pipeline (`run_pipeline.py`)

Reproduce Table 1 (3 seeds + ablation + baselines + aggregate):

```bash
# Single GPU (sequential)
python run_pipeline.py all --dataset mnist --device cuda:0

python run_pipeline.py all --dataset fashion_mnist --device cuda:0

python run_pipeline.py all --dataset cifar10 --device cuda:0 --backbone cnn
```

Two GPUs (seeds 0+1 in parallel):

```bash
python run_pipeline.py all --dataset mnist --device cuda:0 --device2 cuda:1
```

Smoke test (2 epochs):

```bash
python run_pipeline.py all --dataset mnist --device cuda:0 --epochs 2
```

Logs are written to `logs/pipeline_{dataset}_{backbone}_{timestamp}.log` (override with `--log-file`).

### Per-phase commands

```bash
python run_pipeline.py train --dataset fashion_mnist --seeds 0 1 2 --device cuda:0
python run_pipeline.py ablation --dataset fashion_mnist --device cuda:0
python run_pipeline.py baselines --dataset fashion_mnist --device cuda:0
python run_pipeline.py aggregate --runs-dir runs/fashion_mnist
```

### Output layout

```
runs/{dataset}/              # CNN backbone (default)
runs/{dataset}_resnet18/     # if --backbone resnet18
├── seed_0/, seed_1/, seed_2/
│   └── metrics.json
├── aggregated_metrics.json
├── ablation_<timestamp>/
│   └── ablation_results.csv
└── baselines/seed_0/
    └── comparison_results.csv
```

## 5. Backbone: CNN vs ResNet-18

Default backbone is **CNN** (`--backbone cnn`). This is what the paper uses for CIFAR-10.

ResNet-18 is available as an optional experiment:

```bash
python run_pipeline.py all --dataset cifar10 --device cuda:0 --backbone resnet18
# → runs/cifar10_resnet18/
```

**Note:** In our experiments, ResNet-18 underperformed the lightweight CNN on CIFAR-10 (~17% vs ~37% ACC). Use CNN for paper numbers unless exploring backbone ablations.

SVHN (optional RGB dataset) also supports both backbones:

```bash
python run_pipeline.py all --dataset svhn --device cuda:0 --backbone resnet18
```

## 6. CIFAR-10 / RGB datasets

- Use `--skip-viz` for color datasets (auto-enabled in `run_pipeline.py all` for CIFAR-10/SVHN).
- Metrics (ACC, FID, LPIPS, SSIM, PSNR) are computed regardless of visualization flags.

Quick single-seed training:

```bash
python train_cs_wae.py --dataset cifar10 --backbone cnn --seed 0 --device cuda:0 \
  --output-dir runs/cifar10/seed_0 --skip-viz
```

## 7. Manual per-script workflow

If you prefer calling scripts directly:

```bash
# Train one seed
python train_cs_wae.py --dataset mnist --seed 0 --device cuda:0 \
  --output-dir runs/mnist/seed_0

# Ablation (all 5 variants)
python run_ablation_study.py --dataset mnist --seed 0 --device cuda:0

# Baselines
python compare_baselines.py --dataset mnist --seed 0 --device cuda:0 \
  --epochs 50 --output-dir runs/mnist/baselines/seed_0

# Aggregate (only seed_0, seed_1, seed_2 — ignores debug/test dirs)
python aggregate_results.py --runs-dir runs/mnist
```

## 8. Paper figures

After experiments complete:

```bash
python scripts/plot_paper_figures.py
```

Outputs go to `paper_outputs/tables/` and `paper_outputs/figures/`. Update `RUN_PATHS` in the script if your ablation timestamp folders differ.

Build the LaTeX paper:

```bash
cd paper && latexmk -pdf paper.tex
```

## 9. Optional datasets (KMNIST, EMNIST)

Same pipeline, different `--dataset`:

```bash
python run_pipeline.py all --dataset kmnist --device cuda:0
python run_pipeline.py all --dataset emnist_letters --device cuda:0 --skip-advanced-viz
```

EMNIST uses 26 classes; advanced visualization is skipped automatically for heavy datasets.

## 10. Results reference

See [EXPERIMENTS_REPORT.md](EXPERIMENTS_REPORT.md) for full numbers, ablation analysis, and file paths for completed runs.
