# CS-WAE: Class-Structured Spherical Cauchy Wasserstein Auto-Encoders

Official implementation of **CS-WAE** for label-guided generative clustering. The model learns a **hyperspherical latent space** with class-conditional Spherical Cauchy priors and aligns encoded representations via **supervised MMD**, jointly optimizing reconstruction quality and cluster structure.

**Paper datasets:** MNIST, Fashion-MNIST, CIFAR-10 (CNN encoder by default).

| Dataset | CS-WAE ACC (mean ± std, 3 seeds) | Best baseline ACC |
|---------|----------------------------------|-------------------|
| MNIST | **92.9 ± 4.0%** | 89.0% (WAE-MMD) |
| Fashion-MNIST | **79.0 ± 8.0%** | 76.2% (WAE-MMD) |
| CIFAR-10 | **37.2 ± 3.6%** | 27.3% (WAE-MMD) |

Full tables, ablations, and analysis: [docs/EXPERIMENTS_REPORT.md](docs/EXPERIMENTS_REPORT.md).

---

## Installation

```bash
git clone https://github.com/DngBack/CS-WAE.git
cd CS-WAE

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch with CUDA (recommended for full runs).

Optional smoke test:

```bash
python tests/test_ablation.py
```

---

## Reproduce main results

`run_pipeline.py` runs the full experiment suite per dataset: **3-seed training → ablation (5 variants) → baselines (VAE, WAE-MMD, VaDE, CS-WAE) → aggregate**.

```bash
# Table 1 — single GPU
python run_pipeline.py all --dataset mnist --device cuda:0
python run_pipeline.py all --dataset fashion_mnist --device cuda:0
python run_pipeline.py all --dataset cifar10 --device cuda:0 --backbone cnn

# Two GPUs (seeds 0 and 1 in parallel)
python run_pipeline.py all --dataset mnist --device cuda:0 --device2 cuda:1
```

Quick smoke test (2 epochs):

```bash
python run_pipeline.py all --dataset mnist --device cuda:0 --epochs 2
```

Outputs are written to `runs/{dataset}/` (see layout below). Logs go to `logs/pipeline_*.log`.

### Run individual phases

```bash
python run_pipeline.py train     --dataset fashion_mnist --seeds 0 1 2 --device cuda:0
python run_pipeline.py ablation  --dataset fashion_mnist --device cuda:0
python run_pipeline.py baselines --dataset fashion_mnist --device cuda:0
python run_pipeline.py aggregate --runs-dir runs/fashion_mnist
```

Equivalent low-level scripts (`train_cs_wae.py`, `run_ablation_study.py`, `compare_baselines.py`, `aggregate_results.py`) are documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## Repository layout

```
CS-WAE/
├── run_pipeline.py              # End-to-end pipeline (recommended entry point)
├── train_cs_wae.py              # CS-WAE training
├── run_ablation_study.py        # Ablation study (5 variants)
├── compare_baselines.py         # VAE, WAE-MMD, VaDE, CS-WAE
├── aggregate_results.py         # Multi-seed mean ± std
├── src/                         # Models, losses, datasets, trainers
├── scripts/plot_paper_figures.py # Export paper tables and figures
├── tests/test_ablation.py       # Optional forward-pass sanity check
├── docs/
│   ├── REPRODUCIBILITY.md       # Full reproduction guide
│   └── EXPERIMENTS_REPORT.md    # Results and analysis
├── paper/                       # LaTeX source
└── paper_outputs/               # Generated tables and figures
```

After a full run:

```
runs/{dataset}/
├── seed_0/, seed_1/, seed_2/    # metrics.json per seed
├── aggregated_metrics.json      # mean ± std summary
├── ablation_<timestamp>/        # ablation_results.csv
└── baselines/seed_0/            # comparison_results.csv
```

`runs/`, `logs/`, and `data/` are not tracked in git.

---

## Backbone

| Backbone | Flag | Output dir | Notes |
|----------|------|------------|-------|
| Lightweight CNN | `--backbone cnn` (default) | `runs/{dataset}/` | Used for all paper results |
| ResNet-18 | `--backbone resnet18` | `runs/{dataset}_resnet18/` | Optional; underperformed CNN on CIFAR-10 (~17% vs ~37% ACC) |

```bash
# Optional ResNet-18 experiment on CIFAR-10
python run_pipeline.py all --dataset cifar10 --device cuda:0 --backbone resnet18
```

For CIFAR-10 and SVHN, visualization is skipped automatically (`--skip-viz`); all metrics are still computed.

---

## Paper figures

```bash
python scripts/plot_paper_figures.py   # → paper_outputs/
cd paper && latexmk -pdf main.tex
```

See [paper/README.md](paper/README.md) for build details.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Environment, datasets, CLI reference, paper build |
| [docs/EXPERIMENTS_REPORT.md](docs/EXPERIMENTS_REPORT.md) | Full results, ablations, file paths |
| [docs/STAGE0_AUDIT_PROTOCOL.md](docs/STAGE0_AUDIT_PROTOCOL.md) | Canonical latent views, probes, estimators, external evaluators, and result manifests |
| [paper/](paper/) | LaTeX manuscript |

---

## Citation

If you use this code, please cite:

```bibtex
@article{cswae2026,
  title={CS-WAE: Class-Structured Spherical Cauchy Wasserstein Auto-Encoders for Label-Guided Generative Clustering},
  author={Anonymous Authors},
  journal={arXiv preprint},
  year={2026}
}
```

Update author and venue fields when the paper is published.

---

## License

MIT License — see [LICENSE](LICENSE).
