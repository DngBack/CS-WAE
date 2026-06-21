# CS-WAE Main Results — Implementation & Evaluation Report

**Date:** 2026-06-21  
**Dataset:** MNIST  
**Hardware:** 2× NVIDIA A30 (`cuda:0`, `cuda:1`)  
**Environment:** Python 3.10, PyTorch + CUDA, `uv` + `.venv`

---

## 1. Mục tiêu pipeline đã hoàn thành

| ID | Hạng mục | Mô tả | Trạng thái |
|----|----------|-------|------------|
| B1 | Multi-seed main training | CS-WAE × 3 seeds (0, 1, 2), 50 epochs | ✅ |
| A2 | Ablation study | 5 variants × 50 epochs | ✅ |
| A3 | Fair baseline comparison | VAE, WAE-MMD, VaDE, CS-WAE × 50 epochs | ✅ |
| C  | CLI / output chuẩn hóa | `--seed`, `--device`, `--output-dir`, `metrics.json` | ✅ |
| —  | Aggregate multi-seed | `aggregated_metrics.json` mean ± std | ✅ |

**Chưa làm (bước tiếp theo):** Fashion-MNIST, multi-seed baselines, S-VAE, paper figures, theory section.

---

## 2. Kiến trúc & thuật toán (implementation)

### 2.1 Model: `SphericalWAE_Supervised`

**File:** `src/models/cs_wae.py`

```
Input (1×28×28)
    → EncoderCNN → (μ_q, ρ_q) trên S^{d-1}
    → Möbius reparam: z_q ~ Spherical Cauchy(μ_q, ρ_q)
    → DecoderCNN → x_hat
```

| Thành phần | Chi tiết |
|------------|----------|
| Latent space | Hypersphere S^{31} (latent_dim=32) |
| Encoder | 3 conv layers + FC → μ (normalized), s → ρ = σ(s)·(1−ε) |
| Decoder | FC + 3 deconv → Sigmoid output |
| Class priors | `prior_mus` — learnable (10×32), L2-normalized |
| Prior distribution | Spherical Cauchy, ρ_prior = 0.7 |
| Reparameterization | Möbius transform trên sphere (`mobius_reparam`) |

### 2.2 Loss function

**File:** `src/utils/loss.py` — `calculate_cs_wae_loss()`

```
L_total = L_recon + λ_sup(t)·L_sup_MMD + λ_unsup(t)·L_unsup_MMD
```

| Term | Công thức | Weight |
|------|-----------|--------|
| **L_recon** | 0.3·BCE + 0.7·LPIPS(VGG) | cố định |
| **L_sup_MMD** | MMD(z_q[c], z_p[c]) per class c | λ_sup: 0→20 (anneal 20 ep) |
| **L_unsup_MMD** | MMD(z_q, z_p random class) | λ_unsup: 0→50 (anneal 20 ep) |

- Supervised MMD: match latent của từng class với prior của class đó
- Unsupervised MMD: match toàn bộ latent với random class prior (WAE-style)
- MMD kernel: implemented in `src/utils/utils.py` (`mmd_loss`)

### 2.3 Training

**File:** `src/trainers/trainer.py` — `CSWAETrainer`

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam, lr=1e-3 |
| Scheduler | StepLR(step=30, γ=0.5) |
| Batch size | 128 |
| Epochs | 50 |
| Gradient clip | max_norm=1.0 |
| Annealing | λ_sup, λ_unsup linear ramp over 20 epochs |

### 2.4 Evaluation

**File:** `src/metrics/evaluation.py` — `ModelEvaluator.comprehensive_evaluation()`

| Metric | Method |
|--------|--------|
| **ACC** | KMeans(k=10) on latent → Hungarian matching vs labels |
| **NMI** | Normalized mutual information |
| **ARI** | Adjusted Rand index |
| **FID** | pytorch-fid, 10,000 generated vs 10,000 real |
| **SSIM / PSNR** | torchmetrics, test set reconstruction |
| **LPIPS** | VGG-based perceptual loss |

---

## 3. Cách chạy (reproducibility)

### 3.1 Entry point chính

```bash
python train_cs_wae.py \
  --seed <SEED> \
  --device cuda:0 \
  --output-dir runs/mnist/seed_<SEED> \
  [--skip-advanced-viz]
```

### 3.2 Pipeline đã chạy

1. **Phase 1 — Multi-seed** (`run_mnist_pipeline_parallel.sh` + resume)
   - seed 0: full visualization (UMAP, slerp, advanced tests)
   - seed 1, 2: `--skip-advanced-viz`
   - Song song 2 GPU cho seed 0 & 1

2. **Phase 2 — Ablation** (`run_ablation_study.py`)
   - 5 variants, chia 2 GPU, `--skip-aggregate` → `--summarize-only`
   - Output: `runs/mnist/ablation_20260621_061506/`

3. **Phase 3 — Baselines** (`compare_baselines.py`)
   - VAE, WAE-MMD, VaDE, CS-WAE — 50 epochs, seed 0
   - Output: `runs/mnist/baselines/seed_0/`

4. **Phase 4 — Aggregate** (`aggregate_results.py`)
   - Output: `runs/mnist/aggregated_metrics.json`

### 3.3 Reproducibility utilities

| File | Chức năng |
|------|-----------|
| `src/utils/seed.py` | `set_seed()` — Python/NumPy/PyTorch/CuDNN |
| `src/utils/device.py` | `set_device()` — CLI device override |
| `src/utils/run_io.py` | `save_run_metadata()`, `save_metrics()` |
| `src/datasets/loaders.py` | Dataset registry + seeded DataLoader |

Mỗi run lưu: `run_config.json`, `metrics.json`, `training_history.json`, `cs_wae_model.pth`.

---

## 4. Cấu trúc output

```
runs/mnist/
├── seed_0/                    # full viz
│   ├── run_config.json
│   ├── metrics.json
│   ├── training_history.json
│   ├── cs_wae_model.pth
│   └── *.png, fid_images_*/
├── seed_1/                    # skip advanced viz
├── seed_2/
├── aggregated_metrics.json    # mean ± std (3 seeds)
├── per_seed_metrics.csv
├── ablation_20260621_061506/
│   ├── ablation_results.csv
│   ├── baseline/, no_sup_mmd/, euclidean/, vmf_prior/, minimal/
│   └── run_config.json
└── baselines/seed_0/
    ├── comparison_results.csv
    └── VAE/, WAE-MMD/, VaDE/, CS-WAE/
```

> **Note:** `runs/`, `logs/`, `**/fid_images*/`, plots — đã thêm vào `.gitignore` (~355k PNG).

---

## 5. Main results — Multi-seed CS-WAE

### 5.1 Per-seed

| Seed | ACC ↑ | NMI ↑ | ARI ↑ | FID ↓ | SSIM ↑ | LPIPS ↓ | PSNR ↑ |
|------|-------|-------|-------|-------|--------|---------|--------|
| 0 | 87.35% | 88.63% | 84.11% | 26.95 | 0.813 | 0.060 | 16.38 |
| 1 | 95.28% | 88.23% | 89.88% | 15.48 | 0.861 | 0.049 | 17.77 |
| 2 | **96.10%** | **90.08%** | **91.62%** | **15.82** | **0.878** | **0.044** | **18.36** |

### 5.2 Aggregate (report chính cho paper)

| Metric | Mean ± Std |
|--------|------------|
| **ACC** | **92.91 ± 3.95%** |
| **NMI** | **88.98 ± 0.79%** |
| **ARI** | **88.54 ± 3.21%** |
| **FID** | **19.42 ± 5.33** |
| **SSIM** | **0.851 ± 0.028** |
| **LPIPS** | **0.051 ± 0.007** |
| **PSNR** | **17.50 ± 0.83 dB** |

### 5.3 Training convergence (epoch 50)

| Seed | Total loss | Recon | Sup MMD | Unsup MMD |
|------|-----------|-------|---------|-----------|
| 0 | ~1.84 | ~0.081 | ~0.076 | ~0.005 |
| 2 | ~1.79 | ~0.059 | ~0.074 | ~0.005 |

Total loss ~1.8 là **bình thường** (weighted MMD terms). Recon ~0.06–0.08 ổn định.

### 5.4 Nhận xét multi-seed

- **Seed 0 outlier:** ACC thấp hơn ~9%, FID cao hơn ~70% vs seed 2 — có thể bad initialization
- **NMI ổn định** (~0.88–0.90) → latent structure consistent
- **ACC/FID biến thiên nhiều** → cần report mean±std, không chỉ best seed

---

## 6. Baseline comparison (fair, seed 0, 50 epochs)

| Model | ACC ↑ | NMI ↑ | ARI ↑ | FID ↓ | SSIM ↑ |
|-------|-------|-------|-------|-------|--------|
| **CS-WAE** | **86.94%** | **85.03%** | **82.09%** | **15.16** | 0.891 |
| VAE | 60.85% | 55.30% | 45.63% | 18.84 | 0.909 |
| VaDE | 57.79% | 53.37% | 42.37% | 17.45 | 0.909 |
| WAE-MMD | 57.46% | 50.62% | 38.72% | 138.30 | 0.942 |

**Kết luận:**
- CS-WAE **+26–29% ACC** so với mọi baseline
- **FID tốt nhất** (15.16)
- WAE-MMD: SSIM/LPIPS recon cao nhưng FID sụp (138) → pixel recon ≠ sample quality

---

## 7. Ablation study (5 variants, seed 0, 50 epochs)

| Variant | ACC | NMI | FID | SSIM | Insight |
|---------|-----|-----|-----|------|---------|
| **Full CS-WAE** | 84.09% | 88.69% | 24.59 | 0.828 | Reference |
| w/o Supervised MMD | 34.07% | 29.19% | 325.9 | 0.238 | **Collapse** — sup MMD essential |
| Euclidean | 11.35% | 0.00% | 258.9 | 0.251 | **Collapse** — spherical essential |
| vMF prior | 24.54% | 14.74% | 63.0 | **0.966** | Recon tốt, clustering fail |
| Minimal | 11.35% | 0.00% | 240.1 | 0.246 | **Collapse** — cần full design |

**3 claims mạnh cho ablation section:**
1. Supervised MMD không chỉ giúp clustering — **bỏ → model sụp toàn bộ**
2. Spherical manifold **bắt buộc** — Euclidean không hoạt động
3. Cauchy prior **cân bằng** clustering + recon; vMF đổi lấy recon (SSIM 0.97) bằng clustering

---

## 8. So sánh các nguồn kết quả CS-WAE

| Nguồn | ACC | FID | Ghi chú |
|-------|-----|-----|---------|
| Multi-seed mean | 92.9% | 19.4 | 3 seeds, main trainer |
| Multi-seed best (seed 2) | 96.1% | 15.8 | Best single run |
| Multi-seed worst (seed 0) | 87.4% | 26.9 | Outlier |
| Baseline table (seed 0) | 86.9% | 15.2 | Fair comparison protocol |
| Ablation baseline | 84.1% | 24.6 | Ablation trainer (`loss_ablation.py`) |
| Run cũ (`results_cs_wae/`) | 95.8% | 19.9 | Pre-pipeline, 1 seed |

**Discrepancy cần lưu ý:** Ablation baseline (84%) thấp hơn main seed 1/2 (95–96%) do code path khác (`AblationTrainer` + `loss_ablation.py` vs `CSWAETrainer` + `loss.py`). Nên align hoặc note trong paper.

---

## 9. Code changes so với pipeline ban đầu

| File | Thay đổi |
|------|----------|
| `train_cs_wae.py` | CLI: seed, device, output-dir, skip-viz |
| `compare_baselines.py` | 50 epochs, VaDE enabled, seed/device CLI |
| `run_ablation_study.py` | seed, device, parallel workers, summarize-only |
| `aggregate_results.py` | Multi-seed mean ± std |
| `src/utils/seed.py` | Reproducibility |
| `src/utils/device.py` | Device override |
| `src/utils/run_io.py` | JSON I/O |
| `src/utils/loss_ablation.py` | Fix in-place loss bug (`+=` → new tensor) |
| `src/datasets/mnist.py` | Seeded DataLoader |
| `run_mnist_pipeline_parallel.sh` | 2-GPU parallel orchestration |
| `.gitignore` | Ignore runs/, logs/, fid_images/, plots |

---

## 10. Known issues & limitations

| Issue | Impact | Action |
|-------|--------|--------|
| Seed 0 outlier | ACC ±4% variance | Investigate UMAP; thêm seeds hoặc report mean±std |
| Ablation vs main code path | Số liệu ablation baseline thấp hơn | Unify trainer/loss hoặc document |
| Baselines single-seed | Không symmetric với multi-seed main | Chạy baselines 3 seeds (optional) |
| Chỉ MNIST | Không generalize | Fashion-MNIST next |
| S-VAE không chạy | Thiếu `hyperspherical_vae` package | Optional install |
| ~355k PNG trong runs/ | Git bloated | Đã gitignore; metrics nằm trong runs/ |

---

## 11. Số liệu copy-ready cho paper

### Table 1 — Main results (MNIST, 50 epochs, 3 seeds)
> CS-WAE: ACC **92.9±4.0%**, NMI **89.0±0.8%**, ARI **88.5±3.2%**, FID **19.4±5.3**

### Table 2 — Baseline comparison (seed 0)
> CS-WAE vs VAE/VaDE/WAE-MMD — see Section 6

### Table 3 — Ablation
> See Section 7 / `ablation_results.csv`

---

## 12. Bước tiếp theo (recommended priority)

1. **Fashion-MNIST** — generalization (P0 cho conference)
2. **Multi-seed baselines** — fair comparison symmetric
3. **Align ablation code path** — consistent numbers
4. **Export paper tables** — copy metrics CSV ra `docs/` hoặc `paper_results/`
5. **Theory section** — spherical Cauchy + dual MMD motivation
6. **Figure generation** — UMAP, recon grid, ablation bar charts từ saved metrics

---

*Generated from pipeline outputs in `runs/mnist/` — 2026-06-21*
