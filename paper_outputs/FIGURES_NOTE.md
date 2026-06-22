# Ghi chú — Paper outputs đã generate

**Ngày:** 2026-06-22  
**Script:** `scripts/plot_paper_figures.py`  
**Thư mục:** `paper_outputs/`

Mỗi figure có **`.png`** (300 DPI, xem nhanh) và **`.pdf`** (nên dùng trong LaTeX).

---

## Bảng (`tables/`)

| File | Dùng cho section | Nguồn dữ liệu |
|------|------------------|---------------|
| `table01_main_results` | Main results | `runs/*/aggregated_metrics.json` (3 seeds) |
| `table02_baselines` | Baseline comparison | `runs/*/baselines/seed_0/comparison_results.csv` |
| `table03_ablation` | Ablation | `runs/*/ablation_*/ablation_results.csv` |

LaTeX cần `\usepackage{booktabs}`.

---

## Hình (`figures/`)

### fig02 — `baseline_comparison`
So sánh **4 baselines** (CS-WAE, VaDE, VAE, WAE-MMD) trên MNIST và Fashion-MNIST.  
Hai panel: **ACC** (trái) và **FID** (phải). Seed 0, 50 epochs.

→ Đặt ở **§Experiments / Baseline comparison**.

---

### fig03 — `ablation_acc`
Bar chart **ACC** của 5 ablation variants × 2 datasets. Đường đứt nét = chance level (10%).  
Variants: Full, w/o Sup MMD, Euclidean, vMF, Minimal.

→ Đặt ở **§Ablation study**.

---

### fig04 — `tsne_clustering_{mnist|fashion_mnist}`
4 hàng t-SNE (1500 điểm test, seed 0):
1. Ground-truth labels (trên latent CS-WAE)
2. CS-WAE — xanh = cluster đúng, đỏ = sai (KMeans + Hungarian)
3. VaDE — cùng cách tô màu
4. VAE — cùng cách tô màu

Checkpoint: `runs/<dataset>/baselines/seed_0/`.

→ Đặt ở **§Qualitative analysis / Latent space** (kiểu VaDE Fig. 5).

---

### fig05 — `reconstruction_grid_{mnist|fashion_mnist}`
Lưới **4 methods × 2 hàng** (original / reconstruction), 10 ảnh test đầu tiên, cùng input cho mọi method.

→ Đặt ở **§Baseline comparison** hoặc appendix (kiểu WAE Fig. 2).

---

### fig06 — `generation_grid_{mnist|fashion_mnist}`
Lưới **10 classes × 4 methods**, 1 sample/class:
- CS-WAE: sample từ class prior (Spherical Cauchy)
- VaDE: sample từ `mu_c[class]`
- VAE / WAE-MMD: `z ~ N(0,I)`

→ Đặt ở **§Generation quality** (kiểu VaDE Fig. 4).

---

### fig11 — `ablation_recon_{mnist|fashion_mnist}`
Hàng reconstructions của 5 ablation variants (cắt từ `reconstructions.png` mỗi variant).  
MNIST ablation: `ablation_20260621_061506`. Fashion: `ablation_20260621_192630`.

→ Đặt cạnh Table 3 / ablation section — minh họa **collapse** khi bỏ component.

---

### fig15 — `fid_samples_{mnist|fashion_mnist}`
8 ảnh ngẫu nhiên/method từ thư mục FID `fid_images_*/generated/` (seed cố định 42).  
Hữu ích cho narrative **WAE-MMD**: SSIM cao nhưng sample kém.

→ Appendix hoặc discussion.

---

## Hình chưa gen (có sẵn ở `runs/`)

| Nội dung | Path gợi ý |
|----------|-------------|
| UMAP latent space | `runs/<dataset>/seed_0/latent_space_umap.png` |
| Slerp interpolation | `runs/<dataset>/seed_0/slerp_interpolation.png` |
| Spherical interpolation grid | `runs/<dataset>/seed_0/spherical_interpolation_grid.png` |
| Latent traversal | `runs/<dataset>/seed_0/latent_traversal_*.png` |
| Architecture diagram | Chưa có — vẽ tay (TikZ / draw.io) |

Qualitative đẹp nhất: có thể re-run advanced viz trên **seed_2** (best ACC) rồi copy vào paper.

---

## Chạy lại

```bash
source .venv/bin/activate
python scripts/plot_paper_figures.py --device cuda:0

# Chỉ bảng + bar chart (không load model, ~vài giây):
python scripts/plot_paper_figures.py --skip-model-plots
```

Sau khi cập nhật `runs/`, chạy lại script để refresh toàn bộ `paper_outputs/`.
