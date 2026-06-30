# TODO — Hoàn thiện paper (main_v3.tex)

> Cập nhật: 2026-06-30

---

## Ưu tiên cao — Cần có trước submission

### Thí nghiệm cần chạy

- [ ] **Table 3 (cross-model leakage)** — train VAE, WAE-MMD, β-TCVAE, FactorVAE trên MNIST
  ```bash
  python scripts/run_cross_model_diagnostics.py --device cuda --epochs 100
  ```

- [ ] **LP accuracy cho F-CS-WAE MNIST** — điền vào `\todo{run LP}` trong body text §4.2
  ```bash
  python scripts/compute_leakage_diagnostics.py \
    --checkpoint runs_f/mnist/seed_0/best_model.pth \
    --dataset mnist --device cpu
  ```

- [ ] **Table 6 (ablations CIFAR-10)** — so sánh có/không per-class style MMD
  ```bash
  # Với per-class style MMD (đề xuất, delta=1.0)
  python train_f_cs_wae.py --dataset cifar10 --seed 0 --delta-final 1.0

  # Không per-class style MMD (baseline)
  python train_f_cs_wae.py --dataset cifar10 --seed 0 --delta-final 0.0
  ```

- [ ] **Fashion-MNIST 3 seeds**
  ```bash
  python train_f_cs_wae.py --dataset fashion_mnist --seed 0 --device cuda
  python train_f_cs_wae.py --dataset fashion_mnist --seed 1 --device cuda
  python train_f_cs_wae.py --dataset fashion_mnist --seed 2 --device cuda
  ```

- [ ] **MNIST 3 seeds** (để report mean±std)
  ```bash
  python train_f_cs_wae.py --dataset mnist --seed 1 --device cuda
  python train_f_cs_wae.py --dataset mnist --seed 2 --device cuda
  ```

### Hình ảnh cần bổ sung

- [ ] `figures/cross_model_diagnostic.png` — bar chart Global MMD vs Δ_inter (cho §4 cross-model)
- [ ] t-SNE plots của `z_s` tô màu theo class label (cho từng model trong Table 3)

---

## Ưu tiên trung bình

- [ ] Chạy baselines CIFAR-10 seed 1 và seed 2 → cập nhật Table 2
- [ ] Ablation variants còn lại (Table 6): no-z_s, no class MMD, no aux classifier, Gaussian/vMF prior
- [ ] Tính HSIC(z_s, y) cho F-CS-WAE và điền vào Table 3

---

## Trước khi submit

- [ ] Thay preamble bằng AAAI-27 author kit chính thức
- [ ] Xóa appendix "Submission Checklist" (`\section{Submission Checklist}`)
- [ ] Kiểm tra page limit AAAI-27
- [ ] Compile LaTeX và kiểm tra references

---

## Số liệu đã có (KHÔNG cần chạy lại)

| Bảng | Trạng thái |
|------|-----------|
| Table 1 — CIFAR-10 3-seed F-CS-WAE | ✅ Hoàn chỉnh |
| Table 2 — CIFAR-10 baselines (seed 0) | ✅ Hoàn chỉnh |
| Table 4 — MNIST seed-0 comparison | ✅ Hoàn chỉnh |
| Table 5 — MNIST sampling strategies | ✅ Hoàn chỉnh |
| Fig. cifar10_main_metrics, cifar10_class_prior_grid | ✅ Có sẵn |
| Fig. mnist_sampling_*.png, mnist_style_prior_diagnostic | ✅ Có sẵn |
