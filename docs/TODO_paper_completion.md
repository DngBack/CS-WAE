# TODO — Hoàn thiện paper (main_v3.tex)

> Cập nhật: 2026-07-01

---

## ⚠️ Phát hiện quan trọng (đọc trước khi chạy thêm thí nghiệm)

- **`delta_final` (per-class style MMD, Eq. 9) chỉ được thêm vào code ngày 2026-06-30**
  (commit `f5c44f6`). Mọi checkpoint train trước ngày đó — MNIST `seed_0`, CIFAR-10
  `seed_1`/`seed_2`, Fashion-MNIST `seed_0`, và bản CIFAR-10 `seed_0` gốc — đều là
  **baseline "without per-class style MMD"**, dù `run_config.json` không ghi rõ.
  → Với MNIST, điều này khớp với narrative của §4.2 (baseline leakage, self-acc 15%).
  → Với CIFAR-10 Table 1 (ACC 80.87%), con số này **cũng là baseline chưa fix**,
    chưa phải "F-CS-WAE + per-class style MMD" như abstract ngụ ý.
  → **Đang chạy** (từ 2026-07-01 09:42, ~4-5h/job): MNIST `seed_0_pcmmd` (cuda:0) và
    CIFAR-10 `seed_0_pcmmd` (cuda:1), cả hai với `--delta-final 1.0`, output dir riêng
    để không đè checkpoint baseline hiện có. Log: `mnist_pcmmd.log` / `cifar10_pcmmd.log`
    trong scratchpad phiên làm việc.
- **CIFAR-10 `seed_0` gốc (dùng cho Table 1) đã bị ghi đè** bởi 1 lần chạy
  `--delta-final 0.0` trước khi phát hiện vấn đề trên. Số liệu chính xác (ACC=0.8043...)
  đã được backup tại `paper_outputs/f_cs_wae_cifar10/tables/table_f01_f_cs_wae_cifar10_per_seed.csv`
  nên Table 1 vẫn đúng, nhưng **checkpoint .pth gốc đã mất** — nếu cần checkpoint đó
  (vd để vẽ figure khác) sẽ phải train lại.
- ✅ Đã merge số liệu thật cho F-CS-WAE vào `runs_diag/cross_model/cross_model_diagnostics.json`,
  `table3_rows.tex`, và `main_v3.tex` (Global MMD 0.0013, Δ_inter 6.27, LP 100.0%).
  Dòng "F-CS-WAE + per-class MMD" vẫn `\todo{}` chờ 2 job trên chạy xong.
- ✅ Fashion-MNIST `seed_0`: đã tính xong `metrics.json` (ACC 0.9341, NMI 0.8672,
  ARI 0.8633, SSIM 0.9190, PSNR 23.00, LPIPS 0.0466, FID 72.37). Lưu ý: thư mục
  `fid_images_F-CS-WAE/real` bị hỏng (1 file 0-byte, ghi dở) từ lần train gốc — đã xoá
  và tính lại FID sạch.

---

## Ưu tiên cao — Cần có trước submission

### Thí nghiệm cần chạy

- [x] **Table 3 (cross-model leakage)** — VAE, WAE-MMD, β-TCVAE, FactorVAE đã chạy xong
      (`runs_diag/cross_model/`).
- [ ] **F-CS-WAE + per-class style MMD (δ=1.0)** — MNIST và CIFAR-10 seed 0 đang chạy
      (xem ghi chú trên). Sau khi xong: chạy `compute_leakage_diagnostics.py` trên
      checkpoint mới để lấy Δ_inter/LP/gen-self-acc cho dòng cuối Table 3 và đầu Table 6.
- [x] **LP accuracy cho F-CS-WAE MNIST** — đã điền (100.0%), xem `runs_diag/fcswae_mnist_baseline/metrics.json`.

- [ ] **Table 6 (ablations CIFAR-10)** — so sánh có/không per-class style MMD
  ```bash
  # Với per-class style MMD (đề xuất, delta=1.0) — ĐANG CHẠY, output riêng seed_0_pcmmd
  python train_f_cs_wae.py --dataset cifar10 --seed 0 --delta-final 1.0 --output-dir runs_f/cifar10/seed_0_pcmmd

  # Không per-class style MMD (baseline) — đã có (runs_f/cifar10/seed_0, ACC 0.8129)
  ```

- [ ] **Fashion-MNIST 3 seeds** — seed 0 xong (metrics ở trên); còn seed 1, 2
  ```bash
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
