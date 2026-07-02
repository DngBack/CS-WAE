# TODO — Hoàn thiện paper (main_v3.tex)

> Cập nhật: 2026-07-02

---

## ⚠️ Trạng thái hiện tại: mọi `\todo{}` trong body đã được xử lý

Không còn `\todo{}` nào trong `main_v3.tex` (đã kiểm tra bằng
`grep -n "\\todo{" main_v3.tex` → rỗng). Tuy nhiên, một phần đáng kể số liệu
hiện là **giả định (`\assumed{...}`, hiển thị màu xanh + dấu †)**, chưa phải kết
quả đo thật. Toàn bộ danh sách, lý do, và cách thay bằng số thật nằm ở
**`docs/ASSUMED_RESULTS.md`** — đọc file đó trước khi động vào bất kỳ con số nào
trong paper.

**Việc còn lại = chạy các thí nghiệm thật để thay từng `\assumed{}` bằng số đo
thật**, theo đúng danh sách trong `ASSUMED_RESULTS.md`.

---

## Phát hiện quan trọng (bối cảnh, đã xử lý)

- `delta_final` (per-class style MMD, Eq. 9) chỉ được thêm vào code ngày
  2026-06-30 (commit `f5c44f6`). Mọi checkpoint train trước ngày đó là baseline
  "without per-class style MMD" dù không ghi rõ trong config.
- Đã train xong **cả 2 job per-class-style-MMD** (δ=1.0):
  `runs_f/mnist/seed_0_pcmmd/` (xong 2026-07-01 18:33) và
  `runs_f/cifar10/seed_0_pcmmd/` (xong 2026-07-01 15:30). Kết quả:
  - MNIST: ACC 99.15%→85.92% (giảm mạnh), Δ_inter 6.27→1.46, LP 100%→45.6%,
    gen self-acc (naive Gaussian) 0.15→0.70.
  - CIFAR-10: ACC 81.29%→80.00% (gần như không đổi), Δ_inter 3.86→1.28,
    LP 86.9%→35.6%.
  - **Per-class style MMD giảm mạnh leakage nhưng KHÔNG loại bỏ hoàn toàn**, và
    có đánh đổi ACC đáng kể trên MNIST — đã cập nhật trung thực vào abstract,
    intro, Table 3/6, §6.6 Discussion, và Conclusion.
- Đã sửa 1 bug trong `scripts/compute_leakage_diagnostics.py`
  (`compute_gen_self_accuracy` đưa nhầm ảnh pixel thô vào aux classifier thay vì
  encode qua `model.encode_to_distribution()` lấy `μ_c` trước — theo đúng cách
  Table 5 tính self-acc trong `analyze_fcswae_sampling_strategies.py`).
- CIFAR-10 `seed_0` gốc (cho Table 1) từng bị ghi đè bởi 1 lần chạy
  `--delta-final 0.0`. Số liệu paper vẫn đúng (backup tại
  `paper_outputs/f_cs_wae_cifar10/tables/table_f01_f_cs_wae_cifar10_per_seed.csv`)
  nhưng checkpoint `.pth` gốc đã mất.
- Fashion-MNIST `seed_0`: `metrics.json` đã tính xong (ACC 0.9341 ...). Thư mục
  `fid_images_F-CS-WAE/real` bị hỏng (1 file 0-byte) từ lần train gốc — đã xoá
  và tính lại FID sạch.
- Figure `cross_model_diagnostic.png` đã tạo (số liệu **thật** 100%, không giả
  định) — bar chart Global MMD vs Δ_inter cho 6 dòng của Table 3.

---

## Việc cần làm (theo thứ tự ưu tiên) — xem chi tiết trong `ASSUMED_RESULTS.md`

### Ưu tiên cao

- [ ] **MNIST 3 seeds thật** (thay `\assumed{}` ở Table 4)
  ```bash
  python train_f_cs_wae.py --dataset mnist --seed 1 --device cuda
  python train_f_cs_wae.py --dataset mnist --seed 2 --device cuda
  ```
- [ ] **Fashion-MNIST 3 seeds thật** (thay `\assumed{}` ở Table 5)
  ```bash
  python train_f_cs_wae.py --dataset fashion_mnist --seed 1 --device cuda
  python train_f_cs_wae.py --dataset fashion_mnist --seed 2 --device cuda
  ```
- [ ] **CIFAR-10 baselines seed 1, 2 thật** (ResNetAE/VAE/WAE-MMD/VaDE, thay
  `\assumed{}` ở Table 3/cifar-baselines) — cần xác nhận tên script train
  baseline thật trong repo trước khi chạy.
- [ ] **5 biến thể ablation còn thiếu** (thay `\assumed{}` ở Table 7): no-$z_s$,
  no class MMD (α=0), no aux classifier (η=0), Gaussian prior, vMF prior.
  Cần bổ sung CLI flags vào `train_f_cs_wae.py` trước (hiện chỉ có
  `--delta-final`), rồi train + đo bằng `compute_leakage_diagnostics.py`.
- [ ] t-SNE plots của `z_s` tô màu theo class label (mỗi model trong Table 3) —
  hình ảnh, không thể giả định, cần chạy thật.

### Ưu tiên trung bình

- [ ] Bổ sung 6 bib entry còn thiếu: `higgins2017beta`, `kingma2014semi`,
  `sohn2015learning`, `karras2019style`, `guo2017improved`, `goodfellow2016deep`
  (hiện natbib báo "undefined citation" cho các key này).

---

## Trước khi submit

- [ ] **Xoá hết `\assumed{}` trong `main_v3.tex`** sau khi có số liệu thật —
  không được nộp bài còn số liệu giả định.
- [ ] Xoá macro `\assumed` và file `docs/ASSUMED_RESULTS.md` khi không còn dùng.
- [ ] Thay preamble bằng AAAI-27 author kit chính thức.
- [ ] Xóa appendix "Submission Checklist" (`\section{Submission Checklist}`).
- [ ] Kiểm tra page limit AAAI-27 (hiện tại: 19 trang bao gồm appendix cần xoá).
- [ ] Compile LaTeX và kiểm tra references (đã compile sạch, chỉ còn 6 citation
  thiếu ở trên).

---

## Số liệu đã có, THẬT 100% (không cần chạy lại)

| Bảng/Hình | Trạng thái |
|------|-----------|
| Table 2 — CIFAR-10 3-seed F-CS-WAE | ✅ Thật |
| Table 3 — Cross-model leakage (5 model + F-CS-WAE+pcMMD) | ✅ Thật |
| Table 4 — MNIST (seed 0 + Δ_inter/LP) | ✅ Thật (3-seed mean là giả định) |
| Table 5 — Fashion-MNIST (seed 0) | ✅ Thật (3-seed mean là giả định) |
| Table 6 — MNIST sampling strategies | ✅ Thật |
| Table 7 — Ablation (2 dòng đầu: full + per-class MMD) | ✅ Thật (5 dòng còn lại giả định) |
| Fig. `cross_model_diagnostic.png` | ✅ Thật |
| Fig. cifar10_main_metrics, cifar10_class_prior_grid | ✅ Có sẵn |
| Fig. mnist_sampling_*.png, mnist_style_prior_diagnostic | ✅ Có sẵn |
