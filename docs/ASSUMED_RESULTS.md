# ASSUMED / PLACEHOLDER RESULTS — NOT REAL DATA

> Cập nhật: 2026-07-02
> **CẢNH BÁO: Không được nộp paper khi các số liệu trong file này còn nằm trong
> `main_v3.tex`.** Mọi giá trị liệt kê dưới đây là số **giả định**, sinh ra bằng
> quy tắc/logic được ghi rõ bên dưới — không phải kết quả đo thật. Trong
> `main_v3.tex`, mọi số giả định được bọc bởi macro `\assumed{...}`, hiển thị
> màu **xanh dương** kèm dấu `†`, để không ai nhầm với số liệu thật (số liệu
> thật hiển thị màu đen bình thường).

Mục đích của file này: cho phép xem trước layout/độ dài của paper với đầy đủ
bảng biểu trong khi các thí nghiệm thật chưa chạy xong, đồng thời tra cứu chính
xác cần thay số nào, bằng cách gì, khi có kết quả thật.

---

## 1. Table 3 (`tab:cifar-baselines`) — CIFAR-10 baselines, seed 1 & 2

**Vị trí trong paper:** dòng ResNetAE/VAE/WAE-MMD/VaDE.
**Số thật đã có:** seed 0 cho cả 4 baseline (trong `runs/cifar10/baselines/seed_0/`).
**Số giả định:** seed 1, seed 2, và mean/std tổng hợp từ 3 seed.

**Quy tắc sinh số** (deterministic, xem `/tmp/.../gen_assumed.py` đã dùng để tính):
seed1 = seed0 × (1 + d) (metric càng cao càng tốt) hoặc seed0 × (1 − d/2) (FID, càng thấp càng tốt);
seed2 = seed0 × (1 − d/2) hoặc seed0 × (1 + d) tương ứng.
d = 0.05 cho ResNetAE/VAE, d = 0.06 cho WAE-MMD, d = 0.12 cho VaDE.

**Lý do chọn d lớn hơn cho VaDE:** VaDE (Gaussian-mixture prior, EM-style training)
được biết là dễ mất ổn định / sập vào local optima khác nhau giữa các lần chạy hơn
so với các baseline kiến trúc cố định như ResNetAE/VAE/WAE-MMD — nên giả định
phương sai giữa các seed lớn hơn (~2x).

| Method | ACC (mean) | NMI (mean) | ARI (mean) | FID (mean) |
|---|---|---|---|---|
| ResNetAE | 0.222† | 0.103† | 0.050† | 122.0† |
| VAE | 0.224† | 0.096† | 0.052† | 181.5† |
| WAE-MMD | 0.212† | 0.099† | 0.047† | 237.8† |
| VaDE | 0.151† | 0.044† | 0.015† | 485.4† |

**Cách thay bằng số thật:**
```bash
python train_baselines.py --dataset cifar10 --model resnet_ae --seed 1
python train_baselines.py --dataset cifar10 --model resnet_ae --seed 2
# tương tự cho vae / wae_mmd / vade
```
(kiểm tra tên script/flag thật trong repo trước khi chạy — chưa xác nhận tên
chính xác của script train baseline).

---

## 2. Table 4 (`tab:mnist`) — F-CS-WAE MNIST, seed 1 & 2

**Số thật đã có:** seed 0 (`runs_f/mnist/seed_0/metrics.json`), Δ_inter/LP thật
(`runs_diag/fcswae_mnist_baseline/metrics.json`).
**Số giả định:** seed 1, seed 2, mean±std.

**Quy tắc sinh số:** cùng công thức seed1/seed2 như trên, với d = 0.006 — rất nhỏ,
vì MNIST F-CS-WAE là kết quả gần trần (ACC 99.15%) và mức phương sai giữa seed
quan sát được trên CIFAR-10 (Table 2 thật) rất thấp (std ACC = 0.0052 trên nền
0.8087, tức ~0.6% tương đối) — áp dụng cùng tỷ lệ tương đối cho MNIST.
PSNR dùng d/3 (PSNR biến thiên ít hơn tương đối so với ACC/FID trong dữ liệu thật
đã quan sát).

| | seed1 | seed2 | mean | std |
|---|---|---|---|---|
| ACC | 0.9974† | 0.9885† | 0.9925† | 0.0037† |
| NMI | 0.9817† | 0.9729† | 0.9768† | 0.0037† |
| ARI | 0.9872† | 0.9784† | 0.9823† | 0.0037† |
| SSIM | 0.9892† | 0.9804† | 0.9843† | 0.0037† |
| PSNR | 26.93† | 26.85† | 26.89† | 0.03† |
| LPIPS | 0.0120† | 0.0121† | 0.0120† | 0.0000† |
| FID | 73.77† | 74.43† | 74.06† | 0.28† |

Δ_inter (6.27) và LP (100.0%) trong bảng **là số thật** (đo trên seed 0), không
phải giả định — chỉ chưa có bản đo trên seed 1/2.

**Cách thay bằng số thật:**
```bash
python train_f_cs_wae.py --dataset mnist --seed 1 --device cuda
python train_f_cs_wae.py --dataset mnist --seed 2 --device cuda
```

---

## 3. Table 5 (`tab:fashion-mnist`) — Fashion-MNIST F-CS-WAE, seed 1 & 2

**Số thật đã có:** seed 0 (`runs_f/fashion_mnist/seed_0/metrics.json`, tính lại
ngày 2026-07-01 sau khi sửa lỗi ảnh FID hỏng).
**Số giả định:** seed 1, seed 2, mean±std, dùng d = 0.012 (gấp đôi MNIST vì
Fashion-MNIST có đa dạng nội lớp cao hơn MNIST digit, nên giả định phương sai
seed lớn hơn một chút, nhưng vẫn thấp hơn CIFAR-10's d ngầm định ~0.006 dựa theo
std/mean quan sát thật ở Table 2).

| | seed1 | seed2 | mean | std |
|---|---|---|---|---|
| ACC | 0.9453† | 0.9285† | 0.9360† | 0.0070† |
| NMI | 0.8776† | 0.8620† | 0.8689† | 0.0065† |
| ARI | 0.8737† | 0.8581† | 0.8650† | 0.0065† |
| SSIM | 0.9300† | 0.9135† | 0.9208† | 0.0069† |
| PSNR | 23.09† | 22.95† | 23.02† | 0.06† |
| LPIPS | 0.0463† | 0.0472† | 0.0467† | 0.0003† |
| FID | 71.94† | 73.24† | 72.51† | 0.54† |

**Cách thay bằng số thật:**
```bash
python train_f_cs_wae.py --dataset fashion_mnist --seed 1 --device cuda
python train_f_cs_wae.py --dataset fashion_mnist --seed 2 --device cuda
```

---

## 4. Table 7 (`tab:ablations`) — 5 dòng ablation còn thiếu

**Không có checkpoint thật cho bất kỳ dòng nào trong số này.** Đã kiểm tra
`runs_f/cifar10/f_ablation_20260625_183045/no_z_s/` — thư mục **rỗng hoàn toàn**
(không có `model.pth` hay `metrics.json`), nghĩa là lần chạy ablation "no z_s"
gốc (25/6) có thể đã crash ngay khi khởi động hoặc chưa từng thực sự chạy.
Baseline để so sánh: `\ours{}` full, seed 0, δ=0 → ACC 0.8149, NMI 0.6622,
ARI 0.6438, Δ_inter 4.19 (số thật).

### 4.1 No z_s (semantic-only) — ACC 0.825† / NMI 0.674† / ARI 0.655† / Δ_inter N/A
**Logic:** Bỏ hẳn latent style nghĩa là không còn kênh nào khác để "giấu" biến
thiên nội lớp — mọi tín hiệu phân biệt phải dồn vào z_c. Theo trực giác từ
literature disentanglement (loại bỏ capacity "thừa" đôi khi ép representation
còn lại mã hoá thông tin phân biệt tốt hơn), giả định ACC tăng nhẹ so với full
model. Δ_inter = N/A vì không có z_s để đo.
**Rủi ro của giả định:** có thể sai chiều — cũng có khả năng FID/SSIM giảm mạnh
(ít capacity hơn cho tái tạo) kéo theo pipeline huấn luyện kém ổn định hơn, làm
ACC giảm thay vì tăng. Đây là giả định có độ tin cậy thấp nhất trong bảng.

### 4.2 No class MMD (α=0) — ACC 0.58† / NMI 0.46† / ARI 0.35† / Δ_inter 4.30†
**Logic:** Đây là cơ chế chính kéo μ_c về đúng tâm class-conditional prior
(Eq. 8). Không có nó, chỉ còn aux classifier (η) và aggregated MMD (β) tạo cấu
trúc gián tiếp/yếu hơn nhiều. Giả định ACC giảm mạnh (còn hơn chance 10% nhiều
nhờ η vẫn hoạt động, nhưng thấp hơn nhiều so với 0.81). Δ_inter gần như không đổi
vì α chỉ tác động nhánh semantic (z_c), không phải nhánh style.
**Độ tin cậy:** trung bình — hướng giảm mạnh gần như chắc chắn đúng, nhưng biên
độ cụ thể (0.58) chỉ là ước lượng.

### 4.3 No auxiliary classifier (η=0) — ACC 0.77† / NMI 0.615† / ARI 0.575† / Δ_inter 4.15†
**Logic:** Paper mô tả vai trò của term này là "prevents collapse" (Eq. 12) —
vai trò hỗ trợ, không phải cơ chế chính. Giả định giảm nhẹ-vừa phải so với full.
Δ_inter gần như không đổi (η không tác động style).
**Độ tin cậy:** trung bình.

### 4.4 Gaussian semantic prior — ACC 0.775† / NMI 0.625† / ARI 0.585† / Δ_inter 4.05†
**Logic:** Thay hyperspherical Spherical-Cauchy bằng Gaussian mixture cho z_c.
Bài báo lập luận rằng hình học hyperspherical giúp tách góc/hướng tốt hơn cho
clustering (Related Work, S-VAE/Spherical-Cauchy). Giả định giảm nhẹ ACC do mất
lợi thế hình học này. Δ_inter không đổi nhiều (chỉ ảnh hưởng semantic prior, không
phải style regularization).
**Độ tin cậy:** thấp-trung bình — thuần suy luận từ motivation của related work,
chưa có cơ sở thực nghiệm nào trong repo.

### 4.5 vMF semantic prior — ACC 0.805† / NMI 0.655† / ARI 0.630† / Δ_inter 4.10†
**Logic:** von Mises-Fisher cũng là hyperspherical (đuôi nhẹ hơn Spherical
Cauchy). Giả định gần với full model, thấp hơn một chút — phản ánh đúng luận
điểm của paper rằng đuôi nặng của Spherical Cauchy có ích (nhưng không nhiều).
**Độ tin cậy:** thấp-trung bình.

**Cách thay bằng số thật:** cần thêm code hỗ trợ các flag ablation trong
`train_f_cs_wae.py` (hiện chỉ có `--delta-final`; chưa có `--style-dim 0`,
`--alpha-final`, `--eta-final`, hay lựa chọn Gaussian/vMF prior — xem mục
Submission Checklist trong `main_v3.tex` appendix để biết danh sách flag cần bổ
sung), sau đó chạy từng biến thể và đo lại bằng
`scripts/compute_leakage_diagnostics.py`.

---

## 5. KHÔNG được giả định (đã cố tình bỏ qua)

- **t-SNE plots của z_s theo class label** (mỗi model family) — là hình ảnh, không
  thể "giả định" một cách có ý nghĩa; cần chạy thật.
- **6 bib entry còn thiếu** (`higgins2017beta`, `kingma2014semi`,
  `sohn2015learning`, `karras2019style`, `guo2017improved`, `goodfellow2016deep`)
  — đây là trích dẫn thật cần tra đúng, không phải số liệu để giả định.
- **Table 3 (cross-model) và Figure `cross_model_diagnostic.png`** — toàn bộ là
  số liệu **thật**, không cần giả định gì (đã đo xong).

---

## 6. Tổng kết — việc cần làm để loại bỏ hết giả định

1. Chạy MNIST/Fashion-MNIST/CIFAR-10 baselines seed 1, 2 (Mục 1-3).
2. Thêm CLI flags còn thiếu vào `train_f_cs_wae.py` cho 5 biến thể ablation
   (Mục 4), chạy, rồi đo lại bằng `compute_leakage_diagnostics.py`.
3. Sau khi có số thật, tìm-thay từng `\assumed{...}` tương ứng trong
   `main_v3.tex` bằng giá trị đo được (bỏ luôn macro `\assumed`, để lại số đen
   bình thường).
4. Xoá mục đó khỏi file này khi đã thay xong.
5. Khi **toàn bộ** bảng trong `main_v3.tex` không còn `\assumed{}` nào, xoá định
   nghĩa macro `\assumed` ở đầu file và xoá file này.
