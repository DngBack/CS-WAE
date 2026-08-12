# Báo cáo controlled posterior-entropy experiment

**Trạng thái:** Hoàn tất MNIST sweep và CIFAR-10 confirmation  
**Ngày hoàn tất:** 2026-08-12  
**Model:** F-CS-WAE  
**Protocol:** `stage0-1.2.0`  
**Model seed:** 0  
**Source revision ghi trong audit:** `39720db8518f2d96ed9253b02dc4338a1a220b41`
(dirty worktree)

## 1. Tóm tắt điều hành

Experiment này kiểm tra một criticism trực tiếp đối với kết quả leakage của
F-CS-WAE: label leakage trong style latent có thể chỉ xuất hiện vì style
posterior gần deterministic, tức `sigma_s -> 0`.

Thay vì đổi objective hoặc thêm một entropy penalty có nhiều tác động đồng
thời, experiment thêm một variance floor có kiểm soát vào cùng posterior:

\[
\sigma_{\mathrm{eff}}^2(x)
= \exp(\operatorname{clamp}(\log\sigma_s^2(x),-10,10))
+ \sigma_{\mathrm{floor}}^2.
\]

Kết quả chính:

- Trên MNIST, leakage vẫn rất mạnh ở mức lớn nhất `sigma_floor=0.35`:
  stochastic probe accuracy `95.61%`, HSIC `p=0.00498`, và conditional MMD
  lớn gấp `14.51x` global MMD.
- Trên CIFAR-10, leakage vẫn tồn tại rõ tại `sigma_floor=0.15`: probe accuracy
  `28.78%` so với chance `10%`, classifier lower bound `0.256 nats`, HSIC
  `p=0.00498`, và conditional/global MMD ratio `4.97x`.
- Trên CIFAR-10 tại `sigma_floor=0.35`, linear probe giảm về chance
  (`9.76%`). Đây là boundary condition: HSIC vẫn reject và conditional MMD
  vẫn lớn hơn global MMD, nhưng bằng chứng về label information có thể khai
  thác tuyến tính không còn mạnh.
- Reconstruction không collapse theo competence gate đã chốt: tại mức cao
  nhất, final reconstruction loss bằng `1.32x` baseline trên MNIST và `1.26x`
  baseline trên CIFAR-10, thấp hơn ngưỡng loại `2x`.

Vì vậy, experiment hỗ trợ claim sau:

> **Conditional leakage persists after substantially increasing effective
> posterior entropy, including at non-degenerate operating points where
> reconstruction remains competent.**

Claim này không có nghĩa leakage tồn tại ở mọi entropy level hoặc mọi dataset.
Nó cũng không chứng minh raw variance head đã thôi collapse.

## 2. Câu hỏi nghiên cứu và threat được kiểm tra

Kết quả Stage-0 trước experiment này cho thấy:

1. global style MMD nhỏ;
2. label vẫn được dự đoán tốt từ stochastic style code;
3. conditional HSIC bác bỏ độc lập giữa `z_s` và `y`;
4. phần lớn raw style log-variance nằm tại hoặc dưới numerical clamp floor.

Điểm thứ tư tạo ra một cách giải thích cạnh tranh: style sample gần bằng
posterior mean, nên probe chỉ đang khai thác một representation gần
deterministic. Nếu đúng hoàn toàn, leakage có thể biến mất ngay khi posterior
có stochasticity thực sự.

Experiment đặt câu hỏi hẹp:

> Khi tăng trực tiếp stochasticity của posterior sample nhưng giữ nguyên
> architecture và phần còn lại của training protocol, conditional leakage có
> còn quan sát được trước khi model mất competence hay không?

Đây là một intervention lên effective sampling variance. Nó không phải một
remedy để buộc encoder tự học variance lớn và cũng không thay đổi Theorem 1.

## 3. Thiết kế intervention

### 3.1. Unified posterior sampler

Train và audit sử dụng cùng helper `sample_style_posterior`. Posterior sample
được định nghĩa là:

\[
z_s = \mu_s + \sigma_{\mathrm{eff}} \odot \epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

Điều này tránh trường hợp training dùng variance floor nhưng diagnostic script
vẫn dùng công thức reparameterization cũ.

`sigma_floor=0` khôi phục đúng behavior lịch sử. Floor được cộng trong variance
space; vì vậy khi raw log-variance collapse xuống clamp `-10`, effective
standard deviation vẫn gần mức floor được yêu cầu.

### 3.2. Sweep đã khóa trước

| Dataset | Các mức `sigma_floor` | Vai trò |
|---|---|---|
| MNIST | `0, 0.025, 0.05, 0.15, 0.35` | Full discovery sweep |
| CIFAR-10 | `0, 0.15, 0.35` | Fixed confirmation suite |

Các mức CIFAR-10 không được chọn lại sau khi xem đường cong MNIST. Frozen gate
đã quyết định `promote_to_cifar=true`; cả `0.15` và `0.35` đều pass toàn bộ
điều kiện promotion.

### 3.3. Training matrix

- Model family: F-CS-WAE.
- Model seed: `0`.
- Epochs: `300`.
- Semantic dimension: `64`.
- Style dimension: `128`.
- Backbone: ResNet-18.
- `delta_final=0.0`.
- Checkpoint recovery interval: 5 epochs.
- `sigma_floor=0` tái sử dụng paper baseline checkpoint hiện có.
- Các mức `sigma_floor>0` được train với intervention hoạt động trong cả
  training và audit.
- Hai GPU được dùng để chạy các level theo hàng đợi độc lập.

Experiment hiện là single-seed controlled sweep. Nó xác định sự tồn tại và
đường biên của phenomenon, chưa đo between-seed variance.

## 4. Audit protocol

Mỗi checkpoint được audit bằng protocol `stage0-1.2.0`:

- evaluation population: 2.048 held-out test examples;
- primary sampling-contract view: stochastic `z_s_sample`;
- independent Gaussian reference cho global và per-class MMD;
- global và conditional MMD: unbiased multiscale MMD squared U-statistic;
- HSIC: tối đa 1.024 samples, 200 permutations, plus-one p-value;
- stochastic logistic probe: stratified 60/20/20 split;
- feature standardization fit chỉ trên probe-train;
- information proxy:

  \[
  \widehat I_{LB}(Z_s;Y)=\widehat H(Y)
  -\operatorname{CE}_{test}(r_\phi(Y\mid Z_s));
  \]

- external Gen-ACC: 200 generated samples mỗi class, chấm bằng independent
  pixel-space classifier train chỉ trên real images.

External evaluator đạt `99.34%` real-test accuracy trên MNIST và `96.16%` trên
CIFAR-10. Internal re-encoding accuracy không được dùng làm primary Gen-ACC.

### 4.1. Frozen CIFAR promotion gate

Một trong hai mức MNIST `0.15` hoặc `0.35` phải đồng thời thỏa:

1. entropy tăng ít nhất `0.75 nats/dimension` so với baseline;
2. stochastic linear probe accuracy ít nhất `0.20`;
3. HSIC reject ở `p <= 0.05`;
4. mean conditional MMD lớn hơn global MMD;
5. final reconstruction loss không quá `2x` baseline.

Cả hai level đều pass. CIFAR confirmation vì thế được cho phép theo rule đã
khóa, không phải do chọn level đẹp sau khi xem kết quả.

## 5. Kết quả MNIST

Chance accuracy là `10%`.

| `sigma_floor` | Entropy (nats/dim) | Median effective std | Raw floor hit | Probe ACC | `I_LB` (nats) | Global MMD2 | Conditional MMD2 | Cond./global | HSIC p | Gen-ACC | Final recon. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -3.581 | 0.00674 | 100.00% | 99.76% | 2.283 | 0.000579 | 0.016024 | 27.68x | 0.00498 | 10.15% | 0.01259 |
| 0.025 | -2.215 | 0.02589 | 99.16% | 99.51% | 2.273 | 0.000599 | 0.016352 | 27.31x | 0.00498 | 10.05% | 0.01273 |
| 0.05 | -1.553 | 0.05045 | 99.17% | 99.51% | 2.273 | 0.000648 | 0.016135 | 24.88x | 0.00498 | 11.95% | 0.01315 |
| 0.15 | -0.477 | 0.15015 | 99.77% | 98.78% | 2.228 | 0.000654 | 0.012044 | 18.41x | 0.00498 | 37.35% | 0.01520 |
| 0.35 | 0.381 | 0.35006 | 96.50% | 95.61% | 2.159 | 0.000518 | 0.007516 | 14.51x | 0.00498 | 93.50% | 0.01658 |

![MNIST controlled posterior-entropy sweep](../runs_f/entropy_sweep/mnist/entropy_sweep.png)

### 5.1. Quan sát

Entropy tăng đơn điệu tổng cộng `3.962 nats/dimension`, trong khi median
effective standard deviation tăng từ `0.00674` lên `0.35006`. Intervention vì
thế đã tạo posterior samples stochastic rõ ràng.

Leakage giảm theo entropy nhưng không biến mất:

- probe chỉ giảm từ `99.76%` xuống `95.61%`;
- `I_LB` vẫn ở `2.159 nats` tại level cao nhất;
- mọi level đều có HSIC p-value nhỏ nhất có thể với 200 permutations,
  `1/(200+1)=0.004975`;
- conditional MMD giảm nhưng vẫn lớn hơn global MMD hơn `14x` tại level cao
  nhất.

Generation từ global Gaussian prior đồng thời cải thiện mạnh, từ `10.15%` lên
`93.50%` Gen-ACC. Điều này cho thấy increased effective stochasticity không
đơn thuần phá hỏng decoder để làm mất leakage; trên MNIST, nó cải thiện rõ
sampling competence trong khi label information vẫn còn trong posterior style
sample.

Final reconstruction loss tại `sigma_floor=0.35` tăng `31.7%` so với baseline,
vẫn dưới competence threshold `2x`.

### 5.2. Kết luận riêng cho MNIST

MNIST cung cấp evidence mạnh nhất chống lại criticism “leakage chỉ tồn tại vì
posterior sample gần deterministic”. Label leakage vẫn rất lớn sau khi entropy
và effective noise tăng mạnh.

## 6. Kết quả CIFAR-10

Chance accuracy là `10%`.

| `sigma_floor` | Entropy (nats/dim) | Median effective std | Raw floor hit | Probe ACC | `I_LB` (nats) | Global MMD2 | Conditional MMD2 | Cond./global | HSIC p | Gen-ACC | Final recon. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -3.581 | 0.00674 | 100.00% | 61.46% | 1.111 | 0.000249 | 0.005789 | 23.29x | 0.00498 | 12.60% | 0.05140 |
| 0.15 | -0.477 | 0.15015 | 100.00% | 28.78% | 0.256 | 0.000743 | 0.003690 | 4.97x | 0.00498 | 27.35% | 0.05940 |
| 0.35 | 0.369 | 0.35006 | 100.00% | 9.76% | -0.115 | 0.000667 | 0.001287 | 1.93x | 0.00498 | 54.25% | 0.06476 |

![CIFAR-10 controlled posterior-entropy confirmation](../runs_f/entropy_sweep/cifar10/entropy_sweep.png)

### 6.1. Operating point được xác nhận: `sigma_floor=0.15`

So với baseline:

- entropy tăng `3.104 nats/dimension`;
- probe vẫn đạt `28.78%`, cao hơn chance `18.78` percentage points;
- classifier lower bound vẫn dương ở `0.256 nats`;
- HSIC reject với `p=0.00498`;
- conditional MMD lớn gấp `4.97x` global MMD;
- reconstruction loss chỉ tăng `15.6%`;
- external Gen-ACC tăng từ `12.60%` lên `27.35%`.

Đây là CIFAR operating point trực tiếp hỗ trợ claim rằng leakage có thể tồn tại
dưới một posterior đã được làm stochastic rõ rệt mà model chưa collapse.

### 6.2. Boundary condition: `sigma_floor=0.35`

Tại mức cao nhất:

- probe accuracy là `9.76%`, xấp xỉ chance;
- `I_LB=-0.115 nats` không có nghĩa mutual information âm; nó chỉ có nghĩa
  classifier-based estimator không cung cấp positive lower bound ở run này;
- HSIC vẫn reject và conditional MMD vẫn lớn gấp `1.93x` global MMD;
- Gen-ACC tiếp tục tăng tới `54.25%`;
- reconstruction loss tăng `26.0%`, chưa collapse.

Vì probe và information lower bound không còn hỗ trợ label recoverability,
không được dùng level này để claim “strong class leakage persists at maximum
entropy on CIFAR-10”. Nó là bằng chứng rằng dependence/distributional
discrepancy vẫn detect được, đồng thời đánh dấu nơi linear label leakage đã bị
attenuate về chance.

### 6.3. Kết luận riêng cho CIFAR-10

CIFAR-10 xác nhận phenomenon ở mức controlled entropy trung bình `0.15`, nhưng
cũng cho thấy leakage giảm rõ khi noise tiếp tục tăng. Kết quả này mạnh hơn một
MNIST-only result vì nó tái xuất hiện trên ảnh RGB phức tạp hơn, đồng thời đưa
ra một boundary condition trung thực thay vì khẳng định leakage bất biến theo
entropy.

## 7. Kết luận đối với luận điểm của paper

### 7.1. Claim được hỗ trợ

Report hỗ trợ câu sau cho main paper hoặc rebuttal:

> Across MNIST and CIFAR-10, conditional leakage persists at controlled
> stochastic posterior levels that preserve reconstruction competence,
> showing that near-deterministic posterior sampling is not necessary for the
> observed failure.

Một phiên bản thận trọng hơn:

> Conditional leakage persists after substantially increasing effective
> posterior entropy, including at non-degenerate operating points where
> reconstruction remains competent.

Logic của kết luận là existential, không universal: có các operating points
entropy cao hơn baseline nơi leakage vẫn tồn tại. Kết quả CIFAR `0.35` không
mâu thuẫn với claim này.

### 7.2. Điều experiment không chứng minh

Không được suy ra rằng:

- leakage tồn tại ở mọi posterior entropy level;
- mọi factorized model hoặc mọi dataset đều có cùng behavior;
- raw variance head đã tự thoát posterior collapse;
- posterior collapse hoàn toàn không liên quan đến độ lớn leakage;
- probe accuracy là exact mutual information;
- global MMD nhỏ đồng nghĩa exact marginal equality;
- tăng entropy tự động giải quyết representation factorization.

Điều chính xác nhất là near-deterministic effective sampling không phải điều
kiện cần để quan sát failure. Raw variance head vẫn floor-seeking: raw floor-hit
rate là `96.5--100%` ở MNIST và `100%` ở toàn bộ CIFAR sweep.

## 8. Limitations và công việc còn lại

1. Sweep hiện chỉ có model seed 0; chưa có confidence interval qua training
   seeds.
2. Intervention áp variance floor trực tiếp, không chứng minh model tự học
   high-entropy posterior.
3. Default run dùng `--skip-eval`; vì vậy report này chưa có SSIM, LPIPS hoặc
   FID theo từng entropy level. Competence guard hiện dựa trên reconstruction
   loss và independent external Gen-ACC.
4. HSIC p-value bằng minimum resolution của 200 permutations. Nó chứng minh
   rejection ở calibration hiện tại, không lượng hóa effect size một cách độc
   lập.
5. Kết quả chỉ thuộc F-CS-WAE; cross-model DRIT/DIVA audit là experiment riêng
   và chưa được thay thế bởi sweep này.
6. Audit manifest ghi dirty worktree. Checkpoint và result hashes đảm bảo tính
   toàn vẹn artifact, nhưng paper package cuối nên khóa một clean source
   revision tương ứng.

Nếu cần tăng độ chắc trước submission, bước có giá trị cao nhất là lặp lại ba
operating points `{0, 0.15, 0.35}` trên thêm training seeds, thay vì mở rộng
thêm nhiều entropy levels.

## 9. Reproducibility log

### 9.1. Commands

MNIST full sweep:

```bash
.venv/bin/python scripts/run_posterior_entropy_sweep.py \
  --stage mnist --devices cuda:0 cuda:1 \
  --checkpoint-every 5 --skip-existing
```

CIFAR full confirmation command đã chạy:

```bash
.venv/bin/python scripts/run_posterior_entropy_sweep.py \
  --stage cifar --devices cuda:0 cuda:1 \
  --checkpoint-every 5 --skip-existing
```

Lệnh trên được chạy dưới user service `fcswae-entropy-cifar-full`; service chạy
từ 06:11 đến 10:34 ngày 2026-08-12, kết thúc với exit status 0 và không restart.
Runner cũng hỗ trợ `--stage cifar-if-needed` để tự kiểm frozen gate. Cả hai path
đều dùng đúng fixed levels `{0, 0.15, 0.35}`. Trong experiment này, frozen MNIST
decision đã là `true`, nên explicit full-suite execution không thay đổi tập
level hợp lệ hoặc tạo post-hoc level selection.

### 9.2. Artifact chính

- [MNIST summary](../runs_f/entropy_sweep/mnist/entropy_sweep_summary.json)
- [MNIST figure](../runs_f/entropy_sweep/mnist/entropy_sweep.png)
- [Frozen promotion decision](../runs_f/entropy_sweep/mnist/cifar_promotion_decision.json)
- [CIFAR-10 summary](../runs_f/entropy_sweep/cifar10/entropy_sweep_summary.json)
- [CIFAR-10 figure](../runs_f/entropy_sweep/cifar10/entropy_sweep.png)
- Per-level audit:
  `runs_f/entropy_sweep/<dataset>/sigma_<level>/seed_0/audit_protocol_1_2.json`
- Per-level checksum: cùng đường dẫn với suffix `.sha256`.
- Per-level execution log:
  `runs_f/entropy_sweep/<dataset>/sigma_<level>/seed_0/pipeline.log`.

Mỗi audit manifest chứa checkpoint path và SHA-256, evaluation subset hash,
external evaluator hash, model config, protocol version, runtime và repository
revision.

### 9.3. Completion checks

- MNIST có đủ 5 predefined levels.
- CIFAR-10 có đủ 3 fixed confirmation levels.
- Mọi audit dùng `stage0-1.2.0`.
- Audit JSON checksum và checkpoint hash đã được đối chiếu thành công.
- Mọi metric chính trong summaries là finite.
- Mỗi nonzero-floor run có đủ 300 training-history entries và final
  checkpoint.
- Background CIFAR service kết thúc với `Result=success`, exit status 0 và
  không restart.
- Regression suite tại thời điểm viết report: 18 tests pass; 2 legacy ablation
  tests fail vì còn unpack API model cũ thành bốn outputs. Audit protocol và
  training-resume tests đều pass; hai lỗi legacy này không phát sinh từ
  entropy pipeline.

## 10. Vị trí trong roadmap bốn experiment

Controlled posterior entropy là Experiment 2 trong roadmap ICLR 2027 và phần
GPU execution của experiment này đã hoàn tất. Nó xử lý criticism về
near-deterministic posterior, nhưng không thay thế:

- paper-scale synthetic exact-marginal replication;
- cross-model audit trên F-CS-WAE, DRIT và DIVA;
- Shapes3D ground-truth factor audit.

Do đó kết quả này đã củng cố mechanism claim của F-CS-WAE, nhưng chưa tự nó mở
rộng conclusion sang mọi factorized representation model.
