# Theory–experiment mismatch: posterior sample vs. posterior mean

## Thông tin evaluation

- **Ngày chạy:** 2026-08-07 (Asia/Ho_Chi_Minh)
- **Protocol:** `stage0-1.0.0`
- **Evaluation subset:** 2,048 test examples, evaluation seed 0
- **Probe split:** stratified 60/20/20 train/validation/test
- **Standardization:** chỉ fit trên probe-train
- **Probe suite:** logistic, fixed two-layer MLP, RBF-SVM và k-NN
- **HSIC:** multiscale, 1,024 examples, 200 permutations
- **Checkpoint:** MNIST seed 0, Fashion-MNIST seed 0 và CIFAR-10 seeds 0/1/2

Tất cả con số dưới đây được đo từ các checkpoint có sẵn. Không có model nào
được train lại.

## Kết luận chính

**Theory–experiment mismatch không làm thay đổi kết luận về leakage.** Khi
chạy diagnostic trực tiếp trên stochastic posterior sample $z_s$, kết quả gần
như giống posterior mean $\mu_s$.

Điều này cho thấy class leakage không chỉ tồn tại trong deterministic
representation $\mu_s$, mà còn tồn tại trực tiếp trong biến ngẫu nhiên $z_s$
được sử dụng trong lý thuyết và sampling contract.

## 1. Probe accuracy

Đơn vị là test accuracy (%). Chance level của cả ba dataset là 10%.

| Dataset / latent view | Logistic | MLP | RBF-SVM | k-NN |
|---|---:|---:|---:|---:|
| MNIST $\mu_s$ | 99.76 | 99.51 | 99.76 | 97.80 |
| MNIST $z_s$ | 99.76 | 99.51 | 99.76 | 97.80 |
| Fashion-MNIST $\mu_s$ | 88.29 | 90.73 | 93.17 | 85.12 |
| Fashion-MNIST $z_s$ | 88.29 | 90.73 | 93.17 | 85.61 |
| CIFAR-10 $\mu_s$, 3 model seeds | 68.46 ± 6.29 | 70.49 ± 3.83 | 70.24 ± 3.23 | 42.52 ± 6.93 |
| CIFAR-10 $z_s$, 3 model seeds | 68.62 ± 6.19 | 70.57 ± 3.50 | 70.16 ± 3.36 | 42.20 ± 7.11 |

Chênh lệch lớn nhất giữa $z_s$ và $\mu_s$:

- **MNIST:** 0 điểm phần trăm.
- **Fashion-MNIST:** 0.49 điểm phần trăm.
- **CIFAR-10:** tối đa 0.73 điểm phần trăm trên từng seed.

Các sai khác này rất nhỏ so với mức leakage quan sát được. Do đó kết luận
không phụ thuộc vào việc probe sử dụng posterior mean hay posterior sample.

## 2. Distributional diagnostics

| Dataset | $\Delta_{\mathrm{inter}}(\mu_s)$ | $\Delta_{\mathrm{inter}}(z_s)$ | HSIC $p$, cả hai view |
|---|---:|---:|---:|
| MNIST | 6.2641 | 6.2642 | 0.00498 |
| Fashion-MNIST | 4.7912 | 4.7913 | 0.00498 |
| CIFAR-10, 3 model seeds | 3.9538 ± 0.2691 | 3.9538 ± 0.2687 | 0.00498 ở mọi seed/view |

Với 200 plus-one-calibrated permutations, p-value nhỏ nhất có thể đạt được là:

$$
p_{\min}=\frac{1}{201}\approx0.00498.
$$

Cả $\mu_s$ và $z_s$ đều đạt đúng mức nhỏ nhất này ở mọi checkpoint. Vì vậy,
kết quả cung cấp bằng chứng rõ ràng để bác bỏ giả thuyết độc lập
$z_s\perp y$.

Trong khi đó, global MMD của stochastic $z_s$ vẫn rất nhỏ:

- **MNIST:** $0.000572$
- **Fashion-MNIST:** $0.000287$
- **CIFAR-10:** $0.000306\pm0.000054$

Đây là kết quả trực tiếp hỗ trợ luận điểm trung tâm của paper: aggregate
posterior có thể gần prior theo MMD, trong khi stochastic style latent vẫn
chứa rất nhiều class information.

$$
q(z_s)\approx p(z_s)
\quad\not\Rightarrow\quad
q(z_s\mid y)=q(z_s).
$$

## 3. Vì sao $z_s$ và $\mu_s$ gần như giống nhau?

Encoder sử dụng Gaussian reparameterization:

$$
z_s=\mu_s+\sigma_s\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

Kiểm tra trực tiếp các checkpoint cho thấy:

- **MNIST:** 99.9966% giá trị `logvar_s ≤ -10`.
- **Fashion-MNIST:** 99.9989%.
- **CIFAR-10:** 100% ở cả ba model seed.

Model và evaluator clamp `logvar_s` vào khoảng `[-10, 10]`. Vì vậy effective
posterior standard deviation gần như luôn bằng:

$$
\sigma_s\simeq e^{-5}=0.006738.
$$

Trong khi đó, between-class separation nằm trong khoảng $3.95$–$6.26$. Do đó:

$$
\lVert\sigma_s\epsilon\rVert
\ll \text{between-class separation},
$$

và trên thực tế:

$$
z_s\approx\mu_s.
$$

Nói cách khác, stochastic style posterior của các checkpoint hiện tại đã gần
như trở thành một deterministic representation. Posterior sampling thêm quá
ít noise để thay đổi kết luận leakage.

Sự đồng nhất về số liệu này không làm $z_s$ và $\mu_s$ tương đương về mặt lý
thuyết. Với checkpoint có posterior variance không collapse, hai diagnostic
vẫn có thể cho kết quả khác nhau.

## 4. Kết luận cho theory–experiment mismatch

Trước khi bổ sung $z_s$-probe, paper chỉ có thể kết luận chắc chắn:

> Posterior mean representation $\mu_s$ chứa class information.

Sau khi chạy diagnostic mới, paper có thể đưa ra kết luận mạnh hơn:

> Class leakage persists directly in posterior samples $z_s$, the stochastic
> variable appearing in the theoretical sampling contract. Therefore, the
> finding is not an artifact of using posterior means.

Như vậy, criticism về theory–experiment mismatch đã được kiểm tra trực tiếp:
leakage vẫn tồn tại ở chính random variable xuất hiện trong phát biểu lý
thuyết.

Kết quả này **hỗ trợ paper dưới dạng failure-mode/audit paper**, nhưng không
chứng minh F-CS-WAE đã disentangle tốt hoặc đạt class-invariant style. Ngược
lại, nó cho thấy cả deterministic representation lẫn stochastic style latent
đều giữ class information mạnh.

## 5. Các diagnostic vẫn phải đặt tên riêng

Không nên hợp nhất hai latent view dù kết quả hiện tại gần nhau:

- **Mean probe:** probe trên $\mu_s$; đo leakage trong deterministic
  representation.
- **Sample probe:** probe trên posterior sample $z_s$; kiểm tra trực tiếp biến
  ngẫu nhiên trong sampling contract.
- **Sampling-contract MMD/HSIC:** ưu tiên posterior sample $z_s$.
- **Deterministic swap:** tiếp tục dùng $\mu_s$, nhưng phải gọi rõ là
  `mean-swap` hoặc `deterministic mean swap`.
- **JointMMD hiện tại:** dùng $(\mu_c,\mu_s)$, nên phải gọi là mean-view
  JointMMD và không trình bày như stochastic JointMMD.

Sự gần nhau hiện tại là một kết quả thực nghiệm do posterior variance
collapse, không phải một đẳng thức lý thuyết giữa hai diagnostic.

## 6. Đoạn đề xuất đưa vào paper

> *To align the diagnostics with the theoretical sampling contract, we repeat
> the fixed probe suite on seeded posterior samples $z_s$, in addition to
> deterministic posterior means $\mu_s$. The two views yield nearly identical
> recovery: for example, logistic accuracy is $99.76\%$ for both views on
> MNIST and $68.46\pm6.29\%$ versus $68.62\pm6.19\%$ on CIFAR-10. This
> agreement is explained by an almost deterministic learned style posterior,
> whose effective standard deviation is typically $e^{-5}$. Thus, class
> leakage persists in the stochastic variable itself and is not an artifact
> of mean-based diagnostics.*

## 7. Hạn chế khi diễn giải

- MNIST và Fashion-MNIST hiện chỉ có một model seed; không được gắn error bar
  hoặc tuyên bố độ ổn định qua training seeds.
- CIFAR-10 sử dụng ba model seeds và báo cáo mean ± sample standard deviation.
- Mỗi probe test split có 410 examples, nên một dự đoán tương ứng khoảng 0.244
  điểm phần trăm.
- HSIC dùng 200 permutations; phải báo cáo $p=0.00498$, không báo cáo $p=0$.
- Kết quả hiện tại sử dụng một seeded posterior draw cho mỗi example. Vì
  effective posterior variance đã collapse, Monte Carlo variation dự kiến rất
  nhỏ, nhưng các model tương lai vẫn nên kiểm tra nhiều posterior draws.

## 8. Kết luận cuối cùng

Kết quả mới khép được criticism về theory–experiment mismatch và làm luận
điểm chính của paper mạnh hơn:

1. Global MMD của stochastic $z_s$ nhỏ.
2. Sample probes vẫn khôi phục label với accuracy rất cao.
3. HSIC bác bỏ independence ở mọi dataset và model seed.
4. Kết luận gần như không thay đổi khi chuyển từ $\mu_s$ sang $z_s$.

Tuy nhiên, paper cũng phải công khai rằng learned style posterior gần như
deterministic do posterior variance collapse.

## Source result artifacts

- `runs_diag/stage0/mnist/baseline_seed0_samplemean.json`
- `runs_diag/stage0/fashion_mnist/baseline_seed0_samplemean.json`
- `runs_diag/stage0/cifar10/baseline_modelseed0_evalseed0_samplemean.json`
- `runs_diag/stage0/cifar10/baseline_modelseed1_evalseed0_samplemean.json`
- `runs_diag/stage0/cifar10/baseline_modelseed2_evalseed0_samplemean.json`

Mỗi JSON có file `.sha256` đi kèm và chứa dataset, evaluation seed, checkpoint
SHA-256, evaluation configuration và protocol version.
