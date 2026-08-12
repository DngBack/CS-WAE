# Kế hoạch triển khai bốn experiment bắt buộc cho ICLR 2027

## 1. Mục đích của tài liệu

Tài liệu này mô tả kế hoạch triển khai bốn experiment bổ sung cho paper
`fcswae/main_v8_iclr2027.pdf`:

1. cross-model audit trên ba họ mô hình factorized;
2. controlled posterior-entropy sweep;
3. synthetic exact/near-exact marginal matching;
4. audit trên dataset có ground-truth factors.

Mục tiêu không phải mở rộng thành hàng chục ablation rời rạc. Bốn experiment
được chọn vì mỗi experiment xử lý một điểm yếu khác nhau trong evidence chain
hiện tại. Mọi run ngoài bốn nhóm này chỉ được thực hiện nếu cần để xác nhận một
kết quả đã được định nghĩa trước, không dùng để tìm một cấu hình có lợi sau khi
đã xem số liệu.

Đây là tài liệu thiết kế và triển khai. Nó không khẳng định trước rằng leakage
sẽ xuất hiện ở mọi mô hình hoặc mọi mức entropy. Null result phải được giữ lại
và dùng để xác định phạm vi đúng của kết luận.

## 2. Mục tiêu ban đầu của paper

Paper hiện tại không nhằm chứng minh F-CS-WAE là state of the art. Mục tiêu
trung tâm là kiểm tra sampling contract của một factorized generative model:

\[
z_c \sim p(z_c\mid y), \qquad z_s \sim p(z_s),
\]

trong khi encoder tạo ra joint posterior `q(z_c,z_s|y)`. Theorem 1 phân rã
latent mismatch thành bốn phần:

1. global style-prior mismatch: `KL(q(z_s) || p(z_s))`;
2. conditional style leakage: `I(z_s; y)`;
3. semantic-prior mismatch;
4. within-class dependence: `I(z_c; z_s | y)`.

Kết quả hiện tại cho thấy trên F-CS-WAE:

- global style MMD có độ lớn nhỏ;
- label vẫn được dự đoán rất tốt từ stochastic `z_s`;
- conditional HSIC bác bỏ độc lập;
- decoder thực sự sử dụng class information nằm trong style code;
- posterior variance gần như collapse về clamp floor.

Vì vậy, claim hiện tại mạnh ở mức falsification của một implementation cụ thể,
nhưng còn bốn giới hạn:

- chỉ có một model family chính;
- reviewer có thể quy hiện tượng cho posterior collapse;
- exact-marginal counterexample mới chỉ nằm ở phần lý thuyết;
- class label chưa phải một mô tả đầy đủ của ground-truth content/style.

Bốn experiment mới ánh xạ trực tiếp vào bốn giới hạn này. Chúng không thay
Theorem 1 và không thay sampling contract ban đầu; chúng mở rộng evidence cho
những hệ quả của theorem.

## 3. Nguyên tắc chung trước khi triển khai

### 3.1. Chỉ audit native factorization

Một mô hình chỉ được đưa vào cross-model table nếu bản thân mô hình định nghĩa
content/semantic và style/residual subspaces. Không được cắt đôi latent của một
VAE hoặc WAE thông thường rồi gọi nửa sau là `z_s`.

Script `scripts/run_cross_model_diagnostics.py` hiện tại không đạt yêu cầu này:
nó coi toàn bộ Gaussian latent của một số baseline là style, và implementation
beta-TCVAE trong script chỉ là một proxy đơn giản. Các kết quả từ script này
không được dùng cho cross-model claim mới.

### 3.2. Tách posterior mean và posterior sample

- Các claim về distribution và sampling contract dùng stochastic posterior
  sample.
- Probe representation và deterministic swap được báo riêng trên posterior
  mean.
- Không được thay một view bằng view còn lại chỉ vì hai view đang cho kết quả
  gần nhau trên checkpoint cũ.

### 3.3. Dùng cùng estimator và split

Mọi experiment dùng lại nguyên tắc Stage-0:

- fixed held-out evaluation population;
- stratified 60/20/20 probe split;
- standardization fit trên probe-train;
- unbiased multiscale MMD khi evaluation;
- independent RNG streams cho posterior và prior reference;
- plus-one calibrated permutation test cho MMD/HSIC;
- checkpoint, dataset split, source revision và evaluator đều có SHA-256
  manifest.

Khi schema được mở rộng cho model adapters, entropy và ground-truth factors,
protocol phải được bump khỏi `stage0-1.1.0`. Artifact cũ không được silently
pool với artifact mới.

### 3.4. Có competence gate

Leakage chỉ có ý nghĩa nếu mô hình vẫn reconstruct, encode content và generate
ở mức không suy biến. Mỗi model/run phải qua competence gate được chốt trước.
Một model có probe style thấp vì reconstruction hoặc content accuracy đã
collapse không phải là bằng chứng factorization thành công.

### 3.5. Không dùng một statistic thay cho mutual information

Probe accuracy, HSIC và conditional MMD là các proxy khác nhau. Không được ghi
probe accuracy là `I(z_s;y)`. Khi cần một trục information-theoretic, dùng
held-out classifier lower bound:

\[
\widehat I_{\mathrm{LB}}(Z;Y)
= \widehat H(Y)-\operatorname{CE}_{test}(r_\phi(Y\mid Z)).
\]

Giá trị này phải được gọi rõ là classifier-based lower bound, không phải exact
mutual information.

## 4. Hạ tầng chung cần triển khai

Trước khi train model mới, cần tách audit khỏi class `FCSWAE` bằng một adapter
interface thống nhất. Interface tối thiểu:

```text
encode_views(images, domain, generator)
    -> content_mean
    -> content_sample
    -> style_mean
    -> style_sample
    -> style_logvar hoặc style_entropy_metadata

sample_style(domain, n, generator)
decode(content, style, domain)
style_prior_metadata(domain)
model_factorization_metadata()
```

Adapter phải mô tả:

- latent nào thực sự là style theo paper gốc của model;
- prior tương ứng là gì;
- sampler có condition trên class/domain hay không;
- content được lấy từ prior hay từ một source image;
- model có hỗ trợ decoder intervention hay chỉ latent audit.

Sampling Gaussian của F-CS-WAE cũng phải được gom vào một helper duy nhất.
Hiện công thức reparameterization xuất hiện ở model và nhiều diagnostic
scripts. Nếu chỉ sửa `forward()` cho entropy experiment, train và audit có thể
vô tình dùng hai posterior khác nhau.

Các test mới tối thiểu:

- adapter trả đúng named views và shape;
- posterior sampling reproducible với seeded generator;
- posterior/reference RNG độc lập;
- entropy tính từ đúng effective variance;
- synthetic null có MMD rejection rate gần nominal level;
- factor split không rò dữ liệu từ test vào probe/evaluator training.

## 5. Experiment 1: Cross-model audit

### 5.1. Vì sao phải làm

Đây là experiment trả lời câu hỏi: hiện tượng có phải một bug riêng của
F-CS-WAE hay là một failure mode có thể xuất hiện dưới nhiều objective
factorization khác nhau?

Nếu chỉ audit F-CS-WAE, paper có thể bị đọc như một model-debugging report. Nếu
cùng conditional leakage xuất hiện trên các native split-latent models dùng
những prior-alignment mechanisms khác nhau, theorem trở thành một công cụ audit
có tính khái quát hơn.

### 5.2. Model families

Chọn đúng ba họ:

1. **F-CS-WAE**: aggregate MMD cho style;
2. **DRIT**: content/attribute decomposition, KL-to-Gaussian cho attribute;
3. **DIVA**: ba subspaces `z_d`, `z_y`, `z_x`, trong đó residual `z_x` có
   standard Gaussian prior.

Không dùng cả MUNIT và DRIT trong main comparison vì chúng quá gần nhau về
bài toán image translation. DRIT phù hợp hơn với câu hỏi hiện tại vì attribute
code có Gaussian prior regularization rõ ràng.

### 5.3. Dataset chung

Sử dụng two-domain Rotated-MNIST:

- `y`: digit identity, 10 classes;
- `d`: hai domain rotation `-30°` và `+30°`;
- phân bổ domain độc lập với digit và cân bằng theo digit;
- cùng train/test examples và preprocessing cho cả ba model.

DRIT có domain-specific attribute spaces. Vì vậy audit được chạy trong từng
domain:

\[
q(z_s\mid y,d) \stackrel{?}{=} p_d(z_s),
\]

sau đó macro-average hai domain. Pool trực tiếp hai style spaces sẽ làm sai
sampling contract. Với DIVA, primary audited style là `z_x`; `z_d` và `z_y`
được báo như control views.

### 5.4. Metrics

Cho mỗi model, domain và model seed:

- global unbiased MMD2 và 500-permutation null;
- mean conditional MMD2 theo digit;
- conditional/global MMD ratio trong cùng model;
- logistic, MLP và RBF probe trên stochastic style sample;
- label HSIC với 200 permutations;
- mean posterior entropy hoặc model-equivalent stochasticity statistic;
- reconstruction quality;
- content/digit accuracy;
- generation hoặc translation content retention nếu decoder interface hỗ trợ.

Raw MMD không được so trực tiếp giữa các models có style dimension khác nhau.
Main comparison dựa trên within-model conditional/global ratio, probe và
calibrated HSIC.

### 5.5. Run matrix và acceptance gate

Run ba model seeds cho mỗi family trên dataset nhỏ này. Tổng cộng chín jobs,
không phải chín experiment độc lập.

Một checkpoint chỉ được đưa vào leakage comparison nếu:

- reconstruction không collapse;
- semantic/content view dự đoán digit tốt hơn threshold đã chốt trước;
- DRIT translation giữ được digit identity ở mức đủ dùng;
- sampler và latent definitions được ghi trong manifest.

Không định nghĩa trước rằng cả ba model phải leak. Có ba kết quả hợp lệ:

- global discrepancy nhỏ nhưng conditional leakage mạnh: trực tiếp hỗ trợ
  central phenomenon;
- cả global mismatch và leakage đều lớn: sampler fail, nhưng không phải
  marginal paradox;
- leakage thấp: boundary condition cho biết objective/architecture nào tránh
  failure trên setup này.

### 5.6. Ảnh hưởng đến mục tiêu paper

Experiment này **mở rộng phạm vi thực nghiệm**, nhưng không thay claim logic.
Theorem vẫn nói marginal matching không thể là certificate. Cross-model audit
cho thấy failure không nhất thiết phụ thuộc vào spherical prior, MMD training
hay kiến trúc F-CS-WAE.

Nếu chỉ F-CS-WAE leak, paper phải giữ framing “concrete case study”. Nếu nhiều
families leak, có thể đổi framing thành “cross-model audit of factorized
samplers”, nhưng vẫn không được viết rằng mọi factorized model đều thất bại.

## 6. Experiment 2: Controlled posterior entropy

Kết quả hoàn chỉnh và reproducibility log của experiment này được ghi tại
[`CONTROLLED_POSTERIOR_ENTROPY_EXPERIMENT_REPORT.md`](CONTROLLED_POSTERIOR_ENTROPY_EXPERIMENT_REPORT.md).

### 6.0. Implementation entry point

Protocol `stage0-1.2.0` dùng một sampler chung cho train và audit. Sweep MNIST
và promotion gate cố định cho CIFAR được chạy bằng:

```bash
.venv/bin/python scripts/run_posterior_entropy_sweep.py \
  --stage mnist --devices cuda:0 cuda:1 --skip-existing

.venv/bin/python scripts/run_posterior_entropy_sweep.py \
  --stage cifar-if-needed --devices cuda:0 cuda:1 --skip-existing

# Chạy toàn bộ confirmation suite bất kể kết quả gate (explicit override):
.venv/bin/python scripts/run_posterior_entropy_sweep.py \
  --stage cifar --devices cuda:0 cuda:1 --skip-existing
```

Stage thứ hai tự dừng nếu gate MNIST không pass; nó không tự chọn level đẹp
sau khi xem kết quả. Các level MNIST được khóa ở
`{0, 0.025, 0.05, 0.15, 0.35}` và CIFAR confirmation ở `{0, 0.15, 0.35}`.
Stage `cifar` vẫn dùng đúng ba level CIFAR đã khóa, nhưng bỏ qua quyết định
promote khi người chạy chủ động yêu cầu full confirmation suite. Cả hai CIFAR
stage đều đòi hỏi MNIST summary đã hoàn tất để tránh tranh GPU với MNIST.

### 6.0.1. Recovery và durable execution

Mỗi training job ghi nguyên tử `training_checkpoint.pt` sau mỗi 5 epoch. File
này chứa model, optimizer, scheduler, EMA centers, completed history, CPU/CUDA
RNG và DataLoader shuffle-generator state. Vì vậy resume tiếp tục từ đúng epoch
boundary và không âm thầm đổi sampling protocol. `--auto-resume` đã được sweep
runner truyền mặc định; chạy lại cùng command sẽ dùng checkpoint mới nhất.

Không chạy multi-hour sweep trong một interactive terminal/session có TTL.
Khởi chạy dưới user systemd service:

```bash
systemd-run --user --unit=fcswae-entropy-mnist --collect --same-dir \
  --property=Restart=on-failure --property=RestartSec=30s \
  .venv/bin/python scripts/run_posterior_entropy_sweep.py \
  --stage mnist --devices cuda:0 cuda:1 --checkpoint-every 5 --skip-existing

systemctl --user status fcswae-entropy-mnist
journalctl --user -u fcswae-entropy-mnist -f
```

Có thể arm CIFAR stage bằng path trigger trước khi MNIST kết thúc. Trigger chỉ
chạy khi promotion-decision artifact được tạo; script vẫn tự no-op nếu gate
false:

```bash
systemd-run --user --unit=fcswae-entropy-cifar --collect --same-dir \
  --property=Restart=on-failure --property=RestartSec=30s \
  --path-property=PathChanged="$PWD/runs_f/entropy_sweep/mnist/cifar_promotion_decision.json" \
  .venv/bin/python scripts/run_posterior_entropy_sweep.py \
  --stage cifar-if-needed --devices cuda:0 cuda:1 \
  --checkpoint-every 5 --skip-existing
```

Khi đã quyết định chạy toàn bộ CIFAR confirmation suite bất kể gate, thay
`--stage cifar-if-needed` bằng `--stage cifar` trong trigger trên. Đây là
explicit compute override; các level vẫn cố định ở `{0, 0.15, 0.35}`.

Nếu host/service bị restart, chạy lại chính lệnh `systemd-run` sau khi unit cũ
không còn active. Tối đa mất phần epoch kể từ checkpoint gần nhất; artifact đã
hoàn tất và audit hiện hành được reuse.

### 6.1. Vì sao phải làm

Style posterior hiện gần deterministic vì phần lớn raw log-variance nằm dưới
clamp floor. Reviewer có thể cho rằng conditional leakage chỉ cao vì noise gần
bằng không. Đây là threat trực tiếp nhất đối với empirical result, dù không
ảnh hưởng đến Proposition 1 hoặc Theorem 1.

Experiment cần thay đổi stochasticity một cách có kiểm soát, trong khi giữ
architecture, dataset và phần còn lại của objective càng cố định càng tốt.

### 6.2. Intervention

Dùng effective posterior variance floor:

\[
\sigma_{\mathrm{eff}}^2(x)
= \exp(\operatorname{clamp}(\log\sigma_s^2(x),-10,10))
+ \sigma_{\mathrm{floor}}^2.
\]

Candidate levels:

```text
sigma_floor in {0, 0.025, 0.05, 0.15, 0.35}
```

Đây là controlled-noise intervention, không phải bằng chứng raw log-variance
head đã thôi collapse. Cả raw floor-hit rate và effective entropy phải được
báo.

Không dùng per-example KL hoặc entropy penalty làm intervention chính. Các loss
đó đồng thời thay đổi posterior mean, prior fit và capacity, nên khó tách causal
effect của stochasticity.

### 6.3. Trục entropy

Không dùng requested `sigma_floor` làm trục chính. Dùng entropy đo được trên
held-out posterior:

\[
\bar H_s
= \frac{1}{d_s}\mathbb E_x\left[
\frac12\sum_j\log(2\pi e\,\sigma_{\mathrm{eff},j}^2(x))
\right]
\]

với đơn vị `nats/dimension`. Báo thêm median `sigma_eff` và RMS noise norm
`sqrt(sum_j sigma_eff,j^2)` để dễ diễn giải.

### 6.4. Metrics và figure

Mỗi level báo:

- measured entropy và median effective standard deviation;
- raw-logvar clamp-floor hit rate;
- stochastic logistic/RBF probe;
- classifier lower bound `I_LB(z_s;y)`;
- label HSIC và permutation p-value;
- global và mean conditional MMD;
- stochastic conditional HSIC cho term 4;
- external global-prior Gen-ACC;
- SSIM, LPIPS và FID/quality control.

Main figure gồm ba panel dùng chung trục `H_bar_s`:

1. `I_LB` cùng probe accuracy;
2. global MMD và conditional MMD;
3. external Gen-ACC.

Không nên vẽ ba đại lượng khác đơn vị trên một y-axis duy nhất.

### 6.5. Acceptance và interpretation

Kết quả rebuttal mạnh nhất là tồn tại một level mà:

- entropy tăng đáng kể;
- reconstruction/generation chưa collapse;
- held-out probe vẫn trên chance;
- HSIC vẫn reject;
- conditional discrepancy vẫn lớn hơn global discrepancy.

Nếu leakage giảm về chance khi entropy đủ cao, kết quả vẫn phải được báo. Khi
đó kết luận đúng là posterior collapse là empirical mechanism quan trọng trong
implementation hiện tại; theorem vẫn giữ nguyên vì theorem không giả định
posterior deterministic.

Rule promotion sang CIFAR được khóa trước khi đọc sweep. Chỉ chạy ba mức
`{0, 0.15, 0.35}` nếu ít nhất một trong hai mức entropy cao thỏa đồng thời:

- entropy tăng ít nhất `0.75 nats/dimension` so với mức 0;
- stochastic linear probe accuracy ít nhất `0.20` (chance MNIST là `0.10`);
- HSIC reject ở `p <= 0.05`;
- mean conditional MMD lớn hơn global MMD;
- final training reconstruction loss không quá `2x` baseline.

External Gen-ACC vẫn là outcome chính trên figure, nhưng không dùng làm gate:
global-prior Gen-ACC thấp có thể chính là hậu quả của style leakage đang được
đo, nên dùng nó để loại level sẽ tạo selection bias. Reconstruction stability
đóng vai trò competence guard độc lập hơn cho quyết định promotion.

### 6.6. Ảnh hưởng đến mục tiêu paper

Experiment này **thu hẹp hoặc củng cố cơ chế thực nghiệm**, không mở rộng
theorem. Nó quyết định paper có thể viết “leakage persists under clearly
stochastic posteriors” hay chỉ nên viết “the audited checkpoints leak under a
collapsed style posterior”.

Nó cũng ngăn paper vô tình đồng nhất posterior mean với posterior sample. Đây
là phần trực tiếp bảo vệ tính đúng đắn của sampling-contract claim.

## 7. Experiment 3: Exact/near-exact marginal matching

### 7.0. Implementation entry point

Chạy protocol test CPU nhanh trước model sweep:

```bash
.venv/bin/python scripts/run_synthetic_exact_marginal.py
```

Run dùng cho paper (nhiều repetitions/permutations hơn) là:

```bash
.venv/bin/python scripts/run_synthetic_exact_marginal.py \
  --paper \
  --output-dir runs_diag/synthetic_exact_marginal_paper
```

Quick artifacts được giữ ở `runs_diag/synthetic_exact_marginal/`; paper-scale
artifacts được tách riêng ở `runs_diag/synthetic_exact_marginal_paper/` để
không ghi đè protocol test. Global MMD được tính một lần trên cùng Gaussian
sample trong mỗi replication rồi gắn với mọi `kappa`; cách này làm rõ rằng thay
đổi trên đồ thị đến từ label mechanism, không phải thay marginal.

Paper runner chạy các replications độc lập bằng CPU process workers. HSIC
permutations được tính theo batch bằng biểu diễn one-hot
`trace(S^T K_centered S)`, tương đương với permuting full centered label kernel
nhưng không tạo một kernel `n x n` mới cho từng permutation. Benchmark một
paper replication trên host hiện tại mất khoảng 17.7 giây và peak RSS khoảng
1.1 GiB với tám Torch CPU threads.

Sau mỗi replication, runner ghi nguyên tử
`partial_results.json`. Chạy lại cùng command sẽ xác minh scientific config và
skip các replication đã hoàn tất. `results.json`, frozen
`acceptance_criteria.json` và `acceptance_result.json` đều có SHA-256 sidecar.
Một config khác không được resume vào cùng output directory.

Durable paper run:

```bash
systemd-run --user \
  --unit=fcswae-synthetic-paper \
  --collect --same-dir \
  --property=Restart=on-failure \
  --property=RestartSec=30s \
  .venv/bin/python scripts/run_synthetic_exact_marginal.py \
  --paper \
  --output-dir runs_diag/synthetic_exact_marginal_paper

systemctl --user status fcswae-synthetic-paper
journalctl --user -u fcswae-synthetic-paper -f
```

Frozen acceptance được ghi trước paper result:

- mean linear-probe accuracy phải lớn hơn 90% tại từng
  `kappa in {4, 8, 16}`;
- Spearman trend của HSIC và conditional MMD theo bảy `kappa` phải ít nhất
  0.70, đồng thời high-kappa mean phải lớn hơn null-kappa mean;
- global MMD phải giống nhau qua `kappa` trong từng replication, sai số tối đa
  `1e-12`;
- global-MMD rejection count phải nằm trong `[0, 6]`, central 95% prediction
  interval của `Binomial(n=50, p=0.05)`.

### 7.0.1. Paper-scale result đã hoàn tất

Phân tích đầy đủ, proposed paper wording và artifact inventory được ghi tại
[`SYNTHETIC_EXACT_MARGINAL_EXPERIMENT_REPORT.md`](SYNTHETIC_EXACT_MARGINAL_EXPERIMENT_REPORT.md).

Paper run hoàn tất ngày 2026-08-12 với 50/50 replications, `n=2,048`, latent
dimension 128 và 499 permutations cho cả global MMD và HSIC. User service
`fcswae-synthetic-paper` kết thúc với exit status 0, không restart. Artifact
được ghi tại `runs_diag/synthetic_exact_marginal_paper/`.

| `kappa` | Linear probe mean ± std | Global MMD rejection | HSIC rejection | Mean conditional MMD2 |
|---:|---:|---:|---:|---:|
| 0 | 49.58 ± 1.86% | 6% | 2% | 0.000006 |
| 0.5 | 62.48 ± 2.58% | 6% | 100% | 0.000130 |
| 1 | 74.49 ± 2.09% | 6% | 100% | 0.000280 |
| 2 | 83.28 ± 2.07% | 6% | 100% | 0.000405 |
| 4 | 89.51 ± 1.45% | 6% | 100% | 0.000455 |
| 8 | 91.82 ± 1.63% | 6% | 100% | 0.000479 |
| 16 | 92.30 ± 1.51% | 6% | 100% | 0.000478 |

Global MMD mean là khoảng `-9.47e-7` và giống nhau chính xác qua mọi `kappa`
trong từng replication, vì cùng Gaussian sample được dùng trước khi gán label.
Rejection count là `3/50`, tương ứng 6% và nằm trong frozen Binomial interval
`[0, 6]`. HSIC trend có Spearman `1.0`; conditional-MMD trend có Spearman
`0.964`.

Strict composite acceptance có `passes_all=false` vì criterion được khóa yêu
cầu probe lớn hơn 90% tại **từng** `kappa in {4, 8, 16}`, trong khi `kappa=4`
đạt 89.51%, thiếu 0.49 percentage points. Không sửa criterion hoặc chọn lại
`kappa` sau khi xem result. Pedagogical target ban đầu “probe vượt 90% ở
`kappa` lớn” vẫn được quan sát tại `kappa=8` và `16`; estimator-calibration,
trend và exact-marginal invariance gates đều pass. Paper phải báo cả hai facts:
core construction được thực nghiệm hóa thành công, nhưng predeclared strict
all-large-kappa composite gate không pass toàn bộ.

### 7.1. Vì sao phải làm

Proposition 1 đã cho một counterexample exact trên lý thuyết, nhưng reader vẫn
phải tự nối kết proposition với MMD/probe trong experiment. Synthetic figure
biến logical gap thành một kết quả quan sát trực tiếp, đồng thời kiểm tra chính
audit implementation dưới một null distribution đã biết.

### 7.2. Data-generating process

Chọn binary construction như Proposition 1 với `d=128`:

\[
z \sim N(0,I), \qquad
p(y=1\mid z)=\sigma(2\kappa z_1).
\]

Marginal `q(z)` là standard Gaussian chính xác với mọi `kappa`, vì `z` được
sample trước và class chỉ được gán có điều kiện sau đó. Do symmetry, class
marginal cân bằng. Khi `kappa` lớn, class tiến tới partition theo dấu của
`z_1`, nên linear probe có thể tiến tới accuracy gần một. Multiclass softmax
partition vẫn được script hỗ trợ bằng `--n-classes`, nhưng không phải protocol
chính vì binary construction bám sát proposition và đạt target >90% ổn định
với evaluation population hữu hạn.

Sweep:

```text
kappa in {0, 0.5, 1, 2, 4, 8, 16}
```

### 7.3. Protocol

Cho mỗi level:

- sinh độc lập probe-train/validation/test data;
- dùng 2,048 held-out samples cho canonical audit;
- dùng independent Gaussian reference;
- chạy 500 MMD permutations;
- chạy logistic/RBF probe, HSIC và conditional MMD;
- lặp lại khoảng 50 independent replications trên CPU.

Không chọn một seed có MMD p-value lớn. Dưới exact equality null, khoảng 5%
replications vẫn có thể reject ở alpha 0.05. Figure phải báo MMD distribution
hoặc rejection frequency, không chỉ một p-value thuận lợi.

### 7.4. Figure và acceptance gate

Figure cần thể hiện đồng thời:

- analytic statement `q(z)=N(0,I)`;
- mean empirical MMD cùng null band;
- MMD rejection frequency gần nominal level;
- probe accuracy tăng theo `kappa` và vượt 90% ở mức lớn;
- conditional MMD/HSIC tăng dù global marginal không đổi.

Đây cũng là unit/integration test cho audit: nếu MMD systematically reject quá
nhiều dưới construction này, estimator hoặc calibration phải được sửa trước
khi chạy model thật.

### 7.5. Ảnh hưởng đến mục tiêu paper

Experiment này **không tạo claim mới**; nó trực quan hóa Proposition 1. Nó giúp
tách hai thông điệp thường bị lẫn:

- “global MMD nhỏ nhưng statistically nonzero” ở checkpoint thật;
- “marginal equality chính xác nhưng conditional leakage tùy ý lớn” trong
  construction.

Nhờ đó paper không cần phóng đại rằng F-CS-WAE đã đạt exact marginal match.
Exact claim thuộc về synthetic construction; checkpoint claim vẫn chỉ là
numerically small global discrepancy.

## 8. Experiment 4: Ground-truth factor dataset

### 8.1. Vì sao phải làm

Class label chỉ là một proxy cho semantic content. Nó không cho biết model có
tách shape, color, pose và scale đúng hướng hay không. Ground-truth factor
dataset cho phép audit cả hai chiều:

- semantic factor có leak vào style code không;
- style factors có leak ngược vào semantic code không.

### 8.2. Chọn Shapes3D

Chọn Shapes3D thay vì Causal3DIdent hoặc MPI3D cho vòng này. Shapes3D có đầy đủ
480,000 combinations của sáu independent factors:

- floor hue;
- wall hue;
- object hue;
- scale;
- shape;
- orientation.

Chọn `shape` làm designated semantic variable, bốn classes. Năm factors còn
lại là ground-truth style variables.

Causal3DIdent cố ý chứa statistical/causal dependence giữa các ground-truth
factors. Trên dataset đó, `I(z_s;shape)>0` có thể phản ánh data-generating
process thay vì encoder leakage. Nó phù hợp cho một nghiên cứu sau về sampling
contract dưới dependent factors, nhưng không phải clean test cho paper hiện
tại.

### 8.3. Model/data changes

Cần thêm Shapes3D loader đọc HDF5 và trả:

```text
image, shape_label, factor_vector
```

Split đầu tiên là IID full-factor split. Không trộn compositional holdout vào
experiment này; giữ unseen-combination generalization thành scope riêng.

F-CS-WAE cần:

- `n_classes=4`;
- native 64x64 encoder/decoder support;
- không dựa vào decoder 32x32 rồi interpolate lên 64x64;
- manifest ghi factor cardinalities và split hash.

Encoder hiện giả định feature map có spatial size cố định, còn decoder có ba
upsampling blocks cho output 32x32. Cần adaptive encoder pooling hoặc dynamic
feature sizing, và thêm native 64x64 decoder block.

### 8.4. Factor audit

Tạo factor-loading matrix:

| Latent view | Shape | Object hue | Floor hue | Wall hue | Scale | Orientation |
|---|---:|---:|---:|---:|---:|---:|
| semantic `z_c` | expected high | expected low | expected low | expected low | expected low | expected low |
| style `z_s` | expected low | expected high | expected high | expected high | expected high | expected high |

Mỗi ô dùng held-out predictor và calibrated dependence test:

- accuracy/macro-F1 hoặc cross-entropy lower bound cho discrete factors;
- HSIC với multiple-testing correction;
- orientation thêm circular sin/cos regression hoặc circular error;
- posterior sample là primary sampling-contract view;
- posterior mean được báo riêng ở appendix.

Với sáu factor tests, dùng Holm correction hoặc một family-wise procedure đã
chốt trước. Không diễn giải raw p-values riêng lẻ sau khi xem kết quả.

### 8.5. Decoder-level test

Factor-loading matrix chỉ chứng minh information có trong latent. Cần thêm
intervention:

- giữ `z_c`, thay `z_s`: shape phải giữ, style factors được phép đổi;
- giữ `z_s`, thay `z_c`: shape phải đổi, style factors nên giữ.

Train một independent multi-head pixel classifier trên real training images
để chấm shape/color/scale/orientation của decoded images. Evaluator này không
được tham gia generative training.

Main paper dùng một compact heatmap và một swap grid nhỏ; full confusion,
regression errors và factor-wise HSIC để appendix.

### 8.6. Ảnh hưởng đến mục tiêu paper

Experiment này **mở rộng khái niệm content/style vượt ra ngoài class label**.
Nếu factor matrix và interventions cho thấy systematic cross-loading, paper có
thể nói về representation factorization trên known generative factors. Nếu
chỉ class leak nhưng các factor còn lại tách tốt, conclusion phải nói failure
chủ yếu nằm ở designated semantic identity, không phải toàn bộ factorization.

Không được dùng Shapes3D để ngầm thay đổi theorem sang giả định mọi
ground-truth factors độc lập trên dữ liệu tự nhiên. Shapes3D là một controlled
setting nơi independence có thể đạt được, không phải mô hình của mọi dataset.

## 9. Run scope tối thiểu

Để tránh mở rộng không kiểm soát, scope được khóa như sau:

| Experiment | Minimum credible run set |
|---|---|
| Cross-model | 3 native families x 3 seeds trên một common digit setup |
| Entropy | 5 predefined levels trên MNIST seed 0; chỉ xác nhận selected levels trên CIFAR-10 nếu cần |
| Exact marginal | 7 `kappa` levels x khoảng 50 CPU replications |
| Shapes3D | 1 predefined F-CS-WAE configuration; ưu tiên 3 seeds nếu compute cho phép |

Không thêm:

- một họ VAE thường với latent split tùy ý;
- cả MUNIT và DRIT trong main table;
- nhiều conditional-MMD remedy variants;
- combinatorial Shapes3D holdout;
- Causal3DIdent/MPI3D trong cùng submission vòng này;
- hyperparameter sweep nhằm chọn checkpoint đẹp nhất sau khi xem audit.

## 10. Thứ tự triển khai

### Phase 0 — Protocol và test infrastructure

1. Khôi phục environment có PyTorch và chạy test suite.
2. Tạo factorized audit adapter.
3. Gom style posterior sampling vào một helper.
4. Thêm conditional MMD vào canonical audit output.
5. Thêm entropy và `I_LB` metrics.
6. Bump protocol/schema và cập nhật tài liệu Stage-0.

### Phase 1 — Synthetic exact marginal

1. Implement data generator và replication runner.
2. Xác minh MMD null calibration.
3. Sinh figure và JSON manifest.

Experiment này chạy trước vì rẻ và kiểm tra evaluator trước khi GPU jobs bắt
đầu.

### Phase 2 — Controlled entropy

1. Implement effective variance floor.
2. Chạy five-level MNIST sweep.
3. Audit toàn bộ levels bằng stochastic views.
4. Chỉ sau đó mới quyết định selected CIFAR confirmation levels theo rule đã
   chốt, không chọn level chỉ vì đẹp nhất.

### Phase 3 — Cross-model

1. Tạo common Rotated-MNIST dataset.
2. Tích hợp official-model-faithful DRIT và DIVA implementations.
3. Viết adapters và competence tests.
4. Train fixed seed matrix.
5. Chạy cùng canonical audit và aggregate theo model seed.

### Phase 4 — Shapes3D

1. Thêm loader và split manifests.
2. Nâng model lên native 64x64.
3. Train independent factor evaluator.
4. Train F-CS-WAE fixed configuration.
5. Chạy latent factor matrix và decoder interventions.

### Phase 5 — Paper integration

Chỉ sửa abstract/title/conclusion sau khi bốn result packages đã khóa hash.
Không viết trước claim rằng phenomenon generalizes.

## 11. Tích hợp vào paper chín trang

Main paper nên giữ:

1. Theorem và four-term audit map.
2. Exact-marginal synthetic figure.
3. Compact cross-model table.
4. Controlled-entropy figure.
5. Shapes3D factor-loading heatmap.
6. Existing Stage-0 headline table.
7. Existing external latent-swap evidence.

Chuyển xuống appendix:

- full probe suite;
- per-class conditional-MMD bar charts;
- six-point conditional remedy sweep;
- full MMD null plots;
- repeated posterior-draw HSIC tables;
- full Shapes3D factor metrics;
- qualitative grids và confusion matrices;
- architecture/training details cho DRIT và DIVA.

Nếu kết quả hỗ trợ cross-model và ground-truth generality, title có thể chuyển
từ một case-study-centric framing sang:

> Marginal Matching Does Not License Factorized Sampling: Auditing
> Conditional Leakage Across Models and Ground-Truth Factors

Nếu kết quả không hỗ trợ generalization, giữ title/phạm vi hiện tại và trình
bày các null results như boundary conditions.

## 12. Expected impact tổng thể

| Evidence mới | Criticism được xử lý | Điều có thể claim thêm | Điều vẫn không được claim |
|---|---|---|---|
| Cross-model | Chỉ là F-CS-WAE bug | Failure xuất hiện qua nhiều native factorized objectives, nếu số liệu hỗ trợ | Mọi factorized model đều fail |
| Entropy sweep | Leakage chỉ do `sigma -> 0` | Leakage tồn tại dưới stochastic posterior, nếu pass gate | Entropy cao tự động đảm bảo factorization |
| Exact synthetic | Proposition quá trừu tượng | Exact marginal equality không certify independence trong finite-sample audit | F-CS-WAE checkpoint đạt exact equality |
| Shapes3D | Class không đại diện content/style | Audit mở rộng đến known factors | Kết quả áp dụng cho dependent natural factors |

Sau bốn experiment, mục tiêu cốt lõi vẫn là kiểm định sampling contract, không
phải đề xuất một disentanglement method mới. Sự thay đổi quan trọng là evidence
chain sẽ đi từ:

```text
theorem -> một F-CS-WAE case study -> decoder consequence
```

thành:

```text
exact construction
    -> cross-model occurrence
    -> controlled stochasticity
    -> ground-truth factor audit
    -> decoder consequence
```

Đây là mở rộng chiều sâu và phạm vi kiểm chứng của cùng một luận điểm ban đầu,
không phải thay paper bằng một benchmark hoặc một model-comparison study.

## 13. Tài liệu và nguồn liên quan

- Current paper: `fcswae/main_v8_iclr2027.pdf`
- Current evidence roadmap: `fcswae/ICLR2027_EXPERIMENT_ROADMAP.md`
- Canonical audit implementation: `src/metrics/audit_protocol.py`
- F-CS-WAE model: `src/models/f_cs_wae.py`
- Current diagnostic entrypoint: `scripts/compute_leakage_diagnostics.py`
- DRIT: <https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Hsin-Ying_Lee_Diverse_Image-to-Image_Translation_ECCV_2018_paper.pdf>
- DIVA: <https://proceedings.mlr.press/v121/ilse20a.html>
- Official DIVA implementation: <https://github.com/AMLab-Amsterdam/DIVA>
- Shapes3D: <https://github.com/google-deepmind/3d-shapes>
- Causal3DIdent paper: <https://proceedings.neurips.cc/paper/2021/hash/8929c70f8d710e412d38da624b21c3c8-Abstract.html>
