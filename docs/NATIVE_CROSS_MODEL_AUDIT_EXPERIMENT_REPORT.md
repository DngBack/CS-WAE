# Báo cáo native cross-model audit: F-CS-WAE, DRIT và DIVA

**Trạng thái:** Hoàn tất full matrix 3 families x 3 seeds  
**Ngày hoàn tất:** 2026-08-12  
**Cross-model protocol:** `native-cross-model-1.0.0`  
**Stage-0 audit protocol:** `stage0-1.2.0`  
**Dataset:** two-domain Rotated-MNIST  
**Model seeds:** `0, 1, 2`  
**Summary SHA-256:** `2a4f1d3f81f1273fa8f8a01eeebb213f193bae49e9d0607ed4250a55009474ab`

## 1. Kết luận ngắn

Full cross-model experiment đã hoàn tất đủ chín runs, không sử dụng proxy
artifact cũ. Kết quả không cho thấy cùng một phenomenon trên cả ba families.
Thay vào đó, unified audit phân biệt ổn định ba chế độ:

1. **F-CS-WAE:** conditional label leakage rất mạnh, ổn định qua ba seeds,
   trong khi global discrepancy nhỏ hơn conditional discrepancy gần một bậc
   độ lớn;
2. **DRIT:** style code chứa label information, nhưng aggregate posterior đã
   cách rất xa Gaussian prior; đây là wholesale prior-sampler failure, không
   phải marginal-matching paradox;
3. **DIVA:** residual `z_x` gần Gaussian prior, label probes ở chance, content
   và prior-style generation competent; đây là boundary/null result ổn định.

Kết quả vì vậy không hỗ trợ câu:

> Cùng conditional-leakage-under-small-global-discrepancy phenomenon xuất hiện
> ở cả F-CS-WAE, DRIT và DIVA.

Kết quả hỗ trợ câu chính xác hơn:

> A common conditional audit distinguishes qualitatively different outcomes
> across native factorized models: strong conditional leakage in F-CS-WAE,
> wholesale prior mismatch in DRIT, and near-successful factorization in DIVA.

Điều này vẫn tăng giá trị khoa học của paper. DIVA cho thấy audit không phải
một detector luôn trả positive result; DRIT cho thấy chỉ báo failure phải phân
biệt marginal mismatch với conditional leakage; F-CS-WAE cho thấy central
empirical phenomenon lặp lại rất chắc qua training seeds.

## 2. Câu hỏi nghiên cứu

Experiment kiểm tra hai câu hỏi tách biệt:

1. Failure quan sát trên F-CS-WAE có phải chỉ là một seed/model-specific bug?
2. Cùng audit gồm global MMD, conditional MMD, probe và HSIC có phân biệt được
   các native factorization objectives khác nhau hay không?

Sampling contract được audit là:

\[
q(z_s) \stackrel{?}{=} p(z_s),
\qquad
q(z_s\mid y,d) \stackrel{?}{=} p_d(z_s),
\qquad
z_s \perp y\mid d.
\]

Với F-CS-WAE và DIVA, `p_d(z_s)` là cùng standard Gaussian cho cả hai domains.
Với DRIT, attribute space là domain-specific nên mỗi domain phải được audit
riêng. Không pool hai attribute spaces rồi coi chúng là một latent space chung.

## 3. Khác biệt so với proxy experiment cũ

Experiment này không đọc hoặc aggregate bất kỳ artifact nào từ:

```text
runs_diag/cross_model/
```

Legacy script `scripts/run_cross_model_diagnostics.py` dùng generic Gaussian
latents hoặc post-hoc split và không đủ điều kiện cho native cross-model claim.
Protocol mới chỉ đăng ký adapter khi model tự định nghĩa factorized latent:

| Family | Content view | Audited style view | Style prior |
|---|---|---|---|
| F-CS-WAE | hyperspherical semantic `z_c` | Gaussian `z_s` | `N(0,I)` |
| DRIT | shared content after domain-specific stem | domain-specific attribute `z_a` | `N(0,I)` trong từng domain |
| DIVA | class-specific `z_y` | residual `z_x` | `N(0,I)` |

Mỗi audit manifest ghi `legacy_proxy_artifacts_used=false`, native adapter
metadata, checkpoint hash và source-file hashes.

## 4. Model implementations

### 4.1. F-CS-WAE

- semantic dimension: 64;
- style dimension: 32;
- one class center, `rho_prior=0.7`;
- `style_sigma_floor=0`;
- conditional style-MMD remedy tắt: `delta_final=0`;
- giữ global style MMD, supervised semantic structure và auxiliary classifier;
- 100 epochs với phase boundaries được scale về `20/40/70/100`.

### 4.2. DRIT

- content dimension: 64;
- attribute/style dimension: 8;
- domain-specific content stems và Gaussian attribute encoders;
- shared content representation;
- self-reconstruction, cross-cycle, attribute KL, latent regression;
- image adversaries và content adversary.

### 4.3. DIVA

- domain latent `z_d`: 32 dimensions;
- residual/style latent `z_x`: 32 dimensions;
- class-specific/content latent `z_y`: 32 dimensions;
- conditional priors `p(z_d|d)` và `p(z_y|y)`;
- standard Gaussian prior cho `z_x`;
- domain/label auxiliary classifiers.

DRIT và DIVA là native-family common-backbone adaptations cho controlled
28x28 setup, không phải exact reproduction của benchmark architecture gốc.
Các official revisions được dùng để đối chiếu latent definitions và defining
losses:

- DRIT: `f19f50a8fa5f28dffbd93a0ed034da616232d769`;
- DIVA: `4c5282a8e54feee01626f5e8a54595ea570ac169`.

## 5. Dataset và fixed split

Dataset là two-domain Rotated-MNIST:

- domain 0: `-30 degrees`;
- domain 1: `+30 degrees`;
- `y`: digit identity, 10 classes;
- mỗi selected source image xuất hiện đúng một lần ở mỗi domain;
- finite dataset cân bằng chính xác theo digit và domain;
- split seed cố định: 2027.

| Split | Source images/class | Total source images | Paired examples |
|---|---:|---:|---:|
| Train | 2,000 | 20,000 | 40,000 |
| Test | 500 | 5,000 | 10,000 |
| Canonical audit/domain | 100 | 1,000 | 1,000 |

Các families và seeds dùng cùng source-image subsets. Model seed chỉ thay đổi
initialization, posterior draws và DataLoader order, không thay đổi population.

```text
train source-index SHA-256:
ea6ead43cbe8e723fe9ba6d86e59bd71aba06f6ab0d99e4672111177d0ec0219

test source-index SHA-256:
513d4a663c09a89f8a41bb97e8ebf010f594c92f022d39fe8626205f71581d53
```

## 6. Training và evaluation protocol

Mỗi family được train đúng 100 epochs với seeds `{0,1,2}`. Final epoch được
dùng cố định; không chọn checkpoint dựa trên leakage metrics. Checkpoint đầy
đủ model, optimizer, scheduler, RNG và DataLoader-generator state được ghi mỗi
5 epochs để có thể resume tại epoch boundary.

Audit chạy riêng trên mỗi domain rồi báo macro-average hai domains:

- unbiased multiscale global MMD squared U-statistic;
- 500 global-MMD permutations;
- mean per-class conditional MMD đến independent Gaussian references;
- logistic, MLP và RBF-SVM probes trên stochastic style sample;
- stratified 60/20/20 probe split;
- feature standardization fit trên probe-train;
- label HSIC với 200 plus-one calibrated permutations;
- posterior entropy từ effective diagonal variance;
- reconstruction L1;
- content logistic probe;
- digit retention khi giữ content và thay style bằng prior sample.

Global MMD không được so trực tiếp như một normalized score giữa models có
style dimensions khác nhau. Trong-family comparison, absolute discrepancy,
conditional-minus-global gap và probes được đọc cùng nhau.

### 6.1. Independent pixel evaluator

Một Rotated-MNIST CNN độc lập được train 20 epochs chỉ trên real training
images. Nó không tham gia generative training và không nhìn generated images.

| Metric | Kết quả |
|---|---:|
| Real test accuracy | 96.05% |
| Real test cross-entropy | 0.12485 nats |
| Test examples | 10,000 |

Evaluator vượt frozen threshold 95%.

Evaluator checkpoint SHA-256:

```text
4019fcf2a990a9f2c98bf3aa1cef83bc502fc72ef808360443f02c6d887ddaf5
```

## 7. Frozen competence contract

Criteria được ghi trước khi pilot result được đọc:

| Check | Threshold |
|---|---:|
| External real-test accuracy | >=95% |
| Content logistic probe | >=80% |
| Reconstruction L1 | <=0.20 |
| Prior-style content retention | >=70% |

Kết quả được báo theo hai lớp riêng biệt. `Base competence` chỉ gồm external
evaluator, content probe và reconstruction; `frozen all-check` giữ nguyên cả
prior-style retention như protocol gốc:

| Family | Base competence | Frozen all-check gate |
|---|---:|---:|
| F-CS-WAE | 3/3 | 0/3 |
| DRIT | 3/3 | 0/3 |
| DIVA | 3/3 | 3/3 |

F-CS-WAE và DRIT chỉ fail prior-style content-retention check. Cả hai đều pass
external evaluator, content probe và reconstruction checks.

Đây là một design caveat cần báo trung thực: prior-style retention vừa được
dùng làm competence gate, vừa là decoder-level consequence của sampling
contract đang được kiểm tra. Vì criteria đã frozen, không được xóa hoặc đổi
threshold sau khi xem result. Report vì vậy giữ hai cách đọc:

1. **Formal predeclared reading:** chỉ DIVA qua toàn bộ competence gate;
2. **Post-result diagnostic decomposition:** F-CS-WAE và DRIT có competent
   reconstruction/content representation nhưng prior sampler không giữ digit.

Cách đọc thứ hai hữu ích để giải thích failure mechanism, nhưng không được
trình bày như một predeclared gate mới.

## 8. Kết quả tổng hợp

Các giá trị dưới đây là mean +/- sample standard deviation qua ba model seeds.

| Family | Global MMD2 | Conditional MMD2 | Logistic style probe | RBF style probe | Content probe | Rec. L1 | Prior-style retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| F-CS-WAE | 0.003326 +/- 0.000140 | 0.032262 +/- 0.001197 | 91.92 +/- 2.43% | 96.67 +/- 0.95% | 99.42 +/- 0.14% | 0.01510 +/- 0.00036 | 12.37 +/- 1.19% |
| DRIT | 0.255261 +/- 0.008545 | 0.272442 +/- 0.001938 | 34.42 +/- 12.62% | 32.50 +/- 14.12% | 88.25 +/- 1.52% | 0.04284 +/- 0.00211 | 9.87 +/- 0.12% |
| DIVA | -0.0000179 +/- 0.0000274 | 0.000176 +/- 0.000129 | 10.67 +/- 1.04% | 11.83 +/- 0.63% | 96.67 +/- 0.76% | 0.03449 +/- 0.00009 | 89.45 +/- 0.83% |

Chance label accuracy là 10%.

## 9. Per-seed results

### 9.1. F-CS-WAE

| Seed | Global MMD2 | Conditional MMD2 | Cond/global | Logistic | MLP | RBF | HSIC p | Content | Retention |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.003287 | 0.032431 | 9.87 | 94.00% | 93.75% | 97.75% | 0.00498 | 99.25% | 12.55% |
| 1 | 0.003209 | 0.030989 | 9.66 | 92.50% | 91.50% | 96.25% | 0.00498 | 99.50% | 11.10% |
| 2 | 0.003481 | 0.033365 | 9.63 | 89.25% | 92.75% | 96.00% | 0.00498 | 99.50% | 13.45% |

Global MMD exact-equality test rejects in every seed/domain (`p=1/501`). The
checkpoint claim therefore remains “numerically much smaller global than
conditional discrepancy”, not exact marginal match.

HSIC rejects in all six seed-domain evaluations (`p=1/201`). Conditional MMD
is between 9.63 and 9.87 times the positive global statistic. All probe
families recover digit identity from stochastic `z_s` with high accuracy.

Posterior entropy is approximately `-3.58 nats/dimension` in all seed-domain
runs. Thus this particular cross-model checkpoint family still has a
near-deterministic posterior. The separate controlled-entropy experiment is
needed for the claim that leakage persists at higher effective entropy.

### 9.2. DRIT

| Seed | Global MMD2 | Conditional MMD2 | Cond/global | Logistic | MLP | RBF | HSIC p | Content | Retention |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.259014 | 0.272479 | 1.05 | 25.00% | 24.50% | 25.50% | 0.00498 | 87.50% | 9.80% |
| 1 | 0.261287 | 0.274362 | 1.05 | 29.50% | 26.50% | 23.25% | 0.00498 | 90.00% | 10.00% |
| 2 | 0.245482 | 0.270487 | 1.10 | 48.75% | 51.50% | 48.75% | 0.00498 | 87.25% | 9.80% |

DRIT có dependence ổn định theo significance test: HSIC reject ở mọi
seed-domain. Probe effect size thay đổi đáng kể theo seed nhưng luôn cao hơn
chance. Tuy nhiên global MMD rất lớn và conditional/global ratio chỉ khoảng
1.05--1.10. Style posterior entropy cũng thấp, khoảng `-1.57` đến
`-1.88 nats/dimension`.

Kết luận đúng cho DRIT là:

- native attribute code chứa digit information;
- encoded aggregate attribute posterior không match Gaussian sampling prior;
- prior-style translations không giữ content;
- result không cô lập conditional leakage dưới một matched marginal.

### 9.3. DIVA

| Seed | Global MMD2 | Conditional MMD2 | Logistic | MLP | RBF | Raw HSIC p (d0, d1) | Content | Retention |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000096 | 0.000268 | 11.50% | 10.00% | 12.50% | 0.0746, 0.0398 | 96.00% | 90.25% |
| 1 | -0.0000452 | 0.0000288 | 11.00% | 8.75% | 11.25% | 0.2687, 0.0249 | 97.50% | 89.50% |
| 2 | -0.0000181 | 0.000231 | 9.50% | 10.00% | 11.75% | 0.1194, 0.7264 | 96.50% | 88.60% |

Derived paper summary không dùng arithmetic mean p-value. Nó giữ sáu raw
domain-level tests trong mỗi family và áp dụng Holm step-down trong family.
Source audit JSON và protocol cũ không bị sửa.

Tất cả six global-MMD domain tests không reject. Probes nằm quanh chance và
ổn định qua seeds. DIVA posterior entropy khoảng `1.36--1.38 nats/dimension`,
khác rõ posterior collapse ở hai families còn lại.

Hai trong sáu raw domain-level HSIC tests có `p<0.05`, nhưng pattern không lặp
theo domain/seed và probes không tăng. Holm correction trên sáu tests không giữ
lại rejection nào. Đây là derived reporting rule, không phải thay đổi protocol
đã frozen; paper phải giữ cả raw values, nói rõ family được correction và không
claim DIVA independence được chứng minh.

Kết luận thận trọng là audit không tìm thấy evidence ổn định cho practically
recoverable digit leakage trong DIVA `z_x` trên setup này.

## 10. Ba failure/success regimes

```text
F-CS-WAE
  small-but-nonzero global discrepancy
  + much larger conditional discrepancy
  + near-perfect label recovery from style
  + prior-style decoder failure

DRIT
  large global discrepancy
  + similarly large conditional discrepancy
  + moderate, seed-variable label leakage
  + prior-style translation failure

DIVA
  global discrepancy around the finite-sample null
  + tiny conditional discrepancy
  + label probes around chance
  + competent prior-style decoding
```

Theorem 1 dự đoán rằng sampler mismatch có nhiều components. Cross-model result
minh họa đúng lợi ích của decomposition: một aggregate metric duy nhất không
phân biệt được conditional leakage, wholesale prior mismatch và near-success.

## 11. Ảnh hưởng đến claims của paper

### 11.1. Claims được hỗ trợ

1. F-CS-WAE leakage là seed-stable trên controlled two-domain setup.
2. Marginal and conditional diagnostics có thể cho kết luận rất khác nhau.
3. Unified audit phân biệt nhiều failure regimes qua native model families.
4. DRIT có sampler failure ở aggregate-prior level và label information trong
   attribute code.
5. DIVA cung cấp một boundary condition nơi native residual factorization gần
   đạt sampling contract trên setup này.
6. Negative control của DIVA làm giảm khả năng kết quả F-CS-WAE chỉ là artifact
   của estimator luôn reject hoặc probe pipeline luôn overfit.

### 11.2. Claims không được hỗ trợ

Không được viết rằng:

- cùng marginal-matching paradox xuất hiện ở cả ba families;
- mọi factorized model đều leak;
- DRIT có good global marginal matching;
- DIVA có meaningful leakage chỉ dựa trên vài raw HSIC p-values;
- global MMD nhỏ chứng minh exact equality;
- cross-model run tự loại trừ posterior collapse như mechanism;
- F-CS-WAE và DRIT qua toàn bộ predeclared competence gate.

### 11.3. Framing đề xuất

Không dùng universal-negative framing. Main-text wording đề xuất:

> Across three native factorized families, the same audit exposes distinct
> regimes rather than a universal failure: F-CS-WAE exhibits strong and
> seed-stable conditional leakage despite a substantially smaller global
> discrepancy; DRIT fails already at aggregate prior matching; and DIVA
> provides a near-successful boundary case with chance-level residual probes.

Cross-model experiment nên được dùng để chứng minh phạm vi và tính phân biệt
của audit framework. Exact logical counterexample vẫn đến từ synthetic
experiment; strongest trained-model conditional phenomenon vẫn là F-CS-WAE.

## 12. Gợi ý tích hợp vào paper

Main paper nên dùng một compact table gồm:

- global MMD2;
- conditional MMD2 hoặc conditional-minus-global gap;
- stochastic style-probe accuracy;
- HSIC result;
- base reconstruction/content competence;
- prior-style content retention;
- explicit competence-gate status.

Không nên đưa raw conditional/global ratio của DIVA làm headline: unbiased
global MMD có thể âm hoặc gần zero, làm ratio undefined/unstable. Dùng
conditional-minus-global gap hoặc báo hai absolute values cạnh nhau.

Main text nên báo mean +/- std qua three seeds. Domain-level results, raw
p-values, posterior entropy và full probe suite chuyển xuống appendix.

## 13. Limitations

1. DRIT/DIVA là common-backbone adaptations, không phải official benchmark
   architectures được reproduce nguyên xi.
2. Chỉ có một controlled dataset và hai rotations.
3. F-CS-WAE/DRIT posterior entropy thấp; cross-model experiment không thay
   controlled-entropy evidence.
4. Conditional MMD hiện là effect-size statistic, chưa có conditional-MMD
   permutation p-value trong runner.
5. Source runner có trường macro-average p-value mang tính mô tả; derived paper
   summary loại trường này khỏi inference, giữ raw domain tests và Holm theo
   family.
6. Frozen competence gate trộn base competence với prior-sampling outcome;
   paper báo hai cột riêng nhưng vẫn giữ nguyên gate gốc.
7. Audit không chứng minh statistical independence của DIVA; failure to reject
   và chance probes chỉ cho biết chưa tìm thấy dependence hữu dụng ở power hiện
   tại.

## 14. Reproducibility log

Full command:

```bash
.venv/bin/python scripts/run_native_cross_model_audit.py \
  --stage full --seeds 0 1 2 --devices cuda:0 cuda:1 \
  --epochs 100 --evaluator-epochs 20 \
  --checkpoint-every 5 \
  --mmd-permutations 500 --hsic-permutations 200 \
  --probe-epochs 300 \
  --output-root runs_cross_model/native_v1 \
  --auto-resume --skip-existing
```

Run được thực hiện dưới user service
`fcswae-crossmodel-native-full.service`. Service kết thúc bình thường, GPU về
idle, và logs không chứa traceback, RuntimeError, NaN hoặc CUDA OOM.

Completion checks:

- đủ `3 families x 3 seeds`;
- mọi training checkpoint ghi `completed_epochs=100`;
- mọi run có final `model.pth`, `training_history.json` và `audit.json`;
- mọi audit JSON SHA-256 sidecar khớp nội dung;
- summary checksum hợp lệ;
- summary manifest ghi seeds `[0,1,2]`;
- summary manifest ghi `legacy_proxy_artifacts_used=false`.

## 15. Artifact index

Artifacts chính:

- [Frozen acceptance criteria](../runs_cross_model/native_v1/acceptance_criteria.json)
- [Independent evaluator result](../runs_cross_model/native_v1/evaluator/rotated_mnist_cnn_seed0.json)
- [Three-seed cross-model summary](../runs_cross_model/native_v1/cross_model_summary.json)
- [Derived paper summary: raw p-values, Holm, separated competence](../runs_cross_model/native_v1/cross_model_paper_summary.json)
- [F-CS-WAE seed 0 audit](../runs_cross_model/native_v1/fcswae/seed_0/audit.json)
- [F-CS-WAE seed 1 audit](../runs_cross_model/native_v1/fcswae/seed_1/audit.json)
- [F-CS-WAE seed 2 audit](../runs_cross_model/native_v1/fcswae/seed_2/audit.json)
- [DRIT seed 0 audit](../runs_cross_model/native_v1/drit/seed_0/audit.json)
- [DRIT seed 1 audit](../runs_cross_model/native_v1/drit/seed_1/audit.json)
- [DRIT seed 2 audit](../runs_cross_model/native_v1/drit/seed_2/audit.json)
- [DIVA seed 0 audit](../runs_cross_model/native_v1/diva/seed_0/audit.json)
- [DIVA seed 1 audit](../runs_cross_model/native_v1/diva/seed_1/audit.json)
- [DIVA seed 2 audit](../runs_cross_model/native_v1/diva/seed_2/audit.json)

Per-run directory còn chứa:

```text
run_config.json
split_manifest.json
training_checkpoint.pt
training_history.json
model.pth
pipeline.log
audit.json
audit.json.sha256
```

## 16. Việc cần làm tiếp theo

Không cần chạy thêm cross-model seeds hoặc tuning DRIT/DIVA cho submission
scope hiện tại. Tuning sau khi xem audit có nguy cơ biến experiment thành
post-hoc search và không sửa được kết luận đã quan sát.

Hai việc reporting không cần retraining đã hoàn tất:

1. derived summary không dùng arithmetic mean p-value cho inference, báo raw
   domain tests và Holm trong từng family;
2. base competence và frozen all-check gate được báo ở hai cột riêng, trong
   khi artifact/criteria `native-cross-model-1.0.0` vẫn nguyên vẹn.

Trong bốn experiment bắt buộc, trạng thái hiện tại là:

| Experiment | Trạng thái |
|---|---|
| Paper-scale exact marginal | Hoàn tất |
| Controlled posterior entropy, MNIST + CIFAR | Hoàn tất |
| Native cross-model, 3 families x 3 seeds | Hoàn tất |
| Shapes3D ground-truth factor audit | Đã triển khai; seed 0 đang chạy |

Shapes3D seed 0 dùng predefined F-CS-WAE configuration và independent factor
evaluator. Chỉ sau khi seed 0 qua competence gate mới cân nhắc seeds 1--2 để
báo uncertainty. Không cần thêm Causal3DIdent, MPI3D hoặc combinatorial holdout
trong submission này.
