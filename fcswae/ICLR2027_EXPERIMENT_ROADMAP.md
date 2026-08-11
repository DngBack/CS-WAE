# ICLR 2027 evidence audit and experiment roadmap

Updated: 2026-08-11

## Executive decision

The paper is now materially stronger than the previous PDF, because four
cheap reviewer-facing gaps have been resolved from existing
checkpoints:

1. Independent-evaluator latent swaps now show that the decoder actively uses
   class information in the style code.
2. Conditional HSIC on stochastic `(z_c, z_s)` rejects within-class
   independence across ten posterior draws per checkpoint.
3. Global MMD now has a finite-sample permutation calibration.
4. Gen-ACC has been rerun with 1,000 images per class and a style distribution
   fitted only on training data.

The largest remaining reject risk is the posterior-variance-collapse
criticism. A reviewer can still argue that leakage is amplified by an almost
deterministic style posterior. A controlled posterior-noise/entropy retraining
experiment is therefore the next experiment that should consume GPU time.

The second remaining risk is statistical: the conditional-MMD remedy and its
non-monotone trade-off are still single-seed. The third is endpoint validity:
class accuracy improves under conditional style sampling, but quality and
diversity have not yet been measured on the same generated pools.

## Important protocol corrections discovered during this audit

### Independent RNG streams for MMD

The previous Stage-0 implementation seeded the posterior draw and the
Gaussian reference with the same CPU RNG seed. This could couple the two MMD
samples. The evaluator now uses separate streams:

- posterior draw: `seed`;
- Gaussian reference: `seed + 10000`;
- MMD permutation null: `seed + 20000`.

The audit version is bumped from `stage0-1.0.0` to `stage0-1.1.0`. The change
does not reverse the conclusion, but old Stage-0 JSON files must not be mixed
with the corrected artifacts. Corrected global MMD values are:

| Dataset/checkpoint | Corrected global MMD² | Null 95% quantile | Equality-test p |
|---|---:|---:|---:|
| MNIST, seed 0 | 0.000579 | 0.0000245 | 0.001996 |
| Fashion-MNIST, seed 0 | 0.000336 | 0.0000225 | 0.001996 |
| CIFAR-10, seed 0 | 0.000248 | 0.0000188 | 0.001996 |
| CIFAR-10, seed 1 | 0.000353 | 0.0000219 | 0.001996 |
| CIFAR-10, seed 2 | 0.000291 | 0.0000205 | 0.001996 |

CIFAR-10 is `0.000298 ± 0.000052` over model seeds. Exact marginal equality
is rejected at all five checkpoints. Therefore the manuscript must say
“numerically small global discrepancy” or “conditional MMD is 23–28 times
larger,” not “the aggregate posterior is statistically indistinguishable from
the prior.” A small p-value is compatible with a small effect at `n=2048`.

### Training-only conditional style bank

The old Stage-0 Gen-ACC helper estimated class-conditional style statistics
from the evaluation subset, while the manuscript said “encoded training
styles.” The new generation-only diagnostic fits the style bank on 12,000
training examples and evaluates generated pixels with the independent
classifier. These new results replace the 50-images-per-class values.

### Canonical conditional-MMD subset

The archived conditional-MMD figures were generated with an older 5,000-
sample command although the manuscript described the canonical 2,048-example
subset. They have now been regenerated at 2,048 examples with seeded
posterior samples and independent prior streams. The corrected seed-0 rows
are:

| Dataset | δ | Global MMD | Mean conditional MMD | Ratio | Delta_inter |
|---|---:|---:|---:|---:|---:|
| MNIST | 0 | 0.000579 | 0.016024 | 27.68 | 6.264 |
| MNIST | 1 | 0.000812 | 0.001992 | 2.45 | 1.461 |
| CIFAR-10 | 0 | 0.000248 | 0.005789 | 23.30 | 3.801 |
| CIFAR-10 | 1 | 0.000422 | 0.000784 | 1.86 | 1.336 |

The intervention still acts in the predicted direction, and more strongly
than the stale table suggested: conditional MMD falls by 87.6% on MNIST and
86.5% on CIFAR-10.

## Evidence matrix

| Rank | Reviewer question | Current status | Decision |
|---|---|---|---|
| A1 | Is leakage only an artifact of variance collapse? | Open; current posterior std is about 0.00674 | Retrain next; main-paper figure |
| A2 | Does the decoder use leaked style identity? | Resolved for seed-0 MNIST/CIFAR-10 with independent classifiers | Put in main paper now; multi-seed after remedy runs |
| A3 | Does term 4 persist on stochastic variables? | Resolved on all baseline checkpoints and seed-0 `δ=1`, ten draws each | Main text summary; full draws in appendix |
| A4 | What does “small MMD” mean? | Resolved; exact equality is rejected, magnitude is still much below conditional MMD | Correct wording; null plot in appendix |
| B1 | Is conditional-MMD behavior stable across training seeds? | Open for `δ=1,3`; `δ=0` CIFAR seeds already exist | Train selected settings |
| B2 | Is Gen-ACC based on enough images and train-only fitting? | Resolved for unrepaired checkpoints | Replace current table |
| B3 | Did class accuracy improve through mode collapse? | Open | Run quality/diversity on exactly the same sampling strategies |
| C1 | Does the failure recur on controlled factors? | Open | Shapes3D after A/B |
| C2 | Does it recur in another native split-latent model? | Open | Do after Shapes3D or if implementation is cheaper |
| D1 | Does jointly fixing terms 2 and 4 improve sampling? | Open, high-risk research extension | Only after the above evidence is stable |

## New result 1: independent external latent swap

Protocol: deterministic mean swaps
`Dec(mu_c(class a), mu_s(class b))`, 500 draws for each ordered class pair,
an independent pixel classifier, and 95% percentile intervals bootstrapped
over the 90 off-diagonal class pairs.

| Dataset | Objective | Follows semantic donor | Follows style donor | Neither |
|---|---|---:|---:|---:|
| MNIST | `δ=0` | 0.0006 [0.0002, 0.0011] | **0.9928 [0.9916, 0.9940]** | 0.0066 [0.0055, 0.0077] |
| MNIST | `δ=1` | 0.5100 [0.4799, 0.5394] | 0.2337 [0.2037, 0.2640] | 0.2563 [0.2280, 0.2856] |
| CIFAR-10 | `δ=0` | 0.0472 [0.0336, 0.0632] | **0.6392 [0.6132, 0.6654]** | 0.3136 [0.2860, 0.3413] |
| CIFAR-10 | `δ=1` | 0.1874 [0.1642, 0.2113] | 0.3592 [0.3242, 0.3963] | 0.4534 [0.4138, 0.4907] |

This closes the missing causal chain for the audited seed-0 checkpoints:
style contains label information, and replacing style changes decoded class
identity. The `δ=1` comparison is a mechanism check, not yet a statistically
stable treatment effect because it has one model seed.

Main-paper figure:
`figures/external_swap_summary.png`. It should replace the weaker qualitative
MNIST generation grid if the nine-page limit is tight. Full heatmaps and swap
grids belong in the appendix.

## New result 2: stochastic term-4 conditional HSIC

Protocol: classwise conditional HSIC between sampled `(z_c, z_s)`, within-
class permutation calibration, 2,048 held-out examples, 200 permutations, and
ten posterior draws per checkpoint.

| Dataset/checkpoint | Mean-view statistic | Sample statistic across draws | Result |
|---|---:|---:|---|
| MNIST `δ=0`, seed 0 | 0.002440 | 0.002682 ± 0.000005 | all 10 draws p=1/201 |
| Fashion-MNIST `δ=0`, seed 0 | 0.002459 | 0.002459 ± <0.000001 | all 10 draws p=1/201 |
| CIFAR-10 `δ=0`, seed 0 | 0.001924 | 0.002250 ± 0.000004 | all 10 draws p=1/201 |
| CIFAR-10 `δ=0`, seed 1 | 0.001996 | 0.002292 ± 0.000002 | all 10 draws p=1/201 |
| CIFAR-10 `δ=0`, seed 2 | 0.002338 | 0.002338 ± <0.000001 | all 10 draws p=1/201 |
| MNIST `δ=1`, seed 0 | 0.002507 | 0.002507 ± <0.000001 | all 10 draws p=1/201 |
| CIFAR-10 `δ=1`, seed 0 | 0.002283 | 0.002283 ± <0.000001 | all 10 draws p=1/201 |

The key scientific point is not the raw statistic across datasets; kernel
HSIC magnitudes are not directly comparable without their null. The robust
claim is that every stochastic draw rejects conditional independence, and
the term-2 remedy `δ=1` does not remove term 4. This is a clean empirical
illustration of why the four theorem terms cannot substitute for each other.

## New result 3: large-sample external Gen-ACC

Each strategy generates 1,000 images per requested class. The conditional
diagonal Gaussian is fitted on a 12,000-example training-only style bank.

| Dataset | Global Gaussian style | Conditional diagonal style, t=0.25 |
|---|---:|---:|
| MNIST, seed 0 | 0.1047 [0.0989, 0.1109] | 1.0000 [0.9996, 1.0000] |
| CIFAR-10, 3 model seeds | 0.1283 ± 0.0032 | 0.4716 ± 0.0686 |

Brackets for MNIST are 95% Wilson intervals over 10,000 generated images;
CIFAR-10 uncertainty is sample standard deviation over model seeds. Per-seed
CIFAR-10 conditional scores are 0.4016, 0.5386, and 0.4747. The large-sample
result confirms the original direction and shows that Monte Carlo uncertainty
was not driving it.

Do not present `1.0000` as a complete success. It may reflect reduced
diversity, which is why the quality/diversity run below is required.

## Ranked run plan

### Gate 0 — refresh affected artifacts before editing final numbers

Cost: inference only; do immediately.

1. **Pending:** regenerate the five complete Stage-0 JSON files with protocol
   `1.1.0`.
2. **Done:** regenerate conditional-MMD baseline/remedy artifacts with the
   independent global-reference RNG and canonical 2,048-example subset.
3. **Done:** replace the old 50/class Gen-ACC values with the training-only
   1,000/class results.
4. **Done:** insert the external-swap figure and stochastic term-4 result.

Acceptance criterion: every MMD headline must point to a `stage0-1.1.0`
manifest and must not mix old and new RNG streams. Non-MMD diagnostics run
before the version bump remain numerically valid, but should be rerun or
explicitly marked as algorithmically unaffected before packaging the final
supplement.

### Tier A — controlled stochastic-posterior retraining

Impact: highest. Estimated cost from archived runs is roughly 4–9 GPU-hours
per 300-epoch job; two A30s can run two jobs concurrently.

Recommended intervention: use an explicit effective posterior standard-
deviation floor during both training and evaluation,

`sigma_eff = sqrt(exp(clamp(logvar, -10, 10)) + sigma_floor^2)`.

This is preferable for the causal ablation to tuning an entropy-loss weight:
it gives a known stochasticity level and does not let the optimizer satisfy a
weak average-entropy penalty with only a few dimensions. It must be described
as a controlled posterior-noise floor, not as evidence that the raw
log-variance head stopped collapsing.

Pilot on MNIST seed 0:

| Level | `sigma_floor` | Approx. noise norm in 128 dimensions |
|---|---:|---:|
| Current | 0 | 0.076 from the effective clamp |
| Low | 0.05 | 0.57 |
| Medium | 0.15 | 1.70 |
| High | 0.35 | 3.96 |

For every level report:

- median/effective posterior std and log-variance floor-hit rate;
- reconstruction SSIM/LPIPS and FID;
- corrected global and conditional MMD;
- sampled logistic and RBF probe accuracy;
- sampled label HSIC and stochastic conditional HSIC;
- external global-prior Gen-ACC.

Advance at least two nonzero levels to CIFAR-10. Use three CIFAR-10 model
seeds for the selected levels. The experiment succeeds scientifically if at
least one clearly stochastic level retains label recovery above chance with
a held-out confidence interval and calibrated HSIC rejection. Leakage need
not remain near 100%. If leakage disappears, that is also publishable: the
paper must then conclude that variance collapse is the empirical mechanism
for this implementation, while the theorem remains a general audit result.

Recommended figure: x-axis is measured median `sigma_eff`, not the requested
floor. Panel (a) shows sampled probe accuracy and chance; panel (b) shows
global and conditional MMD; optionally panel (c) shows Gen-ACC/FID. This
should be Figure 2 or Figure 3 in the main paper.

### Tier B1 — selected multi-seed conditional-MMD settings

Use CIFAR-10 `δ in {0,1,3}`. The three `δ=0` checkpoints already exist.
Train only seeds 1 and 2 for `δ=1` and `δ=3`: four new jobs, approximately
23 GPU-hours from archived wall times, or about two waves on two A30s.

Primary outcomes: conditional MMD, `Delta_inter`, sampled probe accuracy,
ACC, FID, stochastic term-4 HSIC, and external swap. Report model-seed mean
and sample standard deviation. Do not select a preferred `δ` after looking at
the final metric; frame these as moderate and strong preselected settings.

### Tier B2 — quality and diversity on the generation comparison

Use exactly the generated pools underlying Gen-ACC, with at least 10,000
images per strategy per checkpoint. On CIFAR-10 report:

- FID and preferably KID;
- improved precision/recall or density/coverage in a fixed feature space;
- within-requested-class LPIPS or k-NN feature distance;
- Gen-ACC on the same images.

The key plot is a two-dimensional trade-off, not separate favorable tables:
class fidelity on one axis and diversity/coverage on the other. If the
conditional sampler gains class accuracy while losing recall or intra-class
diversity, state that explicitly. For MNIST, standard ImageNet Inception FID
is hard to interpret; either label it as a consistency metric only or use a
fixed real-image classifier feature space and name the metric accordingly.

### Tier C — controlled-factor generality

Prefer Shapes3D over CelebA. Use `shape` as the designated semantic variable
and treat scale, orientation, and colors as known style factors. Start with an
IID full-factor audit so independence is a property the model can attain;
then add a separately labelled compositional holdout if time permits. Do not
mix the two settings because holding out combinations deliberately introduces
factor correlations in the training distribution.

Required outputs are the same compact audit: global fit, conditional leakage,
stochastic term-4 HSIC, known-factor probes, swaps, and generation on requested
factor combinations. This experiment gives more generality per unit effort
than adding another uncontrolled natural-image label dataset.

### Tier C2 — second native split-latent model

Only start this after the controlled-factor experiment or when a compatible
checkpoint and decoder interface are available. The model must natively emit
separate content/style variables; splitting a vanilla VAE vector in half is
not adequate evidence. Predefine adapters that expose posterior means,
posterior samples, priors, and a decoder, then run the identical audit without
tuning the diagnostic to the result.

### Tier D — joint term-2 and term-4 intervention

This can become the strongest mechanistic experiment but has the highest
risk. Compare base, term-2-only, term-4-only, and combined objectives with
the same seed set. A minibatch conditional-HSIC penalty is computationally
expensive and estimator-sensitive; validate it on a toy distribution before
spending full training runs. This experiment should not delay the entropy
ablation, selected multi-seed runs, or diversity evaluation.

## What belongs in the nine-page main paper

Keep:

1. The decomposition and four-term audit map.
2. Corrected stochastic Stage-0 table.
3. One compact posterior-noise-floor figure.
4. The independent external-swap summary figure.
5. A compact global-versus-conditional MMD table with wording that exact
   marginal equality is rejected.
6. Large-sample Gen-ACC paired with one diversity-sensitive metric.
7. One generality result if Shapes3D or a second model is completed.

Move to appendix:

- full probe suite and per-seed values;
- all ten posterior-draw HSIC values and null summaries;
- MMD null histograms;
- full swap heatmaps/grids;
- all six single-seed delta points;
- per-class confusion matrices and Wilson intervals;
- architecture, schedules, and manifests.

If space is tight, remove the current qualitative MNIST generation grid
before removing the external-swap figure. The swap is a controlled
intervention and carries more evidential weight.

## Commands for completed diagnostics

External swap (change dataset/checkpoint paths for the other runs):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/latent_swap_diagnostics.py \
  --checkpoint runs_f/mnist/seed_0/f_cs_wae_model.pth \
  --dataset mnist --tag mnist_delta0_external_n500 --device cuda \
  --n-per-pair 500 --seed 0 \
  --external-classifier-checkpoint external_classifiers/mnist/grayscale_cnn_4conv_seed0.pth
```

Repeated stochastic conditional HSIC:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/posterior_draw_hsic_diagnostics.py \
  --checkpoint runs_f/cifar10/seed_0/f_cs_wae_model.pth \
  --dataset cifar10 --tag cifar10_delta0_seed0 --device cuda \
  --draws 10 --hsic-permutations 200 --seed 0
```

Global MMD null calibration:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/global_mmd_null_diagnostics.py \
  --checkpoint runs_f/cifar10/seed_0/f_cs_wae_model.pth \
  --dataset cifar10 --tag cifar10_delta0_seed0 --device cuda \
  --n-permutations 500 --seed 0
```

Large-sample Gen-ACC with a training-only style bank:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/generation_accuracy_diagnostics.py \
  --checkpoint runs_f/cifar10/seed_0/f_cs_wae_model.pth \
  --external-classifier-checkpoint external_classifiers/cifar10/wide_resnet_28_10_seed0.pth \
  --dataset cifar10 --tag cifar10_delta0_modelseed0_n1000 --device cuda \
  --n-per-class 1000 --style-bank-size 12000 --seed 0
```

## Stop/go rule

Do not begin a second-model implementation yet. First finish, in order:

1. refresh protocol-1.1 artifacts and manuscript numbers;
2. posterior-noise-floor pilot and selected CIFAR seeds;
3. selected `δ` multi-seed runs;
4. quality/diversity on the existing generation comparison.

After these four items, choose exactly one generality expansion: Shapes3D or
a native split-latent model. Shapes3D is the default recommendation because
it directly addresses the current limitation that class labels are an
imperfect proxy for semantic content.

## Artifact index

- External swap runs: `runs_diag/latent_swap/*_external_n500/`
- External swap aggregate: `runs_diag/latent_swap/external_swap_summary.json`
- Main-paper swap figure: `fcswae/figures/external_swap_summary.png`
- Stochastic term-4 runs: `runs_diag/posterior_draw_hsic/`
- Corrected MMD null runs: `runs_diag/mmd_null/`
- Large-sample generation runs: `runs_diag/generation_accuracy/`
- Sample/mean audit: `runs_diag/stage0/sample_vs_mean_report.md`
