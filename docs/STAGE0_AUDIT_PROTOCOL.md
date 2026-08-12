# Stage-0 Audit Protocol

Protocol version: `stage0-1.2.0`.

This document is authoritative for new audit runs. Historical files under
`runs_diag/` predate Stage 0 and must not be pooled with Stage-0 results
without being regenerated from the original checkpoints.

## 1. Named latent views

Every result names the latent view explicitly.

| Name | Definition | Primary use |
|---|---|---|
| `z_s_sample` | `mu_s + sqrt(exp(clamp(logvar_s,-10,10)) + sigma_floor^2) * epsilon` | Distributional sampling contract: global/conditional MMD and secondary sample probe |
| `mu_s` | Deterministic style posterior mean | Representation probes, deterministic latent swaps and style statistics |
| `mu_c_mu_s` | Pair of deterministic posterior means | JointMMD term-4 proxy |

No table may label a metric only as “on `z_s`” when the implementation uses
`mu_s`. Training, audit and repeated-draw diagnostics call the same posterior
sampler. Posterior sampling uses a seeded CPU generator recorded in the
manifest, so its draws do not depend on accelerator RNG state.

The configured style standard-deviation floor and measured effective
posterior entropy are stored with every new F-CS-WAE audit. A zero floor is
exactly backward-compatible with historical checkpoints.

## 2. Evaluation population

- Default evaluation population: a fixed, stratified-agnostic random subset
  of 2,048 examples from the official test split.
- The subset seed and SHA-256 of the ordered indices are stored in every
  result manifest.
- A table using another sample size must state it in its caption and must not
  silently pool results with the default population.
- HSIC uses at most 1,024 examples because its kernel matrix is quadratic;
  the exact count is stored in the result.

## 3. Probes

All probes share one deterministic stratified 60/20/20
train/validation/test split. Feature mean and standard deviation are fitted
on the probe training partition only. The test partition is never used for
early stopping or hyperparameter selection.

Fixed probes:

- Logistic: one linear layer, Adam, LR `1e-2`, weight decay `1e-4`.
- MLP: hidden dimensions 128 and 64 with ReLU, same optimizer.
- RBF-SVM: `C=1`, `gamma=scale`.
- k-NN: `k=5`, distance weighting.

Validation loss selects the Torch probe checkpoint. Main tables report test
accuracy, macro-F1 and the cross-entropy lower bound
`H_test(y) - CE_test(y|z)`. Probe train and validation metrics remain in JSON
for overfit diagnosis.

## 4. Kernel estimators

Training and evaluation deliberately use different estimators:

- Training MMD remains the non-negative biased V-statistic because it is a
  minibatch optimization loss.
- Evaluation MMD is the unbiased U-statistic and excludes within-sample
  diagonals. Negative finite-sample estimates are retained, not clipped.
- Euclidean kernels average RBF bandwidths
  `{0.5, 1, 2, 5, 10, 20, 50}`.
- Global MMD compares the full stochastic style sample with an independently
  seeded `N(0,I)` reference. Conditional MMD repeats that U-statistic per class
  with independent references and reports the unweighted class mean.
- JointMMD uses a spherical bandwidth ladder
  `{0.1, 0.3, 0.5, 1, 2}` for `mu_c` and the Euclidean ladder for `mu_s`.

HSIC uses the same multi-scale Euclidean kernel and a delta label kernel.
Its biased centered statistic is never interpreted without a plus-one
permutation p-value. Default calibration uses 200 permutations.

Term 4 is cross-checked with classwise conditional HSIC on `(mu_c, mu_s)`.
The semantic block uses the spherical kernel, the standardized style block
uses the Euclidean kernel, and the null independently permutes style codes
within each class. JointMMD and conditional HSIC are reported together; one
is not silently substituted for the other.

## 5. External generated-class evaluator

Primary Gen-ACC is measured directly from generated pixels by an independent
classifier trained only on real images:

- MNIST/Fashion-MNIST: fixed four-convolution CNN, 20 epochs, Adam, no
  augmentation.
- CIFAR-10: WideResNet-28-10, 200 epochs, SGD momentum 0.9, cosine LR,
  crop/flip augmentation on its training partition only. CIFAR channel
  normalization is embedded in the classifier so callers always pass pixels
  in `[0,1]`, including generated images.

The official training set is split 90/10 for classifier training and model
selection. The official test split is used once to report real-test
accuracy. Every checkpoint stores architecture, preprocessing, split sizes,
best validation accuracy and real-test accuracy.

Train an evaluator:

```bash
python3 scripts/train_external_classifier.py \
  --dataset mnist --device cuda:0
```

Internal F-CS-WAE re-encoding scores are retained only under
`generation.internal_robustness_only`; they are not primary Gen-ACC.

## 6. Result provenance

Every Stage-0 diagnostic result is a self-describing JSON containing:

- protocol/schema version;
- dataset and all seeds;
- model checkpoint absolute path and SHA-256;
- external-classifier path and SHA-256, when used;
- run/model configuration;
- evaluation population, latent definitions and estimator settings;
- repository commit and dirty-worktree flag;
- Python, PyTorch and CUDA versions;
- nested metrics with explicit latent-view names.

Each JSON has a `.sha256` sidecar. Example:

```bash
python3 scripts/compute_leakage_diagnostics.py \
  --checkpoint runs_f/mnist/seed_0/f_cs_wae_model.pth \
  --dataset mnist --device cuda:0 \
  --external-classifier-checkpoint \
    external_classifiers/mnist/grayscale_cnn_4conv_seed0.pth
```

The default output is
`runs_diag/stage0/<dataset>/<checkpoint-stem>_seed<eval-seed>.json`.

## 7. Compatibility fields

New JSON files retain the flat keys `global_mmd`, `delta_inter`,
`lp_accuracy`, `hsic`, and `joint_mmd` so old plotters can load them. These
are compatibility aliases only. New paper tables must read the nested,
explicitly named fields.

## 8. Version compatibility

`stage0-1.2.0` changes the random stream consumed by the canonical F-CS-WAE
adapter because it samples both named content and style views, and it adds the
effective entropy, stochastic probe information lower bound and conditional
MMD fields. Therefore values produced by protocol 1.2 must not be averaged
with 1.0/1.1 values. Historical paper evidence remains tagged with its
original protocol until it is regenerated.
