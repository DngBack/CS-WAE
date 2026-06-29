# AAAI-27 Submission Checklist for F-CS-WAE

## Must-Have Experiments

- Run F-CS-WAE on MNIST with seeds `0 1 2`.
- Run F-CS-WAE on Fashion-MNIST with seeds `0 1 2`.
- Keep CIFAR-10 three-seed results, but regenerate final figures after any method change.
- Re-run CS-WAE/simple single-latent baseline under the same evaluation protocol used for F-CS-WAE.
- Re-run VAE, WAE-MMD, VaDE, ResNetAE, and AE+CE/SupCon baselines with matched seeds if compute allows.

## Method Fixes Needed for a Strong AAAI Story

- Implement class-conditional style prior:
  - `p(z_s | y=k) = N(mu_s_k, diag(sigma_s_k^2))`, estimated by EMA or learned parameters.
  - Update sampling code so class-conditional generation uses the conditional style prior.
- Implement per-class style MMD:
  - `mean_k MMD(q(z_s | y=k), N(0,I))`.
  - Compare against current global-only style MMD.
- Optional but valuable:
  - Gradient-reversal classifier on `z_s` to remove label information.
  - HSIC penalty between `z_s` and labels or between `z_s` and `z_c`.

## Ablations

Minimum ablation table:

- Full F-CS-WAE.
- No factorization / single-latent CS-WAE.
- No semantic class MMD.
- No aggregate semantic MMD.
- No style MMD.
- No auxiliary semantic classifier.
- Conditional style prior only.
- Per-class style MMD only.
- Conditional style prior + per-class style MMD.

Style dimension sweep:

- `style_dim = 0, 16, 32, 64, 128`.

Prior structure sweep:

- `n_centers = 1, 2, 4`.
- `rho_prior = 0.3, 0.5, 0.7, 0.9`.

## Figures Needed

- Main pipeline diagram:
  - input -> encoder -> semantic spherical head + style Gaussian head -> decoder.
  - show class-conditional Spherical Cauchy priors and Gaussian/conditional style prior.
- CIFAR-10 main metrics plot.
- Prior samples:
  - CS-WAE/simple baseline.
  - current F-CS-WAE.
  - fixed F-CS-WAE with conditional style alignment.
- Latent UMAP:
  - semantic centers and encoded test data.
- Style diagnostic:
  - `z_s ~ N(0,I)`.
  - `z_s` empirical same-class posterior.
  - conditional style prior sample.
- Ablation trade-off plot:
  - ACC vs FID.

## Writing Tasks

- Replace generic LaTeX preamble with official AAAI-27 style when released.
- Tighten paper to AAAI page limit after experiments are complete.
- Remove appendix checklist from submission version.
- Move detailed failure analysis to either main paper or appendix depending on page budget.
- Ensure all tables report matched protocol and clearly label seed count.

## Submission Claim Target

Weak claim to avoid:

> F-CS-WAE improves clustering.

Strong claim to target:

> Factorized Spherical Cauchy WAE improves label-guided generative clustering, and conditional style alignment resolves the prior-sampling failure mode caused by marginal-only style matching.
