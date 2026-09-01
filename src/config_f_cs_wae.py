"""
Configuration for F-CS-WAE (Factorized Class-Structured Spherical Cauchy WAE).

Designed for CIFAR-10 (3-channel 32×32) with a ResNet-18 backbone.
Supports all datasets via apply_dataset_config().
"""


class FCSWAEConfig:
    """Hyperparameters for F-CS-WAE training and architecture."""

    # --- Architecture ---
    semantic_dim: int = 64     # d_c: semantic latent dimension on S^{d_c-1}
    style_dim: int = 128       # d_s: style latent dimension in R^{d_s}
    n_classes: int = 10
    in_channels: int = 3
    image_size: int = 32
    backbone: str = "resnet18"  # default backbone for CIFAR-10

    # --- Prior parameters ---
    rho_prior: float = 0.7     # Spherical Cauchy concentration for class priors
    n_centers: int = 1         # R: centers per class (multi-center when R > 1)
    ema_momentum: float = 0.95  # tau: EMA decay for class center updates
    epsilon: float = 1e-8

    # --- Training ---
    batch_size: int = 128
    total_epochs: int = 300
    lr: float = 1e-3
    lr_scheduler_step: int = 100
    lr_scheduler_gamma: float = 0.5
    grad_clip: float = 1.0
    num_workers: int = 4
    # Debug/smoke-only cap. None always consumes the full loader.
    max_train_batches: int | None = None

    # --- Phase boundaries (epoch numbers where each phase ends) ---
    # Phase A [0,  50): reconstruction only
    # Phase B [50, 100): add style MMD + cls loss
    # Phase C [100,200): add weak class/agg MMD
    # Phase D [200,300): full objective
    phase_a_end: int = 50
    phase_b_end: int = 100
    phase_c_end: int = 200
    phase_d_end: int = 300
    # Optional zero-indexed epoch at which scheduled objective weights stop
    # increasing. Training and FACT dual updates continue normally.
    phase_weight_freeze_epoch: int | None = None

    # --- Reconstruction loss weights ---
    l1_weight: float = 1.0
    lpips_weight: float = 0.1

    # --- Regularization final weights (reached at end of phase D) ---
    alpha_final: float = 2.0   # L_class: supervised class MMD weight
    beta_final: float = 5.0    # L_agg:   global aggregated semantic MMD weight
    gamma_final: float = 1.0   # L_style: global style prior MMD weight
    delta_final: float = 1.0   # L_style_cls: per-class style MMD weight (enforces z_s ⊥ y)
    joint_contract_final: float = 0.0  # Joint q(z_c,z_s|y) vs factorized prior MMD
    # FACT: clause-aligned, null-calibrated factorized-contract training.
    # Disabled by default so historical runs remain bit-for-bit compatible.
    fact_enabled: bool = False
    fact_start_epoch: int = 100
    fact_style_only_dependence: bool = True
    fact_dual_lr: float = 0.1
    fact_dual_init: float = 1.0
    fact_dual_max: float = 10.0
    # Number of independent matched-null estimates averaged per minibatch
    # class. One preserves the historical FACT estimator exactly.
    fact_null_draws: int = 1
    # Raw MMD and HSIC violations live on very different numerical scales.
    # Dual ascent divides by these fixed positive reference magnitudes before
    # updating its multipliers. Defaults of one preserve historical updates.
    fact_dual_reference_style: float = 1.0
    fact_dual_reference_content: float = 1.0
    fact_dual_reference_dependence: float = 1.0
    # Optional cap on the absolute multiplier change from one epoch. None
    # preserves historical dual ascent. Useful when normalized violations are
    # large during the first active epoch.
    fact_dual_step_max: float | None = None
    # Fixed weight for a matched-null RBF/delta HSIC between sampled style and
    # the categorical condition. Zero preserves the repaired FACT objective.
    fact_label_hsic_weight: float = 0.0
    # Alternating nonlinear label adversary on sampled style. The adversary is
    # trained to predict y, while the encoder minimizes KL(U || q_adv(y|z_s)).
    # Zero disables the branch and preserves historical training exactly.
    fact_label_adversary_weight: float = 0.0
    fact_label_adversary_lr: float = 1e-3
    fact_label_adversary_steps: int = 1
    fact_label_adversary_weight_decay: float = 1e-4
    # Audit-aligned classwise HSIC between posterior means (mu_c, mu_s).
    # Content is stop-gradient; zero preserves all previous objectives.
    fact_mean_hsic_weight: float = 0.0
    fact_style_tolerance: float = 0.0
    fact_content_tolerance: float = 0.0
    fact_dependence_tolerance: float = 0.0
    eta_init: float = 0.1      # L_cls:   auxiliary CE weight at phase B start
    eta_final: float = 0.3     # L_cls:   auxiliary CE weight at phase D end
    lambda_var: float = 0.0    # diversity reg disabled: StyleMMD already handles z_s diversity
    # Controlled posterior-noise intervention.  The effective style variance is
    # exp(clamp(logvar_s, -10, 10)) + style_sigma_floor**2.  Zero exactly
    # preserves the historical checkpoints and training objective.
    style_sigma_floor: float = 0.0

    # --- Evaluation ---
    num_images_for_fid: int = 10000


f_cs_wae_config = FCSWAEConfig()
