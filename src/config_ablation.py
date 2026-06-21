"""
Configuration file for CS-WAE ablation studies
"""

import torch

from .config import config


class AblationConfig:
    """Configuration class for CS-WAE ablation experiments"""

    def __init__(self):
        # Base configuration (inherited from main config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Model parameters
        self.latent_dim = config.latent_dim
        self.n_classes = config.n_classes

        # Training parameters — keep in sync with src/config.py
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.lr = config.lr

        # CS-WAE specific parameters
        self.rho_prior = config.rho_prior
        self.epsilon = config.epsilon

        # Loss weights
        self.bce_weight = config.bce_weight
        self.lpips_weight = config.lpips_weight

        # Annealing parameters
        self.sup_mmd_weight = config.sup_mmd_weight
        self.unsup_mmd_weight = config.unsup_mmd_weight
        self.anneal_epochs = config.anneal_epochs

        # LR scheduler
        self.lr_scheduler_step = config.lr_scheduler_step
        self.lr_scheduler_gamma = config.lr_scheduler_gamma

        # Data parameters
        self.num_workers = config.num_workers

        # Evaluation parameters
        self.num_images_for_fid = config.num_images_for_fid

        # Ablation study variants
        self.ablation_variants = {
            "baseline": {
                "name": "CS-WAE (Baseline)",
                "use_supervised_mmd": True,
                "use_spherical_space": True,
                "prior_type": "spherical_cauchy",
                "description": "Full CS-WAE model with all components",
            },
            "no_sup_mmd": {
                "name": "CS-WAE w/o Supervised MMD",
                "use_supervised_mmd": False,
                "use_spherical_space": True,
                "prior_type": "spherical_cauchy",
                "description": "CS-WAE without supervised MMD loss - tests clustering importance",
            },
            "euclidean": {
                "name": "CS-WAE (Euclidean)",
                "use_supervised_mmd": True,
                "use_spherical_space": False,
                "prior_type": "gaussian",
                "description": "CS-WAE in Euclidean space instead of spherical - tests manifold benefits",
            },
            "vmf_prior": {
                "name": "CS-WAE (von Mises-Fisher)",
                "use_supervised_mmd": True,
                "use_spherical_space": True,
                "prior_type": "von_mises_fisher",
                "description": "CS-WAE with vMF prior instead of spherical Cauchy - tests heavy-tail benefits",
            },
            "minimal": {
                "name": "Minimal CS-WAE",
                "use_supervised_mmd": False,
                "use_spherical_space": False,
                "prior_type": "gaussian",
                "description": "Minimal variant: no supervised MMD, Euclidean space, Gaussian prior",
            },
        }

        # Noise robustness test parameters
        self.noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Gaussian noise std
        self.noise_types = ["gaussian", "salt_pepper", "uniform"]


# Create default ablation config instance
ablation_config = AblationConfig()
