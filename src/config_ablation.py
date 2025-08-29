"""
Configuration file for CS-WAE ablation studies
"""

import torch


class AblationConfig:
    """Configuration class for CS-WAE ablation experiments"""

    def __init__(self):
        # Base configuration (inherited from main config)
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        # Model parameters
        self.latent_dim = 32
        self.n_classes = 10

        # Training parameters
        self.batch_size = 128
        self.epochs = 50  # Reduced for ablation speed
        self.lr = 1e-3

        # CS-WAE specific parameters
        self.rho_prior = 0.7
        self.epsilon = 1e-8

        # Loss weights
        self.bce_weight = 0.3
        self.lpips_weight = 0.7

        # Annealing parameters
        self.sup_mmd_weight = 20.0
        self.unsup_mmd_weight = 50.0
        self.anneal_epochs = 15  # Reduced for faster convergence

        # Data parameters
        self.num_workers = 2

        # Evaluation parameters
        self.num_images_for_fid = 5000  # Reduced for faster evaluation

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
