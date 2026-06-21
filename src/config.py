"""
Configuration file for CS-WAE experiments
"""
import torch

class Config:
    """Configuration class for CS-WAE experiments"""
    
    def __init__(self):
        # Device configuration
        # src/config.py — dòng 11
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Model parameters
        self.latent_dim = 32
        self.n_classes = 10
        
        # Training parameters
        self.batch_size = 128
        self.epochs = 50
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
        self.anneal_epochs = 20

        # LR scheduler (shared with ablation trainer)
        self.lr_scheduler_step = 30
        self.lr_scheduler_gamma = 0.5
        
        # Data parameters
        self.num_workers = 2
        
        # Evaluation parameters
        self.num_images_for_fid = 10000

# Create default config instance
config = Config()
