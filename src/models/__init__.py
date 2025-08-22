# models package init
from .cs_wae import SphericalWAE_Supervised, EncoderCNN, DecoderCNN
from .baselines import VAE, WAE_MMD, S_VAE, VaDE

__all__ = [
    'SphericalWAE_Supervised', 
    'EncoderCNN', 
    'DecoderCNN',
    'VAE', 
    'WAE_MMD', 
    'S_VAE', 
    'VaDE'
]
