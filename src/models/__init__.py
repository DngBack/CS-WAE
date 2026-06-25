# models package init
from .cs_wae import SphericalWAE_Supervised, EncoderCNN, DecoderCNN
from .baselines import VAE, WAE_MMD, S_VAE, VaDE
from .cs_wae_ablation import CSWAEAblation, create_ablation_model

# F-CS-WAE
from .f_cs_wae import FCSWAE, FCSWAEEncoder, ResBlockDecoder, ResBlock, ResBlockUp

# Extended baselines (Groups 1–3)
from .baselines_extended import (
    ResNetAE,
    AEWithCE,
    AEWithSupCon,
    AEWithCenterLoss,
    AEWithTriplet,
    ConditionalVAE,
    ConditionalWAE_MMD,
    GaussianClassPriorWAE,
    MODEL_META,
    ALL_EXTENDED_MODELS,
)

# F-CS-WAE ablation
from .f_cs_wae_ablation import FCSWAEAblation, create_f_cs_wae_ablation, ABLATION_VARIANTS

__all__ = [
    # CS-WAE (original)
    "SphericalWAE_Supervised",
    "EncoderCNN",
    "DecoderCNN",
    "VAE",
    "WAE_MMD",
    "S_VAE",
    "VaDE",
    "CSWAEAblation",
    "create_ablation_model",
    # F-CS-WAE
    "FCSWAE",
    "FCSWAEEncoder",
    "ResBlockDecoder",
    "ResBlock",
    "ResBlockUp",
    # Extended baselines
    "ResNetAE",
    "AEWithCE",
    "AEWithSupCon",
    "AEWithCenterLoss",
    "AEWithTriplet",
    "ConditionalVAE",
    "ConditionalWAE_MMD",
    "GaussianClassPriorWAE",
    "MODEL_META",
    "ALL_EXTENDED_MODELS",
    # F-CS-WAE ablation
    "FCSWAEAblation",
    "create_f_cs_wae_ablation",
    "ABLATION_VARIANTS",
]
