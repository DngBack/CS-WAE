# CS-WAE: Spherical Wasserstein Autoencoder

This repository contains the refactored implementation of CS-WAE (Spherical Wasserstein Autoencoder) with comprehensive baseline comparisons and evaluation metrics.

## Project Structure

```
CS-WAE/
├── src/                          # Main source code directory
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration settings
│   ├── models/                  # Model architectures
│   │   ├── __init__.py
│   │   ├── cs_wae.py           # CS-WAE implementation
│   │   └── baselines.py        # Baseline models (VAE, WAE-MMD, S-VAE, VaDE)
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── utils.py            # Spherical operations and MMD loss
│   │   └── loss.py             # Loss functions
│   ├── trainers/               # Training logic
│   │   ├── __init__.py
│   │   └── trainer.py          # Training classes
│   ├── visualization/          # Visualization functions
│   │   ├── __init__.py
│   │   └── plots.py            # Plotting and visualization
│   ├── metrics/                # Evaluation metrics
│   │   ├── __init__.py
│   │   └── evaluation.py       # Comprehensive evaluation
│   └── datasets/               # Dataset utilities
│       ├── __init__.py
│       └── mnist.py            # MNIST data loading
├── train_cs_wae.py             # Main training script for CS-WAE
├── compare_baselines.py        # Baseline comparison script
├── requirements.txt            # Dependencies
└── README_REFACTORED.md        # This file
```

## Key Features

### Models
- **CS-WAE**: Spherical Wasserstein Autoencoder with supervised learning
- **VAE**: Vanilla Variational Autoencoder
- **WAE-MMD**: Wasserstein Autoencoder with MMD regularization
- **S-VAE**: Spherical VAE
- **VaDE**: Variational Deep Embedding

### Loss Functions
- Combined BCE + LPIPS reconstruction loss
- Supervised and unsupervised MMD losses
- Annealing schedule for regularization weights

### Evaluation Metrics
- **Reconstruction Quality**: SSIM, PSNR, LPIPS
- **Generation Quality**: FID (Fréchet Inception Distance)
- **Clustering Quality**: Accuracy, NMI, ARI

### Visualization
- Random sampling from priors
- Slerp interpolation between latent vectors
- UMAP visualization of latent space
- Training history plots
- Reconstruction comparisons

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install additional packages for S-VAE (optional):
```bash
pip install git+https://github.com/nicola-decao/s-vae-pytorch.git
```

## Usage

### Training CS-WAE
```bash
python train_cs_wae.py
```

This will:
- Train the CS-WAE model on MNIST
- Generate visualizations
- Evaluate the model comprehensively
- Save results to `results_cs_wae/`

### Comparing with Baselines
```bash
python compare_baselines.py
```

This will:
- Train multiple baseline models
- Evaluate all models using the same metrics
- Generate a comparison table
- Save results to `baseline_results/`

### Using Individual Components

```python
from src.models import SphericalWAE_Supervised
from src.datasets.mnist import get_mnist_loaders
from src.trainers.trainer import CSWAETrainer
from src.config import config

# Load data
train_loader, test_loader = get_mnist_loaders()

# Initialize model
model = SphericalWAE_Supervised(config.latent_dim, config.n_classes)

# Train
trainer = CSWAETrainer(model, train_loader)
history = trainer.train()
```

## Configuration

Edit `src/config.py` to modify:
- Model hyperparameters (latent dimension, learning rate, etc.)
- Training settings (batch size, epochs, device)
- Loss weights (BCE vs LPIPS, MMD weights)
- Annealing schedule

## Key Improvements from Notebook

1. **Modular Structure**: Code is organized into logical modules
2. **Reusability**: Components can be imported and used independently
3. **Configuration Management**: Centralized configuration system
4. **Error Handling**: Better error handling and warnings
5. **Documentation**: Comprehensive docstrings and comments
6. **Extensibility**: Easy to add new models or evaluation metrics
7. **Maintainability**: Clear separation of concerns

## Example Results

The refactored code produces the same results as the original notebook but with better organization:

- **Reconstruction Quality**: High-quality reconstructions with LPIPS perceptual loss
- **Latent Space**: Well-separated clusters in spherical latent space
- **Generation**: Diverse and high-quality generated samples
- **Interpolation**: Smooth Slerp interpolations between classes

## Dependencies

- torch
- torchvision
- numpy
- matplotlib
- scikit-learn
- scipy
- umap-learn
- tqdm
- pandas
- pytorch-fid
- lpips
- torchmetrics

## License

[Add your license information here]

## Citation

[Add citation information for your CS-WAE paper here]
