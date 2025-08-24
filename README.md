# CS-WAE: Cauchy-Schwarz Wasserstein Autoencoder

## Overview
CS-WAE is an advanced implementation of the Cauchy-Schwarz Wasserstein Autoencoder, designed for generative modeling and representation learning. This repository provides code for training, evaluating, and analyzing CS-WAE and its ablation variants on datasets such as MNIST.

## Features
- Implementation of CS-WAE and baseline models
- Ablation study framework
- Support for MNIST dataset
- Visualization tools for metrics and latent space
- Modular codebase for easy extension

## Directory Structure
```
├── src/                # Source code (models, trainers, utils, datasets, metrics)
├── data/               # Dataset storage (MNIST)
├── ablation_results/   # Ablation study results
├── exp/                # Experiment outputs and visualizations
├── results_cs_wae/     # Main results
├── logs/               # Training logs
├── requirements.txt    # Python dependencies
├── train_cs_wae.py     # Main training script
├── run_ablation_study.py # Ablation study runner
├── compare_baselines.py  # Baseline comparison
├── test_ablation.py    # Ablation test script
├── README.md           # Project documentation
```

## Installation
1. Clone the repository:
   ```powershell
   git clone https://github.com/DngBack/CS-WAE.git
   cd CS-WAE
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage
- **Train CS-WAE:**
  ```powershell
  python train_cs_wae.py
  ```
- **Run Ablation Study:**
  ```powershell
  python run_ablation_study.py
  ```
- **Compare Baselines:**
  ```powershell
  python compare_baselines.py
  ```
- **Test Ablation:**
  ```powershell
  python test_ablation.py
  ```

## Ablation Study
Ablation studies are organized in `ablation_results/` and can be run using `run_ablation_study.py`. Results for different configurations (e.g., baseline, no_sup_mmd, euclidean) are stored in timestamped folders.

## Results & Visualization
Visualizations and metrics are available in the `exp/` and `ablation_results/` directories, including:
- Latent space UMAP plots
- Reconstruction samples
- Loss history
- Slerp interpolation
- Grid/random samples from priors

Jupyter notebooks in `exp/` and `metrics_cs-wae/` provide further analysis and visualization.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
