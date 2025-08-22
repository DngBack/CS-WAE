# CS-WAE Ablation Study

This ablation study systematically evaluates the contribution of each key component in the CS-WAE architecture.

## Research Questions

### 1. **Supervised MMD Loss (L_MMD-sup)**

- **Question**: Is the supervised clustering force really necessary? Or is unsupervised MMD sufficient?
- **Hypothesis**: Without L_MMD-sup, clusters in latent space will become chaotic, overlapped, and poorly separated.
- **Variant**: `no_sup_mmd` - CS-WAE without supervised MMD loss

### 2. **Spherical Space vs Euclidean Space**

- **Question**: What are the real benefits of implementing the entire model on a curved manifold vs standard flat Euclidean space?
- **Hypothesis**: Euclidean space will not create "tight" and continuous clusters like spherical space. Interpolations will be less smooth.
- **Variant**: `euclidean` - CS-WAE in Euclidean space instead of spherical

### 3. **Spherical Cauchy vs von Mises-Fisher Prior**

- **Question**: Does choosing a "heavy-tailed" spCauchy prior really provide better robustness than the more common "light-tailed" von Mises-Fisher (vMF)?
- **Hypothesis**: On noisy data, the spCauchy model will maintain better performance than the vMF model.
- **Variant**: `vmf_prior` - CS-WAE with von Mises-Fisher prior instead of spherical Cauchy

## Ablation Variants

### Available Variants

1. **`baseline`** - Full CS-WAE (reference)

   - ✅ Supervised MMD Loss
   - ✅ Spherical Space
   - ✅ Spherical Cauchy Prior

2. **`no_sup_mmd`** - Tests importance of supervised clustering

   - ❌ Supervised MMD Loss
   - ✅ Spherical Space
   - ✅ Spherical Cauchy Prior

3. **`euclidean`** - Tests benefits of spherical manifold

   - ✅ Supervised MMD Loss
   - ❌ Spherical Space (uses Euclidean)
   - ❌ Gaussian Prior (instead of spherical)

4. **`vmf_prior`** - Tests heavy-tail benefits

   - ✅ Supervised MMD Loss
   - ✅ Spherical Space
   - ❌ von Mises-Fisher Prior (instead of spherical Cauchy)

5. **`minimal`** - Minimal variant for comparison
   - ❌ Supervised MMD Loss
   - ❌ Spherical Space
   - ❌ Gaussian Prior

## Usage

### Quick Start

```bash
python run_ablation_study.py
```

### Configuration

Edit `src/config_ablation.py` to modify:

- Training parameters (epochs, batch size, learning rate)
- Ablation variants to run
- Noise evaluation settings

### Customize Variants

In `run_ablation_study.py`, modify `variants_to_run` list:

```python
variants_to_run = [
    'baseline',      # Full CS-WAE
    'no_sup_mmd',    # Without supervised MMD
    'euclidean',     # Euclidean space
    'vmf_prior',     # von Mises-Fisher prior
    # 'minimal'      # Comment out for faster testing
]
```

## Evaluation Metrics

### Standard Metrics

- **Clustering Quality**: ACC, NMI, ARI
- **Reconstruction Quality**: SSIM, PSNR, LPIPS
- **Generation Quality**: FID

### Noise Robustness (Optional)

- Tests model performance under different noise conditions
- Noise types: Gaussian, Salt & Pepper, Uniform
- Noise levels: 0.0 to 0.5 standard deviation

## Expected Results

### Hypotheses to Validate

1. **Supervised MMD is crucial for clustering**

   - `baseline` should significantly outperform `no_sup_mmd` on ACC, NMI, ARI
   - Without supervision, clusters should be poorly separated

2. **Spherical space provides better geometry**

   - `baseline` should outperform `euclidean` on clustering metrics
   - Spherical interpolations should be smoother than Euclidean

3. **Heavy-tailed priors are more robust**
   - `baseline` should outperform `vmf_prior` under noise
   - Spherical Cauchy should show better noise robustness

### Success Criteria

A component is considered **essential** if removing it causes:

- > 5% drop in clustering accuracy (ACC)
- > 10% increase in reconstruction error (LPIPS)
- > 20% increase in generation quality (FID)

## Output Structure

```
ablation_results/ablation_YYYYMMDD_HHMMSS/
├── ablation_results.csv           # Main comparison table
├── ablation_comparison.png        # Performance comparison plots
├── training_comparison.png        # Training loss curves
├── noise_robustness_results.csv   # Noise evaluation (if run)
├── noise_robustness_comparison.png
├── experiment_config.txt          # Configuration for reproducibility
├── baseline/                      # Full CS-WAE results
│   ├── loss_history.png
│   ├── reconstructions.png
│   └── baseline_model.pth
├── no_sup_mmd/                    # Without supervised MMD
│   └── ...
└── [other_variants]/
```

## Implementation Details

### Key Technical Features

1. **Modular Architecture**: Each variant shares common components but differs in key aspects
2. **Unified Loss Function**: `calculate_ablation_loss()` handles all variants
3. **Flexible Sampling**: Supports both spherical and Euclidean latent spaces
4. **Comprehensive Evaluation**: Automatic metric calculation and visualization

### Mathematical Differences

- **Spherical vs Euclidean**: Different encoder outputs, sampling methods, and priors
- **Supervised MMD**: Class-conditional vs random prior matching
- **Prior Types**: Möbius reparameterization vs rejection sampling vs Gaussian

## Tips for Analysis

1. **Focus on clustering metrics** (ACC, NMI, ARI) to validate component importance
2. **Check training stability** - some ablations may be harder to train
3. **Examine latent space visualizations** - UMAP projections reveal clustering quality
4. **Compare noise robustness** - heavy-tailed priors should show advantages

## Troubleshooting

### Common Issues

- **GPU Memory**: Reduce batch size in `config_ablation.py`
- **Training Instability**: Some ablations may need different hyperparameters
- **Evaluation Errors**: Check model compatibility with evaluation functions

### Quick Debugging

```bash
# Test single variant (faster)
python -c "
from run_ablation_study import run_single_ablation
from src.datasets.mnist import get_mnist_loaders
train_loader, test_loader = get_mnist_loaders()
run_single_ablation('baseline', train_loader, test_loader, 'debug_results')
"
```

This systematic ablation study will provide strong empirical evidence for the importance of each CS-WAE component and validate your architectural choices.
