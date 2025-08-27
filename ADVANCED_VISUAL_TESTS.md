# Advanced Visual Tests for CS-WAE

This document describes the advanced visual testing capabilities added to the CS-WAE project.

## Overview

The advanced visual tests provide comprehensive analysis of the CS-WAE model's performance and behavior. These tests go beyond basic metrics to offer interpretable insights into the model's latent space structure, class separation, and generation quality.

## Available Tests

### 1. Latent Traversal Analysis
**File**: `src/visualization/advanced_tests.py::plot_latent_traversal()`

**Purpose**: Analyzes how individual latent dimensions affect image generation.

**What it shows**:
- Systematic variation of individual latent dimensions
- Visual understanding of what each dimension controls
- Identifies dimensions responsible for specific features (stroke thickness, orientation, etc.)

**Output**: `latent_traversal_class{X}_sample{Y}.png`

### 2. Class Separation Analysis
**File**: `src/visualization/advanced_tests.py::plot_class_separation_analysis()`

**Purpose**: Evaluates how well different classes are separated in the latent space.

**What it shows**:
- t-SNE visualization of latent representations colored by class
- Distribution histograms showing class overlap
- Prior centers overlaid on the t-SNE plot
- Quality of the learned latent space structure

**Output**: `class_separation_analysis.png`

### 3. Reconstruction Quality by Class
**File**: `src/visualization/advanced_tests.py::plot_reconstruction_quality_by_class()`

**Purpose**: Compares reconstruction quality across different digit classes.

**What it shows**:
- Side-by-side comparison of original vs reconstructed images
- Identifies which classes are easier/harder to reconstruct
- Visual assessment of reconstruction artifacts

**Output**: `reconstruction_quality_by_class.png`

### 4. Prior Distribution Visualization
**File**: `src/visualization/advanced_tests.py::plot_prior_distribution_visualization()`

**Purpose**: Visualizes samples generated from each class-specific prior.

**What it shows**:
- Grid of images generated from each class's prior distribution
- Quality and diversity of class-conditional generation
- Effectiveness of learned prior centers

**Output**: `prior_distribution_visualization.png`

### 5. Spherical Interpolation Grid
**File**: `src/visualization/advanced_tests.py::plot_spherical_interpolation_grid()`

**Purpose**: Creates smooth transitions between multiple class centers on the sphere.

**What it shows**:
- 2D grid showing interpolation between 4 different classes
- Smoothness and continuity of the spherical latent space
- Quality of transitions between different regions

**Output**: `spherical_interpolation_grid.png`

## Usage

### Method 1: Direct Function Calls
```python
from src.visualization.advanced_tests import (
    plot_latent_traversal,
    plot_class_separation_analysis,
    # ... other functions
)

# Use individual functions
plot_latent_traversal(model, test_dataset, class_idx=7, sample_idx=17)
plot_class_separation_analysis(model, test_loader)
```

### Method 2: Run All Tests
```python
from src.visualization.advanced_tests import run_all_advanced_tests

# Run all tests at once
results = run_all_advanced_tests(
    model=model,
    test_dataset=test_dataset,
    test_loader=test_loader,
    save_dir="analysis_results"
)
```

### Method 3: Standalone Script
```bash
python run_advanced_visual_tests.py
```

### Method 4: Integrated with Training
The advanced visual tests are automatically run when using `train_cs_wae.py`.

## Integration Points

### With Training Scripts
- `train_cs_wae.py`: Automatically runs advanced tests after training
- `run_ablation_study.py`: Can be integrated for ablation analysis

### With Evaluation Pipeline
- `src/metrics/evaluation.py`: Complements numerical metrics with visual analysis
- Works alongside FID, SSIM, PSNR, clustering metrics

### With Notebooks
- Easy integration into Jupyter notebooks for interactive analysis
- Compatible with existing notebook workflows

## Dependencies

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `scikit-learn` (for t-SNE)
- `tqdm`

## File Structure

```
src/
├── visualization/
│   ├── __init__.py          # Updated with new exports
│   ├── plots.py             # Existing visualization functions
│   ├── advanced_tests.py    # NEW: Advanced visual tests
│   ├── random.py
│   └── slerp.py
└── ...

run_advanced_visual_tests.py    # NEW: Standalone runner script
```

## Configuration

Tests use the main configuration from `src/config.py`:
- `latent_dim`: Latent space dimensionality
- `n_classes`: Number of classes (10 for MNIST)
- `device`: Computation device

## Output Files

All tests generate high-quality PNG files with descriptive names:
- 150 DPI resolution for publication quality
- Tight bounding boxes to minimize whitespace
- Clear titles and labels for interpretation

## Best Practices

1. **Run after training**: Visual tests are most meaningful on converged models
2. **Use appropriate sample sizes**: Balance between detail and computational cost
3. **Save results**: All outputs are automatically saved with descriptive filenames
4. **Compare across models**: Use consistent parameters when comparing different models
5. **Interpret results**: Visual tests complement but don't replace numerical metrics

## Future Extensions

Potential additions:
- **Latent arithmetic**: Visual operations in latent space
- **Attention maps**: If using attention mechanisms
- **Dynamic visualizations**: Animated transitions
- **Interactive plots**: Web-based exploration tools
- **Comparative analysis**: Side-by-side model comparisons
