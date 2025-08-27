"""
Visualization module for CS-WAE
"""

from .plots import (
    plot_results,
    plot_random_samples_from_priors,
    plot_grid_samples,
    plot_slerp,
    slerp
)

from .advanced_tests import (
    plot_latent_traversal,
    plot_class_separation_analysis,
    plot_reconstruction_quality_by_class,
    plot_prior_distribution_visualization,
    plot_spherical_interpolation_grid,
    run_all_advanced_tests
)

__all__ = [
    'plot_results',
    'plot_random_samples_from_priors',
    'plot_grid_samples',
    'plot_slerp',
    'slerp',
    'plot_latent_traversal',
    'plot_class_separation_analysis',
    'plot_reconstruction_quality_by_class',
    'plot_prior_distribution_visualization',
    'plot_spherical_interpolation_grid',
    'run_all_advanced_tests'
]