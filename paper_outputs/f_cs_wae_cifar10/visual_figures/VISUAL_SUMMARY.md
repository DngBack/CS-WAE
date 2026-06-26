# Visual Figures for F-CS-WAE CIFAR-10

These figures are generated from completed local checkpoints only.

## Generated Figures

- `fig03_available_factorization_samples.*`: single-latent CS-WAE vs full F-CS-WAE prior samples.
- `fig04_cifar10_latent_structure_umap.*`: AE, single-latent CS-WAE, and F-CS-WAE semantic latent structure with class centers.
- `fig05_f_cs_wae_class_conditional_prior_grid.*`: class-conditional prior grid, rows are CIFAR-10 classes.
- `fig06_style_semantic_disentanglement_grid.*`: fixed semantic/vary style and fixed style/vary semantic center diagnostics.
- `fig06b_class_center_slerp.*`: spherical interpolation between learned class centers.
- `fig07_clustering_generation_tradeoff.*`: ACC-FID trade-off scatter.

## Missing Exact Requested Panel

`F-CS-WAE without style MMD` is not plotted because there is no completed checkpoint for `no_style_mmd` in `runs_f/cifar10/f_ablation_20260625_183045`.
Train that variant and re-run this script to make the exact three-column Figure 3.