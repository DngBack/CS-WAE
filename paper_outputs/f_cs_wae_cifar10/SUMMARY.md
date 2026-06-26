# F-CS-WAE CIFAR-10 Paper Outputs

Generated from repository-local experiment artifacts only.

## Main Result

- F-CS-WAE over 3 seeds: ACC 80.87 ± 0.52, NMI 64.96 ± 0.79, ARI 63.25 ± 0.90, FID 83.04 ± 0.86.
- Completed full ablation variant: ACC 81.49, NMI 66.22, ARI 64.38, FID 83.59.

## Caveats

- Ablation completion: 1/9 variants have metrics. The run stopped during `no_z_s` because logging hit disk quota.
- Extended baselines are partial: `ResNetAE` has metrics; `AEWithCE` has a checkpoint but no metrics file in the current output directory.
- The CIFAR-10 comparison table marks whether each row is a 3-seed mean, seed-0 result, or completed ablation run.

## Files

- `tables/table_f01_f_cs_wae_cifar10_main.tex`: main per-seed + mean table.
- `tables/table_f02_cifar10_comparison_available.tex`: available baseline comparison table.
- `tables/table_f03_ablation_status.tex`: ablation completion/status table.
- `figures/fig_f01_f_cs_wae_cifar10_main_metrics.*`: ACC/FID across seeds.
- `figures/fig_f02_f_cs_wae_training_curves.*`: training curves over 300 epochs.
- `figures/fig_f03_cifar10_comparison_available.*`: available ACC/FID comparison.