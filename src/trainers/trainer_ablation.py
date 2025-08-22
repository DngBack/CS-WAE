"""
Training functions for CS-WAE ablation studies
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips
from ..config_ablation import ablation_config
from ..utils.loss_ablation import calculate_ablation_loss, add_noise_to_images


class AblationTrainer:
    """Trainer class for CS-WAE ablation models"""

    def __init__(self, model, train_loader, variant_name, device=None):
        self.model = model
        self.train_loader = train_loader
        self.variant_name = variant_name
        self.device = device or ablation_config.device
        self.model.to(self.device)

        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=ablation_config.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )

        # Initialize LPIPS loss function
        self.loss_fn_vgg = lpips.LPIPS(net="vgg").to(self.device)

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{ablation_config.epochs} [{self.variant_name}]",
        )

        # Accumulators
        losses = {
            "total": 0.0,
            "recon": 0.0,
            "sup_mmd": 0.0,
            "unsup_mmd": 0.0,
            "kld": 0.0,
        }

        # Calculate annealing weights
        anneal_rate = min(1.0, (epoch + 1) / ablation_config.anneal_epochs)
        current_sup_weight = ablation_config.sup_mmd_weight * anneal_rate
        current_unsup_weight = ablation_config.unsup_mmd_weight * anneal_rate

        for data, labels in pbar:
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            x_hat, z_q, mu_q, param_q = self.model(data)

            # Calculate loss
            total_loss, recon_loss, sup_mmd_loss, unsup_mmd_loss, kld_loss = (
                calculate_ablation_loss(
                    data,
                    labels,
                    x_hat,
                    z_q,
                    mu_q,
                    param_q,
                    self.model,
                    self.loss_fn_vgg,
                    current_sup_weight,
                    current_unsup_weight,
                    epoch,
                )
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate losses
            losses["total"] += total_loss.item()
            losses["recon"] += recon_loss.item()
            losses["sup_mmd"] += sup_mmd_loss.item()
            losses["unsup_mmd"] += unsup_mmd_loss.item()
            losses["kld"] += kld_loss.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{total_loss.item():.3f}",
                    "Recon": f"{recon_loss.item():.3f}",
                    "SupMMD": f"{sup_mmd_loss.item():.4f}",
                    "UnsupMMD": f"{unsup_mmd_loss.item():.4f}",
                    "KLD": f"{kld_loss.item():.4f}",
                }
            )

        self.scheduler.step()

        # Calculate average losses
        n_batches = len(self.train_loader)
        avg_losses = {k: v / n_batches for k, v in losses.items()}

        print(
            f"====> Epoch {epoch + 1} [{self.variant_name}] Avg Loss: "
            f"Total={avg_losses['total']:.4f}, Recon={avg_losses['recon']:.4f}, "
            f"SupMMD={avg_losses['sup_mmd']:.4f}, UnsupMMD={avg_losses['unsup_mmd']:.4f}, "
            f"KLD={avg_losses['kld']:.4f}"
        )

        return avg_losses

    def train(self, epochs=None):
        """Full training loop"""
        epochs = epochs or ablation_config.epochs
        history = []

        print(f"Starting training for {self.variant_name} on device: {self.device}")
        print(f"Model configuration: {self.model.variant_config}")

        for epoch in range(epochs):
            avg_losses = self.train_epoch(epoch)
            history.append(avg_losses)

        print(f"Training completed for {self.variant_name}!")
        return history


class NoiseRobustnessEvaluator:
    """Evaluator for noise robustness testing"""

    def __init__(self, device=None):
        self.device = device or ablation_config.device

    def evaluate_noise_robustness(self, model, test_loader, model_name):
        """
        Evaluate model performance under different noise conditions

        Args:
            model: Trained model
            test_loader: Test data loader
            model_name: Name of the model variant

        Returns:
            dict: Noise robustness results
        """
        model.eval()
        results = {}

        for noise_type in ablation_config.noise_types:
            results[noise_type] = {}

            for noise_level in ablation_config.noise_levels:
                print(
                    f"Evaluating {model_name} with {noise_type} noise (level: {noise_level})"
                )

                # Evaluate reconstruction quality under noise
                recon_errors = []

                with torch.no_grad():
                    for images, _ in tqdm(
                        test_loader, desc=f"Noise eval {noise_type}-{noise_level}"
                    ):
                        images = images.to(self.device)

                        # Add noise to input images
                        noisy_images = add_noise_to_images(
                            images, noise_type, noise_level
                        )

                        # Get reconstructions
                        if hasattr(model, "forward"):
                            # Ablation model
                            x_hat, _, _, _ = model(noisy_images)
                        else:
                            # Standard model
                            x_hat, _ = model(noisy_images)

                        # Calculate reconstruction error (MSE)
                        mse = torch.nn.functional.mse_loss(
                            x_hat, images, reduction="none"
                        )
                        recon_errors.append(mse.view(mse.size(0), -1).mean(dim=1))

                # Calculate average reconstruction error
                all_errors = torch.cat(recon_errors, dim=0)
                avg_error = all_errors.mean().item()
                std_error = all_errors.std().item()

                results[noise_type][noise_level] = {
                    "mean_recon_error": avg_error,
                    "std_recon_error": std_error,
                }

                print(f"  Mean reconstruction error: {avg_error:.6f} ± {std_error:.6f}")

        return results

    def compare_noise_robustness(self, results_dict):
        """
        Compare noise robustness across different model variants

        Args:
            results_dict: Dictionary of {model_name: noise_results}
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        # Create comparison table for Gaussian noise
        gaussian_results = {}
        for model_name, results in results_dict.items():
            gaussian_results[model_name] = [
                results["gaussian"][level]["mean_recon_error"]
                for level in ablation_config.noise_levels
            ]

        df = pd.DataFrame(gaussian_results, index=ablation_config.noise_levels)

        print("\n" + "=" * 80)
        print(
            "NOISE ROBUSTNESS COMPARISON (Gaussian Noise - Mean Reconstruction Error)"
        )
        print("=" * 80)
        print(df.round(6))
        print("=" * 80)

        # Plot noise robustness curves
        plt.figure(figsize=(12, 8))
        for model_name, errors in gaussian_results.items():
            plt.plot(
                ablation_config.noise_levels,
                errors,
                "o-",
                label=model_name,
                linewidth=2,
            )

        plt.xlabel("Noise Level (σ)")
        plt.ylabel("Mean Reconstruction Error")
        plt.title("Noise Robustness Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return df
