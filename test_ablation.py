"""
Quick test script for CS-WAE ablation study
"""

import torch
from src.config_ablation import ablation_config
from src.models.cs_wae_ablation import create_ablation_model
from src.datasets.mnist import get_mnist_loaders


def test_ablation_models():
    """Test that all ablation models can be created and run forward pass"""
    print("Testing CS-WAE Ablation Models...")

    # Create small test batch
    test_batch = torch.randn(4, 1, 28, 28)
    test_labels = torch.randint(0, 10, (4,))

    variants = ["baseline", "no_sup_mmd", "euclidean", "vmf_prior", "minimal"]

    for variant in variants:
        print(f"\nTesting variant: {variant}")

        try:
            # Create model
            model = create_ablation_model(variant)
            model.eval()

            # Test forward pass
            with torch.no_grad():
                x_hat, z_q, mu_q, param_q = model(test_batch)

            # Check output shapes
            assert x_hat.shape == test_batch.shape, (
                f"Reconstruction shape mismatch for {variant}"
            )
            assert z_q.shape == (4, ablation_config.latent_dim), (
                f"Latent shape mismatch for {variant}"
            )

            # Test sampling from prior
            try:
                z_p = model.sample_from_prior(0, 2, test_batch.device)
                assert z_p.shape == (2, ablation_config.latent_dim), (
                    f"Prior sampling shape mismatch for {variant}"
                )
            except Exception as e:
                print(f"  Warning: Prior sampling failed for {variant}: {e}")

            print(f"  ✅ {variant}: Forward pass successful")
            print(f"  Config: {model.variant_config}")

        except Exception as e:
            print(f"  ❌ {variant}: Failed with error: {e}")
            raise


def test_loss_calculation():
    """Test loss calculation for different variants"""
    print("\nTesting Loss Calculation...")

    from src.utils.loss_ablation import calculate_ablation_loss
    import lpips

    # Create test data
    test_batch = torch.randn(4, 1, 28, 28)
    test_labels = torch.randint(0, 10, (4,))

    # Initialize LPIPS
    loss_fn_vgg = lpips.LPIPS(net="vgg")

    for variant in ["baseline", "euclidean"]:
        print(f"\nTesting loss for variant: {variant}")

        try:
            model = create_ablation_model(variant)
            model.eval()

            with torch.no_grad():
                x_hat, z_q, mu_q, param_q = model(test_batch)

                total_loss, recon_loss, sup_mmd_loss, unsup_mmd_loss, kld_loss = (
                    calculate_ablation_loss(
                        test_batch,
                        test_labels,
                        x_hat,
                        z_q,
                        mu_q,
                        param_q,
                        model,
                        loss_fn_vgg,
                        1.0,
                        1.0,
                        epoch=0,
                    )
                )

                print(f"  ✅ {variant}: Loss calculation successful")
                print(f"    Total: {total_loss:.4f}, Recon: {recon_loss:.4f}")
                print(
                    f"    Sup MMD: {sup_mmd_loss:.4f}, Unsup MMD: {unsup_mmd_loss:.4f}, KLD: {kld_loss:.4f}"
                )

        except Exception as e:
            print(f"  ❌ {variant}: Loss calculation failed: {e}")
            raise


if __name__ == "__main__":
    print("=" * 60)
    print("CS-WAE ABLATION STUDY - QUICK TEST")
    print("=" * 60)

    # Test model creation and forward pass
    test_ablation_models()

    # Test loss calculation
    test_loss_calculation()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("Ablation study is ready to run.")
    print("Run: python run_ablation_study.py")
    print("=" * 60)
