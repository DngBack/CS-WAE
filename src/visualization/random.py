import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.utils.utils import sample_uniform_sphere, mobius_reparam

"""
Visualization utilities for generating and plotting random samples from prior distributions.
"""

def plot_random_samples_from_priors(model, config, device='cuda', save_dir="."):
    """
    Plot and save images generated from random samples of prior distributions.
    Args:
        model: Trained model with prior means and decoder.
        config (dict): Configuration dictionary with 'n_classes' and 'latent_dim'.
        device (str): Device to run computations on.
        save_dir (str): Directory to save the output image.
    """
    print("Starting random sample generation from priors...")
    model.eval()
    n_classes, latent_dim = config["n_classes"], config["latent_dim"]
    # Create a subplot grid for visualization
    fig, axes = plt.subplots(n_classes, 8, figsize=(8 * 1.5, n_classes * 1.5))
    with torch.no_grad():
        # Normalize prior means for each class
        normalized_prior_mus = F.normalize(model.prior_mus, p=2, dim=1)
        for c in range(n_classes):
            # Expand mean vector for batch sampling
            mu_p = normalized_prior_mus[c].expand(8, -1)
            # Create rho vector for annealing
            rho_p = torch.full((8,), model.rho_p, device=device)
            # Sample random points on sphere
            eps = sample_uniform_sphere(8, latent_dim, device=device)
            # Apply Möbius reparameterization
            z_p = mobius_reparam(eps, mu_p, rho_p, config)
            # Decode latent vectors to images
            generated_images = model.decoder(z_p)
            for i, img in enumerate(generated_images):
                ax = axes[c, i]
                ax.imshow(img.cpu().squeeze(), cmap='gray')
                ax.axis('off')
                # Label the first image in each row with the class index
                if i == 0:
                    ax.text(-10, 14, f'{c}', verticalalignment='center', horizontalalignment='right', fontsize=12, fontweight='bold')
    plt.suptitle("Random samples generated from prior components (Annealing)")
    plt.savefig(f'{save_dir}/random_samples_from_priors.png')
    plt.close(fig)