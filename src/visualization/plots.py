"""
Visualization functions for CS-WAE including random sampling, slerp interpolation, and result plotting
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
import umap
from itertools import product
from ..config import config
from ..utils.utils import sample_uniform_sphere, mobius_reparam


def slerp(p0, p1, t, epsilon=1e-8):
    """
    Spherical Linear Interpolation between two vectors.
    
    Args:
        p0 (torch.Tensor): Starting vector (normalized)
        p1 (torch.Tensor): Ending vector (normalized)
        t (torch.Tensor): Interpolation parameter(s) from 0 to 1
        epsilon (float): Small value to avoid division by zero
        
    Returns:
        torch.Tensor: Interpolated vectors
    """
    # Calculate angle between vectors
    omega = torch.acos(torch.dot(p0, p1).clamp(-1, 1))
    sin_omega = torch.sin(omega)

    # If vectors are too close, return starting vector to avoid division by zero
    if sin_omega.item() < epsilon:
        return p0.unsqueeze(0).expand(len(t), -1)

    # Ensure t has same device as vectors
    t = t.to(p0.device)

    # Slerp formula
    a = torch.sin((1.0 - t) * omega) / sin_omega
    b = torch.sin(t * omega) / sin_omega

    return a.unsqueeze(-1) * p0.unsqueeze(0) + b.unsqueeze(-1) * p1.unsqueeze(0)


def plot_random_samples_from_priors(model, save_dir=".", device=None):
    """Generate and plot random samples from prior distributions"""
    if device is None:
        device = config.device
        
    print("Generating random images from priors...")
    model.eval()
    n_classes, latent_dim = config.n_classes, config.latent_dim
    fig, axes = plt.subplots(n_classes, 8, figsize=(8 * 1.5, n_classes * 1.5))
    
    with torch.no_grad():
        normalized_prior_mus = F.normalize(model.prior_mus, p=2, dim=1)
        for c in range(n_classes):
            mu_p = normalized_prior_mus[c].expand(8, -1)
            rho_p = torch.full((8,), model.rho_p, device=device)
            eps = sample_uniform_sphere(8, latent_dim, device=device)
            z_p = mobius_reparam(eps, mu_p, rho_p)
            generated_images = model.decoder(z_p)
            
            for i, img in enumerate(generated_images):
                ax = axes[c, i]
                ax.imshow(img.cpu().squeeze(), cmap='gray')
                ax.axis('off')
                if i == 0:
                    ax.text(-10, 14, f'{c}', verticalalignment='center', 
                           horizontalalignment='right', fontsize=12, fontweight='bold')
                           
    plt.suptitle("Random Samples from Prior Components (Annealing)")
    plt.savefig(f'{save_dir}/random_samples_from_priors.png')
    plt.close(fig)


def from_latent(net, vec, device=None):
    """Generate image from latent vector"""
    if device is None:
        device = config.device
        
    with torch.no_grad():
        net.eval()
        vec_tensor = torch.from_numpy(vec).unsqueeze(0).to(device).float()
        vec_tensor_normalized = F.normalize(vec_tensor, p=2, dim=1)
        return net.decoder(vec_tensor_normalized).cpu().numpy().reshape(28, 28)


def get_sampling_grid(net, grid):
    """Generate images from a grid of 2D points extended to full latent dimension"""
    base = torch.randn(config.latent_dim - 2)
    image_list = [torch.from_numpy(from_latent(net, np.hstack([vec_2d, base.numpy()]))) 
                  for vec_2d in grid]
    return torch.stack(image_list).unsqueeze(1)


def plot_grid_samples(model, save_dir=".", device=None):
    """Generate and plot samples from a 2D grid"""
    print("Generating images from flat grid (comparison method)...")
    grid_points = list(product(np.linspace(-1.5, 1.5, 8), np.linspace(-1.5, 1.5, 8)))
    results = get_sampling_grid(model, grid_points)
    fig = plt.figure(figsize=(10, 10))
    img_grid = make_grid(results, nrow=8)
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.title("Images Generated from Flat Grid Sampling")
    plt.axis('off')
    plt.savefig(f'{save_dir}/grid_samples.png')
    plt.close(fig)


def plot_slerp(model, save_dir=".", num_steps=10, device=None):
    """Generate and plot Slerp interpolation between prior centers"""
    if device is None:
        device = config.device
        
    print("Generating Slerp interpolation images...")
    model.eval()

    # Choose interesting digit pairs for interpolation
    interpolation_pairs = [(1, 7), (3, 5), (2, 8), (4, 9)]
    num_pairs = len(interpolation_pairs)

    fig, axes = plt.subplots(num_pairs, num_steps, figsize=(num_steps * 1.5, num_pairs * 1.5))

    with torch.no_grad():
        t_values = torch.linspace(0, 1, num_steps)

        for row, (digit_a, digit_b) in enumerate(interpolation_pairs):
            # Get normalized prior vectors
            mu_a = F.normalize(model.prior_mus[digit_a].detach(), dim=0)
            mu_b = F.normalize(model.prior_mus[digit_b].detach(), dim=0)

            # Perform Slerp
            interpolated_z = slerp(mu_a, mu_b, t_values).to(device)

            # Decode intermediate vectors
            generated_images = model.decoder(interpolated_z)

            # Plot images
            for col, img in enumerate(generated_images):
                ax = axes[row, col]
                ax.imshow(img.cpu().squeeze(), cmap='gray')
                ax.axis('off')

                # Label first and last images
                if col == 0:
                    ax.set_title(f'{digit_a}')
                if col == num_steps - 1:
                    ax.set_title(f'{digit_b}')

    plt.suptitle("Slerp Interpolation between Priors")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{save_dir}/slerp_interpolation.png')
    plt.close(fig)
    print(f"Saved Slerp images to '{save_dir}/slerp_interpolation.png'")


def plot_results(history, model, test_loader, save_dir=".", device=None):
    """Plot training history and generate visualization results"""
    if device is None:
        device = config.device
        
    print("Starting result plotting and visualization...")
    
    # Plot training history
    fig = plt.figure(figsize=(12, 8))
    plt.plot([h[0] for h in history], label='Total Loss')
    plt.plot([h[1] for h in history], label='Reconstruction Loss')
    plt.plot([h[2] for h in history], label='Supervised MMD Loss')
    plt.plot([h[3] for h in history], label='Unsupervised MMD Loss')
    plt.title('Training History (WAE with Annealing)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/loss_history.png')
    plt.close(fig)

    # Plot reconstructions
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        x_hat, _ = model(data)
        
        fig = plt.figure(figsize=(20, 4))
        n = 10
        for i in range(n):
            # Original images
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(data[i].cpu().squeeze(), cmap='gray')
            plt.title("Original")
            ax.axis('off')
            
            # Reconstructed images
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(x_hat[i].cpu().squeeze(), cmap='gray')
            plt.title("Reconstructed")
            ax.axis('off')
            
        plt.savefig(f'{save_dir}/reconstructions.png')
        plt.close(fig)

        # UMAP visualization of latent space
        print("Starting latent space projection with UMAP...")
        latent_mus, labels = [], []
        for data, target in tqdm(test_loader, desc="Encoding test set for UMAP"):
            mu_q, _ = model.encode_to_distribution(data.to(device))
            latent_mus.append(mu_q.cpu().numpy())
            labels.append(target.numpy())

        latent_mus = np.concatenate(latent_mus, axis=0)
        labels = np.concatenate(labels, axis=0)

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(latent_mus)

        prior_mus_np = F.normalize(model.prior_mus, p=2, dim=1).cpu().detach().numpy()
        prior_embedding = reducer.transform(prior_mus_np)

        fig = plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5, alpha=0.7)
        plt.scatter(prior_embedding[:, 0], prior_embedding[:, 1], c=range(10), cmap='Spectral', 
                   marker='*', s=500, edgecolor='black', label='Prior Centers')
        plt.title('Latent Space - UMAP (Annealing)')
        plt.legend(handles=scatter.legend_elements(num=10)[0], labels=list(range(10)))
        plt.colorbar(scatter)
        plt.savefig(f'{save_dir}/latent_space_umap.png')
        plt.close(fig)

    # Generate additional visualizations
    plot_random_samples_from_priors(model, save_dir=save_dir, device=device)
    plot_grid_samples(model, save_dir=save_dir, device=device)
