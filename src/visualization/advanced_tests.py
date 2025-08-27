"""
Advanced visual tests for CS-WAE model evaluation and analysis
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE

from ..config import config
from ..utils.utils import sample_uniform_sphere, mobius_reparam


def plot_latent_traversal(model, dataset, class_idx=7, sample_idx=17, num_dims=8, num_steps=7, save_dir=".", device=None):
    """
    Analyze latent space by systematically varying individual dimensions.
    
    Args:
        model: Trained CS-WAE model
        dataset: Test dataset
        class_idx (int): Class index for labeling
        sample_idx (int): Index of sample to use as base
        num_dims (int): Number of dimensions to traverse
        num_steps (int): Number of steps for each dimension
        save_dir (str): Directory to save results
        device: Device to use for computation
    """
    if device is None:
        device = config.device
    
    print(f"\n--- Latent Traversal Analysis: Class {class_idx}, Sample {sample_idx} ---")
    
    model.eval()
    img, label = dataset[sample_idx]
    
    with torch.no_grad():
        # Encode base image
        mu_base, _ = model.encoder(img.unsqueeze(0).to(device))
        mu_base = F.normalize(mu_base, p=2, dim=1).squeeze(0)
        
        # Select representative dimensions to traverse
        dims_to_traverse = np.linspace(0, config.latent_dim-1, num_dims, dtype=int)
        
        fig, axes = plt.subplots(num_dims, num_steps, figsize=(num_steps * 1.5, num_dims * 1.5))
        
        for i, dim_idx in enumerate(dims_to_traverse):
            # Create variation values for this dimension
            traverse_values = torch.linspace(-2.0, 2.0, num_steps)
            
            for j, val in enumerate(traverse_values):
                # Copy base vector and modify specific dimension
                z_modified = mu_base.clone()
                z_modified[dim_idx] = val
                
                # Renormalize to ensure unit sphere constraint
                z_modified = F.normalize(z_modified.unsqueeze(0), p=2, dim=1)
                
                # Decode
                generated_img = model.decoder(z_modified.to(device))
                
                # Plot
                axes[i, j].imshow(generated_img.cpu().squeeze(), cmap='gray')
                axes[i, j].axis('off')
                
                if j == 0:
                    axes[i, j].set_ylabel(f'Dim {dim_idx}', rotation=0, labelpad=20)
                if i == 0:
                    axes[i, j].set_title(f'{val:.1f}')
        
        plt.suptitle(f'Latent Traversal - Class {class_idx}, Sample {sample_idx}', fontsize=16)
        plt.tight_layout()
        
        filename = f'{save_dir}/latent_traversal_class{class_idx}_sample{sample_idx}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved latent traversal analysis: {filename}")
        
        return filename


def plot_class_separation_analysis(model, test_loader, save_dir=".", max_samples=2000, device=None):
    """
    Analyze class separation in latent space using t-SNE and distribution plots.
    
    Args:
        model: Trained CS-WAE model
        test_loader: Test data loader
        save_dir (str): Directory to save results
        max_samples (int): Maximum number of samples to analyze
        device: Device to use for computation
    """
    if device is None:
        device = config.device
    
    print("\n--- Class Separation Analysis in Latent Space ---")
    
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        sample_count = 0
        for images, batch_labels in tqdm(test_loader, desc="Extracting latent representations"):
            if sample_count >= max_samples:
                break
            
            images = images.to(device)
            mu_q, _ = model.encoder(images)
            mu_q = F.normalize(mu_q, p=2, dim=1)
            
            latents.append(mu_q.cpu().numpy())
            labels.append(batch_labels.numpy())
            sample_count += len(batch_labels)
    
    latents = np.concatenate(latents, axis=0)[:max_samples]
    labels = np.concatenate(labels, axis=0)[:max_samples]
    
    # t-SNE visualization
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d = tsne.fit_transform(latents)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # t-SNE scatter plot
    axes[0].scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10', s=5, alpha=0.6)
    axes[0].set_title('t-SNE: Latent Space Class Separation')
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    
    # Add prior centers to t-SNE space
    with torch.no_grad():
        prior_mus = F.normalize(model.prior_mus, p=2, dim=1).cpu().numpy()
        # Project prior centers using the same t-SNE transform
        try:
            prior_2d = tsne.fit_transform(np.vstack([latents, prior_mus]))[-10:]
            axes[0].scatter(prior_2d[:, 0], prior_2d[:, 1], c=range(10), cmap='tab10', 
                           marker='*', s=200, edgecolor='black', label='Prior Centers')
        except Exception:
            print("Warning: Could not project prior centers to t-SNE space")
    
    axes[0].legend()
    
    # Class distribution histogram
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i in range(10):
        class_data = latents_2d[labels == i, 0]
        axes[1].hist(class_data, bins=30, alpha=0.6, label=f'Class {i}', color=colors[i])
    
    axes[1].set_title('Class Distribution in t-SNE Space (Component 1)')
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('Frequency')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    filename = f'{save_dir}/class_separation_analysis.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved class separation analysis: {filename}")
    
    return filename


def plot_reconstruction_quality_by_class(model, test_loader, save_dir=".", samples_per_class=5, device=None):
    """
    Compare reconstruction quality across different classes.
    
    Args:
        model: Trained CS-WAE model
        test_loader: Test data loader
        save_dir (str): Directory to save results
        samples_per_class (int): Number of samples per class to show
        device: Device to use for computation
    """
    if device is None:
        device = config.device
    
    print("\n--- Reconstruction Quality Analysis by Class ---")
    
    model.eval()
    
    # Collect samples for each class
    class_samples = {i: [] for i in range(10)}
    class_originals = {i: [] for i in range(10)}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            reconstructed, _ = model(images)
            
            for i in range(len(images)):
                label = labels[i].item()
                if len(class_samples[label]) < samples_per_class:
                    class_samples[label].append(reconstructed[i].cpu())
                    class_originals[label].append(images[i].cpu())
            
            # Check if we have enough samples
            if all(len(class_samples[i]) >= samples_per_class for i in range(10)):
                break
    
    # Create comparison plot
    fig, axes = plt.subplots(20, samples_per_class, figsize=(samples_per_class * 2, 40))
    
    for class_idx in range(10):
        for sample_idx in range(samples_per_class):
            # Original
            row_orig = class_idx * 2
            axes[row_orig, sample_idx].imshow(class_originals[class_idx][sample_idx].squeeze(), cmap='gray')
            axes[row_orig, sample_idx].axis('off')
            if sample_idx == 0:
                axes[row_orig, sample_idx].set_ylabel(f'Class {class_idx}\nOriginal', rotation=0, labelpad=40)
            
            # Reconstructed
            row_recon = class_idx * 2 + 1
            axes[row_recon, sample_idx].imshow(class_samples[class_idx][sample_idx].squeeze(), cmap='gray')
            axes[row_recon, sample_idx].axis('off')
            if sample_idx == 0:
                axes[row_recon, sample_idx].set_ylabel('Reconstructed', rotation=0, labelpad=40)
    
    plt.suptitle('Reconstruction Quality by Class', fontsize=16)
    plt.tight_layout()
    
    filename = f'{save_dir}/reconstruction_quality_by_class.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved reconstruction quality analysis: {filename}")
    
    return filename


def plot_prior_distribution_visualization(model, save_dir=".", n_samples=1000, device=None):
    """
    Visualize prior distributions and generated samples from each class.
    
    Args:
        model: Trained CS-WAE model
        save_dir (str): Directory to save results
        n_samples (int): Number of samples to generate per class
        device: Device to use for computation
    """
    if device is None:
        device = config.device
    
    print("\n--- Prior Distribution Visualization ---")
    
    model.eval()
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    with torch.no_grad():
        normalized_prior_mus = F.normalize(model.prior_mus, p=2, dim=1)
        
        for class_idx in range(10):
            # Generate samples from this class's prior
            mu_p = normalized_prior_mus[class_idx].expand(n_samples, -1)
            rho_p = torch.full((n_samples,), model.rho_p, device=device)
            eps = sample_uniform_sphere(n_samples, config.latent_dim, device=device)
            z_p = mobius_reparam(eps, mu_p, rho_p)
            
            # Decode to generate images
            generated_images = model.decoder(z_p)
            
            # Create grid of generated images
            grid_size = int(np.sqrt(min(64, n_samples)))  # 8x8 grid
            selected_images = generated_images[:grid_size**2]
            
            # Reshape into grid
            grid = selected_images.view(grid_size, grid_size, 28, 28)
            grid = grid.permute(0, 2, 1, 3).contiguous().view(grid_size*28, grid_size*28)
            
            axes[class_idx].imshow(grid.cpu(), cmap='gray')
            axes[class_idx].set_title(f'Class {class_idx} Prior Samples')
            axes[class_idx].axis('off')
    
    plt.suptitle('Generated Samples from Class-Specific Priors', fontsize=16)
    plt.tight_layout()
    
    filename = f'{save_dir}/prior_distribution_visualization.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved prior distribution visualization: {filename}")
    
    return filename


def plot_spherical_interpolation_grid(model, save_dir=".", selected_classes=None, grid_size=7, device=None):
    """
    Create interpolation grid between multiple class centers on the sphere.
    
    Args:
        model: Trained CS-WAE model
        save_dir (str): Directory to save results
        selected_classes (list): List of 4 class indices for corners
        grid_size (int): Size of interpolation grid
        device: Device to use for computation
    """
    if device is None:
        device = config.device
    
    if selected_classes is None:
        selected_classes = [0, 1, 3, 8]  # Default selection
    
    print(f"\n--- Spherical Interpolation Grid: Classes {selected_classes} ---")
    
    model.eval()
    
    with torch.no_grad():
        normalized_prior_mus = F.normalize(model.prior_mus, p=2, dim=1)
        
        # Get 4 corner points
        top_left = normalized_prior_mus[selected_classes[0]]
        top_right = normalized_prior_mus[selected_classes[1]]
        bottom_left = normalized_prior_mus[selected_classes[2]]
        bottom_right = normalized_prior_mus[selected_classes[3]]
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 1.5, grid_size * 1.5))
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Bilinear interpolation on sphere
                t_vertical = i / (grid_size - 1)
                t_horizontal = j / (grid_size - 1)
                
                # Interpolate top edge using SLERP
                def slerp_safe(p0, p1, t):
                    dot_product = torch.dot(p0, p1).clamp(-1, 1)
                    if dot_product > 0.99:  # Very similar vectors
                        return F.normalize(p0 * (1 - t) + p1 * t, p=2, dim=0)
                    else:
                        omega = torch.acos(dot_product)
                        return (torch.sin((1-t) * omega) * p0 + torch.sin(t * omega) * p1) / torch.sin(omega)
                
                top_interp = slerp_safe(top_left, top_right, t_horizontal)
                bottom_interp = slerp_safe(bottom_left, bottom_right, t_horizontal)
                final_point = slerp_safe(top_interp, bottom_interp, t_vertical)
                
                # Generate image
                generated_img = model.decoder(final_point.unsqueeze(0).to(device))
                
                axes[i, j].imshow(generated_img.cpu().squeeze(), cmap='gray')
                axes[i, j].axis('off')
                
                # Label corners
                if i == 0 and j == 0:
                    axes[i, j].set_title(f'Class {selected_classes[0]}', fontsize=8)
                elif i == 0 and j == grid_size-1:
                    axes[i, j].set_title(f'Class {selected_classes[1]}', fontsize=8)
                elif i == grid_size-1 and j == 0:
                    axes[i, j].set_title(f'Class {selected_classes[2]}', fontsize=8)
                elif i == grid_size-1 and j == grid_size-1:
                    axes[i, j].set_title(f'Class {selected_classes[3]}', fontsize=8)
    
    plt.suptitle(f'Spherical Interpolation Grid: Classes {selected_classes}', fontsize=16)
    plt.tight_layout()
    
    filename = f'{save_dir}/spherical_interpolation_grid.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved spherical interpolation grid: {filename}")
    
    return filename


def run_all_advanced_tests(model, test_dataset, test_loader, save_dir=".", device=None):
    """
    Run all advanced visual tests in sequence.
    
    Args:
        model: Trained CS-WAE model
        test_dataset: Test dataset for single sample access
        test_loader: Test data loader for batch processing
        save_dir (str): Directory to save all results
        device: Device to use for computation
    """
    if device is None:
        device = config.device
    
    print("🎨 Running Advanced Visual Tests for CS-WAE...")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Latent Traversal
    print("\n🔍 Test 1: Latent Traversal Analysis")
    results['latent_traversal_7'] = plot_latent_traversal(
        model, test_dataset, class_idx=7, sample_idx=17, 
        num_dims=6, num_steps=7, save_dir=save_dir, device=device
    )
    results['latent_traversal_2'] = plot_latent_traversal(
        model, test_dataset, class_idx=2, sample_idx=208, 
        num_dims=6, num_steps=7, save_dir=save_dir, device=device
    )
    
    # Test 2: Class Separation Analysis
    print("\n🎯 Test 2: Class Separation Analysis")
    results['class_separation'] = plot_class_separation_analysis(
        model, test_loader, save_dir=save_dir, max_samples=1500, device=device
    )
    
    # Test 3: Reconstruction Quality by Class
    print("\n📊 Test 3: Reconstruction Quality by Class")
    results['reconstruction_quality'] = plot_reconstruction_quality_by_class(
        model, test_loader, save_dir=save_dir, samples_per_class=4, device=device
    )
    
    # Test 4: Prior Distribution Visualization
    print("\n🎲 Test 4: Prior Distribution Visualization")
    results['prior_visualization'] = plot_prior_distribution_visualization(
        model, save_dir=save_dir, n_samples=800, device=device
    )
    
    # Test 5: Spherical Interpolation Grid
    print("\n🌐 Test 5: Spherical Interpolation Grid")
    results['interpolation_grid'] = plot_spherical_interpolation_grid(
        model, save_dir=save_dir, device=device
    )
    
    print("\n" + "=" * 70)
    print("✅ Completed all advanced visual tests!")
    print(f"📁 All results saved in: {save_dir}")
    
    return results
