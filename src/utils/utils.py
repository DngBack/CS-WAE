import torch
import torch.nn.functional as F

"""
Utility functions for sampling, reparameterization, and kernel computation on spheres and manifolds.
"""

def sample_uniform_sphere(n_samples, dim, device='cuda'):
    """
    Sample points uniformly from the surface of a unit sphere.
    Args:
        n_samples (int): Number of samples to generate.
        dim (int): Dimensionality of the sphere.
        device (str): Device to place the tensor on.
    Returns:
        torch.Tensor: Tensor of shape (n_samples, dim) with normalized vectors.
    """
    # Generate random points and normalize to unit sphere
    eta = torch.randn(n_samples, dim, device=device)
    return F.normalize(eta, p=2, dim=1)

def mobius_reparam(eps, mu, rho, config):
    """
    Möbius reparameterization for sampling on a sphere manifold.
    Args:
        eps (torch.Tensor): Noise samples.
        mu (torch.Tensor): Mean direction vectors.
        rho (torch.Tensor): Scaling factors.
        config (dict): Configuration dictionary with 'epsilon' for numerical stability.
    Returns:
        torch.Tensor: Reparameterized and normalized samples.
    """
    # Ensure rho has correct shape for broadcasting
    rho = rho.unsqueeze(-1) if rho.dim() == 1 else rho
    # Compute dot product between eps and mu
    eps_mu_dot = torch.sum(eps * mu, dim=1, keepdim=True)
    # Möbius reparameterization formula
    numerator = (1 - rho**2) * eps + 2 * rho**2 * mu + 2 * rho * eps_mu_dot * mu
    denominator = 1 + 2 * rho * eps_mu_dot + rho**2
    z = numerator / (denominator + config["epsilon"])
    # Normalize to unit sphere
    return F.normalize(z, p=2, dim=1)

def rbf_kernel(x, y, sigma, config):
    """
    Compute the RBF (Gaussian) kernel between two sets of vectors on the sphere.
    Args:
        x (torch.Tensor): First set of vectors.
        y (torch.Tensor): Second set of vectors.
        sigma (float): Kernel bandwidth.
        config (dict): Configuration dictionary with 'epsilon' for numerical stability.
    Returns:
        torch.Tensor: Kernel matrix.
    """
    # Compute squared Euclidean distance using dot product
    dist_sq = 2 - 2 * (x @ y.t())
    return torch.exp(-dist_sq / (2 * sigma**2 + config["epsilon"]))