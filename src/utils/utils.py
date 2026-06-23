import torch
import torch.nn.functional as F
from ..config import config

"""
Utility functions for sampling, reparameterization, and kernel computation on spheres and manifolds.
"""


def to_rgb_for_lpips(tensor: torch.Tensor) -> torch.Tensor:
    """LPIPS expects 3 channels: repeat grayscale, pass RGB through."""
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    return tensor


def sample_uniform_sphere(n_samples, dim, device=None):
    """
    Sample points uniformly from the surface of a unit sphere.
    Args:
        n_samples (int): Number of samples to generate.
        dim (int): Dimensionality of the sphere.
        device (str): Device to place the tensor on.
    Returns:
        torch.Tensor: Tensor of shape (n_samples, dim) with normalized vectors.
    """
    if device is None:
        device = config.device
    # Generate random points and normalize to unit sphere
    eta = torch.randn(n_samples, dim, device=device)
    return F.normalize(eta, p=2, dim=1)


def mobius_reparam(eps, mu, rho):
    """
    Möbius reparameterization for sampling on a sphere manifold.
    Args:
        eps (torch.Tensor): Noise samples.
        mu (torch.Tensor): Mean direction vectors.
        rho (torch.Tensor): Scaling factors.
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
    z = numerator / (denominator + config.epsilon)
    # Normalize to unit sphere
    return F.normalize(z, p=2, dim=1)

def rbf_kernel(x, y, sigma):
    """
    Compute the RBF (Gaussian) kernel between two sets of vectors on the sphere.
    Args:
        x (torch.Tensor): First set of vectors.
        y (torch.Tensor): Second set of vectors.
        sigma (float): Kernel bandwidth.
    Returns:
        torch.Tensor: Kernel matrix.
    """
    # Compute squared Euclidean distance using dot product
    dist_sq = 2 - 2 * (x @ y.t())
    return torch.exp(-dist_sq / (2 * sigma**2 + config.epsilon))


def mmd_loss(q_samples, p_samples, sigma=None):
    """
    Maximum Mean Discrepancy loss for comparing two distributions.
    Args:
        q_samples (torch.Tensor): Samples from first distribution.
        p_samples (torch.Tensor): Samples from second distribution.
        sigma (float, optional): Kernel bandwidth. If None, uses median heuristic.
    Returns:
        torch.Tensor: MMD loss value.
    """
    if q_samples.shape[0] < 2 or p_samples.shape[0] < 2:
        return torch.tensor(0.0, device=config.device)
    
    if sigma is None:
        with torch.no_grad():
            dists = torch.pdist(torch.cat([q_samples, p_samples], dim=0))
            sigma = dists.median()
    
    k_qq = rbf_kernel(q_samples, q_samples, sigma).mean()
    k_pp = rbf_kernel(p_samples, p_samples, sigma).mean()
    k_qp = rbf_kernel(q_samples, p_samples, sigma).mean()
    return k_qq + k_pp - 2 * k_qp