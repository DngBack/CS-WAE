import torch
from .utils import rbf_kernel

"""
Loss functions for CS-WAE, including Maximum Mean Discrepancy (MMD) loss.
"""

def mmd_loss(q_samples, p_samples, device='cuda', sigma=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) loss between two sets of samples using the RBF kernel.
    Args:
        q_samples (torch.Tensor): Samples from the model distribution.
        p_samples (torch.Tensor): Samples from the prior distribution.
        device (str): Device to place the tensor on.
        sigma (float, optional): Kernel bandwidth. If None, estimated from data.
    Returns:
        torch.Tensor: Scalar MMD loss value.
    """
    # Return zero if not enough samples to compute pairwise distances
    if q_samples.shape[0] < 2 or p_samples.shape[0] < 2:
        return torch.tensor(0.0, device=device)
    # Estimate kernel bandwidth if not provided
    if sigma is None:
        with torch.no_grad():
            dists = torch.pdist(torch.cat([q_samples, p_samples], dim=0))
            sigma = dists.median()
    # Compute kernel values
    k_qq = rbf_kernel(q_samples, q_samples, sigma).mean()
    k_pp = rbf_kernel(p_samples, p_samples, sigma).mean()
    k_qp = rbf_kernel(q_samples, p_samples, sigma).mean()
    # MMD loss formula
    return k_qq + k_pp - 2 * k_qp