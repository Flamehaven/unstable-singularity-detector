"""PDE residual metrics for convergence analysis."""

from typing import Dict
import torch


@torch.no_grad()
def pde_residual_stats(residual: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive residual statistics for PDE solving.

    Args:
        residual: PDE residual field [..., H, W, (D)]

    Returns:
        Dictionary containing:
        - res_max: Maximum absolute residual (L-infinity norm)
        - res_l2: L2 norm of residual (RMS)
        - res_mean: Mean absolute residual
    """
    r = residual.reshape(-1).float().abs()
    res_max = r.max().item()
    res_l2 = torch.linalg.vector_norm(r).item() / (r.numel() ** 0.5)
    res_mean = r.mean().item()
    return {"res_max": res_max, "res_l2": res_l2, "res_mean": res_mean}
