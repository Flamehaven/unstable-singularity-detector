"""Self-similar coordinate transformations for singularity analysis."""

import torch
from dataclasses import dataclass


@dataclass
class Similarity:
    """
    Self-similar transformation parameters.

    Transformations:
        x' = x / (T - t)^alpha
        u' = (T - t)^beta * u

    Attributes:
        alpha: Spatial scaling exponent
        beta: Field amplitude scaling exponent
    """
    alpha: float
    beta: float


def forward_transform(
    x: torch.Tensor,
    t: torch.Tensor,
    sim: Similarity
) -> torch.Tensor:
    """
    Transform to self-similar coordinates.

    Args:
        x: Spatial coordinates
        t: Time coordinate
        sim: Similarity parameters

    Returns:
        Transformed coordinates x' = x / (T - t)^alpha
    """
    tau = torch.clamp_min(1.0 - t, 1e-12)
    return x / (tau ** sim.alpha)


def inverse_transform(
    xp: torch.Tensor,
    t: torch.Tensor,
    sim: Similarity
) -> torch.Tensor:
    """
    Transform from self-similar coordinates back to physical space.

    Args:
        xp: Self-similar coordinates
        t: Time coordinate
        sim: Similarity parameters

    Returns:
        Physical coordinates x = x' * (T - t)^alpha
    """
    tau = torch.clamp_min(1.0 - t, 1e-12)
    return xp * (tau ** sim.alpha)
