"""Boundary condition enforcement utilities."""

import torch
from typing import Literal


def enforce_bc(
    u: torch.Tensor,
    bc: torch.Tensor,
    mode: Literal["dirichlet", "neumann"] = "dirichlet"
) -> torch.Tensor:
    """
    Enforce boundary conditions on solution field.

    Args:
        u: Solution field [..., C, *grid]
        bc: Boundary condition values [..., C, *grid]
        mode: Type of boundary condition ("dirichlet" or "neumann")

    Returns:
        Solution field with enforced boundary conditions

    Notes:
        - Dirichlet: Directly override boundary values
        - Neumann: Gradient-based boundary (placeholder implementation)
    """
    if mode == "dirichlet":
        # Dirichlet BC: override values where bc is finite
        u = u.clone()
        u[..., :] = torch.where(torch.isfinite(bc), bc, u)
        return u
    elif mode == "neumann":
        # TODO: Gradient-based boundary implementation
        # For now, return unchanged (placeholder)
        return u
    else:
        raise ValueError(f"Unknown boundary condition mode: {mode}")
