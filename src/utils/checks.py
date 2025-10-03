"""Numerical safety checks for tensor operations."""

import torch


def assert_finite(x: torch.Tensor, name: str) -> None:
    """
    Assert that all values in tensor are finite (no NaN or Inf).

    Args:
        x: Input tensor to check
        name: Name of tensor for error message

    Raises:
        FloatingPointError: If tensor contains non-finite values
    """
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        raise FloatingPointError(
            f"{name} contains non-finite values (count={bad}/{x.numel()})"
        )
