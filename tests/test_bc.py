"""Tests for boundary condition utilities."""

import torch
import pytest
from src.physics.bc import enforce_bc


def test_dirichlet_bc_overwrite():
    """Test Dirichlet boundary condition enforcement."""
    u = torch.zeros(1, 1, 4, 4)
    bc = torch.ones(1, 1, 4, 4)
    out = enforce_bc(u, bc, "dirichlet")
    assert out.sum().item() == 16.0
    assert torch.allclose(out, bc)


def test_dirichlet_bc_partial():
    """Test partial Dirichlet BC (NaN values not overwritten)."""
    u = torch.zeros(2, 2)
    bc = torch.tensor([[1.0, float("nan")], [float("nan"), 2.0]])
    out = enforce_bc(u, bc, "dirichlet")

    assert out[0, 0].item() == 1.0
    assert out[0, 1].item() == 0.0  # NaN in bc -> keep original
    assert out[1, 0].item() == 0.0  # NaN in bc -> keep original
    assert out[1, 1].item() == 2.0


def test_neumann_bc_placeholder():
    """Test Neumann BC (placeholder returns unchanged)."""
    u = torch.ones(1, 1, 3, 3)
    bc = torch.zeros(1, 1, 3, 3)
    out = enforce_bc(u, bc, "neumann")
    assert torch.allclose(out, u)


def test_invalid_bc_mode():
    """Test that invalid BC mode raises ValueError."""
    u = torch.zeros(2, 2)
    bc = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="Unknown boundary condition mode"):
        enforce_bc(u, bc, "invalid_mode")
