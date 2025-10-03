"""Tests for PDE residual metrics."""

import torch
import pytest
from src.utils.metrics import pde_residual_stats


def test_pde_residual_stats_shapes():
    """Test that residual stats returns correct keys and handles zero residual."""
    r = torch.zeros(4, 4, 4)
    stats = pde_residual_stats(r)
    assert set(stats.keys()) == {"res_max", "res_l2", "res_mean"}
    assert stats["res_max"] == 0.0
    assert stats["res_l2"] == 0.0
    assert stats["res_mean"] == 0.0


def test_pde_residual_stats_nonzero():
    """Test residual statistics with non-zero values."""
    r = torch.ones(2, 3, 4) * 0.5
    stats = pde_residual_stats(r)
    assert stats["res_max"] == pytest.approx(0.5, abs=1e-6)
    assert stats["res_l2"] == pytest.approx(0.5, abs=1e-6)
    assert stats["res_mean"] == pytest.approx(0.5, abs=1e-6)


def test_pde_residual_stats_mixed():
    """Test residual statistics with mixed positive/negative values."""
    r = torch.tensor([[[-1.0, 2.0], [0.5, -0.5]]])
    stats = pde_residual_stats(r)
    assert stats["res_max"] == pytest.approx(2.0, abs=1e-6)
    assert stats["res_mean"] == pytest.approx(1.0, abs=1e-6)
