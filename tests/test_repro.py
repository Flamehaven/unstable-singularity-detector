"""Tests for reproducibility utilities."""

import torch
import numpy as np
import pytest
from src.utils.repro import set_global_seed


def test_seed_fix_torch():
    """Test that PyTorch random operations are deterministic."""
    set_global_seed(123, deterministic=True)
    a = torch.randn(10)

    set_global_seed(123, deterministic=True)
    b = torch.randn(10)

    assert torch.allclose(a, b), "Seeds should produce identical results"


def test_seed_fix_numpy():
    """Test that NumPy random operations are deterministic."""
    set_global_seed(456, deterministic=True)
    a = np.random.randn(10)

    set_global_seed(456, deterministic=True)
    b = np.random.randn(10)

    assert np.allclose(a, b), "Seeds should produce identical results"


def test_seed_different_values():
    """Test that different seeds produce different results."""
    set_global_seed(111, deterministic=True)
    a = torch.randn(10)

    set_global_seed(222, deterministic=True)
    b = torch.randn(10)

    assert not torch.allclose(a, b), "Different seeds should produce different results"


def test_deterministic_flag():
    """Test that deterministic mode is enabled."""
    set_global_seed(2025, deterministic=True)

    # Check CUDA backend settings
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
