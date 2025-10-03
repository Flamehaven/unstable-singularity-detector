"""Tests for numerical safety checks."""

import torch
import pytest
from src.utils.checks import assert_finite


def test_assert_finite_valid():
    """Test that finite tensors pass the check."""
    x = torch.tensor([1.0, 2.0, 3.0])
    assert_finite(x, "x")  # Should not raise


def test_assert_finite_nan_raises():
    """Test that NaN values raise FloatingPointError."""
    x = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(FloatingPointError, match="non-finite values"):
        assert_finite(x, "x")


def test_assert_finite_inf_raises():
    """Test that Inf values raise FloatingPointError."""
    x = torch.tensor([1.0, float("inf"), 3.0])
    with pytest.raises(FloatingPointError, match="non-finite values"):
        assert_finite(x, "x")


def test_assert_finite_mixed_raises():
    """Test that mixed NaN/Inf values raise FloatingPointError."""
    x = torch.tensor([float("nan"), float("inf"), 1.0])
    with pytest.raises(FloatingPointError, match="non-finite values"):
        assert_finite(x, "x")


def test_assert_finite_multidimensional():
    """Test finite check on multidimensional tensors."""
    x = torch.randn(3, 4, 5)
    assert_finite(x, "multidim")  # Should not raise

    x[1, 2, 3] = float("nan")
    with pytest.raises(FloatingPointError):
        assert_finite(x, "multidim")
