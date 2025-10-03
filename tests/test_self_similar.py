"""Tests for self-similar coordinate transformations."""

import torch
import pytest
from src.physics.self_similar import Similarity, forward_transform, inverse_transform


def test_similarity_roundtrip():
    """Test forward and inverse transform roundtrip."""
    x = torch.tensor([1.0, 2.0, 3.0])
    t = torch.tensor(0.5)
    sim = Similarity(alpha=0.3, beta=0.0)

    xp = forward_transform(x, t, sim)
    xr = inverse_transform(xp, t, sim)

    assert torch.allclose(x, xr, atol=1e-12)


def test_forward_transform_scaling():
    """Test that forward transform scales correctly."""
    x = torch.tensor([1.0])
    t = torch.tensor(0.0)  # tau = 1.0
    sim = Similarity(alpha=0.5, beta=0.0)

    xp = forward_transform(x, t, sim)
    # x' = x / (1.0)^0.5 = x / 1.0 = x
    assert torch.allclose(xp, x)


def test_inverse_transform_scaling():
    """Test that inverse transform scales correctly."""
    xp = torch.tensor([2.0])
    t = torch.tensor(0.5)  # tau = 0.5
    sim = Similarity(alpha=1.0, beta=0.0)

    x = inverse_transform(xp, t, sim)
    # x = x' * (0.5)^1.0 = 2.0 * 0.5 = 1.0
    assert torch.allclose(x, torch.tensor([1.0]))


def test_clamping_near_singularity():
    """Test that transformation handles near-singularity times."""
    x = torch.tensor([1.0])
    t = torch.tensor(1.0)  # tau = 0, clamped to 1e-12
    sim = Similarity(alpha=0.5, beta=0.0)

    xp = forward_transform(x, t, sim)
    # Should not produce inf/nan due to clamping
    assert torch.isfinite(xp).all()


def test_different_exponents():
    """Test transformations with various exponents."""
    x = torch.tensor([4.0])
    t = torch.tensor(0.75)  # tau = 0.25

    # Test alpha=0.5: x' = 4.0 / (0.25)^0.5 = 4.0 / 0.5 = 8.0
    sim1 = Similarity(alpha=0.5, beta=0.0)
    xp1 = forward_transform(x, t, sim1)
    assert torch.allclose(xp1, torch.tensor([8.0]), atol=1e-6)

    # Test alpha=1.0: x' = 4.0 / (0.25)^1.0 = 4.0 / 0.25 = 16.0
    sim2 = Similarity(alpha=1.0, beta=0.0)
    xp2 = forward_transform(x, t, sim2)
    assert torch.allclose(xp2, torch.tensor([16.0]), atol=1e-6)
