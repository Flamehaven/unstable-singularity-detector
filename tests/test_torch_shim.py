"""
Tests for torch_shim utilities.

Validates edge case handling and bug fixes:
- arange with step=0, positive/negative steps
- abs recursion fix
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.torch_shim import arange, tensor, TensorShim, abs_tensor


class TestArangeEdgeCases:
    """Test arange function with various edge cases."""

    def test_arange_positive_step(self):
        """Test ascending sequence with positive step."""
        result = arange(0, 5, 1)
        expected = [0.0, 1.0, 2.0, 3.0, 4.0]

        assert result.tolist() == expected, f"Expected {expected}, got {result.tolist()}"

    def test_arange_positive_step_fractional(self):
        """Test ascending sequence with fractional step."""
        result = arange(0, 2, 0.5)
        expected = [0.0, 0.5, 1.0, 1.5]

        assert len(result.tolist()) == len(expected)
        for a, b in zip(result.tolist(), expected):
            assert abs(a - b) < 1e-9, f"Values differ: {a} vs {b}"

    def test_arange_negative_step(self):
        """Test descending sequence with negative step."""
        result = arange(5, 0, -1)
        expected = [5.0, 4.0, 3.0, 2.0, 1.0]

        assert result.tolist() == expected, f"Expected {expected}, got {result.tolist()}"

    def test_arange_negative_step_fractional(self):
        """Test descending sequence with fractional negative step."""
        result = arange(2, 0, -0.5)
        expected = [2.0, 1.5, 1.0, 0.5]

        assert len(result.tolist()) == len(expected)
        for a, b in zip(result.tolist(), expected):
            assert abs(a - b) < 1e-9, f"Values differ: {a} vs {b}"

    def test_arange_zero_step_raises_error(self):
        """Test that step=0 raises ValueError (matches PyTorch behavior)."""
        with pytest.raises(ValueError, match="step must be non-zero"):
            arange(0, 5, 0)

    def test_arange_empty_range_positive_step(self):
        """Test empty result when start >= end with positive step."""
        result = arange(5, 0, 1)  # start > end with positive step
        assert result.tolist() == [], f"Expected empty list, got {result.tolist()}"

    def test_arange_empty_range_negative_step(self):
        """Test empty result when start <= end with negative step."""
        result = arange(0, 5, -1)  # start < end with negative step
        assert result.tolist() == [], f"Expected empty list, got {result.tolist()}"

    def test_arange_single_element(self):
        """Test range that produces single element."""
        result = arange(0, 0.5, 1)
        expected = [0.0]

        assert result.tolist() == expected, f"Expected {expected}, got {result.tolist()}"

    def test_arange_large_step(self):
        """Test range with step larger than range."""
        result = arange(0, 1, 10)
        expected = [0.0]

        assert result.tolist() == expected, f"Expected {expected}, got {result.tolist()}"


class TestAbsRecursionFix:
    """Test that abs() method doesn't cause infinite recursion."""

    def test_abs_method_no_recursion(self):
        """Test Tensor.abs() method works without recursion."""
        t = tensor([-1, -2, -3, 4, 5])
        result = t.abs()

        expected = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert result.tolist() == expected, f"Expected {expected}, got {result.tolist()}"

    def test_abs_dunder_method(self):
        """Test __abs__() magic method."""
        t = tensor([-1, -2, -3])
        result = abs(t)  # Calls __abs__()

        expected = [1.0, 2.0, 3.0]
        assert result.tolist() == expected, f"Expected {expected}, got {result.tolist()}"

    def test_abs_with_mixed_values(self):
        """Test abs with mixed positive/negative values."""
        t = tensor([-5, 3, -0, 2, -10])
        result = t.abs()

        expected = [5.0, 3.0, 0.0, 2.0, 10.0]
        assert result.tolist() == expected, f"Expected {expected}, got {result.tolist()}"

    def test_abs_module_function(self):
        """Test module-level abs_tensor function."""
        t = tensor([-1, -2, -3])
        result = abs_tensor(t)

        expected = [1.0, 2.0, 3.0]
        assert result.tolist() == expected, f"Expected {expected}, got {result.tolist()}"


class TestTensorShimBasics:
    """Basic functionality tests for TensorShim."""

    def test_tensor_creation(self):
        """Test basic tensor creation."""
        t = tensor([1, 2, 3])
        assert t.tolist() == [1.0, 2.0, 3.0]

    def test_tensor_scalar(self):
        """Test scalar tensor creation."""
        t = tensor(5)
        assert t.tolist() == [5.0]

    def test_tensor_mean(self):
        """Test mean calculation."""
        t = tensor([1, 2, 3, 4, 5])
        assert t.mean() == 3.0

    def test_tensor_item(self):
        """Test item extraction from 1-element tensor."""
        t = tensor(42)
        assert t.item() == 42.0

    def test_tensor_item_multi_element_raises(self):
        """Test item() raises error for multi-element tensor."""
        t = tensor([1, 2, 3])
        with pytest.raises(ValueError, match="1-element tensors"):
            t.item()

    def test_tensor_negation(self):
        """Test unary negation."""
        t = tensor([1, -2, 3])
        result = -t

        expected = [-1.0, 2.0, -3.0]
        assert result.tolist() == expected


class TestComparisonWithPyTorch:
    """
    Comparison tests to verify behavior matches PyTorch.

    Note: These tests document expected behavior but don't require PyTorch.
    """

    def test_arange_behavior_doc(self):
        """Document PyTorch arange behavior for reference."""
        # PyTorch: torch.arange(0, 5, 1) -> [0, 1, 2, 3, 4]
        assert arange(0, 5, 1).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

        # PyTorch: torch.arange(5, 0, -1) -> [5, 4, 3, 2, 1]
        assert arange(5, 0, -1).tolist() == [5.0, 4.0, 3.0, 2.0, 1.0]

        # PyTorch: torch.arange(0, 5, 0) -> RuntimeError: step must be nonzero
        with pytest.raises(ValueError):
            arange(0, 5, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
