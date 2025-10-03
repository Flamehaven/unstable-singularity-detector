"""
Minimal torch utilities for testing and prototyping.

WARNING: This is NOT a production replacement for PyTorch.
Use actual PyTorch (torch==2.4.0) for all production code.

This module provides reference implementations for testing edge cases
and understanding expected behavior.
"""

from __future__ import annotations
import builtins
import math
from typing import List, Sequence, Union

Number = Union[int, float]


class TensorShim:
    """Minimal tensor for testing purposes only."""

    def __init__(self, data: Sequence[Number], shape: tuple = None):
        if isinstance(data, (int, float)):
            self._data = [float(data)]
        else:
            self._data = [float(x) for x in data]

        if shape is None:
            self.shape = (len(self._data),)
        else:
            self.shape = shape

    def __neg__(self) -> TensorShim:
        return TensorShim([-x for x in self._data], self.shape)

    def __abs__(self) -> TensorShim:
        # Delegate to module-level abs function to avoid recursion
        return abs_tensor(self)

    def abs(self) -> TensorShim:
        """Compute absolute value (fixed recursion bug)."""
        # FIXED: Was `return abs(self)` which caused infinite recursion
        # Now correctly delegates to builtins.abs
        return abs_tensor(self)

    def mean(self) -> float:
        """Compute mean value."""
        return sum(self._data) / max(1, len(self._data))

    def item(self) -> float:
        """Extract scalar value."""
        if len(self._data) != 1:
            raise ValueError(f"item() only works for 1-element tensors, got {len(self._data)}")
        return self._data[0]

    def tolist(self) -> List[float]:
        """Convert to Python list."""
        return list(self._data)

    def __repr__(self) -> str:
        return f"TensorShim({self._data[:5]}{'...' if len(self._data) > 5 else ''}, shape={self.shape})"


def abs_tensor(t: TensorShim) -> TensorShim:
    """
    Compute absolute value of tensor.

    FIXED: Separate function to avoid recursion in Tensor.abs() method.
    """
    return TensorShim([builtins.abs(x) for x in t._data], t.shape)


def arange(start: Number, end: Number, step: Number = 1.0) -> TensorShim:
    """
    Generate a range of values with proper edge case handling.

    FIXED ISSUES:
    1. ValueError when step == 0 (matches PyTorch behavior)
    2. Support for negative step (descending sequences)
    3. Proper termination conditions for both directions

    Args:
        start: Starting value (inclusive)
        end: Ending value (exclusive)
        step: Step size (default: 1.0)

    Returns:
        TensorShim with generated sequence

    Raises:
        ValueError: If step is zero

    Examples:
        >>> arange(0, 5, 1).tolist()
        [0.0, 1.0, 2.0, 3.0, 4.0]

        >>> arange(5, 0, -1).tolist()
        [5.0, 4.0, 3.0, 2.0, 1.0]

        >>> arange(0, 5, 0)
        ValueError: step must be non-zero
    """
    # Fix 1: Raise ValueError for step == 0 (matches PyTorch)
    if step == 0:
        raise ValueError("step must be non-zero")

    values = []
    current = float(start)
    end_val = float(end)
    step_val = float(step)

    # Fix 2: Branch on sign of step for correct termination
    if step_val > 0:
        # Positive step: ascending sequence
        while current < end_val - 1e-12:
            values.append(current)
            current += step_val
    else:
        # Negative step: descending sequence
        while current > end_val + 1e-12:
            values.append(current)
            current += step_val  # step is negative, so this decreases current

    return TensorShim(values)


def linspace(start: Number, end: Number, steps: int) -> TensorShim:
    """
    Generate linearly spaced values.

    Args:
        start: Starting value
        end: Ending value (inclusive)
        steps: Number of values to generate

    Returns:
        TensorShim with linearly spaced values
    """
    if steps <= 1:
        return TensorShim([float(start)])

    step_size = (end - start) / (steps - 1)
    values = [float(start + i * step_size) for i in range(steps)]
    return TensorShim(values)


def tensor(data: Union[Sequence[Number], Number]) -> TensorShim:
    """Create a tensor from data."""
    if isinstance(data, (int, float)):
        return TensorShim([data])
    return TensorShim(data)


__all__ = [
    "TensorShim",
    "arange",
    "linspace",
    "tensor",
    "abs_tensor",
]
