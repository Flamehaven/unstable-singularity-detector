"""Physics-specific utilities for fluid dynamics."""

from .bc import enforce_bc
from .self_similar import Similarity, forward_transform, inverse_transform

__all__ = ["enforce_bc", "Similarity", "forward_transform", "inverse_transform"]
