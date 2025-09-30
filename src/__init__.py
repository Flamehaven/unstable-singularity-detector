"""
Unstable Singularity Detector

Revolutionary implementation of DeepMind's breakthrough discovery in fluid dynamics singularities.
Based on the paper "Discovery of Unstable Singularities" (arXiv:2509.14185).
"""

__version__ = "1.0.0"
__author__ = "Flamehaven Research"

from .unstable_singularity_detector import (
    UnstableSingularityDetector,
    SingularityDetectionResult,
    SingularityType
)
from .pinn_solver import PINNSolver, PINNConfig
from .gauss_newton_optimizer import (
    HighPrecisionGaussNewton,
    GaussNewtonOptimizer,
    GaussNewtonConfig,
    AdaptivePrecisionOptimizer
)
from .fluid_dynamics_sim import FluidDynamicsSimulator, SimulationConfig

__all__ = [
    "UnstableSingularityDetector",
    "SingularityDetectionResult",
    "SingularityType",
    "PINNSolver",
    "PINNConfig",
    "GaussNewtonOptimizer",
    "HighPrecisionGaussNewton",
    "GaussNewtonConfig",
    "AdaptivePrecisionOptimizer",
    "FluidDynamicsSimulator",
    "SimulationConfig"
]
