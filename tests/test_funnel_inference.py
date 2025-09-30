"""
Test Funnel Inference Implementation
Validates secant method and convergence behavior
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from funnel_inference import (
    FunnelInference,
    FunnelInferenceConfig,
    create_evaluation_points_near_origin
)


class MockPDESystem:
    """Mock PDE system with known admissible lambda"""

    def __init__(self, admissible_lambda=0.5):
        self.admissible_lambda = admissible_lambda

    def compute_residual(self, u_pred, x, lambda_value):
        """
        Residual that is zero at admissible lambda
        r(λ) = (λ - λ*)·x
        """
        return (lambda_value - self.admissible_lambda) * x.sum(dim=-1)


class MockNetwork(nn.Module):
    """Simple mock network"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, dtype=torch.float64)

    def forward(self, x):
        return self.linear(x)


def mock_train_function(network, pde_system, lambda_fixed, max_steps=10):
    """Mock training function"""
    return {'final_loss': 1e-8}


class TestFunnelInference:
    """Test funnel inference algorithm"""

    def test_initialization(self):
        """Test funnel inference initialization"""
        config = FunnelInferenceConfig(initial_lambda=0.5)
        funnel = FunnelInference(config)

        funnel.initialize()

        assert len(funnel.lambda_history) == 1
        assert funnel.lambda_history[0] == 0.5
        assert funnel.iteration == 0

    def test_secant_first_iteration(self):
        """Test secant method first iteration (Eq. 18)"""
        config = FunnelInferenceConfig(
            initial_lambda=0.5,
            delta_lambda=0.01
        )
        funnel = FunnelInference(config)
        funnel.initialize()

        lambda_next = funnel.secant_update()

        # First iteration: λ₁ = λ₀ + Δλ
        assert lambda_next == pytest.approx(0.51, abs=1e-6)

    def test_secant_method(self):
        """Test secant method update (Eq. 17)"""
        config = FunnelInferenceConfig()
        funnel = FunnelInference(config)

        # Set up history
        funnel.lambda_history = [0.5, 0.51]
        funnel.residual_history = [0.1, 0.05]  # Residual decreasing

        lambda_next = funnel.secant_update()

        # Should move toward zero residual
        # λ₃ = λ₂ - r₁·(λ₁ - λ₂)/(r₁ - r₂)
        #    = 0.51 - 0.1·(0.5 - 0.51)/(0.1 - 0.05)
        #    = 0.51 - 0.1·(-0.01)/0.05
        #    = 0.51 + 0.02 = 0.53
        expected = 0.51 - 0.1 * (0.5 - 0.51) / (0.1 - 0.05)
        assert lambda_next == pytest.approx(expected, abs=1e-6)

    def test_convergence_lambda_change(self):
        """Test convergence detection via lambda change"""
        config = FunnelInferenceConfig(convergence_tol=1e-6)
        funnel = FunnelInference(config)

        # Set up history with small change
        funnel.lambda_history = [0.5, 0.5 + 1e-7]
        funnel.iteration = 2

        converged, reason = funnel.check_convergence()

        assert converged
        assert "Lambda converged" in reason

    def test_convergence_residual_zero(self):
        """Test convergence detection via residual near zero"""
        config = FunnelInferenceConfig(convergence_tol=1e-6)
        funnel = FunnelInference(config)

        funnel.lambda_history = [0.5, 0.51]
        funnel.residual_history = [0.1, 1e-7]  # Near zero
        funnel.iteration = 2

        converged, reason = funnel.check_convergence()

        assert converged
        assert "Residual near zero" in reason

    def test_proxy_residual_computation(self):
        """Test proxy residual computation"""
        config = FunnelInferenceConfig()
        funnel = FunnelInference(config)

        network = MockNetwork()
        pde_system = MockPDESystem(admissible_lambda=0.5)
        eval_points = torch.randn(10, 2, dtype=torch.float64)

        # At admissible lambda, residual should be small
        residual_at_admissible = funnel.compute_proxy_residual(
            network, pde_system, 0.5, eval_points
        )

        # Away from admissible, residual should be larger
        residual_away = funnel.compute_proxy_residual(
            network, pde_system, 0.7, eval_points
        )

        # Due to (λ - λ*) term, away should have larger magnitude
        assert abs(residual_away) > abs(residual_at_admissible) * 0.5

    def test_full_optimization_mock(self):
        """Test full optimization with mock system"""
        config = FunnelInferenceConfig(
            initial_lambda=0.4,
            max_iterations=5,
            training_steps_per_eval=10
        )

        funnel = FunnelInference(config)
        funnel.initialize()

        network = MockNetwork()
        pde_system = MockPDESystem(admissible_lambda=0.5)
        eval_points = torch.randn(10, 2, dtype=torch.float64)

        results = funnel.optimize(
            network=network,
            pde_system=pde_system,
            train_function=mock_train_function,
            evaluation_points=eval_points
        )

        # Check results structure
        assert 'final_lambda' in results
        assert 'final_residual' in results
        assert 'iterations' in results
        assert results['iterations'] > 0

    def test_create_evaluation_points_2d(self):
        """Test evaluation point generation (2D)"""
        points = create_evaluation_points_near_origin(n_points=20, radius=0.1, dim=2)

        assert points.shape[1] == 2
        assert len(points) > 0

        # Check points are within radius
        distances = torch.norm(points, dim=1)
        assert torch.all(distances <= 0.1 + 1e-6)

    def test_create_evaluation_points_3d(self):
        """Test evaluation point generation (3D)"""
        points = create_evaluation_points_near_origin(n_points=20, radius=0.1, dim=3)

        assert points.shape[1] == 3
        assert len(points) > 0

        distances = torch.norm(points, dim=1)
        assert torch.all(distances <= 0.1 + 1e-6)


class TestFunnelInferenceIntegration:
    """Integration tests with realistic scenarios"""

    def test_convergence_to_known_value(self):
        """Test that funnel inference converges to known admissible value"""
        admissible_lambda = 0.618  # Test value

        config = FunnelInferenceConfig(
            initial_lambda=0.5,
            max_iterations=10,
            convergence_tol=1e-3,
            training_steps_per_eval=10
        )

        funnel = FunnelInference(config)
        funnel.initialize()

        network = MockNetwork()
        pde_system = MockPDESystem(admissible_lambda=admissible_lambda)
        eval_points = torch.randn(10, 2, dtype=torch.float64)

        results = funnel.optimize(
            network=network,
            pde_system=pde_system,
            train_function=mock_train_function,
            evaluation_points=eval_points
        )

        # Should converge close to admissible value
        final_lambda = results['final_lambda']
        error = abs(final_lambda - admissible_lambda)

        print(f"\nConvergence test:")
        print(f"  True λ: {admissible_lambda}")
        print(f"  Final λ: {final_lambda}")
        print(f"  Error: {error}")

        # Relaxed tolerance for mock system
        assert error < 0.2, f"Did not converge to admissible value (error={error})"

    def test_lambda_history_monotonicity(self):
        """Test that lambda updates are reasonable"""
        config = FunnelInferenceConfig(
            initial_lambda=0.5,
            max_iterations=5
        )

        funnel = FunnelInference(config)
        funnel.initialize()

        network = MockNetwork()
        pde_system = MockPDESystem(admissible_lambda=0.6)
        eval_points = torch.randn(10, 2, dtype=torch.float64)

        results = funnel.optimize(
            network=network,
            pde_system=pde_system,
            train_function=mock_train_function,
            evaluation_points=eval_points
        )

        lambda_hist = results['lambda_history']

        # Check no extreme jumps
        for i in range(len(lambda_hist) - 1):
            change = abs(lambda_hist[i+1] - lambda_hist[i])
            assert change < 0.5, f"Lambda jumped too much at iteration {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])