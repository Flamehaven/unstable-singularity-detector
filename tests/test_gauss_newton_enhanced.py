"""
Tests for Enhanced Gauss-Newton Optimizer
Validates rank-1 Hessian, EMA, and machine precision convergence
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gauss_newton_optimizer_enhanced import (
    HighPrecisionGaussNewtonEnhanced,
    GaussNewtonConfig,
    Rank1HessianEstimator,
    EMAHessianApproximation
)


class TestRank1HessianEstimator:
    """Test rank-1 Hessian estimator"""

    def test_initialization(self):
        """Test estimator initialization"""
        estimator = Rank1HessianEstimator(n_params=10, batch_size=5)

        assert estimator.n_params == 10
        assert estimator.batch_size == 5

    def test_hessian_vector_product(self):
        """Test Hv computation"""
        n_params = 5
        n_residuals = 10

        estimator = Rank1HessianEstimator(n_params=n_params, batch_size=n_residuals)

        # Create test data
        jacobian = torch.randn(n_residuals, n_params, dtype=torch.float64)
        vector = torch.randn(n_params, dtype=torch.float64)

        # Compute Hv using estimator
        Hv_estimated = estimator.estimate_hessian_vector_product(jacobian, vector)

        # Compute exact Hv = (J^T J)v
        JTJ = torch.matmul(jacobian.T, jacobian)
        Hv_exact = torch.matmul(JTJ, vector)

        # Should match when batch_size == n_residuals
        assert torch.allclose(Hv_estimated, Hv_exact, atol=1e-10)

    def test_batch_sampling(self):
        """Test batch sampling produces valid output"""
        estimator = Rank1HessianEstimator(n_params=10, batch_size=3)

        jacobian = torch.randn(20, 10, dtype=torch.float64)
        vector = torch.randn(10, dtype=torch.float64)

        Hv = estimator.estimate_hessian_vector_product(jacobian, vector)

        assert Hv.shape == (10,)
        assert not torch.isnan(Hv).any()


class TestEMAHessianApproximation:
    """Test EMA Hessian approximation"""

    def test_initialization(self):
        """Test EMA initialization"""
        ema = EMAHessianApproximation(n_params=10, decay=0.9)

        assert ema.n_params == 10
        assert ema.decay == 0.9
        assert not ema.initialized

    def test_update(self):
        """Test EMA update"""
        ema = EMAHessianApproximation(n_params=5, decay=0.9)

        jacobian = torch.randn(10, 5, dtype=torch.float64)

        ema.update(jacobian)

        assert ema.initialized
        assert ema.ema_diag.shape == (5,)
        assert torch.all(ema.ema_diag > 0)

    def test_preconditioner(self):
        """Test preconditioner computation"""
        ema = EMAHessianApproximation(n_params=5, decay=0.9)

        jacobian = torch.randn(10, 5, dtype=torch.float64)
        ema.update(jacobian)

        precond = ema.get_preconditioner(damping=1e-6)

        assert precond.shape == (5,)
        assert torch.all(precond > 0)
        assert torch.all(torch.isfinite(precond))

    def test_ema_smoothing(self):
        """Test EMA smooths updates"""
        ema = EMAHessianApproximation(n_params=5, decay=0.95)

        # First update
        j1 = torch.ones(10, 5, dtype=torch.float64)
        ema.update(j1)
        diag1 = ema.ema_diag.clone()

        # Second update with different values
        j2 = torch.ones(10, 5, dtype=torch.float64) * 2
        ema.update(j2)
        diag2 = ema.ema_diag.clone()

        # EMA should smooth: diag2 should be between diag1 and new values
        expected_smooth = 0.95 * diag1 + 0.05 * torch.sum(j2**2, dim=0)

        assert torch.allclose(diag2, expected_smooth, atol=1e-10)


class TestGaussNewtonConfig:
    """Test configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = GaussNewtonConfig()

        assert config.learning_rate == 1e-3
        assert config.damping_factor == 1e-6
        assert config.tolerance == 1e-12
        assert config.use_ema_hessian is True
        assert config.use_rank1_hessian is True
        assert config.auto_learning_rate is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = GaussNewtonConfig(
            learning_rate=1e-2,
            tolerance=1e-10,
            use_ema_hessian=False
        )

        assert config.learning_rate == 1e-2
        assert config.tolerance == 1e-10
        assert config.use_ema_hessian is False


class TestHighPrecisionGaussNewtonEnhanced:
    """Test enhanced Gauss-Newton optimizer"""

    def test_initialization(self):
        """Test optimizer initialization"""
        config = GaussNewtonConfig()
        optimizer = HighPrecisionGaussNewtonEnhanced(config)

        assert optimizer.config == config
        assert optimizer.rank1_estimator is None  # Not initialized until optimize()
        assert optimizer.ema_hessian is None

    def test_quadratic_convergence(self):
        """Test convergence on simple quadratic problem"""
        # Setup problem: minimize ||Ax - b||²
        n_params = 5
        n_residuals = 10

        true_params = torch.randn(n_params, dtype=torch.float64)
        A = torch.randn(n_residuals, n_params, dtype=torch.float64)
        b = torch.matmul(A, true_params)

        def compute_residual(params):
            return torch.matmul(A, params) - b

        def compute_jacobian(params):
            return A

        # Configure optimizer
        config = GaussNewtonConfig(
            max_iterations=50,
            tolerance=1e-12,
            verbose=False
        )

        optimizer = HighPrecisionGaussNewtonEnhanced(config)

        # Optimize from random start
        initial = torch.randn(n_params, dtype=torch.float64)
        results = optimizer.optimize(compute_residual, compute_jacobian, initial)

        # Validate convergence
        assert results['converged']
        assert results['loss'] < 1e-10

        # Check parameter accuracy
        param_error = torch.norm(results['parameters'] - true_params).item()
        assert param_error < 1e-6

    def test_machine_precision_achievement(self):
        """Test that optimizer can achieve machine precision (< 10^-12)"""
        n_params = 3
        n_residuals = 6

        # Use fixed seed for reproducible test
        torch.manual_seed(42)
        true_params = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        A = torch.randn(n_residuals, n_params, dtype=torch.float64)
        b = torch.matmul(A, true_params)

        def compute_residual(params):
            return torch.matmul(A, params) - b

        def compute_jacobian(params):
            return A

        config = GaussNewtonConfig(
            max_iterations=100,
            tolerance=1e-12,
            gradient_clip=10.0,  # Increased for ill-conditioned problems
            verbose=False
        )

        optimizer = HighPrecisionGaussNewtonEnhanced(config)

        initial = torch.zeros(n_params, dtype=torch.float64)
        results = optimizer.optimize(compute_residual, compute_jacobian, initial)

        # Machine precision check
        assert results['loss'] < 1e-12, f"Loss {results['loss']:.2e} >= 1e-12"
        print(f"[+] Machine precision achieved: loss = {results['loss']:.2e}")

    def test_enhanced_components_initialization(self):
        """Test that enhanced components are initialized during optimization"""
        n_params = 5

        true_params = torch.randn(n_params, dtype=torch.float64)
        A = torch.randn(10, n_params, dtype=torch.float64)
        b = torch.matmul(A, true_params)

        def compute_residual(params):
            return torch.matmul(A, params) - b

        def compute_jacobian(params):
            return A

        config = GaussNewtonConfig(
            max_iterations=10,
            use_ema_hessian=True,
            use_rank1_hessian=True,
            verbose=False
        )

        optimizer = HighPrecisionGaussNewtonEnhanced(config)

        initial = torch.zeros(n_params, dtype=torch.float64)
        results = optimizer.optimize(compute_residual, compute_jacobian, initial)

        # Check components were initialized
        assert optimizer.rank1_estimator is not None
        assert optimizer.ema_hessian is not None
        assert optimizer.ema_hessian.initialized

    def test_history_tracking(self):
        """Test that optimization history is tracked"""
        n_params = 3

        true_params = torch.randn(n_params, dtype=torch.float64)
        A = torch.randn(6, n_params, dtype=torch.float64)
        b = torch.matmul(A, true_params)

        def compute_residual(params):
            return torch.matmul(A, params) - b

        def compute_jacobian(params):
            return A

        config = GaussNewtonConfig(
            max_iterations=20,
            auto_learning_rate=True,
            verbose=False
        )

        optimizer = HighPrecisionGaussNewtonEnhanced(config)

        initial = torch.zeros(n_params, dtype=torch.float64)
        results = optimizer.optimize(compute_residual, compute_jacobian, initial)

        # Check history exists
        assert len(results['loss_history']) > 0
        assert len(results['gradient_norm_history']) > 0
        assert len(results['damping_history']) > 0
        assert len(results['step_size_history']) > 0

        # Check monotonic decrease (mostly)
        losses = results['loss_history']
        for i in range(len(losses) - 1):
            # Allow some increase due to line search failures
            if i > 5:  # After initial iterations
                assert losses[i+1] <= losses[i] * 1.1  # At most 10% increase

    def test_auto_learning_rate(self):
        """Test automated learning rate updates"""
        n_params = 5

        true_params = torch.randn(n_params, dtype=torch.float64)
        A = torch.randn(10, n_params, dtype=torch.float64)
        b = torch.matmul(A, true_params)

        def compute_residual(params):
            return torch.matmul(A, params) - b

        def compute_jacobian(params):
            return A

        config = GaussNewtonConfig(
            max_iterations=30,
            auto_learning_rate=True,
            lr_update_freq=5,
            verbose=False
        )

        optimizer = HighPrecisionGaussNewtonEnhanced(config)

        initial = torch.zeros(n_params, dtype=torch.float64)
        results = optimizer.optimize(compute_residual, compute_jacobian, initial)

        # Learning rate should have been updated
        assert len(results['lr_history']) > 0


class TestIntegration:
    """Integration tests"""

    def test_all_features_enabled(self):
        """Test optimizer with all features enabled"""
        n_params = 10
        n_residuals = 20

        true_params = torch.randn(n_params, dtype=torch.float64)
        A = torch.randn(n_residuals, n_params, dtype=torch.float64)
        b = torch.matmul(A, true_params)

        def compute_residual(params):
            return torch.matmul(A, params) - b

        def compute_jacobian(params):
            return A

        config = GaussNewtonConfig(
            max_iterations=100,
            tolerance=1e-12,
            use_ema_hessian=True,
            use_rank1_hessian=True,
            auto_learning_rate=True,
            adaptive_damping=True,
            line_search=True,
            verbose=False
        )

        optimizer = HighPrecisionGaussNewtonEnhanced(config)

        initial = torch.randn(n_params, dtype=torch.float64)
        results = optimizer.optimize(compute_residual, compute_jacobian, initial)

        assert results['converged']
        assert results['loss'] < 1e-10

        print(f"\n[+] All features integration test:")
        print(f"  Final loss: {results['loss']:.2e}")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Time: {results['optimization_time']:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])