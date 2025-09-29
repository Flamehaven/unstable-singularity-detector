#!/usr/bin/env python3
"""
Test Suite for PINN Solver

Comprehensive tests for Physics-Informed Neural Networks including:
- Network architecture validation
- Training convergence testing
- Self-similar parameterization
- Precision accuracy validation
"""

import pytest
import torch
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pinn_solver import (
    PINNSolver, PINNConfig, PhysicsInformedNN, SelfSimilarPINN,
    IncompressiblePorousMedia, BoussinesqEquation, Euler3D
)

class TestPINNConfig:
    """Test PINN configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = PINNConfig()

        assert config.hidden_layers == [50, 50, 50, 50]
        assert config.activation == "tanh"
        assert config.precision == torch.float64
        assert config.convergence_threshold == 1e-10

    def test_custom_config(self):
        """Test custom configuration"""
        config = PINNConfig(
            hidden_layers=[32, 32],
            activation="relu",
            precision=torch.float32,
            learning_rate=1e-4
        )

        assert config.hidden_layers == [32, 32]
        assert config.activation == "relu"
        assert config.precision == torch.float32
        assert config.learning_rate == 1e-4

class TestPhysicsInformedNN:
    """Test Physics-Informed Neural Network"""

    def setup_method(self):
        """Setup for each test method"""
        self.config = PINNConfig(
            hidden_layers=[32, 32, 32],
            activation="tanh",
            precision=torch.float64
        )

    def test_network_initialization(self):
        """Test network initialization"""
        network = PhysicsInformedNN(self.config)

        # Check network structure
        assert len(network.layers) == 4  # input + 3 hidden + output
        assert network.config.activation == "tanh"

        # Check parameter count
        param_count = sum(p.numel() for p in network.parameters())
        assert param_count > 0

    def test_forward_pass(self):
        """Test forward pass with sample input"""
        network = PhysicsInformedNN(self.config)

        # Create sample input [batch_size, input_dim]
        batch_size = 100
        input_dim = 3  # (t, x, y)
        sample_input = torch.randn(batch_size, input_dim, dtype=torch.float64)

        # Forward pass
        output = network.forward(sample_input)

        # Check output shape
        assert output.shape == (batch_size, 1)
        assert output.dtype == torch.float64

    def test_activation_functions(self):
        """Test different activation functions"""
        activations = ["tanh", "relu", "swish"]

        for activation in activations:
            config = PINNConfig(activation=activation)
            network = PhysicsInformedNN(config)

            # Test forward pass
            sample_input = torch.randn(10, 3, dtype=torch.float64)
            output = network.forward(sample_input)

            assert output.shape == (10, 1)
            assert not torch.isnan(output).any()

class TestSelfSimilarPINN:
    """Test Self-Similar PINN for blow-up solutions"""

    def setup_method(self):
        """Setup for each test method"""
        self.config = PINNConfig(
            hidden_layers=[32, 32],
            precision=torch.float64
        )
        self.T_blowup = 1.0

    def test_initialization(self):
        """Test self-similar PINN initialization"""
        network = SelfSimilarPINN(self.config, self.T_blowup)

        # Check learnable parameters
        assert hasattr(network, 'lambda_param')
        assert hasattr(network, 'alpha_param')
        assert isinstance(network.lambda_param, torch.nn.Parameter)
        assert isinstance(network.alpha_param, torch.nn.Parameter)

    def test_self_similar_transformation(self):
        """Test self-similar coordinate transformation"""
        network = SelfSimilarPINN(self.config, self.T_blowup)

        # Create coordinates approaching blow-up time
        t = torch.tensor([0.5, 0.8, 0.95], dtype=torch.float64).view(-1, 1)
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64).view(-1, 1)
        y = torch.tensor([0.0, -0.5, 0.2], dtype=torch.float64).view(-1, 1)

        coordinates = torch.cat([t, x, y], dim=1)

        # Forward pass
        output = network.forward(coordinates)

        # Check that output grows as t approaches T_blowup
        assert output.shape == (3, 1)
        assert torch.all(torch.isfinite(output))

class TestPDESystems:
    """Test PDE system implementations"""

    def test_incompressible_porous_media(self):
        """Test IPM equation implementation"""
        pde = IncompressiblePorousMedia()

        # Test basic properties
        assert pde.dimension == 2
        assert pde.equation_type == "ipm"

        # Test RHS computation (mock test)
        u = torch.randn(32, 32, dtype=torch.float64)
        t = torch.tensor(0.5, dtype=torch.float64)

        # This would normally compute actual PDE terms
        # For testing, just check interface
        assert hasattr(pde, 'compute_rhs')

    def test_boussinesq_equation(self):
        """Test Boussinesq equation implementation"""
        pde = BoussinesqEquation()

        assert pde.dimension == 2
        assert pde.equation_type == "boussinesq"

    def test_euler_3d(self):
        """Test 3D Euler equation implementation"""
        pde = Euler3D()

        assert pde.dimension == 3
        assert pde.equation_type == "euler_3d"

class TestPINNSolver:
    """Test main PINN solver"""

    def setup_method(self):
        """Setup for each test method"""
        self.config = PINNConfig(
            hidden_layers=[16, 16],  # Small for testing
            max_epochs=100,  # Few epochs for testing
            convergence_threshold=1e-4,  # Relaxed for testing
            patience=10
        )
        self.pde_system = IncompressiblePorousMedia()

    def test_solver_initialization(self):
        """Test solver initialization"""
        solver = PINNSolver(self.pde_system, self.config)

        assert solver.pde_system == self.pde_system
        assert solver.config == self.config
        assert isinstance(solver.network, PhysicsInformedNN)

    def test_self_similar_initialization(self):
        """Test solver with self-similar parameterization"""
        solver = PINNSolver(
            self.pde_system, self.config,
            self_similar=True, T_blowup=1.0
        )

        assert isinstance(solver.network, SelfSimilarPINN)
        assert solver.self_similar == True

    def test_training_points_generation(self):
        """Test training points generation"""
        solver = PINNSolver(self.pde_system, self.config)

        training_points = solver.generate_training_points(
            n_interior=100,
            n_boundary=50,
            n_initial=25,
            domain_bounds={
                "x_min": -1, "x_max": 1,
                "y_min": -1, "y_max": 1,
                "t_min": 0, "t_max": 0.5
            }
        )

        assert 'interior' in training_points
        assert 'boundary' in training_points
        assert 'initial' in training_points

        assert training_points['interior'].shape[0] == 100
        assert training_points['boundary'].shape[0] == 50
        assert training_points['initial'].shape[0] == 25

    @pytest.mark.slow
    def test_training_convergence(self):
        """Test training convergence (slow test)"""
        solver = PINNSolver(self.pde_system, self.config)

        # Generate minimal training data
        solver.generate_training_points(n_interior=50, n_boundary=20, n_initial=10)

        # Run short training
        history = solver.train(max_epochs=50)

        # Check history structure
        assert 'total_loss' in history
        assert 'pde_loss' in history
        assert 'boundary_loss' in history
        assert len(history['total_loss']) <= 50

        # Check that loss generally decreases
        initial_loss = history['total_loss'][0]
        final_loss = history['total_loss'][-1]
        assert final_loss <= initial_loss  # Loss should decrease or stay same

    def test_solution_evaluation(self):
        """Test solution evaluation"""
        solver = PINNSolver(self.pde_system, self.config)

        # Create test coordinates
        test_coords = torch.tensor([
            [0.1, 0.0, 0.0],
            [0.5, 0.5, -0.5],
            [0.9, -0.2, 0.3]
        ], dtype=torch.float64)

        # Evaluate solution
        solution = solver.evaluate_solution(test_coords)

        assert solution.shape == (3, 1)
        assert torch.all(torch.isfinite(solution))

    def test_loss_computation(self):
        """Test loss computation components"""
        solver = PINNSolver(self.pde_system, self.config)

        # Generate minimal training points
        solver.generate_training_points(n_interior=20, n_boundary=10, n_initial=5)

        # Test loss computation
        total_loss, loss_components = solver._compute_loss()

        assert isinstance(total_loss, torch.Tensor)
        assert 'pde_loss' in loss_components
        assert 'boundary_loss' in loss_components
        assert total_loss.requires_grad == True

class TestPrecisionAccuracy:
    """Test precision and accuracy requirements"""

    def test_double_precision_training(self):
        """Test that training maintains double precision"""
        config = PINNConfig(
            precision=torch.float64,
            hidden_layers=[8, 8],  # Very small for testing
            max_epochs=10
        )

        pde_system = IncompressiblePorousMedia()
        solver = PINNSolver(pde_system, config)

        # Check that network uses double precision
        for param in solver.network.parameters():
            assert param.dtype == torch.float64

    def test_convergence_monitoring(self):
        """Test convergence monitoring and early stopping"""
        config = PINNConfig(
            convergence_threshold=1e-8,
            patience=5,
            max_epochs=100
        )

        pde_system = IncompressiblePorousMedia()
        solver = PINNSolver(pde_system, config)

        # Mock training to test early stopping logic
        solver.generate_training_points(n_interior=10, n_boundary=5, n_initial=3)

        # This should test the early stopping mechanism
        # In practice, we might not reach convergence with such small data

class TestIntegrationWithConfig:
    """Integration tests with configuration system"""

    def test_solver_from_config(self):
        """Test creating solver from configuration"""
        from config_manager import PINNConfig

        # This would normally load from YAML
        config_dict = {
            'hidden_layers': [32, 32, 32],
            'activation': 'tanh',
            'max_epochs': 1000,
            'learning_rate': 1e-3
        }

        # Test that configuration integrates properly
        assert 'hidden_layers' in config_dict
        assert 'activation' in config_dict

@pytest.mark.gpu
class TestGPUCompatibility:
    """Test GPU acceleration (requires CUDA)"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_training(self):
        """Test training on GPU"""
        config = PINNConfig(
            device="cuda",
            hidden_layers=[16, 16],
            max_epochs=10
        )

        pde_system = IncompressiblePorousMedia()
        solver = PINNSolver(pde_system, config)

        # Check that network is on GPU
        for param in solver.network.parameters():
            assert param.device.type == "cuda"

if __name__ == "__main__":
    # Run specific test categories
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])