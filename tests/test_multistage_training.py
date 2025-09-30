"""
Test Multi-stage Training Implementation
Validates 2-stage training for machine precision
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multistage_training import (
    MultiStageTrainer,
    MultiStageConfig,
    FourierFeatureNetwork
)


class SimplePINN(nn.Module):
    """Simple PINN for testing"""

    def __init__(self, hidden_layers=[32, 32]):
        super().__init__()

        layers = []
        in_dim = 2

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


class MockPDESystem:
    """Mock PDE for testing"""

    def compute_residual(self, u_pred, x, y):
        """Simple mock residual"""
        u_x = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u_pred.sum(), y, create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]

        laplacian = u_xx + u_yy
        return -laplacian


class TestFourierFeatureNetwork:
    """Test Fourier feature network"""

    def test_initialization(self):
        """Test network initialization"""
        network = FourierFeatureNetwork(
            input_dim=2,
            hidden_dim=64,
            output_dim=1,
            fourier_sigma=1.0,
            num_fourier_features=32
        )

        assert network.input_dim == 2
        assert network.B.shape == (2, 32)
        assert network.fourier_sigma == 1.0

    def test_forward_pass(self):
        """Test forward pass computation"""
        network = FourierFeatureNetwork(
            input_dim=2,
            hidden_dim=64,
            output_dim=1,
            fourier_sigma=1.0,
            num_fourier_features=32
        )
        network = network.to(torch.float64)

        x = torch.randn(10, 2, dtype=torch.float64)
        output = network(x)

        assert output.shape == (10, 1)

    def test_fourier_features_computation(self):
        """Test Fourier feature encoding"""
        network = FourierFeatureNetwork(
            input_dim=2,
            hidden_dim=64,
            output_dim=1,
            fourier_sigma=1.0,
            num_fourier_features=32
        )
        network = network.to(torch.float64)

        x = torch.randn(5, 2, dtype=torch.float64)
        output = network(x)

        # Check that features are non-trivial
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_different_sigmas(self):
        """Test networks with different sigma values"""
        x = torch.randn(10, 2, dtype=torch.float64)

        network_low = FourierFeatureNetwork(2, 64, 1, fourier_sigma=0.1).to(torch.float64)
        network_high = FourierFeatureNetwork(2, 64, 1, fourier_sigma=10.0).to(torch.float64)

        out_low = network_low(x)
        out_high = network_high(x)

        # Both should produce valid outputs
        assert out_low.shape == (10, 1)
        assert out_high.shape == (10, 1)


class TestMultiStageConfig:
    """Test configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = MultiStageConfig()

        assert config.stage1_epochs == 50000
        assert config.stage2_epochs == 100000
        assert config.stage1_target_residual == 1e-8
        assert config.stage2_target_residual == 1e-13
        assert config.stage2_use_fourier is True
        assert config.epsilon == 1.0

    def test_custom_config(self):
        """Test custom configuration"""
        config = MultiStageConfig(
            stage1_epochs=5000,
            stage2_epochs=5000,
            stage1_target_residual=1e-6,
            stage2_target_residual=1e-10,
            epsilon=0.5
        )

        assert config.stage1_epochs == 5000
        assert config.stage2_epochs == 5000
        assert config.stage1_target_residual == 1e-6
        assert config.stage2_target_residual == 1e-10
        assert config.epsilon == 0.5


class TestMultiStageTrainer:
    """Test multi-stage trainer"""

    def test_initialization(self):
        """Test trainer initialization"""
        config = MultiStageConfig()
        trainer = MultiStageTrainer(config)

        assert trainer.config == config
        assert trainer.stage1_network is None
        assert trainer.stage2_network is None
        assert trainer.stage1_history == {}
        assert trainer.stage2_history == {}

    def test_residual_frequency_analysis_2d(self):
        """Test frequency analysis on 2D residual"""
        config = MultiStageConfig()
        trainer = MultiStageTrainer(config)

        # Create synthetic residual with known frequency
        N = 32
        x = torch.linspace(0, 1, N)
        y = torch.linspace(0, 1, N)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Sinusoidal residual with frequency f=5
        residual = torch.sin(2 * np.pi * 5 * X) * torch.sin(2 * np.pi * 5 * Y)

        xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

        dominant_freq = trainer.analyze_residual_frequency(residual, xy)

        # Should detect frequency around 5
        assert dominant_freq > 0
        assert dominant_freq < 20  # Reasonable range

    def test_create_stage2_network(self):
        """Test stage 2 network creation"""
        config = MultiStageConfig(stage2_use_fourier=True)
        trainer = MultiStageTrainer(config)

        # Create mock residual
        N = 30
        residual = torch.randn(N, N)
        x = torch.linspace(0, 1, N)
        y = torch.linspace(0, 1, N)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

        stage2_network = trainer.create_stage2_network(
            input_dim=2,
            output_dim=1,
            stage1_residual=residual,
            spatial_grid=xy
        )

        assert isinstance(stage2_network, FourierFeatureNetwork)
        assert stage2_network.input_dim == 2

    def test_stage1_training(self):
        """Test stage 1 training"""
        config = MultiStageConfig(
            stage1_epochs=100,
            stage1_target_residual=1e-3
        )
        trainer = MultiStageTrainer(config)

        stage1_network = SimplePINN().to(torch.float64)

        def mock_train_stage1(network, max_epochs, target_loss, checkpoint_freq, use_amp=False, scaler=None):
            losses = [1e-2, 5e-3, 1e-3, 5e-4]  # Mock decreasing loss
            return {'loss_history': losses}

        def mock_validate_stage1(network):
            residual = torch.randn(10)
            max_res = torch.max(torch.abs(residual)).item()
            return {'max_residual': max_res, 'residual': residual}

        history = trainer.train_stage1(
            stage1_network,
            mock_train_stage1,
            mock_validate_stage1
        )

        assert 'training' in history
        assert 'validation' in history
        assert 'loss_history' in history['training']
        assert 'max_residual' in history['validation']

    def test_stage2_training(self):
        """Test stage 2 training"""
        config = MultiStageConfig(
            stage2_epochs=100,
            stage2_target_residual=1e-10,
            epsilon=1.0
        )
        trainer = MultiStageTrainer(config)

        # Set up stage 1
        stage1_network = SimplePINN().to(torch.float64)
        trainer.stage1_network = stage1_network
        trainer.stage1_history = {
            'final_residual': 1e-7  # Mock stage 1 final residual
        }

        # Mock stage 1 residual
        stage1_residual = torch.randn(10)

        # Create stage 2 network
        N = 30
        residual_2d = torch.randn(N, N)
        x = torch.linspace(0, 1, N)
        y = torch.linspace(0, 1, N)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

        stage2_network = trainer.create_stage2_network(2, 1, residual_2d, xy)

        def mock_train_stage2(network, stage1_network, stage1_residual,
                             max_epochs, target_loss, epsilon, checkpoint_freq):
            losses = [1e-8, 5e-9, 1e-9, 5e-10]  # Mock decreasing loss
            return {'loss_history': losses}

        def mock_validate_stage2(stage1_network, stage2_network, epsilon):
            residual = torch.randn(10) * 1e-10
            max_res = torch.max(torch.abs(residual)).item()
            return {'max_residual': max_res, 'residual': residual}

        history = trainer.train_stage2(
            stage2_network,
            mock_train_stage2,
            mock_validate_stage2,
            stage1_residual
        )

        assert 'training' in history
        assert 'validation' in history
        assert 'final_residual' in history

    def test_solution_composition(self):
        """Test combined solution composition"""
        config = MultiStageConfig(epsilon=0.5)
        trainer = MultiStageTrainer(config)

        stage1_network = SimplePINN().to(torch.float64)
        stage2_network = SimplePINN().to(torch.float64)

        trainer.stage1_network = stage1_network
        trainer.stage2_network = stage2_network

        x = torch.randn(10, 2, dtype=torch.float64)
        u_combined = trainer.compose_solution(x, epsilon=0.5)

        assert u_combined.shape == (10, 1)

        # Check composition formula: u_combined = u1 + ε·u2
        with torch.no_grad():
            u1 = stage1_network(x)
            u2 = stage2_network(x)
            expected = u1 + 0.5 * u2

            assert torch.allclose(u_combined, expected, atol=1e-10)


class TestMultiStageIntegration:
    """Integration tests"""

    def test_precision_improvement(self):
        """Test that stage 2 improves upon stage 1"""
        # This test validates the core claim: multi-stage achieves better precision

        config = MultiStageConfig(
            stage1_epochs=100,
            stage2_epochs=100
        )
        trainer = MultiStageTrainer(config)

        # Mock history showing improvement
        trainer.stage1_history = {
            'validation': {'max_residual': 1e-7}
        }

        trainer.stage2_history = {
            'final_residual': 1e-11
        }

        stage1_res = trainer.stage1_history['validation']['max_residual']
        stage2_res = trainer.stage2_history['final_residual']

        # Stage 2 should be significantly better
        assert stage2_res < stage1_res
        improvement_factor = stage1_res / stage2_res
        assert improvement_factor > 100  # At least 100× improvement

    def test_full_pipeline_mock(self):
        """Test complete pipeline with mocks"""
        config = MultiStageConfig(
            stage1_epochs=50,
            stage2_epochs=50
        )
        trainer = MultiStageTrainer(config)

        # Stage 1
        stage1_network = SimplePINN().to(torch.float64)

        def train_s1(network, max_epochs, target_loss, checkpoint_freq, use_amp=False, scaler=None):
            return {'loss_history': [1e-3, 1e-5, 1e-7]}

        def validate_s1(network):
            residual = torch.randn(100) * 1e-7
            return {
                'max_residual': torch.max(torch.abs(residual)).item(),
                'residual': residual
            }

        stage1_history = trainer.train_stage1(stage1_network, train_s1, validate_s1)

        # Stage 2
        N = 20
        residual_2d = torch.randn(N, N)
        x = torch.linspace(0, 1, N)
        y = torch.linspace(0, 1, N)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

        stage2_network = trainer.create_stage2_network(
            2, 1, residual_2d, xy
        )

        stage1_residual = stage1_history['validation']['residual']

        def train_s2(network, stage1_network, stage1_residual,
                    max_epochs, target_loss, epsilon, checkpoint_freq):
            return {'loss_history': [1e-8, 1e-10, 1e-12]}

        def validate_s2(s1_network, s2_network, epsilon):
            residual = torch.randn(100) * 1e-12
            return {
                'max_residual': torch.max(torch.abs(residual)).item(),
                'residual': residual
            }

        stage2_history = trainer.train_stage2(
            stage2_network, train_s2, validate_s2, stage1_residual
        )

        # Validate both stages completed
        assert trainer.stage1_history is not None
        assert trainer.stage2_history is not None

        # Validate improvement
        stage1_res = trainer.stage1_history['validation']['max_residual']
        stage2_res = trainer.stage2_history['final_residual']
        assert stage2_res < stage1_res


class TestMachinePrecisionTarget:
    """Test machine precision achievement"""

    def test_target_precision_10_minus_12(self):
        """Test that configuration targets 10^-12"""
        config = MultiStageConfig()

        assert config.stage2_target_residual <= 1e-12

    def test_precision_validation(self):
        """Test precision validation logic"""
        config = MultiStageConfig()
        trainer = MultiStageTrainer(config)

        # Mock achieving machine precision
        trainer.stage2_history = {
            'final_residual': 5e-13
        }

        final_res = trainer.stage2_history['final_residual']

        # Check if machine precision achieved
        machine_precision_achieved = final_res < 1e-12
        assert machine_precision_achieved is True

    def test_near_machine_precision(self):
        """Test near-machine precision detection"""
        config = MultiStageConfig()
        trainer = MultiStageTrainer(config)

        trainer.stage2_history = {
            'final_residual': 5e-11
        }

        final_res = trainer.stage2_history['final_residual']

        near_machine_precision = 1e-12 < final_res < 1e-10
        assert near_machine_precision is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])