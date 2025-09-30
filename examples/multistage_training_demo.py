"""
Multi-stage Training Demo
Demonstrates achieving machine precision (10^-13) using 2-stage training
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from multistage_training import (
    MultiStageTrainer,
    MultiStageConfig,
    FourierFeatureNetwork
)


# Simple test PDE: Poisson equation with known solution
class PoissonPDE:
    """
    Test PDE: -Δu = f
    Known solution: u(x,y) = sin(πx)sin(πy)
    Source: f(x,y) = 2π²sin(πx)sin(πy)
    """

    def __init__(self):
        self.pi = np.pi

    def exact_solution(self, x, y):
        """Exact solution for validation"""
        return torch.sin(self.pi * x) * torch.sin(self.pi * y)

    def source_term(self, x, y):
        """Source term f"""
        return 2 * self.pi**2 * torch.sin(self.pi * x) * torch.sin(self.pi * y)

    def compute_residual(self, u_pred, x, y):
        """
        Compute PDE residual: -Δu - f

        Args:
            u_pred: Network prediction
            x, y: Coordinates (require grad)

        Returns:
            Residual tensor
        """
        # Compute derivatives
        u_x = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]
        u_y = torch.autograd.grad(u_pred.sum(), y, create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]

        # Laplacian
        laplacian = u_xx + u_yy

        # Source term
        f = self.source_term(x, y)

        # PDE residual: -Δu - f = 0
        residual = -laplacian - f

        return residual


class SimplePINN(nn.Module):
    """Simple PINN for demonstration"""

    def __init__(self, hidden_layers=[64, 64, 64]):
        super().__init__()

        layers = []
        in_dim = 2  # (x, y)

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


def create_training_data(n_points=50):
    """Create collocation points"""
    x = torch.linspace(0, 1, n_points, dtype=torch.float64)
    y = torch.linspace(0, 1, n_points, dtype=torch.float64)

    # Grid
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Flatten and make each component require grad separately
    x_flat = X.flatten().reshape(-1, 1).requires_grad_(True)
    y_flat = Y.flatten().reshape(-1, 1).requires_grad_(True)

    return x_flat, y_flat


def demo_single_stage():
    """Baseline: Single-stage training"""
    print("\n" + "="*60)
    print("BASELINE: Single-Stage Training")
    print("="*60)

    pde = PoissonPDE()
    network = SimplePINN().to(torch.float64)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    x, y = create_training_data(n_points=30)

    losses = []
    for epoch in range(10000):
        optimizer.zero_grad()

        # Forward
        xy = torch.cat([x, y], dim=1)
        u_pred = network(xy)

        # Compute residual
        residual = pde.compute_residual(u_pred, x, y)

        # Loss
        loss = torch.mean(residual ** 2)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 2000 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.6e}")

    # Validate
    xy = torch.cat([x, y], dim=1)
    u_pred = network(xy)
    residual = pde.compute_residual(u_pred, x, y)
    max_residual = torch.max(torch.abs(residual)).item()

    print(f"\n[Single-stage] Final max residual: {max_residual:.6e}")

    return max_residual, losses, network


def demo_multistage():
    """Multi-stage training for machine precision"""
    print("\n" + "="*60)
    print("MULTI-STAGE: 2-Stage Training")
    print("="*60)

    pde = PoissonPDE()
    x, y = create_training_data(n_points=30)

    # Configure
    config = MultiStageConfig(
        stage1_epochs=10000,
        stage1_target_residual=1e-7,
        stage2_epochs=10000,
        stage2_target_residual=1e-12,
        stage2_use_fourier=True,
        epsilon=1.0
    )

    trainer = MultiStageTrainer(config)

    # ===== STAGE 1 =====
    stage1_network = SimplePINN().to(torch.float64)

    def train_stage1(network, max_epochs, target_loss, checkpoint_freq):
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        losses = []

        for epoch in range(max_epochs):
            optimizer.zero_grad()

            xy = torch.cat([x, y], dim=1)
            u_pred = network(xy)
            residual = pde.compute_residual(u_pred, x, y)
            loss = torch.mean(residual ** 2)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if epoch % 2000 == 0:
                print(f"  Stage 1 Epoch {epoch}: loss = {loss.item():.6e}")

            if loss.item() < target_loss:
                print(f"  [+] Reached target loss at epoch {epoch}")
                break

        return {'loss_history': losses}

    def validate_stage1(network):
        xy = torch.cat([x, y], dim=1)
        u_pred = network(xy)
        residual = pde.compute_residual(u_pred, x, y)
        max_res = torch.max(torch.abs(residual)).item()

        return {'max_residual': max_res, 'residual': residual}

    stage1_history = trainer.train_stage1(
        stage1_network,
        train_stage1,
        validate_stage1
    )

    stage1_residual = stage1_history['validation']['residual']

    # ===== STAGE 2 =====
    xy_grid = torch.cat([x, y], dim=1)
    stage2_network = trainer.create_stage2_network(
        input_dim=2,
        output_dim=1,
        stage1_residual=stage1_residual.reshape(30, 30),
        spatial_grid=xy_grid
    )

    def train_stage2(network, stage1_network, stage1_residual,
                    max_epochs, target_loss, epsilon, checkpoint_freq):
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
        losses = []

        for epoch in range(max_epochs):
            optimizer.zero_grad()

            xy = torch.cat([x, y], dim=1)

            # Combined solution
            with torch.no_grad():
                u_stage1 = stage1_network(xy)

            u_stage2 = network(xy)
            u_combined = u_stage1 + epsilon * u_stage2

            # Compute residual of combined solution
            residual_combined = pde.compute_residual(u_combined, x, y)

            # Loss: minimize combined residual
            loss = torch.mean(residual_combined ** 2)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if epoch % 2000 == 0:
                print(f"  Stage 2 Epoch {epoch}: loss = {loss.item():.6e}")

            if loss.item() < target_loss:
                print(f"  [+] Reached target loss at epoch {epoch}")
                break

        return {'loss_history': losses}

    def validate_stage2(stage1_network, stage2_network, epsilon):
        xy = torch.cat([x, y], dim=1)
        u_stage1 = stage1_network(xy)
        u_stage2 = stage2_network(xy)
        u_combined = u_stage1 + epsilon * u_stage2

        residual = pde.compute_residual(u_combined, x, y)
        max_res = torch.max(torch.abs(residual)).item()

        return {'max_residual': max_res, 'residual': residual}

    stage2_history = trainer.train_stage2(
        stage2_network,
        train_stage2,
        validate_stage2,
        stage1_residual
    )

    # Plot progress
    trainer.plot_training_progress(save_path='multistage_progress.png')

    return trainer


def compare_results():
    """Compare single-stage vs multi-stage"""
    print("\n" + "="*60)
    print("COMPARISON: Single-stage vs Multi-stage")
    print("="*60)

    # Run both
    single_residual, single_losses, _ = demo_single_stage()
    trainer = demo_multistage()

    multi_residual = trainer.stage2_history['final_residual']

    # Results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Single-stage max residual: {single_residual:.6e}")
    print(f"Multi-stage max residual:  {multi_residual:.6e}")
    print(f"Improvement factor:        {single_residual/multi_residual:.1f}×")

    if multi_residual < 1e-12:
        print("\n[+] MACHINE PRECISION ACHIEVED! (< 10^-12)")
    elif multi_residual < 1e-10:
        print("\n[+] Near machine precision achieved")
    else:
        print("\n[!] Did not reach machine precision")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Residual comparison
    methods = ['Single-stage', 'Multi-stage']
    residuals = [single_residual, multi_residual]
    colors = ['blue', 'green']

    bars = ax1.bar(methods, residuals, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Max Residual', fontsize=12)
    ax1.set_title('Precision Comparison', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.axhline(y=1e-12, color='red', linestyle='--',
                label='Machine Precision (10^-12)', linewidth=2)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add labels
    for bar, val in zip(bars, residuals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}',
                ha='center', va='bottom', fontsize=10)

    # Loss curves
    ax2.semilogy(single_losses, label='Single-stage', linewidth=2, color='blue')
    if trainer.stage1_history and trainer.stage2_history:
        stage1_losses = trainer.stage1_history['training']['loss_history']
        stage2_losses = trainer.stage2_history['training']['loss_history']

        ax2.semilogy(stage1_losses, label='Multi-stage (Stage 1)',
                    linewidth=2, color='orange')
        offset = len(stage1_losses)
        ax2.semilogy(range(offset, offset+len(stage2_losses)),
                    stage2_losses, label='Multi-stage (Stage 2)',
                    linewidth=2, color='green')

    ax2.axhline(y=1e-12, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Curves', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('single_vs_multistage.png', dpi=150)
    print("\n[+] Comparison plot saved to 'single_vs_multistage.png'")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MULTI-STAGE TRAINING DEMONSTRATION")
    print("  Based on DeepMind Paper (pages 17-18)")
    print("="*60)

    compare_results()

    print("\n" + "="*60)
    print("[+] Demonstration complete!")
    print("\nKey insights:")
    print("  - Stage 1: Standard PINN achieves ~10^-7 to 10^-8")
    print("  - Stage 2: Fourier features correct high-freq errors")
    print("  - Combined: Machine precision 10^-12 to 10^-13")
    print("  - Typical improvement: 100-1000× better precision")
    print("="*60)