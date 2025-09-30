"""
Funnel Inference Demo
Demonstrates lambda optimization using DeepMind's funnel inference method
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from funnel_inference import (
    FunnelInference,
    FunnelInferenceConfig,
    create_evaluation_points_near_origin
)


# Simple 1D example: Find admissible lambda for toy PDE
class ToyPDE:
    """
    Toy PDE system for demonstration:
    u_t + λ·u + u² = 0

    Admissible λ values exist where smooth solutions can be found
    """

    def compute_residual(self, u_pred, x, lambda_value):
        """Compute PDE residual"""
        x.requires_grad_(True)

        # Compute derivatives
        u_t = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]

        # PDE: u_t + λ·u + u² = 0
        residual = u_t + lambda_value * u_pred + u_pred ** 2

        return residual.squeeze()


class SimpleNetwork(nn.Module):
    """Simple MLP for toy problem"""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_toy_network(network, pde_system, lambda_fixed, max_steps=1000):
    """
    Train network with fixed lambda
    Returns training info
    """
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    # Training points
    x_train = torch.linspace(0, 1, 50).reshape(-1, 1)
    x_train.requires_grad_(True)

    network.train()
    for step in range(max_steps):
        optimizer.zero_grad()

        u_pred = network(x_train)
        residuals = pde_system.compute_residual(u_pred, x_train, lambda_fixed)

        loss = torch.mean(residuals ** 2)
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"  Step {step}: loss = {loss.item():.6e}")

    return {'final_loss': loss.item()}


def demo_funnel_inference_toy():
    """Run funnel inference on toy problem"""

    print("[*] Funnel Inference Demo - Toy 1D PDE")
    print("=" * 60)

    # Setup
    network = SimpleNetwork()
    pde_system = ToyPDE()

    # Evaluation points near origin
    eval_points = torch.linspace(0, 0.1, 10).reshape(-1, 1)

    # Configure funnel inference
    config = FunnelInferenceConfig(
        initial_lambda=0.5,
        delta_lambda=0.01,
        max_iterations=10,
        convergence_tol=1e-4,
        training_steps_per_eval=500,
        verbose=True
    )

    # Run funnel inference
    funnel = FunnelInference(config)
    funnel.initialize()

    results = funnel.optimize(
        network=network,
        pde_system=pde_system,
        train_function=train_toy_network,
        evaluation_points=eval_points
    )

    # Display results
    print("\n[+] Results:")
    print(f"  Final λ: {results['final_lambda']:.10f}")
    print(f"  Final residual: {results['final_residual']:.6e}")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Converged: {results['converged']}")

    # Plot funnel
    funnel.plot_funnel(save_path='funnel_inference_toy.png')

    return results


def demo_realistic_scenario():
    """
    More realistic scenario: IPM equation with known admissible lambda

    We know from paper that λ_stable = 1.0285722760222
    Test if funnel inference can find this value
    """
    print("\n[*] Realistic Scenario: IPM Stable Lambda")
    print("=" * 60)

    # Known admissible lambda from paper
    true_lambda = 1.0285722760222
    print(f"Ground truth λ (paper): {true_lambda:.10f}")

    # Initialize near but not at the true value
    initial_guess = true_lambda + 0.05
    print(f"Initial guess: {initial_guess:.10f}")
    print(f"Initial error: {abs(initial_guess - true_lambda):.6e}")

    # For realistic scenario, would need actual IPM PDE system
    # Here we demonstrate the workflow

    print("\n[!] Note: Full IPM implementation requires:")
    print("    - IncompressiblePorousMedia PDE class")
    print("    - Self-similar coordinate transformation")
    print("    - High-precision PINN architecture")
    print("    - Gauss-Newton optimizer")
    print("\n    See pinn_solver.py and fluid_dynamics_sim.py for complete implementation")


def analyze_funnel_shape():
    """
    Analyze funnel shape by scanning lambda values

    This demonstrates Figure 5 from paper:
    - Residual vs lambda has funnel shape
    - Minimum is at admissible lambda
    """
    print("\n[*] Funnel Shape Analysis")
    print("=" * 60)

    network = SimpleNetwork()
    pde_system = ToyPDE()
    eval_points = torch.linspace(0, 0.1, 10).reshape(-1, 1)

    # Scan lambda values
    lambda_range = np.linspace(0.3, 0.7, 20)
    residuals = []

    for lam in lambda_range:
        # Train briefly with this lambda
        _ = train_toy_network(network, pde_system, lam, max_steps=200)

        # Evaluate residual
        network.eval()
        with torch.no_grad():
            u_pred = network(eval_points)
            res = pde_system.compute_residual(u_pred, eval_points, lam)
            max_res = torch.max(torch.abs(res)).item()
            residuals.append(max_res)

        print(f"λ = {lam:.3f}: r̂ = {max_res:.6e}")

    # Plot funnel shape
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_range, residuals, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Lambda (λ)', fontsize=14)
    plt.ylabel('Max Residual', fontsize=14)
    plt.title('Funnel Shape: Residual vs Lambda', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Mark minimum
    min_idx = np.argmin(residuals)
    plt.plot(lambda_range[min_idx], residuals[min_idx], 'r*',
            markersize=20, label=f'Minimum at λ={lambda_range[min_idx]:.3f}')
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig('funnel_shape_analysis.png', dpi=150)
    print("\n[+] Plot saved to 'funnel_shape_analysis.png'")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  FUNNEL INFERENCE DEMONSTRATION")
    print("  Based on DeepMind Paper (pages 16-17)")
    print("="*60 + "\n")

    # Demo 1: Toy problem
    try:
        results = demo_funnel_inference_toy()
    except Exception as e:
        print(f"[!] Toy demo error: {e}")

    # Demo 2: Realistic scenario
    demo_realistic_scenario()

    # Demo 3: Funnel shape analysis
    try:
        analyze_funnel_shape()
    except Exception as e:
        print(f"[!] Funnel analysis error: {e}")

    print("\n" + "="*60)
    print("[+] Demonstration complete!")
    print("\nKey insights:")
    print("  - Funnel inference finds admissible λ via secant method")
    print("  - Residual has funnel shape with minimum at admissible value")
    print("  - Requires iterative training + residual evaluation")
    print("  - Converges in ~10 iterations for well-behaved problems")
    print("="*60)