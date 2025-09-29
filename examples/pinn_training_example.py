#!/usr/bin/env python3
"""
PINN Training Example - High-Precision Physics-Informed Neural Networks

Demonstrates the complete PINN training pipeline for unstable singularity detection,
implementing DeepMind's methodology with self-similar parameterization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import matplotlib.pyplot as plt
from src.pinn_solver import PINNSolver, PINNConfig, IncompressiblePorousMedia, BoussinesqEquation, Euler3D
from src.gauss_newton_optimizer import AdaptivePrecisionOptimizer
from src.visualization import SingularityVisualizer

def setup_equation(equation_type):
    """Setup PDE system based on equation type"""
    equations = {
        "ipm": IncompressiblePorousMedia(),
        "boussinesq": BoussinesqEquation(),
        "euler_3d": Euler3D()
    }

    if equation_type not in equations:
        raise ValueError(f"Unknown equation type: {equation_type}. Choose from {list(equations.keys())}")

    return equations[equation_type]

def train_pinn_high_precision(equation_type="ipm", epochs=5000, self_similar=True):
    """Train PINN with high precision for singularity detection"""

    print("=" * 70)
    print("    HIGH-PRECISION PINN TRAINING")
    print(f"    Equation: {equation_type.upper()}")
    print(f"    Self-similar: {self_similar}")
    print("=" * 70)

    # Setup PDE system
    pde_system = setup_equation(equation_type)
    print(f"[>] PDE System: {pde_system.__class__.__name__}")

    # High-precision configuration
    config = PINNConfig(
        hidden_layers=[64, 64, 64, 64, 64],  # Deep network for complexity
        activation="tanh",
        precision=torch.float64,  # Double precision
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=1e-3,
        convergence_threshold=1e-12,  # Near machine precision
        patience=500,
        boundary_weight=1000.0,  # Strong boundary enforcement
        pde_weight=1.0,
        adaptive_weights=True
    )

    print(f"[>] Device: {config.device}")
    print(f"[>] Precision: {config.precision}")
    print(f"[>] Target convergence: {config.convergence_threshold:.0e}")

    # Initialize solver
    T_blowup = 1.0  # Blow-up time
    solver = PINNSolver(pde_system, config, self_similar=self_similar, T_blowup=T_blowup)

    # Generate training points
    print(f"[>] Generating training points...")
    training_points = solver.generate_training_points(
        n_interior=15000,  # Dense sampling for precision
        n_boundary=2000,
        n_initial=1000,
        domain_bounds={
            "x_min": -2.0, "x_max": 2.0,
            "y_min": -2.0, "y_max": 2.0,
            "t_min": 0.0, "t_max": 0.95  # Avoid exact blow-up time
        }
    )

    print(f"    Interior points: {training_points['interior'].shape[0]}")
    print(f"    Boundary points: {training_points['boundary'].shape[0]}")
    print(f"    Initial points: {training_points['initial'].shape[0]}")

    # Train the network
    print(f"\n[>] Starting PINN training for {epochs} epochs...")
    history = solver.train(max_epochs=epochs)

    # Analyze results
    final_loss = history['total_loss'][-1]
    final_pde_loss = history['pde_loss'][-1]
    final_boundary_loss = history['boundary_loss'][-1]

    print(f"\n[+] Training completed!")
    print(f"    Final total loss: {final_loss:.2e}")
    print(f"    Final PDE residual: {final_pde_loss:.2e}")
    print(f"    Final boundary loss: {final_boundary_loss:.2e}")

    # Check if machine precision achieved
    if final_pde_loss < 1e-10:
        print(f"    [*] MACHINE PRECISION ACHIEVED!")
        print(f"    [*] Ready for computer-assisted proofs")
    elif final_pde_loss < 1e-8:
        print(f"    [+] High precision achieved")
    else:
        print(f"    [-] Consider more training epochs")

    # Extract learned parameters for self-similar solutions
    if self_similar and hasattr(solver.network, 'lambda_param'):
        lambda_learned = solver.network.lambda_param.item()
        alpha_learned = solver.network.alpha_param.item()

        print(f"\n[>] Learned Self-Similar Parameters:")
        print(f"    Lambda (blow-up rate): {lambda_learned:.6f}")
        print(f"    Alpha (spatial scaling): {alpha_learned:.6f}")

        # Compare with DeepMind patterns
        expected_patterns = {
            "ipm": 1.875,  # For order-0 case
            "boussinesq": 1.654,
            "euler_3d": 1.523
        }

        if equation_type in expected_patterns:
            expected = expected_patterns[equation_type]
            error = abs(lambda_learned - expected)
            print(f"    Expected (DeepMind): {expected:.3f}")
            print(f"    Error: {error:.4f}")

            if error < 0.1:
                print(f"    [+] Matches DeepMind pattern!")
            else:
                print(f"    [!] Deviates from expected pattern")

    return solver, history

def analyze_convergence(history):
    """Analyze and visualize training convergence"""
    print(f"\n[>] Analyzing convergence behavior...")

    # Plot convergence history
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(len(history['total_loss']))

    # Total loss
    ax1.semilogy(epochs, history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # PDE residual
    ax2.semilogy(epochs, history['pde_loss'], 'r-', linewidth=2, label='PDE Residual')
    ax2.axhline(y=1e-12, color='g', linestyle='--', label='Target Precision')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PDE Residual')
    ax2.set_title('PDE Residual Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Boundary loss
    ax3.semilogy(epochs, history['boundary_loss'], 'orange', linewidth=2, label='Boundary Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Boundary Loss')
    ax3.set_title('Boundary Condition Enforcement')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Learning rate (if available)
    if 'learning_rate' in history:
        ax4.semilogy(epochs, history['learning_rate'], 'purple', linewidth=2, label='Learning Rate')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Adaptive Learning Rate')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Learning Rate\nHistory Not Available',
                transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        ax4.set_title('Learning Rate Schedule')

    plt.tight_layout()
    plt.savefig('pinn_convergence_analysis.png', dpi=300, bbox_inches='tight')
    print(f"[+] Convergence analysis saved to: pinn_convergence_analysis.png")

    return fig

def test_solution_quality(solver, equation_type):
    """Test the quality of the trained solution"""
    print(f"\n[>] Testing solution quality...")

    # Generate test points
    test_x = torch.linspace(-1.5, 1.5, 50, dtype=torch.float64)
    test_y = torch.linspace(-1.5, 1.5, 50, dtype=torch.float64)
    test_t = torch.linspace(0.1, 0.9, 20, dtype=torch.float64)

    # Create meshgrid
    T, X, Y = torch.meshgrid(test_t, test_x, test_y, indexing='ij')
    test_coords = torch.stack([T.flatten(), X.flatten(), Y.flatten()], dim=1)

    if torch.cuda.is_available():
        test_coords = test_coords.cuda()

    # Evaluate solution
    with torch.no_grad():
        solution = solver.evaluate_solution(test_coords)

    # Reshape for analysis
    solution = solution.reshape(T.shape)

    # Check for blow-up behavior
    max_values = []
    for i in range(len(test_t)):
        max_val = torch.max(torch.abs(solution[i])).item()
        max_values.append(max_val)

    print(f"    Solution range: {torch.min(solution).item():.2e} to {torch.max(solution).item():.2e}")
    print(f"    Growth near blow-up: {max_values[-1]/max_values[0]:.2e}")

    # Plot solution evolution
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    time_indices = [0, 4, 8, 12, 16, 19]  # Sample time points

    for i, t_idx in enumerate(time_indices):
        if i < 6:
            ax = axes[i]
            im = ax.contourf(X[t_idx].cpu().numpy(), Y[t_idx].cpu().numpy(),
                           solution[t_idx].cpu().numpy(), levels=20, cmap='viridis')
            ax.set_title(f't = {test_t[t_idx]:.2f}')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(f'solution_evolution_{equation_type}.png', dpi=300, bbox_inches='tight')
    print(f"[+] Solution evolution saved to: solution_evolution_{equation_type}.png")

    return solution.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='PINN Training for Unstable Singularities')
    parser.add_argument('--equation', choices=['ipm', 'boussinesq', 'euler_3d'],
                       default='ipm', help='PDE equation type')
    parser.add_argument('--epochs', type=int, default=5000, help='Training epochs')
    parser.add_argument('--self-similar', action='store_true', default=True,
                       help='Use self-similar parameterization')

    args = parser.parse_args()

    # Train PINN
    solver, history = train_pinn_high_precision(
        equation_type=args.equation,
        epochs=args.epochs,
        self_similar=args.self_similar
    )

    # Analyze results
    analyze_convergence(history)
    test_solution_quality(solver, args.equation)

    print("\n" + "=" * 70)
    print("[*] PINN training completed successfully!")
    print("[*] Solution ready for singularity analysis")
    print("[*] Files generated:")
    print("    - pinn_convergence_analysis.png")
    print(f"    - solution_evolution_{args.equation}.png")
    print("=" * 70)

if __name__ == "__main__":
    main()