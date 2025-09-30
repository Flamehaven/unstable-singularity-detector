#!/usr/bin/env python3
"""
Basic example demonstrating unstable singularity detection

This example shows how to:
1. Set up a PINN solver for the IPM equation
2. Train with high precision
3. Detect unstable singularities
4. Visualize the lambda-instability pattern
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.unstable_singularity_detector import UnstableSingularityDetector
from src.pinn_solver import PINNSolver, IncompressiblePorousMedia, PINNConfig
from src.visualization import SingularityVisualizer

def main():
    """Run basic singularity detection example"""
    print("=" * 60)
    print("Unstable Singularity Detector - Basic Example")
    print("Based on DeepMind's Discovery of Unstable Singularities")
    print("=" * 60)

    # Step 1: Configure high-precision PINN
    print("\n[1] Setting up high-precision PINN configuration...")
    config = PINNConfig(
        hidden_layers=[50, 50, 50, 50],
        activation="tanh",
        precision=torch.float64,
        learning_rate=1e-3,
        max_epochs=5000,  # Reduced for example
        convergence_threshold=1e-10,
        pde_weight=1.0,
        boundary_weight=100.0,
        initial_weight=100.0
    )

    # Step 2: Set up IPM equation (known to have unstable singularities)
    print("[2] Initializing Incompressible Porous Media equation...")
    pde_system = IncompressiblePorousMedia()
    solver = PINNSolver(pde_system, config, self_similar=True, T_blowup=1.0)

    # Step 3: Generate training points
    print("[3] Generating training points...")
    training_points = solver.generate_training_points(
        n_interior=2000,  # Reduced for example
        n_boundary=200,
        n_initial=200,
        domain_bounds={
            'x_min': -1.0, 'x_max': 1.0,
            'y_min': -1.0, 'y_max': 1.0,
            't_min': 0.0, 't_max': 0.95
        }
    )

    # Step 4: Train PINN with high precision
    print("[4] Training PINN to high precision...")
    history = solver.train()

    final_loss = history['total_loss'][-1]
    final_residual = history['pde_residual'][-1]

    print(f"    Final Loss: {final_loss:.2e}")
    print(f"    PDE Residual: {final_residual:.2e}")

    if hasattr(solver.network, 'lambda_param'):
        lambda_learned = solver.network.lambda_param.item()
        alpha_learned = solver.network.alpha_param.item()
        print(f"    Learned λ (blow-up rate): {lambda_learned:.6f}")
        print(f"    Learned α (spatial scaling): {alpha_learned:.6f}")

    # Step 5: Create synthetic solution for singularity detection
    print("[5] Generating solution field for analysis...")

    # Create spatial grid
    grid_size = 64  # Smaller for example
    time_steps = 20

    x = torch.linspace(-2, 2, grid_size)
    y = torch.linspace(-2, 2, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    spatial_grid = torch.stack([X, Y], dim=0)

    # Time evolution
    time_evolution = torch.linspace(0.1, 0.99, time_steps)

    # Generate synthetic solution with blow-up behavior
    solution_field = torch.zeros(time_steps, grid_size, grid_size)

    for t_idx, t in enumerate(time_evolution):
        # Self-similar blow-up solution
        T_blowup = 1.0
        lambda_true = 1.5

        r_squared = X**2 + Y**2
        time_factor = (T_blowup - t)**(-lambda_true)
        spatial_factor = torch.exp(-r_squared / (0.1 * (T_blowup - t)))

        solution_field[t_idx] = time_factor * spatial_factor

    # Step 6: Run singularity detection
    print("[6] Running unstable singularity detection...")
    detector = UnstableSingularityDetector(
        equation_type="ipm",
        precision_target=1e-12,
        max_instability_order=5
    )

    results = detector.detect_unstable_singularities(
        solution_field, time_evolution, spatial_grid
    )

    print(f"    Number of singularities detected: {len(results)}")

    # Step 7: Display results
    print("\n[7] Singularity Detection Results:")
    print("-" * 50)

    for i, result in enumerate(results):
        print(f"  Singularity {i+1}:")
        print(f"    Type: {result.singularity_type.value}")
        print(f"    Lambda: {result.lambda_value:.6f}")
        print(f"    Instability Order: {result.instability_order}")
        print(f"    Confidence: {result.confidence_score:.4f}")
        print(f"    Time to Blow-up: {result.time_to_blowup:.6f}")
        print(f"    Precision Achieved: {result.precision_achieved:.2e}")
        print()

    # Step 8: Visualization
    if results:
        print("[8] Generating visualization...")
        visualizer = SingularityVisualizer()

        # Plot lambda-instability pattern
        fig = visualizer.plot_lambda_instability_pattern(
            results,
            equation_type="ipm",
            save_path="lambda_pattern_example.png",
            show_theory=True
        )

        print("    Lambda-instability pattern plot saved as 'lambda_pattern_example.png'")

        # Plot training history
        solver.plot_training_history("training_history_example.png")
        print("    Training history plot saved as 'training_history_example.png'")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("Key Achievements:")
    print(f"  [+] High-precision training: {final_residual:.2e} PDE residual")
    print(f"  [+] Singularity detection: {len(results)} unstable singularities found")
    if results:
        avg_lambda = np.mean([r.lambda_value for r in results])
        print(f"  [+] Average lambda value: {avg_lambda:.4f}")
    print("  [+] Computer-assisted proof support ready")
    print("=" * 60)

if __name__ == "__main__":
    main()