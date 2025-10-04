"""
2D Boussinesq E2E Pipeline

Demonstrates Boussinesq equation solving:
- 2D grid (faster than 3D)
- Temperature-driven convection
- Lambda prediction for Boussinesq
- Full multistage training
- Conservation of energy verification

Usage:
    python examples/e2e_boussinesq_2d.py --output results/boussinesq_2d
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unstable_singularity_detector import UnstableSingularityDetector
from physics.bc import apply_boundary_conditions


class Boussinesq2DNetwork(nn.Module):
    """PINN for 2D Boussinesq equation."""

    def __init__(self, layers=[2, 64, 64, 64, 2]):  # 2 inputs (x,y), 2 outputs (u,T)
        super().__init__()
        self.activation = nn.Tanh()

        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))

        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        h = inputs
        for linear in self.linears[:-1]:
            h = self.activation(linear(h))
        output = self.linears[-1](h)
        u = output[:, 0:1]  # Velocity
        T = output[:, 1:2]  # Temperature
        return u, T


class Boussinesq2DPipeline:
    """Complete 2D Boussinesq E2E pipeline."""

    def __init__(self, output_dir: Path, grid_size: int = 64):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grid_size = grid_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

        self.results = {
            "metadata": self._collect_metadata(),
            "initial": {},
            "funnel": {},
            "training": {},
            "final": {}
        }

        print(f"[*] 2D Boussinesq Pipeline")
        print(f"    Grid: {grid_size}x{grid_size}")
        print(f"    Device: {self.device}")

    def _collect_metadata(self):
        import subprocess
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except:
            git_commit = "unknown"

        return {
            "timestamp": datetime.now().isoformat(),
            "git_commit": git_commit,
            "equation": "boussinesq_2d",
            "grid_size": self.grid_size
        }

    def step1_initial_conditions(self):
        """Setup 2D Boussinesq initial conditions."""
        print("\n[Step 1] 2D Boussinesq Initial Conditions")

        # 2D grid
        x = torch.linspace(0, 1, self.grid_size, dtype=self.dtype, device=self.device)
        y = torch.linspace(0, 1, self.grid_size, dtype=self.dtype, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Initial temperature: T(x,y,0) = sin(pi*x)*sin(pi*y)
        T0 = torch.sin(torch.pi * X) * torch.sin(torch.pi * Y)

        # Initial velocity: u(x,y,0) = 0
        u0 = torch.zeros_like(T0)

        # Apply boundary conditions (T=0, u=0 on boundaries)
        T0_bc = T0.clone()
        T0_bc[0, :] = 0.0
        T0_bc[-1, :] = 0.0
        T0_bc[:, 0] = 0.0
        T0_bc[:, -1] = 0.0

        # Conservation: initial energy
        initial_energy = torch.sum(T0_bc ** 2).item()

        print(f"  Initial temperature: sin(pi*x)*sin(pi*y)")
        print(f"  Initial velocity: 0")
        print(f"  Initial energy: {initial_energy:.6f}")
        print(f"  Boundary: Dirichlet (T=0, u=0)")

        self.results["initial"]["energy"] = initial_energy

        return {
            "X": X,
            "Y": Y,
            "T0": T0_bc,
            "u0": u0
        }

    def step2_lambda_prediction(self):
        """Predict lambda for Boussinesq."""
        print("\n[Step 2] Lambda Prediction (Boussinesq)")

        detector = UnstableSingularityDetector(equation_type="boussinesq")

        lambda_0 = detector.predict_next_unstable_lambda(current_order=0)
        lambda_1 = detector.predict_next_unstable_lambda(current_order=1)

        print(f"  Lambda_0 (stable): {lambda_0:.10f}")
        print(f"  Lambda_1 (unstable): {lambda_1:.10f}")
        print(f"  Ground truth (paper): Lambda_0 = 2.4142135624")
        print(f"  Error: {abs(lambda_0 - 2.4142135624) / 2.4142135624 * 100:.3f}%")

        self.results["funnel"]["lambda_0"] = lambda_0
        self.results["funnel"]["lambda_1"] = lambda_1

        return lambda_0

    def step3_funnel_training(self, lambda_init: float):
        """Funnel inference for 2D Boussinesq."""
        print(f"\n[Step 3] Funnel Inference")

        net = Boussinesq2DNetwork().to(self.device).to(self.dtype)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        lambda_current = lambda_init
        residual_history = []

        for iteration in range(10):
            epoch_losses = []

            for epoch in range(30):
                x = torch.rand(256, 1, dtype=self.dtype, device=self.device, requires_grad=True)
                y = torch.rand(256, 1, dtype=self.dtype, device=self.device, requires_grad=True)

                u, T = net(x, y)

                # Compute derivatives
                T_x = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
                T_xx = torch.autograd.grad(T_x.sum(), x, create_graph=True)[0]
                T_y = torch.autograd.grad(T.sum(), y, create_graph=True)[0]
                T_yy = torch.autograd.grad(T_y.sum(), y, create_graph=True)[0]

                # Boussinesq residual (simplified): nabla^2 T + lambda * u * T = 0
                laplacian_T = T_xx + T_yy
                residual = laplacian_T + lambda_current * u * T

                loss = torch.mean(residual ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_residual = np.mean(epoch_losses)
            residual_history.append(avg_residual)

            print(f"  Iter {iteration+1}: lambda={lambda_current:.6f}, residual={avg_residual:.2e}")

            if avg_residual < 5e-5:
                print(f"  [+] Converged")
                break

            # Update lambda
            if iteration > 0:
                lambda_current = lambda_current + 0.01 * np.sign(residual_history[-2] - residual_history[-1])

        self.results["funnel"]["final_lambda"] = lambda_current
        self.results["funnel"]["residual_history"] = residual_history

        return {
            "lambda_star": lambda_current,
            "net": net
        }

    def step4_multistage_training(self, lambda_star: float, net_init: nn.Module):
        """Multi-stage training."""
        print("\n[Step 4] Multi-stage Training")

        net = net_init

        # Stage 1: Adam
        print("  [Stage 1] Adam -> 1e-6")
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

        stage1_losses = []
        for epoch in range(150):
            x = torch.rand(512, 1, dtype=self.dtype, device=self.device, requires_grad=True)
            y = torch.rand(512, 1, dtype=self.dtype, device=self.device, requires_grad=True)

            u, T = net(x, y)

            T_x = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
            T_xx = torch.autograd.grad(T_x.sum(), x, create_graph=True)[0]
            T_y = torch.autograd.grad(T.sum(), y, create_graph=True)[0]
            T_yy = torch.autograd.grad(T_y.sum(), y, create_graph=True)[0]

            laplacian = T_xx + T_yy
            residual = laplacian + lambda_star * u * T
            loss = torch.mean(residual ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stage1_losses.append(loss.item())

            if epoch % 50 == 0:
                print(f"    Epoch {epoch}: loss={loss.item():.2e}")

        print(f"  Stage 1 final: {stage1_losses[-1]:.2e}")

        # Stage 2: Refinement
        print("  [Stage 2] Refinement -> 1e-8")
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

        stage2_losses = []
        for epoch in range(80):
            x = torch.rand(512, 1, dtype=self.dtype, device=self.device, requires_grad=True)
            y = torch.rand(512, 1, dtype=self.dtype, device=self.device, requires_grad=True)

            u, T = net(x, y)

            T_x = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
            T_xx = torch.autograd.grad(T_x.sum(), x, create_graph=True)[0]
            T_y = torch.autograd.grad(T.sum(), y, create_graph=True)[0]
            T_yy = torch.autograd.grad(T_y.sum(), y, create_graph=True)[0]

            laplacian = T_xx + T_yy
            residual = laplacian + lambda_star * u * T
            loss = torch.mean(residual ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stage2_losses.append(loss.item())

        print(f"  Stage 2 final: {stage2_losses[-1]:.2e}")

        self.results["training"]["stage1_final"] = stage1_losses[-1]
        self.results["training"]["stage2_final"] = stage2_losses[-1]
        self.results["training"]["stage1_history"] = stage1_losses[::10]
        self.results["training"]["stage2_history"] = stage2_losses[::5]

        return {
            "net": net,
            "stage1_residual": stage1_losses[-1],
            "stage2_residual": stage2_losses[-1]
        }

    def step5_conservation_check(self, net: nn.Module, initial_energy: float):
        """Check energy conservation."""
        print("\n[Step 5] Energy Conservation Check")

        x = torch.linspace(0, 1, self.grid_size, dtype=self.dtype, device=self.device)
        y = torch.linspace(0, 1, self.grid_size, dtype=self.dtype, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        with torch.no_grad():
            _, T_final = net(X.reshape(-1, 1), Y.reshape(-1, 1))
            final_energy = torch.sum(T_final ** 2).item()

        violation = abs(final_energy - initial_energy) / abs(initial_energy) if initial_energy != 0 else abs(final_energy)

        print(f"  Initial energy: {initial_energy:.6f}")
        print(f"  Final energy: {final_energy:.6f}")
        print(f"  Violation: {violation:.2e}")
        print(f"  Status: {'PASS' if violation < 0.1 else 'FAIL'}")

        self.results["final"]["initial_energy"] = initial_energy
        self.results["final"]["final_energy"] = final_energy
        self.results["final"]["violation"] = violation

        return violation

    def generate_report(self):
        """Generate PDF report."""
        print("\n[Report] Generating PDF...")

        pdf_path = self.output_dir / "boussinesq_2d_report.pdf"

        with PdfPages(pdf_path) as pdf:
            # Page 1: Funnel convergence
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.semilogy(self.results["funnel"]["residual_history"], 'b-', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Residual')
            ax.set_title('2D Boussinesq: Lambda Funnel Convergence')
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 2: Training stages
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.semilogy(self.results["training"]["stage1_history"], 'b-')
            ax1.set_title('Stage 1: Adam')
            ax1.set_xlabel('Epoch (subsampled)')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)

            ax2.semilogy(self.results["training"]["stage2_history"], 'r-')
            ax2.set_title('Stage 2: Refinement')
            ax2.set_xlabel('Epoch (subsampled)')
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Page 3: Summary
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            ax.axis('off')

            summary = f"""
2D Boussinesq Pipeline Summary

Equation: Boussinesq (temperature-driven convection)
Grid: {self.results['metadata']['grid_size']}x{self.results['metadata']['grid_size']}
Timestamp: {self.results['metadata']['timestamp']}

Lambda Prediction:
  Predicted λ₀: {self.results['funnel']['lambda_0']:.6f}
  Ground truth: 2.4142135624
  Error: {abs(self.results['funnel']['lambda_0'] - 2.4142135624) / 2.4142135624 * 100:.3f}%

Funnel Inference:
  Final λ: {self.results['funnel']['final_lambda']:.6f}
  Iterations: {len(self.results['funnel']['residual_history'])}

Multi-stage Training:
  Stage 1: {self.results['training']['stage1_final']:.2e}
  Stage 2: {self.results['training']['stage2_final']:.2e}

Energy Conservation:
  Initial: {self.results['final']['initial_energy']:.6f}
  Final: {self.results['final']['final_energy']:.6f}
  Violation: {self.results['final']['violation']:.2e}

Status: COMPLETE
            """

            ax.text(0.1, 0.5, summary, fontsize=11, family='monospace',
                   verticalalignment='center')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        print(f"  [+] PDF: {pdf_path}")

        # JSON metrics
        json_path = self.output_dir / "boussinesq_2d_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  [+] JSON: {json_path}")

    def run(self):
        """Execute pipeline."""
        start = time.time()

        print("=" * 60)
        print("2D BOUSSINESQ E2E PIPELINE")
        print("=" * 60)

        data = self.step1_initial_conditions()
        lambda_init = self.step2_lambda_prediction()
        funnel_result = self.step3_funnel_training(lambda_init)
        training_result = self.step4_multistage_training(funnel_result["lambda_star"], funnel_result["net"])
        self.step5_conservation_check(training_result["net"], self.results["initial"]["energy"])
        self.generate_report()

        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"COMPLETE ({elapsed:.1f}s)")
        print(f"Results: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/boussinesq_2d"))
    parser.add_argument("--grid-size", type=int, default=64)
    args = parser.parse_args()

    pipeline = Boussinesq2DPipeline(args.output, args.grid_size)
    pipeline.run()


if __name__ == "__main__":
    main()
