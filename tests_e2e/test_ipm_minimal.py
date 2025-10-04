"""
Minimal IPM E2E test for CI pipeline validation.

Tests the full pipeline from initial conditions to lambda convergence
on a reduced-scale problem suitable for continuous integration.
"""

import json
import pytest
import torch
from pathlib import Path

from src.unstable_singularity_detector import UnstableSingularityDetector
from src.funnel_inference import FunnelInference
from src.pinn_solver import PINNSolver


class TestIPMMinimal:
    """Minimal IPM end-to-end test suite."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory for test artifacts."""
        out_dir = tmp_path / "e2e_ipm"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def test_ipm_lambda_prediction(self):
        """Validate IPM lambda prediction accuracy (order 0)."""
        detector = UnstableSingularityDetector(equation_type="ipm")

        # Test order 0 (exact formula from paper Fig 2e)
        lambda_pred = detector.predict_next_unstable_lambda(
            previous_lambda=None,
            order=0
        )

        # Ground truth from DeepMind paper
        lambda_expected = 1.0285722760222
        error = abs(lambda_pred - lambda_expected) / lambda_expected

        assert error < 0.01, f"Lambda error {error:.4%} exceeds 1% threshold"

    def test_ipm_funnel_convergence_minimal(self, output_dir):
        """Test funnel inference convergence on minimal IPM problem."""

        # Minimal configuration for CI
        config = {
            "equation": "ipm",
            "grid_size": 16,
            "epochs": 50,
            "precision": "fp64",
            "target_residual": 1e-6
        }

        funnel = FunnelInference(
            equation_type="ipm",
            max_iterations=10,
            tolerance=1e-3
        )

        # Run funnel with reduced computational budget
        result = funnel.run(
            initial_guess=1.0,
            max_epochs=config["epochs"]
        )

        # Validate convergence
        assert result["converged"], "Funnel failed to converge"
        assert result["iterations"] <= 10, f"Too many iterations: {result['iterations']}"

        # Save metrics
        metrics = {
            "test": "ipm_funnel_minimal",
            "config": config,
            "final_lambda": float(result["lambda"]),
            "iterations": result["iterations"],
            "final_residual": float(result.get("residual", 0)),
            "converged": result["converged"]
        }

        with open(output_dir / "ipm_funnel_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    def test_ipm_boundary_conditions(self):
        """Validate boundary condition implementation."""
        from src.physics.bc import apply_boundary_conditions

        # Small test grid
        u = torch.randn(8, 8, 8, dtype=torch.float64)

        # Apply Dirichlet BC
        u_bc = apply_boundary_conditions(
            u,
            bc_type="dirichlet",
            bc_value=0.0
        )

        # Verify boundaries are zero
        assert torch.allclose(u_bc[0, :, :], torch.zeros_like(u_bc[0, :, :]))
        assert torch.allclose(u_bc[-1, :, :], torch.zeros_like(u_bc[-1, :, :]))
        assert torch.allclose(u_bc[:, 0, :], torch.zeros_like(u_bc[:, 0, :]))
        assert torch.allclose(u_bc[:, -1, :], torch.zeros_like(u_bc[:, -1, :]))

    def test_conservation_validation(self):
        """Test conservation law checking on minimal problem."""
        from src.utils.metrics import check_conservation

        # Synthetic conserved quantity
        initial_mass = 1.0
        final_mass = 1.0001

        violation = check_conservation(
            initial_value=initial_mass,
            final_value=final_mass,
            tolerance=1e-3
        )

        assert violation < 1e-3, f"Conservation violation: {violation}"


class TestLineageTracking:
    """Test result lineage and reproducibility tracking."""

    def test_metadata_collection(self, tmp_path):
        """Validate automatic metadata collection."""
        import subprocess
        import platform

        metadata = {
            "git_commit": subprocess.getoutput("git rev-parse HEAD").strip(),
            "git_branch": subprocess.getoutput("git branch --show-current").strip(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "precision": "fp64",
            "device": "cpu"
        }

        # Save lineage
        lineage_file = tmp_path / "lineage.json"
        with open(lineage_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Validate required fields
        assert len(metadata["git_commit"]) == 40, "Invalid git commit hash"
        assert metadata["torch_version"], "Missing PyTorch version"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
