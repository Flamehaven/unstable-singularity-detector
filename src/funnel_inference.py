"""
Funnel Inference for Lambda Optimization
Based on DeepMind paper "Discovery of Unstable Singularities" (pages 16-17)

Core Principle:
- At admissible λ*, equation residual r(λ) reaches minimum (funnel bottom)
- Use secant method to find zero of residual function
- Only valid near admissible values (local optimization)
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class FunnelInferenceConfig:
    """Configuration for Funnel Inference algorithm"""
    initial_lambda: float = 0.5
    delta_lambda: float = 1e-3  # Initial perturbation for secant method
    max_iterations: int = 20
    convergence_tol: float = 1e-6  # Convergence in lambda value
    residual_region: str = "origin"  # "origin" or "global"
    training_steps_per_eval: int = 5000  # Steps to train before evaluating residual
    min_training_loss: float = 1e-8  # Minimum loss to consider relaxed
    verbose: bool = True


class FunnelInference:
    """
    Funnel Inference algorithm for finding admissible lambda values

    Algorithm (from paper Eq. 17-18):
    1. Initialize with λ₀
    2. Train network with fixed λ until relaxed
    3. Measure residual r̂(λ) at origin
    4. Update λ using secant method: λₙ₊₁ = λₙ - r̂ₙ₋₁ · (λₙ₋₁ - λₙ)/(r̂ₙ₋₁ - r̂ₙ)
    5. Repeat until convergence
    """

    def __init__(self, config: FunnelInferenceConfig):
        self.config = config
        self.lambda_history = []
        self.residual_history = []
        self.training_loss_history = []
        self.iteration = 0

    def initialize(self, lambda_init: Optional[float] = None):
        """Initialize funnel inference with starting lambda"""
        if lambda_init is not None:
            self.config.initial_lambda = lambda_init

        self.lambda_history = [self.config.initial_lambda]
        self.residual_history = []
        self.training_loss_history = []
        self.iteration = 0

        logger.info(f"Funnel Inference initialized with λ₀ = {self.config.initial_lambda:.10f}")

    def compute_proxy_residual(self,
                               network: torch.nn.Module,
                               pde_system,
                               lambda_value: float,
                               evaluation_points: torch.Tensor) -> float:
        """
        Compute proxy residual r̂(λ) at evaluation points

        Paper methodology:
        - Focus on small neighborhood near origin
        - Measure residual with maximum absolute value
        - This captures admissibility signal

        Args:
            network: Trained PINN network
            pde_system: PDE system with residual computation
            lambda_value: Current lambda value
            evaluation_points: Points to evaluate (typically near origin)

        Returns:
            Signed residual value (maximum magnitude residual)
        """
        network.eval()

        with torch.no_grad():
            # Forward pass
            u_pred = network(evaluation_points)

            # Compute PDE residual at these points
            # This is problem-specific, delegated to pde_system
            residuals = pde_system.compute_residual(
                u_pred, evaluation_points, lambda_value
            )

            # Find residual with maximum absolute value (keep sign)
            abs_residuals = torch.abs(residuals)
            max_idx = torch.argmax(abs_residuals)
            signed_residual = residuals[max_idx].item()

        logger.debug(f"Proxy residual at λ={lambda_value:.6f}: r̂={signed_residual:.6e}")
        return signed_residual

    def secant_update(self) -> float:
        """
        Compute next lambda using secant method (Eq. 17)

        Formula: λₙ₊₁ = λₙ - r̂ₙ₋₁ · (λₙ₋₁ - λₙ)/(r̂ₙ₋₁ - r̂ₙ)

        Returns:
            Next lambda value
        """
        n = len(self.lambda_history)

        # For first iteration, use simple perturbation (Eq. 18)
        if n == 1:
            lambda_next = self.lambda_history[-1] + self.config.delta_lambda
            logger.info(f"Iteration 1: λ₁ = λ₀ + Δλ = {lambda_next:.10f}")
            return lambda_next

        # Secant method (need at least 2 history points)
        if n < 2 or len(self.residual_history) < 2:
            raise RuntimeError("Need at least 2 evaluations for secant method")

        λn = self.lambda_history[-1]
        λn_1 = self.lambda_history[-2]
        rn = self.residual_history[-1]
        rn_1 = self.residual_history[-2]

        # Avoid division by zero
        if abs(rn - rn_1) < 1e-15:
            logger.warning("Residuals too similar, perturbing lambda")
            lambda_next = λn + np.sign(rn) * self.config.delta_lambda
        else:
            # Secant formula
            lambda_next = λn - rn_1 * (λn_1 - λn) / (rn_1 - rn)

        logger.info(f"Secant update: λ{n} = {λn:.10f} → λ{n+1} = {lambda_next:.10f}")
        logger.debug(f"  r̂{n-2}={rn_1:.6e}, r̂{n-1}={rn:.6e}")

        return lambda_next

    def check_convergence(self) -> Tuple[bool, str]:
        """
        Check if funnel inference has converged

        Convergence criteria:
        1. Lambda change is small
        2. Residual is near zero
        3. Max iterations reached

        Returns:
            (converged: bool, reason: str)
        """
        if len(self.lambda_history) < 2:
            return False, "Insufficient history"

        # Check lambda convergence
        lambda_change = abs(self.lambda_history[-1] - self.lambda_history[-2])
        if lambda_change < self.config.convergence_tol:
            return True, f"Lambda converged (Δλ = {lambda_change:.2e})"

        # Check residual near zero
        if len(self.residual_history) > 0:
            current_residual = abs(self.residual_history[-1])
            if current_residual < self.config.convergence_tol * 10:
                return True, f"Residual near zero (r̂ = {current_residual:.2e})"

        # Check max iterations
        if self.iteration >= self.config.max_iterations:
            return True, f"Max iterations reached ({self.config.max_iterations})"

        return False, "Continuing"

    def run_iteration(self,
                     network: torch.nn.Module,
                     pde_system,
                     train_function: Callable,
                     evaluation_points: torch.Tensor) -> Dict:
        """
        Run one iteration of funnel inference

        Steps:
        1. Train network with current lambda (fixed)
        2. Evaluate proxy residual r̂(λ)
        3. Update lambda using secant method
        4. Check convergence

        Args:
            network: PINN network
            pde_system: PDE system
            train_function: Function that trains network with fixed lambda
            evaluation_points: Points to evaluate residual

        Returns:
            Iteration info dictionary
        """
        current_lambda = self.lambda_history[-1]

        logger.info(f"\n[Iteration {self.iteration + 1}] λ = {current_lambda:.10f}")
        logger.info("-" * 60)

        # 1. Train with fixed lambda
        logger.info(f"Training network with λ = {current_lambda:.10f}...")
        train_info = train_function(
            network=network,
            pde_system=pde_system,
            lambda_fixed=current_lambda,
            max_steps=self.config.training_steps_per_eval
        )

        final_loss = train_info.get('final_loss', float('inf'))
        self.training_loss_history.append(final_loss)

        # Check if training converged sufficiently
        if final_loss > self.config.min_training_loss:
            logger.warning(f"Training loss {final_loss:.2e} > target {self.config.min_training_loss:.2e}")

        # 2. Evaluate proxy residual
        proxy_residual = self.compute_proxy_residual(
            network, pde_system, current_lambda, evaluation_points
        )
        self.residual_history.append(proxy_residual)

        logger.info(f"Proxy residual: r̂(λ) = {proxy_residual:.6e}")

        # 3. Update lambda using secant method
        if self.iteration < self.config.max_iterations - 1:
            lambda_next = self.secant_update()
            self.lambda_history.append(lambda_next)

        # 4. Check convergence
        converged, reason = self.check_convergence()
        self.iteration += 1

        info = {
            'iteration': self.iteration,
            'lambda': current_lambda,
            'proxy_residual': proxy_residual,
            'training_loss': final_loss,
            'converged': converged,
            'convergence_reason': reason
        }

        if converged:
            logger.info(f"\n[+] Converged: {reason}")
            logger.info(f"[+] Final λ = {self.lambda_history[-1]:.10f}")

        return info

    def optimize(self,
                network: torch.nn.Module,
                pde_system,
                train_function: Callable,
                evaluation_points: torch.Tensor) -> Dict:
        """
        Run full funnel inference optimization

        Returns:
            Optimization results with final lambda
        """
        logger.info("=" * 60)
        logger.info("Starting Funnel Inference Optimization")
        logger.info("=" * 60)

        iteration_results = []

        while self.iteration < self.config.max_iterations:
            info = self.run_iteration(
                network, pde_system, train_function, evaluation_points
            )
            iteration_results.append(info)

            if info['converged']:
                break

        final_lambda = self.lambda_history[-1]
        final_residual = self.residual_history[-1] if self.residual_history else float('inf')

        results = {
            'final_lambda': final_lambda,
            'final_residual': final_residual,
            'iterations': self.iteration,
            'lambda_history': self.lambda_history.copy(),
            'residual_history': self.residual_history.copy(),
            'iteration_results': iteration_results,
            'converged': iteration_results[-1]['converged'] if iteration_results else False
        }

        logger.info("=" * 60)
        logger.info(f"Funnel Inference Complete")
        logger.info(f"  Final λ: {final_lambda:.10f}")
        logger.info(f"  Final r̂: {final_residual:.6e}")
        logger.info(f"  Iterations: {self.iteration}")
        logger.info("=" * 60)

        return results

    def plot_funnel(self, save_path: Optional[str] = None):
        """
        Visualize funnel inference convergence

        Creates Figure 5/6 style plots:
        - Lambda vs iteration
        - Residual vs iteration
        - Residual vs lambda (funnel shape)
        """
        if len(self.lambda_history) < 2:
            logger.warning("Insufficient data to plot funnel")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Lambda convergence
        ax1 = axes[0]
        iterations = range(len(self.lambda_history))
        ax1.plot(iterations, self.lambda_history, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Lambda (λ)', fontsize=12)
        ax1.set_title('Lambda Convergence', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Residual evolution
        ax2 = axes[1]
        if self.residual_history:
            ax2.plot(range(len(self.residual_history)), self.residual_history, 's-',
                    linewidth=2, markersize=8, color='red')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Proxy Residual r̂(λ)', fontsize=12)
            ax2.set_title('Residual Evolution', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        # 3. Funnel plot (residual vs lambda)
        ax3 = axes[2]
        if self.residual_history:
            # Plot actual trajectory
            lambda_vals = self.lambda_history[:len(self.residual_history)]
            ax3.plot(lambda_vals, self.residual_history, 'o-',
                    linewidth=2, markersize=8, color='blue', label='Trajectory')

            # Mark start and end
            ax3.plot(lambda_vals[0], self.residual_history[0], 'g^',
                    markersize=12, label='Start')
            ax3.plot(lambda_vals[-1], self.residual_history[-1], 'r*',
                    markersize=15, label='Final')

            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3, label='Zero')
            ax3.set_xlabel('Lambda (λ)', fontsize=12)
            ax3.set_ylabel('Proxy Residual r̂(λ)', fontsize=12)
            ax3.set_title('Funnel Shape', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Funnel plot saved to {save_path}")

        plt.show()


def create_evaluation_points_near_origin(n_points: int = 20,
                                         radius: float = 0.1,
                                         dim: int = 2) -> torch.Tensor:
    """
    Create evaluation points in small neighborhood near origin

    Paper methodology: Focus residual evaluation near singularity location

    Args:
        n_points: Number of evaluation points
        radius: Radius of neighborhood
        dim: Spatial dimension (2D or 3D)

    Returns:
        Evaluation points tensor [n_points, dim]
    """
    if dim == 2:
        # Polar grid near origin
        angles = torch.linspace(0, 2 * np.pi, n_points)
        radii = torch.linspace(0.01, radius, 5)

        points = []
        for r in radii:
            for theta in angles:
                x = r * torch.cos(theta)
                y = r * torch.sin(theta)
                points.append([x.item(), y.item()])

        return torch.tensor(points, dtype=torch.float64)

    elif dim == 3:
        # Spherical grid near origin
        n_theta = int(np.sqrt(n_points))
        n_phi = n_theta

        theta = torch.linspace(0, np.pi, n_theta)
        phi = torch.linspace(0, 2 * np.pi, n_phi)

        points = []
        for r in torch.linspace(0.01, radius, 3):
            for t in theta:
                for p in phi:
                    x = r * torch.sin(t) * torch.cos(p)
                    y = r * torch.sin(t) * torch.sin(p)
                    z = r * torch.cos(t)
                    points.append([x.item(), y.item(), z.item()])

        return torch.tensor(points, dtype=torch.float64)

    else:
        raise ValueError(f"Unsupported dimension: {dim}")