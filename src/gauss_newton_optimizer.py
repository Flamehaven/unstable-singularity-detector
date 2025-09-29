"""
High-Precision Gauss-Newton Optimizer for PINNs
Based on DeepMind "Discovery of Unstable Singularities" (arXiv:2509.14185)

Core Features:
- Second-order optimization for near machine precision
- Adaptive damping for stability
- Line search for robust convergence
- Specialized for Physics-Informed Neural Networks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import warnings
from scipy.linalg import solve, LinAlgError
from torch.autograd import grad
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GaussNewtonConfig:
    """Configuration for Gauss-Newton optimizer"""
    learning_rate: float = 1e-3
    damping_factor: float = 1e-6  # Levenberg-Marquardt damping
    max_iterations: int = 1000
    tolerance: float = 1e-12  # Near machine precision target
    line_search: bool = True
    line_search_max_iter: int = 20
    line_search_c1: float = 1e-4  # Armijo condition parameter
    adaptive_damping: bool = True
    damping_increase: float = 10.0
    damping_decrease: float = 0.1
    gradient_clip: float = 1.0
    verbose: bool = True
    precision: torch.dtype = torch.float64

class ResidualFunction(ABC):
    """Abstract base class for residual functions"""

    @abstractmethod
    def compute_residual(self, parameters: torch.Tensor) -> torch.Tensor:
        """Compute residual vector r(θ) for parameters θ"""
        pass

    @abstractmethod
    def compute_jacobian(self, parameters: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian matrix J = ∂r/∂θ"""
        pass

class PINNResidualFunction(ResidualFunction):
    """
    Residual function for Physics-Informed Neural Networks

    Computes combined residual from:
    - PDE residual at collocation points
    - Boundary condition residuals
    - Initial condition residuals
    """

    def __init__(self, network: nn.Module, pde_system, training_points: Dict,
                 weights: Dict[str, float] = None):
        self.network = network
        self.pde_system = pde_system
        self.training_points = training_points

        # Default weights for different loss components
        if weights is None:
            weights = {'pde': 1.0, 'boundary': 100.0, 'initial': 100.0}
        self.weights = weights

        # Flatten network parameters for optimization
        self.param_shapes = []
        self.param_sizes = []

        for param in self.network.parameters():
            self.param_shapes.append(param.shape)
            self.param_sizes.append(param.numel())

        self.total_params = sum(self.param_sizes)
        logger.info(f"PINN has {self.total_params} parameters")

    def _set_network_parameters(self, flat_params: torch.Tensor):
        """Set network parameters from flattened vector"""
        start_idx = 0
        with torch.no_grad():
            for param, size, shape in zip(self.network.parameters(),
                                         self.param_sizes, self.param_shapes):
                param_slice = flat_params[start_idx:start_idx + size]
                param.data = param_slice.view(shape)
                start_idx += size

    def _get_network_parameters(self) -> torch.Tensor:
        """Get flattened network parameters"""
        params = []
        for param in self.network.parameters():
            params.append(param.data.flatten())
        return torch.cat(params)

    def _compute_derivatives(self, u: torch.Tensor,
                           coordinates: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute derivatives using automatic differentiation"""
        x, y, t = coordinates[:, 0:1], coordinates[:, 1:2], coordinates[:, 2:3]

        # First derivatives
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u.sum(), y, create_graph=True, retain_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True, retain_graph=True)[0]

        # Second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True, retain_graph=True)[0]
        u_zz = torch.zeros_like(u_xx)

        return {
            'u': u, 'u_t': u_t, 'u_x': u_x, 'u_y': u_y, 'u_z': torch.zeros_like(u_x),
            'u_xx': u_xx, 'u_yy': u_yy, 'u_zz': u_zz,
            'x': x, 'y': y, 'z': torch.zeros_like(x), 't': t
        }

    def compute_residual(self, parameters: torch.Tensor) -> torch.Tensor:
        """
        Compute combined residual vector for PINN

        Args:
            parameters: Flattened network parameters

        Returns:
            Residual vector [N_total, 1]
        """
        # Set network parameters
        self._set_network_parameters(parameters)

        residuals = []

        # 1. PDE residual at interior points
        interior_points = self.training_points['interior']
        interior_points.requires_grad_(True)

        u_interior = self.network(interior_points)
        derivatives = self._compute_derivatives(u_interior, interior_points)
        pde_residual = self.pde_system.pde_residual(**derivatives)

        weighted_pde = torch.sqrt(self.weights['pde']) * pde_residual
        residuals.append(weighted_pde.flatten())

        # 2. Boundary condition residual
        boundary_points = self.training_points['boundary']
        u_boundary = self.network(boundary_points)

        x_b, y_b, t_b = boundary_points[:, 0:1], boundary_points[:, 1:2], boundary_points[:, 2:3]
        z_b = torch.zeros_like(x_b)
        boundary_target = self.pde_system.boundary_condition(x_b, y_b, z_b, t_b)

        boundary_residual = u_boundary - boundary_target
        weighted_boundary = torch.sqrt(self.weights['boundary']) * boundary_residual
        residuals.append(weighted_boundary.flatten())

        # 3. Initial condition residual
        initial_points = self.training_points['initial']
        u_initial = self.network(initial_points)

        x_i, y_i, t_i = initial_points[:, 0:1], initial_points[:, 1:2], initial_points[:, 2:3]
        z_i = torch.zeros_like(x_i)
        initial_target = self.pde_system.initial_condition(x_i, y_i, z_i)

        initial_residual = u_initial - initial_target
        weighted_initial = torch.sqrt(self.weights['initial']) * initial_residual
        residuals.append(weighted_initial.flatten())

        # Combine all residuals
        total_residual = torch.cat(residuals)
        return total_residual

    def compute_jacobian(self, parameters: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian matrix J = ∂r/∂θ using automatic differentiation

        Args:
            parameters: Flattened network parameters [P]

        Returns:
            Jacobian matrix [N, P] where N is number of residuals, P is number of parameters
        """
        # Set network parameters
        self._set_network_parameters(parameters)

        # Compute residual
        residual = self.compute_residual(parameters)
        n_residuals = len(residual)

        # Compute Jacobian using autograd
        jacobian_rows = []

        for i in range(n_residuals):
            # Compute gradient of i-th residual w.r.t. all parameters
            if residual[i].requires_grad:
                grad_i = torch.autograd.grad(residual[i], self.network.parameters(),
                                           retain_graph=(i < n_residuals - 1),
                                           allow_unused=True)

                # Flatten and concatenate gradients
                grad_flat = []
                for g in grad_i:
                    if g is not None:
                        grad_flat.append(g.flatten())
                    else:
                        grad_flat.append(torch.zeros(self.param_sizes[len(grad_flat)]))

                jacobian_rows.append(torch.cat(grad_flat))
            else:
                jacobian_rows.append(torch.zeros(self.total_params))

        jacobian = torch.stack(jacobian_rows)
        return jacobian

class HighPrecisionGaussNewton:
    """
    High-precision Gauss-Newton optimizer with Levenberg-Marquardt damping

    Implements the methodology from DeepMind paper:
    - Second-order convergence for extreme precision
    - Adaptive damping for stability near singularities
    - Line search for robust convergence
    - Specialized for Physics-Informed Neural Networks
    """

    def __init__(self, residual_function: ResidualFunction, config: GaussNewtonConfig):
        self.residual_function = residual_function
        self.config = config

        # Optimization state
        self.current_parameters = None
        self.current_residual = None
        self.current_jacobian = None
        self.current_loss = None

        # History tracking
        self.loss_history = []
        self.gradient_norm_history = []
        self.damping_history = []
        self.step_size_history = []

        # Adaptive damping state
        self.damping = config.damping_factor

        logger.info("Initialized High-Precision Gauss-Newton Optimizer")
        logger.info(f"Target tolerance: {config.tolerance:.2e}")

    def _compute_loss(self, residual: torch.Tensor) -> float:
        """Compute loss as 0.5 * ||r||²"""
        return 0.5 * torch.sum(residual**2).item()

    def _compute_gradient(self, residual: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Compute gradient g = J^T r"""
        return torch.matmul(jacobian.T, residual)

    def _solve_gauss_newton_system(self, jacobian: torch.Tensor,
                                 residual: torch.Tensor) -> torch.Tensor:
        """
        Solve the Gauss-Newton system with Levenberg-Marquardt damping:
        (J^T J + λI) δ = -J^T r

        Args:
            jacobian: Jacobian matrix [N, P]
            residual: Residual vector [N]

        Returns:
            Step direction δ [P]
        """
        JTJ = torch.matmul(jacobian.T, jacobian)
        JTr = torch.matmul(jacobian.T, residual)

        # Add Levenberg-Marquardt damping
        n_params = JTJ.shape[0]
        damped_JTJ = JTJ + self.damping * torch.eye(n_params, dtype=self.config.precision)

        try:
            # Solve linear system
            step = torch.linalg.solve(damped_JTJ, -JTr)
            return step
        except RuntimeError as e:
            logger.warning(f"Linear system solve failed: {e}")
            # Fallback to gradient descent step
            gradient = self._compute_gradient(residual, jacobian)
            return -self.config.learning_rate * gradient

    def _line_search(self, parameters: torch.Tensor, step: torch.Tensor,
                    current_loss: float) -> Tuple[float, bool]:
        """
        Armijo line search to find appropriate step size

        Args:
            parameters: Current parameters
            step: Search direction
            current_loss: Current loss value

        Returns:
            Step size and success flag
        """
        if not self.config.line_search:
            return 1.0, True

        alpha = 1.0
        c1 = self.config.line_search_c1

        # Compute directional derivative
        current_residual = self.residual_function.compute_residual(parameters)
        current_jacobian = self.residual_function.compute_jacobian(parameters)
        gradient = self._compute_gradient(current_residual, current_jacobian)
        directional_derivative = torch.dot(gradient, step).item()

        for i in range(self.config.line_search_max_iter):
            # Try step
            test_params = parameters + alpha * step
            test_residual = self.residual_function.compute_residual(test_params)
            test_loss = self._compute_loss(test_residual)

            # Armijo condition
            if test_loss <= current_loss + c1 * alpha * directional_derivative:
                return alpha, True

            # Reduce step size
            alpha *= 0.5

        logger.warning("Line search failed, using small step")
        return 0.01, False

    def _update_damping(self, loss_reduction: float, predicted_reduction: float):
        """
        Update Levenberg-Marquardt damping factor based on step quality

        Args:
            loss_reduction: Actual loss reduction
            predicted_reduction: Predicted loss reduction from quadratic model
        """
        if not self.config.adaptive_damping:
            return

        if predicted_reduction > 0:
            ratio = loss_reduction / predicted_reduction

            if ratio > 0.75:
                # Good step, decrease damping
                self.damping *= self.config.damping_decrease
            elif ratio < 0.25:
                # Poor step, increase damping
                self.damping *= self.config.damping_increase
        else:
            # Increase damping if predicted reduction is not positive
            self.damping *= self.config.damping_increase

        # Clamp damping to reasonable range
        self.damping = torch.clamp(torch.tensor(self.damping), 1e-12, 1e6).item()

    def optimize(self, initial_parameters: torch.Tensor) -> Dict[str, any]:
        """
        Main optimization loop

        Args:
            initial_parameters: Starting parameters

        Returns:
            Optimization results dictionary
        """
        logger.info("Starting high-precision Gauss-Newton optimization...")

        # Initialize
        self.current_parameters = initial_parameters.clone().detach()
        self.current_parameters.requires_grad_(True)

        # History
        self.loss_history = []
        self.gradient_norm_history = []
        self.damping_history = []
        self.step_size_history = []

        start_time = time.time()

        for iteration in range(self.config.max_iterations):
            # Compute residual and Jacobian
            self.current_residual = self.residual_function.compute_residual(self.current_parameters)
            self.current_jacobian = self.residual_function.compute_jacobian(self.current_parameters)
            self.current_loss = self._compute_loss(self.current_residual)

            # Compute gradient
            gradient = self._compute_gradient(self.current_residual, self.current_jacobian)
            gradient_norm = torch.norm(gradient).item()

            # Record history
            self.loss_history.append(self.current_loss)
            self.gradient_norm_history.append(gradient_norm)
            self.damping_history.append(self.damping)

            # Check convergence
            if self.current_loss < self.config.tolerance:
                logger.info(f"Converged! Loss: {self.current_loss:.2e} < {self.config.tolerance:.2e}")
                break

            if gradient_norm < self.config.tolerance:
                logger.info(f"Gradient convergence! ||g||: {gradient_norm:.2e}")
                break

            # Compute Gauss-Newton step
            step = self._solve_gauss_newton_system(self.current_jacobian, self.current_residual)

            # Clip gradient for stability
            step_norm = torch.norm(step).item()
            if step_norm > self.config.gradient_clip:
                step = step * (self.config.gradient_clip / step_norm)
                step_norm = self.config.gradient_clip

            # Line search
            step_size, line_search_success = self._line_search(
                self.current_parameters, step, self.current_loss
            )
            self.step_size_history.append(step_size)

            # Predicted reduction (for damping update)
            predicted_reduction = -torch.dot(gradient, step).item() - 0.5 * step_size**2 * step_norm**2

            # Update parameters
            old_loss = self.current_loss
            self.current_parameters = self.current_parameters + step_size * step

            # Compute new loss for damping update
            new_residual = self.residual_function.compute_residual(self.current_parameters)
            new_loss = self._compute_loss(new_residual)
            loss_reduction = old_loss - new_loss

            # Update damping
            self._update_damping(loss_reduction, predicted_reduction)

            # Progress logging
            if self.config.verbose and (iteration % 100 == 0 or iteration < 10):
                logger.info(f"Iter {iteration:4d}: Loss={self.current_loss:.2e}, "
                          f"||g||={gradient_norm:.2e}, λ={self.damping:.2e}, "
                          f"step_size={step_size:.3f}")

        end_time = time.time()

        # Final results
        final_residual = self.residual_function.compute_residual(self.current_parameters)
        final_loss = self._compute_loss(final_residual)
        final_jacobian = self.residual_function.compute_jacobian(self.current_parameters)
        final_gradient = self._compute_gradient(final_residual, final_jacobian)
        final_gradient_norm = torch.norm(final_gradient).item()

        results = {
            'parameters': self.current_parameters,
            'loss': final_loss,
            'gradient_norm': final_gradient_norm,
            'iterations': iteration + 1,
            'converged': final_loss < self.config.tolerance or final_gradient_norm < self.config.tolerance,
            'optimization_time': end_time - start_time,
            'loss_history': self.loss_history,
            'gradient_norm_history': self.gradient_norm_history,
            'damping_history': self.damping_history,
            'step_size_history': self.step_size_history
        }

        logger.info(f"Optimization completed in {results['optimization_time']:.2f}s")
        logger.info(f"Final loss: {final_loss:.2e}")
        logger.info(f"Final gradient norm: {final_gradient_norm:.2e}")
        logger.info(f"Converged: {results['converged']}")

        return results

    def plot_convergence(self, save_path: Optional[str] = None):
        """Plot optimization convergence history"""
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Loss history
        ax1.semilogy(self.loss_history)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Convergence')
        ax1.grid(True, alpha=0.3)

        # Gradient norm history
        ax2.semilogy(self.gradient_norm_history)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_title('Gradient Norm Convergence')
        ax2.grid(True, alpha=0.3)

        # Damping history
        ax3.semilogy(self.damping_history)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Damping Factor')
        ax3.set_title('Levenberg-Marquardt Damping')
        ax3.grid(True, alpha=0.3)

        # Step size history
        ax4.plot(self.step_size_history)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Step Size')
        ax4.set_title('Line Search Step Size')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Convergence plot saved to {save_path}")

        plt.show()

class AdaptivePrecisionOptimizer:
    """
    Adaptive optimizer that switches between first-order and second-order methods
    based on convergence progress and precision requirements
    """

    def __init__(self, residual_function: ResidualFunction,
                 adam_lr: float = 1e-3, gn_config: GaussNewtonConfig = None):
        self.residual_function = residual_function
        self.adam_lr = adam_lr

        if gn_config is None:
            gn_config = GaussNewtonConfig()
        self.gn_config = gn_config

        # Switch threshold (when to move from Adam to Gauss-Newton)
        self.switch_threshold = 1e-6
        self.switched_to_gn = False

    def optimize(self, initial_parameters: torch.Tensor,
                adam_epochs: int = 5000) -> Dict[str, any]:
        """
        Two-stage optimization: Adam followed by Gauss-Newton

        Args:
            initial_parameters: Starting parameters
            adam_epochs: Number of Adam epochs before switching

        Returns:
            Combined optimization results
        """
        logger.info("Starting adaptive precision optimization...")

        # Stage 1: Adam optimization for initial convergence
        logger.info("Stage 1: Adam optimization...")
        parameters = initial_parameters.clone()

        # Create temporary network for Adam optimization
        temp_network = self.residual_function.network
        adam_optimizer = torch.optim.Adam(temp_network.parameters(), lr=self.adam_lr)

        adam_losses = []

        for epoch in range(adam_epochs):
            adam_optimizer.zero_grad()

            # Compute loss
            residual = self.residual_function.compute_residual(
                self.residual_function._get_network_parameters()
            )
            loss = 0.5 * torch.sum(residual**2)

            # Backward pass
            loss.backward()
            adam_optimizer.step()

            adam_losses.append(loss.item())

            # Check if ready to switch to Gauss-Newton
            if loss.item() < self.switch_threshold and epoch > 1000:
                logger.info(f"Switching to Gauss-Newton at epoch {epoch}, loss: {loss.item():.2e}")
                self.switched_to_gn = True
                break

            if epoch % 1000 == 0:
                logger.info(f"Adam epoch {epoch}: Loss = {loss.item():.2e}")

        # Get parameters after Adam
        adam_final_params = self.residual_function._get_network_parameters()

        # Stage 2: Gauss-Newton for high precision
        if self.switched_to_gn:
            logger.info("Stage 2: Gauss-Newton high-precision optimization...")
            gn_optimizer = HighPrecisionGaussNewton(self.residual_function, self.gn_config)
            gn_results = gn_optimizer.optimize(adam_final_params)
        else:
            logger.info("Adam reached max epochs, creating dummy GN results")
            gn_results = {
                'parameters': adam_final_params,
                'loss': adam_losses[-1],
                'gradient_norm': np.nan,
                'iterations': 0,
                'converged': False,
                'optimization_time': 0.0,
                'loss_history': [],
                'gradient_norm_history': [],
                'damping_history': [],
                'step_size_history': []
            }

        # Combined results
        combined_results = {
            'final_parameters': gn_results['parameters'],
            'final_loss': gn_results['loss'],
            'adam_losses': adam_losses,
            'adam_epochs': len(adam_losses),
            'switched_to_gn': self.switched_to_gn,
            'gn_results': gn_results,
            'total_precision_achieved': gn_results['loss'] if self.switched_to_gn else adam_losses[-1]
        }

        logger.info(f"Adaptive optimization completed!")
        logger.info(f"Final precision: {combined_results['total_precision_achieved']:.2e}")

        return combined_results

# Example usage and testing
if __name__ == "__main__":
    print("[T] Testing High-Precision Gauss-Newton Optimizer...")

    # This would normally be run with a real PINN setup
    # For demonstration, we'll create a simple quadratic test function

    class QuadraticResidual(ResidualFunction):
        """Simple quadratic test function for optimizer validation"""

        def __init__(self, n_params: int = 10, n_residuals: int = 20):
            self.n_params = n_params
            self.n_residuals = n_residuals

            # Random target parameters and measurement matrix
            self.target = torch.randn(n_params, dtype=torch.float64)
            self.A = torch.randn(n_residuals, n_params, dtype=torch.float64)
            self.b = torch.matmul(self.A, self.target) + 0.01 * torch.randn(n_residuals, dtype=torch.float64)

        def compute_residual(self, parameters: torch.Tensor) -> torch.Tensor:
            """Residual: r = Ax - b"""
            return torch.matmul(self.A, parameters) - self.b

        def compute_jacobian(self, parameters: torch.Tensor) -> torch.Tensor:
            """Jacobian: J = A"""
            return self.A

    # Test the optimizer
    test_function = QuadraticResidual(n_params=5, n_residuals=10)

    config = GaussNewtonConfig(
        learning_rate=1e-2,
        damping_factor=1e-6,
        max_iterations=100,
        tolerance=1e-12,
        line_search=True,
        adaptive_damping=True,
        verbose=True
    )

    optimizer = HighPrecisionGaussNewton(test_function, config)

    # Random initial parameters
    initial_params = torch.randn(5, dtype=torch.float64)

    print(f"True parameters: {test_function.target.numpy()}")
    print(f"Initial parameters: {initial_params.numpy()}")
    print(f"Initial loss: {0.5 * torch.sum(test_function.compute_residual(initial_params)**2).item():.2e}")

    # Optimize
    results = optimizer.optimize(initial_params)

    print(f"\nOptimization Results:")
    print(f"Final parameters: {results['parameters'].numpy()}")
    print(f"Final loss: {results['loss']:.2e}")
    print(f"Iterations: {results['iterations']}")
    print(f"Converged: {results['converged']}")
    print(f"Optimization time: {results['optimization_time']:.3f}s")

    # Check accuracy
    parameter_error = torch.norm(results['parameters'] - test_function.target).item()
    print(f"Parameter error: {parameter_error:.2e}")

    # Plot convergence
    optimizer.plot_convergence("gauss_newton_convergence.png")

    print("\n[*] Gauss-Newton optimizer test completed!")
    print(f"[+] Achieved precision: {results['loss']:.2e}")
    print("[!] Ready for integration with PINN solver!")