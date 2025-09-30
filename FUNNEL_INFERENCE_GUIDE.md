# Funnel Inference Implementation Guide

## Overview

**Funnel Inference** is DeepMind's method for finding admissible λ values in self-similar blow-up solutions (Paper pages 16-17). This document explains the implementation and usage.

---

## Theory

### What is Funnel Inference?

At an **admissible λ value** (λ*), smooth solutions to self-similar PDEs exist. The residual function r(λ) has a characteristic "funnel" shape:

```
     r(λ)
      |
      |     *
      |    / \
      |   /   \
      |  /     \
    0 +--------*-------- λ
      |       λ*
      |
```

- **Bottom of funnel** (λ*): Minimum residual → smooth solution exists
- **Away from λ***: Residual increases → no smooth solution

### Algorithm (from Paper)

**Secant Method for Zero-Finding:**

1. Initialize with λ₀
2. For iteration n:
   - Train PINN with fixed λₙ until convergence
   - Evaluate residual r̂(λₙ) near origin
   - Update: λₙ₊₁ = λₙ - r̂ₙ₋₁ · (λₙ₋₁ - λₙ)/(r̂ₙ₋₁ - r̂ₙ)
3. Stop when |λₙ₊₁ - λₙ| < tolerance

**First Iteration:** λ₁ = λ₀ + Δλ (small perturbation)

---

## Usage

### Basic Example

```python
from funnel_inference import FunnelInference, FunnelInferenceConfig
from funnel_inference import create_evaluation_points_near_origin

# 1. Configure
config = FunnelInferenceConfig(
    initial_lambda=0.5,              # Starting guess
    delta_lambda=0.01,               # Initial perturbation
    max_iterations=20,               # Max secant iterations
    convergence_tol=1e-6,            # Stop when Δλ < this
    training_steps_per_eval=5000,    # Train steps per iteration
    min_training_loss=1e-8,          # Target training loss
    verbose=True
)

# 2. Initialize
funnel = FunnelInference(config)
funnel.initialize()

# 3. Create evaluation points (near singularity)
eval_points = create_evaluation_points_near_origin(
    n_points=20,
    radius=0.1,
    dim=2  # 2D problem
)

# 4. Run optimization
results = funnel.optimize(
    network=pinn_network,
    pde_system=pde,
    train_function=training_callback,
    evaluation_points=eval_points
)

# 5. Get results
print(f"Final λ: {results['final_lambda']:.10f}")
print(f"Iterations: {results['iterations']}")
print(f"Converged: {results['converged']}")

# 6. Visualize
funnel.plot_funnel(save_path='funnel_convergence.png')
```

### Training Callback

The `train_function` must have this signature:

```python
def train_pinn_with_fixed_lambda(network, pde_system, lambda_fixed, max_steps):
    """
    Train PINN with fixed lambda value

    Args:
        network: Neural network (PINN)
        pde_system: PDE system with residual computation
        lambda_fixed: Fixed lambda value for this iteration
        max_steps: Maximum training steps

    Returns:
        dict with 'final_loss': float
    """
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    for step in range(max_steps):
        optimizer.zero_grad()

        # Compute PDE residual with fixed lambda
        loss = compute_pinn_loss(network, pde_system, lambda_fixed)

        loss.backward()
        optimizer.step()

    return {'final_loss': loss.item()}
```

### PDE System Requirements

Your PDE system must implement:

```python
class MyPDESystem:
    def compute_residual(self, u_pred, x, lambda_value):
        """
        Compute PDE residual for given solution and lambda

        Args:
            u_pred: Network output [N, ...]
            x: Spatial coordinates [N, dim]
            lambda_value: Current lambda value

        Returns:
            residual: Tensor [N]
        """
        # Compute derivatives using autograd
        u_x = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]

        # PDE residual (example: u_t + λ·u = 0)
        residual = u_t + lambda_value * u_pred

        return residual
```

---

## Complete Example: IPM Equation

```python
import torch
import torch.nn as nn
from funnel_inference import FunnelInference, FunnelInferenceConfig
from pinn_solver import PINNSolver, IncompressiblePorousMedia

# Known from paper: IPM stable λ = 1.0285722760222
# We'll see if funnel inference finds it

# 1. Setup PINN
class IPMPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

network = IPMPINN()
pde_system = IncompressiblePorousMedia()

# 2. Define training function
def train_ipm_pinn(network, pde_system, lambda_fixed, max_steps=5000):
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    # Collocation points in self-similar coordinates
    y = torch.randn(1000, 2, requires_grad=True)

    for step in range(max_steps):
        optimizer.zero_grad()

        u_pred = network(y)
        residual = pde_system.compute_residual(u_pred, y, lambda_fixed)
        loss = torch.mean(residual ** 2)

        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"  Step {step}: loss = {loss.item():.6e}")

    return {'final_loss': loss.item()}

# 3. Run funnel inference
config = FunnelInferenceConfig(
    initial_lambda=1.0,  # Start near expected value
    delta_lambda=0.001,
    max_iterations=15,
    convergence_tol=1e-5,
    training_steps_per_eval=5000
)

funnel = FunnelInference(config)
funnel.initialize()

eval_points = create_evaluation_points_near_origin(n_points=30, radius=0.1, dim=2)

results = funnel.optimize(
    network=network,
    pde_system=pde_system,
    train_function=train_ipm_pinn,
    evaluation_points=eval_points
)

# 4. Compare with paper
paper_lambda = 1.0285722760222
final_lambda = results['final_lambda']
error = abs(final_lambda - paper_lambda)

print(f"\nResults:")
print(f"  Paper λ:  {paper_lambda:.10f}")
print(f"  Found λ:  {final_lambda:.10f}")
print(f"  Error:    {error:.6e}")
print(f"  Error %:  {error/paper_lambda*100:.3f}%")

# 5. Visualize funnel
funnel.plot_funnel(save_path='ipm_funnel_inference.png')
```

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_lambda` | 0.5 | Starting λ value |
| `delta_lambda` | 1e-3 | Initial perturbation for secant |
| `max_iterations` | 20 | Maximum iterations |
| `convergence_tol` | 1e-6 | Stop when Δλ < this |
| `residual_region` | "origin" | Where to evaluate residual |
| `training_steps_per_eval` | 5000 | Training steps per iteration |
| `min_training_loss` | 1e-8 | Target training loss |
| `verbose` | True | Print progress |

---

## Tips for Success

### 1. Good Initial Guess

Use empirical formula for initialization:

```python
from unstable_singularity_detector import UnstableSingularityDetector

detector = UnstableSingularityDetector(equation_type="ipm")
lambda_init = detector.predict_next_unstable_lambda(order=1)

config = FunnelInferenceConfig(initial_lambda=lambda_init)
```

### 2. Sufficient Training

Each iteration needs the PINN to converge:

```python
config = FunnelInferenceConfig(
    training_steps_per_eval=10000,  # Increase if not converging
    min_training_loss=1e-10         # Lower threshold
)
```

### 3. Evaluation Points

Focus on region near singularity:

```python
# For IPM/Boussinesq: singularity at origin
eval_points = create_evaluation_points_near_origin(
    n_points=50,     # More points = better signal
    radius=0.05,     # Small radius near origin
    dim=2
)
```

### 4. Monitoring Convergence

Check funnel shape during optimization:

```python
# After each iteration
print(f"λ = {current_lambda:.8f}, r̂ = {residual:.6e}")

# Final check
funnel.plot_funnel()  # Should see funnel converging to minimum
```

---

## Troubleshooting

### Problem: Not Converging

**Symptoms:** Lambda oscillates or diverges

**Solutions:**
1. Reduce `delta_lambda` (e.g., 1e-4)
2. Increase `training_steps_per_eval`
3. Use better initial guess (empirical formula)
4. Check PDE residual computation

### Problem: Training Loss High

**Symptoms:** `min_training_loss` not reached

**Solutions:**
1. Increase training steps
2. Lower learning rate
3. Use better optimizer (Gauss-Newton)
4. Check network architecture

### Problem: Residual Not Decreasing

**Symptoms:** r̂(λ) doesn't show funnel shape

**Solutions:**
1. Check evaluation points are near singularity
2. Verify PDE residual formula
3. Increase network capacity
4. Try different λ initialization

---

## Integration with Existing Code

### With PINN Solver

```python
from pinn_solver import PINNSolver, PINNConfig
from funnel_inference import FunnelInference, FunnelInferenceConfig

# Create PINN solver
pinn_config = PINNConfig(
    hidden_layers=[64, 64, 64],
    precision=torch.float64
)
solver = PINNSolver(pde_system, pinn_config)

# Wrap training function
def train_with_solver(network, pde_system, lambda_fixed, max_steps):
    solver.set_lambda(lambda_fixed)
    history = solver.train(max_epochs=max_steps)
    return {'final_loss': history['total_loss'][-1]}

# Run funnel inference
config = FunnelInferenceConfig()
funnel = FunnelInference(config)
results = funnel.optimize(
    network=solver.network,
    pde_system=pde_system,
    train_function=train_with_solver,
    evaluation_points=eval_points
)
```

### With Gauss-Newton Optimizer

```python
from gauss_newton_optimizer import HighPrecisionGaussNewton

def train_with_gn(network, pde_system, lambda_fixed, max_steps):
    optimizer = HighPrecisionGaussNewton(
        network.parameters(),
        residual_function=lambda: pde_system.compute_residual(..., lambda_fixed)
    )

    for step in range(max_steps):
        optimizer.step()

    return {'final_loss': optimizer.get_loss()}
```

---

## Validation

Compare with paper ground truth:

```python
# IPM Ground Truth (from paper Table, page 4)
ground_truth = {
    "stable": 1.0285722760222,
    "1st_unstable": 0.4721297362414,
    "2nd_unstable": 0.3149620267088,
    "3rd_unstable": 0.2415604743989
}

# Run funnel inference for 1st unstable
config = FunnelInferenceConfig(initial_lambda=0.47)
funnel = FunnelInference(config)
results = funnel.optimize(...)

# Validate
expected = ground_truth["1st_unstable"]
found = results['final_lambda']
error = abs(found - expected) / expected

assert error < 0.01, f"Error too large: {error:.3%}"
print(f"[+] Validation passed: error = {error:.3%}")
```

---

## References

- **DeepMind Paper:** arXiv:2509.14185v1
- **Funnel Inference:** Pages 16-17, Equations 17-18
- **Figure 5:** Smoothness signal in λ (funnel plots)
- **Figure 6:** Funnel validation plots

---

## Next Steps

After implementing Funnel Inference, proceed to:

1. **Multi-stage Training** (paper pages 17-18)
   - Stage 1: Coarse solution (10^-8)
   - Stage 2: Fourier features refinement (10^-13)

2. **Full Gauss-Newton Optimizer** (paper pages 7-8)
   - Rank-1 unbiased estimator
   - Exponential moving average
   - Automated learning rate

3. **Computer-Assisted Proofs**
   - Interval arithmetic validation
   - Spectral analysis
   - Rigorous bounds

---

**Status:** ✅ Funnel Inference fully implemented and tested (2025-09-30)