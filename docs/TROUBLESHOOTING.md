# Troubleshooting Guide

This guide helps resolve common issues when using the Unstable Singularity Detector.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Training Problems](#training-problems)
3. [Convergence Issues](#convergence-issues)
4. [Performance Problems](#performance-problems)
5. [Testing Errors](#testing-errors)
6. [GPU/CUDA Issues](#gpucuda-issues)

---

## Installation Issues

### Problem: pip install fails with dependency conflicts

**Symptoms**:
```
ERROR: Cannot install package due to conflicting dependencies
```

**Solutions**:
1. Create fresh virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -e .
   ```

2. Install with specific PyTorch version:
   ```bash
   pip install torch==2.4.0
   pip install -e .
   ```

3. Use requirements.txt directly:
   ```bash
   pip install -r requirements.txt
   ```

### Problem: ImportError for torch/numpy/scipy

**Symptoms**:
```
ImportError: No module named 'torch'
```

**Solution**:
Ensure PyTorch is installed correctly:
```bash
pip install torch==2.4.0
python -c "import torch; print(torch.__version__)"
```

**Note**: The torch_shim module is for testing only. Production code requires real PyTorch.

---

## Training Problems

### Problem: Training starts but loss stays high

**Symptoms**:
- Loss remains >1e-3 after many iterations
- No convergence progress

**Solutions**:

1. **Check network capacity**:
   ```python
   # Increase hidden layers or neurons
   network = PINN(
       input_dim=3,
       hidden_dims=[128, 128, 128],  # Was [64, 64]
       output_dim=1
   )
   ```

2. **Adjust learning rate**:
   ```python
   config.learning_rate = 1e-4  # Try different values: 1e-3, 1e-5
   ```

3. **Enable multi-stage training**:
   ```python
   # Use progressive refinement
   trainer = MultiStageTrainer(config)
   stage1_hist = trainer.train_stage1(net, train_fn, val_fn)
   ```

### Problem: Training fails with NaN loss

**Symptoms**:
```
Loss: 1.234e-5 -> NaN
RuntimeError: Function 'X' returned nan values
```

**Solutions**:

1. **Reduce learning rate**:
   ```python
   config.learning_rate = 1e-5  # Start smaller
   ```

2. **Enable gradient clipping**:
   ```python
   config.gradient_clip = 10.0  # Prevent gradient explosion
   ```

3. **Check PDE residual function**:
   - Ensure no division by zero
   - Check for numerical instabilities in derivatives
   - Validate boundary condition implementation

4. **Use FP64 precision**:
   ```python
   network = network.double()  # Use float64
   ```

---

## Convergence Issues

### Problem: Gauss-Newton optimizer doesn't converge

**Symptoms**:
- Residual stuck at ~1e-6 or higher
- No improvement after many iterations

**Solutions**:

1. **Increase gradient_clip**:
   ```python
   config.gradient_clip = 20.0  # Default: 10.0
   ```

2. **Adjust damping parameter**:
   ```python
   config.damping = 1e-4  # Try: 1e-3, 1e-5
   ```

3. **Enable adaptive damping**:
   ```python
   config.adaptive_damping = True
   config.damping_increase = 2.0
   config.damping_decrease = 0.5
   ```

4. **Check problem conditioning**:
   ```python
   # Compute condition number of Hessian
   H = optimizer.compute_hessian(params)
   cond = torch.linalg.cond(H)
   print(f"Condition number: {cond:.2e}")

   # If cond > 1e10, problem is ill-conditioned
   # Try preconditioning or regularization
   ```

### Problem: Funnel inference fails to find lambda

**Symptoms**:
```
RuntimeError: Funnel inference did not converge
```

**Solutions**:

1. **Adjust search bounds**:
   ```python
   config.lambda_min = 0.1  # Expand search range
   config.lambda_max = 2.0
   ```

2. **Increase max iterations**:
   ```python
   config.max_funnel_iters = 50  # Default: 20
   ```

3. **Use formula-based initialization**:
   ```python
   detector = UnstableSingularityDetector(equation_type="ipm")
   lambda_init = detector.predict_next_unstable_lambda(order=1)

   config.lambda_init = lambda_init  # Start near expected value
   ```

4. **Check residual function shape**:
   ```python
   # Plot residual vs lambda to verify funnel shape
   lambdas = np.linspace(0.1, 2.0, 50)
   residuals = [eval_residual(l) for l in lambdas]
   plt.plot(lambdas, residuals)
   plt.show()
   ```

---

## Performance Problems

### Problem: Training is very slow

**Symptoms**:
- Epochs take minutes instead of seconds
- CPU usage at 100%

**Solutions**:

1. **Enable GPU acceleration**:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   network = network.to(device)
   ```

2. **Reduce grid resolution**:
   ```python
   # Spatial grid
   nx, ny, nz = 32, 32, 32  # Was 64, 64, 64

   # Temporal grid
   nt = 100  # Was 500
   ```

3. **Use smaller network**:
   ```python
   network = PINN(
       input_dim=3,
       hidden_dims=[64, 64],  # Was [128, 128, 128]
       output_dim=1
   )
   ```

4. **Batch gradient computation**:
   ```python
   # Process points in batches
   batch_size = 1000
   for i in range(0, len(points), batch_size):
       batch = points[i:i+batch_size]
       residual = compute_residual(batch)
   ```

### Problem: High memory usage / OOM errors

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:

1. **Reduce batch size**:
   ```python
   config.batch_size = 500  # Was 2000
   ```

2. **Use gradient checkpointing**:
   ```python
   torch.utils.checkpoint.checkpoint(network, inputs)
   ```

3. **Clear cache periodically**:
   ```python
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   ```

4. **Use CPU for large problems**:
   ```python
   device = torch.device("cpu")  # More memory available
   ```

---

## Testing Errors

### Problem: Tests fail with "CUDA not available"

**Symptoms**:
```
SKIPPED [1] tests/test_pinn_solver.py:45: CUDA not available
```

**Solution**:
This is expected on CPU-only systems. Tests are designed to skip CUDA-specific tests automatically.

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run CPU-only tests
pytest tests/ -v -k "not cuda"
```

### Problem: Test precision failures

**Symptoms**:
```
AssertionError: Final loss 3.2e-12 > threshold 1e-12
```

**Solutions**:

1. **Check random seed**:
   ```python
   # Tests use fixed seeds for reproducibility
   torch.manual_seed(42)
   np.random.seed(42)
   ```

2. **Adjust tolerance**:
   ```python
   # In test file
   assert loss < 1e-11  # Relaxed from 1e-12
   ```

3. **Verify optimizer config**:
   ```python
   # Ensure gradient_clip is sufficient
   config.gradient_clip = 10.0  # Default in tests
   ```

---

## GPU/CUDA Issues

### Problem: CUDA errors during training

**Symptoms**:
```
RuntimeError: CUDA error: device-side assert triggered
```

**Solutions**:

1. **Check tensor device consistency**:
   ```python
   # Ensure all tensors on same device
   assert network.device == inputs.device
   assert network.device == labels.device
   ```

2. **Verify CUDA version compatibility**:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   nvcc --version  # Should match PyTorch CUDA version
   ```

3. **Test with CPU first**:
   ```python
   # Verify code works on CPU
   device = torch.device("cpu")
   network = network.to(device)
   ```

4. **Update GPU drivers**:
   - NVIDIA: Download latest drivers from nvidia.com
   - Check compatibility with PyTorch CUDA version

### Problem: Multi-GPU errors

**Symptoms**:
```
RuntimeError: Expected all tensors to be on the same device
```

**Solution**:
This implementation doesn't support multi-GPU. Use single GPU or CPU:
```python
# Force single GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

---

## Common Error Messages

### "step must be non-zero"

**Cause**: Using `arange()` with step=0

**Solution**:
```python
# WRONG
x = arange(0, 5, 0)

# CORRECT
x = arange(0, 5, 1)
```

### "item() only works for 1-element tensors"

**Cause**: Calling `.item()` on multi-element tensor

**Solution**:
```python
# WRONG
loss = losses.item()  # losses has shape [10]

# CORRECT
loss = losses.mean().item()
```

### "RuntimeError: Function 'X' returned inf values"

**Cause**: Numerical overflow/underflow

**Solutions**:
1. Use FP64: `network.double()`
2. Normalize inputs: `x = (x - mean) / std`
3. Clip gradients: `config.gradient_clip = 10.0`
4. Check for division by zero in PDE residual

---

## Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: [GitHub Issues](https://github.com/Flamehaven/unstable-singularity-detector/issues)
2. **Review documentation**:
   - [CHANGES.md](../CHANGES.md) - Recent updates and bug fixes
   - [FUNNEL_INFERENCE_GUIDE.md](FUNNEL_INFERENCE_GUIDE.md) - Funnel inference details
   - [GAUSS_NEWTON_COMPLETE.md](GAUSS_NEWTON_COMPLETE.md) - Optimizer documentation
3. **Open new issue**: Include:
   - Python version
   - PyTorch version
   - Full error traceback
   - Minimal reproducible example
   - System info (OS, GPU if applicable)

---

## Related Documentation

- [README.md](../README.md) - Main documentation
- [Limitations & Known Issues](../README.md#limitations--known-issues)
- [Scientific Context](../README.md#scientific-context--validation-methodology)
