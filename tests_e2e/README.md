# E2E Test Suite - Minimal PDE Scenarios for CI

## Overview

End-to-end test suite for continuous integration with reduced computational requirements. These tests validate the full pipeline from initial conditions to lambda convergence on small-grid, short-epoch scenarios.

## Design Principles

1. **CI-Friendly**: Small grids (16x16x16 max), short epochs (<100)
2. **CPU-Only**: FP64 precision, no GPU requirements
3. **Fast Execution**: <5 minutes per test
4. **Regression Detection**: Validates numerical consistency across commits

## Test Categories

### 1. IPM (Incompressible Porous Media)
- Minimal boundary conditions test
- Lambda funnel convergence validation
- Conservation law verification

### 2. 2D Boussinesq
- Self-similar solution validation
- Multi-stage training pipeline
- Residual history tracking

### 3. Coordinate Transform
- Boundary condition preservation
- Numerical stability checks
- Grid transformation accuracy

## Metrics Collected

Each test automatically generates:
- Final residual values
- Conservation quantity violations
- Lambda estimates vs ground truth
- Seed sensitivity analysis
- Benchmark execution time
- Hardware/software configuration

## Running Tests

```bash
# Run all E2E tests (CI mode)
pytest tests_e2e/ -v --tb=short

# Run specific test category
pytest tests_e2e/test_ipm_minimal.py -v

# Generate metric reports
pytest tests_e2e/ --json-report --json-report-file=results/e2e_metrics.json
```

## Expected Outputs

- JSON metric cards in `results/e2e/`
- Convergence plots in `results/plots/e2e/`
- Failure case logs with hyperparameter suggestions
- Lineage tracking (commit SHA, Docker hash, seed, precision)
