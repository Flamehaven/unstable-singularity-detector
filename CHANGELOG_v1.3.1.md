# Changelog - v1.3.1

**Release Date**: 2025-10-04
**Focus**: E2E Validation & Reproducibility (Patch 21 Implementation)

---

## 🚀 New Features

### E2E Reproduction Pipeline
- **Added** `examples/e2e_ipm_pipeline.py`: Full end-to-end IPM reproduction pipeline
  - Explicit initial conditions with boundary setup
  - Stage 1/2/3 training with automatic log collection (residuals, conservation, condition numbers)
  - Lambda-funnel convergence curve generation
  - Failure case documentation with hyperparameter suggestions
  - Complete lineage tracking (git commit, seed, precision, hardware)

### Test Stratification
- **Added** `tests_e2e/` directory: Minimal PDE scenarios for CI
  - Small-grid tests (16x16x16 max) for fast execution (<5 min)
  - CPU-only FP64 for maximum compatibility
  - Regression detection without overnight experiments
  - 5 new E2E test cases

### Metric Standardization
- **Added** `scripts/generate_metric_report.py`: Auto-report generator
  - Standardized metric schema (residual, vorticity, conservation, lambda, seed sensitivity, runtime)
  - JSON/CSV export formats
  - Auto-generated markdown tables for README
  - CI artifact generation for GitHub Actions

### CI/CD Integration
- **Added** `.github/workflows/e2e-metrics.yml`: Automated E2E validation
  - Matrix testing across Python 3.9, 3.10, 3.11
  - Automatic metric collection and README updates
  - PR comment integration with metric summaries
  - Manual trigger for full pipeline (2-hour timeout safety)

---

## 📚 Documentation

### High-Precision Modes Guide
- **Added** comprehensive platform compatibility matrix (OS/BLAS/NumPy/PyTorch)
- **Added** FP64 vs FP128 requirement table
- **Added** hardware recommendations for different use cases
- **Added** precision enablement code examples

### Paper-to-Code Mapping
- **Added** complete traceability table: Paper references → Code locations → Test coverage
- **Added** validation methodology documentation
- **Added** explicit known gaps section
- Direct links to implementation files with line numbers

### Implementation Documentation
- **Added** `PATCH_21_IMPLEMENTATION.md`: Complete implementation report
  - Detailed component descriptions
  - Validation results
  - Impact assessment
  - Breaking changes analysis (none)

---

## 🔧 Improvements

### Lineage Tracking
- Auto-collect git commit SHA, branch, timestamp
- Record PyTorch version, device (CPU/GPU), precision mode
- Track seed for reproducibility
- Include hardware platform information

### Failure Intelligence
- Automatic failure case documentation with timestamp
- Root cause suggestions based on error patterns
- Hyperparameter recommendations for retry
- Plot generation even on partial failures

---

## 📊 Metrics & Validation

### New Metric Schema
```json
{
  "final_residual": "Final PDE residual",
  "max_vorticity": "Maximum vorticity magnitude",
  "conservation_violation": "Mass/energy conservation error",
  "lambda_estimate": "Estimated lambda parameter",
  "seed_sensitivity": "Std dev across seeds",
  "benchmark_time": "Total execution time (s)"
}
```

### E2E Test Results
- 5 new E2E tests: **5 passed**
- Previous tests: **99 passed, 2 skipped** (unchanged)
- Total: **104 passed, 2 skipped**

---

## 🗂️ File Structure Changes

### New Files
```
tests_e2e/
├── README.md                           # E2E test documentation
└── test_ipm_minimal.py                 # Minimal IPM E2E tests

examples/
└── e2e_ipm_pipeline.py                 # Full E2E pipeline

scripts/
└── generate_metric_report.py           # Metric report generator

.github/workflows/
└── e2e-metrics.yml                     # E2E CI/CD workflow

PATCH_21_IMPLEMENTATION.md              # Implementation report
CHANGELOG_v1.3.1.md                     # This file
```

### Modified Files
```
README.md                               # +82 lines
  - High-Precision Modes section
  - Paper-to-Code Mapping table

setup.py                                # Version 1.0.0 → 1.3.1
pyproject.toml                          # Version 1.0.0 → 1.3.1
```

---

## 📈 Statistics

- **Code Added**: ~850 lines
- **Documentation Added**: ~200 lines
- **Tests Added**: ~180 lines
- **Total New Content**: ~1,230 lines
- **Breaking Changes**: 0
- **Deprecations**: 0

---

## 🔄 Migration Guide

No migration required. All changes are additive:
- Existing code unchanged
- New optional features
- Backward compatible

To use new E2E pipeline:
```bash
# Run E2E IPM pipeline
python examples/e2e_ipm_pipeline.py --config configs/e2e/ipm_minimal.yaml

# Generate metric reports
python scripts/generate_metric_report.py --input results/e2e --output results/metric_summary.md

# Run E2E tests
pytest tests_e2e/ -v
```

---

## 🎯 Impact Summary

### Before v1.3.1
- Formula-based validation only
- Manual verification required
- Ad-hoc experiments
- Limited traceability

### After v1.3.1
- ✅ Numerical validation with reproducible pipelines
- ✅ Automated lineage tracking + CI validation
- ✅ Standardized E2E scenarios with metric cards
- ✅ Full audit trail (commit → config → results)

---

## 🙏 Acknowledgments

This release implements recommendations from Patch 21 review:
- E2E reproduction scenarios for scientific validation
- Platform-specific precision guidance
- Metric standardization for cross-study comparison
- Documentation traceability for audit readiness

---

## 🔗 Links

- **Release**: [v1.3.1](https://github.com/Flamehaven/unstable-singularity-detector/releases/tag/v1.3.1)
- **Implementation Report**: [PATCH_21_IMPLEMENTATION.md](PATCH_21_IMPLEMENTATION.md)
- **E2E Tests**: [tests_e2e/](tests_e2e/)
- **Paper Mapping**: [README.md#paper-to-code-mapping](README.md#paper-to-code-mapping)

---

**Full Changelog**: [v1.0.0...v1.3.1](https://github.com/Flamehaven/unstable-singularity-detector/compare/v1.0.0...v1.3.1)
