# Contributing to Unstable Singularity Detector

Thank you for your interest in contributing to the Unstable Singularity Detector project! This implementation is based on DeepMind's breakthrough research in fluid dynamics and we welcome contributions from the scientific community.

## 🚀 Quick Start

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/unstable-singularity-detector.git
   cd unstable-singularity-detector
   ```

2. **Create Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

## 📋 Types of Contributions

### 🔬 Scientific Contributions
- **New PDE Systems**: Implement additional fluid dynamics equations
- **Optimization Methods**: Improve convergence and precision
- **Mathematical Analysis**: Enhance singularity classification
- **Validation**: Compare with analytical solutions or other methods

### 💻 Technical Contributions
- **Performance Optimization**: GPU acceleration, memory efficiency
- **Visualization**: Better plots and interactive tools
- **Documentation**: Examples, tutorials, API improvements
- **Testing**: Unit tests, integration tests, benchmarks

### 🐛 Bug Reports and Fixes
- Report issues with clear reproduction steps
- Include system information and error messages
- Submit fixes with tests when possible

## 🛠️ Development Guidelines

### Code Style
- **Python**: Follow PEP 8, use `black` for formatting
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style for all classes and functions
- **ASCII Only**: No Unicode emojis in code (for cross-platform compatibility)

### Testing Requirements
- **Unit Tests**: All new functionality must include tests
- **Integration Tests**: For complex workflows
- **Precision Tests**: Numerical accuracy validation
- **GPU Tests**: When CUDA functionality is modified

### Documentation
- **API Documentation**: Auto-generated from docstrings
- **Examples**: Working code examples for new features
- **Mathematical Background**: For complex algorithms
- **Performance Benchmarks**: When applicable

## 🔬 Scientific Standards

### Mathematical Rigor
- **Cite Sources**: Reference papers and mathematical foundations
- **Numerical Precision**: Maintain machine-precision accuracy where possible
- **Validation**: Compare with known analytical solutions
- **Error Analysis**: Quantify and report numerical errors

### Reproducibility
- **Random Seeds**: Set for all stochastic processes
- **Dependencies**: Pin versions for critical dependencies
- **Environment**: Document system requirements clearly
- **Data**: Include test datasets or generation scripts

## 📝 Pull Request Process

### Before Submitting
1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write Tests**
   ```python
   # tests/test_your_feature.py
   def test_your_new_functionality():
       assert expected_behavior()
   ```

3. **Run Full Test Suite**
   ```bash
   pytest tests/ --cov=src/ --cov-report=html
   black src/ tests/ examples/
   flake8 src/ tests/ examples/
   ```

4. **Update Documentation**
   - Update README.md if adding major features
   - Add docstrings and type hints
   - Create examples if appropriate

### Pull Request Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Scientific enhancement

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] GPU tests pass (if applicable)
- [ ] Precision tests maintain accuracy

## Mathematical/Scientific Validation
- [ ] Compared with analytical solutions (if available)
- [ ] Validated against DeepMind's published results
- [ ] Error analysis completed
- [ ] Convergence properties analyzed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes without discussion
```

## 🎯 Priority Areas

### High Priority
1. **Navier-Stokes Extension**: Full 3D implementation
2. **Parallel Computing**: MPI-based distributed training
3. **Computer-Assisted Proofs**: Rigorous error bounds
4. **Additional PDE Systems**: More fluid dynamics equations

### Medium Priority
1. **Performance Optimization**: Memory usage, GPU utilization
2. **Visualization Enhancements**: Interactive 3D plots
3. **Example Applications**: Real-world fluid dynamics problems
4. **Documentation**: Tutorials and mathematical background

### Research Opportunities
1. **New Singularity Types**: Explore beyond current classifications
2. **Machine Learning**: Improved neural network architectures
3. **Optimization**: Novel training algorithms for PINNs
4. **Applications**: Climate modeling, turbulence, etc.

## 🤝 Community Guidelines

### Code of Conduct
- **Respectful**: Be respectful and inclusive to all contributors
- **Constructive**: Provide constructive feedback and criticism
- **Collaborative**: Work together toward common scientific goals
- **Professional**: Maintain professional standards in all interactions

### Communication
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: For sensitive issues or direct collaboration inquiries

### Attribution
- **Scientific Credit**: Proper citation of original research and contributors
- **Code Attribution**: Credit for significant code contributions
- **Collaboration**: Acknowledge collaborative efforts appropriately

## 📚 Resources

### Scientific Background
- [DeepMind Blog Post](https://deepmind.google/discover/blog/discovering-new-solutions-to-century-old-problems-in-fluid-dynamics/)
- [arXiv Paper](https://arxiv.org/pdf/2509.14185v1)
- [Navier-Stokes Millennium Problem](https://www.claymath.org/millennium-problems/navier-stokes-equation)

### Technical Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Scientific Computing with Python](https://scipy-lectures.org/)
- [Numerical Methods for PDEs](https://link.springer.com/book/10.1007/978-3-319-32726-6)

### Development Tools
- [pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [Pre-commit Hooks](https://pre-commit.com/)

## 🏆 Recognition

Contributors will be recognized through:
- **GitHub Contributors Page**: Automatic recognition
- **Paper Acknowledgments**: For significant scientific contributions
- **Documentation**: Listed in project documentation
- **Presentations**: Credit in conference presentations and talks

## 📞 Contact

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and community discussions
- **Email**: research@flamehaven.ai (for direct collaboration)

---

**"From mathematical discovery to collaborative science - building the future of computational fluid dynamics together."**