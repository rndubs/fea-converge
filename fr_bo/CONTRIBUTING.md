
# Contributing to FR-BO

Thank you for your interest in contributing to FR-BO (Failure-Robust Bayesian Optimization)! This document provides guidelines and instructions for contributing.

---

## Current Status

**FR-BO is currently in development (v0.1.0) and NOT production-ready.**

The core implementation exists (4,062 LOC) but lacks:
- ✅ Test suite (NOW COMPLETE!)
- ✅ Basic examples (NOW COMPLETE!)
- ⚠️ Full documentation (IN PROGRESS)
- ⚠️ Validation against real FEA problems

**Priority contributions:**
1. Running tests and fixing bugs
2. Adding documentation and docstrings
3. Validating against Smith FEA
4. Adding professional logging

---

## Development Setup

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- BoTorch 0.9+
- GPyTorch 1.11+

### Installation

```bash
# Clone repository
cd /path/to/fea-converge/fr_bo

# Install in development mode with uv (recommended)
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_gp_models.py -v

# Run with coverage
pytest tests/ --cov=fr_bo --cov-report=html

# Run specific test
pytest tests/test_acquisition.py::TestFailureRobustEI::test_positive_values -v
```

---

## Project Structure

```
fr_bo/
├── __init__.py           # Package initialization (v0.1.0)
├── optimizer.py          # Main FRBOOptimizer class
├── gp_models.py          # Dual GP system (objective + failure)
├── acquisition.py        # FailureRobustEI acquisition function
├── multi_task.py         # Multi-task GP for transfer learning
├── early_termination.py  # Trajectory monitoring
├── risk_scoring.py       # Pre-simulation risk assessment
├── visualization.py      # Plotting utilities
├── synthetic_data.py     # Synthetic training data generation
├── simulator.py          # Simulation executor wrapper
├── objective.py          # Objective function definitions
├── parameters.py         # Parameter space encoding
├── utils.py              # Helper functions
├── tests/                # Test suite
│   ├── conftest.py       # Shared fixtures
│   ├── test_gp_models.py
│   ├── test_acquisition.py
│   ├── test_parameters.py
│   └── test_integration.py
├── examples/             # Usage examples
│   ├── basic_optimization.py
│   └── smith_integration_example.py
├── pyproject.toml        # Dependencies and configuration
├── README.md             # User documentation
└── CONTRIBUTING.md       # This file
```

---

## Contribution Workflow

### 1. Pick an Issue

Check the [issues](https://github.com/rndubs/fea-converge/issues) or see [Roadmap](#roadmap) below for tasks.

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 3. Make Changes

Follow the [Code Style](#code-style) guidelines below.

### 4. Add Tests

**All new code must include tests.**

```python
# tests/test_new_feature.py
import pytest

def test_new_feature():
    """Test description."""
    # Arrange
    ...
    # Act
    ...
    # Assert
    assert result == expected
```

### 5. Run Tests

```bash
# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=fr_bo

# Format code
black fr_bo/ tests/
ruff check fr_bo/ tests/
```

### 6. Commit Changes

```bash
git add .
git commit -m "Add feature: brief description

Longer description of what changed and why.
Fixes #123"
```

### 7. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

---

## Code Style

### Python Style

We follow [PEP 8](https://peps.python.org/pep-0008/) with some modifications:

- **Line length:** 100 characters (not 79)
- **Formatter:** Black
- **Linter:** Ruff

```bash
# Format code
black fr_bo/ tests/

# Check linting
ruff check fr_bo/ tests/

# Auto-fix linting issues
ruff check --fix fr_bo/ tests/
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief one-line description.

    Longer description if needed, explaining what the function does,
    any important details, and gotchas.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When this error occurs

    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
    pass
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import Dict, List, Optional, Tuple
import torch

def predict(
    model: torch.nn.Module,
    data: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prediction with type hints."""
    pass
```

---

## Testing Guidelines

### Test Organization

- **Unit tests:** Test individual components in isolation
- **Integration tests:** Test complete workflows
- **Fixtures:** Share test data via pytest fixtures in `conftest.py`

### Test Structure

```python
class TestMyComponent:
    """Test suite for MyComponent."""

    def test_initialization(self):
        """Test that component initializes correctly."""
        component = MyComponent()
        assert component is not None

    def test_basic_functionality(self):
        """Test basic functionality."""
        component = MyComponent()
        result = component.do_something()
        assert result == expected

    def test_edge_case_empty_input(self):
        """Test edge case with empty input."""
        component = MyComponent()
        with pytest.raises(ValueError):
            component.do_something([])
```

### Test Coverage

Aim for **80%+ test coverage** on new code:

```bash
pytest tests/ --cov=fr_bo --cov-report=term-missing
```

---

## Documentation

### README Updates

When adding new features, update `README.md` with:
- Feature description
- Usage example
- API changes

### Docstring Standards

All public functions/classes need docstrings:

```python
class FRBOOptimizer:
    """
    Failure-Robust Bayesian Optimization for FEA convergence.

    FR-BO uses a dual Gaussian process system to simultaneously model
    the objective function and failure probability, enabling optimization
    in scenarios where simulations frequently fail.

    Attributes:
        simulator: Simulation executor
        config: Optimization configuration
        trials: List of completed trials
        dual_gp: Dual GP system (initialized after first batch)

    Example:
        >>> from fr_bo.optimizer import FRBOOptimizer, OptimizationConfig
        >>> config = OptimizationConfig(n_sobol_trials=20, n_frbo_trials=50)
        >>> optimizer = FRBOOptimizer(simulator=my_sim, config=config)
        >>> results = optimizer.optimize()
    """
    pass
```

---

## Roadmap

### Phase 1: Testing & Validation (Current)

**Priority: Critical**

- [x] Create test suite (test_gp_models.py, test_acquisition.py, etc.)
- [ ] Run full test suite and fix failing tests
- [ ] Achieve 80%+ test coverage
- [ ] Add edge case tests
- [ ] Validate on standard BO benchmarks (Branin, Hartmann, etc.)

### Phase 2: Documentation (In Progress)

**Priority: High**

- [x] Create basic usage example
- [x] Create Smith integration example
- [x] Write CONTRIBUTING.md
- [ ] Add comprehensive docstrings to all public APIs
- [ ] Expand README.md with full API documentation
- [ ] Create tutorial notebooks (optional)

### Phase 3: Validation (Next)

**Priority: High**

- [ ] Validate against Smith FEA simulations
- [ ] Compare performance vs. CONFIG/GP-Classification/SHEBO
- [ ] Benchmark computational cost
- [ ] Document known limitations and failure modes

### Phase 4: Production Polish

**Priority: Medium**

- [ ] Add professional logging throughout
- [ ] Add progress bars and status updates
- [ ] Error handling and graceful degradation
- [ ] Configuration validation
- [ ] Performance profiling and optimization

### Phase 5: Release v1.0.0

**Priority: Future**

- [ ] All tests passing
- [ ] 80%+ test coverage
- [ ] Complete documentation
- [ ] Validated against real FEA
- [ ] Performance benchmarked
- [ ] Ready for production use

---

## Issue Labels

When creating issues, use these labels:

- `bug`: Something isn't working
- `documentation`: Improvements or additions to documentation
- `enhancement`: New feature or request
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `priority: high`: High priority
- `priority: low`: Low priority
- `testing`: Related to test suite
- `validation`: Needs validation

---

## Questions?

- Check the [README.md](README.md) for general information
- Review [RESEARCH.md](../RESEARCH.md) for technical details
- Check existing [issues](https://github.com/rndubs/fea-converge/issues)
- Ask in pull request comments

---

## License

By contributing to FR-BO, you agree that your contributions will be licensed under the same license as the fea-converge project.

---

**Thank you for contributing to FR-BO!** 🚀
