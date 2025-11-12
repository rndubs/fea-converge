# Contributing to CONFIG Optimizer

This document provides instructions for setting up the development environment and contributing to the CONFIG (Constrained Efficient Global Optimization) implementation.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Running Examples](#running-examples)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)

## Development Setup

### Prerequisites

- Python 3.11 or higher
- `uv` package manager (recommended) or `pip`
- Git

### Installation with uv (Recommended)

1. **Clone the repository**:
   ```bash
   cd config/
   ```

2. **Sync dependencies**:
   ```bash
   uv sync
   ```

3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```

4. **Install development dependencies**:
   ```bash
   uv sync --all-extras
   ```

### Installation with pip

Alternatively, if you don't have `uv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Verify Installation

Check that the installation was successful:

```bash
python -c "import config_optimizer; print(config_optimizer.__version__)"
```

## Project Structure

```
config/
├── src/config_optimizer/        # Main package source code
│   ├── core/                    # Core optimizer components
│   │   └── controller.py        # Main CONFIG controller
│   ├── models/                  # GP models
│   │   └── gp_models.py         # Objective and constraint GPs
│   ├── acquisition/             # Acquisition functions
│   │   └── config_acquisition.py # LCB-based acquisition
│   ├── monitoring/              # Monitoring utilities
│   │   └── violation_monitor.py # Violation tracking
│   ├── solvers/                 # Black box solvers
│   │   └── black_box_solver.py  # Synthetic test functions
│   ├── utils/                   # Utility functions
│   │   ├── beta_schedule.py     # Beta parameter computation
│   │   ├── constraints.py       # Constraint formulations
│   │   └── sampling.py          # LHS and Sobol sampling
│   └── visualization/           # Visualization tools
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── examples/                    # Example scripts
├── IMPLEMENTATION_PLAN.md       # Detailed implementation plan
├── pyproject.toml               # Package configuration
└── CONTRIBUTING.md              # This file
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_beta_schedule.py -v

# Specific test function
pytest tests/unit/test_beta_schedule.py::test_compute_beta_basic -v
```

### Run Tests with Coverage

```bash
pytest --cov=src/config_optimizer --cov-report=html
# View coverage report at htmlcov/index.html
```

### Run Tests in Verbose Mode

```bash
pytest -v -s  # -s shows print statements
```

## Running Examples

### Basic Example

```bash
cd examples/
python basic_example.py
```

This will:
1. Set up a black box solver (Branin function)
2. Configure and run CONFIG optimizer
3. Display results and statistics
4. Generate a violation plot

### Custom Example

Create your own optimization problem:

```python
from config_optimizer.core.controller import CONFIGController, CONFIGConfig
import numpy as np

# Define your objective function
def my_objective(x):
    # Your simulation code here
    return {
        'objective_value': ...,
        'final_residual': ...,
        'iterations': ...,
        'converged': ...
    }

# Configure optimizer
config = CONFIGConfig(
    bounds=np.array([[lower1, upper1], [lower2, upper2]]),
    constraint_configs={'convergence': {'tolerance': 1e-8}},
    n_init=20,
    n_max=100
)

# Run optimization
optimizer = CONFIGController(config, my_objective)
results = optimizer.optimize()
```

## Development Workflow

### Adding New Features

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement your feature** with tests

3. **Run tests**:
   ```bash
   pytest
   ```

4. **Format code** (if using black/ruff):
   ```bash
   black src/ tests/
   ruff check src/ tests/
   ```

5. **Commit changes**:
   ```bash
   git add .
   git commit -m "Add: your feature description"
   ```

6. **Push and create pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Testing New Components

When adding new components, ensure you:

1. **Write unit tests** for individual functions
2. **Write integration tests** for component interactions
3. **Test edge cases** and error handling
4. **Document** the component with docstrings

Example test structure:

```python
def test_new_component_basic():
    """Test basic functionality."""
    result = new_component()
    assert result is not None

def test_new_component_edge_case():
    """Test edge case handling."""
    with pytest.raises(ValueError):
        new_component(invalid_input)
```

## Code Style

### Python Style Guide

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write comprehensive docstrings (Google style)
- Keep functions focused and modular
- Maximum line length: 100 characters

### Docstring Format

```python
def function_name(arg1: type1, arg2: type2) -> return_type:
    """
    Brief description of function.
    
    More detailed description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When validation fails
        
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
    """
    pass
```

### Testing Guidelines

- Test names should be descriptive: `test_component_behavior_expected_result`
- Each test should test one specific behavior
- Use fixtures for common setup
- Mock external dependencies
- Aim for >80% code coverage

## Debugging Tips

### Enable Verbose Output

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Debug Specific Components

```python
# In your code
import pdb; pdb.set_trace()  # Python debugger

# Or use IPython
from IPython import embed; embed()
```

### Visualize GP Models

```python
# Plot GP predictions
mean, std = gp.predict(test_points)
plt.plot(test_points, mean)
plt.fill_between(test_points, mean-2*std, mean+2*std, alpha=0.3)
plt.show()
```

## Common Issues and Solutions

### Issue: PyTorch/CUDA Installation

If you encounter CUDA-related issues:

```bash
# Install CPU-only version
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: GP Fitting Fails

If GP fitting produces warnings:

- Check for NaN or Inf values in your data
- Ensure sufficient data points (>10 recommended)
- Try standardizing your inputs/outputs
- Increase numerical stability with jitter

### Issue: Empty Optimistic Feasible Set

If F_opt is empty:

- Increase beta to expand the feasible set
- Check constraint formulations
- Verify initial sampling covers feasible regions
- Consider problem reformulation

## Getting Help

- **Issues**: Open an issue on GitHub
- **Questions**: Check IMPLEMENTATION_PLAN.md for detailed algorithm description
- **Examples**: See examples/ directory for working code

## License

This project is part of the fea-converge repository and follows its licensing terms.
