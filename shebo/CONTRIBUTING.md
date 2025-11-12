# Contributing to SHEBO

This guide provides instructions for setting up the development environment and contributing to the SHEBO (Surrogate Optimization with Hidden Constraints) project.

> **📘 For Developers**: See [DEVELOPMENT.md](DEVELOPMENT.md) for comprehensive documentation including:
> - Exact environment reproduction with pinned dependencies
> - Testing environment setup and best practices
> - Architecture overview and component descriptions
> - Extension guide (adding new features, models, acquisition functions)
> - Development workflows and troubleshooting
>
> This file (CONTRIBUTING.md) covers the basics. DEVELOPMENT.md provides in-depth technical documentation.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Using SHEBO](#using-shebo)
- [Contributing Guidelines](#contributing-guidelines)

## Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Installing uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd shebo
```

### 2. Create Virtual Environment and Install Dependencies

Using uv (recommended):

```bash
# Create virtual environment
uv venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install package in development mode with dependencies
uv pip install -e ".[dev]"
```

Alternative using pip:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Run tests to verify everything is working
pytest tests/ -v
```

## Development Setup

### Installing Development Dependencies

Development dependencies include testing, linting, and formatting tools:

```bash
uv pip install -e ".[dev]"
```

This installs:
- **pytest**: Testing framework
- **pytest-cov**: Code coverage
- **black**: Code formatter
- **ruff**: Linter
- **mypy**: Type checker

### Code Formatting and Linting

Before committing, ensure your code is properly formatted and linted:

```bash
# Format code with black
black shebo/ tests/

# Lint with ruff
ruff check shebo/ tests/

# Type checking with mypy
mypy shebo/
```

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_optimizer.py -v
```

### Run with Coverage Report

```bash
pytest tests/ -v --cov=shebo --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Run Specific Test

```bash
pytest tests/test_optimizer.py::TestSHEBOOptimizer::test_optimizer_initialization -v
```

## Project Structure

```
shebo/
├── shebo/                      # Main package
│   ├── __init__.py
│   ├── core/                   # Core optimization components
│   │   ├── optimizer.py        # Main SHEBO optimizer
│   │   ├── surrogate_manager.py  # Surrogate model management
│   │   ├── constraint_discovery.py  # Constraint discovery
│   │   └── acquisition.py      # Acquisition functions
│   ├── models/                 # Neural network models
│   │   ├── convergence_nn.py   # NN architectures
│   │   └── ensemble.py         # Ensemble models
│   ├── utils/                  # Utilities
│   │   ├── black_box_solver.py  # Test solver
│   │   └── synthetic_data.py   # Data generation
│   └── visualization/          # Plotting tools
│       └── plots.py
├── tests/                      # Test suite
│   ├── test_optimizer.py
│   ├── test_ensemble.py
│   ├── test_constraint_discovery.py
│   └── test_black_box_solver.py
├── examples/                   # Example scripts
│   └── simple_optimization.py
├── data/                       # Data directory (generated)
├── pyproject.toml             # Package configuration
├── IMPLEMENTATION_PLAN.md     # Detailed implementation plan
└── CONTRIBUTING.md            # This file
```

## Using SHEBO

### Quick Start Example

```python
import numpy as np
from shebo import SHEBOOptimizer
from shebo.utils.black_box_solver import create_test_objective

# Define parameter bounds
bounds = np.array([
    [1e6, 1e10],    # penalty parameter
    [1e-8, 1e-4],   # tolerance
    [0.0, 1.0],     # timestep
    [0.0, 1.0]      # damping
])

# Create objective function
objective = create_test_objective(n_params=4, random_seed=42)

# Initialize optimizer
optimizer = SHEBOOptimizer(
    bounds=bounds,
    objective_fn=objective,
    n_init=20,
    budget=100,
    random_seed=42
)

# Run optimization
result = optimizer.run()

# Access results
print(f"Best performance: {result.best_performance}")
print(f"Best parameters: {result.best_params}")
print(f"Success rate: {sum(result.convergence_history)/len(result.convergence_history)*100:.1f}%")
```

### Run Example Script

```bash
cd examples
python simple_optimization.py
```

This will:
1. Run SHEBO optimization
2. Print results
3. Generate visualization plots

### Creating Custom Objective Functions

Your objective function should have this signature:

```python
def objective_function(params: np.ndarray) -> Dict[str, Any]:
    """Custom objective function.

    Args:
        params: Parameter vector

    Returns:
        Dictionary with:
            - 'output': Simulation output dict with keys:
                - 'convergence_status': bool
                - 'residual_history': List[float]
                - 'iterations': int
                - 'solve_time': float
                - 'penetration_max': float
                - 'jacobian_min': float
                - 'contact_pairs': int
                - 'all_values': List[float]
                - 'expected_contact': bool
            - 'performance': float (metric to minimize)
    """
    # Your simulation code here
    output = run_simulation(params)
    performance = compute_performance(output)

    return {
        'output': output,
        'performance': performance
    }
```

## Contributing Guidelines

### Submitting Changes

1. **Fork the repository**

2. **Create a feature branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

3. **Make your changes**
   - Write clear, documented code
   - Add tests for new functionality
   - Ensure all tests pass
   - Format and lint your code

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description of feature"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/my-new-feature
   ```

6. **Create a Pull Request**

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for all public functions/classes
- Keep lines under 100 characters
- Use meaningful variable names

### Development Best Practices

Based on comprehensive code review and fixes, follow these critical practices:

#### Ensemble Training
- ✅ **DO**: Train each network independently with separate optimizers
- ✅ **DO**: Use manual optimization mode in PyTorch Lightning
- ❌ **DON'T**: Average losses and backpropagate once (defeats ensemble diversity)

```python
# GOOD: Independent training
def training_step(self, batch, batch_idx):
    optimizers = self.optimizers()
    for network, optimizer in zip(self.networks, optimizers):
        optimizer.zero_grad()
        loss = compute_loss(network(x), y)
        self.manual_backward(loss)
        optimizer.step()

# BAD: Shared gradients
def training_step(self, batch, batch_idx):
    total_loss = sum(compute_loss(net(x), y) for net in self.networks)
    return total_loss  # All networks get same gradients!
```

#### Feature Normalization
- ✅ **DO**: Always normalize features when ranges span multiple orders of magnitude
- ✅ **DO**: Use StandardScaler for neural networks
- ✅ **DO**: Fit normalizer once on first batch, then transform consistently
- ❌ **DON'T**: Pass raw features with scales like 1e-8 to 1e10 to neural networks

#### Device Management
- ✅ **DO**: Move models to device on initialization
- ✅ **DO**: Move data to device before training
- ✅ **DO**: Ensure device consistency in predictions
- ❌ **DON'T**: Assume CPU-only usage

#### Data Validation
- ✅ **DO**: Check for NaN/Inf before training
- ✅ **DO**: Validate minimum samples per class for classification
- ✅ **DO**: Warn about severe class imbalance
- ✅ **DO**: Use constants for magic numbers (MIN_SAMPLES_FOR_TRAINING = 10)

#### Logging and Error Handling
- ✅ **DO**: Use Python logging module, not print statements
- ✅ **DO**: Catch specific exceptions (RuntimeError, ValueError)
- ✅ **DO**: Log at appropriate levels (debug, info, warning, error)
- ❌ **DON'T**: Use broad `except Exception` that hides bugs
- ❌ **DON'T**: Use print() for library code

#### Iteration vs Sample Counting
- ✅ **DO**: Track iterations explicitly for model update schedules
- ✅ **DO**: Pass iteration count to update functions
- ❌ **DON'T**: Accumulate sample counts that don't represent actual new samples

#### Checkpointing
- ✅ **DO**: Save optimization state periodically
- ✅ **DO**: Include random state for reproducibility
- ✅ **DO**: Provide both automatic and manual checkpoint methods

### Testing Requirements

- All new code must have accompanying tests
- Maintain or improve code coverage
- Tests should be independent and reproducible
- Use fixtures for common test setup
- **Test correctness, not just structure** (e.g., verify ensemble diversity, not just that it runs)
- Test edge cases: small datasets, imbalanced data, NaN/Inf values

### Documentation

- Update docstrings for any changed functionality
- Add examples for new features
- Update CONTRIBUTING.md if adding new workflows
- Comment complex algorithms

## Common Issues and Solutions

### PyTorch Installation

If you encounter issues with PyTorch:

```bash
# CPU-only version
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU version (CUDA 11.8)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors

If you get import errors:

```bash
# Reinstall in development mode
uv pip install -e .
```

### Test Failures

If tests fail:

1. Ensure you have all dependencies: `uv pip install -e ".[dev]"`
2. Check Python version: `python --version` (should be 3.8+)
3. Try running tests individually to isolate issues
4. Check for stale `.pyc` files: `find . -type d -name __pycache__ -exec rm -rf {} +`

## Getting Help

- Check the [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed design documentation
- Review existing tests for usage examples
- Open an issue on GitHub for bugs or feature requests

## License

[Specify your license here]

## Acknowledgments

SHEBO is part of the FEA-Converge project for optimizing finite element contact convergence using machine learning surrogate models.
