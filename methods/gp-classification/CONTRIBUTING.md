# Contributing to GP Classification

This guide describes how to set up the development environment and work with the GP Classification codebase.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Running Tests](#running-tests)
- [Running Examples](#running-examples)
- [Code Style](#code-style)
- [Adding New Features](#adding-new-features)

## Environment Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Installation Steps

1. **Clone the repository**

```bash
cd gp-classification
```

2. **Install dependencies using uv**

```bash
uv sync --extra dev
```

This will:
- Create a virtual environment in `.venv/`
- Install all required dependencies (PyTorch, BoTorch, GPyTorch, etc.)
- Install development dependencies (pytest, black, ruff)

3. **Activate the virtual environment**

```bash
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

4. **Verify installation**

```bash
uv run python -c "import gp_classification; print(gp_classification.__version__)"
```

You should see: `0.1.0`

## Project Structure

```
gp-classification/
├── src/gp_classification/       # Main package source code
│   ├── __init__.py             # Package initialization
│   ├── data.py                 # Trial database and data management
│   ├── models.py               # GP models (variational classifier, dual model)
│   ├── acquisition.py          # Acquisition functions (CEI, entropy, adaptive)
│   ├── optimizer.py            # Main GP Classification optimizer
│   ├── use_cases.py            # Parameter suggestion, validation, real-time estimation
│   ├── visualization.py        # Plotting and visualization tools
│   └── mock_solver.py          # Mock Smith solver for testing
├── tests/                       # Test suite
│   ├── test_data.py            # Data management tests
│   ├── test_models.py          # GP model tests
│   ├── test_optimizer.py       # Optimizer tests
│   ├── test_mock_solver.py     # Mock solver tests
│   └── test_integration.py     # End-to-end integration tests
├── examples/                    # Example scripts
│   └── basic_optimization.py   # Basic optimization example
├── pyproject.toml              # Project configuration and dependencies
├── IMPLEMENTATION_PLAN.md      # Detailed implementation specification
├── CONTRIBUTING.md             # This file
└── README.md                   # Project README
```

## Development Workflow

### Making Changes

1. **Create a new branch** (if working in a git repository)

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** in the appropriate files

3. **Run tests** to ensure nothing broke

```bash
uv run pytest
```

4. **Format code** with black

```bash
uv run black src/ tests/
```

5. **Lint code** with ruff

```bash
uv run ruff check src/ tests/
```

6. **Commit changes**

```bash
git add .
git commit -m "Description of changes"
```

## Running Tests

### Run all tests

```bash
uv run pytest
```

### Run specific test file

```bash
uv run pytest tests/test_data.py
```

### Run specific test function

```bash
uv run pytest tests/test_data.py::test_database_initialization
```

### Run tests with coverage

```bash
uv run pytest --cov=gp_classification --cov-report=html
```

Coverage report will be generated in `htmlcov/index.html`

### Run tests with verbose output

```bash
uv run pytest -v
```

### Run only fast tests (skip slow integration tests)

```bash
uv run pytest -m "not slow"
```

## Running Examples

### Basic Optimization Example

```bash
cd examples
uv run python basic_optimization.py
```

This will:
- Run a complete optimization with 60 iterations
- Generate parameter suggestions
- Perform validation
- Create visualizations in `examples/output/`
- Save trial database to CSV

### Output Files

After running the example, you'll find:
- `output/convergence_landscape.png` - 2D convergence probability heatmap
- `output/uncertainty_map.png` - Prediction uncertainty visualization
- `output/optimization_history.png` - Convergence rate and best objective over time
- `output/parameter_importance.png` - ARD-based parameter importance
- `output/calibration.png` - Model calibration curve
- `output/dashboard.png` - Comprehensive summary dashboard
- `output/trial_database.csv` - Complete trial history

## Code Style

### Formatting

We use **Black** for code formatting with a line length of 100:

```bash
uv run black src/ tests/ examples/
```

### Linting

We use **Ruff** for fast Python linting:

```bash
uv run ruff check src/ tests/ examples/
```

To automatically fix issues:

```bash
uv run ruff check --fix src/ tests/ examples/
```

### Type Hints

We encourage the use of type hints for function signatures:

```python
def train_model(
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int = 100,
) -> Tuple[Model, Likelihood]:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong
    """
    ...
```

## Adding New Features

### Adding a New Acquisition Function

1. **Create the acquisition class** in `src/gp_classification/acquisition.py`:

```python
class MyNewAcquisition(AcquisitionFunction):
    """Description of your acquisition function."""

    def __init__(self, ...):
        super().__init__(model=...)
        # Initialize parameters

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate acquisition function."""
        # Implementation
        return acquisition_values
```

2. **Add tests** in `tests/test_acquisition.py`:

```python
def test_my_new_acquisition():
    """Test MyNewAcquisition."""
    # Create test data
    # Initialize acquisition
    # Verify behavior
    assert ...
```

3. **Update `__init__.py`** to export the new class

4. **Add documentation** in docstrings and update README if needed

### Adding a New Use Case

1. **Implement in `src/gp_classification/use_cases.py`**

2. **Add tests** in `tests/test_use_cases.py`

3. **Create example** in `examples/` directory

4. **Update documentation**

## Testing with Mock Solver

The mock Smith solver (`mock_solver.py`) simulates realistic contact convergence behavior without requiring the actual FEA solver.

### Creating a Mock Solver

```python
from gp_classification.mock_solver import MockSmithSolver

# Easy problem
solver = MockSmithSolver(
    random_seed=42,
    noise_level=0.05,
    difficulty="easy"
)

# Test simulation
converged, iterations = solver.simulate({
    "penalty_stiffness": 1e5,
    "gap_tolerance": 1e-7,
    ...
})
```

### Difficulty Levels

- **easy**: Wide convergence region, ~70-80% success rate with random sampling
- **medium**: Moderate convergence region, ~40-60% success rate
- **hard**: Narrow convergence region, ~20-40% success rate

### Generating Synthetic Datasets

```python
from gp_classification.mock_solver import SyntheticDataGenerator, get_default_parameter_bounds

bounds = get_default_parameter_bounds()
generator = SyntheticDataGenerator(bounds, random_seed=42)

# Generate using Latin Hypercube Sampling
dataset = generator.generate_dataset_with_solver(
    solver=solver,
    n_samples=100,
    sampling_method="latin_hypercube"
)
```

## Integration with Real Smith Solver

To use with the actual Smith/Tribol solver:

1. **Create a wrapper function**:

```python
def smith_simulator(parameters: Dict[str, float]) -> Tuple[bool, Optional[float]]:
    """
    Run Smith/Tribol simulation with given parameters.

    Args:
        parameters: Solver parameters

    Returns:
        (converged, objective_value)
    """
    # Set up Smith input deck with parameters
    # Run simulation
    # Parse output
    # Return convergence status and iteration count
    converged = ...  # Parse from solver output
    iterations = ...  # Parse from solver output

    return converged, iterations if converged else None
```

2. **Use with optimizer**:

```python
optimizer = GPClassificationOptimizer(
    parameter_bounds=your_bounds,
    simulator=smith_simulator,  # Use real solver
    n_initial_samples=20,
    verbose=True,
)
```

## Common Issues

### PyTorch Installation

If PyTorch doesn't detect your GPU:

```bash
# Check CUDA availability
uv run python -c "import torch; print(torch.cuda.is_available())"
```

For CPU-only systems, the default PyTorch installation works fine.

### Memory Issues

For large datasets (>1000 trials):
- Increase `n_inducing_points` for variational GP (default: 100)
- Use batch prediction for visualizations
- Consider data subsampling for initial development

### Convergence Issues in Training

If GP classifier training doesn't converge:
- Increase `n_epochs` (default: 500)
- Adjust learning rate (try 0.01 or 0.1)
- Check data balance (need both converged and failed trials)
- Ensure sufficient data (at least 20 trials)

## Getting Help

- Review `IMPLEMENTATION_PLAN.md` for detailed design specifications
- Check existing tests for usage examples
- Run example scripts to see complete workflows
- Consult BoTorch documentation: https://botorch.org/
- Consult GPyTorch documentation: https://gpytorch.ai/

## Summary of Commands

```bash
# Setup
uv sync --extra dev

# Development
uv run pytest                      # Run all tests
uv run pytest --cov=gp_classification  # With coverage
uv run black src/ tests/          # Format code
uv run ruff check src/ tests/     # Lint code

# Examples
uv run python examples/basic_optimization.py

# Package
uv build                          # Build package
```

Happy coding! 🚀
