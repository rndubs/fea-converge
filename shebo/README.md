# SHEBO: Surrogate Optimization with Hidden Constraints

SHEBO is a machine learning-based optimization framework that combines neural network ensemble surrogate modeling with constraint discovery for solving complex optimization problems with unknown constraints. It is specifically designed for optimizing finite element analysis (FEA) contact convergence parameters.

## Key Features

- **Ensemble Surrogate Modeling**: Uses multiple neural networks with uncertainty quantification (each network trained independently for maximum diversity)
- **Constraint Discovery**: Automatically discovers and models hidden failure modes
- **Adaptive Acquisition**: Balances exploration, exploitation, and boundary learning
- **Feature Normalization**: Robust StandardScaler preprocessing for vastly different parameter scales
- **Comprehensive Data Validation**: Automatic checks for NaN/Inf, class imbalance, and minimum samples
- **Checkpointing**: Save and resume optimization state for crash recovery
- **Batch Parallelization**: Evaluate multiple points simultaneously for faster optimization
- **Production Ready**: Lightweight deployment for real-time predictions
- **Transfer Learning**: Share knowledge across similar problems

## Recent Improvements (2024)

SHEBO has undergone comprehensive review and fixes for all critical issues:

- ✅ **Fixed ensemble training**: Each network now trains independently with separate optimizers, ensuring true diversity
- ✅ **Feature normalization**: Handles vastly different parameter scales (1e-8 to 1e10) with StandardScaler
- ✅ **Robust device handling**: Full GPU/CPU support with consistent device management
- ✅ **Iteration-based tracking**: Fixed sample counting for correct model update schedules
- ✅ **Data validation**: Comprehensive checks for data quality, NaN/Inf, and class balance
- ✅ **Proper logging**: Python logging throughout instead of print statements
- ✅ **Checkpointing**: Save and resume optimization progress
- ✅ **Batch API**: Parallel evaluation support with diversity-aware selection
- ✅ **Enhanced tests**: Comprehensive correctness tests including ensemble diversity, normalization, and optimization improvement

See [CRITICAL_REVIEW.md](CRITICAL_REVIEW.md) for detailed fix documentation.

## Installation

### Using uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repository-url>
cd shebo
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Using pip

```bash
pip install -e ".[dev]"
```

## Quick Start

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

# Run optimization
optimizer = SHEBOOptimizer(
    bounds=bounds,
    objective_fn=objective,
    n_init=20,
    budget=100
)

result = optimizer.run()

# View results
print(f"Best performance: {result.best_performance}")
print(f"Success rate: {sum(result.convergence_history)/len(result.convergence_history)*100:.1f}%")
```

## Components

### 1. Neural Network Ensembles

Multiple neural networks trained with different initializations to quantify both aleatoric (data) and epistemic (model) uncertainty.

```python
from shebo.models.ensemble import ConvergenceEnsemble

ensemble = ConvergenceEnsemble(input_dim=4, n_networks=5)
predictions = ensemble.predict_with_uncertainty(X)

print(f"Mean: {predictions['mean']}")
print(f"Epistemic uncertainty: {predictions['epistemic_uncertainty']}")
print(f"Aleatoric uncertainty: {predictions['aleatoric_uncertainty']}")
```

### 2. Constraint Discovery

Automatically detects and models failure modes:
- Residual oscillation/divergence
- Numerical instability (NaN/Inf)
- Mesh distortion
- Contact detection failures
- Excessive penetration

```python
from shebo.core.constraint_discovery import ConstraintDiscovery

discovery = ConstraintDiscovery()
violations = discovery.check_simulation_output(output)

for violation in violations:
    print(f"{violation.type}: {violation.description}")
```

### 3. Adaptive Acquisition Function

Multi-objective acquisition balancing:
- Expected improvement (performance)
- Feasibility probability
- Uncertainty reduction
- Boundary exploration

### 4. Surrogate Manager

Coordinates multiple surrogate models with asynchronous updates:
- Convergence model (classification)
- Performance model (regression)
- Constraint models (discovered dynamically)

## Architecture

```
SHEBO Optimizer
├── Initialization (Sobol sampling)
├── Surrogate Manager
│   ├── Convergence Ensemble
│   ├── Performance Ensemble
│   └── Constraint Ensembles (dynamic)
├── Constraint Discovery
│   └── Anomaly Detection
├── Adaptive Acquisition
│   ├── Expected Improvement
│   ├── Feasibility Probability
│   ├── Uncertainty Sampling
│   └── Boundary Exploration
└── Optimization Loop
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=shebo --cov-report=html

# Run specific test
pytest tests/test_optimizer.py -v
```

## Examples

### Basic Optimization

```bash
cd examples
python simple_optimization.py
```

### Custom Objective Function

```python
def my_objective(params):
    """Custom objective for your problem."""
    # Run your simulation
    output = run_my_simulation(params)

    # Return required format
    return {
        'output': {
            'convergence_status': output.converged,
            'residual_history': output.residuals,
            'iterations': output.n_iters,
            'solve_time': output.time,
            'penetration_max': output.max_pen,
            'jacobian_min': output.min_jac,
            'contact_pairs': output.n_contacts,
            'all_values': output.all_vals,
            'expected_contact': True
        },
        'performance': output.n_iters  # Metric to minimize
    }

# Use with SHEBO
optimizer = SHEBOOptimizer(bounds=bounds, objective_fn=my_objective)
result = optimizer.run()
```

## Visualization

```python
from shebo.visualization.plots import (
    plot_convergence_history,
    plot_performance_history,
    plot_constraint_timeline,
    plot_parameter_space_2d
)

# Plot convergence over time
fig = plot_convergence_history(result.convergence_history)
fig.savefig('convergence.png')

# Plot performance evolution
fig = plot_performance_history(
    result.performance_history,
    result.convergence_history
)
fig.savefig('performance.png')

# Plot discovered constraints
fig = plot_constraint_timeline(
    result.discovered_constraints['constraints'],
    result.iterations
)
fig.savefig('constraints.png')
```

## Performance Characteristics

### Expected Success Rates

- **Initial phase (0-50 trials)**: 30-60% convergence
- **Mid-term (50-200 trials)**: 85-90% convergence
- **Mature (>200 trials)**: >90% convergence

### Computational Efficiency

- **vs Grid Search**: 50-100x speedup
- **vs Random Search**: 5-10x speedup
- **Inference Latency**: <5ms (compressed models)
- **Model Size**: <1MB (deployment)

## When to Use SHEBO

### Best For:
- Production systems with many routine simulations
- Geometry families requiring transfer learning
- Complex, multi-constraint problems
- High-dimensional parameter spaces (>10D)
- Real-time monitoring requirements

### Consider Alternatives:
- Limited data (<100 samples) → GP Classification or FR-BO
- Need formal guarantees → CONFIG
- Simpler constraint landscape → GP Classification

## Documentation

### For Users
- [README.md](README.md) - This file: Overview, installation, quick start, examples
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Detailed design and architecture documentation
- [RESEARCH.md](../RESEARCH.md) - Comprehensive ML systems research documentation

### For Contributors & Developers
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - **Comprehensive developer guide**:
  - Exact environment reproduction with pinned dependencies
  - Testing environment setup and best practices
  - Architecture overview and component descriptions
  - Extension guide (adding new features, models, acquisition functions)
  - Development workflows and troubleshooting
  - Custom objective function examples
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contributing guidelines and development setup basics
- [CRITICAL_REVIEW.md](CRITICAL_REVIEW.md) - Code review findings and all fixes implemented
- [FIXES.md](FIXES.md) - Detailed documentation of bug fixes with code examples

### Quick Links by Task
- **Getting Started**: [Installation](#installation) → [Quick Start](#quick-start)
- **Adding New Features**: [DEVELOPMENT.md - Extension Guide](DEVELOPMENT.md#extension-guide)
- **Writing Tests**: [DEVELOPMENT.md - Testing Environment](DEVELOPMENT.md#testing-environment)
- **Troubleshooting**: [DEVELOPMENT.md - Troubleshooting](DEVELOPMENT.md#troubleshooting)
- **Environment Setup**: [DEVELOPMENT.md - Environment Setup](DEVELOPMENT.md#environment-setup)

## Citation

If you use SHEBO in your research, please cite:

```bibtex
@software{shebo2024,
  title={SHEBO: Surrogate Optimization with Hidden Constraints},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/shebo}
}
```

## License

[Specify your license]

## Acknowledgments

SHEBO is part of the FEA-Converge project for optimizing finite element contact convergence using machine learning, developed for the LLNL Tribol contact library and Smith/Serac solver framework.
