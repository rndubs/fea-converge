# Contributing to FR-BO Development

This guide provides instructions for setting up the development environment, running tests, and contributing to the Failure-Robust Bayesian Optimization (FR-BO) implementation.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Using the FR-BO Optimizer](#using-the-fr-bo-optimizer)
- [Creating Synthetic Datasets](#creating-synthetic-datasets)
- [Extending the Implementation](#extending-the-implementation)
- [Code Quality](#code-quality)

## Development Setup

### Prerequisites

- Python 3.8 or higher
- `uv` package manager (recommended) or `pip`
- Git for version control

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fea-converge
   ```

2. **Create virtual environment with uv**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -e ".[dev]"
   ```

   This installs:
   - **Core dependencies**: PyTorch, BoTorch, Ax Platform, GPyTorch, NumPy, SciPy, Pandas
   - **Visualization**: Matplotlib, Plotly
   - **ML tools**: scikit-learn
   - **Development tools**: pytest, black, ruff, mypy

4. **Verify installation**:
   ```bash
   python -c "import fr_bo; print('FR-BO installed successfully')"
   ```

## Project Structure

```
fea-converge/
├── fr_bo/                      # Main package
│   ├── __init__.py
│   ├── parameters.py           # Parameter space definition
│   ├── objective.py            # Objective function
│   ├── simulator.py            # Simulation executors
│   ├── gp_models.py            # Dual GP system
│   ├── acquisition.py          # FREI acquisition function
│   ├── optimizer.py            # Main FR-BO optimizer
│   ├── early_termination.py    # Early stopping logic
│   ├── multi_task.py           # Multi-task GP
│   ├── risk_scoring.py         # Risk assessment
│   ├── visualization.py        # Plotting tools
│   ├── synthetic_data.py       # Test data generation
│   └── utils.py                # Utility functions
│
├── tests/                      # Test suite
│   ├── test_parameters.py
│   ├── test_objective.py
│   ├── test_simulator.py
│   ├── test_gp_models.py
│   ├── test_synthetic_data.py
│   └── test_integration.py
│
├── fr-bo/                      # Implementation plans
│   └── IMPLEMENTATION_PLAN.md
│
├── pyproject.toml              # Package configuration
├── README.md                   # Project overview
├── RESEARCH.md                 # Technical documentation
└── CONTRIBUTING.md             # This file
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
pytest tests/test_parameters.py
pytest tests/test_integration.py
```

### Run with Coverage Report

```bash
pytest --cov=fr_bo --cov-report=html
```

Then open `htmlcov/index.html` to view the coverage report.

### Run Specific Test Functions

```bash
pytest tests/test_optimizer.py::test_complete_optimization_workflow
```

### Skip Slow Tests

Some integration tests are marked as slow. To skip them:

```bash
pytest -m "not slow"
```

## Using the FR-BO Optimizer

### Basic Example

```python
from fr_bo.optimizer import FRBOOptimizer, OptimizationConfig
from fr_bo.simulator import SyntheticSimulator

# Create synthetic simulator (for testing without real solver)
simulator = SyntheticSimulator(random_seed=42)

# Configure optimization
config = OptimizationConfig(
    n_sobol_trials=20,      # Initial random exploration
    n_frbo_trials=80,       # FR-BO optimization
    random_seed=42,
    max_iterations=1000,
    timeout=3600.0,
)

# Create optimizer
optimizer = FRBOOptimizer(simulator=simulator, config=config)

# Run optimization
results = optimizer.optimize()

# Access results
print(f"Best objective: {results['best_objective']}")
print(f"Best parameters: {results['best_parameters']}")
print(f"Success rate: {results['metrics']['overall']['success_rate']:.2%}")
```

### Visualization

```python
from fr_bo.visualization import OptimizationVisualizer
from fr_bo.parameters import encode_parameters

visualizer = OptimizationVisualizer()

# Plot convergence history
visualizer.plot_convergence_history(
    trials=results['trials'],
    save_path='convergence.png'
)

# Plot parameter space (2D projection)
visualizer.plot_parameter_space_2d(
    trials=results['trials'],
    param_encoder=encode_parameters,
    save_path='parameter_space.png',
    method='pca'
)

# Create interactive dashboard
visualizer.create_interactive_dashboard(
    trials=results['trials'],
    save_path='dashboard.html'
)
```

### Risk Assessment

```python
from fr_bo.risk_scoring import RiskScorer, ParameterValidator
from fr_bo.parameters import encode_parameters

# Create risk scorer
risk_scorer = RiskScorer()

# Add historical trials
for trial in results['trials']:
    params_encoded = encode_parameters(trial.parameters)
    risk_scorer.add_trial(params_encoded, trial.result.converged)

# Assess risk for new parameters
new_params = {"penalty_stiffness": 1e6, ...}
params_encoded = encode_parameters(new_params)

risk_assessment = risk_scorer.assess_risk(
    params_encoded,
    failure_model=optimizer.dual_gp.failure_classifier.model
)

print(f"Risk level: {risk_assessment.risk_level}")
print(f"Risk score: {risk_assessment.risk_score:.3f}")
print(f"Recommendation: {risk_assessment.recommendation}")
```

## Creating Synthetic Datasets

### Generate Benchmark Datasets

```python
from fr_bo.synthetic_data import create_benchmark_dataset

# Simple scenario (3D parameter space)
dataset_simple = create_benchmark_dataset(
    scenario_name="simple",
    n_train=100,
    n_test=50,
    random_seed=42
)

# Complex scenario (5D parameter space)
dataset_complex = create_benchmark_dataset(
    scenario_name="complex",
    n_train=200,
    n_test=50,
    random_seed=42
)

# Access data
X_train = dataset_simple["train"]["X"]
y_objective = dataset_simple["train"]["y_objective"]
y_converged = dataset_simple["train"]["y_converged"]
```

### Custom Scenarios

```python
from fr_bo.synthetic_data import SyntheticDataGenerator

generator = SyntheticDataGenerator(random_seed=42)

# Define custom scenario
from fr_bo.synthetic_data import SyntheticScenario
import numpy as np

custom_scenario = SyntheticScenario(
    name="custom",
    description="Custom test scenario",
    optimal_regions=[
        {
            "center": np.array([0.4, 0.4, 0.5]),
            "radius": 0.2,
            "success_prob": 0.95,
            "mean_objective": 0.1,
        }
    ],
    failure_regions=[
        {
            "center": np.array([0.9, 0.9, 0.5]),
            "radius": 0.15,
        }
    ],
    parameter_dim=3,
    baseline_success_rate=0.35,
)

# Generate data
X, y_obj, y_conv = generator.generate_training_data(custom_scenario, n_samples=100)
```

## Extending the Implementation

### Adding Custom Acquisition Functions

Create a new acquisition function in `fr_bo/acquisition.py`:

```python
class CustomAcquisition(AnalyticAcquisitionFunction):
    def __init__(self, model, best_f, ...):
        super().__init__(model=model)
        self.best_f = best_f

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        # Implement acquisition logic
        pass
```

### Custom Simulators

To integrate with a real solver, implement the simulation interface:

```python
from fr_bo.simulator import SimulationResult

class MyCustomSimulator:
    def run(self, parameters, max_iterations, timeout):
        # Run your simulation
        # ...

        return SimulationResult(
            converged=converged,
            iterations=iterations,
            max_iterations=max_iterations,
            time_elapsed=time_elapsed,
            timeout=timeout,
            final_residual=residual,
            contact_pressure_max=pressure,
            penetration_max=penetration,
            severe_instability=instability,
            residual_history=residuals,
            active_set_sizes=active_sets,
        )
```

Then use it with the optimizer:

```python
simulator = MyCustomSimulator()
optimizer = FRBOOptimizer(simulator=simulator, config=config)
results = optimizer.optimize()
```

### Multi-Task Optimization

For optimization across multiple geometries:

```python
from fr_bo.multi_task import GeometryOptimizationManager, GeometricFeatureExtractor

# Create manager
manager = GeometryOptimizationManager()

# Register geometries
for geom_id, geom_data in enumerate(geometries):
    manager.register_geometry(geom_id, geom_data)

# Add trials
for trial_result in trials:
    manager.add_trial(
        geometry_id=trial_result['geometry_id'],
        parameters=trial_result['parameters'],
        objective=trial_result['objective'],
        converged=trial_result['converged']
    )

# Train multi-task GP
manager.train_multi_task_gp()

# Get recommendations for new geometry
recommendations = manager.recommend_for_new_geometry(
    geometry_id=new_geom_id,
    candidate_parameters=candidate_grid,
    top_k=3
)
```

## Code Quality

### Formatting

Format code with Black:

```bash
black fr_bo tests
```

### Linting

Check code quality with Ruff:

```bash
ruff check fr_bo tests
```

Auto-fix issues:

```bash
ruff check --fix fr_bo tests
```

### Type Checking

Run type checking with mypy:

```bash
mypy fr_bo
```

### Pre-commit Checks

Before committing, run:

```bash
black fr_bo tests
ruff check --fix fr_bo tests
pytest
```

## Testing Guidelines

### Writing Tests

1. **Test one thing**: Each test should verify one specific behavior
2. **Use descriptive names**: `test_optimizer_tracks_best` not `test_1`
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use fixtures**: For common setup code
5. **Mock expensive operations**: Use synthetic data instead of real simulations

### Test Example

```python
def test_objective_function_converged():
    """Test objective function for converged simulation."""
    # Arrange
    obj_func = ObjectiveFunction()

    # Act
    objective = obj_func.compute(
        converged=True,
        iterations=20,
        max_iterations=100,
        time_elapsed=1.0,
        timeout=100.0,
    )

    # Assert
    assert objective > 0
    assert objective < 1.0
```

## Performance Tips

1. **Use smaller trial counts for development**: Set `n_sobol_trials=5, n_frbo_trials=5` for quick iteration
2. **Profile slow code**: Use `pytest --durations=10` to find slow tests
3. **Cache GP training**: Retraining is expensive, do it strategically
4. **Use inducing points**: For large datasets (>500 points), use sparse GPs

## Common Issues

### Issue: Import Errors

**Solution**: Ensure package is installed in editable mode:
```bash
uv pip install -e .
```

### Issue: Tests Fail with "No module named 'fr_bo'"

**Solution**: Make sure you're in the virtual environment:
```bash
source .venv/bin/activate
```

### Issue: Out of Memory During GP Training

**Solution**: Reduce batch size or use inducing points:
```python
# In gp_models.py, use inducing points
inducing_points = train_X[:100]  # Use subset
```

### Issue: Slow Tests

**Solution**: Use smaller trial counts or skip slow tests:
```bash
pytest -m "not slow" --durations=10
```

## Getting Help

- Check the [RESEARCH.md](RESEARCH.md) for detailed technical documentation
- See [fr-bo/IMPLEMENTATION_PLAN.md](fr-bo/IMPLEMENTATION_PLAN.md) for implementation details
- Open an issue on GitHub for bugs or feature requests

## Contributing Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run tests and quality checks
6. Commit with clear messages: `git commit -m "Add feature X"`
7. Push and create a pull request

Thank you for contributing to FR-BO!
