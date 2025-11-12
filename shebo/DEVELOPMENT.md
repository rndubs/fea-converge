# SHEBO Development Guide

This guide provides comprehensive documentation for developers working on SHEBO, including environment setup, testing, architecture overview, and extension guidelines.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Testing Environment](#testing-environment)
- [Architecture Overview](#architecture-overview)
- [Extension Guide](#extension-guide)
- [Development Workflows](#development-workflows)
- [Troubleshooting](#troubleshooting)

## Environment Setup

### Exact Environment Reproduction

For complete reproducibility, we provide multiple options:

#### Option 1: Using uv with pyproject.toml (Recommended for Development)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone <repository-url>
cd shebo

# Create virtual environment and install
uv venv --python 3.10
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e ".[dev]"
```

#### Option 2: Using pip with exact versions

For maximum reproducibility, you can freeze exact package versions:

```bash
# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install from pyproject.toml
pip install -e ".[dev]"

# Freeze exact versions for reproducibility
pip freeze > requirements-lock.txt
```

To reproduce from frozen requirements:

```bash
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

### Python Version

**Recommended**: Python 3.10
**Minimum**: Python 3.8
**Tested**: Python 3.8, 3.9, 3.10, 3.11

### Core Dependencies

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| torch | 2.0.0 | Neural network training |
| pytorch-lightning | 2.0.0 | Training framework |
| numpy | 1.24.0 | Numerical operations |
| scipy | 1.10.0 | Optimization, sampling |
| scikit-learn | 1.3.0 | Preprocessing, metrics |
| imbalanced-learn | 0.11.0 | Handling class imbalance |
| matplotlib | 3.7.0 | Visualization |
| seaborn | 0.12.0 | Statistical plots |
| plotly | 5.14.0 | Interactive visualizations |
| pydantic | 2.0.0 | Data validation |

### Development Dependencies

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| pytest | 7.4.0 | Testing framework |
| pytest-cov | 4.1.0 | Code coverage |
| black | 23.0.0 | Code formatting |
| ruff | 0.0.280 | Linting |
| mypy | 1.4.0 | Type checking |

### GPU Support (Optional)

For GPU acceleration:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# ROCm (AMD GPUs)
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
```

Verify GPU availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

## Testing Environment

### Running Tests

SHEBO uses pytest with comprehensive test coverage requirements.

#### Basic Test Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_optimizer.py -v

# Run specific test class
pytest tests/test_optimizer.py::TestSHEBOOptimizer -v

# Run specific test method
pytest tests/test_optimizer.py::TestSHEBOOptimizer::test_initialization -v

# Run with coverage report
pytest tests/ -v --cov=shebo --cov-report=html
# Open htmlcov/index.html to view coverage

# Run with coverage threshold (fail if below 80%)
pytest tests/ --cov=shebo --cov-fail-under=80
```

#### Test Organization

```
tests/
├── test_optimizer.py           # Main optimizer tests
├── test_ensemble.py            # Ensemble model tests
├── test_surrogate_manager.py   # Surrogate management tests
├── test_acquisition.py         # Acquisition function tests
├── test_constraint_discovery.py # Constraint discovery tests
├── test_black_box_solver.py    # Test solver validation
├── test_correctness.py         # Correctness validation tests
└── conftest.py                 # Shared fixtures (if needed)
```

#### Writing New Tests

**Structure Tests** - Verify components initialize and run:
```python
def test_optimizer_initialization():
    """Test that optimizer initializes correctly."""
    optimizer = SHEBOOptimizer(bounds=bounds, objective_fn=obj, n_init=10)
    assert optimizer.bounds.shape == (4, 2)
    assert optimizer.iteration == 0
```

**Correctness Tests** - Verify actual functionality:
```python
def test_optimization_improves():
    """Test that optimization finds better solutions."""
    optimizer = SHEBOOptimizer(bounds=bounds, objective_fn=obj, budget=50)
    result = optimizer.run()

    # Compare best to median
    performances = [p for p, c in zip(result.performance_history,
                                     result.convergence_history) if c]
    best = result.best_performance
    median = np.median(performances)

    assert best < median, "Optimizer should find better than median"
```

**Edge Case Tests** - Verify robustness:
```python
def test_handles_all_failures():
    """Test behavior when all initial samples fail."""
    def failing_objective(params):
        return {'output': {'convergence_status': False, ...}, 'performance': None}

    optimizer = SHEBOOptimizer(bounds=bounds, objective_fn=failing_objective)
    # Should not crash, should use exploration
    result = optimizer.run()
```

### Test Fixtures

Create reusable test components in `tests/conftest.py`:

```python
import pytest
import numpy as np
from shebo.utils.black_box_solver import create_test_objective

@pytest.fixture
def standard_bounds():
    """Standard 4D parameter bounds."""
    return np.array([
        [1e6, 1e10],
        [1e-8, 1e-4],
        [0.0, 1.0],
        [0.0, 1.0]
    ])

@pytest.fixture
def test_objective():
    """Standard test objective function."""
    return create_test_objective(n_params=4, random_seed=42)

@pytest.fixture
def optimizer(standard_bounds, test_objective):
    """Initialized optimizer."""
    from shebo import SHEBOOptimizer
    return SHEBOOptimizer(
        bounds=standard_bounds,
        objective_fn=test_objective,
        n_init=10,
        random_seed=42
    )
```

### Continuous Integration

The test suite is designed for CI/CD integration. Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest tests/ --cov=shebo --cov-fail-under=80
```

## Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                      SHEBO Optimizer                         │
├─────────────────────────────────────────────────────────────┤
│  Initialization (Sobol Sampling) → Main Loop → Results      │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┴────────┬──────────────┬─────────────────┐
    │                 │              │                 │
┌───▼────────┐  ┌────▼─────┐  ┌────▼────────┐  ┌────▼──────┐
│ Surrogate  │  │Constraint│  │ Adaptive    │  │Objective  │
│ Manager    │  │Discovery │  │Acquisition  │  │Function   │
└───┬────────┘  └────┬─────┘  └────┬────────┘  └───────────┘
    │                │              │
    │         ┌──────┴──────┐       │
    │         │             │       │
┌───▼─────┐  ┌▼────────┐  ┌▼───────▼─────┐
│Feature  │  │Anomaly  │  │Multi-Objective│
│Normalizer│  │Detection│  │Optimization  │
└─────────┘  └─────────┘  └──────────────┘
    │
┌───┴──────────────────────────────┐
│         Ensemble Models          │
├──────────────┬───────────────────┤
│ Convergence  │  Performance      │
│  Ensemble    │   Ensemble        │
│ (Binary)     │  (Regression)     │
└──────────────┴───────────────────┘
```

### Component Descriptions

#### SHEBOOptimizer (`shebo/core/optimizer.py`)

**Purpose**: Main optimization loop coordinator
**Key Methods**:
- `run()` - Execute optimization
- `_initialize()` - Sobol sampling
- `_update_surrogates()` - Trigger model updates
- `_select_next_point()` - Get next evaluation point
- `_get_next_batch()` - Get batch of points for parallel evaluation
- `save_checkpoint()` / `load_checkpoint()` - State persistence

**Extension Points**:
- Override `_initialize()` for custom initialization strategies
- Modify `_select_next_point()` for different acquisition strategies
- Customize `_should_retrain()` for different update schedules

#### SurrogateManager (`shebo/core/surrogate_manager.py`)

**Purpose**: Manages all surrogate models and training
**Key Methods**:
- `update_models()` - Train/update all models
- `predict()` - Get predictions with uncertainty
- `add_constraint_model()` - Dynamically add new constraint models
- `_train_model()` - Internal training logic
- `_validate_training_data()` - Data quality checks

**Extension Points**:
- Modify `_train_model()` for custom training procedures
- Override `_validate_training_data()` for domain-specific validation
- Add new model types beyond convergence/performance/constraints

#### Ensemble Models (`shebo/models/ensemble.py`)

**Purpose**: Neural network ensembles with uncertainty quantification
**Types**:
- `ConvergenceEnsemble` - Binary classification
- `PerformanceEnsemble` - Regression

**Key Methods**:
- `training_step()` - Independent network training
- `predict_with_uncertainty()` - Predictions + epistemic + aleatoric uncertainty
- `configure_optimizers()` - Separate optimizer per network

**Extension Points**:
- Modify network architecture in `ConvergenceNN` or `PerformanceNN`
- Change uncertainty quantification method
- Add new ensemble types (e.g., multi-class classification)

#### AdaptiveAcquisition (`shebo/core/acquisition.py`)

**Purpose**: Multi-objective acquisition function
**Components**:
- Expected Improvement (EI)
- Feasibility probability
- Uncertainty sampling
- Boundary exploration

**Extension Points**:
- Add new acquisition components
- Modify weights in `compute_acquisition()`
- Implement alternative optimization strategies in `optimize()`

#### ConstraintDiscovery (`shebo/core/constraint_discovery.py`)

**Purpose**: Automatic failure mode detection
**Detects**:
- Residual oscillation/divergence
- Numerical instability (NaN/Inf)
- Mesh quality issues
- Contact detection failures
- Excessive penetration

**Extension Points**:
- Add new constraint types
- Modify anomaly detection thresholds
- Implement domain-specific failure patterns

## Extension Guide

### Adding a New Acquisition Function

Create a new acquisition component:

```python
# In shebo/core/acquisition.py

def _compute_probability_of_feasibility(self, x: torch.Tensor) -> torch.Tensor:
    """Compute probability that points are feasible."""
    try:
        conv_pred = self.surrogate_manager.predict(x, 'convergence')
        prob_feasible = conv_pred['mean'].squeeze()
        return prob_feasible
    except (RuntimeError, ValueError) as e:
        logging.getLogger(__name__).debug(f"Could not compute PoF: {e}")
        return torch.zeros(len(x))

def _compute_knowledge_gradient(self, x: torch.Tensor) -> torch.Tensor:
    """Compute knowledge gradient for value of information."""
    # Your implementation here
    pass
```

Then integrate it in `compute_acquisition()`:

```python
def compute_acquisition(self, x: torch.Tensor, phase: str = 'balanced') -> torch.Tensor:
    ei = self._compute_expected_improvement(x)
    pof = self._compute_probability_of_feasibility(x)
    uncertainty = self._compute_uncertainty(x)
    boundary = self._compute_boundary_score(x)
    kg = self._compute_knowledge_gradient(x)  # New component

    if phase == 'balanced':
        return 0.3 * ei + 0.3 * pof + 0.2 * uncertainty + 0.1 * boundary + 0.1 * kg
    # ... other phases
```

### Adding a New Constraint Type

In `shebo/core/constraint_discovery.py`:

```python
def _check_energy_conservation(self, output: Dict[str, Any]) -> Optional[ConstraintViolation]:
    """Check if energy is conserved (custom constraint)."""
    if 'energy_history' not in output:
        return None

    energy = np.array(output['energy_history'])
    energy_drift = np.abs(energy[-1] - energy[0]) / (np.abs(energy[0]) + 1e-10)

    if energy_drift > 0.1:  # 10% drift threshold
        return ConstraintViolation(
            type='energy_conservation',
            description=f'Energy drift {energy_drift:.2%} exceeds 10% threshold',
            severity=np.clip(energy_drift / 0.1, 0, 1),
            parameters={}
        )
    return None
```

Then add to `check_simulation_output()`:

```python
def check_simulation_output(self, output: Dict[str, Any]) -> List[ConstraintViolation]:
    violations = []

    # Existing checks
    v = self._check_convergence_status(output)
    if v: violations.append(v)
    # ... other checks

    # New check
    v = self._check_energy_conservation(output)
    if v: violations.append(v)

    return violations
```

### Adding a New Surrogate Model Type

For a multi-class classifier:

```python
# In shebo/models/ensemble.py

class FailureModeEnsemble(pl.LightningModule):
    """Ensemble for multi-class failure mode classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_networks: int = 5,
        hidden_dims: List[int] = [128, 64, 32],
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # Manual optimization

        self.networks = nn.ModuleList([
            self._create_network(input_dim, num_classes, hidden_dims)
            for _ in range(n_networks)
        ])
        self.learning_rate = learning_rate

        # Move to device
        self.to(device)

    def _create_network(self, input_dim: int, num_classes: int,
                       hidden_dims: List[int]) -> nn.Module:
        """Create single classifier network."""
        layers = []
        dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))

        return nn.Sequential(*layers)

    def training_step(self, batch, batch_idx):
        x, y = batch
        optimizers = self.optimizers()

        total_loss = 0.0
        for network, optimizer in zip(self.networks, optimizers):
            optimizer.zero_grad()

            logits = network(x)
            loss = nn.CrossEntropyLoss()(logits, y.long())

            self.manual_backward(loss)
            optimizer.step()

            total_loss += loss.detach()

        avg_loss = total_loss / len(self.networks)
        self.log('train_loss', avg_loss)
        return avg_loss

    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict with uncertainty quantification."""
        self.eval()

        with torch.no_grad():
            # Get logits from each network
            all_logits = torch.stack([net(x) for net in self.networks])

            # Softmax to get probabilities
            all_probs = torch.softmax(all_logits, dim=-1)

            # Mean probabilities (predictive distribution)
            mean_probs = all_probs.mean(dim=0)

            # Epistemic uncertainty (variance across ensemble)
            epistemic = all_probs.var(dim=0).mean(dim=-1)

            # Aleatoric uncertainty (predictive entropy)
            aleatoric = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        return {
            'probabilities': mean_probs,
            'predicted_class': mean_probs.argmax(dim=-1),
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric
        }

    def configure_optimizers(self):
        return [
            torch.optim.Adam(network.parameters(), lr=self.learning_rate)
            for network in self.networks
        ]
```

Then integrate in `SurrogateManager`:

```python
def __init__(self, ...):
    # ... existing code
    self.models['failure_modes'] = FailureModeEnsemble(
        input_dim=input_dim,
        num_classes=5,  # e.g., 5 failure types
        n_networks=n_networks
    ).to(self.device)
```

### Adding Custom Initialization Strategies

Override the initialization in a custom optimizer:

```python
from shebo import SHEBOOptimizer
import numpy as np

class CustomInitOptimizer(SHEBOOptimizer):
    """Optimizer with custom initialization strategy."""

    def _initialize(self):
        """Custom initialization using Latin Hypercube instead of Sobol."""
        from scipy.stats import qmc

        # Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=self.bounds.shape[0], seed=self.random_seed)
        unit_samples = sampler.random(n=self.n_init)

        # Scale to bounds
        samples = qmc.scale(
            unit_samples,
            self.bounds[:, 0],
            self.bounds[:, 1]
        )

        # Evaluate initial samples
        for params in samples:
            self._evaluate_and_store(params)

        # Update surrogates
        self._update_surrogates()
```

### Custom Objective Functions

Integrate with your own simulation:

```python
import numpy as np
from typing import Dict, Any

def my_fem_objective(params: np.ndarray) -> Dict[str, Any]:
    """
    Custom FEM simulation objective.

    Args:
        params: [penalty, tolerance, timestep, damping]

    Returns:
        Dictionary with 'output' and 'performance' keys
    """
    penalty, tolerance, timestep, damping = params

    # Run your FEM simulation
    from my_fem_solver import run_contact_simulation

    result = run_contact_simulation(
        penalty=penalty,
        tolerance=tolerance,
        timestep=timestep,
        damping=damping
    )

    # Extract required output format
    output = {
        'convergence_status': result.converged,
        'residual_history': result.residuals.tolist(),
        'iterations': result.num_iterations,
        'solve_time': result.wall_time,
        'penetration_max': result.max_penetration,
        'jacobian_min': result.min_jacobian_eigenvalue,
        'contact_pairs': result.num_contact_pairs,
        'all_values': result.all_residuals.tolist(),
        'expected_contact': True
    }

    # Define performance metric (what to minimize)
    if result.converged:
        performance = result.num_iterations  # Minimize iterations
    else:
        performance = None

    return {
        'output': output,
        'performance': performance
    }

# Use with SHEBO
from shebo import SHEBOOptimizer

bounds = np.array([
    [1e6, 1e10],
    [1e-8, 1e-4],
    [0.001, 0.1],
    [0.0, 1.0]
])

optimizer = SHEBOOptimizer(
    bounds=bounds,
    objective_fn=my_fem_objective,
    n_init=20,
    budget=200,
    checkpoint_dir='./checkpoints',
    checkpoint_frequency=10
)

result = optimizer.run()
```

## Development Workflows

### Adding a New Feature

1. **Create feature branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Write tests first (TDD)**
   ```bash
   # Create test file
   touch tests/test_my_feature.py

   # Write failing tests
   pytest tests/test_my_feature.py -v  # Should fail
   ```

3. **Implement feature**
   - Write code in appropriate module
   - Follow best practices from CONTRIBUTING.md
   - Add type hints and docstrings

4. **Make tests pass**
   ```bash
   pytest tests/test_my_feature.py -v  # Should pass
   ```

5. **Run full test suite**
   ```bash
   pytest tests/ -v --cov=shebo
   ```

6. **Format and lint**
   ```bash
   black shebo/ tests/
   ruff check shebo/ tests/
   mypy shebo/
   ```

7. **Commit and push**
   ```bash
   git add .
   git commit -m "Add feature: description"
   git push origin feature/my-new-feature
   ```

### Debugging Optimization Issues

Enable detailed logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shebo_debug.log'),
        logging.StreamHandler()
    ]
)

# Run optimization
optimizer = SHEBOOptimizer(...)
result = optimizer.run()
```

Check convergence history:

```python
import matplotlib.pyplot as plt

# Plot convergence rate over time
conv_rate = np.cumsum(result.convergence_history) / np.arange(1, len(result.convergence_history) + 1)
plt.plot(conv_rate)
plt.xlabel('Iteration')
plt.ylabel('Success Rate')
plt.title('Convergence Rate Over Time')
plt.savefig('convergence_rate.png')
```

Analyze ensemble predictions:

```python
# Check ensemble diversity
from shebo.models.ensemble import ConvergenceEnsemble

ensemble = optimizer.surrogate_manager.models['convergence']
test_point = torch.randn(1, 4)

predictions = []
for network in ensemble.networks:
    network.eval()
    with torch.no_grad():
        pred = network(test_point).item()
        predictions.append(pred)

print(f"Predictions: {predictions}")
print(f"Std dev: {np.std(predictions):.4f}")  # Should be > 0.01 for good diversity
```

### Profiling Performance

Identify bottlenecks:

```python
import cProfile
import pstats
from pstats import SortKey

# Profile optimization
profiler = cProfile.Profile()
profiler.enable()

result = optimizer.run()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats(SortKey.CUMULATIVE)
stats.print_stats(20)  # Top 20 functions
```

Memory profiling:

```bash
pip install memory_profiler

# Add @profile decorator to functions you want to profile
python -m memory_profiler examples/simple_optimization.py
```

## Troubleshooting

### Common Issues

#### 1. "CUDA out of memory"

**Solution**: Reduce batch size or use CPU

```python
# Force CPU usage
optimizer = SHEBOOptimizer(..., device='cpu')

# Or reduce ensemble size
manager = SurrogateManager(n_networks=3, ...)  # Instead of 5
```

#### 2. "All ensemble networks give same predictions"

**Cause**: Ensemble training bug (should be fixed in current version)
**Check**: Verify using manual optimization mode

```python
ensemble = ConvergenceEnsemble(input_dim=4)
assert ensemble.automatic_optimization == False, "Should use manual optimization"
```

#### 3. "Poor convergence rate"

**Possible causes**:
- Insufficient initial samples
- Poor feature scaling
- Inappropriate bounds

**Solutions**:
```python
# Increase initial samples
optimizer = SHEBOOptimizer(n_init=50, ...)  # Instead of 20

# Check feature scaling
from shebo.utils.preprocessing import FeatureNormalizer
normalizer = FeatureNormalizer()
normalizer.fit(X_train)
print(f"Feature means: {normalizer.scaler.mean_}")
print(f"Feature stds: {normalizer.scaler.scale_}")

# Verify bounds are reasonable
print(f"Bounds range: {bounds[:, 1] / bounds[:, 0]}")
```

#### 4. "NaN in training loss"

**Cause**: Numerical instability
**Solution**: Check data normalization and learning rate

```python
# Verify normalization is applied
assert optimizer.surrogate_manager.normalizer.fitted, "Normalizer should be fitted"

# Reduce learning rate
ensemble = ConvergenceEnsemble(learning_rate=1e-4, ...)  # Instead of 1e-3
```

#### 5. "Tests failing on import"

**Solution**: Reinstall in development mode

```bash
pip install -e .  # Or uv pip install -e .
```

#### 6. "ModuleNotFoundError: No module named 'shebo.xxx'"

**Cause**: Missing `__init__.py` or incorrect installation
**Solution**:

```bash
# Check __init__.py exists
ls shebo/__init__.py shebo/core/__init__.py shebo/models/__init__.py

# Reinstall
pip uninstall shebo
pip install -e .
```

### Getting Help

1. **Check documentation**:
   - README.md - Overview and quick start
   - CONTRIBUTING.md - Contributing guidelines
   - DEVELOPMENT.md - This file
   - IMPLEMENTATION_PLAN.md - Detailed design

2. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Run tests with verbose output**:
   ```bash
   pytest tests/ -vv -s
   ```

4. **Check existing issues**: Search GitHub issues for similar problems

5. **Create minimal reproducible example**:
   ```python
   import numpy as np
   from shebo import SHEBOOptimizer
   from shebo.utils.black_box_solver import create_test_objective

   # Minimal example that reproduces the issue
   bounds = np.array([[0, 1], [0, 1]])
   obj = create_test_objective(n_params=2)
   opt = SHEBOOptimizer(bounds=bounds, objective_fn=obj, n_init=5)
   result = opt.run()  # Issue occurs here
   ```

## Additional Resources

- **PyTorch Lightning Docs**: https://lightning.ai/docs/pytorch/stable/
- **PyTorch Docs**: https://pytorch.org/docs/stable/
- **Bayesian Optimization**: "A Tutorial on Bayesian Optimization" by Shahriari et al.
- **Ensemble Methods**: "Uncertainty in Deep Learning" by Yarin Gal
- **Contact Mechanics**: Tribol documentation (if available)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Pull request process
- Development best practices

## License

[Specify your license]
