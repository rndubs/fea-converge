# Critical Fixes for SHEBO Implementation

This document provides detailed fixes for the critical issues identified in the code review.

## Fix 1: Ensemble Training Independence

### Problem
All networks trained with same gradient, defeating ensemble diversity.

### Solution
Train each network with its own optimizer and separate backward passes:

```python
# In ConvergenceEnsemble.__init__, create separate optimizers
def __init__(self, ...):
    super().__init__()
    self.save_hyperparameters()

    self.networks = nn.ModuleList([
        ConvergenceNN(input_dim, hidden_dims, dropout)
        for _ in range(n_networks)
    ])
    self.learning_rate = learning_rate

    # Create separate optimizer for each network
    self.automatic_optimization = False  # Disable automatic optimization

def training_step(self, batch, batch_idx):
    x, y = batch

    # Get all optimizers
    optimizers = self.optimizers()
    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    # Train each network independently
    total_loss = 0
    for network, optimizer in zip(self.networks, optimizers):
        optimizer.zero_grad()
        pred = network(x)
        loss = nn.BCELoss()(pred, y)
        self.manual_backward(loss)
        optimizer.step()
        total_loss += loss.detach()

    avg_loss = total_loss / len(self.networks)
    self.log('train_loss', avg_loss, prog_bar=True)
    return avg_loss

def configure_optimizers(self):
    # Return list of optimizers, one per network
    return [torch.optim.Adam(net.parameters(), lr=self.learning_rate)
            for net in self.networks]
```

---

## Fix 2: Sample Count Tracking

### Problem
`self.sample_count += len(X)` adds total size, not new samples.

### Solution
Redesign to pass iteration count instead of using cumulative sample count:

```python
# In SurrogateManager
def __init__(self, ...):
    # ... existing code ...
    self.sample_count = 0  # Tracks total samples seen
    self.last_update = {
        'convergence': 0,
        'performance': 0,
        'constraints': {}
    }

def should_update(self, model_name: str, current_iteration: int) -> bool:
    """Check if model should be updated."""
    if model_name == 'constraints':
        return False

    freq = self.update_schedules[model_name]
    iterations_since_last = current_iteration - self.last_update[model_name]

    return iterations_since_last >= freq and current_iteration > 0

def update_models(self, X, y_convergence, y_performance=None,
                  y_constraints=None, current_iteration=0, **kwargs):
    """Update models based on iteration count."""
    self.sample_count = len(X)  # Total samples in dataset

    # Update convergence model
    if self.should_update('convergence', current_iteration):
        self.last_update['convergence'] = current_iteration
        # ... train model ...

    # Similar for other models
```

Then in optimizer:
```python
def _update_surrogates(self):
    # ...
    self.surrogate_manager.update_models(
        X, y_convergence, y_performance, y_constraints,
        current_iteration=self.iteration  # Pass current iteration
    )
```

---

## Fix 3: Device Handling

### Problem
Models and data not consistently moved to correct device.

### Solution
Add comprehensive device management:

```python
# In SurrogateManager.__init__
def __init__(self, ...):
    # ... existing code ...

    # Move all initial models to device
    self.models['convergence'].to(self.device)
    self.models['performance'].to(self.device)

def add_constraint(self, name, constraint_type='binary'):
    """Add new constraint model."""
    if constraint_type == 'binary':
        model = ConvergenceEnsemble(self.input_dim, self.n_networks)
    else:
        model = PerformanceEnsemble(self.input_dim, n_networks=self.n_networks)

    # CRITICAL: Move to device immediately
    model = model.to(self.device)
    self.models['constraints'][name] = model

def _train_model(self, model, X, y, batch_size, val_split, model_name):
    """Train model with proper device handling."""
    # Move data to device BEFORE creating datasets
    X = X.to(self.device)
    y = y.to(self.device)

    # Ensure model is on correct device
    model = model.to(self.device)

    # ... rest of training ...
```

---

## Fix 4: Feature Normalization

### Problem
No normalization of features with vastly different scales.

### Solution
Add preprocessing pipeline:

```python
# New file: shebo/utils/preprocessing.py
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

class FeatureNormalizer:
    """Normalizes features for neural network training."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: np.ndarray):
        """Fit normalizer to data."""
        self.scaler.fit(X)
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform."""
        return self.scaler.inverse_transform(X)

# In SurrogateManager.__init__
def __init__(self, ...):
    # ... existing code ...
    self.normalizer = FeatureNormalizer()
    self.normalizer_fitted = False

def update_models(self, X, ...):
    """Update with normalization."""
    # Fit normalizer on first call
    if not self.normalizer_fitted:
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        self.normalizer.fit(X_np)
        self.normalizer_fitted = True

    # Normalize features
    X_np = X.numpy() if isinstance(X, torch.Tensor) else X
    X_normalized = self.normalizer.transform(X_np)
    X = torch.tensor(X_normalized, dtype=torch.float32)

    # Continue with training...

def predict(self, x, model_name='convergence'):
    """Predict with normalization."""
    # Normalize input
    if self.normalizer_fitted:
        x_np = x.numpy() if isinstance(x, torch.Tensor) else x
        x_normalized = self.normalizer.transform(x_np)
        x = torch.tensor(x_normalized, dtype=torch.float32)

    # Move to device and predict
    x = x.to(self.device)
    # ... rest of prediction ...
```

---

## Fix 5: Performance Data Shape

### Problem
Performance data duplicated incorrectly.

### Solution Option 1 - Single Output:
```python
# In SurrogateManager, change PerformanceEnsemble initialization
self.models = {
    'convergence': ConvergenceEnsemble(input_dim, n_networks),
    'performance': PerformanceEnsemble(
        input_dim,
        output_dim=1,  # Single output, not 2
        n_networks=n_networks
    ),
    'constraints': {}
}

# In optimizer._update_surrogates
y_performance = None
if any(self.convergence_status):
    perf_array = np.array(self.performance_values)
    perf_log = np.log1p(perf_array).reshape(-1, 1)  # Single column
    y_performance = torch.tensor(perf_log, dtype=torch.float32)

# In acquisition._compute_expected_improvement
perf_pred = self.surrogate_manager.predict(x, 'performance')
mean = perf_pred['mean'].squeeze()  # Now 1D
std = torch.sqrt(perf_pred['uncertainty'].squeeze())
```

### Solution Option 2 - Actual Separate Metrics:
```python
# In optimizer._evaluate_and_store
def _evaluate_and_store(self, x):
    result = self.objective_fn(x)
    output = result['output']

    # Store BOTH iteration count AND solve time
    iterations = output.get('iterations', 100)
    solve_time = output.get('solve_time', 1.0)

    self.all_params.append(x)
    self.all_outputs.append(output)
    self.convergence_status.append(converged)

    # Store as tuple
    self.performance_values.append((iterations, solve_time))

# In _update_surrogates
if any(self.convergence_status):
    # Extract both metrics
    iterations = np.array([p[0] for p in self.performance_values])
    times = np.array([p[1] for p in self.performance_values])

    # Log transform both
    iters_log = np.log1p(iterations).reshape(-1, 1)
    times_log = np.log1p(times).reshape(-1, 1)

    y_performance = torch.tensor(
        np.hstack([iters_log, times_log]),
        dtype=torch.float32
    )
```

---

## Fix 6: Data Validation

### Problem
No validation of training data quality.

### Solution:
```python
# In surrogate_manager._train_model
def _train_model(self, model, X, y, batch_size, val_split, model_name):
    """Train with data validation."""

    # Validate no NaN/Inf
    if torch.isnan(X).any() or torch.isinf(X).any():
        print(f"Warning: NaN/Inf in features for {model_name}, skipping training")
        return

    if torch.isnan(y).any() or torch.isinf(y).any():
        print(f"Warning: NaN/Inf in labels for {model_name}, skipping training")
        return

    # Check minimum samples
    n_samples = len(X)
    if n_samples < 10:
        print(f"Warning: Only {n_samples} samples for {model_name}, skipping training")
        return

    # For classification, check class balance
    if len(y.shape) == 2 and y.shape[1] == 1:  # Binary classification
        n_positive = (y == 1).sum().item()
        n_negative = (y == 0).sum().item()

        if n_positive < 3 or n_negative < 3:
            print(f"Warning: Insufficient samples per class ({n_positive}/{n_negative})")
            return

        # Warn about severe imbalance
        imbalance_ratio = max(n_positive, n_negative) / min(n_positive, n_negative)
        if imbalance_ratio > 10:
            print(f"Warning: Severe class imbalance ({imbalance_ratio:.1f}:1)")

    # Adjust validation split for small datasets
    if n_samples < 50:
        val_split = max(0.1, min(0.2, 5 / n_samples))  # At least 5 validation samples

    # ... continue with training ...
```

---

## Implementation Priority

1. **Fix ensemble training** (Critical for correctness)
2. **Add feature normalization** (Critical for training stability)
3. **Fix device handling** (Critical for GPU usage)
4. **Fix sample count tracking** (Important for correct behavior)
5. **Add data validation** (Important for robustness)
6. **Fix performance data shape** (Nice to have, workaround exists)

## Testing the Fixes

After applying fixes, run these tests:

```python
# Test ensemble diversity
ensemble = ConvergenceEnsemble(input_dim=4, n_networks=5)
# Train on data
# Check that networks give different predictions
preds = [net(test_X) for net in ensemble.networks]
assert not all(torch.allclose(preds[0], p) for p in preds[1:])

# Test normalization
manager = SurrogateManager(input_dim=4)
X_raw = torch.tensor([[1e8, 1e-6, 0.5, 0.5]])
# Train with normalization
# Check predictions are sensible

# Test device handling
if torch.cuda.is_available():
    manager = SurrogateManager(input_dim=4, device='cuda')
    # Ensure all operations work on GPU
```
