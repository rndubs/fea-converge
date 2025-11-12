# Surrogate Optimization with Hidden Constraints (SHEBO) Implementation Plan

## Overview

SHEBO combines surrogate modeling with constraint discovery, using ensemble approaches for robust feasibility prediction and adaptive sampling to map unknown convergence boundaries. This method excels when constraints are completely unknown a priori and when building reusable surrogate models across geometry families.

**Key Innovation**: Multiple surrogates (neural networks or GPs) for objectives and discovered constraints, using ensemble disagreement to quantify epistemic uncertainty and active learning to reveal hidden failure modes.

**Best For**: Production systems with many simulations, complex multi-constraint problems, transfer learning across geometries, and discovering unexpected failure modes.

---

## High-Level Implementation Checklist

### Phase 1: Foundation and Infrastructure (Weeks 1-2)
- [ ] Set up PyTorch/PyTorch Lightning environment
- [ ] Design data schema for multi-constraint, multi-geometry storage
- [ ] Implement database backend (PostgreSQL or MongoDB)
- [ ] Create parameter space encoding with feature engineering
- [ ] Build FE simulation executor with comprehensive output capture
- [ ] Set up experiment tracking (MLflow or Weights & Biases)

### Phase 2: Neural Network Ensemble Architecture (Weeks 3-5)
- [ ] Design NN architecture for convergence prediction (128→64→32, dropout)
- [ ] Implement ensemble training (5-10 networks with different initializations)
- [ ] Create training loop with early stopping and validation
- [ ] Implement ensemble prediction aggregation (mean, variance)
- [ ] Add uncertainty quantification (aleatoric + epistemic)
- [ ] Configure SMOTE for class imbalance if convergence failures <10%

### Phase 3: Surrogate Manager (Week 6)
- [ ] Implement Surrogate Manager coordinating multiple models
- [ ] Create performance surrogate (regression NN for iteration count, solve time)
- [ ] Build convergence surrogate (classification NN ensemble)
- [ ] Add constraint surrogates (one per discovered constraint type)
- [ ] Implement asynchronous model updates (fast: every 10 samples, expensive: every 50)
- [ ] Create model versioning and checkpointing

### Phase 4: Constraint Discovery Module (Week 7)
- [ ] Design expected behavior profiles (residual patterns, bounds)
- [ ] Implement anomaly detection for simulation outputs
- [ ] Create constraint labeling system for discovered failure modes:
  - [ ] Residual oscillations/increase
  - [ ] NaN/Inf values
  - [ ] Mesh quality degradation
  - [ ] Contact detection failures
  - [ ] Memory/time step crashes
- [ ] Build automatic surrogate creation for new constraints
- [ ] Implement constraint integration into acquisition function

### Phase 5: Adaptive Acquisition Function (Week 8)
- [ ] Implement multi-objective scalarization:
  - [ ] Performance optimization (EI)
  - [ ] Feasibility probability
  - [ ] Uncertainty reduction (entropy)
  - [ ] Boundary exploration
- [ ] Create adaptive weighting schedule (exploration → exploitation)
- [ ] Implement acquisition optimization with 10 restarts
- [ ] Add diversity promotion for batch parallelization
- [ ] Configure active learning integration (uncertainty sampling, QBC)

### Phase 6: Main Workflow Orchestration (Week 9)
- [ ] Implement space-filling initialization (Sobol/LHS)
- [ ] Create surrogate training pipeline
- [ ] Build constraint discovery loop
- [ ] Implement adaptive acquisition optimization
- [ ] Add termination criteria:
  - [ ] Budget exhausted
  - [ ] No new constraints in 30 iterations
  - [ ] Best solution stable for N iterations

### Phase 7: Transfer Learning and Multi-Fidelity (Week 10)
- [ ] Implement geometric feature extraction for similarity
- [ ] Build multi-task NN with geometry descriptors as inputs
- [ ] Create transfer learning pipeline (pre-train, fine-tune)
- [ ] Implement multi-fidelity extension (low/high fidelity models)
- [ ] Build auto-regressive correction: high_fi = α × low_fi + GP_correction
- [ ] Create budget allocation (cheap pre-screening, expensive refinement)

### Phase 8: Use Case Features (Week 11)
- [ ] Build surrogate models for geometry families
- [ ] Implement real-time surrogate-based monitoring (<10ms inference)
- [ ] Create pre-simulation prediction with uncertainty visualization
- [ ] Build online surrogate updates as simulations complete
- [ ] Implement model deployment for compute nodes

### Phase 9: Visualization and Dashboards (Week 12)
- [ ] Create ensemble disagreement heatmaps
- [ ] Build constraint discovery timeline visualization
- [ ] Implement multi-objective Pareto frontier plots
- [ ] Add real-time monitoring dashboard
- [ ] Create uncertainty visualization (violin plots, confidence intervals)

### Phase 10: Testing and Production Deployment (Weeks 13-14)
- [ ] Unit tests for NN training and ensemble predictions
- [ ] Integration tests for constraint discovery
- [ ] Validation on test geometries and known failure modes
- [ ] Load testing for real-time inference (<10ms)
- [ ] Model compression for lightweight deployment (<1MB)
- [ ] Production deployment infrastructure (model serving, monitoring)
- [ ] Documentation and user training

---

## Detailed Implementation Specifications

### 1. SHEBO Methodology Overview

**Core Problem**: Optimization when constraints are completely unknown a priori.

For contact convergence:
- Known constraint: convergence required
- Unknown constraints: numerical instability patterns, mesh distortion, unphysical states
- SHEBO discovers these during optimization

**Core Approach**:
1. Build **multiple surrogates** for objectives and discovered constraints
2. Use **ensemble disagreement** to quantify epistemic uncertainty
3. Employ **active learning** to sample high-uncertainty regions
4. **Discover hidden constraints** through anomaly detection
5. **Integrate constraints** into acquisition as they're discovered

**Surrogate Types**:
- Polynomial chaos expansions: Fast, limited complexity
- Radial basis functions: Local interpolation
- Gaussian processes: Uncertainty-aware, moderate scale
- **Neural network ensembles**: Flexible, high-capacity (recommended)
- Gradient-boosted trees: Discontinuities, categorical features

**For Contact Convergence**: Neural network ensembles recommended for:
- Complex, high-dimensional parameter spaces (>10 parameters)
- Non-smooth convergence boundaries
- Multiple interacting constraints
- Transfer learning across geometries

**Hidden Constraint Identification Workflow**:
1. During optimization, monitor for unexplained failures
2. Introduce new constraint models for discovered failure modes
3. Actively sample near suspected constraint boundaries
4. Integrate constraints into acquisition function
5. Iterate until no new constraint violations over N iterations

### 2. Ensemble Surrogate Modeling for Constraints

**Ensemble Architecture**:

Train 5-10 neural networks with:
- Identical architecture
- Different random initializations
- Different dropout patterns (MC dropout as alternative)
- Different training data subsets (bagging, optional)

Each network k predicts convergence probability: P_k(converge|x)

**Network Architecture for Convergence**:

```python
import torch
import torch.nn as nn

class ConvergenceNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

**Training Configuration**:

```python
import pytorch_lightning as pl

class ConvergenceEnsemble(pl.LightningModule):
    def __init__(self, input_dim, n_networks=5):
        super().__init__()
        self.networks = nn.ModuleList([
            ConvergenceNN(input_dim) for _ in range(n_networks)
        ])

    def forward(self, x):
        # Get predictions from all networks
        predictions = [net(x) for net in self.networks]
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)

        # Train each network independently
        losses = []
        for pred in predictions:
            loss = nn.BCELoss()(pred, y)
            losses.append(loss)

        total_loss = sum(losses) / len(losses)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Training with early stopping
from pytorch_lightning.callbacks import EarlyStopping

trainer = pl.Trainer(
    max_epochs=500,
    callbacks=[EarlyStopping(monitor='val_loss', patience=20)]
)
```

**Uncertainty Quantification**:

```python
def predict_with_uncertainty(ensemble, x):
    """
    Predict convergence probability with uncertainty quantification.

    Returns:
        mean: Ensemble mean prediction
        aleatoric_unc: Inherent randomness (average predictive entropy)
        epistemic_unc: Model uncertainty (variance of predictions)
    """
    predictions = []
    for network in ensemble.networks:
        network.eval()
        with torch.no_grad():
            pred = network(x)
            predictions.append(pred)

    predictions = torch.stack(predictions)

    # Ensemble mean
    mean = predictions.mean(dim=0)

    # Epistemic uncertainty (model uncertainty)
    epistemic_unc = predictions.var(dim=0)

    # Aleatoric uncertainty (average predictive entropy)
    epsilon = 1e-8  # Numerical stability
    entropy = -(predictions * torch.log(predictions + epsilon) +
                (1 - predictions) * torch.log(1 - predictions + epsilon))
    aleatoric_unc = entropy.mean(dim=0)

    return {
        'mean': mean,
        'epistemic_uncertainty': epistemic_unc,
        'aleatoric_uncertainty': aleatoric_unc,
        'total_uncertainty': epistemic_unc + aleatoric_unc
    }
```

**High Uncertainty Flags**:
- **Epistemic > 0.3**: Regions needing more data (acquisition target)
- **Aleatoric > 0.5**: Inherent randomness (boundary regions, chaotic dynamics)
- **Total > 0.5**: High-risk predictions (defer to expert review)

**Alternative: GP Ensembles**:

```python
# Different kernel families
kernels = [
    gpytorch.kernels.MaternKernel(nu=1.5),  # Matérn-3/2
    gpytorch.kernels.MaternKernel(nu=2.5),  # Matérn-5/2
    gpytorch.kernels.RBFKernel()            # Squared exponential
]

# Sample hyperparameters from posterior
gp_ensemble = []
for kernel in kernels:
    model = SingleTaskGP(train_X, train_Y, covar_module=kernel)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    gp_ensemble.append(model)

# Ensemble predictions
predictions = [gp(test_X).mean for gp in gp_ensemble]
ensemble_mean = torch.stack(predictions).mean(dim=0)
ensemble_var = torch.stack(predictions).var(dim=0)
```

**GP vs NN Ensembles**:
- **GPs**: More theoretically grounded, better for low-dimensional (<10D), smaller datasets
- **NNs**: More scalable to high dimensions (>10D), large datasets (>500 samples), complex boundaries

**Hybrid Approach**:
- GP for low-dimensional critical parameters (penalty, tolerance)
- NN for high-dimensional auxiliary parameters (mesh settings, solver options)

### 3. SHEBO System Architecture

**Surrogate Manager**:

Coordinates multiple models with asynchronous updates:

```python
class SurrogateManager:
    def __init__(self):
        self.models = {
            'convergence': ConvergenceEnsemble(input_dim),
            'performance': PerformanceNN(input_dim),  # Regression
            'constraints': {}  # Dynamically added as discovered
        }
        self.update_schedules = {
            'convergence': 10,  # Retrain every 10 samples
            'performance': 10,
            'constraints': 50   # Less frequent for discovered constraints
        }
        self.sample_count = 0

    def add_constraint(self, name, constraint_type='binary'):
        """
        Add new constraint model when failure mode discovered.
        """
        if constraint_type == 'binary':
            model = ConvergenceEnsemble(input_dim)  # Same architecture
        else:
            model = PerformanceNN(input_dim)  # Continuous constraint

        self.models['constraints'][name] = model
        self.update_schedules[f'constraints.{name}'] = 50
        print(f"New constraint discovered: {name}")

    def update_models(self, new_data):
        """
        Asynchronously update models based on schedules.
        """
        self.sample_count += 1

        for model_name, model in self.models.items():
            if model_name == 'constraints':
                for con_name, con_model in model.items():
                    if self.sample_count % self.update_schedules[f'constraints.{con_name}'] == 0:
                        self.train_model(con_model, new_data)
            else:
                if self.sample_count % self.update_schedules[model_name] == 0:
                    self.train_model(model, new_data)

    def train_model(self, model, data):
        """Train individual model."""
        # Implement training logic
        pass

    def predict(self, x, model_name='convergence'):
        """Get predictions from specific model."""
        model = self.models[model_name]
        return predict_with_uncertainty(model, x)
```

**Constraint Discovery Module**:

Implements anomaly detection:

```python
class ConstraintDiscovery:
    def __init__(self):
        self.discovered_constraints = {}
        self.expected_behaviors = {
            'residual_monotonic': 'Residual should decrease monotonically',
            'penetration_bounded': 'Penetration should remain < 1e-3',
            'no_nan_inf': 'No NaN or Inf values',
            'mesh_quality': 'Jacobian determinant > 0',
        }

    def check_simulation_output(self, output):
        """
        Analyze simulation output for anomalies.

        Args:
            output: Dict with simulation results
                - residual_history: List of residual norms
                - penetration_max: Maximum penetration
                - convergence_status: Boolean
                - jacobian_min: Minimum Jacobian determinant
                - etc.

        Returns:
            List of discovered constraint violations
        """
        violations = []

        # Check residual pattern
        if output['residual_history']:
            residuals = output['residual_history']
            # Check for non-monotonic decrease (after first 5 iters)
            if len(residuals) > 5:
                recent = residuals[-10:]
                if any(recent[i] > recent[i-1] for i in range(1, len(recent))):
                    violations.append({
                        'type': 'residual_oscillation',
                        'severity': 'medium',
                        'description': 'Residual oscillating/increasing'
                    })

        # Check for NaN/Inf
        if any(np.isnan(val) or np.isinf(val) for val in output.get('all_values', [])):
            violations.append({
                'type': 'numerical_instability',
                'severity': 'high',
                'description': 'NaN or Inf detected'
            })

        # Check penetration bounds
        if output.get('penetration_max', 0) > 1e-3:
            violations.append({
                'type': 'excessive_penetration',
                'severity': 'high',
                'description': f'Penetration {output["penetration_max"]} exceeds limit'
            })

        # Check mesh quality
        if output.get('jacobian_min', 1) <= 0:
            violations.append({
                'type': 'mesh_distortion',
                'severity': 'high',
                'description': 'Inverted elements detected'
            })

        # Check contact detection
        if output.get('contact_pairs', 0) == 0 and output.get('expected_contact', True):
            violations.append({
                'type': 'contact_detection_failure',
                'severity': 'medium',
                'description': 'No contact pairs found'
            })

        return violations

    def update_discovered_constraints(self, violations, surrogate_manager):
        """
        Add new constraint surrogates for discovered failure modes.
        """
        for violation in violations:
            con_type = violation['type']
            if con_type not in self.discovered_constraints:
                # New constraint discovered
                self.discovered_constraints[con_type] = {
                    'first_seen': len(surrogate_manager.sample_count),
                    'frequency': 1,
                    'severity': violation['severity']
                }
                # Add surrogate model
                surrogate_manager.add_constraint(con_type, constraint_type='binary')
            else:
                self.discovered_constraints[con_type]['frequency'] += 1
```

**Adaptive Acquisition Function**:

Balances multiple objectives with adaptive weighting:

```python
class AdaptiveAcquisition:
    def __init__(self, surrogate_manager):
        self.surrogate_manager = surrogate_manager
        self.iteration = 0

    def compute_acquisition(self, x, phase='exploration'):
        """
        Compute multi-objective acquisition value.

        α(x) = w_1·EI(x) + w_2·P(feasible|x) + w_3·H(x) + w_4·boundary_prox(x)
        """
        # 1. Expected Improvement (performance optimization)
        ei = self.compute_expected_improvement(x)

        # 2. Feasibility probability
        conv_pred = self.surrogate_manager.predict(x, 'convergence')
        p_feasible = conv_pred['mean']

        # 3. Entropy (uncertainty reduction)
        uncertainty = conv_pred['total_uncertainty']
        entropy = -(p_feasible * torch.log(p_feasible + 1e-8) +
                    (1 - p_feasible) * torch.log(1 - p_feasible + 1e-8))

        # 4. Boundary proximity
        boundary_prox = torch.exp(-5 * (p_feasible - 0.5)**2)

        # Adaptive weights based on phase
        weights = self.get_adaptive_weights(phase)

        acquisition = (weights['ei'] * ei +
                       weights['feasibility'] * p_feasible +
                       weights['uncertainty'] * uncertainty +
                       weights['boundary'] * boundary_prox)

        return acquisition

    def get_adaptive_weights(self, phase):
        """
        Adaptive weighting schedule.
        """
        schedules = {
            'exploration': {'ei': 0.1, 'feasibility': 0.2, 'uncertainty': 0.5, 'boundary': 0.2},
            'boundary_learning': {'ei': 0.2, 'feasibility': 0.2, 'uncertainty': 0.1, 'boundary': 0.5},
            'exploitation': {'ei': 0.5, 'feasibility': 0.3, 'uncertainty': 0.1, 'boundary': 0.1}
        }
        return schedules.get(phase, schedules['exploration'])

    def compute_expected_improvement(self, x):
        """Compute EI using performance surrogate."""
        perf_pred = self.surrogate_manager.predict(x, 'performance')
        # Implement EI calculation
        # ...
        return ei
```

**Main SHEBO Workflow**:

```python
class SHEBOOptimizer:
    def __init__(self, bounds, n_init=20, budget=200):
        self.bounds = bounds
        self.n_init = n_init
        self.budget = budget
        self.surrogate_manager = SurrogateManager()
        self.constraint_discovery = ConstraintDiscovery()
        self.adaptive_acquisition = AdaptiveAcquisition(self.surrogate_manager)
        self.data = []
        self.iteration = 0

    def run(self):
        """Main SHEBO optimization loop."""

        # Initialize with space-filling design
        X_init = self.generate_initial_samples(self.n_init)
        for x in X_init:
            self.evaluate_and_store(x)

        # Train initial surrogates
        self.surrogate_manager.update_models(self.data)

        # Main loop
        while self.iteration < self.budget:
            self.iteration += 1

            # Determine phase
            phase = self.determine_phase()

            # Discover constraints from recent failures
            self.discover_constraints()

            # Update constraint surrogates
            self.surrogate_manager.update_models(self.data)

            # Adaptive acquisition
            next_x = self.optimize_acquisition(phase)

            # Evaluate
            self.evaluate_and_store(next_x)

            # Check termination
            if self.check_termination():
                break

        return self.get_best_solution()

    def discover_constraints(self):
        """Check recent simulations for constraint violations."""
        for sample in self.data[-10:]:  # Check last 10 samples
            violations = self.constraint_discovery.check_simulation_output(sample['output'])
            if violations:
                self.constraint_discovery.update_discovered_constraints(
                    violations, self.surrogate_manager
                )

    def determine_phase(self):
        """Determine optimization phase based on progress."""
        if self.iteration < 30:
            return 'exploration'
        elif self.iteration < 100:
            return 'boundary_learning'
        else:
            return 'exploitation'

    def check_termination(self):
        """
        Termination criteria:
        - Budget exhausted
        - No new constraints in last 30 iterations
        - Best solution stable
        """
        if self.iteration >= self.budget:
            return True

        # Check for new constraints
        recent_discoveries = [
            con for con, info in self.constraint_discovery.discovered_constraints.items()
            if info['first_seen'] > self.iteration - 30
        ]
        no_new_constraints = len(recent_discoveries) == 0

        # Check solution stability
        best_stable = self.check_best_solution_stable(window=15)

        return no_new_constraints and best_stable and self.iteration > 100
```

### 4. Constraint Modeling Specifics

**Convergence Surrogate**:

Binary classification NN ensemble:

```python
class ConvergenceSurrogate:
    def __init__(self, input_dim, use_feature_engineering=True):
        self.ensemble = ConvergenceEnsemble(input_dim)
        self.use_feature_engineering = use_feature_engineering

    def engineer_features(self, raw_params):
        """
        Add engineered features to improve predictions.
        """
        features = raw_params.copy()

        # Add ratio features
        features['penalty_over_stiffness'] = raw_params['penalty'] / raw_params['material_stiffness']
        features['timestep_over_contact_timescale'] = raw_params['timestep'] / raw_params['contact_timescale']

        # Add interaction terms
        features['penalty_x_tolerance'] = raw_params['penalty'] * raw_params['tolerance']

        return features

    def train(self, X, y, class_balance=True):
        """
        Train ensemble with optional SMOTE for imbalanced classes.
        """
        # Engineer features
        if self.use_feature_engineering:
            X = self.engineer_features(X)

        # Handle class imbalance
        if class_balance and y.mean() < 0.1:  # <10% convergence rate
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            print(f"SMOTE: Augmented minority class. New class balance: {y.mean():.2f}")

        # Train ensemble
        self.ensemble.train(X, y)
```

**Performance Surrogate**:

Regression NN ensemble for iteration count and solve time:

```python
class PerformanceNN(nn.Module):
    def __init__(self, input_dim, output_dim=2):  # [log(iters), log(time)]
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Training: MSE loss on log-transformed targets
def train_performance_surrogate(model, X_success, y_iters, y_time):
    """
    Train only on successful trials.
    Log transform ensures positive predictions and stabilizes training.
    """
    y = torch.stack([torch.log(y_iters), torch.log(y_time)], dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    # ...
```

**Hidden Constraint Surrogates**:

One surrogate per discovered failure mode:

```python
# Example: Residual oscillation constraint
oscillation_surrogate = ConvergenceEnsemble(input_dim)
# Labels: 1 if residual oscillates, 0 if monotonic decrease
oscillation_labels = [detect_oscillation(sim['residual_history']) for sim in data]
oscillation_surrogate.train(X, oscillation_labels)

# Example: Mesh distortion constraint
mesh_surrogate = ConvergenceEnsemble(input_dim)
# Labels: 1 if mesh distorts (jacobian ≤ 0), 0 if valid
mesh_labels = [sim['jacobian_min'] <= 0 for sim in data]
mesh_surrogate.train(X, mesh_labels)
```

**Multi-Fidelity Extension**:

Use cheap low-fidelity simulations for pre-screening:

```python
class MultiFidelitySurrogate:
    def __init__(self):
        self.low_fidelity_model = ConvergenceEnsemble(input_dim)
        self.high_fidelity_model = ConvergenceEnsemble(input_dim)
        self.correction_gp = None

    def train(self, X_low, y_low, X_high, y_high):
        """
        Train low and high fidelity models, plus correction.
        """
        # Train low-fidelity model
        self.low_fidelity_model.train(X_low, y_low)

        # Train high-fidelity model
        self.high_fidelity_model.train(X_high, y_high)

        # Train auto-regressive correction: high = α × low + GP_correction
        low_fi_predictions = self.low_fidelity_model.predict(X_high)
        residuals = y_high - low_fi_predictions

        self.correction_gp = SingleTaskGP(X_high, residuals)
        mll = ExactMarginalLogLikelihood(self.correction_gp.likelihood, self.correction_gp)
        fit_gpytorch_model(mll)

    def predict(self, x):
        """Predict using multi-fidelity model."""
        low_fi_pred = self.low_fidelity_model.predict(x)
        correction = self.correction_gp(x).mean
        high_fi_pred = low_fi_pred + correction
        return high_fi_pred

# Workflow
# 1. Run cheap coarse mesh simulations (low fidelity) - 60-70% cost reduction
# 2. Pre-screen: if P_converge_low_fi > 0.7, proceed to high fidelity
# 3. Run expensive fine mesh simulations (high fidelity) on promising candidates
# 4. Train correction model
```

### 5. SHEBO Implementation Approach

**Ax Integration** (Custom Components):

SHEBO not natively in Ax, but can integrate:

```python
from ax import *

class SHEBOModelBridge(ModelBridge):
    """
    Custom ModelBridge wrapping SHEBO surrogate manager.
    """
    def __init__(self, surrogate_manager, **kwargs):
        self.surrogate_manager = surrogate_manager
        super().__init__(**kwargs)

    def _predict(self, X):
        """Use SHEBO surrogates for predictions."""
        predictions = self.surrogate_manager.predict(X, 'convergence')
        return predictions['mean'], predictions['epistemic_uncertainty']

# GenerationStrategy with custom SHEBO step
gs = GenerationStrategy([
    GenerationStep(model=Models.SOBOL, num_trials=20),  # Initialization
    GenerationStep(model=SHEBOModelBridge, num_trials=-1)  # SHEBO optimization
])

# Use with Ax Service API
ax_client = AxClient(generation_strategy=gs)
```

**PyTorch-Based Surrogates**:

Leverage PyTorch for GPU acceleration:

```python
# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ensemble.to(device)

# Batch predictions for efficiency
X_batch = torch.tensor(X_candidates).to(device)
with torch.no_grad():
    predictions = ensemble(X_batch)
```

**PyTorch Lightning for Training**:

Handles boilerplate (checkpointing, logging, early stopping):

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='convergence-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=True
)

trainer = Trainer(
    max_epochs=500,
    callbacks=[checkpoint_callback, early_stop_callback],
    gpus=1 if torch.cuda.is_available() else 0
)

trainer.fit(ensemble, train_dataloader, val_dataloader)
```

**Active Learning Integration**:

Use modAL or custom implementations:

```python
from modAL.uncertainty import uncertainty_sampling, entropy_sampling
from modAL.disagreement import max_disagreement_sampling

def query_by_committee(ensemble, X_pool):
    """
    Query-by-committee: maximize ensemble disagreement.
    """
    predictions = []
    for network in ensemble.networks:
        with torch.no_grad():
            pred = network(X_pool)
            predictions.append(pred)

    predictions = torch.stack(predictions)
    disagreement = predictions.var(dim=0)

    # Select point with maximum disagreement
    query_idx = disagreement.argmax()
    return X_pool[query_idx]

# Combined BO + Active Learning acquisition
def combined_acquisition(x, surrogate_manager, lambda_unc=0.5):
    """
    α_SHEBO(x) = α_BO(x) × (1 + λ × uncertainty(x))

    λ decays over time (explore early, exploit late).
    """
    ei = compute_expected_improvement(x, surrogate_manager)
    unc = surrogate_manager.predict(x, 'convergence')['total_uncertainty']

    return ei * (1 + lambda_unc * unc)
```

**Scalability via Approximations**:

For large parameter spaces (>20 dimensions):

1. **Random Projections**:
   ```python
   from sklearn.random_projection import GaussianRandomProjection
   projector = GaussianRandomProjection(n_components=10)
   X_reduced = projector.fit_transform(X)
   ```

2. **Active Subspaces**:
   ```python
   # Identify critical dimensions via gradient analysis
   gradients = compute_gradients(surrogate, X)
   C = gradients.T @ gradients  # Covariance matrix
   eigenvalues, eigenvectors = np.linalg.eig(C)
   # Use top-k eigenvectors as active subspace
   active_dims = eigenvectors[:, :k]
   ```

3. **Inducing Points for GPs**:
   ```python
   from gpytorch.models import ApproximateGP
   inducing_points = X[::10]  # Subsample or k-means
   model = VariationalGP(inducing_points)
   ```

4. **Batch Parallelization**:
   ```python
   # Evaluate 5-10 points simultaneously
   batch = select_diverse_batch(candidates, batch_size=5)
   results = parallel_evaluate(batch)
   ```

### 6. Use Case: Building Surrogate Models for Geometry Families

**Multi-Task NN with Geometry Descriptors**:

```python
class MultiGeometryNN(nn.Module):
    def __init__(self, param_dim, geom_feature_dim, output_dim=1):
        super().__init__()

        # Geometry encoder
        self.geom_encoder = nn.Sequential(
            nn.Linear(geom_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Combined network
        combined_dim = param_dim + 16  # Parameters + encoded geometry
        self.network = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid()  # For convergence probability
        )

    def forward(self, params, geom_features):
        geom_encoded = self.geom_encoder(geom_features)
        combined = torch.cat([params, geom_encoded], dim=1)
        return self.network(combined)
```

**Geometric Features**:
```python
def extract_geometry_features(geometry):
    """
    Extract features describing geometry for transfer learning.
    """
    features = {
        'contact_area': compute_contact_area(geometry),
        'mesh_size': geometry.element_count,
        'avg_element_size': geometry.avg_element_size,
        'material_contrast': max_stiffness / min_stiffness,
        'aspect_ratio': max_dimension / min_dimension,
        'gap_mean': np.mean(geometry.gap_distribution),
        'gap_variance': np.var(geometry.gap_distribution),
        'gap_max': np.max(geometry.gap_distribution),
        'curvature_max': compute_max_curvature(geometry)
    }
    return features
```

**Transfer Learning Pipeline**:

```python
class TransferLearning:
    def __init__(self):
        self.base_model = None
        self.geometry_database = {}

    def pretrain_on_database(self, geometries, data):
        """
        Pre-train on large geometry database (10-20 cases).
        """
        # Collect data across all geometries
        X_params = []
        X_geom = []
        y = []

        for geom_id, geom_data in data.items():
            geom_features = extract_geometry_features(geometries[geom_id])
            for sample in geom_data:
                X_params.append(sample['parameters'])
                X_geom.append(geom_features)
                y.append(sample['converged'])

        X_params = torch.tensor(X_params)
        X_geom = torch.tensor(X_geom)
        y = torch.tensor(y)

        # Train multi-geometry model
        self.base_model = MultiGeometryNN(param_dim, geom_feature_dim)
        self.train_model(self.base_model, X_params, X_geom, y)

    def fine_tune_for_geometry(self, new_geometry, new_samples, n_epochs=50):
        """
        Fine-tune pre-trained model on new geometry with few samples.
        """
        # Extract features for new geometry
        geom_features = extract_geometry_features(new_geometry)

        # Prepare data
        X_params = torch.tensor([s['parameters'] for s in new_samples])
        X_geom = torch.tensor([geom_features] * len(new_samples))
        y = torch.tensor([s['converged'] for s in new_samples])

        # Fine-tune with lower learning rate
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=1e-4)
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            pred = self.base_model(X_params, X_geom)
            loss = nn.BCELoss()(pred, y)
            loss.backward()
            optimizer.step()

        return self.base_model

# Expected performance
# - 70% fewer samples needed vs training from scratch
# - 20-30 samples sufficient for new geometry vs 100+ without transfer
```

### 7. Use Case: Real-Time Surrogate-Based Monitoring

**Lightweight Deployment**:

```python
class LightweightSurrogate:
    def __init__(self, model_path):
        # Load compressed model
        self.model = torch.load(model_path)
        self.model.eval()

        # Model compression for deployment
        self.compress_model()

    def compress_model(self):
        """
        Compress model for deployment (<1MB).
        """
        # Quantization (reduce precision)
        self.model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )

        # Pruning (remove small weights)
        import torch.nn.utils.prune as prune
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.3)

    def predict(self, x):
        """
        Fast inference (<10ms).
        """
        with torch.no_grad():
            return self.model(x)

# Deploy on compute nodes
# - Model size: <1MB after compression
# - Inference time: <10ms
# - Negligible overhead during simulation
```

**Mid-Simulation Monitoring**:

```python
class RealTimeMonitor:
    def __init__(self, surrogate):
        self.surrogate = surrogate

    def check_convergence_probability(self, iteration, residual_history, parameters):
        """
        Every K iterations, predict eventual convergence.
        """
        # Extract current state
        features = self.extract_trajectory_features(iteration, residual_history, parameters)

        # Query surrogate
        P_converge = self.surrogate.predict(features)

        return P_converge

    def extract_trajectory_features(self, iteration, residual_history, parameters):
        """
        Features from partial simulation state.
        """
        features = parameters.copy()

        # Trajectory features
        features['current_iteration'] = iteration
        features['current_residual'] = residual_history[-1]
        features['residual_rate'] = np.log(residual_history[-1]) - np.log(residual_history[max(0, -10)])
        features['stagnation'] = sum(1 for i in range(1, min(10, len(residual_history)))
                                       if abs(residual_history[-i] - residual_history[-i-1]) < 1e-12)

        return features

# Integration with simulation
if iteration % 5 == 0 and iteration > 10:
    P_converge = monitor.check_convergence_probability(iteration, residual_history, parameters)
    if P_converge < 0.3:
        # Terminate early
        terminate_simulation("Low convergence probability")

# Expected savings: 40% compute time by avoiding doomed simulations
```

### 8. Use Case: Pre-Simulation Prediction

**Interactive Prediction Interface**:

```python
class PreSimulationPredictor:
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def predict_outcomes(self, parameters):
        """
        Predict multiple outcomes before simulation.
        """
        # Get ensemble predictions
        pred = predict_with_uncertainty(self.ensemble, parameters)

        # Convergence probability
        P_converge = pred['mean'].item()
        uncertainty = pred['total_uncertainty'].item()

        # Expected iterations if converges (from performance surrogate)
        # ...

        # Visualization
        self.visualize_prediction(P_converge, uncertainty)

        return {
            'P_converge': P_converge,
            'uncertainty': uncertainty,
            'confidence': 'high' if uncertainty < 0.15 else 'medium' if uncertainty < 0.3 else 'low',
            'expected_iterations': expected_iters,
            'expected_time': expected_time,
            'risk_level': 'low' if P_converge > 0.8 else 'medium' if P_converge > 0.5 else 'high'
        }

    def visualize_prediction(self, mean, uncertainty):
        """
        Violin plots and confidence intervals.
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Convergence probability distribution
        samples = np.random.normal(mean, np.sqrt(uncertainty), 1000)
        samples = np.clip(samples, 0, 1)  # Bound to [0,1]

        ax1.violinplot([samples], positions=[0], showmeans=True, showextrema=True)
        ax1.set_ylabel('Convergence Probability')
        ax1.set_title(f'Predicted: {mean:.2f} ± {np.sqrt(uncertainty):.2f}')
        ax1.axhline(y=0.5, color='r', linestyle='--', label='Decision boundary')
        ax1.legend()

        # Uncertainty breakdown
        # (Would need separate aleatoric/epistemic values)
        ax2.bar(['Aleatoric', 'Epistemic'], [0.1, uncertainty], color=['blue', 'orange'])
        ax2.set_ylabel('Uncertainty')
        ax2.set_title('Uncertainty Breakdown')

        plt.tight_layout()
        return fig

    def suggest_alternatives(self, parameters, P_threshold=0.8):
        """
        If high risk, suggest safer alternatives.
        """
        if self.predict_outcomes(parameters)['P_converge'] < P_threshold:
            # Optimize for nearest high-probability parameters
            from scipy.optimize import minimize

            def objective(x):
                pred = predict_with_uncertainty(self.ensemble, x)
                # Minimize: distance + penalty for low P_converge
                return np.linalg.norm(x - parameters) + 10 * (P_threshold - pred['mean'])**2

            result = minimize(objective, parameters, bounds=parameter_bounds)
            alternative = result.x

            return {
                'status': 'High risk detected',
                'original_P_converge': self.predict_outcomes(parameters)['P_converge'],
                'suggested_alternative': alternative,
                'alternative_P_converge': self.predict_outcomes(alternative)['P_converge'],
                'distance': np.linalg.norm(alternative - parameters)
            }
        else:
            return {'status': 'Low risk, proceed'}
```

**Online Surrogate Updates**:

```python
class OnlineUpdater:
    def __init__(self, ensemble, update_frequency=10):
        self.ensemble = ensemble
        self.update_frequency = update_frequency
        self.new_data = []

    def add_result(self, parameters, outcome):
        """Add new simulation result."""
        self.new_data.append({'parameters': parameters, 'outcome': outcome})

        if len(self.new_data) >= self.update_frequency:
            self.update_ensemble()

    def update_ensemble(self):
        """
        Incremental update of ensemble.
        """
        # Combine with historical data
        X = torch.tensor([d['parameters'] for d in self.new_data])
        y = torch.tensor([d['outcome'] for d in self.new_data])

        # Fine-tune ensemble (few epochs, low learning rate)
        for network in self.ensemble.networks:
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
            for epoch in range(10):
                optimizer.zero_grad()
                pred = network(X)
                loss = nn.BCELoss()(pred, y)
                loss.backward()
                optimizer.step()

        # Clear buffer
        self.new_data = []
        print("Ensemble updated with new data")
```

### 9. Visualization Requirements

**Ensemble Disagreement Heatmap**:

```python
def visualize_ensemble_disagreement(ensemble, bounds):
    """
    2D parameter projection colored by ensemble disagreement.
    """
    # Generate grid
    x1 = np.linspace(bounds[0,0], bounds[0,1], 100)
    x2 = np.linspace(bounds[1,0], bounds[1,1], 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Compute disagreement
    disagreement = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = torch.tensor([[X1[i,j], X2[i,j]]])
            pred = predict_with_uncertainty(ensemble, x)
            disagreement[i,j] = pred['epistemic_uncertainty'].item()

    # Plot
    plt.figure(figsize=(10, 8))
    im = plt.contourf(X1, X2, disagreement, levels=20, cmap='RdYlBu_r')
    plt.colorbar(im, label='Ensemble Disagreement (Epistemic Uncertainty)')

    # Overlay sample locations
    # (would need historical data)
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title('Ensemble Disagreement Heatmap\n(Red: high uncertainty, needs data; Blue: confident)')

    return plt
```

**Constraint Discovery Timeline**:

```python
def visualize_constraint_timeline(constraint_discovery, iteration_max):
    """
    Timeline showing discovered constraints over iterations.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    y_pos = 0
    for con_name, info in constraint_discovery.discovered_constraints.items():
        # Plot constraint discovery as horizontal line
        ax.barh(y_pos, iteration_max - info['first_seen'],
                left=info['first_seen'],
                height=0.8,
                label=con_name,
                alpha=0.7)

        # Annotate with details
        ax.text(info['first_seen'], y_pos,
                f" {con_name}\n Freq: {info['frequency']}, Severity: {info['severity']}",
                va='center', fontsize=9)

        y_pos += 1

    ax.set_xlabel('Iteration')
    ax.set_yticks([])
    ax.set_title('Constraint Discovery Timeline')
    ax.axvline(x=0, color='black', linewidth=2)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    return fig
```

**Multi-Objective Pareto Frontier**:

```python
def visualize_pareto_frontier(data, feasibility_model):
    """
    3D surface: parameter space with time vs accuracy trade-off.
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract converged samples
    converged = [d for d in data if d['converged']]

    params1 = [d['parameters'][0] for d in converged]
    params2 = [d['parameters'][1] for d in converged]
    times = [d['solve_time'] for d in converged]
    errors = [d['final_residual'] for d in converged]

    # Color by error (accuracy)
    scatter = ax.scatter(params1, params2, times, c=errors, cmap='viridis',
                         s=50, alpha=0.6)
    fig.colorbar(scatter, label='Final Residual (Error)', shrink=0.5)

    # Identify Pareto-optimal points (non-dominated)
    pareto_points = compute_pareto_frontier(times, errors)
    ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2],
               color='red', s=200, marker='*', label='Pareto Optimal', zorder=10)

    # Feasibility constraint surface (transparent)
    # (Would need to compute from model)

    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_zlabel('Solve Time')
    ax.set_title('Multi-Objective Trade-off: Time vs Accuracy')
    ax.legend()

    return fig
```

---

## Performance Characteristics

### Expected Success Rates

**Initial Phase (0-50 trials)**:
- Random initialization: 30-40% convergence
- Ensemble learning begins: 50-60% by trial 50
- Constraint discovery identifies 2-3 failure modes

**Mid-term Phase (50-200 trials)**:
- Success rate: 85-90%
- Ensemble surrogates well-trained
- Most constraints discovered

**Mature Phase (>200 trials)**:
- Success rate: >90%
- Excellent generalization to new geometries (with transfer learning)
- Robust multi-constraint handling

### Computational Efficiency

- **vs Grid Search**: 50-100x speedup
- **vs Random Search**: 5-10x speedup
- **Sample Efficiency**: Fair initially, Excellent with transfer learning
- **Training Cost**: High initially (100+ samples), amortized over many geometries

### Production Performance

- **Inference Latency**: <5ms (lightweight NN deployment)
- **Model Size**: <1MB (compressed)
- **Batch Throughput**: 1000+ predictions/sec (GPU)
- **Real-time Monitoring**: <1% simulation overhead

---

## Method Strengths and Weaknesses

### Strengths
- Handles complex multi-constraint problems naturally
- Robust ensemble uncertainty quantification
- Discovers hidden failure modes proactively
- Scales to high dimensions with NNs
- Transfer learning across geometry families
- Production-ready deployment (fast inference)
- Multi-fidelity integration for cost reduction

### Weaknesses
- Requires substantial training data initially (100-200 samples)
- Ensemble management adds implementation complexity
- May miss rare failure modes (requires diverse data)
- Computationally expensive training (GPU recommended)
- More complex than GP-based methods

---

## Deployment Recommendations

**When to Use SHEBO**:
1. Production systems with many routine simulations
2. Geometry families requiring transfer learning
3. Complex, multi-constraint problems
4. High-dimensional parameter spaces (>10D)
5. When amortizing training cost over many uses
6. Real-time monitoring requirements

**When to Use Alternative**:
- Limited data (<100 samples) → GP Classification or FR-BO
- Need formal guarantees → CONFIG
- Simpler constraint landscape → GP Classification

**Deployment Strategy (Weeks 9-12)**:
- Train on aggregated data from GP Classification phase (200+ samples)
- Deploy lightweight NNs for real-time validation and monitoring
- Maintain continuous learning pipeline
- Use for production-scale applications

---

## Implementation Tools

### Required Software
- **Python 3.8+**
- **PyTorch 1.12+** (GPU support recommended)
- **PyTorch Lightning** (training infrastructure)
- **NumPy, SciPy, Pandas**
- **scikit-learn** (preprocessing, SMOTE)
- **imbalanced-learn** (class balancing)

### Recommended Tools
- **MLflow** or **Weights & Biases** (experiment tracking)
- **Optuna** (hyperparameter tuning)
- **ONNX** (model export for deployment)
- **TorchServe** (model serving)
- **Plotly/Dash** (interactive dashboards)

### Infrastructure
- **Database**: PostgreSQL or MongoDB (multi-geometry data)
- **Caching**: Redis (frequent predictions)
- **Model Registry**: MLflow Model Registry
- **CI/CD**: GitHub Actions + Docker
- **Monitoring**: Prometheus + Grafana (production metrics)

---

## Next Steps

1. Review implementation plan and resource requirements
2. Set up PyTorch/Lightning development environment
3. Design database schema for multi-geometry, multi-constraint data
4. Implement basic NN ensemble for convergence prediction
5. Create constraint discovery module prototype
6. Test on initial simulation data (50-100 samples)
7. Develop visualization prototypes
8. Establish training and deployment pipelines
9. Plan transfer learning experiments across geometries
10. Schedule integration with production workflow

---

## References and Further Reading

- Neural Network Ensembles: Dietterich, "Ensemble Methods in Machine Learning"
- Uncertainty Quantification: Gal & Ghahramani, "Dropout as a Bayesian Approximation"
- Active Learning: Settles, "Active Learning Literature Survey"
- Multi-Fidelity Optimization: Forrester et al., "Multi-Fidelity Optimization via Surrogate Modelling"
- Transfer Learning: Pan & Yang, "A Survey on Transfer Learning"
- SMOTE for Imbalanced Classes: Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique"
- PyTorch Lightning: https://pytorch-lightning.readthedocs.io/
- Model Compression: Han et al., "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding"
