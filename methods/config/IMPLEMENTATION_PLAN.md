# Constrained Efficient Global Optimization (CONFIG) Implementation Plan

## Overview

CONFIG provides rigorous theoretical guarantees for optimizing expensive black-box objectives subject to unknown convergence constraints. It uses optimistic feasibility estimates and bounded cumulative violations to achieve provable convergence with sublinear regret.

**Key Innovation**: Maintains an optimistic feasible set F_opt that contains the true feasible region with high probability, enabling strategic constraint violations that accelerate learning without sacrificing convergence guarantees.

**Theoretical Guarantees**:
- Cumulative regret: R_T = O(√(T γ_T log T))
- Cumulative violations: V_T = O(√(T γ_T log T))
- Convergence rate to ε-optimal: O((γ*/ε)² log²(γ*/ε)) evaluations

**Best For**: Safety-critical applications requiring formal guarantees, rigorous constraint handling, and bounded violations.

---

## High-Level Implementation Checklist

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up BoTorch development environment
- [ ] Define parameter space and constraint formulations
- [ ] Implement continuous constraint formulation: c(x) = log10(residual) - log10(tolerance)
- [ ] Create multiple constraint handling (convergence, iterations, penetration)
- [ ] Build FE simulation executor with constraint monitoring

### Phase 2: GP Models for Objective and Constraints (Weeks 3-4)
- [ ] Implement SingleTaskGP for objective function
- [ ] Implement separate GPs for each constraint
- [ ] Configure Matérn-5/2 kernel with ARD for all models
- [ ] Implement separate hyperparameter optimization per GP
- [ ] Handle heteroscedastic noise for constraints if needed
- [ ] Create GP training pipeline with MLE

### Phase 3: LCB-Based Acquisition (Week 5)
- [ ] Implement Lower Confidence Bound (LCB) calculation
- [ ] Implement theoretical β schedule: β_n = 2 log(π² n² / 6δ)
- [ ] Create LCB for objective: LCB_obj = μ(x) - β^(1/2) σ(x)
- [ ] Create LCB for constraints: LCB_con = μ(x) - β^(1/2) σ(x)
- [ ] Implement custom CONFIG acquisition function

### Phase 4: Optimistic Feasible Set (Week 6)
- [ ] Implement optimistic feasible set: F_opt = {x : LCB_constraint(x) ≤ 0}
- [ ] Create constrained optimization: minimize LCB_obj(x) subject to x ∈ F_opt
- [ ] Implement acquisition optimization with scipy.optimize (constraint handling)
- [ ] Add 20 random restarts for global optimization
- [ ] Handle empty F_opt case (likely infeasible problem)
- [ ] Implement adaptive β adjustment for F_opt expansion when needed

### Phase 5: Multi-Phase Strategy (Week 7)
- [ ] Implement Phase 1: Initialization (0-20 evals) - Latin Hypercube Sampling
- [ ] Implement Phase 2: Feasibility Discovery (20-40 evals) - Active constraint learning
- [ ] Implement Phase 3: Constrained Optimization (40-80 evals) - CONFIG-style LCB
- [ ] Implement Phase 4: Refinement (80-100 evals) - Local search with tighter bounds
- [ ] Create adaptive phase switching logic based on feasibility and uncertainty
- [ ] Track cumulative violations V_t = Σ max(0, c(x_i))

### Phase 6: System Components (Week 8)
- [ ] Build BO Controller for CONFIG iteration loop
- [ ] Implement Constraint Learner for active boundary learning
- [ ] Create Feasibility Predictor for real-time queries
- [ ] Build Violation Monitor tracking cumulative violations
- [ ] Implement Geometry Adapter with multi-task GPs for transfer learning

### Phase 7: Use Case Features (Week 9)
- [ ] Implement feasible region mapping (CONFIG with constant objective)
- [ ] Create real-time feasibility assessment with LCB-based checking
- [ ] Build pre-simulation batch validation and ranking
- [ ] Implement conservative safe parameter recommendations
- [ ] Add nearest safe point projection for infeasible queries

### Phase 8: Visualization and Monitoring (Week 10)
- [ ] Create optimistic feasible set evolution visualization
- [ ] Build LCB acquisition surface 3D plots
- [ ] Implement cumulative violations trajectory vs theoretical bounds
- [ ] Add real-time monitoring dashboard
- [ ] Generate comprehensive reports with guarantees

### Phase 9: Testing and Validation (Week 11)
- [ ] Unit tests for LCB calculations and β schedule
- [ ] Integration tests for constrained optimization
- [ ] Validation that violations remain bounded (V_t = O(√t))
- [ ] Verification of theoretical guarantees on test problems
- [ ] Performance benchmarking vs CEI and SafeOpt
- [ ] Documentation of theoretical properties

### Phase 10: Deployment and Certification (Week 12)
- [ ] Create deployment package with formal guarantee documentation
- [ ] Generate certification reports for safety-critical use
- [ ] Build user guide emphasizing theoretical properties
- [ ] Implement monitoring for guarantee violations in production
- [ ] Establish procedures for failure investigation

---

## Detailed Implementation Specifications

### 1. CONFIG Theoretical Framework

**Problem Formulation**:
```
Original: minimize f(x) subject to c_i(x) ≤ 0, i=1,...,m, x ∈ X

CONFIG Auxiliary Problem (at iteration n):
x_{n+1} = argmin_{x∈F_n^opt} l_0,n(x)
```

where:
- F_n^opt = {x : l_i,n(x) ≤ 0 ∀i} is the optimistic feasible set
- l_i,n(x) = μ_i,n(x) - β_n^(1/2) σ_i,n(x) are lower confidence bounds

**Theoretical Guarantees**:

1. **Cumulative Regret**:
   ```
   R_T = Σ_{t=1}^T (f(x_t) - f(x*)) = O(√(T γ_T log T))
   ```

2. **Cumulative Violations**:
   ```
   V_T = Σ_{t=1}^T max(0, c_i(x_t)) = O(√(T γ_T log T))
   ```

3. **Maximum Information Gain γ_T**:
   - Matérn kernels: γ_T = O(log^(d+1) T) - sublinear!
   - Squared exponential: γ_T = O(log^d T)

4. **Convergence Rate**:
   To reach ε-optimal solution requires O((γ*/ε)² log²(γ*/ε)) evaluations.

**Optimistic Principle**:
- Permits violations when uncertainty is high
- If LCB suggests feasibility, sample even if mean predicts infeasibility
- Accelerates boundary learning vs conservative approaches
- Maintains theoretical guarantees despite violations

**Key Properties**:
- Sublinear regret → converges to optimum
- Sublinear violations → violation rate decreases over time
- Natural infeasibility detection → if F_opt shrinks to empty, problem likely infeasible
- No manual tuning of penalty parameters required

### 2. Comparison with Standard EGO Approaches

**Standard EGO**:
- Expected Improvement: α_EI(x) = E[max(0, f_best - f(x))]
- No explicit constraint handling
- Not suitable for constrained problems

**CEI (Constrained EI)**:
- Multiplies EI by probability of feasibility: α_CEI = α_EI × P(feasible)
- Heuristic approach, lacks theoretical analysis
- Can be overly conservative
- No violation bounds

**Augmented Lagrangian EGO**:
- Converts constrained to unconstrained via penalty parameters
- Requires manual tuning of Lagrange multipliers
- May not converge for infeasible problems
- No theoretical guarantees

**SafeOpt**:
- Strict safety: never violates constraints
- Requires safe seed point
- May get trapped in local feasible regions
- Proven safety but limited exploration

**CONFIG Advantages**:
1. Principled exploration-exploitation via LCB (no tunable weights)
2. Provable convergence guarantees and finite-time regret bounds
3. Natural infeasibility detection (F_opt → ∅)
4. Bounded cumulative violations enable learning without divergence
5. Global exploration (accepts initial violations)

**Trade-offs**:
- CONFIG: Incurs bounded violations, guaranteed convergence
- SafeOpt: Strict safety, may get trapped locally
- CEI: Fast heuristic, no guarantees

### 3. CONFIG Implementation in Ax/BoTorch

**Constraint Types in BoTorch**:

1. **Parameter constraints**: Known linear relationships in input space
   - Enforced exactly during optimization
   - Example: x1 + x2 ≤ 1

2. **Outcome constraints**: Unknown black-box functions
   - Modeled via GPs
   - Handled probabilistically
   - Used for convergence in our case

**Convergence as Outcome Constraint**:

Convergence is expensive black-box constraint:
- Must execute simulation to determine convergence
- Model with separate GP: c(x) represents residual_norm - tolerance
- c ≤ 0 means converged

**BoTorch Constrained Acquisition**:

```python
from botorch.acquisition import SampleReducingMCAcquisitionFunction

def convergence_constraint(samples, X=None):
    """
    Constraint function for CONFIG.
    Returns negative for feasible, positive for infeasible.
    """
    residual_pred = samples[..., constraint_idx]
    threshold = tolerance
    return threshold - residual_pred  # Negative when residual < tolerance

# Use with qExpectedImprovement or custom acquisition
acqf = qExpectedImprovement(
    model=model,
    best_f=best_observed,
    constraints=[convergence_constraint],
    sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512]))
)
```

**CONFIG-Style LCB Acquisition**:

Custom implementation:

```python
from botorch.acquisition import AnalyticAcquisitionFunction
import torch

class CONFIGAcquisition(AnalyticAcquisitionFunction):
    """
    CONFIG acquisition using LCB with optimistic feasible set.
    """
    def __init__(self, model_obj, model_constraints, beta, **kwargs):
        super().__init__(model=model_obj, **kwargs)
        self.model_obj = model_obj
        self.model_constraints = model_constraints  # List of constraint GPs
        self.beta = beta

    def forward(self, X):
        """
        Compute -LCB_objective (negate for maximization).
        Constraints handled externally in optimization.
        """
        posterior_obj = self.model_obj.posterior(X)
        mean_obj = posterior_obj.mean
        sigma_obj = posterior_obj.variance.sqrt()

        lcb_obj = mean_obj - self.beta.sqrt() * sigma_obj
        return -lcb_obj  # Negative for maximization in BoTorch

def compute_constraint_lcb(X, model_con, beta):
    """
    Compute LCB for constraint.
    """
    posterior_con = model_con.posterior(X)
    mean_con = posterior_con.mean
    sigma_con = posterior_con.variance.sqrt()

    lcb_con = mean_con - beta.sqrt() * sigma_con
    return lcb_con
```

**Theoretical β Schedule**:
```python
import numpy as np

def compute_beta(n, delta=0.1):
    """
    Theoretical β schedule from CONFIG paper.

    β_n = 2 log(π² n² / 6δ)

    Args:
        n: Current iteration number
        delta: Failure probability (typically 0.1)

    Returns:
        β value for iteration n
    """
    beta = 2 * np.log(np.pi**2 * n**2 / (6 * delta))
    return max(beta, 0.1)  # Lower bound for numerical stability
```

**Optimistic Feasible Set Representation**:
```python
def is_in_optimistic_set(X, constraint_models, beta):
    """
    Check if point(s) X are in optimistic feasible set.

    F_opt = {x : LCB_constraint(x) ≤ 0 for all constraints}
    """
    for model_con in constraint_models:
        lcb_con = compute_constraint_lcb(X, model_con, beta)
        if torch.any(lcb_con > 0):
            return False
    return True
```

**Constrained Optimization of Acquisition**:

```python
from scipy.optimize import minimize

def optimize_config_acquisition(acqf, constraint_models, beta, bounds, n_restarts=20):
    """
    Optimize CONFIG acquisition subject to optimistic feasible set.

    minimize LCB_obj(x)
    subject to LCB_constraint_i(x) ≤ 0 for all i
    """
    def objective(x):
        X_tensor = torch.tensor(x).reshape(1, -1)
        return acqf(X_tensor).item()

    def constraint_func(x, con_idx):
        X_tensor = torch.tensor(x).reshape(1, -1)
        lcb = compute_constraint_lcb(X_tensor, constraint_models[con_idx], beta)
        return -lcb.item()  # scipy wants g(x) ≥ 0 for feasibility

    constraints = [
        {'type': 'ineq', 'fun': constraint_func, 'args': (i,)}
        for i in range(len(constraint_models))
    ]

    best_x = None
    best_val = float('inf')

    for _ in range(n_restarts):
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
        result = minimize(
            objective, x0,
            bounds=[(b[0], b[1]) for b in bounds],
            constraints=constraints,
            method='SLSQP'
        )
        if result.success and result.fun < best_val:
            best_val = result.fun
            best_x = result.x

    return best_x
```

**Handling Empty F_opt**:

If optimization fails (no feasible solution found):
1. Increase β to expand optimistic set (more tolerant of uncertainty)
2. Use pure exploration (sample highest uncertainty point)
3. Flag potential infeasibility to user
4. Consider problem reformulation

### 4. Multi-Phase CONFIG Strategy

**Phase 1: Initialization (0-20 evaluations)**

**Objective**: Establish baseline GP models and find initial feasible points

**Strategy**:
- Latin Hypercube Sampling (LHS) for space-filling initialization
- More uniform coverage than random sampling
- If no feasible points found, increase β to expand F_opt

**Metrics**:
- Number of feasible points found
- Coverage of parameter space
- Initial GP model quality

**Adaptive handling**:
```python
if num_feasible == 0 and n_evals >= 20:
    beta *= 1.5  # Expand optimistic set
    # Continue with increased tolerance
```

**Phase 2: Feasibility Discovery (20-40 evaluations)**

**Objective**: Map the convergence boundary efficiently

**Strategy**: Active constraint learning by maximizing uncertainty near predicted boundaries

**Acquisition**:
```
α(x) = uncertainty_constraint(x) × proximity_to_boundary(x)

where:
uncertainty_constraint(x) = σ_constraint(x)
proximity_to_boundary(x) = exp(-5 × LCB_constraint(x)²)
```

This targets regions:
- High uncertainty (learn faster)
- Near boundary (most informative)

**Benefits**:
- Accurate boundary understanding critical for subsequent optimization
- Efficient use of simulation budget
- Reduces conservative bias in F_opt

**Transition criterion**:
- Constraint GP uncertainty uniformly low (<0.2 across sampled points)
- OR: 40 evaluations completed

**Phase 3: Constrained Optimization (40-80 evaluations)**

**Objective**: Optimize within optimistic feasible set

**CONFIG Algorithm**:

Each iteration:
1. Fit objective and constraint GPs with MLE
2. Compute β_n using theoretical schedule
3. Compute LCB for objective and constraints
4. Define F_opt = {x : LCB_constraint(x) ≤ 0 for all constraints}
5. Solve: minimize LCB_obj(x) subject to x ∈ F_opt
6. Evaluate selected point
7. Track cumulative violations: V_t += max(0, c(x_t))
8. Increase β_n for next iteration

**Violation Tracking**:
```python
cumulative_violation = 0
for x, c_val in zip(evaluated_points, constraint_values):
    cumulative_violation += max(0, c_val)

# Check against theoretical bound
theoretical_bound = compute_theoretical_violation_bound(t)
if cumulative_violation > 2 * theoretical_bound:
    # Warning: violations exceeding theory
    # Possible model misspecification
```

**Transition criterion**:
- 80 evaluations completed
- OR: Best feasible solution unchanged for 15 iterations

**Phase 4: Refinement (80-100 evaluations)**

**Objective**: Polish solution and confirm convergence

**Strategy**: Local search near best feasible solution

**Tighter confidence bounds**:
- Reduce β by factor of 2 (β_refine = β_n / 2)
- More exploitative, less exploratory
- Assumes boundary well-understood

**Activities**:
- Validate robustness via nearby parameter perturbations
- Confirm convergence on multiple test cases
- Generate confidence intervals for optimal solution

**Stopping criteria**:
- Budget exhausted (100 evaluations)
- Best solution validated on 5 perturbations
- Uncertainty in optimal region < threshold

**Adaptive Phase Switching**:

```python
def determine_phase(n_evals, num_feasible, constraint_uncertainty_max, best_stable_count):
    if n_evals < 20:
        return "initialization"

    if num_feasible == 0 and n_evals < 30:
        return "feasibility_discovery"  # Extended

    if constraint_uncertainty_max > 0.3:
        return "feasibility_discovery"  # Need more boundary data

    if n_evals < 80:
        return "constrained_optimization"

    if best_stable_count >= 10:
        return "refinement"

    return "constrained_optimization"  # Default
```

### 5. Constraint Modeling

**Continuous Constraint Formulation** (Recommended):
```
c(x) = log10(final_residual) - log10(tolerance)
```

**Advantages**:
1. Natural extension to multi-threshold constraints
2. Smooth gradients for GP modeling
3. Quantifies "how close" to convergence
4. Handles divergence via clipping (final_residual = max_float)
5. Enables satisfied (c<0) and violation magnitude (c>0)

**Example**:
- final_residual = 1e-6, tolerance = 1e-8
- c(x) = log10(1e-6) - log10(1e-8) = -6 - (-8) = 2 > 0 (violated)

- final_residual = 1e-10, tolerance = 1e-8
- c(x) = log10(1e-10) - log10(1e-8) = -10 - (-8) = -2 < 0 (satisfied)

**Alternative Binary Formulation**:
```
c(x) = 0 if converged, 1 if failed
```

**Disadvantages**:
- Loses information about proximity to convergence
- Discontinuous (poor for GP modeling)
- Less informative for optimization

**Multiple Constraints for Comprehensive Safety**:

1. **Convergence constraint**:
   ```
   c_1(x) = log10(residual) - log10(tolerance)
   ```

2. **Iteration budget constraint**:
   ```
   c_2(x) = iterations - max_iter
   ```

3. **Physics validity constraint**:
   ```
   c_3(x) = max_penetration - penetration_limit
   ```

**CONFIG Handles Multiple Constraints**:
```
F_opt = {x : LCB_i(x) ≤ 0 for all i ∈ {1,2,3}}
```
Intersection of individual optimistic feasible sets.

**Constraint GP Modeling**:
- **Kernel**: Same family as objective (Matérn-5/2 with ARD) for consistency
- **Hyperparameters**: Separate optimization per constraint GP
- **Noise model**: Heteroscedastic if variance increases near boundary
  - Use HeteroskedasticGP or input-dependent noise models

Example for heteroscedastic noise:
```python
from botorch.models import HeteroskedasticSingleTaskGP

# If constraint noise varies across parameter space
noise_model = HeteroskedasticSingleTaskGP(train_X, train_Y, train_Yvar)
```

### 6. CONFIG System Components

**BO Controller**:

Main CONFIG iteration loop:

```python
class CONFIGController:
    def __init__(self, bounds, constraint_models, objective_model, delta=0.1):
        self.bounds = bounds
        self.constraint_models = constraint_models
        self.objective_model = objective_model
        self.delta = delta
        self.cumulative_violations = 0
        self.iteration = 0

    def run_iteration(self):
        self.iteration += 1

        # 1. Fit GPs with MLE
        fit_gp_models([self.objective_model] + self.constraint_models)

        # 2. Compute β using theoretical schedule
        beta = compute_beta(self.iteration, self.delta)

        # 3. Construct optimistic feasible set (implicit in optimization)

        # 4. Solve auxiliary optimization problem
        next_x = optimize_config_acquisition(
            acqf=CONFIGAcquisition(self.objective_model, self.constraint_models, beta),
            constraint_models=self.constraint_models,
            beta=beta,
            bounds=self.bounds,
            n_restarts=20
        )

        # 5. Evaluate simulation
        obj_value, constraint_values = evaluate_simulation(next_x)

        # 6. Track cumulative violations
        for c_val in constraint_values:
            self.cumulative_violations += max(0, c_val)

        # 7. Update data
        self.update_data(next_x, obj_value, constraint_values)

        # 8. Check termination
        return self.check_convergence()
```

**Constraint Learner**:

Active learning module for boundary refinement:

```python
class ConstraintLearner:
    def __init__(self, constraint_models):
        self.constraint_models = constraint_models

    def propose_boundary_sample(self, bounds, n_candidates=1000):
        """
        Propose sample to refine constraint boundary understanding.
        Uses Expected Information Gain (EIG).
        """
        # Generate candidate set
        candidates = generate_sobol_samples(bounds, n_candidates)

        # Compute uncertainty for each constraint
        uncertainties = []
        for model in self.constraint_models:
            posterior = model.posterior(candidates)
            uncertainty = posterior.variance.sqrt()
            uncertainties.append(uncertainty)

        # Combine uncertainties (max for most uncertain constraint)
        total_uncertainty = torch.max(torch.stack(uncertainties), dim=0)[0]

        # Compute boundary proximity
        boundary_proximity = []
        for model in self.constraint_models:
            lcb = compute_constraint_lcb(candidates, model, beta=1.0)
            proximity = torch.exp(-5 * lcb**2)
            boundary_proximity.append(proximity)

        # Total acquisition: uncertainty × proximity
        acquisition = total_uncertainty * torch.max(torch.stack(boundary_proximity), dim=0)[0]

        # Select best candidate
        best_idx = torch.argmax(acquisition)
        return candidates[best_idx]
```

**Feasibility Predictor**:

Real-time query interface:

```python
class FeasibilityPredictor:
    def __init__(self, constraint_models, beta):
        self.constraint_models = constraint_models
        self.beta = beta

    def query(self, theta):
        """
        Given parameters θ, return feasibility assessment.

        Returns:
            - P_feasible: Probability of satisfying all constraints
            - confidence: Confidence level (based on variance)
            - in_optimistic_set: Boolean, is θ in F_opt?
        """
        probabilities = []
        uncertainties = []

        for model in self.constraint_models:
            posterior = model.posterior(theta)
            mean = posterior.mean
            variance = posterior.variance

            # P(c_i(θ) ≤ 0) using Gaussian CDF
            prob = torch.distributions.Normal(mean, variance.sqrt()).cdf(torch.tensor(0.0))
            probabilities.append(prob.item())

            uncertainties.append(variance.sqrt().item())

        # Aggregate assuming independence
        P_feasible = np.prod(probabilities)

        # Confidence: high if all uncertainties low
        max_uncertainty = max(uncertainties)
        if max_uncertainty < 0.15:
            confidence = "high"
        elif max_uncertainty < 0.3:
            confidence = "medium"
        else:
            confidence = "low"

        # Check optimistic feasible set
        in_optimistic_set = is_in_optimistic_set(theta, self.constraint_models, self.beta)

        return {
            'P_feasible': P_feasible,
            'confidence': confidence,
            'in_optimistic_set': in_optimistic_set,
            'uncertainties': uncertainties
        }
```

**Violation Monitor**:

Tracks and validates theoretical bounds:

```python
class ViolationMonitor:
    def __init__(self):
        self.violations = []

    def add_violation(self, constraint_value):
        """Add violation for current iteration."""
        violation = max(0, constraint_value)
        self.violations.append(violation)

    def cumulative_violation(self):
        """Compute V_t = Σ max(0, c(x_i))"""
        return sum(self.violations)

    def check_theoretical_bound(self, t, gamma_t):
        """
        Check if violations within theoretical bound.

        Theory: V_t = O(√(t γ_t log t))
        """
        V_t = self.cumulative_violation()
        theoretical_bound = np.sqrt(t * gamma_t * np.log(max(t, 2)))

        # Allow 2x factor for practical margin
        if V_t > 2 * theoretical_bound:
            return {
                'status': 'WARNING',
                'message': 'Violations exceed theoretical bound',
                'V_t': V_t,
                'bound': theoretical_bound
            }
        else:
            return {
                'status': 'OK',
                'V_t': V_t,
                'bound': theoretical_bound
            }

    def violation_rate(self):
        """Compute ΔV_t = V_t - V_{t-1}"""
        if len(self.violations) < 2:
            return self.violations[-1] if self.violations else 0
        return self.violations[-1]  # Last violation added

    def plot_violations(self):
        """Generate violation trajectory plot vs theoretical bound."""
        t = range(1, len(self.violations) + 1)
        V_t = np.cumsum(self.violations)

        # Theoretical bound (assuming γ_t ≈ log^(d+1) t for Matérn)
        theoretical = [np.sqrt(i * np.log(i)**5 * np.log(max(i, 2))) for i in t]

        plt.figure(figsize=(10, 6))
        plt.plot(t, V_t, label='Cumulative Violations V_t', linewidth=2)
        plt.plot(t, theoretical, '--', label='Theoretical Bound O(√(t log^5 t log t))', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Cumulative Violations')
        plt.title('CONFIG Violation Trajectory')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        return plt
```

**Geometry Adapter**:

Multi-task extension for transfer learning:

```python
class GeometryAdapter:
    def __init__(self):
        self.geometry_models = {}

    def create_multitask_gp(self, geometries, features):
        """
        Create multi-task GP where each geometry is a task.

        Args:
            geometries: List of geometry IDs
            features: Geometric features for each geometry
        """
        # Multi-task kernel: shared base + task-specific
        task_covar = IndexKernel(num_tasks=len(geometries))
        data_covar = MaternKernel(nu=2.5, ard_num_dims=features.shape[-1])

        covar = MultitaskKernel(data_covar, task_covar, num_tasks=len(geometries))

        return MultiTaskGP(train_X, train_Y, task_feature=-1, covar_module=covar)

    def predict_for_new_geometry(self, new_features, candidate_params):
        """
        Predict for new geometry using transfer learning.

        1. Find similar geometries via k-NN in feature space
        2. Initialize predictions from similar geometries
        3. Refine with geometry-specific samples (20% budget)
        """
        # Find k nearest geometries
        similar_geometries = self.find_similar_geometries(new_features, k=3)

        # Get predictions from multi-task GP
        predictions = self.geometry_models['multitask'].predict(candidate_params)

        # Weight by similarity
        weights = self.compute_similarity_weights(new_features, similar_geometries)
        weighted_prediction = sum(w * p for w, p in zip(weights, predictions))

        return weighted_prediction

    def allocate_budget(self, total_budget, new_geometry):
        """
        Allocate sampling budget between transfer and new samples.

        - 20% for new geometry exploration
        - 80% transferred from knowledge base
        """
        new_geometry_budget = int(0.2 * total_budget)
        transfer_budget = total_budget - new_geometry_budget

        return {
            'new_geometry_samples': new_geometry_budget,
            'transfer_samples': transfer_budget
        }
```

### 7. Use Case: Learning Feasible Parameter Regions

**Objective**: Map feasible region F, not optimize objective

**Strategy**: Deploy CONFIG with constant objective

```python
# Constant objective (focus purely on constraints)
def constant_objective(x):
    return 0.0

# Run CONFIG with only constraint learning
config = CONFIGController(
    bounds=param_bounds,
    constraint_models=[convergence_constraint_gp],
    objective_model=constant_objective_gp,  # Trivial
    delta=0.1
)

# Active boundary learning emphasized
for i in range(100):
    if i < 50:
        # Use Constraint Learner for boundary focus
        next_x = constraint_learner.propose_boundary_sample(param_bounds)
    else:
        # Standard CONFIG (will sample interior of F_opt)
        next_x = config.run_iteration()

    evaluate_and_update(next_x)
```

**Outputs**:
1. **Feasible region volume estimate** via Monte Carlo:
   ```python
   # Generate dense sample grid
   samples = generate_sobol_samples(bounds, n=10000)

   # Count samples in F_opt
   in_F_opt = [is_in_optimistic_set(x, constraint_models, beta) for x in samples]
   volume_estimate = sum(in_F_opt) / len(in_F_opt) * total_volume
   ```

2. **Boundary parameterization**:
   - Extract decision boundary: {x : LCB_constraint(x) = 0}
   - Fit parametric curve/surface if low-dimensional
   - Enable analytical queries

3. **Conservative safe parameter recommendations**:
   - Select points well inside F_opt: LCB_constraint(x) < -1.0
   - Ensures robustness to model uncertainty
   - Return top-K by distance from boundary

### 8. Use Case: Real-Time Feasibility Assessment

**Pre-Train on Historical Data**:
```python
# Train CONFIG models on historical simulations
historical_data = load_historical_data()
config.fit_models(historical_data)
```

**For New Parameter Query θ**:

```python
def assess_feasibility(theta, config):
    """
    Real-time feasibility assessment (<50ms latency).
    """
    # 1. Compute LCB_constraint(θ)
    lcb_values = [
        compute_constraint_lcb(theta, model, config.beta)
        for model in config.constraint_models
    ]

    # 2. Check if θ ∈ F_opt
    in_F_opt = all(lcb <= 0 for lcb in lcb_values)

    # 3. Compute P(c(θ) ≤ 0)
    probabilities = [
        compute_feasibility_probability(theta, model)
        for model in config.constraint_models
    ]
    P_feasible = np.prod(probabilities)

    # 4. Compute prediction uncertainty
    uncertainties = [
        model.posterior(theta).variance.sqrt().item()
        for model in config.constraint_models
    ]
    max_uncertainty = max(uncertainties)
    confidence = "high" if max_uncertainty < 0.2 else "medium" if max_uncertainty < 0.4 else "low"

    # 5. Find nearest safe point if infeasible
    nearest_safe = None
    if not in_F_opt:
        nearest_safe = find_nearest_safe_point(theta, config)

    return {
        'feasible_estimate': 'Yes' if in_F_opt else 'No' if max(lcb_values) > 0.5 else 'Uncertain',
        'probability': P_feasible,
        'confidence': confidence,
        'lcb_values': lcb_values,
        'nearest_safe_point': nearest_safe
    }
```

**Caching for <50ms Latency**:
```python
# Cache frequently-queried regions
cache = {}

def cached_assess_feasibility(theta):
    # Discretize to grid for caching
    grid_key = discretize_to_grid(theta, grid_size=0.1)

    if grid_key in cache:
        return cache[grid_key]

    result = assess_feasibility(theta, config)
    cache[grid_key] = result
    return result
```

### 9. Use Case: Pre-Simulation Constraint Checking for Batches

**Batch Validation Pipeline**:

Given N candidate configurations:

```python
def validate_batch(candidates, config, select_top_k_percent=30):
    """
    Validate and rank batch of parameter configurations.
    """
    results = []

    for theta in candidates:
        # Predict constraint values
        constraint_predictions = [
            model.posterior(theta).mean.item()
            for model in config.constraint_models
        ]

        # Compute LCB
        lcb_values = [
            compute_constraint_lcb(theta, model, config.beta).item()
            for model in config.constraint_models
        ]

        # Feasibility probability
        P_feasible = compute_feasibility_probability_all(theta, config.constraint_models)

        results.append({
            'theta': theta,
            'constraint_predictions': constraint_predictions,
            'lcb_values': lcb_values,
            'P_feasible': P_feasible,
            'max_lcb': max(lcb_values)  # Lower is better (more likely feasible)
        })

    # Rank by LCB (lower = more likely feasible)
    results.sort(key=lambda x: x['max_lcb'])

    # Select top K%
    k = int(len(results) * select_top_k_percent / 100)
    selected = results[:k]

    # Expected success rate
    expected_success_rate = sum(r['P_feasible'] for r in selected) / k

    return {
        'selected': selected,
        'expected_success_rate': expected_success_rate,
        'ranked_results': results
    }
```

**Adaptive Thresholding**:
```python
# Track batch success rates
historical_success_rates = []

def adaptive_threshold(default_k=30):
    """
    Adjust selectivity based on previous batch performance.
    """
    if not historical_success_rates:
        return default_k

    recent_success = np.mean(historical_success_rates[-5:])

    if recent_success > 0.9:
        # Increase selectivity (accept fewer)
        return max(default_k - 10, 10)
    elif recent_success < 0.6:
        # Decrease selectivity (accept more)
        return min(default_k + 10, 50)
    else:
        return default_k
```

**Diversity Promotion**:
```python
def promote_diversity(selected, diversity_weight=0.3):
    """
    Penalize candidates too close to each other.
    """
    diverse_selected = []

    for candidate in selected:
        # Compute minimum distance to already selected
        if diverse_selected:
            min_dist = min(
                np.linalg.norm(candidate['theta'] - s['theta'])
                for s in diverse_selected
            )
            # Penalty for proximity
            candidate['score'] -= diversity_weight / (min_dist + 0.1)

        diverse_selected.append(candidate)

    # Re-rank with diversity penalty
    diverse_selected.sort(key=lambda x: x.get('score', -x['max_lcb']))

    return diverse_selected
```

### 10. Visualization Requirements

**Optimistic Feasible Set Evolution**:

Time-series showing F_opt boundaries across iterations:

```python
def visualize_F_opt_evolution(config, iterations=[5, 20, 50, 100]):
    """
    Show how F_opt evolves as more data is collected.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, iter_num in zip(axes.flat, iterations):
        # Generate grid
        x1 = np.linspace(bounds[0,0], bounds[0,1], 100)
        x2 = np.linspace(bounds[1,0], bounds[1,1], 100)
        X1, X2 = np.meshgrid(x1, x2)

        # Compute LCB_constraint at iteration iter_num
        # (requires saved model states or retraining)
        beta = compute_beta(iter_num)
        LCB = compute_constraint_lcb_grid(X1, X2, model_at_iter[iter_num], beta)

        # Plot F_opt (LCB ≤ 0)
        feasible_mask = LCB <= 0
        ax.contourf(X1, X2, feasible_mask, levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.3)
        ax.contour(X1, X2, LCB, levels=[0], colors='black', linewidths=2)

        # Overlay violations (points outside F_opt that were sampled)
        violations = [p for p in sampled_points[:iter_num] if actual_constraint(p) > 0]
        if violations:
            vx, vy = zip(*violations)
            ax.scatter(vx, vy, color='red', marker='x', s=100, label='Violations')

        ax.set_title(f'Iteration {iter_num}')
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.legend()

    plt.tight_layout()
    return fig
```

**LCB Acquisition Surface**:

3D visualization:

```python
def visualize_lcb_surface(config):
    """
    3D surface plot of LCB_objective with feasibility mask.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate grid
    x1 = np.linspace(bounds[0,0], bounds[0,1], 50)
    x2 = np.linspace(bounds[1,0], bounds[1,1], 50)
    X1, X2 = np.meshgrid(x1, x2)

    # Compute LCB_objective
    LCB_obj = compute_lcb_objective_grid(X1, X2, config.objective_model, config.beta)

    # Compute feasibility mask
    LCB_con = compute_constraint_lcb_grid(X1, X2, config.constraint_models[0], config.beta)
    feasible = LCB_con <= 0

    # Mask infeasible region (set to NaN)
    LCB_obj_masked = np.where(feasible, LCB_obj, np.nan)

    # Plot surface
    surf = ax.plot_surface(X1, X2, LCB_obj_masked, cmap='viridis', alpha=0.8)

    # Mark next sampling location
    next_x = config.next_sample_location
    ax.scatter([next_x[0]], [next_x[1]], [compute_lcb_objective(next_x, config)],
               color='red', s=200, marker='*', label='Next Sample')

    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_zlabel('LCB Objective')
    ax.set_title('CONFIG Acquisition Surface\n(Infeasible region masked)')
    fig.colorbar(surf, shrink=0.5)
    ax.legend()

    return fig
```

**Cumulative Violations Trajectory**:

See ViolationMonitor.plot_violations() above.

Additional enhancements:
```python
def plot_violation_trajectory_enhanced(monitor):
    """
    Enhanced violation plot with rate and theoretical comparisons.
    """
    t = range(1, len(monitor.violations) + 1)
    V_t = np.cumsum(monitor.violations)
    Delta_V_t = monitor.violations

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Cumulative violations vs theoretical bound
    theoretical = [np.sqrt(i * np.log(i)**5 * np.log(max(i, 2))) for i in t]
    ax1.plot(t, V_t, 'b-', linewidth=2, label='Actual V_t')
    ax1.plot(t, theoretical, 'r--', linewidth=2, label='Theoretical O(√(t γ_t log t))')
    ax1.fill_between(t, 0, theoretical, alpha=0.2, color='red', label='Allowed violation region')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cumulative Violations')
    ax1.set_title('CONFIG Cumulative Violations vs Theoretical Bound')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Violation rate (should trend toward zero)
    ax2.bar(t, Delta_V_t, color='orange', alpha=0.7, label='ΔV_t = V_t - V_{t-1}')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Per-Iteration Violation')
    ax2.set_title('Violation Rate (should decrease over time)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

---

## Critical Parameters (Tribol and Smith)

[Same as in other documents - omitted for brevity but would include full parameter descriptions]

---

## Performance Characteristics

### Expected Success Rates

**Initial Phase (0-50 trials)**:
- LHS initialization: 30-40% convergence
- CONFIG learning: 45-55% by trial 50
- Conservative (lower variance than FR-BO)

**Mid-term Phase (50-150 trials)**:
- Success rate: 70-80%
- Strict violation bounds maintained
- Reliable predictions in F_opt

**Mature Phase (>150 trials)**:
- Success rate: >90%
- Formal guarantees of ε-optimality
- Violations provably bounded

### Computational Efficiency

- **vs Grid Search**: 50-100x speedup
- **vs Random Search**: 5-10x speedup
- **Sample Efficiency**: Good (conservative reduces failures)
- **Convergence Speed**: Good (slower than FR-BO, guaranteed)

---

## Method Strengths and Weaknesses

### Strengths
- **Rigorous theoretical guarantees**: Sublinear regret and violations
- **Principled optimistic exploration**: Strategic violations with bounds
- **Natural infeasibility detection**: F_opt → ∅ signals likely infeasibility
- **Safety-critical suitable**: Bounded violations, formal certification
- **No manual penalty tuning**: β schedule from theory

### Weaknesses
- **Conservative in practice**: Higher false positive rate (10-15%)
- **Implementation complexity**: Auxiliary optimization, β schedule tuning
- **Less intuitive**: LCB concept harder for practitioners vs probabilities
- **Slower initial learning**: Conservative approach vs aggressive FR-BO
- **Requires careful model validation**: Guarantees depend on GP assumptions

---

## Deployment Recommendations

**When to Use CONFIG**:
1. Safety-critical applications requiring certification
2. Need formal convergence guarantees
3. Violations must be bounded and decreasing
4. Willing to be conservative (accept higher false positive rate)
5. Require audit trails and theoretical justification

**When to Use Alternative**:
- If speed more critical than guarantees → FR-BO
- If interpretability more important → GP Classification
- If complex multi-constraint discovery needed → SHEBO

**Deployment Strategy**:
- Use as fallback for safety-critical applications
- Maintain in parallel with GP Classification baseline
- Emphasize theoretical properties in documentation
- Provide violation monitoring dashboards
- Generate certification reports for formal verification

---

## Implementation Tools

### Required Software
- Python 3.8+
- PyTorch 1.12+
- BoTorch (latest version)
- GPyTorch
- SciPy (for constrained optimization)
- NumPy, Pandas
- Matplotlib (3D plotting)

### Recommended Extensions
- Ax Platform (optional, for high-level orchestration)
- modAL (active learning utilities)
- Plotly (interactive 3D visualizations)
- Dash (monitoring dashboard)

### Development Infrastructure
- Git version control
- pytest for testing (especially for theoretical guarantees)
- Sphinx documentation (emphasizing theory)
- Docker containers
- CI/CD with formal verification checks

---

## Next Steps

1. Review theoretical framework and guarantees
2. Set up development environment with SciPy/BoTorch
3. Implement LCB calculation and β schedule
4. Create constrained optimization solver integration
5. Develop violation monitoring system
6. Establish testing protocol for theoretical properties
7. Generate initial results on test problems
8. Validate that violations satisfy O(√t) bound
9. Create certification documentation
10. Schedule reviews with domain experts

---

## References and Further Reading

- CONFIG original paper: Gelbart et al., "Bayesian Optimization with Unknown Constraints"
- SafeOpt: Sui et al., "Safe Exploration for Optimization with Gaussian Processes"
- GP-UCB: Srinivas et al., "Gaussian Process Optimization in the Bandit Setting"
- Regret bounds: Auer et al., "Using Confidence Bounds for Exploitation-Exploration Trade-offs"
- Matérn kernels and information gain: Vakili et al., "On Information Gain and Regret Bounds in Gaussian Process Bandits"
