# Failure-Robust Bayesian Optimization (FR-BO) Implementation Plan

## Overview

FR-BO treats simulation failures as informative constraints rather than nuisances, jointly modeling convergence feasibility and performance objectives through dual Gaussian processes with failure-aware acquisition functions. This approach learns failure boundaries by maintaining a regression GP for the objective function J(θ) alongside a classification GP for failure probability P_fail(θ).

**Key Innovation**: FREI(θ) = EI(θ) × (1 - P_fail(θ)) naturally balances optimization potential with feasibility likelihood.

**Expected Performance**: 3-8x convergence speedup versus standard BO in simulation-heavy applications.

---

## High-Level Implementation Checklist

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up Ax/BoTorch development environment
- [ ] Define parameter space for Tribol and Smith parameters
- [ ] Implement parameter encoding (log-scale, one-hot, linear normalization)
- [ ] Define objective function J(θ) with convergence priority weighting
- [ ] Create basic FE simulation executor wrapper for Tribol+Smith

### Phase 2: Dual GP System (Weeks 3-4)
- [ ] Implement SingleTaskGP for objective regression (trained on successful trials)
- [ ] Implement variational GP classifier (GP_vi) for failure probability
- [ ] Configure Matérn-5/2 kernel with ARD for both GPs
- [ ] Implement dual GP training pipeline with hyperparameter optimization
- [ ] Create trial status tracking system (COMPLETED, FAILED, ABANDONED, RUNNING)

### Phase 3: FR-BO Acquisition Function (Week 5)
- [ ] Implement custom FailureRobustEI acquisition class
- [ ] Inherit from SampleReducingMCAcquisitionFunction
- [ ] Implement multiplicative combination of EI and feasibility probability
- [ ] Add floor padding strategy for failed evaluations
- [ ] Configure acquisition optimization (10 random restarts + L-BFGS-B)

### Phase 4: Three-Phase Workflow (Week 6)
- [ ] Implement Phase 1: Sobol quasi-random initialization (Trials 1-20)
- [ ] Implement Phase 2: FR-BO iterations (Trials 21-200)
- [ ] Implement Phase 3: Post-optimization validation and sensitivity analysis
- [ ] Configure automatic Sobol→FR-BO transition via GenerationStrategy
- [ ] Implement GP retraining schedule (every 50 trials)

### Phase 5: Early Termination System (Week 7)
- [ ] Implement convergence trajectory monitoring (every 5 iterations after iter 10)
- [ ] Create GP trajectory model using Matérn-3/2 kernel
- [ ] Implement trajectory extrapolation to max_iterations
- [ ] Add early termination logic (P(convergence) < 0.2 with confidence > 0.8)
- [ ] Integrate termination feedback into FR-BO data loop

### Phase 6: Use Case Features (Week 8)
- [ ] Implement multi-task GP for uncertainty quantification across geometries
- [ ] Create geometric feature extraction module
- [ ] Implement UCB-style parameter recommendation for new geometries
- [ ] Build pre-simulation risk scoring system
- [ ] Create safe alternative suggestion via constrained optimization

### Phase 7: Visualization and Reporting (Week 9)
- [ ] Create iteration diagram flowchart
- [ ] Build system architecture diagram (layered)
- [ ] Implement parameter space visualization (2D projection via PCA/t-SNE)
- [ ] Add real-time dashboard for monitoring optimization progress
- [ ] Generate convergence metrics and performance reports

### Phase 8: Testing and Validation (Week 10)
- [ ] Unit tests for parameter encoding/decoding
- [ ] Integration tests for dual GP training
- [ ] Validation on test geometries
- [ ] Performance benchmarking vs standard BO
- [ ] Documentation and user guide

---

## Detailed Implementation Specifications

### 1. Theoretical Foundation

**Problem Formulation**: minimize J(θ) subject to θ ∉ ΘF where ΘF represents the learned failure region.

**Three Failure Handling Strategies**:
1. **Floor padding**: Replace failed evaluations with minimum observed successful value (adaptive, promotes fast convergence)
2. **Binary classification**: Use separate variational GP classifier (conservative, reduces failure rate)
3. **Combined approaches**: Apply both (most robust but computationally expensive)

**Mathematical Approach**: Unlike hard constraints, FR-BO allows occasional strategic violations during learning when uncertainty is high, accelerating boundary discovery.

**Convergence**: Empirical studies demonstrate 3-8x convergence speedup versus standard BO.

### 2. Ax/BoTorch Implementation Architecture

**BoTorch Components**:
- Low-level PyTorch-based GP modeling with automatic differentiation
- Automatic differentiation for acquisition optimization
- SingleTaskGP for objective regression
- Variational GP classifier (GP_vi) with BernoulliLikelihood

**Ax Components**:
- High-level experiment orchestration
- Trial status tracking (COMPLETED, FAILED, ABANDONED, RUNNING)
- Parameter transformations and normalization
- GenerationStrategy for automatic phase transitions

**Dual GP Configuration**:
```python
# Objective GP (trained only on successful trials)
model_obj = SingleTaskGP(train_X=X_success, train_Y=y_obj)

# Failure classifier GP (trained on all trials)
model_fail = GP_vi(X_all, y_failed)  # Binary labels
```

**Extension Mechanisms**:
- Custom models via construct_inputs classmethods
- Custom acquisitions via @acqf_input_constructor decorators
- Custom transforms for missing data handling (floor padding patterns)
- Inducing points for scalability (reducing O(n³) to O(nm²) for m inducing points)
- Pending observations to prevent redundant evaluation of running trials

### 3. Parameter Space and Objective Design

**Parameter Encoding Schemes**:

1. **Log-scale transforms** (for high-dynamic-range parameters):
   - Penalty stiffness: [10³, 10⁸]
   - Tolerances: [10⁻⁸, 10⁻⁴]

2. **One-hot encoding** (for categorical choices):
   - Enforcement method: {mortar, penalty, augmented_lagrange}
   - Solver type: {Newton, NewtonLineSearch, TrustRegion, L-BFGS}

3. **Linear normalization** (for bounded continuous parameters):
   - Iterations: [10, 1000]
   - Search expansion: [1.1, 1.5]

**Objective Function** (prioritizing convergence, then efficiency, then speed):
```
J(θ) = 10.0 × (1-converged) + 1.0 × (iters/max_iters) + 0.5 × (time/timeout)
```

This formulation:
- Treats non-convergence as catastrophic (dominates objective)
- Enables optimization of computational efficiency among converging configurations
- Severe numerical instabilities receive 2× floor padding (double penalty)
- Early successful convergence receives 0.9× actual value (reward for efficiency)

**ARD (Automatic Relevance Determination) Kernels**:
- Automatically detect parameter importance through length scale learning
- Parameters with longer length scales contribute less to predictions (candidates for removal)
- Matérn-5/2 kernel provides twice-differentiable smoothness
- Appropriate for iterative solver convergence surfaces
- Avoids overfitting compared to RBF kernels

### 4. Three-Phase FR-BO Workflow

**Phase 1 (Trials 1-20): Sobol Initialization**
- Quasi-random initialization explores parameter space uniformly
- Train initial dual GPs on observed successes and failures
- Establish baseline performance
- Identify obviously problematic regions
- Space-filling phase ensures diverse training data

**Phase 2 (Trials 21-200): FR-BO Iterations**

Each iteration:
1. Update dual GPs with new data
2. Compute FREI over candidate set using Monte Carlo acquisition
3. Select next θ maximizing FREI
4. Execute FE simulation with early termination monitoring
5. Record results and binary convergence label

Optimization details:
- Use 10 random restarts plus L-BFGS-B for acquisition optimization
- Retrain GPs every 50 trials for hyperparameter adaptation
- Monitor convergence indicators during simulation

**Phase 3 (Post-optimization): Validation and Deployment**
- Validate best configuration on held-out test geometries
- Perform Sobol sensitivity analysis to identify critical parameters
- Generate feasibility probability maps for visualization
- Deploy optimal parameters with uncertainty estimates
- Provide predicted success probability

**Component Interactions**:
- AxClient orchestration layer manages workflow
- GenerationStrategy handles Sobol→FR-BO transition automatically
- Dual GPs predict objective mean/variance and failure probability independently
- FREI acquisition optimizes product of EI and (1-P_fail) via multi-start gradient ascent
- FE executor wraps Tribol+Smith simulation, monitors convergence, triggers early termination

### 5. Use Case: Uncertainty Quantification for New Geometries

**Multi-Task GP Extension**:
- Each geometry represents one task
- Extract geometric features as task descriptors:
  - Contact area ratio
  - Mesh density
  - Material property contrast
  - Gap distribution statistics
- MTGP shares statistical strength across similar geometries
- Maintains geometry-specific predictions

**Workflow for New Geometry**:
1. Extract geometric features from new geometry
2. Query MTGP across parameter grid
3. Compute success probability and uncertainty
4. Recommend parameters via UCB-style scoring:
   ```
   score = P(success) - β×uncertainty  (where β=2 for 95% confidence)
   ```
5. Return top-3 configurations with:
   - Predicted success rates
   - Confidence intervals
   - Expected performance metrics

### 6. Use Case: Real-Time Monitoring for Early Termination

**Trajectory-Based Prediction**:

At iteration checkpoints (every 5 iterations after iteration 10):
1. Fit GP to partial residual trajectory: log(residual) vs iteration number
2. Extrapolate to max_iterations using posterior predictive distribution
3. Compute P(convergence) with confidence estimate
4. If P(convergence) < 0.2 with confidence > 0.8, terminate early

**GP Trajectory Model**:
- Uses Matérn-3/2 kernel for residual evolution
- Captures typical exponential decay patterns
- Provides uncertainty quantification for predictions

**Benefits**:
- Saves 30-50% compute on failing configurations
- Updates FR-BO with failure data
- Improves future predictions
- Enables resource reallocation to promising trials

**Feature Extraction**:
- Current iteration number
- Log residual norm
- Residual derivative (convergence rate)
- Contact active set size
- Penetration magnitude
- Stagnation detection (iterations without progress)

### 7. Use Case: Pre-Simulation Validation

**Risk Scoring System** (combines three factors):
```
risk_score = 0.5 × P_fail(θ) +
             0.3 × (1 - distance_to_nearest_failure) +
             0.2 × (1 - local_success_rate)
```

**Risk Thresholds**:
- **> 0.7**: "HIGH RISK - Do not run"
  - Provide suggested safe alternative
  - Project to nearest high-probability region
- **0.4-0.7**: "MODERATE RISK - Proceed with caution"
  - Display confidence intervals
  - Suggest parameter adjustments
- **< 0.4**: "LOW RISK - Proceed"
  - Estimated success probability
  - Expected performance metrics

**Safe Alternative Generation**:
Constrained optimization:
```
minimize ||θ_new - θ_proposed||²
subject to P_fail(θ_new) < 0.2
```

This finds the nearest parameter set with acceptable failure probability.

### 8. Visualization Requirements

**Iteration Diagram (Flowchart)**:
```
Current data D_j
    ↓
Update dual GPs
    ↓
Predict failure regions (2D heatmap projection)
    ↓
Compute FREI acquisition surface
    ↓
Optimize FREI (show multiple restarts converging)
    ↓
Execute simulation
    ├─→ Success → Update D_{j+1}
    └─→ Failure → Update D_{j+1}
    ↓
Loop (check convergence criteria and budget)
```

**System Architecture Diagram (Layered)**:

*Top Layer - User Interface*:
- Parameter input forms
- Real-time dashboards
- 3D geometry viewer
- Risk assessment display

*Middle Layer - Orchestration*:
- Ax orchestration engine
- Dual GP models (separate boxes):
  - Objective GP
  - Failure VGPC
- FREI acquisition optimizer
- Early stopping logic
- Trial status manager

*Bottom Layer - Execution*:
- FE executor (Tribol+Smith simulation engine)
- Convergence monitoring module
- Results database
- Checkpoint/restart system

*Data Flow*:
Parameter suggestion → Simulation → Result extraction → GP update → Next iteration

**Parameter Space Visualization (2D Projection)**:

Use PCA or t-SNE for dimensionality reduction:
- **Color regions by P_fail** from classifier:
  - Green: P_fail < 0.2 (safe region)
  - Yellow: 0.2 ≤ P_fail ≤ 0.5 (boundary)
  - Red: P_fail > 0.5 (failure region)
- **Overlay trial markers**:
  - Blue dots: successful trials
  - Red X markers: failed trials
  - Gold star: current best
- **FREI acquisition contours**: Show where next samples likely
- **Uncertainty bands**: Transparency indicates prediction confidence

This reveals how FR-BO navigates around failure regions while optimizing.

---

## Critical Tribol and Smith Parameters

### Tribol Contact Parameters

**Penalty stiffness** (k_penalty):
- Most convergence-critical parameter
- Range: 10³ to 10⁸ times material stiffness-to-element-size ratio (E/h)
- Too low → excessive penetration
- Too high → ill-conditioning and Newton-Raphson failures
- Penalty normalization by nodal tributary area improves mesh-independence

**Gap tolerances**:
- Control penetration detection
- Typical values: 1e-6 to 1e-9 in normalized coordinates
- Tighter tolerances improve accuracy but demand more nonlinear iterations

**Mortar integration scheme**:
- Quadrature point density: 3-4 points per triangular cell, 2×2 for quadrilaterals
- Multiplier space interpolation choices: linear/bilinear (stable), quadratic (accurate), piecewise linear
- Search expansion factor: typically 1.1-1.5 for dynamic problems
- Projection tolerance: governs geometric accuracy in non-conforming mesh mapping

### Smith Solver Convergence Parameters

**Solver type**:
- Newton (standard)
- NewtonLineSearch (default, with backtracking) - recommended for contact
- TrustRegion (bounded steps)
- L-BFGS (quasi-Newton)
- Contact problems strongly benefit from line search capabilities

**Tolerances**:
- Absolute tolerance: default 1e-12, tighten to 1e-10 for contact
- Relative tolerance: default 1e-8, tighten to 1e-10 or 1e-12 for contact
- Both must be satisfied for convergence

**Iteration limits**:
- Maximum iterations: default 20, contact problems typically need 50-100
- Accommodates active set changes during primal-dual iterations
- Line search iterations: default 0, recommended 5-10 for contact

**Time stepping**:
- Initial time step: 1e-4 to 1e-3 during contact closure
- Prevents penetration instabilities
- Minimum and maximum time step bounds
- Trust region scaling: default 0.1

**Linear solver**:
- Direct methods (SuperLU): robust but memory-intensive
- Iterative methods (GMRES with preconditioners): scalable but convergence-sensitive
- Choice fundamentally alters solution robustness

### Output Metrics for ML Training

**Contact metrics**:
- Contact pressure distributions (via Lagrange multiplier fields)
- Penetration variables (interpenetration depth)
  - Mortar methods: 1e-9 scale violations (essentially exact)
  - Penalty methods: 1e-5 to 1e-3 depending on stiffness
- Gap function values (signed distances)
- Maximum penetration
- Gap constraint residuals

**Convergence indicators**:
- Nonlinear iteration counts
- Contact constraint residual norms
- Force residual norms
- Convergence rates (Newton method progress)
- Active contact set size and changes per iteration

**ML training data**:
- Residual norms (continuous) → regression
- Binary convergence status (converged/failed) → classification
- Contact force smoothness during sliding
- Patch test results for validation

---

## Performance Characteristics

### Expected Success Rates by Phase

**Initial Phase (0-50 trials)**:
- Random/Sobol: 30-40% convergence rate
- FR-BO learning: 50-60% by trial 50
- Rapid failure boundary identification

**Mid-term Phase (50-150 trials)**:
- FR-BO achieves: 75-85% success rate
- Failure-aware sampling becomes effective
- Parameter recommendations increasingly reliable

**Mature Phase (>150 trials)**:
- Success rate: >90%
- Near-optimal parameters found in 50-70 samples
- Robust predictions for new geometries

### Computational Efficiency Gains

- **vs Grid Search**: 50-100x speedup for 10+ parameters
- **vs Random Search**: 5-10x speedup
- **Early Termination**: Additional 30-40% savings on failures
- **Total Reduction**: 60-80% reduction in simulation costs

---

## Method Strengths and Weaknesses

### Strengths
- Fast convergence via strategic violations
- Handles failures naturally (not discarded, but informative)
- Proven in expensive simulation contexts
- Relatively simple theoretical foundation
- Natural uncertainty quantification
- Effective for initial parameter discovery

### Weaknesses
- Violations may be unacceptable in some contexts (safety-critical)
- Requires tuning floor padding strategy
- Limited formal guarantees for convergence (heuristic vs theoretical)
- May require careful selection of objective function weighting

---

## Integration with FE Workflow

### Pre-Processing
- Geometry analyzer extracts features before mesh generation
- Feature vector queries FR-BO model for initial parameter recommendations
- CAD plugin presents suggestions with confidence indicators
- User choices logged for continuous improvement

### Solver Integration
- Modified Tribol/Smith input deck includes ML-suggested parameters
- Metadata tracking (model version, confidence, alternatives)
- Optional: runtime adaptive parameter adjustment
- Simulation output includes ML provenance

### Post-Processing
- Automated extraction of convergence metrics and parameters
- Structured data pushed to central database
- Triggers model retraining when sufficient new data accumulated
- Updated models versioned and deployed via CI/CD

### Continuous Learning Loop
Weekly automated workflow:
1. Fetch new simulation data from database
2. Retrain dual GPs on expanded dataset
3. Validate on held-out test set
4. If validation accuracy exceeds threshold, promote to production
5. Deploy updated model artifacts
6. Notify users of model updates

---

## Recommended Starting Approach

**Why FR-BO is Suitable for Advanced Optimization**:
1. Aggressive parameter optimization when higher success rates needed
2. Critical simulations where finding optimal parameters justifies more trials
3. Scenarios where strategic failures provide valuable boundary information
4. When global optimality is critical and failures are acceptable

**Deployment Strategy**:
- Use after initial GP Classification baseline (Weeks 5-8)
- Apply to critical simulations requiring optimal parameters
- Compare with GP Classification on test cases
- Measure convergence speed and final solution quality
- Maintain both systems for different use case requirements

---

## Implementation Tools and Technologies

### Required Software
- Python 3.8+
- PyTorch 1.12+
- Ax Platform (Facebook/Meta)
- BoTorch (Bayesian Optimization library)
- GPyTorch (GP implementation)
- NumPy, SciPy, Pandas
- Matplotlib, Plotly for visualization

### Development Environment
- Version control: Git
- Testing: pytest
- Documentation: Sphinx
- CI/CD: GitHub Actions or GitLab CI
- Containerization: Docker for reproducible environments

### Data Management
- Database: PostgreSQL or MongoDB for trial history
- Caching: Redis for frequently-queried predictions
- Model versioning: MLflow or DVC
- Experiment tracking: Weights & Biases or MLflow

---

## Next Steps

1. Review and approve this implementation plan
2. Set up development environment and dependencies
3. Begin Phase 1: Foundation implementation
4. Establish testing framework and CI/CD pipeline
5. Create initial test geometries and baseline simulations
6. Schedule weekly progress reviews and milestone checkpoints
