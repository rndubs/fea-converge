# Machine Learning Systems for Contact Convergence in Finite Element Simulations

**Tribol contact solver and Smith solver enable state-of-the-art mortar-based contact mechanics, but convergence remains challenging. Four distinct machine learning optimization approaches—FR-BO, GP classification, CONFIG, and SHEBO—offer complementary strategies for predicting and resolving convergence failures through intelligent parameter selection.**

The LLNL Tribol library implements Puso-Laursen mortar contact methods with exact constraint enforcement, while Smith/Serac provides implicit nonlinear solvers with sophisticated time integration. Together they create a powerful but parameter-sensitive contact simulation framework. This report presents complete architectural designs for four ML-driven optimization systems, each addressing convergence issues through different theoretical foundations and practical workflows.

## Tribol and Smith parameter landscape

### Critical Tribol contact parameters

Tribol provides **mortar-based contact enforcement** as its primary method, utilizing Lagrange multipliers to represent contact pressures with exact constraint satisfaction. The segment-to-segment formulation eliminates node-to-surface locking while maintaining optimal convergence rates for large deformation problems.

**Penalty stiffness** (k_penalty) represents the most convergence-critical parameter, typically ranging from 10³ to 10⁸ times the material stiffness-to-element-size ratio (E/h). Too low allows excessive penetration; too high causes ill-conditioning and Newton-Raphson failures. **Penalty normalization** by nodal tributary area improves mesh-independence but requires careful calibration. **Gap tolerances** control penetration detection, with typical values from 1e-6 to 1e-9 in normalized coordinates—tighter tolerances improve accuracy but demand more nonlinear iterations.

The mortar integration scheme uses triangular or quadrilateral subdivision cells with **quadrature point density** of 3-4 points per triangular cell or 2×2 for quadrilaterals. **Multiplier space interpolation** choices—linear/bilinear (guaranteed stable), quadratic (higher accuracy), or piecewise linear—directly affect stability. The **search expansion factor** (typically 1.1-1.5) enlarges contact candidate regions for dynamic problems, while **projection tolerance** governs geometric accuracy in mapping between non-conforming meshes.

### Smith solver convergence parameters

Smith/Serac implements multiple nonlinear solver variants: **Newton** (standard), **NewtonLineSearch** (default, with backtracking), **TrustRegion** (bounded steps), and **L-BFGS** (quasi-Newton). Contact problems strongly benefit from line search capabilities to handle the non-smooth optimization landscape introduced by inequality constraints.

**Absolute tolerance** (default 1e-12) and **relative tolerance** (default 1e-8) both must be satisfied for convergence. Contact problems require tightening these to 1e-10 or 1e-12 absolute for acceptable constraint satisfaction. **Maximum iterations** defaults to 20 but contact problems typically need 50-100 to accommodate active set changes during primal-dual iterations. **Line search iterations** (default 0, recommended 5-10) enable backtracking when full Newton steps violate constraints or cause divergence.

**Time step size** parameters—initial, minimum, maximum—critically affect contact initiation. Starting with small steps (1e-4 to 1e-3) during contact closure prevents penetration instabilities. The **trust region scaling** (default 0.1) limits step magnification in trust region methods. Linear solver selection between direct methods (SuperLU, robust but memory-intensive) and iterative methods (GMRES with preconditioners, scalable but convergence-sensitive) fundamentally alters solution robustness for the contact-augmented system.

### Output metrics for ML training

Tribol exposes **contact pressure distributions** via Lagrange multiplier fields, enabling visualization of constraint enforcement quality. **Penetration variables** show interpenetration depth with mortar methods achieving 1e-9 scale violations (essentially exact) versus penalty methods' 1e-5 to 1e-3 depending on stiffness. **Gap function values** provide signed distances between surfaces, while **maximum penetration** and **gap constraint residuals** quantify constraint violation magnitudes.

Convergence indicators include **nonlinear iteration counts**, **contact constraint residual norms**, **force residual norms**, and **convergence rates** tracking Newton method progress. **Active contact set** size and changes per iteration reveal constraint activation/deactivation patterns. For ML training, combining residual norms (continuous) with binary convergence status (converged/failed) enables both regression and classification approaches. Contact force smoothness during sliding and patch test results provide validation metrics for algorithm correctness.

## System 1: Failure-Robust Bayesian Optimization

FR-BO treats simulation failures as informative constraints rather than nuisances, jointly modeling convergence feasibility and performance objectives through dual Gaussian processes with failure-aware acquisition functions.

### FR-BO theoretical foundation

Standard Bayesian Optimization ignores or arbitrarily penalizes failed trials, discarding valuable information about failure regions. **FR-BO learns failure boundaries** by maintaining a regression GP for the objective function J(θ) alongside a classification GP for failure probability P_fail(θ). The failure-robust expected improvement acquisition function combines these: **FREI(θ) = EI(θ) × (1 - P_fail(θ))**, naturally balancing optimization potential with feasibility likelihood.

Three failure handling strategies exist: **Floor padding** replaces failed evaluations with the minimum observed successful value (adaptive, promotes fast convergence), **binary classification** uses a separate variational GP classifier (conservative, reduces failure rate), and **combined approaches** apply both (most robust but computationally expensive). Empirical studies demonstrate 3-8x convergence speedup versus standard BO in simulation-heavy applications.

The mathematical problem formulation becomes: minimize J(θ) subject to θ ∉ ΘF where ΘF represents the learned failure region. Unlike hard constraints, FR-BO allows occasional strategic violations during learning when uncertainty is high, accelerating boundary discovery.

### Ax/BoTorch implementation architecture

BoTorch provides low-level PyTorch-based GP modeling with automatic differentiation for acquisition optimization, while Ax manages high-level experiment orchestration, trial status tracking, and parameter transformations. Trial statuses (COMPLETED, FAILED, ABANDONED, RUNNING) enable sophisticated failure handling workflows.

The implementation pattern uses custom generation strategies with **dual GP configuration**: one SingleTaskGP for objective regression trained only on successful trials, plus a variational GP classifier (GP_vi) trained on all trials with binary convergence labels. The custom FailureRobustEI acquisition class inherits from SampleReducingMCAcquisitionFunction and implements multiplicative combination of expected improvement and feasibility probability.

**Extension mechanisms** include: custom models via construct_inputs classmethods, custom acquisitions via @acqf_input_constructor decorators, custom transforms for missing data handling (floor padding patterns), and inducing points for scalability (reducing O(n³) complexity to O(nm²) for m inducing points). Pending observations prevent redundant evaluation of running trials in parallel settings.

### Parameter space and objective design for contact convergence

**Parameter encoding** transforms Tribol and Smith parameters into normalized [0,1] coordinates: log-scale transforms for penalty stiffness [10³,10⁸] and tolerances [10⁻⁸,10⁻⁴], one-hot encoding for categorical choices (enforcement method, solver type), and linear normalization for bounded continuous parameters (iterations [10,1000], search expansion [1.1,1.5]).

The **objective function** prioritizes convergence, then efficiency, then speed:
```
J(θ) = 10.0×(1-converged) + 1.0×(iters/max_iters) + 0.5×(time/timeout)
```

This formulation treats non-convergence as catastrophic (dominates objective) while enabling optimization of computational efficiency among converging configurations. Severe numerical instabilities receive 2× floor padding (double penalty), while early successful convergence receives 0.9× actual value (reward for efficiency).

**ARD (Automatic Relevance Determination) kernels** automatically detect parameter importance through length scale learning—parameters with longer length scales contribute less to predictions and can be fixed or removed. The Matérn-5/2 kernel with ARD provides twice-differentiable smoothness appropriate for iterative solver convergence surfaces while avoiding overfitting.

### FR-BO workflow for FE contact problems

**Phase 1 (Trials 1-20)**: Sobol quasi-random initialization explores parameter space uniformly. Train initial dual GPs on observed successes and failures. Establish baseline performance and identify any obviously problematic regions. This space-filling phase ensures diverse training data before activating acquisition-driven sampling.

**Phase 2 (Trials 21-200)**: FR-BO iterations optimize FREI acquisition function using 10 random restarts plus L-BFGS-B. Each iteration: (1) update dual GPs with new data, (2) compute FREI over candidate set using Monte Carlo acquisition, (3) select next θ maximizing FREI, (4) execute FE simulation with early termination monitoring, (5) record results and binary convergence label. Retrain GPs every 50 trials to incorporate hyperparameter adaptation.

**Phase 3 (Post-optimization)**: Validate best configuration on held-out test geometries. Perform Sobol sensitivity analysis to identify critical parameters. Generate feasibility probability maps for visualization. Deploy optimal parameters with uncertainty estimates and predicted success probability.

**Component interactions** flow through the AxClient orchestration layer: GenerationStrategy manages Sobol→FR-BO transition automatically. Dual GPs predict objective mean/variance and failure probability independently. FREI acquisition optimizes the product of EI and (1-P_fail) via multi-start gradient ascent. The FE executor wraps Tribol+Smith simulation, monitors convergence indicators, and triggers early termination when trajectory extrapolation predicts failure with 80% confidence.

### FR-BO use case implementations

**Uncertainty quantification for new geometries** employs multi-task GP extensions where each geometry represents one task. Extract geometric features (contact area ratio, mesh density, material property contrast, gap distribution statistics) as task descriptors. The MTGP shares statistical strength across similar geometries while maintaining geometry-specific predictions. For a new geometry, query the MTGP across a parameter grid, compute success probability and uncertainty, then recommend parameters via UCB-style scoring: score = P(success) - β×uncertainty where β=2 provides 95% confidence bounds. Return the top-3 configurations with predicted success rates and confidence intervals.

**Real-time monitoring for early termination** fits a GP to the partial residual trajectory log(residual) vs iteration number at iteration checkpoints (every 5 iterations after iteration 10). Extrapolate to max_iterations using the posterior predictive distribution. If P(convergence) < 0.2 with confidence > 0.8, terminate early and label as "predicted failure." This saves 30-50% compute on failing configurations while updating FR-BO with failure data that improves future predictions. The GP trajectory model uses a Matérn-3/2 kernel for residual evolution, capturing typical exponential decay patterns.

**Pre-simulation validation** computes a risk score combining: 0.5×P_fail(θ) from the classifier + 0.3×(1-distance_to_nearest_failure) measuring proximity to known failures + 0.2×(1-local_success_rate) in a k-nearest-neighbor region. Risk thresholds: >0.7 "HIGH RISK - Do not run" with suggested safe alternative via constrained optimization projecting to nearest high-probability region; 0.4-0.7 "MODERATE RISK - Proceed with caution"; <0.4 "LOW RISK - Proceed." Safe alternatives optimize: minimize ||θ_new - θ_proposed||² subject to P_fail(θ_new) < 0.2.

### FR-BO visualization requirements

**Iteration diagram** (flowchart): Current data D_j → Update dual GPs → Predict failure regions (2D heatmap projection) → Compute FREI acquisition surface → Optimize FREI (show multiple restarts converging) → Execute simulation (branch success/failure) → Update D_{j+1} → Loop. Annotate with convergence criteria and budget tracking.

**System architecture diagram** (layered): Top layer shows user interface (parameter input forms, real-time dashboards, 3D geometry viewer). Middle layer contains Ax orchestration engine, dual GP models (show separate boxes for objective GP and failure VGPC), FREI acquisition optimizer, and early stopping logic. Bottom layer depicts FE executor (Tribol+Smith simulation engine), convergence monitoring module, and results database. Arrows show trial lifecycle: parameter suggestion → simulation → result extraction → GP update → next iteration.

**Parameter space visualization** (2D projection via PCA or t-SNE): Color regions by P_fail from the classifier—green (P_fail < 0.2), yellow (0.2 ≤ P_fail ≤ 0.5), red (P_fail > 0.5). Overlay successful trials (blue dots), failed trials (red X markers), current best (gold star), and FREI acquisition contours. This reveals how FR-BO navigates around failure regions while optimizing.

## System 2: Gaussian Process Optimization with Sign-Based Classification

GP classification models the binary convergence outcome directly as a probabilistic prediction, enabling constrained optimization where feasibility itself becomes a probabilistic constraint integrated into acquisition functions.

### GP classification for convergence prediction

GPs for classification place a prior on a latent function f(x) ~ GP(μ, K) then squash through a link function to predict binary outcomes: P(converged|x) = σ(f(x)) where σ is the logistic sigmoid or probit function Φ. The Bernoulli likelihood P(y|f) = σ(f)^y × (1-σ(f))^(1-y) renders the posterior p(f|y) non-Gaussian, requiring approximation methods.

**Laplace approximation** finds the posterior mode f̂ via Newton iterations, then approximates the posterior as Gaussian q(f) = N(f|f̂, H⁻¹) where H is the Hessian. **Expectation propagation** iteratively refines local Gaussian approximations for better accuracy. **Variational inference** (used in GPyTorch) optimizes a variational distribution to lower-bound the marginal likelihood, providing scalable inference through inducing points and stochastic gradients.

The **prediction process** computes: p(y*=1|D) ≈ ∫ σ(f*) p(f*|D) df* using Monte Carlo integration or analytical approximations. The variational GP classifier provides both convergence probability and epistemic uncertainty (model uncertainty about the classification boundary location).

Sign-based classification naturally suits binary convergence outcomes: y=1 (residual < tolerance, constraints satisfied) versus y=0 (divergence, numerical failure, excessive iterations). The probabilistic output P(converged) ∈ [0,1] enables risk-aware decision making and soft constraint handling in acquisition functions.

### Constrained optimization integration

**Constrained Expected Improvement** multiplies objective EI by feasibility probability: α_CEI(x) = α_EI(x) × P(feasible|x). This product formulation naturally balances finding good solutions in feasible regions—high EI with low feasibility is penalized, low EI with high feasibility wastes evaluations, and high EI with high feasibility receives maximum acquisition value.

Alternative **entropy-based acquisitions** include: **Max-value Entropy Search (MES)** reducing entropy about the optimal value f*, **Joint Entropy Search (JES)** reducing joint entropy over input-output pairs, and **PESC (Predictive Entropy Search with Constraints)** specifically designed for constrained problems. These information-theoretic approaches maximize information gain about the optimal solution location.

**Feasibility modeling** maintains an independent GP for the convergence constraint c(x) with Probability of Feasibility: PoF(x) = P(c(x) ≤ 0) computed via the GP posterior. For binary convergence, the GP classifier directly provides P(converged|x) = P(feasible|x).

**Exploration-exploitation balance**: Early iterations emphasize exploration (high uncertainty regions via entropy acquisition), middle iterations balance both (CEI with equal weighting), late iterations emphasize exploitation (high EI in known feasible regions). Adaptive weighting schedules adjust based on feasible region discovery rate.

### BoTorch implementation with classification

**ModelListGP architecture** combines a SingleTaskGP for the objective (trained on successful trials only) with a variational GP classifier for convergence (trained on all trials):

```python
model_obj = SingleTaskGP(train_X=X_success, train_Y=y_obj)
model_con = GP_vi(X_all, y_converged)  # Variational GP classifier
model = ModelListGP(model_obj, model_con)
```

The **variational GP classifier** (GP_vi) inherits from ApproximateGP and GPyTorchModel, using CholeskyVariationalDistribution with inducing points for scalability. The likelihood is BernoulliLikelihood providing the sigmoid transform. Training maximizes the evidence lower bound (ELBO) using Adam optimization.

**Constrained acquisition implementation** uses qLogExpectedImprovement with a custom constraint function:

```python
def convergence_constraint(Z, model_con, X=None):
    y_con = Z[..., 1]  # Convergence model output dimension
    prob = model_con.likelihood(y_con).probs
    return prob - threshold  # Negative if infeasible
```

**Reparameterization trick**: BoTorch samples latent functions as f(x) = μ(x) + L(x)ε where L(x)L(x)^T = K(x,x) and ε ~ N(0,I). This enables automatic differentiation through the acquisition function for gradient-based optimization. For classification, samples are transformed through the likelihood's sigmoid to get probability samples.

**Handling mixed parameter types**: MixedSingleTaskGP accommodates both continuous and categorical parameters (enforcement method, solver type) by treating categoricals with appropriate kernels (Hamming distance for small cardinality, one-hot encoding with standard kernels for larger sets).

### Three-phase exploration strategy

**Phase 1: Initial Exploration (Iterations 1-20)** maps the convergence landscape using entropy-based acquisition maximizing uncertainty reduction about the classification boundary. Focus on broad coverage to establish where the feasible/infeasible transition occurs. Use Sobol quasi-random sampling for space-filling initialization, then switch to entropy-driven sampling to actively learn boundary locations.

**Phase 2: Boundary Refinement (Iterations 21-50)** concentrates samples near the decision boundary P(converge) ≈ 0.5. This region contains maximum uncertainty about convergence and maximum information about optimal parameters likely residing on or near the boundary. Acquisition becomes: α(x) = CEI(x) × entropy(x) × boundary_proximity(x) where boundary_proximity = exp(-5×(P(converge)-0.5)²) peaks at P=0.5.

**Phase 3: Exploitation (Iterations 51+)** optimizes performance in known feasible regions using CEI with high weight on EI. By this phase, the classifier accurately predicts feasibility with low uncertainty in most regions, enabling confident optimization. Adaptive weighting: w_feas starts 0.8 (early), decreases to 0.2 (late) while w_perf inversely increases.

### GP classification system architecture

**Data layer** maintains a time-series database of (parameters, convergence_status, performance_metrics, geometry_metadata) enabling both classification and regression models plus transfer learning across geometries.

**Model layer** contains two components: (1) Variational GP Classifier for convergence prediction with inducing points (100-200 for datasets >500 samples), Matérn-5/2 kernel with ARD, BernoulliLikelihood, (2) Exact GP Regressor for objective trained only on converged trials, using the same kernel family for consistency.

**Acquisition layer** implements constrained EI with entropy bonuses. The acquisition optimizer uses 10 random restarts with L-BFGS-B for continuous parameters, cycling through categorical choices for mixed spaces. For batch optimization (parallel simulations), uses SobolQMCNormalSampler with 1024 samples and diversity promotion via determinantal point processes.

**Execution layer** provides parameter validation (physics-based sanity checks: penalty > min_threshold, tolerance < max_gap), pre-simulation convergence probability checks (reject if P(converge) < 0.3), simulation execution with monitoring, and automated labeling of convergence outcomes based on residual satisfaction and iteration budgets.

### Use case adaptations for GP classification

**Initial parameter suggestions** employs clustering of historically successful parameters using k-means (k=3-5 clusters). For a new geometry, extract geometric features, find similar historical geometries via k-nearest-neighbors in feature space, filter their successful parameter sets, cluster these, and present cluster centers with predicted convergence probabilities. Each suggestion includes: parameter values, estimated P(converge), confidence level (high if σ<0.15, medium if σ<0.3, low otherwise), and expected performance if converged.

**Real-time convergence probability estimation** provides interactive exploration: generate a parameter grid (50×50 for 2D projections, coarser for higher dimensions), evaluate GP classifier at all grid points (vectorized for efficiency), visualize as heatmaps with contours at P=0.5, P=0.7, P=0.9, and overlay historical trials. Users interactively adjust parameters and see convergence probability update in real-time (<100ms latency). Uncertainty visualization shows regions requiring more data via saturation/transparency.

**Pre-simulation validation** implements a multi-stage checking pipeline: (1) ML prediction via GP classifier returns P(converge) and uncertainty, (2) Physics rules check parameter bounds and relationships (penalty vs material stiffness, gap tolerance vs element size), (3) Historical comparison finds similar configurations in database. Aggregate into a composite risk score. For failed validation, the system projects to the nearest high-probability region using constrained optimization: minimize ||θ_new - θ_proposed||² subject to P(converge|θ_new) ≥ 0.8 and physics constraints.

### GP classification visualizations

**Convergence landscape** (2D heatmap): Show probability surface P(converge|penalty_stiffness, gap_tolerance) with penalty on log-scale x-axis, gap_tolerance on log-scale y-axis. Color gradient from red (P≈0) through yellow (P≈0.5) to green (P≈1). Contour lines at 0.5, 0.7, 0.9 probabilities. Overlay data points: blue circles (converged), red crosses (failed), with marker size proportional to performance quality. This reveals the "safe operating region" and convergence boundary shape.

**Classification confidence** (uncertainty map): Same 2D projection but color-coded by predictive standard deviation σ—dark blue (low uncertainty, confident predictions) to bright yellow (high uncertainty, needs more data). This identifies where additional sampling would most reduce prediction uncertainty. Superimpose acquisition function contours showing next sampling locations.

**Decision boundary evolution** (time-series animation): Sequence of frames showing how the decision boundary P(converge)=0.5 evolves as trials accumulate. Early frames show wide uncertainty bands; later frames show tight, well-defined boundaries. Helps communicate learning progress and convergence of the optimization process.

## System 3: Constrained Efficient Global Optimization (CONFIG)

CONFIG provides rigorous theoretical guarantees for optimizing expensive black-box objectives subject to unknown convergence constraints, using optimistic feasibility estimates and bounded cumulative violations.

### CONFIG theoretical framework

Unlike heuristic constraint handling (probability weighting, penalty methods), **CONFIG provides provable convergence** with sublinear regret and bounded constraint violations. The algorithm maintains an **optimistic feasible set** that contains the true feasible region with high probability, enabling strategic constraint violations that accelerate learning without sacrificing convergence guarantees.

The original problem: minimize f(x) subject to c_i(x) ≤ 0 for i=1,...,m and x ∈ X becomes an auxiliary problem at each iteration: x_{n+1} = argmin_{x∈F_n^opt} l_0,n(x) where F_n^opt = {x : l_i,n(x) ≤ 0 ∀i} is the optimistic feasible set and l_i,n(x) = μ_i,n(x) - β_n^(1/2) σ_i,n(x) are lower confidence bounds.

**Theoretical guarantees**: Cumulative regret R_T = O(√(T γ_T log T)) and cumulative violations V_T = O(√(T γ_T log T)) where γ_T is the maximum information gain (sublinear for Matérn and squared exponential kernels, e.g., γ_T = O(log^(d+1) T) for Matérn kernels). Convergence rate to ε-optimal solution: O((γ*/ε)² log²(γ*/ε)) evaluations.

The **optimistic principle** permits violations when uncertainty is high—if the lower confidence bound suggests feasibility, sample that point even if the mean prediction indicates infeasibility. This accelerates boundary learning compared to conservative approaches (SafeOpt) that never violate but may get stuck, or heuristic methods (probability weighting) lacking convergence guarantees.

### Comparison with standard EGO approaches

**Standard EGO** uses expected improvement α_EI(x) = E[max(0, f_best - f(x))] without explicit constraint handling. **CEI (Constrained EI)** multiplies EI by probability of feasibility P(feasible|x) but lacks theoretical analysis and can be overly conservative. **Augmented Lagrangian EGO** converts constrained problems to unconstrained via penalty parameters, but requires manual tuning and may not converge for infeasible problems.

**CONFIG advantages**: (1) Principled exploration-exploitation balance via LCB-based acquisition (no tunable weights), (2) Provable convergence guarantees and finite-time regret bounds, (3) Natural infeasibility detection—if optimistic set shrinks to empty, problem is likely infeasible, (4) Bounded cumulative violations enable limited constraint violations for faster learning without divergence.

**SafeOpt advantages**: Never violates constraints but requires a safe seed point and may get trapped in local feasible regions. **CONFIG advantages**: Accepts initial violations, globally explores, theoretically guaranteed convergence. Trade-off: CONFIG incurs bounded violations; SafeOpt maintains strict safety.

### CONFIG implementation in Ax/BoTorch

**Constraint types**: BoTorch distinguishes parameter constraints (known linear relationships in input space, enforced exactly) from outcome constraints (unknown black-box functions modeled via GPs, handled probabilistically).

For contact convergence, convergence is an **outcome constraint**: the simulation must be executed to determine convergence, making it an expensive black-box constraint function. Model with a separate GP: c(x) represents residual_norm - tolerance, where c≤0 means converged.

**BoTorch constrained acquisition** implements SampleReducingMCAcquisitionFunction with constraints as a list of callables. Each constraint function takes Monte Carlo samples and returns negative for feasible, positive for infeasible:

```python
def convergence_constraint(samples, X=None):
    residual_pred = samples[..., constraint_idx]
    threshold = tolerance
    return threshold - residual_pred  # Negative when residual < tolerance
```

**CONFIG-style LCB acquisition** requires custom implementation inheriting from AnalyticAcquisitionFunction (for deterministic posteriors) or MCAcquisitionFunction (for noisy observations). Compute LCB = μ(x) - β^(1/2) σ(x) where β follows the theoretical schedule β_n = 2 log(π² n² / 6δ) for failure probability δ.

**Optimistic feasible set** represented as: F_opt = {x : LCB_constraint(x) ≤ 0}. The optimization becomes: minimize LCB_objective(x) subject to x ∈ F_opt. Implement via constrained acquisition: alpha_CONFIG(x) = -LCB_objective(x) with constraint LCB_constraint(x) ≤ 0. Optimize using scipy.optimize with constraint handling.

### Multi-phase CONFIG strategy for contact problems

**Phase 1: Initialization (0-20 evaluations)** uses Latin Hypercube Sampling to explore the parameter space and identify any initial feasible points. If no feasible points found, increase LCB β to expand the optimistic set (more tolerant of uncertainty). This phase establishes baseline GP models and prevents degenerate cases where the optimistic set is empty.

**Phase 2: Feasibility Discovery (20-40 evaluations)** focuses on **active constraint learning** by maximizing uncertainty near predicted constraint boundaries. Modify acquisition to: α(x) = uncertainty_constraint(x) × proximity_to_boundary(x) where proximity = exp(-5×LCB_constraint(x)²). This maps the convergence boundary efficiently, critical for accurate subsequent optimization.

**Phase 3: Constrained Optimization (40-80 evaluations)** implements CONFIG-style LCB acquisition within the optimistic feasible set. Each iteration: (1) Compute LCB for objective and constraints, (2) Define F_opt using LCB_constraint ≤ 0, (3) Optimize LCB_objective over F_opt using 20 random restarts, (4) Evaluate selected point, (5) Update GPs and increase β_n. Track cumulative violations to ensure they remain sublinear.

**Phase 4: Refinement (80-100 evaluations)** performs local search near the best feasible solution using tighter confidence bounds (smaller β). This phase polishes the solution, confirms convergence, and validates robustness via nearby parameter perturbations.

**Adaptive phase switching** based on: (1) Feasible point availability—if still zero after 30 evaluations, remain in discovery longer, (2) Boundary understanding—if constraint GP uncertainty uniformly low, advance to optimization, (3) Solution stability—if best feasible solution unchanged for 10 iterations, transition to refinement.

### CONFIG constraint modeling

**Constraint formulation** treats convergence as a continuous function: c(x) = log10(final_residual) - log10(tolerance). This formulation: (1) Naturally extends to multi-threshold constraints (different tolerances for different metrics), (2) Provides smooth gradients for GP modeling, (3) Handles non-convergence via clipping (final_residual = max_float if diverged), (4) Enables both satisfied (c<0) and quantified violation magnitude (c>0).

**Alternative binary formulation**: c(x) = 0 if converged, 1 if failed. Simpler but loses information about "how close" to convergence, making GP modeling less effective. The continuous formulation better captures the convergence landscape topology.

**Multiple constraints** for comprehensive safety: c_1(x) = residual - tolerance (convergence), c_2(x) = iterations - max_iter (budget), c_3(x) = max_penetration - penetration_limit (physics validity). CONFIG handles multiple constraints via intersection of optimistic feasible sets: F_opt = {x : LCB_i(x) ≤ 0 ∀i}.

**Constraint GP modeling**: Use same kernel family as objective (Matérn-5/2 with ARD) for consistency. Separate hyperparameter optimization per constraint GP. For constraints with heteroscedastic noise (e.g., iteration count variance increases near boundary), use heteroscedastic noise models or input-dependent noise.

### CONFIG system components

**BO Controller** implements the CONFIG iteration loop: (1) Fit objective and constraint GPs with maximum likelihood estimation, (2) Compute LCB bounds using theoretical β schedule, (3) Construct optimistic feasible set F_opt, (4) Solve auxiliary optimization problem min LCB_obj subject to x ∈ F_opt, (5) Evaluate simulation, (6) Track cumulative violations and regret, (7) Repeat until budget exhausted or convergence criteria met.

**Constraint Learner** actively targets the constraint boundary using information-theoretic acquisition. After sufficient data (>30 samples), compute Expected Information Gain about the constraint boundary location. Sample points maximizing EIG to refine boundary understanding. This module provides the boundary refinement phase, complementing CONFIG's optimization focus.

**Feasibility Predictor** provides real-time queries: given θ, return (P_feasible, confidence). Uses the constraint GPs to compute P(c_i(θ) ≤ 0) for all constraints, aggregates via independence assumption: P(feasible|θ) = ∏_i P(c_i(θ) ≤ 0). Confidence quantified via prediction variance—low variance indicates confident predictions. Enables pre-simulation filtering and dashboard visualization.

**Violation Monitor** tracks cumulative constraint violations: V_t = ∑_{i=1}^t max(0, c(x_i)). CONFIG theory guarantees V_t = O(√t), so violation rate decreases over time. If violations exceed theoretical bounds significantly, signal possible model misspecification or infeasible problem. Adaptive: if violations approaching budget, increase β to be more conservative.

**Geometry Adapter** extends CONFIG to multiple geometries via multi-task GPs. Each geometry becomes a task with shared base kernels and geometry-specific variation. For a new geometry, leverage transferred knowledge: (1) Initialize with predictions from similar geometries, (2) Refine with geometry-specific samples, (3) Allocate budget proportionally: 20% initial exploration for new geometry, 80% transferred from knowledge base.

### CONFIG use case implementations

**Learning feasible parameter regions**: Deploy CONFIG with objective = constant (focus purely on constraint satisfaction). This degenerates to safe Bayesian optimization focused on mapping F. Use active boundary learning to concentrate samples near constraint boundary (P≈0.5). Output: feasible region volume estimate via Monte Carlo, boundary parameterization, conservative safe parameter recommendations (well inside feasible region for robustness).

**Real-time feasibility assessment**: Pre-train CONFIG models on historical data from similar contact problems. For new parameter query θ: (1) Compute LCB_constraint(θ) using trained models, (2) Check if θ ∈ F_opt (LCB ≤ 0), (3) Return: feasible_estimate (Yes/No/Uncertain), probability P(c(θ)≤0), confidence (high if σ_constraint(θ) < 0.2), nearest_safe_point (if infeasible). Cache frequently-queried regions for <50ms latency.

**Pre-simulation constraint checking for batches**: Given a candidate batch of N configurations, predict c(θ) for each. Rank by LCB_constraint (lower = more likely feasible). Select top K% for evaluation. Expected success rate: ∑_i P(c(θ_i)≤0) / K. Adaptive thresholding: if success rate in previous batch was high, increase selectivity; if low, be more permissive. Diversity promotion: penalize candidates too close to each other in parameter space to ensure broad exploration.

### CONFIG visualizations

**Optimistic feasible set evolution**: 2D parameter space projection showing F_opt boundaries across iterations. Frame 1 (iter 5): Large, uncertain feasible region (wide confidence bands). Frame 20: Narrowing feasible region as boundary is learned. Frame 50: Well-defined boundary, F_opt tightly bounds true feasible region. Frame 100: Converged feasible region with interior optimum marked. Overlay violations (red dots outside F_opt) showing they decrease in frequency over time.

**LCB acquisition surface**: 3D surface plot with parameters on x-y axes, LCB_objective on z-axis. Show feasibility mask (F_opt in color, infeasible region greyed out). The minimum of LCB within F_opt marks the next sampling location. Illustrates how CONFIG navigates the tradeoff between optimistic boundary (includes uncertain regions) and objective optimization (prefers low LCB values).

**Cumulative violations trajectory**: Time series plot with iteration on x-axis, cumulative violations on y-axis (log scale). Show actual V_t (solid line) versus theoretical bound O(√t) (dashed line). Include violation rate ΔV_t = V_t - V_{t-1} (bar chart, should trend toward zero). Demonstrates that violations are bounded and decrease, validating CONFIG's theoretical properties.

## System 4: Surrogate Optimization with Hidden Constraints (SHEBO)

SHEBO combines surrogate modeling with constraint discovery, using ensemble approaches for robust feasibility prediction and adaptive sampling to map unknown convergence boundaries.

### SHEBO methodology overview

SHEBO addresses optimization when constraints are **completely unknown a priori**—not just their functional form, but even which constraints exist. For contact convergence, we know a priori that convergence is required, but SHEBO principles apply when discovering unexpected failure modes (numerical instability, mesh distortion, unphysical states).

The core idea: build **multiple surrogates** (typically neural networks or GPs) for both objectives and discovered constraints, use ensemble disagreement to quantify epistemic uncertainty, and employ **active learning** to sample high-uncertainty regions that might reveal hidden constraints.

**Surrogate types**: Polynomial chaos expansions (fast, limited complexity), radial basis functions (local interpolation), Gaussian processes (uncertainty-aware), neural network ensembles (flexible, high-capacity), gradient-boosted trees (discontinuities, categorical features). For contact convergence: neural network ensembles recommended for complex, high-dimensional parameter spaces with non-smooth convergence boundaries.

**Hidden constraint identification**: (1) During optimization, monitor for unexplained failures (divergence, crashes), (2) Introduce new constraint models for discovered failure modes, (3) Actively sample near suspected constraint boundaries to refine understanding, (4) Integrate constraints into acquisition function as they're discovered, (5) Iterate until no new constraint violations observed over N consecutive iterations.

### Ensemble surrogate modeling for constraints

**Ensemble architecture**: Train 5-10 neural networks with identical architecture but different initializations and dropout patterns. Each network predicts convergence probability P_k(converge|x). Ensemble mean captures aleatoric uncertainty (inherent randomness), ensemble variance captures epistemic uncertainty (model uncertainty from limited data).

**Network architecture for convergence**: Input layer (parameters, normalized), 3 hidden layers (128→64→32 neurons, ReLU activation), dropout layers (p=0.2), output layer (1 neuron, sigmoid activation for probability). Train with binary cross-entropy loss on convergence labels. Use early stopping on validation set to prevent overfitting.

**Uncertainty quantification**: Total uncertainty = aleatoric + epistemic. Aleatoric: average predictive entropy E_p[H(p)] = -E[p log p + (1-p) log(1-p)]. Epistemic: variance of ensemble predictions Var[P_k(x)]. High epistemic uncertainty indicates regions needing more data (acquisition target). High aleatoric uncertainty indicates inherent randomness (boundary regions, chaotic dynamics).

**Alternative: GP ensembles** using different kernel families (Matérn-3/2, Matérn-5/2, RBF) and hyperparameter samples from the posterior. More theoretically grounded than NNs but less scalable to high dimensions and large datasets. Hybrid approach: GP for low-dimensional critical parameters, NN for high-dimensional auxiliary parameters.

### SHEBO system architecture

**Surrogate manager** maintains a zoo of models: performance surrogate (regression), convergence surrogate (classification), constraint surrogates (one per discovered constraint type). Models updated asynchronously—fast surrogates (NNs) retrain every 10 samples, expensive surrogates (GPs with hyperparameter optimization) retrain every 50 samples.

**Constraint discovery module** implements anomaly detection on simulation outputs. Define expected behavior profiles (residual should decrease monotonically, penetration should remain bounded). Monitor for violations: residual increase, oscillations, NaN/Inf values, unphysical states (negative pressures, inverted elements). When anomaly detected: (1) Label as new constraint, (2) Train surrogate for this constraint, (3) Add to acquisition function as penalty term.

**Adaptive acquisition** balances multiple objectives: (1) Optimize performance (EI), (2) Ensure feasibility (constraint satisfaction probability), (3) Reduce uncertainty (entropy), (4) Explore boundaries (sample near P≈0.5). Use scalarization: α(x) = w_1 EI(x) + w_2 P(feasible|x) + w_3 H(x) + w_4 boundary_proximity(x). Weights adapt: early exploration (w_3=0.5), mid boundary learning (w_4=0.5), late exploitation (w_1=0.5, w_2=0.3).

**Workflow**: Initialize with space-filling design → Train initial surrogates → Discover constraints from failures → Update constraint surrogates → Adaptive acquisition → Evaluate next point → Retrain surrogates → Repeat. Termination: budget exhausted OR no new constraints discovered in last 30 iterations OR best solution stable.

### SHEBO constraint modeling specifics

**Convergence surrogate**: Binary classification NN ensemble predicting P(converge|parameters). Input features include both raw parameters and engineered features (penalty/material_stiffness ratio, timestep/contact_timescale ratio). Augment training with SMOTE (Synthetic Minority Oversampling) if convergence failures are rare (<10% of samples) to balance classes.

**Performance surrogate**: Regression NN ensemble predicting log(iteration_count) and log(solve_time) for converged cases. Separate model or multi-output architecture. Trained only on successful trials to avoid contamination from failures. Log transform ensures positive predictions and stabilizes training.

**Hidden constraints**: Beyond convergence, discover: (1) Mesh quality degradation (Jacobian determinant), (2) Contact detection failures (no contact pairs found), (3) Time step crashes (timestep → 0), (4) Load balancing issues (parallel efficiency), (5) Memory overflow. Each gets a dedicated surrogate when first detected.

**Multi-fidelity extension**: Use cheap low-fidelity simulations (coarse mesh, loose tolerances) to pre-screen, expensive high-fidelity simulations for refinement. Train surrogates on both, use auto-regressive models: high_fidelity(x) = α × low_fidelity(x) + GP_correction(x). Reduces total computational cost by 60-70% versus high-fidelity only.

### SHEBO implementation approach

**Ax integration**: SHEBO is not natively implemented in Ax/BoTorch but can be constructed via custom components. Use Ax's ServiceAPI for flexible experiment management. Implement custom ModelBridge that wraps the ensemble surrogate manager. GenerationStrategy with custom generation steps: (1) Sobol initialization, (2) Custom SHEBO acquisition.

**PyTorch-based surrogate**: Leverage PyTorch for NN surrogates with GPU acceleration. Use PyTorch Lightning for training loop boilerplate. Ensemble via Bayesian neural networks (Monte Carlo dropout at test time) or explicit ensemble (train multiple networks, average predictions). Integrate with BoTorch objectives and constraints via wrapper classes.

**Active learning integration**: Use acquisition functions from modAL or custom implementations. For boundary sampling: uncertainty sampling (maximize predictive variance), query-by-committee (maximize ensemble disagreement), expected model change (maximize expected parameter change). Combine with BO acquisition: α_SHEBO(x) = α_BO(x) × (1 + λ × uncertainty(x)) where λ decays over time (explore early, exploit late).

**Scalability via approximations**: For large parameter spaces (>20 dimensions), use: (1) Random projections to 10-dimensional subspace, (2) Active subspaces identifying critical dimensions, (3) Warped GPs for non-stationary behavior, (4) Inducing point methods for constraint GPs (reduce n³ cost), (5) Batch parallelization with diversity promotion (evaluate 5-10 points simultaneously).

### SHEBO use case implementations

**Building surrogate models for geometry families**: Collect data across multiple geometries (10-20 representative cases). Train multi-task NN with geometry descriptors as auxiliary inputs: inputs = [contact_parameters, geometry_features], output = convergence_probability. Geometry features: contact_area, mesh_size, material_contrast, aspect_ratio. Transfer learning: pre-train on large geometry database, fine-tune on specific new geometry with 20-30 samples. Expected performance: 70% fewer samples needed versus training from scratch.

**Real-time surrogate-based monitoring**: Deploy lightweight NN surrogates (<1MB) on simulation compute nodes. Every K iterations, extract current parameter state and residual trajectory. Feed to surrogate for convergence prediction. If P(eventual_converge) < 0.3, terminate early. Surrogate inference time <10ms enables negligible overhead. Savings: 40% compute time by avoiding doomed simulations.

**Pre-simulation prediction**: User specifies parameters → Query ensemble surrogates → Return prediction distribution (mean±std convergence probability, mean±std expected iterations if converges, mean±std solve time) → Visualize uncertainty (violin plots, confidence intervals) → Highlight high-risk parameters (ensemble disagreement >0.3) → Suggest alternatives if high-risk (nearest low-risk parameters via constrained optimization). Update surrogates online as new simulations complete.

### SHEBO visualizations

**Ensemble disagreement heatmap**: 2D parameter projection with color = standard deviation of ensemble predictions. High std (red/yellow) indicates epistemic uncertainty—regions needing more data. Low std (blue) indicates confident predictions. Overlay sample locations showing how sampling targets high-uncertainty regions initially, then shifts to low-uncertainty as learning progresses. Demonstrates active learning effectiveness.

**Constraint discovery timeline**: Horizontal timeline showing discovered constraints over iterations. Each constraint marked with: iteration discovered, type (convergence failure, numerical instability, mesh distortion), frequency of occurrence, current surrogate accuracy (cross-validation AUC). Illustrates how SHEBO progressively learns about failure modes, not just a single known constraint.

**Multi-objective Pareto frontier**: For time vs accuracy trade-off, show 3D surface: x=parameter1, y=parameter2, z=time, color=error. Overlay Pareto-optimal points (non-dominated solutions). Show feasibility constraint as transparent plane/surface—points below are infeasible. User can select preferred trade-off point from Pareto set based on priorities.

## Comparison framework and recommendations

### Method comparison across use cases

**For uncertainty quantification → initial parameter suggestions:**

**FR-BO** learns failure regions rapidly through strategic violations, providing optimal starting points after 50-100 trials. Best when: prior knowledge is limited, willing to tolerate early failures, need globally optimal initialization. Convergence speed: **Excellent**. Sample efficiency: **Good** (3-5x vs random). Setup complexity: **Medium** (dual GPs, custom acquisition).

**GP Classification** provides probabilistic predictions enabling risk-aware suggestions. Clustering successful parameters gives interpretable starting points. Best when: need confidence intervals on suggestions, want multiple alternative starting points, require explainability. Convergence speed: **Good**. Sample efficiency: **Excellent** (entropy-driven boundary learning). Setup complexity: **Medium** (classification GP, clustering).

**CONFIG** focuses on constrained optimization rather than initialization, so less suited for this use case. Can provide safe conservative starting points (well within F_opt) but slower than FR-BO/GP. Convergence speed: **Good**. Sample efficiency: **Fair**. Setup complexity: **High** (LCB calculations, auxiliary optimization).

**SHEBO** requires significant upfront data collection for ensemble training (100+ samples), making it slower for single-geometry initialization. Excels for geometry families where surrogates transfer. Convergence speed: **Fair** (initially slow, fast after training). Sample efficiency: **Fair to Excellent** (poor initially, excellent with transfer learning). Setup complexity: **High** (ensemble management, constraint discovery).

**Recommendation**: Start with **GP Classification** for interpretable, risk-aware initial suggestions. If global optimality is critical and failures are acceptable, use **FR-BO**.

### Method comparison for real-time monitoring

**FR-BO** trajectory-based early termination uses GP extrapolation of residual evolution. Requires fitting GP mid-simulation (<100ms overhead). Termination accuracy 85-90%. Best when: simulations are expensive (hours), can afford mid-simulation interruption, want probabilistic termination criteria. Computational overhead: **Low**. Prediction accuracy: **Excellent**. Implementation complexity: **Medium**.

**GP Classification** applied to partial convergence indicators (current residual, iteration, rate) predicts eventual outcome. Fast inference (<10ms) enables frequent checks. Best when: simulations have intermediate checkpoints, need very low overhead, can formulate features from partial state. Computational overhead: **Very Low**. Prediction accuracy: **Good to Excellent**. Implementation complexity: **Low to Medium**.

**CONFIG** focuses on parameter selection rather than runtime monitoring, so not directly applicable. Could predict convergence probability before starting simulation for batch filtering, but not for mid-simulation termination. Recommendation: **Not recommended** for this use case.

**SHEBO** with lightweight NN surrogates provides fast inference (<5ms) and high accuracy (>90%) after sufficient training. Best when: many simulations run routinely, can amortize surrogate training cost, need production deployment. Computational overhead: **Very Low**. Prediction accuracy: **Excellent** (after training). Implementation complexity: **High** (training infrastructure, deployment).

**Recommendation**: For one-off simulations, use **GP Classification trajectory extrapolation**. For production systems with many simulations, invest in **SHEBO ensemble surrogates**.

### Method comparison for pre-simulation validation

**FR-BO** risk scoring combines failure classifier, distance to known failures, and local success rate. Provides nearest safe alternative via projection. Best when: parameters proposed by users, need both validation and correction, can afford 10-100ms latency. Validation accuracy: **Excellent** (>90%). False positive rate: **Low** (<5%). Interpretability: **Good** (multi-factor risk score).

**GP Classification** multi-stage validation (ML + physics rules + historical comparison) provides comprehensive checking. Constrained optimization for safe alternatives. Best when: need high confidence validation, want physics-grounded constraints, require audit trails. Validation accuracy: **Excellent** (>95%). False positive rate: **Very Low** (<2%). Interpretability: **Excellent** (staged checks).

**CONFIG** optimistic feasible set provides natural validation: check if θ ∈ F_opt. LCB-based checking is theoretically grounded. Conservative (may reject good parameters near boundary). Best when: want rigorous validation with guarantees, acceptable to be overly cautious, need formal certification. Validation accuracy: **Good** (>85%). False positive rate: **Medium** (10-15%, conservative). Interpretability: **Medium** (LCB concept less intuitive).

**SHEBO** ensemble-based validation with uncertainty quantification flags high-disagreement cases for manual review. Handles multiple constraint types naturally. Best when: constraints are complex and evolving, need confidence estimation, acceptable to defer uncertain cases. Validation accuracy: **Excellent** (>92%). False positive rate: **Low** (<5%). Interpretability: **Good** (ensemble uncertainty intuitive).

**Recommendation**: For critical applications requiring maximum safety, use **GP Classification multi-stage validation**. For balanced performance with theoretical backing, use **CONFIG**.

### Strengths and weaknesses summary

**FR-BO Strengths**: Fast convergence via strategic violations, handles failures naturally, proven in expensive simulation contexts, relatively simple theory. **Weaknesses**: Violations may be unacceptable in some contexts, requires tuning floor padding strategy, limited formal guarantees for convergence.

**GP Classification Strengths**: Interpretable probability outputs, natural constraint handling via multiplication, entropy-based acquisition for active learning, integrates seamlessly with standard BO. **Weaknesses**: Classification accuracy depends on boundary complexity, may need many samples near boundary, non-Gaussian posterior requires approximation.

**CONFIG Strengths**: Rigorous theoretical guarantees (sublinear regret and violations), principled optimistic exploration, natural infeasibility detection, well-suited for safety-critical applications. **Weaknesses**: Conservative in practice (higher false positive rate), auxiliary optimization adds complexity, requires careful β schedule tuning, less intuitive for practitioners.

**SHEBO Strengths**: Handles complex multi-constraint problems, ensemble uncertainty quantification is robust, discovers hidden failure modes, scales to high dimensions with NNs. **Weaknesses**: Requires substantial training data initially, ensemble management adds complexity, constraint discovery may miss rare failure modes, computationally expensive training.

### Method selection decision tree

**Primary criterion: Data availability**
- **<50 samples**: Use FR-BO or GP Classification with active learning
- **50-200 samples**: Any method viable, choose based on other criteria
- **>200 samples**: SHEBO excels with ensemble surrogates

**Secondary criterion: Failure tolerance**
- **Violations unacceptable**: CONFIG (guaranteed bounds) or conservative GP Classification
- **Limited violations OK**: FR-BO (fastest learning)
- **Violations acceptable for learning**: SHEBO (discovers hidden constraints)

**Tertiary criterion: Theoretical rigor required**
- **Formal guarantees needed**: CONFIG (only method with proven bounds)
- **Probabilistic reasoning sufficient**: GP Classification or FR-BO
- **Empirical performance sufficient**: SHEBO

**Quaternary criterion: Computational budget**
- **Limited (<100 trials)**: FR-BO or GP Classification
- **Moderate (100-300 trials)**: FR-BO, GP Classification, or CONFIG
- **Extensive (>300 trials)**: SHEBO (amortizes training cost)

### Recommended starting approach

**Phase 1: Quick wins (Weeks 1-4)**: Implement **GP Classification system** as baseline. Reasons: (1) Balances all criteria well, (2) Moderate complexity, (3) Interpretable probability outputs stakeholders understand, (4) Natural pre-simulation validation, (5) Extensible to other use cases. Start with 20 Sobol samples, 80 GP-guided samples. Target: 70-80% success rate, identify reliable "safe" parameter ranges.

**Phase 2: Advanced optimization (Weeks 5-8)**: Add **FR-BO module** for aggressive parameter optimization when higher success rates are needed. Use FR-BO for critical simulations where finding optimal parameters justifies more trials. Compare FR-BO vs GP Classification on test cases, measure convergence speed and final solution quality.

**Phase 3: Production deployment (Weeks 9-12)**: If success warrants production deployment across many geometries, develop **SHEBO ensemble surrogates**. Train on aggregated data from Phases 1-2 (200+ samples). Deploy lightweight NNs for real-time validation and monitoring. Maintain CONFIG as fallback for safety-critical applications requiring formal guarantees.

**Parallel exploration**: Throughout all phases, collect structured data enabling future transfer learning: geometry features, material properties, mesh statistics, convergence outcomes, parameter configurations. This dataset becomes invaluable for multi-task learning and generalizing across new contact problems.

## Implementation priorities and workflow integration

### Generalization strategy for evolving parameter sets

**ARD kernel approach**: Automatic Relevance Determination kernels learn parameter importance via length scale hyperparameters. Parameters with large length scales (>1.0 in normalized space) contribute little to predictions—candidates for removal. Parameters with small length scales (<0.1) are critical—retain and possibly increase sampling density.

**Sobol sensitivity analysis**: Compute first-order indices S_i = Var[E(J|θ_i)] / Var(J) and total-order indices S_Ti measuring parameter i's contribution including interactions. Rank parameters: S_Ti > 0.1 (critical), 0.01 < S_Ti < 0.1 (moderate), S_Ti < 0.01 (negligible). Remove negligible parameters; fix moderate parameters at nominal values; focus optimization on critical parameters.

**Adding new parameters**: When Tribol/Smith updates introduce new parameters: (1) Add to parameter space with broad initial ranges, (2) Sobol initialization across expanded space (add 10-20 samples), (3) GP automatically models interactions via cross-terms in kernel, (4) ARD determines importance within 20-30 samples, (5) Adjust future sampling density based on learned importance. No retraining from scratch required—GP updates incrementally.

**Removing parameters**: Fix at recommended values determined by: (1) Sobol analysis (insensitive parameters → any reasonable value), (2) Physics intuition (dimensional analysis, asymptotic limits), (3) Computational efficiency (choose values minimizing cost among equivalent outcomes). Document removed parameters and fixed values for reproducibility and future reconsideration if new physics discovered.

**Modular parameter configuration**: Maintain YAML-based parameter registry:
```yaml
parameters:
  penalty_stiffness:
    type: continuous
    range: [1e3, 1e8]
    scale: log
    importance: critical
  enforcement_method:
    type: categorical
    values: [mortar, penalty, augmented_lagrange]
    importance: moderate
  search_expansion:
    type: continuous
    range: [1.0, 2.0]
    scale: linear
    importance: low
```

### Real-time monitoring architecture for mid-simulation detection

**Architecture components**:

**Monitoring agent** (lightweight process on simulation node): Every N iterations (N=5-10), extract state vector: [current_iteration, residual_norm, residual_derivative, contact_active_set_size, penetration_max, jacobian_condition_estimate]. Serialize to shared memory buffer (zero-copy IPC).

**Predictor service** (dedicated process or remote service): Poll shared memory, deserialize state vectors, apply trained GP/NN predictor to forecast convergence probability. If P(converge) < termination_threshold (0.2-0.3) with confidence > 0.8, send termination signal via signal handler (SIGUSR1) or socket.

**Simulation wrapper** (modified Tribol/Smith driver): Register signal handler for graceful termination. On signal receipt: (1) Save current state to checkpoint file, (2) Write termination metadata (reason, iteration, predicted outcome), (3) Clean up resources, (4) Exit with status code indicating early termination.

**Data feedback loop**: Terminated simulations provide partially-converged data. Extract features: convergence_rate = log(residual_i) - log(residual_{i-k}) per k iterations, contact_stability = variance(active_set_size), stagnation = iterations_without_progress. These features improve future early termination predictions via reinforcement learning: reward = compute_time_saved if correctly terminated, penalty = wasted_cost if incorrectly terminated.

**Dashboard visualization**: Real-time web interface showing: (1) Running simulations (list with parameters, current iteration, predicted convergence probability with confidence bars), (2) Residual trajectory plots with GP prediction bands, (3) Termination history (correctly terminated, incorrectly terminated, completed naturally), (4) Resource savings (compute-hours saved, success rate). Update every 10 seconds via WebSocket.

### Integration with finite element workflow

**Pre-processing integration**: Before mesh generation, geometry analyzer extracts features (contact surface area, gap distribution, material property range). Feature vector queries the trained ML model for initial parameter recommendations. CAD plugin (Python API) presents suggestions directly in pre-processor GUI with confidence indicators. User accepts, modifies, or overrides recommendations—all choices logged for model improvement.

**Solver integration**: Modified Tribol/Smith input deck includes ML-suggested parameters with metadata (model version, confidence, alternatives). Optional: solver queries ML service at runtime for adaptive parameter adjustment (e.g., adaptive penalty stiffness based on contact evolution). Simulation output includes ML metadata for provenance tracking.

**Post-processing integration**: Analysis scripts automatically extract convergence metrics, parameter configurations, and simulation outcomes. Structured data pushed to central database via REST API. Database triggers model retraining pipeline when sufficient new data accumulated (e.g., 50 new samples). Updated models versioned and deployed automatically via CI/CD.

**Continuous learning loop**: Weekly cron job: (1) Fetch new simulation data from database, (2) Retrain GP/NN models on expanded dataset, (3) Validate on held-out test set, (4) If validation accuracy exceeds threshold, promote to production, (5) Generate and distribute updated model artifacts, (6) Notify users of model updates via email/Slack. Model versioning enables rollback if issues detected.

## Expected performance characteristics and deployment considerations

### Anticipated success rates and convergence behavior

**Initial performance** (0-50 trials): Random/Sobol initialization typically achieves 30-40% convergence rate for challenging contact problems with broad parameter ranges. FR-BO and GP Classification begin learning failure boundaries, success rate increases to 50-60% by trial 50. CONFIG remains conservative, 45-55% success but lower variance.

**Mid-term performance** (50-150 trials): ML models accurately predict feasibility. FR-BO achieves 75-85% success rate through failure-aware sampling. GP Classification reaches 80-90% via boundary refinement. CONFIG achieves 70-80% with strict violation bounds. SHEBO ensembles achieve 85-90% if sufficient training data.

**Mature performance** (>150 trials): Convergence landscape well-understood. All methods achieve >90% success rate. FR-BO finds near-optimal parameters fastest (50-70 samples to optimum). CONFIG provides formal guarantees of ε-optimality. SHEBO generalizes best to new geometries (transfer learning).

**Computational efficiency gains**: Versus grid search (exponential in dimensions), ML methods provide 50-100x speedup for 10+ parameters. Versus random search, 5-10x speedup. Early termination saves additional 30-40% compute time on failures. Total: 60-80% reduction in simulation costs.

### Scalability to production environments

**Multi-user scenarios**: Central database and model service support concurrent users. Parameter recommendations personalized via user history and geometry similarity. Caching common queries reduces latency to <50ms. Load balancing across model service replicas handles 100+ concurrent requests.

**Large parameter spaces** (>20 dimensions): Use dimensionality reduction (PCA, autoencoders, active subspaces) projecting to 10-15 dimensions. Train models in reduced space, maintain forward/inverse transformations. Alternative: Hierarchical optimization—coarse search over all parameters, fine search over critical subset.

**Multi-fidelity integration**: Maintain separate models for fast low-fidelity (coarse mesh, loose tolerance) and expensive high-fidelity. Sequential strategy: (1) Run low-fidelity prediction, (2) If P(converge) > 0.7, proceed to high-fidelity, (3) If uncertain (0.3 < P < 0.7), run medium-fidelity, (4) If P < 0.3, skip or suggest parameter modifications. Reduces high-fidelity evaluations by 60-70%.

**Ensemble model updates**: Asynchronous training allows staggered updates minimizing service disruption. Blue-green deployment: new model version deployed alongside old, gradual traffic shift, rollback if issues. Model ensembles enable individual model updates without full retraining—replace 1-2 ensemble members with retrained versions.

### Uncertainty quantification and confidence bounds

All four systems provide uncertainty estimates enabling **risk-aware decision making**:

**Prediction confidence**: GP posterior standard deviation quantifies prediction uncertainty. High σ indicates regions needing more data. Confidence intervals: 68% (±1σ), 95% (±2σ), 99% (±3σ). Present to users: "70% probability of convergence (95% CI: 60-80%)".

**Model confidence**: Ensemble disagreement (std of predictions across ensemble members) quantifies model uncertainty. Flags regions where models disagree fundamentally, suggesting complex physics or insufficient data. Trigger: if std > 0.3, recommend caution or additional sampling.

**Decision thresholds**: Calibrate probability thresholds to user risk tolerance. Conservative (P > 0.9, accept <50% of proposals but >98% converge), balanced (P > 0.7, accept ~70%, ~90% converge), aggressive (P > 0.5, accept >85%, ~75% converge). Allow users to select operating point on precision-recall curve.

**Conformal prediction**: Post-hoc calibration ensuring probabilistic predictions are well-calibrated. Compute calibration curves on validation set, apply temperature scaling or Platt scaling to adjust predicted probabilities. Result: if model predicts 70% convergence, actual success rate is 70% ± 3%.

This comprehensive system design provides four complementary approaches to resolving contact convergence issues through machine learning. Each method offers distinct advantages: FR-BO for rapid convergence despite failures, GP Classification for interpretable probabilistic reasoning, CONFIG for theoretical guarantees, and SHEBO for complex multi-constraint discovery. Starting with GP Classification and strategically expanding to FR-BO and SHEBO provides a pragmatic deployment path balancing immediate value with long-term scalability.