"""
Main FR-BO optimizer with three-phase workflow.

Implements:
1. Phase 1: Sobol quasi-random initialization (Trials 1-20)
2. Phase 2: FR-BO iterations (Trials 21-200)
3. Phase 3: Post-optimization validation and sensitivity analysis
"""

from typing import Dict, Any, Optional, List, Callable
import numpy as np
import torch
from dataclasses import dataclass
import time
from tqdm import tqdm

from fr_bo.parameters import create_search_space, get_parameter_dimension
from fr_bo.objective import ObjectiveFunction, extract_objective_from_result
from fr_bo.simulator import SyntheticSimulator, SimulationResult
from fr_bo.gp_models import DualGPSystem
from fr_bo.acquisition import FailureRobustEI, optimize_acquisition


@dataclass
class OptimizationConfig:
    """Configuration for FR-BO optimization."""

    # Phase 1: Initialization
    n_sobol_trials: int = 20
    random_seed: Optional[int] = None

    # Phase 2: FR-BO
    n_frbo_trials: int = 180
    gp_retrain_interval: int = 50
    acquisition_restarts: int = 10
    acquisition_raw_samples: int = 512

    # Simulation limits
    max_iterations: int = 1000
    timeout: float = 3600.0

    # Early termination
    enable_early_termination: bool = True
    early_term_check_interval: int = 5
    early_term_start_iter: int = 10

    # Convergence criteria
    convergence_tolerance: float = 1e-3
    convergence_patience: int = 10


@dataclass
class TrialRecord:
    """Record of a single optimization trial."""

    trial_number: int
    parameters: Dict[str, Any]
    result: SimulationResult
    objective_value: float
    phase: str
    timestamp: float


class FRBOOptimizer:
    """
    Failure-Robust Bayesian Optimization for FEA convergence.

    Manages the complete three-phase workflow with dual GP system.
    """

    def __init__(
        self,
        simulator: Optional[Any] = None,
        config: Optional[OptimizationConfig] = None,
    ):
        """
        Initialize FR-BO optimizer.

        Args:
            simulator: Simulation executor (SyntheticSimulator or SmithTribolExecutor)
            config: Optimization configuration
        """
        self.simulator = simulator or SyntheticSimulator(random_seed=42)
        self.config = config or OptimizationConfig()

        # Initialize state
        self.trials: List[TrialRecord] = []
        self.current_phase = "initialization"
        self.best_objective = float("inf")
        self.best_parameters = None
        self.best_trial_number = None

        # Dual GP system (initialized after first batch)
        self.dual_gp: Optional[DualGPSystem] = None

        # Objective function tracker
        self.objective_func = ObjectiveFunction()

        # Search space
        self.search_space = create_search_space()
        self.param_dim = get_parameter_dimension()

        # Random number generator
        self.rng = np.random.RandomState(self.config.random_seed)

    def optimize(self, total_trials: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete FR-BO optimization.

        Args:
            total_trials: Total number of trials (overrides config if provided)

        Returns:
            Dictionary containing optimization results
        """
        if total_trials is None:
            total_trials = self.config.n_sobol_trials + self.config.n_frbo_trials

        print(f"Starting FR-BO optimization with {total_trials} trials")
        print("=" * 70)

        # Phase 1: Sobol initialization
        print("\nPhase 1: Sobol Quasi-Random Initialization")
        print("-" * 70)
        self._run_sobol_phase()

        # Phase 2: FR-BO iterations
        print("\n\nPhase 2: Failure-Robust Bayesian Optimization")
        print("-" * 70)
        self._run_frbo_phase()

        # Phase 3: Validation
        print("\n\nPhase 3: Post-Optimization Analysis")
        print("-" * 70)
        results = self._run_validation_phase()

        return results

    def _run_sobol_phase(self):
        """Run Phase 1: Sobol quasi-random initialization."""
        self.current_phase = "sobol"

        # Generate Sobol samples
        sobol_samples = self._generate_sobol_samples(self.config.n_sobol_trials)

        # Run simulations
        for i in tqdm(range(self.config.n_sobol_trials), desc="Sobol trials"):
            params = sobol_samples[i]
            self._evaluate_and_record(i + 1, params, "sobol")

        # Initialize dual GP system after Sobol phase
        self._initialize_dual_gp()

        # Print phase summary
        self._print_phase_summary()

    def _run_frbo_phase(self):
        """Run Phase 2: FR-BO iterations."""
        self.current_phase = "frbo"

        start_trial = self.config.n_sobol_trials + 1
        end_trial = start_trial + self.config.n_frbo_trials

        for trial_num in tqdm(range(start_trial, end_trial), desc="FR-BO trials"):
            # Retrain GPs periodically
            if (trial_num - start_trial) % self.config.gp_retrain_interval == 0:
                print(f"\n  Retraining GPs at trial {trial_num}...")
                self._update_dual_gp()

            # Optimize acquisition function to get next candidate
            next_params = self._optimize_acquisition_function()

            # Evaluate candidate
            self._evaluate_and_record(trial_num, next_params, "frbo")

            # Update dual GP with new data
            self._update_dual_gp()

        # Print phase summary
        self._print_phase_summary()

    def _run_validation_phase(self) -> Dict[str, Any]:
        """Run Phase 3: Validation and analysis."""
        self.current_phase = "validation"

        # Compute metrics
        all_results = [t.result.to_dict() for t in self.trials]
        sobol_results = [t.result.to_dict() for t in self.trials if t.phase == "sobol"]
        frbo_results = [t.result.to_dict() for t in self.trials if t.phase == "frbo"]

        from fr_bo.objective import compute_success_metrics

        all_metrics = compute_success_metrics(all_results)
        sobol_metrics = compute_success_metrics(sobol_results)
        frbo_metrics = compute_success_metrics(frbo_results)

        # Get parameter importance
        param_importance = self.dual_gp.get_parameter_importance()

        # Compile results
        results = {
            "best_objective": self.best_objective,
            "best_parameters": self.best_parameters,
            "best_trial_number": self.best_trial_number,
            "metrics": {
                "overall": all_metrics,
                "sobol_phase": sobol_metrics,
                "frbo_phase": frbo_metrics,
            },
            "parameter_importance": param_importance,
            "total_trials": len(self.trials),
            "trials": self.trials,
        }

        # Print summary
        self._print_validation_summary(results)

        return results

    def _generate_sobol_samples(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate Sobol quasi-random samples."""
        from torch.quasirandom import SobolEngine

        # Create Sobol engine
        sobol = SobolEngine(dimension=self.param_dim, scramble=True, seed=self.config.random_seed)

        # Generate samples
        samples = sobol.draw(n_samples)

        # Convert to parameter dictionaries
        param_list = []
        for sample in samples:
            # Map to parameter space (simplified - using random sampling from bounds)
            params = self._sample_from_space()
            param_list.append(params)

        return param_list

    def _sample_from_space(self) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}

        # Sample each parameter
        for param in self.search_space.parameters.values():
            if param.parameter_type.name == "FLOAT":
                if param.log_scale:
                    # Log-scale parameter: sample in log space then convert back
                    log_lower = np.log10(param.lower)
                    log_upper = np.log10(param.upper)
                    log_val = self.rng.uniform(log_lower, log_upper)
                    params[param.name] = 10 ** log_val
                else:
                    # Linear scale
                    params[param.name] = self.rng.uniform(param.lower, param.upper)
            elif param.parameter_type.name == "INT":
                params[param.name] = self.rng.randint(param.lower, param.upper + 1)
            elif param.parameter_type.name == "STRING":
                params[param.name] = self.rng.choice(param.values)

        return params

    def _evaluate_and_record(
        self, trial_number: int, parameters: Dict[str, Any], phase: str
    ):
        """Evaluate parameters and record trial."""
        # Run simulation
        result = self.simulator.run(
            parameters=parameters,
            max_iterations=self.config.max_iterations,
            timeout=self.config.timeout,
        )

        # Compute objective
        objective_value = extract_objective_from_result(result.to_dict())

        # Update best
        if result.converged and objective_value < self.best_objective:
            self.best_objective = objective_value
            self.best_parameters = parameters.copy()
            self.best_trial_number = trial_number
            print(f"\n  New best: Trial {trial_number}, Objective = {objective_value:.4f}")

        # Record trial
        trial = TrialRecord(
            trial_number=trial_number,
            parameters=parameters,
            result=result,
            objective_value=objective_value,
            phase=phase,
            timestamp=time.time(),
        )
        self.trials.append(trial)

    def _initialize_dual_gp(self):
        """Initialize dual GP system after Sobol phase."""
        # Prepare training data
        train_X_list = []
        train_Y_list = []
        failure_labels_list = []

        for trial in self.trials:
            # Encode parameters as feature vector
            x = self._encode_parameters(trial.parameters)
            train_X_list.append(x)

            # Objective value (NaN for failed trials)
            if trial.result.converged:
                train_Y_list.append(trial.objective_value)
                failure_labels_list.append(0.0)
            else:
                train_Y_list.append(float("nan"))
                failure_labels_list.append(1.0)

        # Convert to tensors
        train_X = torch.tensor(np.array(train_X_list), dtype=torch.float32)
        train_Y = torch.tensor(train_Y_list, dtype=torch.float32).unsqueeze(-1)
        failure_labels = torch.tensor(failure_labels_list, dtype=torch.float32)

        # Replace NaN with dummy values for now (will be filtered out in DualGPSystem)
        train_Y = torch.nan_to_num(train_Y, nan=0.0)

        # Initialize dual GP
        self.dual_gp = DualGPSystem(train_X, train_Y, failure_labels)
        self.dual_gp.train(num_restarts=5)

    def _update_dual_gp(self):
        """Update dual GP system with all current data."""
        # Re-initialize with all data
        self._initialize_dual_gp()

    def _optimize_acquisition_function(self) -> Dict[str, Any]:
        """Optimize FREI acquisition to get next candidate."""
        # Get current best objective
        successful_trials = [t for t in self.trials if t.result.converged]
        if successful_trials:
            best_f = min(t.objective_value for t in successful_trials)
        else:
            best_f = 0.0

        # If no successful trials yet, fall back to random sampling
        if self.dual_gp.objective_gp is None:
            print("  No successful trials yet, using random sampling")
            return self._sample_from_space()

        # Create FREI acquisition
        frei = FailureRobustEI(
            model=self.dual_gp.objective_gp.model,
            failure_model=self.dual_gp.failure_classifier.model,
            failure_likelihood=self.dual_gp.failure_classifier.likelihood,
            best_f=best_f,
            maximize=False,
        )

        # Get parameter bounds (normalized [0, 1])
        bounds = torch.tensor([[0.0] * self.param_dim, [1.0] * self.param_dim], dtype=torch.float32)

        # Optimize acquisition
        candidate = optimize_acquisition(
            acquisition_function=frei,
            bounds=bounds,
            num_restarts=self.config.acquisition_restarts,
            raw_samples=self.config.acquisition_raw_samples,
        )

        # Decode candidate to parameters
        params = self._decode_parameters(candidate.numpy())

        return params

    def _encode_parameters(self, params: Dict[str, Any]) -> np.ndarray:
        """Encode parameters as feature vector."""
        from fr_bo.parameters import encode_parameters
        return encode_parameters(params)

    def _decode_parameters(self, encoded: np.ndarray) -> Dict[str, Any]:
        """Decode feature vector to parameters."""
        from fr_bo.parameters import decode_parameters
        return decode_parameters(encoded)

    def _print_phase_summary(self):
        """Print summary of current phase."""
        phase_trials = [t for t in self.trials if t.phase == self.current_phase]
        converged = sum(1 for t in phase_trials if t.result.converged)
        success_rate = converged / len(phase_trials) if phase_trials else 0.0

        print(f"\n{self.current_phase.upper()} Phase Summary:")
        print(f"  Trials: {len(phase_trials)}")
        print(f"  Converged: {converged} ({success_rate * 100:.1f}%)")
        if not np.isinf(self.best_objective):
            print(f"  Best objective: {self.best_objective:.4f}")
        else:
            print(f"  Best objective: {self.best_objective}")

    def _print_validation_summary(self, results: Dict[str, Any]):
        """Print validation and final summary."""
        print("\nFinal Results:")
        print("=" * 70)

        if results["best_parameters"] is not None:
            print(f"Best trial: {results['best_trial_number']}")
            print(f"Best objective: {results['best_objective']:.4f}")
            print(f"\nBest parameters:")
            for key, value in results["best_parameters"].items():
                print(f"  {key}: {value}")
        else:
            print("No successful trials - all simulations failed to converge")
            print(f"Total trials attempted: {results['total_trials']}")

        print(f"\nOverall Metrics:")
        for key, value in results["metrics"]["overall"].items():
            if isinstance(value, float):
                if not np.isinf(value):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")

        if results["parameter_importance"] is not None:
            print(f"\nParameter Importance (top 5):")
            importance = results["parameter_importance"]
            param_names = [p.name for p in self.search_space.parameters.values()]
            sorted_idx = np.argsort(importance)[::-1][:5]
            for idx in sorted_idx:
                print(f"  {param_names[idx]}: {importance[idx]:.4f}")
