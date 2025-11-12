"""Main SHEBO optimizer orchestrating the complete workflow."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass, field

from shebo.core.surrogate_manager import SurrogateManager
from shebo.core.constraint_discovery import ConstraintDiscovery
from shebo.core.acquisition import AdaptiveAcquisition


@dataclass
class SHEBOResult:
    """Result from SHEBO optimization."""
    best_params: np.ndarray
    best_performance: float
    convergence_history: List[bool] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    discovered_constraints: Dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    all_params: List[np.ndarray] = field(default_factory=list)
    all_outputs: List[Dict[str, Any]] = field(default_factory=list)


class SHEBOOptimizer:
    """SHEBO (Surrogate Optimization with Hidden Constraints) optimizer.

    Combines ensemble surrogate modeling with constraint discovery for
    robust optimization with unknown constraints.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        objective_fn: Callable,
        n_init: int = 20,
        budget: int = 200,
        n_networks: int = 5,
        convergence_update_freq: int = 10,
        performance_update_freq: int = 10,
        constraint_update_freq: int = 50,
        checkpoint_dir: Optional[str] = None,
        checkpoint_frequency: int = 10,
        random_seed: Optional[int] = None
    ):
        """Initialize SHEBO optimizer.

        Args:
            bounds: Parameter bounds of shape (n_params, 2)
            objective_fn: Objective function that takes parameters and returns
                         dictionary with 'output' (simulation results) and
                         'performance' (metric to minimize)
            n_init: Number of initial space-filling samples
            budget: Total evaluation budget
            n_networks: Number of networks in each ensemble
            convergence_update_freq: Update frequency for convergence model
            performance_update_freq: Update frequency for performance model
            constraint_update_freq: Update frequency for constraint models
            checkpoint_dir: Directory to save checkpoints (None to disable)
            checkpoint_frequency: Save checkpoint every N iterations
            random_seed: Random seed for reproducibility
        """
        self.bounds = bounds
        self.objective_fn = objective_fn
        self.n_init = n_init
        self.budget = budget
        self.random_seed = random_seed
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        # Get dimensionality
        self.n_params = bounds.shape[0]

        # Initialize components
        self.surrogate_manager = SurrogateManager(
            input_dim=self.n_params,
            n_networks=n_networks,
            convergence_update_freq=convergence_update_freq,
            performance_update_freq=performance_update_freq,
            constraint_update_freq=constraint_update_freq
        )

        self.constraint_discovery = ConstraintDiscovery()

        self.adaptive_acquisition = AdaptiveAcquisition(
            surrogate_manager=self.surrogate_manager,
            bounds=bounds
        )

        # Data storage
        self.all_params: List[np.ndarray] = []
        self.all_outputs: List[Dict[str, Any]] = []
        self.convergence_status: List[bool] = []
        self.performance_values: List[float] = []

        self.iteration = 0
        self.best_params: Optional[np.ndarray] = None
        self.best_performance = float('inf')

    def run(self) -> SHEBOResult:
        """Run SHEBO optimization.

        Returns:
            SHEBOResult containing optimization results
        """
        print("=" * 60)
        print("SHEBO Optimization Starting")
        print("=" * 60)
        print(f"Budget: {self.budget} evaluations")
        print(f"Initial samples: {self.n_init}")
        print(f"Parameter space dimension: {self.n_params}")
        print()

        # Phase 1: Space-filling initialization
        print("Phase 1: Space-filling initialization")
        self._initialize()

        # Train initial surrogates
        print("\nTraining initial surrogate models...")
        self._update_surrogates()

        # Phase 2: Main optimization loop
        print("\nPhase 2: Adaptive optimization")
        print("-" * 60)

        while self.iteration < self.budget:
            self.iteration += 1

            # Determine optimization phase
            phase = self._determine_phase()

            # Discover constraints from recent failures
            self._discover_constraints()

            # Update surrogates (based on schedules)
            self._update_surrogates()

            # Optimize acquisition to get next point
            next_x = self._get_next_point(phase)

            # Evaluate
            self._evaluate_and_store(next_x)

            # Print progress
            if self.iteration % 10 == 0 or self.iteration == self.budget:
                self._print_progress()

            # Save checkpoint periodically
            if (self.checkpoint_dir is not None and
                self.iteration % self.checkpoint_frequency == 0):
                self.save_checkpoint()

            # Check termination criteria
            if self._check_termination():
                print(f"\nEarly termination at iteration {self.iteration}")
                # Save final checkpoint
                if self.checkpoint_dir is not None:
                    self.save_checkpoint()
                break

        # Final summary
        self._print_final_summary()

        return self._create_result()

    def _initialize(self) -> None:
        """Initialize with space-filling design (Sobol sequence)."""
        from scipy.stats import qmc

        # Generate Sobol samples
        sampler = qmc.Sobol(d=self.n_params, scramble=True, seed=self.random_seed)
        samples = sampler.random(n=self.n_init)

        # Scale to bounds
        l_bounds = self.bounds[:, 0]
        u_bounds = self.bounds[:, 1]
        samples_scaled = qmc.scale(samples, l_bounds, u_bounds)

        # Evaluate all initial samples
        for i, x in enumerate(samples_scaled):
            print(f"  Initializing {i+1}/{self.n_init}", end='\r')
            self._evaluate_and_store(x)

        print(f"  Initialization complete: {self.n_init} samples evaluated")

    def _evaluate_and_store(self, x: np.ndarray) -> None:
        """Evaluate objective function and store results.

        Args:
            x: Parameter vector to evaluate
        """
        # Evaluate objective
        result = self.objective_fn(x)

        output = result['output']
        performance = result.get('performance', float('inf'))
        converged = output.get('convergence_status', False)

        # Store data
        self.all_params.append(x)
        self.all_outputs.append(output)
        self.convergence_status.append(converged)
        self.performance_values.append(performance)

        # Update best
        if converged and performance < self.best_performance:
            self.best_performance = performance
            self.best_params = x
            self.adaptive_acquisition.update_best_performance(performance)

    def _discover_constraints(self) -> None:
        """Discover constraints from recent simulations."""
        # Check last 10 samples for constraint violations
        recent_start = max(0, len(self.all_outputs) - 10)
        recent_outputs = self.all_outputs[recent_start:]

        for idx, output in enumerate(recent_outputs):
            violations = self.constraint_discovery.check_simulation_output(output)
            if violations:
                actual_iter = recent_start + idx
                self.constraint_discovery.update_discovered_constraints(
                    violations,
                    actual_iter,
                    self.surrogate_manager
                )

    def _update_surrogates(self) -> None:
        """Update surrogate models with all available data."""
        if len(self.all_params) < 5:  # Need minimum data
            return

        # Convert to tensors
        X = torch.tensor(np.array(self.all_params), dtype=torch.float32)
        y_convergence = torch.tensor(
            np.array(self.convergence_status),
            dtype=torch.float32
        ).reshape(-1, 1)

        # Performance data (only for successful runs)
        y_performance = None
        if any(self.convergence_status):
            # Log transform for stability
            perf_array = np.array(self.performance_values)
            # Single output: log-transformed performance metric
            perf_log = np.log1p(perf_array).reshape(-1, 1)
            y_performance = torch.tensor(perf_log, dtype=torch.float32)

        # Constraint labels
        y_constraints = None
        if self.constraint_discovery.discovered_constraints:
            constraint_labels = self.constraint_discovery.get_constraint_labels(
                self.all_outputs
            )
            y_constraints = {
                name: torch.tensor(labels, dtype=torch.float32)
                for name, labels in constraint_labels.items()
            }

        # Update models
        self.surrogate_manager.update_models(
            X,
            y_convergence,
            y_performance,
            y_constraints,
            current_iteration=self.iteration
        )

    def _determine_phase(self) -> str:
        """Determine optimization phase based on progress.

        Returns:
            Phase name
        """
        if self.iteration < 30:
            return 'exploration'
        elif self.iteration < 100:
            return 'boundary_learning'
        else:
            return 'exploitation'

    def _get_next_point(self, phase: str) -> np.ndarray:
        """Get next point by optimizing acquisition function.

        Args:
            phase: Optimization phase

        Returns:
            Next parameter vector to evaluate
        """
        next_x, acq_value = self.adaptive_acquisition.optimize(
            phase=phase,
            n_restarts=10
        )
        return next_x

    def _get_next_batch(self, phase: str, batch_size: int) -> np.ndarray:
        """Get batch of diverse points for parallel evaluation.

        Args:
            phase: Optimization phase
            batch_size: Number of points to select

        Returns:
            Array of shape (batch_size, n_params)
        """
        return self.adaptive_acquisition.select_batch(
            phase=phase,
            batch_size=batch_size,
            diversity_weight=0.5
        )

    def evaluate_batch(
        self,
        params_batch: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of parameter vectors.

        This method can be overridden to implement parallel evaluation.
        By default, evaluates sequentially.

        Args:
            params_batch: List of parameter vectors

        Returns:
            List of results from objective function
        """
        results = []
        for params in params_batch:
            result = self.objective_fn(params)
            results.append(result)
        return results

    def run_batch(
        self,
        batch_size: int = 5,
        parallel: bool = False
    ) -> SHEBOResult:
        """Run SHEBO optimization with batch evaluation.

        Args:
            batch_size: Number of points to evaluate in parallel
            parallel: Whether to use parallel evaluation (requires override)

        Returns:
            SHEBOResult containing optimization results
        """
        print("=" * 60)
        print("SHEBO Batch Optimization Starting")
        print("=" * 60)
        print(f"Budget: {self.budget} evaluations")
        print(f"Initial samples: {self.n_init}")
        print(f"Batch size: {batch_size}")
        print(f"Parameter space dimension: {self.n_params}")
        print()

        # Phase 1: Space-filling initialization
        print("Phase 1: Space-filling initialization")
        self._initialize()

        # Train initial surrogates
        print("\nTraining initial surrogate models...")
        self._update_surrogates()

        # Phase 2: Main optimization loop with batches
        print("\nPhase 2: Adaptive batch optimization")
        print("-" * 60)

        while self.iteration < self.budget:
            # Determine optimization phase
            phase = self._determine_phase()

            # Discover constraints from recent failures
            self._discover_constraints()

            # Update surrogates (based on schedules)
            self._update_surrogates()

            # Get batch of points
            batch_params = self._get_next_batch(phase, batch_size)

            # Evaluate batch
            batch_results = self.evaluate_batch(batch_params)

            # Store all results
            for params, result in zip(batch_params, batch_results):
                self.iteration += 1
                output = result['output']
                performance = result.get('performance', float('inf'))
                converged = output.get('convergence_status', False)

                self.all_params.append(params)
                self.all_outputs.append(output)
                self.convergence_status.append(converged)
                self.performance_values.append(performance)

                # Update best
                if converged and performance < self.best_performance:
                    self.best_performance = performance
                    self.best_params = params
                    self.adaptive_acquisition.update_best_performance(performance)

                if self.iteration >= self.budget:
                    break

            # Print progress
            if self.iteration % 10 == 0 or self.iteration >= self.budget:
                self._print_progress()

            # Save checkpoint periodically
            if (self.checkpoint_dir is not None and
                self.iteration % self.checkpoint_frequency == 0):
                self.save_checkpoint()

            # Check termination criteria
            if self._check_termination():
                print(f"\nEarly termination at iteration {self.iteration}")
                if self.checkpoint_dir is not None:
                    self.save_checkpoint()
                break

        # Final summary
        self._print_final_summary()

        return self._create_result()

    def _check_termination(self) -> bool:
        """Check termination criteria.

        Returns:
            True if should terminate
        """
        # Budget exhausted
        if self.iteration >= self.budget:
            return True

        # Need minimum iterations
        if self.iteration < 100:
            return False

        # No new constraints in last 30 iterations
        no_new_constraints = not self.constraint_discovery.has_new_constraints_since(
            self.iteration - 30
        )

        # Best solution stable (no improvement in last 15 iterations)
        best_stable = self._check_best_stable(window=15)

        return no_new_constraints and best_stable

    def _check_best_stable(self, window: int = 15) -> bool:
        """Check if best solution is stable.

        Args:
            window: Window size to check

        Returns:
            True if best hasn't improved in window iterations
        """
        if self.iteration < window:
            return False

        # Check if there's been improvement in last window iterations
        recent_perfs = self.performance_values[-window:]
        recent_converged = self.convergence_status[-window:]

        for perf, converged in zip(recent_perfs, recent_converged):
            if converged and perf < self.best_performance:
                return False

        return True

    def _print_progress(self) -> None:
        """Print progress update."""
        n_converged = sum(self.convergence_status)
        convergence_rate = n_converged / len(self.convergence_status) * 100

        phase = self._determine_phase()

        print(f"Iteration {self.iteration}/{self.budget} | "
              f"Phase: {phase:20s} | "
              f"Converged: {n_converged}/{len(self.convergence_status)} ({convergence_rate:.1f}%) | "
              f"Best: {self.best_performance:.4f} | "
              f"Constraints: {len(self.constraint_discovery.discovered_constraints)}")

    def _print_final_summary(self) -> None:
        """Print final optimization summary."""
        print("\n" + "=" * 60)
        print("SHEBO Optimization Complete")
        print("=" * 60)
        print(f"Total evaluations: {len(self.all_params)}")
        print(f"Successful convergences: {sum(self.convergence_status)}")
        print(f"Convergence rate: {sum(self.convergence_status)/len(self.convergence_status)*100:.1f}%")
        print(f"Best performance: {self.best_performance:.6f}")
        print(f"Discovered constraints: {len(self.constraint_discovery.discovered_constraints)}")

        if self.constraint_discovery.discovered_constraints:
            print("\nDiscovered Constraint Summary:")
            for name, info in self.constraint_discovery.discovered_constraints.items():
                print(f"  - {name:30s}: "
                      f"frequency={info['frequency']:3d}, "
                      f"severity={info['severity']}")

        print("\nBest parameters:")
        if self.best_params is not None:
            for i, val in enumerate(self.best_params):
                print(f"  param_{i}: {val:.6f}")
        print("=" * 60)

    def _create_result(self) -> SHEBOResult:
        """Create result object.

        Returns:
            SHEBOResult
        """
        return SHEBOResult(
            best_params=self.best_params if self.best_params is not None else np.array([]),
            best_performance=self.best_performance,
            convergence_history=self.convergence_status.copy(),
            performance_history=self.performance_values.copy(),
            discovered_constraints=self.constraint_discovery.get_summary(),
            iterations=self.iteration,
            all_params=self.all_params.copy(),
            all_outputs=self.all_outputs.copy()
        )

    def save_checkpoint(self, filepath: Optional[str] = None) -> None:
        """Save optimization checkpoint.

        Args:
            filepath: Path to save checkpoint (None to use default)
        """
        import os
        import pickle

        if self.checkpoint_dir is None and filepath is None:
            return  # Checkpointing disabled

        if filepath is None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            filepath = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_iter_{self.iteration}.pkl'
            )

        checkpoint = {
            'iteration': self.iteration,
            'all_params': self.all_params,
            'all_outputs': self.all_outputs,
            'convergence_status': self.convergence_status,
            'performance_values': self.performance_values,
            'best_params': self.best_params,
            'best_performance': self.best_performance,
            'discovered_constraints': self.constraint_discovery.discovered_constraints,
            'random_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Checkpoint saved: {filepath}")

        # Also save models
        model_dir = os.path.join(os.path.dirname(filepath), 'models')
        self.surrogate_manager.save_models(model_dir)

    def load_checkpoint(self, filepath: str) -> None:
        """Load optimization checkpoint and resume.

        Args:
            filepath: Path to checkpoint file
        """
        import os
        import pickle

        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        self.iteration = checkpoint['iteration']
        self.all_params = checkpoint['all_params']
        self.all_outputs = checkpoint['all_outputs']
        self.convergence_status = checkpoint['convergence_status']
        self.performance_values = checkpoint['performance_values']
        self.best_params = checkpoint['best_params']
        self.best_performance = checkpoint['best_performance']
        self.constraint_discovery.discovered_constraints = checkpoint['discovered_constraints']

        # Restore random states
        np.random.set_state(checkpoint['random_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])

        # Load models
        model_dir = os.path.join(os.path.dirname(filepath), 'models')
        if os.path.exists(model_dir):
            self.surrogate_manager.load_models(model_dir)

        print(f"Checkpoint loaded: {filepath}")
        print(f"Resuming from iteration {self.iteration}")
