"""
Main GP Classification optimizer with three-phase exploration strategy.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import BernoulliLikelihood
from torch.quasirandom import SobolEngine

from .acquisition import AdaptiveAcquisition, optimize_acquisition
from .data import TrialDatabase
from .models import DualModel, VariationalGPClassifier


class GPClassificationOptimizer:
    """
    GP Classification-based optimizer for contact convergence problems.

    Implements three-phase strategy:
    - Phase 1 (iter 1-20): Entropy-based exploration for boundary discovery
    - Phase 2 (iter 21-50): Boundary refinement near P(converge)=0.5
    - Phase 3 (iter 51+): Constrained exploitation with CEI

    Features:
    - Variational GP classification for convergence prediction
    - Dual model architecture (convergence + objective)
    - Adaptive acquisition function switching
    - Automatic hyperparameter optimization
    """

    def __init__(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        simulator: Callable[[Dict[str, float]], Tuple[bool, Optional[float]]],
        n_initial_samples: int = 20,
        phase1_end: int = 20,
        phase2_end: int = 50,
        n_inducing_points: int = 100,
        verbose: bool = True,
    ):
        """
        Initialize GP Classification optimizer.

        Args:
            parameter_bounds: Dictionary mapping parameter names to (min, max) bounds
            simulator: Function that takes parameters and returns (converged, objective_value)
            n_initial_samples: Number of Sobol samples for initialization
            phase1_end: Last iteration of Phase 1 (exploration)
            phase2_end: Last iteration of Phase 2 (boundary refinement)
            n_inducing_points: Number of inducing points for variational GP
            verbose: Whether to print progress
        """
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())
        self.simulator = simulator
        self.n_initial_samples = n_initial_samples
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.n_inducing_points = n_inducing_points
        self.verbose = verbose

        # Initialize database
        self.database = TrialDatabase(parameter_bounds)

        # Models (initialized after initial sampling)
        self.dual_model: Optional[DualModel] = None

        # Current iteration
        self.iteration = 0

        # Bounds tensor for optimization
        bounds_list = [parameter_bounds[name] for name in self.parameter_names]
        self.bounds_tensor = torch.tensor(bounds_list, dtype=torch.float64).T

    def _parameters_to_tensor(self, parameters: Dict[str, float]) -> torch.Tensor:
        """Convert parameter dict to tensor."""
        return torch.tensor(
            [parameters[name] for name in self.parameter_names], dtype=torch.float64
        )

    def _tensor_to_parameters(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Convert tensor to parameter dict."""
        return {name: float(tensor[i]) for i, name in enumerate(self.parameter_names)}

    def _generate_sobol_samples(self, n_samples: int) -> torch.Tensor:
        """Generate Sobol quasi-random samples within bounds."""
        dim = len(self.parameter_names)
        sobol = SobolEngine(dimension=dim, scramble=True)
        samples = sobol.draw(n_samples).to(dtype=torch.float64)

        # Scale to bounds
        lower = self.bounds_tensor[0]
        upper = self.bounds_tensor[1]
        samples = lower + (upper - lower) * samples

        return samples

    def _evaluate_simulator(self, parameters: Dict[str, float]) -> Tuple[bool, Optional[float]]:
        """
        Evaluate simulator and handle errors.

        Args:
            parameters: Parameter dictionary

        Returns:
            (converged, objective_value)
        """
        try:
            converged, objective_value = self.simulator(parameters)
            return converged, objective_value
        except Exception as e:
            if self.verbose:
                print(f"Simulation error: {e}")
            return False, None

    def _initialize_with_sobol(self) -> None:
        """Initialize database with Sobol samples."""
        if self.verbose:
            print(f"\n=== Phase 1: Initial Exploration ===")
            print(f"Generating {self.n_initial_samples} Sobol samples...")

        samples = self._generate_sobol_samples(self.n_initial_samples)

        for i, sample in enumerate(samples):
            parameters = self._tensor_to_parameters(sample)
            converged, objective = self._evaluate_simulator(parameters)

            self.database.add_trial(
                parameters=parameters,
                converged=converged,
                objective_value=objective,
            )

            self.iteration += 1

            if self.verbose and (i + 1) % 5 == 0:
                stats = self.database.get_statistics()
                print(
                    f"  Sample {i + 1}/{self.n_initial_samples}: "
                    f"Convergence rate = {stats['convergence_rate']:.1%}"
                )

    def _update_models(self) -> None:
        """Update GP models with latest data."""
        # Get all trials for convergence classifier
        X_all, y_converged, _ = self.database.get_training_data(converged_only=False)

        # Get successful trials for objective model
        try:
            X_success, _, y_objective = self.database.get_training_data(converged_only=True)
        except ValueError:
            # No successful trials yet
            X_success = None
            y_objective = None

        # Initialize or update dual model
        if self.dual_model is None:
            self.dual_model = DualModel(
                train_X_all=X_all,
                train_Y_converged=y_converged,
                train_X_success=X_success,
                train_Y_objective=y_objective,
                n_inducing_points=self.n_inducing_points,
            )
        else:
            # Update with new data
            self.dual_model = DualModel(
                train_X_all=X_all,
                train_Y_converged=y_converged,
                train_X_success=X_success,
                train_Y_objective=y_objective,
                n_inducing_points=self.n_inducing_points,
            )

        # Train models
        self.dual_model.train_models(
            train_X_all=X_all,
            train_Y_converged=y_converged,
            n_epochs=300,
            learning_rate=0.05,
            verbose=False,
        )

    def _get_next_candidate(self) -> torch.Tensor:
        """Get next candidate point using adaptive acquisition."""
        # Create convergence predictor function (with gradients for optimization)
        def convergence_predictor(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.dual_model.predict_convergence_with_grad(X)

        # Get best observed value
        best_trial = self.database.get_best_trial()
        if best_trial is not None and best_trial.objective_value is not None:
            best_f = torch.tensor(best_trial.objective_value, dtype=torch.float64)
            model = self.dual_model.objective_model
        else:
            best_f = None
            model = None

        # Create adaptive acquisition function
        acq_fn = AdaptiveAcquisition(
            model=model,
            best_f=best_f,
            convergence_predictor=convergence_predictor,
            current_iteration=self.iteration,
            phase1_end=self.phase1_end,
            phase2_end=self.phase2_end,
        )

        # Optimize acquisition function
        candidate = optimize_acquisition(
            acquisition_fn=acq_fn,
            bounds=self.bounds_tensor,
            num_restarts=10,
            raw_samples=512,
        )

        return candidate

    def optimize(self, n_iterations: int = 100) -> Dict[str, float]:
        """
        Run optimization loop.

        Args:
            n_iterations: Total number of iterations (including initial samples)

        Returns:
            Best parameters found
        """
        # Initial sampling
        if self.iteration == 0:
            self._initialize_with_sobol()

        # Main optimization loop
        while self.iteration < n_iterations:
            # Update models
            self._update_models()

            # Determine phase
            if self.iteration <= self.phase1_end:
                phase = "Exploration"
            elif self.iteration <= self.phase2_end:
                phase = "Boundary Refinement"
            else:
                phase = "Exploitation"

            if self.verbose:
                if self.iteration == self.phase1_end + 1:
                    print(f"\n=== Phase 2: Boundary Refinement ===")
                elif self.iteration == self.phase2_end + 1:
                    print(f"\n=== Phase 3: Exploitation ===")

            # Get next candidate
            candidate = self._get_next_candidate()
            parameters = self._tensor_to_parameters(candidate)

            # Evaluate
            converged, objective = self._evaluate_simulator(parameters)

            # Add to database
            self.database.add_trial(
                parameters=parameters,
                converged=converged,
                objective_value=objective,
            )

            self.iteration += 1

            # Print progress
            if self.verbose:
                stats = self.database.get_statistics()
                best = stats.get("best_objective", float("inf"))
                prob, _ = self.dual_model.predict_convergence(candidate.unsqueeze(0))

                print(
                    f"Iter {self.iteration:3d} [{phase:20s}]: "
                    f"Converged={converged}, "
                    f"P(conv)={prob.item():.3f}, "
                    f"Best={best:.4f}, "
                    f"Rate={stats['convergence_rate']:.1%}"
                )

        # Return best parameters
        best_trial = self.database.get_best_trial()
        if best_trial is None:
            raise ValueError("No converged trials found!")

        if self.verbose:
            print(f"\n=== Optimization Complete ===")
            print(f"Best objective: {best_trial.objective_value:.6f}")
            print(f"Best parameters:")
            for name, value in best_trial.parameters.items():
                print(f"  {name}: {value:.6f}")

        return best_trial.parameters

    def predict_convergence_landscape(
        self, param1_name: str, param2_name: str, resolution: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate 2D convergence probability landscape.

        Args:
            param1_name: First parameter name (x-axis)
            param2_name: Second parameter name (y-axis)
            resolution: Grid resolution

        Returns:
            X: Grid for param1 [resolution, resolution]
            Y: Grid for param2 [resolution, resolution]
            P: Convergence probabilities [resolution, resolution]
        """
        if self.dual_model is None:
            raise ValueError("Model not trained yet")

        # Get bounds
        bounds1 = self.parameter_bounds[param1_name]
        bounds2 = self.parameter_bounds[param2_name]

        # Create grid
        x = torch.linspace(bounds1[0], bounds1[1], resolution, dtype=torch.float64)
        y = torch.linspace(bounds2[0], bounds2[1], resolution, dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing="xy")

        # Get indices of these parameters
        idx1 = self.parameter_names.index(param1_name)
        idx2 = self.parameter_names.index(param2_name)

        # Create full parameter tensors (fix other parameters at their mid-range)
        grid_points = []
        for i in range(resolution):
            for j in range(resolution):
                # Start with mid-range values for all parameters
                params = torch.zeros(len(self.parameter_names), dtype=torch.float64)
                for k, name in enumerate(self.parameter_names):
                    lower, upper = self.parameter_bounds[name]
                    params[k] = (lower + upper) / 2.0

                # Set the two parameters we're varying
                params[idx1] = X[i, j]
                params[idx2] = Y[i, j]

                grid_points.append(params)

        grid_tensor = torch.stack(grid_points)

        # Predict convergence probabilities
        probs, _ = self.dual_model.predict_convergence(grid_tensor)
        P = probs.reshape(resolution, resolution)

        return X, Y, P

    def get_statistics(self) -> Dict:
        """Get comprehensive optimization statistics."""
        stats = self.database.get_statistics()
        stats["iteration"] = self.iteration
        stats["phase"] = (
            "exploration"
            if self.iteration <= self.phase1_end
            else "boundary_refinement"
            if self.iteration <= self.phase2_end
            else "exploitation"
        )

        return stats
