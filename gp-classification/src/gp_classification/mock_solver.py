"""
Mock Smith solver for testing without actual FEA solver.

Simulates contact convergence behavior based on realistic physics-inspired models.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class MockSmithSolver:
    """
    Black-box simulator that mimics Smith/Tribol contact solver behavior.

    Implements:
    - Realistic convergence boundaries based on penalty stiffness and tolerances
    - Objective function (iteration count) for converged simulations
    - Stochastic failures near boundaries
    - Physics-inspired parameter interactions
    """

    def __init__(
        self,
        random_seed: Optional[int] = None,
        noise_level: float = 0.1,
        difficulty: str = "medium",
    ):
        """
        Initialize mock solver.

        Args:
            random_seed: Random seed for reproducibility
            noise_level: Amount of stochastic noise in convergence decisions
            difficulty: Problem difficulty ('easy', 'medium', 'hard')
        """
        self.rng = np.random.RandomState(random_seed)
        self.noise_level = noise_level
        self.difficulty = difficulty

        # Define optimal regions based on difficulty
        if difficulty == "easy":
            self.optimal_penalty = 1e5
            self.optimal_tolerance = 1e-7
            self.convergence_margin = 0.5  # Wide convergence region
        elif difficulty == "medium":
            self.optimal_penalty = 5e5
            self.optimal_tolerance = 5e-8
            self.convergence_margin = 0.3
        else:  # hard
            self.optimal_penalty = 1e6
            self.optimal_tolerance = 1e-8
            self.convergence_margin = 0.15  # Narrow convergence region

    def simulate(self, parameters: Dict[str, float]) -> Tuple[bool, Optional[float]]:
        """
        Simulate contact convergence for given parameters.

        Args:
            parameters: Dictionary of solver parameters

        Returns:
            (converged, objective_value)
            - converged: True if simulation converged
            - objective_value: Iteration count (lower is better) if converged, None if failed
        """
        # Extract key parameters (with defaults)
        penalty = parameters.get("penalty_stiffness", 1e5)
        tolerance = parameters.get("gap_tolerance", 1e-7)
        abs_tol = parameters.get("absolute_tolerance", 1e-10)
        rel_tol = parameters.get("relative_tolerance", 1e-8)
        max_iter = parameters.get("max_iterations", 50)
        timestep = parameters.get("timestep", 1e-4)

        # Compute convergence probability based on parameter distances from optimal
        prob_converge = self._compute_convergence_probability(
            penalty, tolerance, abs_tol, rel_tol, timestep
        )

        # Add stochastic noise
        noise = self.rng.randn() * self.noise_level
        prob_converge = np.clip(prob_converge + noise, 0, 1)

        # Determine convergence
        converged = self.rng.rand() < prob_converge

        if not converged:
            return False, None

        # Compute objective (iteration count)
        # Better parameters -> fewer iterations
        base_iterations = self._compute_base_iterations(
            penalty, tolerance, abs_tol, rel_tol, timestep
        )

        # Add some noise
        iteration_noise = self.rng.randn() * 2.0
        iterations = max(5, min(max_iter, base_iterations + iteration_noise))

        return True, iterations

    def _compute_convergence_probability(
        self,
        penalty: float,
        tolerance: float,
        abs_tol: float,
        rel_tol: float,
        timestep: float,
    ) -> float:
        """
        Compute convergence probability based on parameter values.

        Physics-inspired model:
        - Penalty too low -> penetration, failure
        - Penalty too high -> ill-conditioning, failure
        - Tolerance too tight -> excessive iterations, potential failure
        - Tolerance too loose -> accuracy issues
        - Timestep affects stability
        """
        # Penalty distance (log scale)
        penalty_score = 1.0 - min(
            1.0, abs(np.log10(penalty) - np.log10(self.optimal_penalty)) / 2.0
        )

        # Tolerance distance (log scale)
        tol_score = 1.0 - min(
            1.0, abs(np.log10(tolerance) - np.log10(self.optimal_tolerance)) / 2.0
        )

        # Solver tolerance consistency
        tol_ratio = abs_tol / (rel_tol + 1e-16)
        tol_consistency = 1.0 if 1e-4 < tol_ratio < 1e-2 else 0.5

        # Timestep stability (prefer medium timesteps)
        optimal_dt = 1e-4
        timestep_score = 1.0 - min(1.0, abs(np.log10(timestep) - np.log10(optimal_dt)) / 2.0)

        # Penalty-tolerance interaction
        # High penalty requires tighter tolerance
        interaction_penalty = 0.0
        if penalty > 1e6 and tolerance > 1e-7:
            interaction_penalty = 0.3

        # Combine scores
        base_prob = (
            0.4 * penalty_score
            + 0.3 * tol_score
            + 0.1 * tol_consistency
            + 0.2 * timestep_score
            - interaction_penalty
        )

        # Apply convergence margin based on difficulty
        prob = (base_prob - (1 - self.convergence_margin)) / self.convergence_margin
        prob = np.clip(prob, 0, 1)

        return prob

    def _compute_base_iterations(
        self,
        penalty: float,
        tolerance: float,
        abs_tol: float,
        rel_tol: float,
        timestep: float,
    ) -> float:
        """
        Compute expected iteration count for converged simulations.

        Better parameters -> fewer iterations needed.
        """
        # Base iteration count
        base = 20.0

        # Penalty contribution (too high or too low increases iterations)
        penalty_factor = 1.0 + 5.0 * (np.log10(penalty) - np.log10(self.optimal_penalty)) ** 2

        # Tolerance contribution (tighter tolerance -> more iterations)
        tol_factor = 1.0 + 10.0 * abs(
            np.log10(tolerance) - np.log10(self.optimal_tolerance)
        )

        # Solver tolerance (tighter -> more iterations, but more accuracy)
        solver_tol_factor = -2.0 * np.log10(rel_tol + 1e-16) / 10.0

        # Timestep (smaller timestep -> smaller changes -> fewer NR iterations per step)
        timestep_factor = 1.0 - 0.5 * np.log10(timestep / 1e-4)

        total_iterations = (
            base * penalty_factor * tol_factor + solver_tol_factor + timestep_factor
        )

        return max(5.0, total_iterations)


class SyntheticDataGenerator:
    """
    Generate synthetic datasets for testing and benchmarking.
    """

    def __init__(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        random_seed: Optional[int] = None,
    ):
        """
        Initialize synthetic data generator.

        Args:
            parameter_bounds: Dictionary mapping parameter names to (min, max) bounds
            random_seed: Random seed for reproducibility
        """
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())
        self.rng = np.random.RandomState(random_seed)

    def generate_random_parameters(self, n_samples: int) -> list[Dict[str, float]]:
        """
        Generate random parameter sets within bounds.

        Args:
            n_samples: Number of parameter sets to generate

        Returns:
            List of parameter dictionaries
        """
        parameter_sets = []

        for _ in range(n_samples):
            params = {}
            for name in self.parameter_names:
                lower, upper = self.parameter_bounds[name]

                # Use log-uniform sampling for parameters that span many orders of magnitude
                if upper / lower > 100:  # More than 2 orders of magnitude
                    value = 10 ** self.rng.uniform(np.log10(lower), np.log10(upper))
                else:
                    value = self.rng.uniform(lower, upper)

                params[name] = value

            parameter_sets.append(params)

        return parameter_sets

    def generate_latin_hypercube(self, n_samples: int) -> list[Dict[str, float]]:
        """
        Generate parameter sets using Latin Hypercube Sampling.

        Args:
            n_samples: Number of parameter sets to generate

        Returns:
            List of parameter dictionaries
        """
        from scipy.stats import qmc

        n_params = len(self.parameter_names)

        # Create Latin Hypercube sampler
        sampler = qmc.LatinHypercube(d=n_params, seed=self.rng.randint(0, 10000))
        samples = sampler.random(n=n_samples)

        # Scale to bounds
        parameter_sets = []
        for sample in samples:
            params = {}
            for i, name in enumerate(self.parameter_names):
                lower, upper = self.parameter_bounds[name]

                # Use log-uniform scaling for wide-range parameters
                if upper / lower > 100:
                    value = 10 ** (
                        np.log10(lower) + sample[i] * (np.log10(upper) - np.log10(lower))
                    )
                else:
                    value = lower + sample[i] * (upper - lower)

                params[name] = value

            parameter_sets.append(params)

        return parameter_sets

    def generate_dataset_with_solver(
        self,
        solver: MockSmithSolver,
        n_samples: int,
        sampling_method: str = "latin_hypercube",
    ) -> list[Tuple[Dict[str, float], bool, Optional[float]]]:
        """
        Generate complete dataset by evaluating solver.

        Args:
            solver: Mock solver instance
            n_samples: Number of samples to generate
            sampling_method: 'random' or 'latin_hypercube'

        Returns:
            List of (parameters, converged, objective_value) tuples
        """
        # Generate parameter sets
        if sampling_method == "latin_hypercube":
            parameter_sets = self.generate_latin_hypercube(n_samples)
        else:
            parameter_sets = self.generate_random_parameters(n_samples)

        # Evaluate with solver
        dataset = []
        for params in parameter_sets:
            converged, objective = solver.simulate(params)
            dataset.append((params, converged, objective))

        return dataset


def create_default_mock_solver() -> MockSmithSolver:
    """
    Create default mock solver with standard settings.

    Returns:
        MockSmithSolver instance
    """
    return MockSmithSolver(random_seed=42, noise_level=0.1, difficulty="medium")


def get_default_parameter_bounds() -> Dict[str, Tuple[float, float]]:
    """
    Get default parameter bounds for testing.

    Returns:
        Dictionary of parameter bounds
    """
    return {
        "penalty_stiffness": (1e3, 1e8),
        "gap_tolerance": (1e-9, 1e-6),
        "absolute_tolerance": (1e-12, 1e-8),
        "relative_tolerance": (1e-10, 1e-6),
        "max_iterations": (20, 100),
        "timestep": (1e-6, 1e-2),
    }
