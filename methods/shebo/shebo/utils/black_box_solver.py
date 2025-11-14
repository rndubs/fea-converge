"""Black box FEA solver simulator for testing without Smith solver.

This module simulates the behavior of a finite element contact solver,
including convergence characteristics, residual histories, and various
failure modes that can occur in contact mechanics simulations.
"""

from typing import Dict, Any, Optional, List
import numpy as np


class BlackBoxFEASolver:
    """Simulates FEA contact solver behavior.

    Models complex convergence landscape with:
    - Parameter-dependent convergence probability
    - Realistic residual evolution patterns
    - Multiple failure modes (oscillation, divergence, mesh issues, etc.)
    - Performance metrics (iterations, solve time)
    """

    def __init__(
        self,
        param_names: Optional[List[str]] = None,
        noise_level: float = 0.1,
        random_seed: Optional[int] = None
    ):
        """Initialize black box solver.

        Args:
            param_names: Names of parameters (optional)
            noise_level: Noise level for stochasticity (0-1)
            random_seed: Random seed for reproducibility
        """
        self.param_names = param_names or []
        self.noise_level = noise_level
        self.rng = np.random.RandomState(random_seed)

    def solve(self, params: np.ndarray) -> Dict[str, Any]:
        """Simulate FEA solve with given parameters.

        Args:
            params: Parameter vector (e.g., [penalty, tolerance, timestep, ...])

        Returns:
            Dictionary with simulation outputs:
                - convergence_status: bool
                - residual_history: List[float]
                - iterations: int
                - solve_time: float
                - penetration_max: float
                - jacobian_min: float
                - contact_pairs: int
                - all_values: List[float]
                - expected_contact: bool
        """
        # Extract key parameters (assuming first few are critical)
        # For contact convergence, typical parameters:
        # params[0]: penalty parameter
        # params[1]: tolerance
        # params[2]: timestep (if applicable)
        # params[3+]: other solver parameters

        # Compute convergence probability based on parameters
        p_converge = self._compute_convergence_probability(params)

        # Determine if this run converges
        converged = self.rng.random() < p_converge

        # Generate residual history
        residual_history = self._generate_residual_history(params, converged)

        # Determine specific failure mode if failed
        failure_mode = None
        if not converged:
            failure_mode = self._determine_failure_mode(params)

        # Generate output based on convergence and failure mode
        output = self._generate_output(
            params,
            converged,
            residual_history,
            failure_mode
        )

        return output

    def _compute_convergence_probability(self, params: np.ndarray) -> float:
        """Compute convergence probability based on parameters.

        Models a complex landscape with:
        - Sweet spot in parameter space
        - Penalty parameter effects
        - Tolerance effects
        - Interaction effects

        Args:
            params: Parameter vector

        Returns:
            Convergence probability (0-1)
        """
        # Ensure params has at least 2 dimensions
        n_params = len(params)

        # Define "ideal" parameter values (sweet spot)
        ideal_params = np.array([1e8, 1e-6] + [0.5] * (n_params - 2))

        # Compute distance from ideal in parameter space
        # Normalize by reasonable ranges
        ranges = np.array([1e9, 1e-5] + [1.0] * (n_params - 2))
        normalized_dist = np.linalg.norm((params - ideal_params) / ranges)

        # Base probability decreases with distance from ideal
        base_prob = np.exp(-0.5 * normalized_dist)

        # Penalty parameter effects (params[0])
        if n_params > 0:
            penalty = params[0]
            # Too low penalty: poor convergence
            if penalty < 1e6:
                base_prob *= 0.3
            # Too high penalty: numerical issues
            elif penalty > 1e10:
                base_prob *= 0.5

        # Tolerance effects (params[1])
        if n_params > 1:
            tolerance = params[1]
            # Too loose tolerance: not converged
            if tolerance > 1e-4:
                base_prob *= 0.6
            # Too tight tolerance: hard to achieve
            elif tolerance < 1e-8:
                base_prob *= 0.7

        # Add some noise for realism
        noise = self.rng.normal(0, self.noise_level * 0.1)
        prob = np.clip(base_prob + noise, 0.05, 0.95)

        return prob

    def _determine_failure_mode(self, params: np.ndarray) -> str:
        """Determine which failure mode occurs.

        Args:
            params: Parameter vector

        Returns:
            Failure mode name
        """
        # Different failure modes based on parameter values
        n_params = len(params)

        if n_params > 0:
            penalty = params[0]

            # High penalty can cause oscillations
            if penalty > 5e9:
                if self.rng.random() < 0.6:
                    return 'residual_oscillation'

            # Very high penalty can cause numerical instability
            if penalty > 1e10:
                if self.rng.random() < 0.4:
                    return 'numerical_instability'

            # Low penalty can cause divergence
            if penalty < 1e6:
                if self.rng.random() < 0.5:
                    return 'residual_divergence'

        if n_params > 1:
            tolerance = params[1]

            # Tight tolerance can cause stagnation
            if tolerance < 1e-8:
                if self.rng.random() < 0.3:
                    return 'residual_stagnation'

        # Random failures
        failure_modes = [
            'mesh_distortion',
            'contact_detection_failure',
            'excessive_penetration'
        ]

        return self.rng.choice(failure_modes)

    def _generate_residual_history(
        self,
        params: np.ndarray,
        converged: bool
    ) -> List[float]:
        """Generate realistic residual history.

        Args:
            params: Parameter vector
            converged: Whether simulation converges

        Returns:
            List of residual values
        """
        max_iters = 100

        if converged:
            # Monotonic decrease to convergence
            n_iters = self.rng.randint(10, 50)
            initial_residual = self.rng.uniform(1e-2, 1e0)

            # Compute final residual based on tolerance
            if len(params) > 1:
                final_residual = params[1] * self.rng.uniform(0.1, 0.9)
            else:
                final_residual = 1e-7

            # Exponential decay
            residuals = initial_residual * np.exp(
                -np.linspace(0, 8, n_iters)
            )
            residuals[-1] = final_residual

            # Add small noise
            noise = self.rng.normal(0, 0.01, n_iters)
            residuals = residuals * (1 + noise)
            residuals = np.maximum(residuals, final_residual)

        else:
            # Failed convergence - depends on failure mode
            n_iters = max_iters
            initial_residual = self.rng.uniform(1e-2, 1e0)

            failure_mode = self._determine_failure_mode(params)

            if failure_mode == 'residual_oscillation':
                # Oscillating residual
                trend = np.linspace(0, 0.5, n_iters)
                oscillation = 0.3 * np.sin(np.linspace(0, 10 * np.pi, n_iters))
                residuals = initial_residual * np.exp(-trend) * (1 + oscillation)

            elif failure_mode == 'residual_divergence':
                # Diverging residual
                residuals = initial_residual * np.exp(np.linspace(0, 3, n_iters))

            elif failure_mode == 'residual_stagnation':
                # Stagnated residual
                residuals = initial_residual * np.exp(-np.linspace(0, 1, n_iters))
                residuals[20:] = residuals[20] * (1 + self.rng.normal(0, 0.001, n_iters - 20))

            else:
                # Generic slow convergence
                residuals = initial_residual * np.exp(-np.linspace(0, 2, n_iters))

        return residuals.tolist()

    def _generate_output(
        self,
        params: np.ndarray,
        converged: bool,
        residual_history: List[float],
        failure_mode: Optional[str]
    ) -> Dict[str, Any]:
        """Generate complete simulation output.

        Args:
            params: Parameter vector
            converged: Convergence status
            residual_history: Residual history
            failure_mode: Failure mode if not converged

        Returns:
            Complete output dictionary
        """
        n_iters = len(residual_history)

        # Base output
        output = {
            'convergence_status': converged,
            'residual_history': residual_history,
            'iterations': n_iters,
            'expected_contact': True
        }

        # Solve time (depends on iterations and parameters)
        base_time_per_iter = 0.1  # seconds
        output['solve_time'] = base_time_per_iter * n_iters * (1 + self.rng.normal(0, 0.1))

        # Penetration
        if converged:
            output['penetration_max'] = self.rng.uniform(1e-6, 1e-4)
        else:
            if failure_mode == 'excessive_penetration':
                output['penetration_max'] = self.rng.uniform(1e-2, 1e-1)
            else:
                output['penetration_max'] = self.rng.uniform(1e-5, 1e-3)

        # Jacobian (mesh quality)
        if converged:
            output['jacobian_min'] = self.rng.uniform(0.3, 1.0)
        else:
            if failure_mode == 'mesh_distortion':
                output['jacobian_min'] = self.rng.uniform(-0.5, 0.0)
            else:
                output['jacobian_min'] = self.rng.uniform(0.1, 0.8)

        # Contact pairs
        if failure_mode == 'contact_detection_failure':
            output['contact_pairs'] = 0
        else:
            output['contact_pairs'] = self.rng.randint(10, 100)

        # All values for NaN/Inf check
        all_values = residual_history.copy()
        all_values.extend([
            output['penetration_max'],
            output['jacobian_min'],
            float(output['contact_pairs'])
        ])

        # Inject NaN/Inf for numerical instability
        if failure_mode == 'numerical_instability':
            nan_idx = self.rng.randint(0, len(all_values))
            all_values[nan_idx] = np.nan

        output['all_values'] = all_values

        return output


def create_test_objective(
    n_params: int = 4,
    noise_level: float = 0.1,
    random_seed: Optional[int] = None
) -> callable:
    """Create a test objective function using the black box solver.

    Args:
        n_params: Number of parameters
        noise_level: Noise level for stochasticity
        random_seed: Random seed

    Returns:
        Objective function compatible with SHEBO optimizer
    """
    solver = BlackBoxFEASolver(
        noise_level=noise_level,
        random_seed=random_seed
    )

    def objective(params: np.ndarray) -> Dict[str, Any]:
        """Objective function for SHEBO.

        Args:
            params: Parameter vector

        Returns:
            Dictionary with 'output' and 'performance'
        """
        output = solver.solve(params)

        # Performance metric: iteration count (to minimize)
        performance = output['iterations']

        return {
            'output': output,
            'performance': performance
        }

    return objective
