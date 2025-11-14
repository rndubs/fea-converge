"""
Black box FE simulation executor wrapper.

This module provides both a real Smith/Tribol executor and a synthetic
simulator for testing without the actual solver.
"""

from typing import Dict, Any, Optional, List
import time
import numpy as np
from dataclasses import dataclass
import warnings


@dataclass
class SimulationResult:
    """Result of a finite element simulation."""

    converged: bool
    iterations: int
    max_iterations: int
    time_elapsed: float
    timeout: float
    final_residual: float
    contact_pressure_max: float
    penetration_max: float
    severe_instability: bool
    residual_history: Optional[List[float]] = None
    active_set_sizes: Optional[List[int]] = None

    @property
    def objective_value(self) -> float:
        """
        Compute objective value from simulation result.

        For minimization: lower is better. We want to minimize residual and iterations.
        """
        if not self.converged:
            # Failed simulations get a penalty
            return float('inf')
        # Objective combines residual and iteration count
        return self.final_residual + 0.01 * self.iterations

    @property
    def failed(self) -> bool:
        """Return True if simulation failed (did not converge or severe instability)."""
        return not self.converged or self.severe_instability

    def __post_init__(self):
        """Ensure bool types are Python bools, not NumPy bools."""
        self.converged = bool(self.converged)
        self.severe_instability = bool(self.severe_instability)
        self.iterations = int(self.iterations)
        self.max_iterations = int(self.max_iterations)
        self.time_elapsed = float(self.time_elapsed)
        self.timeout = float(self.timeout)
        self.final_residual = float(self.final_residual)
        self.contact_pressure_max = float(self.contact_pressure_max)
        self.penetration_max = float(self.penetration_max)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "time_elapsed": self.time_elapsed,
            "timeout": self.timeout,
            "final_residual": self.final_residual,
            "contact_pressure_max": self.contact_pressure_max,
            "penetration_max": self.penetration_max,
            "severe_instability": self.severe_instability,
            "residual_history": self.residual_history,
            "active_set_sizes": self.active_set_sizes,
        }


class SyntheticSimulator:
    """
    Synthetic finite element simulator for testing without real solver.

    This simulator creates realistic failure surfaces and convergence behavior
    based on typical FEA contact problem characteristics.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize synthetic simulator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(random_seed)

        # Define "good" parameter regions (normalized space)
        self.optimal_regions = [
            {
                "center": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
                "radius": 0.3,
                "success_prob": 0.95,
                "mean_iters": 30,
            },
            {
                "center": np.array([0.2, 0.6, 0.4, 0.4, 0.4, 0.5, 0.4, 0.6, 0.4]),
                "radius": 0.25,
                "success_prob": 0.90,
                "mean_iters": 40,
            },
            {
                "center": np.array([0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
                "radius": 0.2,
                "success_prob": 0.70,
                "mean_iters": 60,
            },
        ]

        # Define "bad" parameter regions (guaranteed failures)
        self.failure_regions = [
            {
                "center": np.array([0.1, 0.9, 0.1, 0.9, 0.9, 0.8, 0.2, 0.1, 0.9]),
                "radius": 0.15,
            },
            {
                "center": np.array([0.9, 0.1, 0.9, 0.1, 0.1, 0.2, 0.9, 0.9, 0.1]),
                "radius": 0.12,
            },
        ]

    def _normalize_parameters(self, params: Dict[str, Any]) -> np.ndarray:
        """Normalize parameters to [0, 1] for distance calculations."""
        # Extract and normalize continuous parameters
        # Log-scale parameters
        penalty = params.get("penalty_stiffness", 1e5)
        gap_tol = params.get("gap_tolerance", 1e-7)
        proj_tol = params.get("projection_tolerance", 1e-7)
        abs_tol = params.get("abs_tolerance", 1e-8)
        rel_tol = params.get("rel_tolerance", 1e-6)

        continuous = [
            (np.log10(penalty) - 3) / 5 if penalty > 0 else 0.5,  # 1e3 to 1e8
            (np.log10(gap_tol) + 12) / 9 if gap_tol > 0 else 0.5,  # 1e-12 to 1e-3
            (np.log10(proj_tol) + 12) / 9 if proj_tol > 0 else 0.5,  # 1e-12 to 1e-3
            (np.log10(abs_tol) + 14) / 11 if abs_tol > 0 else 0.5,  # 1e-14 to 1e-3
            (np.log10(rel_tol) + 14) / 11 if rel_tol > 0 else 0.5,  # 1e-14 to 1e-3
            params.get("search_expansion", 0.05) / 0.1,  # 0.0 to 0.1
            params.get("max_iterations", 500) / 1000,  # 100 to 1000
            params.get("line_search_iters", 10) / 20,  # 5 to 20
            params.get("trust_region_scaling", 0.25) / 0.5,  # 0.1 to 0.5
        ]
        return np.array(continuous)

    def _compute_convergence_probability(self, params_normalized: np.ndarray) -> float:
        """
        Compute convergence probability based on parameter location.

        Args:
            params_normalized: Normalized parameter vector

        Returns:
            Convergence probability [0, 1]
        """
        # Start with base probability
        prob = 0.4

        # Check if in optimal regions (increase probability)
        for region in self.optimal_regions:
            dist = np.linalg.norm(params_normalized - region["center"])
            if dist < region["radius"]:
                weight = 1.0 - (dist / region["radius"])
                prob += weight * (region["success_prob"] - prob)

        # Check if in failure regions (decrease probability)
        for region in self.failure_regions:
            dist = np.linalg.norm(params_normalized - region["center"])
            if dist < region["radius"]:
                weight = 1.0 - (dist / region["radius"])
                prob *= (1.0 - 0.8 * weight)  # Reduce probability significantly

        # Add some noise
        prob += self.rng.normal(0, 0.05)
        return np.clip(prob, 0.0, 1.0)

    def _simulate_convergence_trajectory(
        self,
        max_iterations: int,
        initial_residual: float,
        convergence_rate: float,
        will_converge: bool,
    ) -> tuple:
        """
        Simulate residual convergence trajectory.

        Args:
            max_iterations: Maximum number of iterations
            initial_residual: Starting residual norm
            convergence_rate: Convergence rate (smaller = faster)
            will_converge: Whether this simulation will converge

        Returns:
            Tuple of (final_iterations, residual_history, converged)
        """
        residuals = []
        current_residual = initial_residual

        for i in range(max_iterations):
            # Exponential decay with noise
            decay = np.exp(-convergence_rate * (i + 1))
            noise = self.rng.normal(1.0, 0.1)
            current_residual = initial_residual * decay * noise

            # Add occasional spikes (contact active set changes)
            if self.rng.random() < 0.1:
                current_residual *= self.rng.uniform(1.2, 2.0)

            residuals.append(current_residual)

            # Check convergence
            if will_converge and current_residual < 1e-8:
                return i + 1, residuals, True

            # Check for stagnation (failure indicator)
            if not will_converge and i > 20 and len(residuals) >= 10:
                if residuals[-1] > residuals[-10] * 0.9:
                    # Stagnated - terminate early
                    return i + 1, residuals, False

        # Reached max iterations
        return max_iterations, residuals, current_residual < 1e-8

    def run(
        self,
        parameters: Dict[str, Any],
        max_iterations: int = 1000,
        timeout: float = 3600.0,
    ) -> SimulationResult:
        """
        Run synthetic simulation with given parameters.

        Args:
            parameters: Dictionary of solver parameters
            max_iterations: Maximum iterations
            timeout: Maximum wall-clock time

        Returns:
            SimulationResult object
        """
        start_time = time.time()

        # Normalize parameters
        params_normalized = self._normalize_parameters(parameters)

        # Compute convergence probability
        conv_prob = self._compute_convergence_probability(params_normalized)

        # Decide if this trial will converge
        will_converge = self.rng.random() < conv_prob

        # Set convergence rate based on parameters
        if will_converge:
            # Faster convergence in optimal regions
            base_rate = 0.15
            for region in self.optimal_regions:
                dist = np.linalg.norm(params_normalized - region["center"])
                if dist < region["radius"]:
                    base_rate += 0.1 * (1.0 - dist / region["radius"])
        else:
            # Slow/no convergence
            base_rate = 0.02

        # Add parameter-specific effects
        convergence_rate = base_rate * (1.0 + 0.5 * params_normalized[0])

        # Simulate convergence trajectory
        initial_residual = self.rng.uniform(1.0, 10.0)
        final_iters, residuals, converged = self._simulate_convergence_trajectory(
            max_iterations, initial_residual, convergence_rate, will_converge
        )

        # Compute elapsed time (roughly proportional to iterations)
        time_per_iter = self.rng.uniform(0.01, 0.05)
        time_elapsed = final_iters * time_per_iter

        # Check for severe instability (divergence)
        severe_instability = False
        if not converged and len(residuals) > 10:
            if residuals[-1] > residuals[0] * 10:
                severe_instability = True

        # Generate synthetic contact metrics
        contact_pressure_max = self.rng.uniform(1e6, 1e8) if converged else 0.0
        penetration_max = self.rng.uniform(1e-9, 1e-6) if converged else 1e-3

        # Generate active set sizes
        active_set_sizes = [
            int(self.rng.uniform(100, 500)) for _ in range(final_iters)
        ]

        return SimulationResult(
            converged=converged,
            iterations=final_iters,
            max_iterations=max_iterations,
            time_elapsed=time_elapsed,
            timeout=timeout,
            final_residual=residuals[-1],
            contact_pressure_max=contact_pressure_max,
            penetration_max=penetration_max,
            severe_instability=severe_instability,
            residual_history=residuals,
            active_set_sizes=active_set_sizes,
        )


class SmithTribolExecutor:
    """
    Real Smith/Tribol simulation executor.

    This would interface with the actual LLNL Tribol contact library
    and Smith/Serac solver framework when available.
    """

    def __init__(self, smith_path: str, tribol_config: Optional[Dict] = None):
        """
        Initialize Smith/Tribol executor.

        Args:
            smith_path: Path to Smith executable
            tribol_config: Tribol configuration dictionary
        """
        self.smith_path = smith_path
        self.tribol_config = tribol_config or {}

        warnings.warn(
            "SmithTribolExecutor is not fully implemented. "
            "Use SyntheticSimulator for testing."
        )

    def run(
        self,
        parameters: Dict[str, Any],
        max_iterations: int = 1000,
        timeout: float = 3600.0,
    ) -> SimulationResult:
        """
        Run real Smith/Tribol simulation.

        Args:
            parameters: Dictionary of solver parameters
            max_iterations: Maximum iterations
            timeout: Maximum wall-clock time

        Returns:
            SimulationResult object
        """
        raise NotImplementedError(
            "Real Smith/Tribol execution not implemented. "
            "This requires network access to build Smith. "
            "Use SyntheticSimulator for testing."
        )
