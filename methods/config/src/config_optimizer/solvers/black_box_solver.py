"""
Black box solver for testing CONFIG without Smith/Serac.

Provides synthetic test functions that mimic FEA convergence behavior.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    converged: bool
    final_residual: float
    iterations: int
    max_penetration: Optional[float] = None
    objective_value: Optional[float] = None
    

class BlackBoxSolver:
    """
    Simulates FEA solver behavior for testing purposes.
    
    Provides multiple test functions with different characteristics:
    - Branin: 2D optimization with convergence constraints
    - Hartmann6: 6D optimization problem
    - Constrained quadratic: Simple test with clear feasible region
    """
    
    def __init__(
        self,
        problem_type: str = "branin",
        noise_level: float = 0.01,
        seed: int = None
    ):
        """
        Initialize black box solver.
        
        Args:
            problem_type: Type of test problem ('branin', 'hartmann6', 'quadratic')
            noise_level: Standard deviation of observation noise
            seed: Random seed for reproducibility
        """
        self.problem_type = problem_type
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)
        
        # Define problem-specific parameters
        if problem_type == "branin":
            self.bounds = np.array([[-5, 10], [0, 15]])
            self.dim = 2
        elif problem_type == "hartmann6":
            self.bounds = np.array([[0, 1]] * 6)
            self.dim = 6
        elif problem_type == "quadratic":
            self.bounds = np.array([[-2, 2]] * 2)
            self.dim = 2
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    
    def evaluate(self, x: np.ndarray) -> SimulationResult:
        """
        Evaluate the black box function at point x.
        
        Args:
            x: Parameter vector to evaluate
            
        Returns:
            SimulationResult with objective and constraint values
        """
        x = np.atleast_1d(x)
        
        # Compute objective value
        if self.problem_type == "branin":
            obj_value = self._branin(x)
        elif self.problem_type == "hartmann6":
            obj_value = self._hartmann6(x)
        elif self.problem_type == "quadratic":
            obj_value = self._constrained_quadratic(x)
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")
        
        # Add observation noise
        obj_value += self.rng.normal(0, self.noise_level)
        
        # Simulate convergence behavior based on location
        converged, residual, iterations = self._simulate_convergence(x)
        
        # Simulate penetration (for some problems)
        max_penetration = self._simulate_penetration(x)
        
        return SimulationResult(
            converged=converged,
            final_residual=residual,
            iterations=iterations,
            max_penetration=max_penetration,
            objective_value=obj_value
        )
    
    def _branin(self, x: np.ndarray) -> float:
        """
        Branin function (standard optimization test function).
        
        Has 3 global minima in the domain [-5,10] x [0,15].
        """
        x1, x2 = x[0], x[1]
        a = 1.0
        b = 5.1 / (4 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8 * np.pi)
        
        return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
    
    def _hartmann6(self, x: np.ndarray) -> float:
        """
        6-dimensional Hartmann function.
        
        Has multiple local minima, global minimum around -3.32.
        """
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ])
        P = 1e-4 * np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ])
        
        outer_sum = 0
        for i in range(4):
            inner_sum = 0
            for j in range(6):
                inner_sum += A[i, j] * (x[j] - P[i, j])**2
            outer_sum += alpha[i] * np.exp(-inner_sum)
        
        return -outer_sum
    
    def _constrained_quadratic(self, x: np.ndarray) -> float:
        """
        Simple constrained quadratic function.
        
        Minimum at origin, with circular feasible region.
        """
        return np.sum(x**2)
    
    def _simulate_convergence(self, x: np.ndarray) -> tuple:
        """
        Simulate convergence behavior based on parameter location.
        
        Returns:
            (converged, final_residual, iterations)
        """
        # Define feasible region (problem-dependent)
        if self.problem_type == "branin":
            # Feasible region: avoid corners
            dist_from_center = np.linalg.norm(x - np.array([2.5, 7.5]))
            feasible_prob = 1.0 / (1.0 + np.exp((dist_from_center - 6) / 2))
            
        elif self.problem_type == "hartmann6":
            # Feasible region: interior of domain
            dist_from_center = np.linalg.norm(x - 0.5)
            feasible_prob = 1.0 / (1.0 + np.exp((dist_from_center - 0.4) / 0.1))
            
        elif self.problem_type == "quadratic":
            # Feasible region: circle of radius 1.5
            dist_from_origin = np.linalg.norm(x)
            feasible_prob = 1.0 / (1.0 + np.exp((dist_from_origin - 1.5) / 0.3))
        
        else:
            feasible_prob = 0.5
        
        # Add randomness
        feasible_prob = np.clip(feasible_prob + self.rng.normal(0, 0.1), 0, 1)
        
        # Determine convergence
        converged = self.rng.random() < feasible_prob
        
        if converged:
            # Converged: low residual, reasonable iterations
            final_residual = 10 ** self.rng.uniform(-12, -8)
            iterations = self.rng.integers(10, 50)
        else:
            # Failed: high residual, more iterations
            final_residual = 10 ** self.rng.uniform(-5, -2)
            iterations = self.rng.integers(50, 100)
        
        return converged, final_residual, iterations
    
    def _simulate_penetration(self, x: np.ndarray) -> float:
        """
        Simulate maximum penetration (physics constraint).
        
        Returns:
            Maximum penetration value
        """
        # Simple model: penetration increases near boundaries
        normalized_x = (x - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        
        # Distance to nearest boundary
        dist_to_boundary = np.min([
            np.min(normalized_x),
            np.min(1 - normalized_x)
        ])
        
        # Penetration inversely related to distance
        base_penetration = 0.01 / (dist_to_boundary + 0.1)
        
        # Add noise
        penetration = base_penetration * (1 + self.rng.normal(0, 0.2))
        
        return max(0, penetration)


class SyntheticDataGenerator:
    """
    Generate synthetic datasets for testing CONFIG.
    """
    
    @staticmethod
    def generate_training_data(
        solver: BlackBoxSolver,
        n_samples: int,
        bounds: np.ndarray,
        sampling_method: str = "lhs",
        seed: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic training data.
        
        Args:
            solver: Black box solver instance
            n_samples: Number of samples to generate
            bounds: Parameter bounds
            sampling_method: 'lhs', 'sobol', or 'random'
            seed: Random seed
            
        Returns:
            Dictionary with 'X', 'y_obj', 'y_constraints', 'converged'
        """
        from ..utils.sampling import generate_candidate_set
        
        # Generate sample points
        X = generate_candidate_set(bounds, n_samples, sampling_method, seed)
        
        # Evaluate all points
        results = [solver.evaluate(x) for x in X]
        
        # Extract data
        y_obj = np.array([r.objective_value for r in results])
        converged = np.array([r.converged for r in results])
        final_residuals = np.array([r.final_residual for r in results])
        iterations = np.array([r.iterations for r in results])
        
        return {
            'X': X,
            'y_obj': y_obj,
            'converged': converged,
            'final_residuals': final_residuals,
            'iterations': iterations,
            'results': results
        }
