"""
Objective function design with convergence priority weighting.

This module implements the composite objective function that prioritizes
convergence, then efficiency, then speed.
"""

from typing import Dict, Any
import numpy as np


class ObjectiveFunction:
    """
    Composite objective function for FEA convergence optimization.

    Objective: J(θ) = 10.0 × (1-converged) + 1.0 × (iters/max_iters) + 0.5 × (time/timeout)

    This formulation:
    - Treats non-convergence as catastrophic (dominates objective)
    - Enables optimization of computational efficiency among converging configurations
    - Severe numerical instabilities receive 2× floor padding (double penalty)
    - Early successful convergence receives 0.9× actual value (reward for efficiency)
    """

    def __init__(
        self,
        convergence_weight: float = 10.0,
        iteration_weight: float = 1.0,
        time_weight: float = 0.5,
        floor_padding_multiplier: float = 2.0,
        success_reward_multiplier: float = 0.9,
    ):
        """
        Initialize objective function.

        Args:
            convergence_weight: Weight for convergence penalty
            iteration_weight: Weight for iteration efficiency
            time_weight: Weight for computational time
            floor_padding_multiplier: Multiplier for severe failures
            success_reward_multiplier: Multiplier for early success
        """
        self.convergence_weight = convergence_weight
        self.iteration_weight = iteration_weight
        self.time_weight = time_weight
        self.floor_padding_multiplier = floor_padding_multiplier
        self.success_reward_multiplier = success_reward_multiplier

        # Track minimum observed successful value for floor padding
        self.min_successful_value = None

    def compute(
        self,
        converged: bool,
        iterations: int,
        max_iterations: int,
        time_elapsed: float,
        timeout: float,
        severe_instability: bool = False,
    ) -> float:
        """
        Compute the objective value.

        Args:
            converged: Whether simulation converged successfully
            iterations: Number of iterations used
            max_iterations: Maximum allowed iterations
            time_elapsed: Wall-clock time in seconds
            timeout: Maximum allowed time
            severe_instability: Whether severe numerical instability occurred

        Returns:
            Objective value (lower is better)
        """
        if converged:
            # Normalized iteration count
            iter_ratio = iterations / max_iterations

            # Normalized time
            time_ratio = time_elapsed / timeout

            # Compute base objective
            objective = (
                self.iteration_weight * iter_ratio +
                self.time_weight * time_ratio
            )

            # Reward early convergence
            if iter_ratio < 0.3:  # Converged in less than 30% of max iterations
                objective *= self.success_reward_multiplier

            # Update minimum successful value
            if self.min_successful_value is None:
                self.min_successful_value = objective
            else:
                self.min_successful_value = min(self.min_successful_value, objective)

            return objective

        else:
            # Non-convergence: apply floor padding
            if self.min_successful_value is not None:
                base_penalty = self.min_successful_value
            else:
                # If no successful trials yet, use theoretical maximum
                base_penalty = self.iteration_weight + self.time_weight

            # Convergence penalty dominates
            objective = self.convergence_weight + base_penalty

            # Apply additional penalty for severe instabilities
            if severe_instability:
                objective *= self.floor_padding_multiplier

            return objective

    def reset(self):
        """Reset the objective function state (e.g., for new optimization run)."""
        self.min_successful_value = None


def extract_objective_from_result(result: Dict[str, Any]) -> float:
    """
    Extract objective value from simulation result dictionary.

    Args:
        result: Dictionary containing simulation results

    Returns:
        Computed objective value
    """
    obj_func = ObjectiveFunction()

    return obj_func.compute(
        converged=result["converged"],
        iterations=result["iterations"],
        max_iterations=result.get("max_iterations", 1000),
        time_elapsed=result["time_elapsed"],
        timeout=result.get("timeout", 3600.0),
        severe_instability=result.get("severe_instability", False),
    )


def compute_success_metrics(results: list) -> Dict[str, float]:
    """
    Compute success rate and convergence metrics from a list of results.

    Args:
        results: List of simulation result dictionaries

    Returns:
        Dictionary of metrics
    """
    if not results:
        return {
            "success_rate": 0.0,
            "mean_iterations": 0.0,
            "mean_time": 0.0,
            "median_objective": np.inf,
            "total_trials": 0,
            "successful_trials": 0,
        }

    converged_results = [r for r in results if r.get("converged", False)]
    success_rate = len(converged_results) / len(results)

    if converged_results:
        mean_iterations = np.mean([r["iterations"] for r in converged_results])
        mean_time = np.mean([r["time_elapsed"] for r in converged_results])
        objectives = [extract_objective_from_result(r) for r in converged_results]
        median_objective = np.median(objectives)
    else:
        mean_iterations = 0.0
        mean_time = 0.0
        median_objective = np.inf

    return {
        "success_rate": success_rate,
        "mean_iterations": mean_iterations,
        "mean_time": mean_time,
        "median_objective": median_objective,
        "total_trials": len(results),
        "successful_trials": len(converged_results),
    }
