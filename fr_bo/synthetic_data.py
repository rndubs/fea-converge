"""
Synthetic dataset generation for testing FR-BO without real solver.

Creates realistic test scenarios with known failure regions and
optimal parameter configurations.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class SyntheticScenario:
    """Description of a synthetic test scenario."""

    name: str
    description: str
    optimal_regions: List[Dict]
    failure_regions: List[Dict]
    parameter_dim: int
    baseline_success_rate: float


class SyntheticDataGenerator:
    """
    Generate synthetic datasets for testing FR-BO components.
    """

    def __init__(self, random_seed: Optional[int] = 42):
        """
        Initialize synthetic data generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(random_seed)

    def generate_scenario_simple(self) -> SyntheticScenario:
        """
        Generate simple scenario with 2 optimal regions and 2 failure regions.

        Returns:
            SyntheticScenario object
        """
        return SyntheticScenario(
            name="simple",
            description="Two optimal regions with clear failure zones",
            optimal_regions=[
                {
                    "center": np.array([0.3, 0.3, 0.5]),
                    "radius": 0.2,
                    "success_prob": 0.95,
                    "mean_objective": 0.1,
                },
                {
                    "center": np.array([0.7, 0.7, 0.5]),
                    "radius": 0.15,
                    "success_prob": 0.90,
                    "mean_objective": 0.15,
                },
            ],
            failure_regions=[
                {
                    "center": np.array([0.1, 0.9, 0.5]),
                    "radius": 0.15,
                },
                {
                    "center": np.array([0.9, 0.1, 0.5]),
                    "radius": 0.15,
                },
            ],
            parameter_dim=3,
            baseline_success_rate=0.4,
        )

    def generate_scenario_complex(self) -> SyntheticScenario:
        """
        Generate complex scenario with multiple overlapping regions.

        Returns:
            SyntheticScenario object
        """
        return SyntheticScenario(
            name="complex",
            description="Multiple overlapping regions with narrow corridors",
            optimal_regions=[
                {
                    "center": np.array([0.25, 0.25, 0.5, 0.5, 0.5]),
                    "radius": 0.15,
                    "success_prob": 0.90,
                    "mean_objective": 0.12,
                },
                {
                    "center": np.array([0.75, 0.75, 0.5, 0.5, 0.5]),
                    "radius": 0.12,
                    "success_prob": 0.85,
                    "mean_objective": 0.18,
                },
                {
                    "center": np.array([0.5, 0.5, 0.8, 0.5, 0.5]),
                    "radius": 0.1,
                    "success_prob": 0.95,
                    "mean_objective": 0.08,
                },
            ],
            failure_regions=[
                {
                    "center": np.array([0.1, 0.1, 0.5, 0.5, 0.5]),
                    "radius": 0.2,
                },
                {
                    "center": np.array([0.9, 0.9, 0.5, 0.5, 0.5]),
                    "radius": 0.2,
                },
                {
                    "center": np.array([0.5, 0.5, 0.1, 0.5, 0.5]),
                    "radius": 0.15,
                },
            ],
            parameter_dim=5,
            baseline_success_rate=0.3,
        )

    def generate_training_data(
        self,
        scenario: SyntheticScenario,
        n_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training data based on scenario.

        Args:
            scenario: Synthetic scenario
            n_samples: Number of samples to generate

        Returns:
            Tuple of (X, y_objective, y_converged)
        """
        X = self.rng.rand(n_samples, scenario.parameter_dim)
        y_objective = np.zeros(n_samples)
        y_converged = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            x = X[i]

            # Compute success probability
            success_prob = self._compute_success_probability(x, scenario)

            # Determine if converged
            converged = self.rng.rand() < success_prob
            y_converged[i] = converged

            if converged:
                # Compute objective based on distance to optimal regions
                objective = self._compute_objective(x, scenario)
                y_objective[i] = objective
            else:
                # Failed trial - assign penalty
                y_objective[i] = 10.0 + self.rng.rand()

        return X, y_objective, y_converged

    def _compute_success_probability(
        self, x: np.ndarray, scenario: SyntheticScenario
    ) -> float:
        """Compute success probability for parameter vector."""
        prob = scenario.baseline_success_rate

        # Increase probability in optimal regions
        for region in scenario.optimal_regions:
            dist = np.linalg.norm(x - region["center"])
            if dist < region["radius"]:
                weight = 1.0 - (dist / region["radius"])
                prob += weight * (region["success_prob"] - prob)

        # Decrease probability in failure regions
        for region in scenario.failure_regions:
            dist = np.linalg.norm(x - region["center"])
            if dist < region["radius"]:
                weight = 1.0 - (dist / region["radius"])
                prob *= (1.0 - 0.9 * weight)

        # Add noise
        prob += self.rng.normal(0, 0.05)

        return np.clip(prob, 0.0, 1.0)

    def _compute_objective(
        self, x: np.ndarray, scenario: SyntheticScenario
    ) -> float:
        """Compute objective value for successful trial."""
        # Find distance to nearest optimal region
        min_objective = 1.0

        for region in scenario.optimal_regions:
            dist = np.linalg.norm(x - region["center"])
            # Quadratic penalty from center
            objective = region["mean_objective"] + 0.5 * (dist / region["radius"]) ** 2
            min_objective = min(min_objective, objective)

        # Add noise
        min_objective += self.rng.normal(0, 0.02)

        return max(0.0, min_objective)

    def generate_test_geometries(
        self, n_geometries: int = 5
    ) -> List[Dict]:
        """
        Generate synthetic geometry data for multi-task GP testing.

        Args:
            n_geometries: Number of geometries to generate

        Returns:
            List of geometry dictionaries
        """
        geometries = []

        for i in range(n_geometries):
            geometry = {
                "geometry_id": i,
                "total_surface_area": self.rng.uniform(1.0, 10.0),
                "potential_contact_area": self.rng.uniform(0.1, 5.0),
                "volume": self.rng.uniform(1.0, 100.0),
                "num_elements": int(self.rng.uniform(1000, 10000)),
                "youngs_moduli": [
                    self.rng.uniform(1e9, 2e11),
                    self.rng.uniform(1e9, 2e11),
                ],
                "initial_gaps": list(self.rng.uniform(1e-4, 1e-2, size=20)),
                "num_contact_pairs": int(self.rng.uniform(10, 500)),
            }
            geometries.append(geometry)

        return geometries

    def generate_convergence_trajectory(
        self,
        max_iterations: int = 100,
        will_converge: bool = True,
        convergence_rate: Optional[float] = None,
    ) -> Tuple[List[int], List[float], List[int]]:
        """
        Generate synthetic convergence trajectory.

        Args:
            max_iterations: Maximum iterations
            will_converge: Whether trajectory converges
            convergence_rate: Optional convergence rate

        Returns:
            Tuple of (iterations, residuals, active_set_sizes)
        """
        if convergence_rate is None:
            convergence_rate = 0.15 if will_converge else 0.02

        iterations = []
        residuals = []
        active_set_sizes = []

        initial_residual = self.rng.uniform(1.0, 10.0)
        current_residual = initial_residual
        active_set = int(self.rng.uniform(100, 500))

        for i in range(max_iterations):
            iterations.append(i)

            # Exponential decay with noise
            decay = np.exp(-convergence_rate * i)
            noise = self.rng.normal(1.0, 0.1)
            current_residual = initial_residual * decay * noise

            # Occasional spikes (active set changes)
            if self.rng.rand() < 0.1:
                current_residual *= self.rng.uniform(1.2, 2.0)
                active_set = int(active_set * self.rng.uniform(0.8, 1.2))

            residuals.append(max(current_residual, 1e-15))
            active_set_sizes.append(active_set)

            # Check convergence
            if will_converge and current_residual < 1e-10:
                break

        return iterations, residuals, active_set_sizes


def create_benchmark_dataset(
    scenario_name: str = "simple",
    n_train: int = 100,
    n_test: int = 50,
    random_seed: int = 42,
) -> Dict:
    """
    Create benchmark dataset for testing.

    Args:
        scenario_name: Name of scenario ("simple" or "complex")
        n_train: Number of training samples
        n_test: Number of test samples
        random_seed: Random seed

    Returns:
        Dictionary containing train/test splits
    """
    generator = SyntheticDataGenerator(random_seed)

    if scenario_name == "simple":
        scenario = generator.generate_scenario_simple()
    elif scenario_name == "complex":
        scenario = generator.generate_scenario_complex()
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    # Generate train data
    X_train, y_obj_train, y_conv_train = generator.generate_training_data(
        scenario, n_train
    )

    # Generate test data
    X_test, y_obj_test, y_conv_test = generator.generate_training_data(
        scenario, n_test
    )

    return {
        "scenario": scenario,
        "train": {
            "X": X_train,
            "y_objective": y_obj_train,
            "y_converged": y_conv_train,
        },
        "test": {
            "X": X_test,
            "y_objective": y_obj_test,
            "y_converged": y_conv_test,
        },
    }


if __name__ == "__main__":
    # Generate and save benchmark datasets
    import pickle
    from pathlib import Path

    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario_name in ["simple", "complex"]:
        print(f"Generating {scenario_name} scenario...")
        dataset = create_benchmark_dataset(scenario_name, n_train=200, n_test=50)

        with open(output_dir / f"{scenario_name}_benchmark.pkl", "wb") as f:
            pickle.dump(dataset, f)

        print(f"  Saved to {output_dir / f'{scenario_name}_benchmark.pkl'}")
        print(f"  Train success rate: {dataset['train']['y_converged'].mean():.2%}")
        print(f"  Test success rate: {dataset['test']['y_converged'].mean():.2%}")
