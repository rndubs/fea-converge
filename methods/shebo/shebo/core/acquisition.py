"""Adaptive acquisition functions for SHEBO optimization."""

from typing import Dict, Optional, Tuple
import torch
import numpy as np
from scipy.optimize import minimize, differential_evolution


class AdaptiveAcquisition:
    """Adaptive acquisition function balancing multiple objectives.

    Combines expected improvement, feasibility probability, uncertainty reduction,
    and boundary exploration with adaptive weighting.
    """

    def __init__(
        self,
        surrogate_manager: 'SurrogateManager',  # type: ignore
        bounds: np.ndarray,
        best_performance: float = float('inf')
    ):
        """Initialize adaptive acquisition.

        Args:
            surrogate_manager: Surrogate manager for predictions
            bounds: Parameter bounds of shape (n_params, 2)
            best_performance: Best performance value seen so far
        """
        self.surrogate_manager = surrogate_manager
        self.bounds = bounds
        self.best_performance = best_performance
        self.iteration = 0

    def compute_acquisition(
        self,
        x: torch.Tensor,
        phase: str = 'exploration'
    ) -> torch.Tensor:
        """Compute multi-objective acquisition value.

        α(x) = w1·EI(x) + w2·P(feasible|x) + w3·H(x) + w4·boundary_prox(x)

        Args:
            x: Parameter tensor of shape (batch_size, n_params) or (n_params,)
            phase: Optimization phase ('exploration', 'boundary_learning', 'exploitation')

        Returns:
            Acquisition value(s)
        """
        # Ensure x is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 1. Expected Improvement (performance optimization)
        ei = self._compute_expected_improvement(x)

        # 2. Feasibility probability
        conv_pred = self.surrogate_manager.predict(x, 'convergence')
        p_feasible = conv_pred['mean']

        # 3. Uncertainty (epistemic for exploration)
        uncertainty = conv_pred['epistemic_uncertainty']

        # 4. Boundary proximity (encourage boundary exploration)
        boundary_prox = self._compute_boundary_proximity(p_feasible)

        # Get adaptive weights based on phase
        weights = self._get_adaptive_weights(phase)

        # Compute weighted acquisition
        acquisition = (
            weights['ei'] * ei +
            weights['feasibility'] * p_feasible +
            weights['uncertainty'] * uncertainty +
            weights['boundary'] * boundary_prox
        )

        return acquisition.squeeze()

    def _compute_expected_improvement(
        self,
        x: torch.Tensor,
        xi: float = 0.01
    ) -> torch.Tensor:
        """Compute Expected Improvement.

        Args:
            x: Parameter tensor
            xi: Exploration-exploitation trade-off parameter

        Returns:
            Expected improvement values
        """
        try:
            perf_pred = self.surrogate_manager.predict(x, 'performance')
            mean = perf_pred['mean'].squeeze()  # Single output
            uncertainty = perf_pred['uncertainty'].squeeze()

            # Handle edge case where uncertainty is zero or very small
            std = torch.sqrt(uncertainty + 1e-8)

            # EI calculation (for minimization)
            improvement = self.best_performance - mean - xi
            z = improvement / (std + 1e-8)

            # Compute EI using standard normal CDF and PDF
            # Using error function approximation for numerical stability
            from torch.special import erf

            # CDF(z) = 0.5 * (1 + erf(z / sqrt(2)))
            cdf_z = 0.5 * (1.0 + erf(z / np.sqrt(2)))

            # PDF(z) = exp(-z^2 / 2) / sqrt(2 * pi)
            pdf_z = torch.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)

            ei = improvement * cdf_z + std * pdf_z
            ei = torch.clamp(ei, min=0)

            # Handle NaN values
            ei = torch.nan_to_num(ei, nan=0.0)

            return ei

        except (RuntimeError, ValueError, AttributeError) as e:
            # Performance model not ready or other specific errors
            import logging
            logging.getLogger(__name__).debug(
                f"Could not compute EI, returning zeros: {str(e)}"
            )
            return torch.zeros(len(x))

    def _compute_boundary_proximity(
        self,
        p_feasible: torch.Tensor,
        sharpness: float = 5.0
    ) -> torch.Tensor:
        """Compute proximity to feasibility boundary.

        Encourages sampling near p=0.5 (decision boundary).

        Args:
            p_feasible: Feasibility probability
            sharpness: Controls how sharply peaked the function is

        Returns:
            Boundary proximity values
        """
        return torch.exp(-sharpness * (p_feasible - 0.5) ** 2)

    def _get_adaptive_weights(self, phase: str) -> Dict[str, float]:
        """Get adaptive weights based on optimization phase.

        Args:
            phase: Optimization phase

        Returns:
            Dictionary of weights
        """
        schedules = {
            'exploration': {
                'ei': 0.1,
                'feasibility': 0.2,
                'uncertainty': 0.5,
                'boundary': 0.2
            },
            'boundary_learning': {
                'ei': 0.2,
                'feasibility': 0.2,
                'uncertainty': 0.1,
                'boundary': 0.5
            },
            'exploitation': {
                'ei': 0.5,
                'feasibility': 0.3,
                'uncertainty': 0.1,
                'boundary': 0.1
            }
        }
        return schedules.get(phase, schedules['exploration'])

    def optimize(
        self,
        phase: str = 'exploration',
        n_restarts: int = 10,
        method: str = 'L-BFGS-B'
    ) -> Tuple[np.ndarray, float]:
        """Optimize acquisition function to find next point.

        Args:
            phase: Optimization phase
            n_restarts: Number of random restarts
            method: Optimization method ('L-BFGS-B' or 'differential_evolution')

        Returns:
            Tuple of (best_x, best_acquisition_value)
        """
        if method == 'differential_evolution':
            return self._optimize_differential_evolution(phase)
        else:
            return self._optimize_multistart(phase, n_restarts)

    def _optimize_multistart(
        self,
        phase: str,
        n_restarts: int
    ) -> Tuple[np.ndarray, float]:
        """Optimize using multi-start L-BFGS-B.

        Args:
            phase: Optimization phase
            n_restarts: Number of random restarts

        Returns:
            Best point and acquisition value
        """
        best_x = None
        best_acq = -float('inf')

        # Define objective (negative acquisition for minimization)
        def objective(x_np: np.ndarray) -> float:
            x_tensor = torch.tensor(x_np, dtype=torch.float32).reshape(1, -1)
            acq = self.compute_acquisition(x_tensor, phase)
            return -acq.item()  # Negative for minimization

        # Multi-start optimization
        for _ in range(n_restarts):
            # Random initialization within bounds
            x0 = np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1]
            )

            # Optimize
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(low, high) for low, high in self.bounds]
            )

            # Update best
            if -result.fun > best_acq:
                best_acq = -result.fun
                best_x = result.x

        return best_x, best_acq

    def _optimize_differential_evolution(
        self,
        phase: str
    ) -> Tuple[np.ndarray, float]:
        """Optimize using differential evolution (global optimizer).

        Args:
            phase: Optimization phase

        Returns:
            Best point and acquisition value
        """
        def objective(x_np: np.ndarray) -> float:
            x_tensor = torch.tensor(x_np, dtype=torch.float32).reshape(1, -1)
            acq = self.compute_acquisition(x_tensor, phase)
            return -acq.item()

        result = differential_evolution(
            objective,
            bounds=[(low, high) for low, high in self.bounds],
            maxiter=100,
            seed=None
        )

        return result.x, -result.fun

    def update_best_performance(self, performance: float) -> None:
        """Update best performance seen so far.

        Args:
            performance: New performance value
        """
        if performance < self.best_performance:
            self.best_performance = performance

    def select_batch(
        self,
        phase: str,
        batch_size: int,
        diversity_weight: float = 0.5
    ) -> np.ndarray:
        """Select diverse batch of points for parallel evaluation.

        Args:
            phase: Optimization phase
            batch_size: Number of points to select
            diversity_weight: Weight for diversity term (0-1)

        Returns:
            Array of shape (batch_size, n_params)
        """
        selected_points = []

        for i in range(batch_size):
            if i == 0:
                # First point: pure acquisition optimization
                x, _ = self.optimize(phase)
            else:
                # Subsequent points: balance acquisition and diversity
                x = self._optimize_with_diversity(
                    phase,
                    selected_points,
                    diversity_weight
                )

            selected_points.append(x)

        return np.array(selected_points)

    def _optimize_with_diversity(
        self,
        phase: str,
        existing_points: list,
        diversity_weight: float
    ) -> np.ndarray:
        """Optimize with diversity penalty.

        Args:
            phase: Optimization phase
            existing_points: Already selected points
            diversity_weight: Weight for diversity term

        Returns:
            Next point
        """
        existing_array = np.array(existing_points)

        def objective(x_np: np.ndarray) -> float:
            x_tensor = torch.tensor(x_np, dtype=torch.float32).reshape(1, -1)
            acq = self.compute_acquisition(x_tensor, phase).item()

            # Diversity penalty (negative minimum distance)
            distances = np.linalg.norm(existing_array - x_np, axis=1)
            min_dist = np.min(distances)

            # Combined objective
            return -(acq - diversity_weight * (-min_dist))

        # Optimize
        x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=[(low, high) for low, high in self.bounds]
        )

        return result.x
