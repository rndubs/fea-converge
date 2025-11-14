"""
Early termination system with trajectory-based prediction.

Monitors convergence trajectories during simulation and predicts
whether convergence will be achieved, enabling early termination
of failing simulations.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


class TrajectoryGP(ExactGP):
    """
    Gaussian Process for modeling convergence trajectories.

    Uses Matérn-3/2 kernel to capture typical exponential decay patterns
    in residual evolution.
    """

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: GaussianLikelihood):
        """
        Initialize trajectory GP.

        Args:
            train_x: Iteration numbers
            train_y: Log residual values
            likelihood: Gaussian likelihood
        """
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=1.5)  # Matérn-3/2 kernel
        )

    def forward(self, x):
        """Forward pass."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class EarlyTerminationPredictor:
    """
    Predicts convergence likelihood based on partial residual trajectory.

    Uses GP extrapolation to predict whether simulation will converge
    within the maximum iteration limit.
    """

    def __init__(
        self,
        convergence_threshold: float = 1e-10,
        confidence_threshold: float = 0.8,
        probability_threshold: float = 0.2,
    ):
        """
        Initialize early termination predictor.

        Args:
            convergence_threshold: Residual threshold for convergence
            confidence_threshold: Minimum confidence for termination decision
            probability_threshold: Minimum convergence probability to continue
        """
        self.convergence_threshold = convergence_threshold
        self.confidence_threshold = confidence_threshold
        self.probability_threshold = probability_threshold

    def should_terminate(
        self,
        iterations: List[int],
        residuals: List[float],
        max_iterations: int,
        current_iteration: int,
    ) -> Tuple[bool, float, float]:
        """
        Determine if simulation should be terminated early.

        Args:
            iterations: List of iteration numbers
            residuals: List of corresponding residual values
            max_iterations: Maximum allowed iterations
            current_iteration: Current iteration number

        Returns:
            Tuple of (should_terminate, convergence_probability, confidence)
        """
        # Need at least 5 points for reliable prediction
        if len(iterations) < 5:
            return False, 1.0, 0.0

        # Check for obvious divergence
        if self._is_diverging(residuals):
            return True, 0.0, 1.0

        # Fit GP to trajectory
        try:
            gp_model = self._fit_trajectory_gp(iterations, residuals)
        except Exception:
            # If fitting fails, don't terminate
            return False, 0.5, 0.0

        # Extrapolate to max_iterations
        conv_prob, confidence = self._predict_convergence(
            gp_model, current_iteration, max_iterations
        )

        # Decide termination
        should_terminate = (
            conv_prob < self.probability_threshold and
            confidence > self.confidence_threshold
        )

        return should_terminate, conv_prob, confidence

    def _is_diverging(self, residuals: List[float]) -> bool:
        """
        Check if residuals are clearly diverging.

        Args:
            residuals: List of residual values

        Returns:
            True if diverging
        """
        if len(residuals) < 5:
            return False

        # Check if recent residuals are increasing
        recent = residuals[-5:]
        if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
            # Monotonically increasing
            if recent[-1] > recent[0] * 10:
                return True

        # Check for stagnation at high residual
        if len(residuals) >= 20:
            recent_20 = residuals[-20:]
            mean_recent = np.mean(recent_20)
            std_recent = np.std(recent_20)

            if mean_recent > 1.0 and std_recent < mean_recent * 0.1:
                # Stagnated at high residual
                return True

        return False

    def _fit_trajectory_gp(
        self, iterations: List[int], residuals: List[float]
    ) -> TrajectoryGP:
        """
        Fit GP to residual trajectory.

        Args:
            iterations: Iteration numbers
            residuals: Residual values

        Returns:
            Trained TrajectoryGP model
        """
        # Convert to tensors
        train_x = torch.tensor(iterations, dtype=torch.float32).unsqueeze(-1)

        # Log-transform residuals for better GP modeling
        log_residuals = np.log10(np.maximum(residuals, 1e-15))
        train_y = torch.tensor(log_residuals, dtype=torch.float32)

        # Create model
        likelihood = GaussianLikelihood()
        model = TrajectoryGP(train_x, train_y, likelihood)

        # Train model
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        # Optimize
        for i in range(50):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()

        return model

    def _predict_convergence(
        self,
        model: TrajectoryGP,
        current_iteration: int,
        max_iterations: int,
    ) -> Tuple[float, float]:
        """
        Predict probability of convergence by max_iterations.

        Args:
            model: Trained trajectory GP
            current_iteration: Current iteration
            max_iterations: Maximum iterations

        Returns:
            Tuple of (convergence_probability, confidence)
        """
        # Predict at max_iterations
        test_x = torch.tensor([[float(max_iterations)]], dtype=torch.float32)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = model.likelihood(model(test_x))
            mean_log_residual = prediction.mean.item()
            std_log_residual = prediction.stddev.item()

        # Convert back from log space
        predicted_residual = 10 ** mean_log_residual

        # Compute probability of convergence
        # P(residual < threshold) assuming log-normal distribution
        log_threshold = np.log10(self.convergence_threshold)
        z_score = (log_threshold - mean_log_residual) / max(std_log_residual, 1e-6)

        from scipy.stats import norm
        conv_prob = norm.cdf(z_score)

        # Confidence = inverse of uncertainty
        # High uncertainty -> low confidence
        confidence = 1.0 / (1.0 + std_log_residual)

        return conv_prob, confidence

    def extract_features(
        self,
        iterations: List[int],
        residuals: List[float],
        active_set_sizes: Optional[List[int]] = None,
    ) -> dict:
        """
        Extract features from convergence trajectory for ML models.

        Args:
            iterations: Iteration numbers
            residuals: Residual values
            active_set_sizes: Optional active set sizes

        Returns:
            Dictionary of features
        """
        if len(residuals) < 2:
            return {}

        features = {
            "current_residual": residuals[-1],
            "initial_residual": residuals[0],
            "residual_ratio": residuals[-1] / max(residuals[0], 1e-15),
            "mean_residual": np.mean(residuals),
            "std_residual": np.std(residuals),
            "min_residual": np.min(residuals),
            "num_iterations": len(iterations),
        }

        # Convergence rate (slope of log residual)
        if len(residuals) >= 5:
            log_residuals = np.log10(np.maximum(residuals, 1e-15))
            # Linear fit to recent trajectory
            recent_iters = iterations[-5:]
            recent_log_res = log_residuals[-5:]
            slope = np.polyfit(recent_iters, recent_log_res, 1)[0]
            features["convergence_rate"] = slope

        # Stagnation detection
        if len(residuals) >= 10:
            recent_10 = residuals[-10:]
            stagnation = np.std(recent_10) / (np.mean(recent_10) + 1e-15)
            features["stagnation_score"] = stagnation

        # Active set features
        if active_set_sizes is not None and len(active_set_sizes) > 0:
            features["current_active_set_size"] = active_set_sizes[-1]
            if len(active_set_sizes) >= 5:
                # Active set stability
                recent_active = active_set_sizes[-5:]
                features["active_set_stability"] = np.std(recent_active) / (
                    np.mean(recent_active) + 1e-6
                )

        return features


def monitor_simulation(
    residual_callback: callable,
    max_iterations: int,
    check_interval: int = 5,
    start_checking_at: int = 10,
    predictor: Optional[EarlyTerminationPredictor] = None,
) -> Tuple[bool, List[float], str]:
    """
    Monitor simulation with early termination checks.

    Args:
        residual_callback: Function that runs simulation and returns (iter, residual, converged)
        max_iterations: Maximum iterations
        check_interval: Check for termination every N iterations
        start_checking_at: Start checking after N iterations
        predictor: EarlyTerminationPredictor instance

    Returns:
        Tuple of (converged, residual_history, termination_reason)
    """
    if predictor is None:
        predictor = EarlyTerminationPredictor()

    iterations = []
    residuals = []
    termination_reason = "max_iterations"

    for i in range(max_iterations):
        # Run one iteration
        current_iter, residual, converged = residual_callback()

        iterations.append(current_iter)
        residuals.append(residual)

        # Check convergence
        if converged:
            termination_reason = "converged"
            return True, residuals, termination_reason

        # Check for early termination
        if i >= start_checking_at and i % check_interval == 0:
            should_terminate, conv_prob, confidence = predictor.should_terminate(
                iterations, residuals, max_iterations, i
            )

            if should_terminate:
                termination_reason = f"early_termination (P_conv={conv_prob:.3f}, conf={confidence:.3f})"
                return False, residuals, termination_reason

    return False, residuals, termination_reason
