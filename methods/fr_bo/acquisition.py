"""
Failure-Robust Expected Improvement (FREI) acquisition function.

This module implements the custom acquisition function that balances
optimization potential with feasibility likelihood.
"""

from typing import Optional
import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
import math


class FailureRobustEI(AnalyticAcquisitionFunction):
    """
    Failure-Robust Expected Improvement acquisition function.

    FREI(θ) = EI(θ) × (1 - P_fail(θ))

    This naturally balances:
    - EI: Expected improvement over current best
    - (1 - P_fail): Probability of success

    The acquisition promotes exploration in promising regions while
    avoiding high-failure-probability areas.
    """

    def __init__(
        self,
        model: Model,
        failure_model: Model,
        best_f: float,
        maximize: bool = False,
        failure_penalty_weight: float = 1.0,
        failure_likelihood: Optional[object] = None,
    ):
        """
        Initialize FREI acquisition function.

        Args:
            model: GP model for objective function
            failure_model: GP classifier for failure probability
            best_f: Best observed objective value
            maximize: If True, maximize objective (use negative EI)
            failure_penalty_weight: Weight for failure penalty (default 1.0)
            failure_likelihood: Optional likelihood for failure model
        """
        super().__init__(model=model)
        self.failure_model = failure_model
        self.failure_likelihood = failure_likelihood
        self.best_f = best_f
        self.maximize = maximize
        self.failure_penalty_weight = failure_penalty_weight

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """
        Compute FREI acquisition value.

        Args:
            X: Input tensor of shape (batch_size, d)

        Returns:
            FREI values of shape (batch_size,)
        """
        # Compute Expected Improvement
        ei = self._compute_expected_improvement(X)

        # Compute success probability (1 - failure probability)
        success_prob = self._compute_success_probability(X)

        # Combine: FREI = EI × (1 - P_fail)^weight
        frei = ei * torch.pow(success_prob, self.failure_penalty_weight)

        # Ensure output has correct shape - squeeze q-batch dimension
        # For q=1, we should return shape [batch_size] not [batch_size, 1]
        if frei.dim() > 1:
            frei = frei.squeeze(-1)

        return frei

    def _compute_expected_improvement(self, X: Tensor) -> Tensor:
        """
        Compute Expected Improvement.

        Args:
            X: Input tensor

        Returns:
            EI values
        """
        # Get posterior
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        variance = posterior.variance.squeeze(-1)
        sigma = torch.sqrt(variance)

        # Compute standardized improvement
        if self.maximize:
            improvement = mean - self.best_f
        else:
            improvement = self.best_f - mean

        # Compute EI using closed form
        # EI = improvement * Φ(Z) + sigma * φ(Z)
        # where Z = improvement / sigma

        # Handle zero variance case
        sigma = sigma.clamp(min=1e-9)
        Z = improvement / sigma

        # Standard normal CDF and PDF
        normal = torch.distributions.Normal(0, 1)
        Phi_Z = normal.cdf(Z)
        phi_Z = torch.exp(normal.log_prob(Z))

        ei = improvement * Phi_Z + sigma * phi_Z

        # Clamp to non-negative
        ei = ei.clamp(min=0.0)

        return ei

    def _compute_success_probability(self, X: Tensor) -> Tensor:
        """
        Compute success probability from failure classifier.

        Args:
            X: Input tensor of shape (..., d)

        Returns:
            Success probability (1 - P_fail) of shape matching X.shape[:-1]
        """
        # Get failure probability from classifier
        with torch.no_grad():
            # Get latent distribution
            latent_dist = self.failure_model(X)

            # Get failure probability via Bernoulli likelihood
            # Use provided likelihood if available
            if self.failure_likelihood is not None:
                pred_dist = self.failure_likelihood(latent_dist)
                # pred_dist.probs gives us P(y=1) for Bernoulli
                if hasattr(pred_dist, "probs"):
                    failure_prob = pred_dist.probs
                else:
                    failure_prob = pred_dist.mean

                # Keep shape as-is to match EI dimensions for proper broadcasting
                # pred_dist.probs already has the right shape
                pass
            else:
                # Fallback: use sigmoid of mean
                # Keep mean shape for consistency
                failure_prob = torch.sigmoid(latent_dist.mean)

        # Compute success probability
        success_prob = 1.0 - failure_prob

        # Clamp to avoid numerical issues
        success_prob = success_prob.clamp(min=1e-6, max=1.0)

        return success_prob


class UpperConfidenceBound:
    """
    Upper Confidence Bound acquisition for initial exploration.

    UCB(θ) = μ(θ) + β × σ(θ)

    Useful for exploration phase when failure data is limited.
    """

    def __init__(self, model: Model, beta: float = 2.0, maximize: bool = False):
        """
        Initialize UCB acquisition.

        Args:
            model: GP model
            beta: Exploration parameter (typically 2.0 for 95% confidence)
            maximize: If True, maximize objective
        """
        self.model = model
        self.beta = beta
        self.maximize = maximize

    def __call__(self, X: Tensor) -> Tensor:
        """
        Compute UCB acquisition value.

        Args:
            X: Input tensor

        Returns:
            UCB values
        """
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        sigma = torch.sqrt(posterior.variance.squeeze(-1))

        if self.maximize:
            ucb = mean + self.beta * sigma
        else:
            ucb = -(mean - self.beta * sigma)  # LCB for minimization

        return ucb


class ProbabilityOfImprovement:
    """
    Probability of Improvement acquisition function.

    PI(θ) = P(f(θ) < f* - ξ)

    Useful for risk-averse optimization when failures are costly.
    """

    def __init__(self, model: Model, best_f: float, xi: float = 0.01, maximize: bool = False):
        """
        Initialize PI acquisition.

        Args:
            model: GP model
            best_f: Best observed value
            xi: Improvement threshold
            maximize: If True, maximize objective
        """
        self.model = model
        self.best_f = best_f
        self.xi = xi
        self.maximize = maximize

    def __call__(self, X: Tensor) -> Tensor:
        """
        Compute PI acquisition value.

        Args:
            X: Input tensor

        Returns:
            PI values
        """
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze(-1)
        sigma = torch.sqrt(posterior.variance.squeeze(-1))

        # Compute standardized improvement
        if self.maximize:
            Z = (mean - self.best_f - self.xi) / sigma.clamp(min=1e-9)
        else:
            Z = (self.best_f - mean - self.xi) / sigma.clamp(min=1e-9)

        # Compute probability
        normal = torch.distributions.Normal(0, 1)
        pi = normal.cdf(Z)

        return pi


def optimize_acquisition(
    acquisition_function,
    bounds: Tensor,
    num_restarts: int = 10,
    raw_samples: int = 512,
):
    """
    Optimize acquisition function using multi-start L-BFGS-B.

    Args:
        acquisition_function: Acquisition function to optimize
        bounds: Parameter bounds (2 x d tensor)
        num_restarts: Number of random restarts
        raw_samples: Number of initial random samples

    Returns:
        Tuple of (best candidate point, acquisition value)
    """
    from botorch.optim import optimize_acqf

    # Use BoTorch's optimize_acqf
    candidate, acq_value = optimize_acqf(
        acq_function=acquisition_function,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    return candidate, acq_value.item()
