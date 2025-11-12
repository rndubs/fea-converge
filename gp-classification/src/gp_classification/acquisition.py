"""
Acquisition functions for GP classification-based optimization.
"""

from typing import Callable, Optional

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models import SingleTaskGP
from botorch.sampling import SobolQMCNormalSampler
from torch import Tensor


class EntropyAcquisition(AcquisitionFunction):
    """
    Entropy-based acquisition for boundary discovery.

    Maximizes uncertainty about convergence prediction:
    H(P) = -P·log(P) - (1-P)·log(1-P)

    Peaks at P=0.5 (maximum uncertainty), useful for Phase 1 exploration.
    """

    def __init__(
        self,
        convergence_predictor: Callable[[Tensor], tuple[Tensor, Tensor]],
        epsilon: float = 1e-8,
    ):
        """
        Initialize entropy acquisition.

        Args:
            convergence_predictor: Function that returns (prob, std) for inputs
            epsilon: Small value to avoid log(0)
        """
        super().__init__(model=None)
        self.convergence_predictor = convergence_predictor
        self.epsilon = epsilon

    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate entropy acquisition.

        Args:
            X: Candidate points [batch_size, q, D] or [batch_size, D]

        Returns:
            Entropy values [batch_size]
        """
        # Handle both batched and unbatched inputs
        original_shape = X.shape
        if X.ndim == 2:
            X = X.unsqueeze(1)  # Add q dimension

        batch_size, q, d = X.shape
        X_flat = X.reshape(-1, d)

        # Get convergence probabilities (keep gradients)
        probs, _ = self.convergence_predictor(X_flat)
        probs = probs.reshape(batch_size, q)

        # Clip probabilities to avoid log(0)
        probs = torch.clamp(probs, self.epsilon, 1 - self.epsilon)

        # Compute entropy: H(P) = -P·log(P) - (1-P)·log(1-P)
        entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs))

        # Average over q dimension if present
        if q > 1:
            entropy = entropy.mean(dim=-1)
        else:
            entropy = entropy.squeeze(-1)

        return entropy


class BoundaryProximityAcquisition(AcquisitionFunction):
    """
    Acquisition that favors points near the decision boundary.

    Uses Gaussian weighting centered at P=0.5:
    α(x) = exp(-5·(P(converge|x) - 0.5)²)

    Useful for Phase 2 boundary refinement.
    """

    def __init__(
        self,
        convergence_predictor: Callable[[Tensor], tuple[Tensor, Tensor]],
        sharpness: float = 5.0,
    ):
        """
        Initialize boundary proximity acquisition.

        Args:
            convergence_predictor: Function that returns (prob, std) for inputs
            sharpness: Controls width of Gaussian around P=0.5
        """
        super().__init__(model=None)
        self.convergence_predictor = convergence_predictor
        self.sharpness = sharpness

    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate boundary proximity acquisition.

        Args:
            X: Candidate points [batch_size, q, D] or [batch_size, D]

        Returns:
            Proximity values [batch_size]
        """
        # Handle both batched and unbatched inputs
        if X.ndim == 2:
            X = X.unsqueeze(1)

        batch_size, q, d = X.shape
        X_flat = X.reshape(-1, d)

        # Get convergence probabilities
        probs, _ = self.convergence_predictor(X_flat)
        probs = probs.reshape(batch_size, q)

        # Compute proximity: exp(-sharpness·(P - 0.5)²)
        proximity = torch.exp(-self.sharpness * (probs - 0.5) ** 2)

        # Average over q dimension if present
        if q > 1:
            proximity = proximity.mean(dim=-1)
        else:
            proximity = proximity.squeeze(-1)

        return proximity


class ConstrainedEI(AcquisitionFunction):
    """
    Constrained Expected Improvement.

    Combines expected improvement with convergence probability:
    α_CEI(x) = α_EI(x) × P(feasible|x)

    Used for Phase 3 exploitation.
    """

    def __init__(
        self,
        model: SingleTaskGP,
        best_f: Tensor,
        convergence_predictor: Callable[[Tensor], tuple[Tensor, Tensor]],
        feasibility_threshold: float = 0.5,
        n_samples: int = 512,
    ):
        """
        Initialize constrained expected improvement.

        Args:
            model: Objective GP model
            best_f: Best observed objective value
            convergence_predictor: Function that returns (prob, std) for inputs
            feasibility_threshold: Minimum probability for feasibility
            n_samples: Number of MC samples for EI computation
        """
        super().__init__(model=model)
        self.best_f = best_f
        self.convergence_predictor = convergence_predictor
        self.feasibility_threshold = feasibility_threshold
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_samples]))

    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate constrained EI.

        Args:
            X: Candidate points [batch_size, q, D] or [batch_size, D]

        Returns:
            Constrained EI values [batch_size]
        """
        # Handle both batched and unbatched inputs
        if X.ndim == 2:
            X = X.unsqueeze(1)

        batch_size, q, d = X.shape
        X_flat = X.reshape(-1, d)

        # Compute Expected Improvement
        with torch.no_grad():
            posterior = self.model.posterior(X)
            samples = self.sampler(posterior)

            # EI = E[max(best_f - f(x), 0)]
            improvement = torch.clamp(self.best_f - samples, min=0.0)
            ei = improvement.mean(dim=0)

            if q > 1:
                ei = ei.mean(dim=1)
            else:
                ei = ei.squeeze(1)

        # Get convergence probabilities
        probs, _ = self.convergence_predictor(X_flat)
        probs = probs.reshape(batch_size, q)

        if q > 1:
            probs = probs.mean(dim=-1)
        else:
            probs = probs.squeeze(-1)

        # Constrained EI = EI × P(feasible)
        cei = ei * probs

        return cei


class AdaptiveAcquisition(AcquisitionFunction):
    """
    Adaptive acquisition that combines multiple strategies based on iteration.

    Implements the three-phase strategy:
    - Phase 1 (iter 1-20): Entropy-based exploration
    - Phase 2 (iter 21-50): Boundary refinement
    - Phase 3 (iter 51+): Constrained exploitation
    """

    def __init__(
        self,
        model: Optional[SingleTaskGP],
        best_f: Optional[Tensor],
        convergence_predictor: Callable[[Tensor], tuple[Tensor, Tensor]],
        current_iteration: int,
        phase1_end: int = 20,
        phase2_end: int = 50,
    ):
        """
        Initialize adaptive acquisition.

        Args:
            model: Objective GP model (can be None in early iterations)
            best_f: Best observed objective value (can be None in early iterations)
            convergence_predictor: Function that returns (prob, std) for inputs
            current_iteration: Current optimization iteration
            phase1_end: Last iteration of Phase 1
            phase2_end: Last iteration of Phase 2
        """
        super().__init__(model=model)
        self.best_f = best_f
        self.convergence_predictor = convergence_predictor
        self.current_iteration = current_iteration
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end

        # Initialize component acquisitions
        self.entropy_acq = EntropyAcquisition(convergence_predictor)
        self.boundary_acq = BoundaryProximityAcquisition(convergence_predictor)

        if model is not None and best_f is not None:
            self.cei_acq = ConstrainedEI(model, best_f, convergence_predictor)
        else:
            self.cei_acq = None

    def forward(self, X: Tensor) -> Tensor:
        """
        Evaluate adaptive acquisition based on current phase.

        Args:
            X: Candidate points [batch_size, q, D] or [batch_size, D]

        Returns:
            Acquisition values [batch_size]
        """
        if self.current_iteration <= self.phase1_end:
            # Phase 1: Pure exploration
            return self.entropy_acq(X)

        elif self.current_iteration <= self.phase2_end:
            # Phase 2: Boundary refinement with entropy bonus
            entropy = self.entropy_acq(X)
            boundary = self.boundary_acq(X)

            # Combine with weights
            w_entropy = 0.4
            w_boundary = 0.6

            return w_entropy * entropy + w_boundary * boundary

        else:
            # Phase 3: Exploitation with boundary awareness
            if self.cei_acq is None:
                # Fallback to boundary if CEI not available
                return self.boundary_acq(X)

            cei = self.cei_acq(X)
            boundary = self.boundary_acq(X)

            # Adaptive weighting based on iteration
            progress = (self.current_iteration - self.phase2_end) / 50.0
            progress = min(progress, 1.0)

            w_cei = 0.5 + 0.4 * progress  # 0.5 -> 0.9
            w_boundary = 1.0 - w_cei  # 0.5 -> 0.1

            return w_cei * cei + w_boundary * boundary

    def get_phase(self) -> str:
        """Get current optimization phase."""
        if self.current_iteration <= self.phase1_end:
            return "exploration"
        elif self.current_iteration <= self.phase2_end:
            return "boundary_refinement"
        else:
            return "exploitation"


def optimize_acquisition(
    acquisition_fn: AcquisitionFunction,
    bounds: Tensor,
    num_restarts: int = 10,
    raw_samples: int = 512,
) -> Tensor:
    """
    Optimize acquisition function to find next query point.

    Args:
        acquisition_fn: Acquisition function to optimize
        bounds: Parameter bounds [2, D]
        num_restarts: Number of random restarts for optimization
        raw_samples: Number of raw samples for initialization

    Returns:
        Best candidate point [D]
    """
    from botorch.optim import optimize_acqf

    candidates, acq_value = optimize_acqf(
        acq_function=acquisition_fn,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    return candidates.squeeze(0)
