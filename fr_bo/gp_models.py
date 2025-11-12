"""
Dual Gaussian Process system for FR-BO.

Implements:
1. SingleTaskGP for objective regression (trained on successful trials)
2. Variational GP classifier for failure probability prediction
"""

from typing import Optional, Tuple
import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood
import numpy as np


class ObjectiveGP:
    """
    Gaussian Process for objective function regression.

    Trains only on successful trials, using Matérn-5/2 kernel with ARD.
    """

    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor):
        """
        Initialize objective GP.

        Args:
            train_X: Training inputs (successful trials only)
            train_Y: Training objectives (successful trials only)
        """
        self.train_X = train_X
        self.train_Y = train_Y

        # Create SingleTaskGP with Matérn-5/2 kernel and ARD
        self.model = SingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            outcome_transform=Standardize(m=1),
        )

        # Configure Matérn-5/2 kernel with ARD (no priors for simpler optimization)
        self.model.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_X.shape[-1],
            ),
        )

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def optimize_hyperparameters(self, num_restarts: int = 5):
        """
        Optimize GP hyperparameters via MLE.

        Args:
            num_restarts: Number of random restarts for optimization
        """
        self.model.train()
        self.model.likelihood.train()

        best_loss = float("inf")
        best_state_dict = None

        for restart in range(num_restarts):
            # Random initialization
            if restart > 0:
                self._initialize_hyperparameters()

            # Optimize
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

            for i in range(100):
                optimizer.zero_grad()
                output = self.model(self.train_X)
                loss = -self.mll(output, self.train_Y.squeeze())
                loss.backward()
                optimizer.step()

            # Check if this is the best restart
            final_loss = loss.item()
            if final_loss < best_loss:
                best_loss = final_loss
                best_state_dict = self.model.state_dict()

        # Load best model
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        self.model.eval()
        self.model.likelihood.eval()

    def _initialize_hyperparameters(self):
        """Randomly initialize hyperparameters."""
        self.model.covar_module.base_kernel.lengthscale = torch.rand_like(
            self.model.covar_module.base_kernel.lengthscale
        ) * 2.0 + 0.1

    def predict(self, test_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions at test points.

        Args:
            test_X: Test inputs

        Returns:
            Tuple of (mean, variance) predictions
        """
        self.model.eval()
        self.model.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model.posterior(test_X)
            mean = posterior.mean
            variance = posterior.variance

        return mean, variance

    def get_lengthscales(self) -> np.ndarray:
        """
        Get learned lengthscales for ARD interpretation.

        Returns:
            Array of lengthscales (indicates parameter importance)
        """
        lengthscales = self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        return lengthscales.flatten()


class FailureClassifierGP(ApproximateGP):
    """
    Variational Gaussian Process classifier for failure probability.

    Trains on all trials (successful=0, failed=1) using variational inference
    with Bernoulli likelihood.
    """

    def __init__(self, train_X: torch.Tensor, inducing_points: Optional[torch.Tensor] = None):
        """
        Initialize failure classifier GP.

        Args:
            train_X: Training inputs
            inducing_points: Optional inducing points for scalability
        """
        # Use inducing points if provided, otherwise use all training points
        if inducing_points is None:
            inducing_points = train_X[:min(100, len(train_X))]

        # Variational distribution
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )

        # Variational strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        # Mean and covariance modules (no priors for simpler optimization)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_X.shape[-1],
            ),
        )

    def forward(self, x):
        """Forward pass."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class FailureClassifier:
    """Wrapper for failure classifier GP with training utilities."""

    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor):
        """
        Initialize failure classifier.

        Args:
            train_X: Training inputs (all trials)
            train_Y: Binary failure labels (0=success, 1=failure)
        """
        self.train_X = train_X
        self.train_Y = train_Y

        # Create model and likelihood
        self.model = FailureClassifierGP(train_X)
        self.likelihood = BernoulliLikelihood()

        self.mll = VariationalELBO(self.likelihood, self.model, num_data=train_Y.numel())

    def train_model(self, num_epochs: int = 200, lr: float = 0.1):
        """
        Train the failure classifier.

        Args:
            num_epochs: Number of training epochs
            lr: Learning rate
        """
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {"params": self.model.parameters()},
            {"params": self.likelihood.parameters()},
        ], lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = self.model(self.train_X)
            loss = -self.mll(output, self.train_Y)
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.likelihood.eval()

    def predict_failure_probability(
        self, test_X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict failure probability at test points.

        Args:
            test_X: Test inputs

        Returns:
            Tuple of (failure_probability, uncertainty)
        """
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            # Get latent function distribution
            latent_dist = self.model(test_X)

            # Convert to probability via likelihood
            pred_dist = self.likelihood(latent_dist)

            # Get probability of failure (class 1)
            failure_prob = pred_dist.mean

            # Uncertainty (variance of latent function)
            uncertainty = latent_dist.variance

        return failure_prob, uncertainty


class DualGPSystem:
    """
    Combined system managing both objective GP and failure classifier.
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        failure_labels: torch.Tensor,
    ):
        """
        Initialize dual GP system.

        Args:
            train_X: All training inputs
            train_Y: Objective values (for successful trials, NaN for failed)
            failure_labels: Binary labels (0=success, 1=failure)
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.failure_labels = failure_labels

        # Split data for objective GP (successful trials only)
        success_mask = failure_labels == 0
        self.train_X_success = train_X[success_mask]
        self.train_Y_success = train_Y[success_mask]

        # Initialize models
        if len(self.train_X_success) > 0:
            self.objective_gp = ObjectiveGP(self.train_X_success, self.train_Y_success)
        else:
            self.objective_gp = None

        self.failure_classifier = FailureClassifier(train_X, failure_labels)

    def train(self, num_restarts: int = 5):
        """
        Train both GPs.

        Args:
            num_restarts: Number of restarts for objective GP hyperparameter optimization
        """
        # Train objective GP (if we have successful trials)
        if self.objective_gp is not None:
            self.objective_gp.optimize_hyperparameters(num_restarts=num_restarts)

        # Train failure classifier
        self.failure_classifier.train_model()

    def predict(
        self, test_X: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Predict both objective and failure probability.

        Args:
            test_X: Test inputs

        Returns:
            Tuple of (obj_mean, obj_var, failure_prob, failure_uncertainty)
        """
        # Predict objective
        if self.objective_gp is not None:
            obj_mean, obj_var = self.objective_gp.predict(test_X)
        else:
            obj_mean, obj_var = None, None

        # Predict failure probability
        failure_prob, failure_unc = self.failure_classifier.predict_failure_probability(test_X)

        return obj_mean, obj_var, failure_prob, failure_unc

    def get_parameter_importance(self) -> Optional[np.ndarray]:
        """
        Get parameter importance from ARD lengthscales.

        Returns:
            Array of importance scores (inverse of lengthscales)
        """
        if self.objective_gp is not None:
            lengthscales = self.objective_gp.get_lengthscales()
            # Inverse lengthscale = importance
            importance = 1.0 / lengthscales
            # Normalize
            importance = importance / importance.sum()
            return importance
        return None
