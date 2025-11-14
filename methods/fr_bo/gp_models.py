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
from botorch.fit import fit_gpytorch_mll
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
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

        # Create likelihood without priors
        likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(1e-6)
        )

        # Create SingleTaskGP with custom likelihood
        # Only use Standardize transform if we have more than 1 sample
        if train_X.shape[0] > 1:
            self.model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                likelihood=likelihood,
                outcome_transform=Standardize(m=1),
            )
        else:
            # For single sample, don't use outcome transform
            self.model = SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                likelihood=likelihood,
            )

        # Configure Matérn-5/2 kernel with ARD
        self.model.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_X.shape[-1],
            ),
        )

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def optimize_hyperparameters(self, num_restarts: int = 3):
        """
        Optimize GP hyperparameters via MLE using BoTorch's fit utility.

        Args:
            num_restarts: Number of random restarts for optimization
        """
        self.model.train()
        self.model.likelihood.train()

        try:
            # Use BoTorch's fit utility which handles priors correctly
            fit_gpytorch_mll(self.mll)
        except Exception as e:
            # Fallback to simple optimization without prior validation
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
            for i in range(50):
                optimizer.zero_grad()
                output = self.model(self.train_X)
                loss = -self.mll(output, self.train_Y.squeeze())
                loss.backward()
                optimizer.step()

        self.model.eval()
        self.model.likelihood.eval()

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
            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)

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
        # Ensure train_Y is 1D for Bernoulli likelihood
        self.train_Y = train_Y.squeeze()

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
        # Squeeze failure_labels to ensure 1D tensor for boolean indexing
        failure_labels_1d = failure_labels.squeeze()
        success_mask = failure_labels_1d == 0
        self.train_X_success = train_X[success_mask]
        self.train_Y_success = train_Y[success_mask]

        # Track counts
        self.n_successes = int(success_mask.sum())
        self.n_failures = int((~success_mask).sum())

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

    def train_models(self, gp_restarts: int = 5, classifier_epochs: int = 200):
        """
        Train both GPs with detailed control over training parameters.

        Args:
            gp_restarts: Number of restarts for objective GP hyperparameter optimization
            classifier_epochs: Number of training epochs for failure classifier
        """
        # Train objective GP (if we have successful trials)
        if self.objective_gp is not None:
            self.objective_gp.optimize_hyperparameters(num_restarts=gp_restarts)

        # Train failure classifier
        self.failure_classifier.train_model(num_epochs=classifier_epochs)

    def predict_objective(self, test_X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict objective function values.

        Args:
            test_X: Test inputs

        Returns:
            Tuple of (mean, variance) predictions
        """
        if self.objective_gp is not None:
            return self.objective_gp.predict(test_X)
        else:
            # Return dummy predictions if no objective GP
            return torch.zeros(test_X.shape[0]), torch.ones(test_X.shape[0])

    def predict_failure_probability(self, test_X: torch.Tensor) -> torch.Tensor:
        """
        Predict failure probability (without uncertainty).

        Args:
            test_X: Test inputs

        Returns:
            Failure probability tensor
        """
        failure_prob, _ = self.failure_classifier.predict_failure_probability(test_X)
        return failure_prob

    def predict(self, test_X: torch.Tensor) -> dict:
        """
        Predict both objective and failure probability.

        Args:
            test_X: Test inputs

        Returns:
            Dictionary with keys: 'objective_mean', 'objective_variance',
            'failure_probability', 'success_probability'
        """
        # Predict objective
        if self.objective_gp is not None:
            obj_mean, obj_var = self.objective_gp.predict(test_X)
        else:
            obj_mean, obj_var = None, None

        # Predict failure probability
        failure_prob, failure_unc = self.failure_classifier.predict_failure_probability(test_X)
        success_prob = 1.0 - failure_prob

        return {
            'objective_mean': obj_mean,
            'objective_variance': obj_var,
            'failure_probability': failure_prob,
            'success_probability': success_prob,
        }

    def predict_with_uncertainty(
        self, test_X: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Predict both objective and failure probability with uncertainty.

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
