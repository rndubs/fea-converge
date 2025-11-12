"""
GP models for convergence classification and objective regression.
"""

from typing import Optional, Tuple

import gpytorch
import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)


class VariationalGPClassifier(ApproximateGP):
    """
    Variational GP Classifier for binary convergence prediction.

    Uses:
    - Variational inference with inducing points for scalability
    - Matérn-5/2 kernel with ARD (Automatic Relevance Determination)
    - Bernoulli likelihood with logistic link function
    - ELBO (Evidence Lower Bound) optimization

    Outputs:
    - P(converged|x): Convergence probability in [0,1]
    - Epistemic uncertainty about classification boundary
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        n_inducing_points: int = 100,
        learn_inducing_locations: bool = True,
    ):
        """
        Initialize variational GP classifier.

        Args:
            train_X: Training inputs [N, D]
            train_Y: Binary labels [N, 1] (0=failed, 1=converged)
            n_inducing_points: Number of inducing points for scalability
            learn_inducing_locations: Whether to optimize inducing point locations
        """
        # Ensure consistent dtype
        train_X = train_X.to(dtype=torch.float64)
        train_Y = train_Y.to(dtype=torch.float64)

        # Initialize inducing points using k-means clustering
        from sklearn.cluster import KMeans

        n_points = min(n_inducing_points, len(train_X))

        if len(train_X) > n_points:
            kmeans = KMeans(n_clusters=n_points, random_state=42, n_init=10)
            kmeans.fit(train_X.numpy())
            inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float64)
        else:
            inducing_points = train_X[:n_points].clone()

        # Variational distribution
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))

        # Variational strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )

        super().__init__(variational_strategy)

        # Mean function (constant)
        self.mean_module = ConstantMean()

        # Covariance function (Matérn-5/2 with ARD)
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_X.size(-1),
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
        )

        # Convert to float64
        self.double()

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Forward pass through the GP.

        Args:
            x: Input tensor [N, D]

        Returns:
            MultivariateNormal distribution over latent function values
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def train_variational_gp_classifier(
    model: VariationalGPClassifier,
    likelihood: BernoulliLikelihood,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    n_epochs: int = 500,
    learning_rate: float = 0.05,
    verbose: bool = False,
) -> Tuple[VariationalGPClassifier, BernoulliLikelihood]:
    """
    Train variational GP classifier using ELBO optimization.

    Args:
        model: Variational GP classifier
        likelihood: Bernoulli likelihood
        train_X: Training inputs [N, D]
        train_Y: Binary labels [N, 1]
        n_epochs: Number of training epochs
        learning_rate: Adam optimizer learning rate
        verbose: Whether to print training progress

    Returns:
        Trained model and likelihood
    """
    # Ensure consistent dtype
    train_X = train_X.to(dtype=torch.float64)
    train_Y = train_Y.to(dtype=torch.float64)

    # Convert likelihood to float64
    likelihood = likelihood.double()

    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=learning_rate,
    )

    # Marginal log likelihood (ELBO)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_Y.numel())

    # Training loop
    best_loss = float("inf")
    patience = 50
    patience_counter = 0

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -mll(output, train_Y.squeeze())
        loss.backward()
        optimizer.step()

        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {loss.item():.4f}")

    model.eval()
    likelihood.eval()

    return model, likelihood


def predict_convergence_probability(
    model: VariationalGPClassifier,
    likelihood: BernoulliLikelihood,
    X: torch.Tensor,
    n_samples: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict convergence probability with uncertainty.

    Args:
        model: Trained variational GP classifier
        likelihood: Trained Bernoulli likelihood
        X: Test inputs [N, D]
        n_samples: Number of samples for Monte Carlo integration

    Returns:
        probs: Convergence probabilities [N]
        std: Predictive standard deviation [N]
    """
    # Ensure consistent dtype
    X = X.to(dtype=torch.float64)

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # Get latent function distribution
        latent_dist = model(X)

        # Sample from latent distribution
        latent_samples = latent_dist.rsample(torch.Size([n_samples]))

        # Transform through likelihood (sigmoid)
        prob_samples = torch.sigmoid(latent_samples)

        # Compute mean and std
        probs = prob_samples.mean(dim=0)
        std = prob_samples.std(dim=0)

    return probs, std


class DualModel:
    """
    Combined model for convergence classification and objective regression.

    Architecture:
    - VariationalGPClassifier: Models P(converged|x) for all trials
    - SingleTaskGP: Models objective f(x) for converged trials only

    This enables constrained optimization where convergence is a probabilistic constraint.
    """

    def __init__(
        self,
        train_X_all: torch.Tensor,
        train_Y_converged: torch.Tensor,
        train_X_success: Optional[torch.Tensor] = None,
        train_Y_objective: Optional[torch.Tensor] = None,
        n_inducing_points: int = 100,
    ):
        """
        Initialize dual model.

        Args:
            train_X_all: All trial parameters [N, D]
            train_Y_converged: Binary convergence labels [N, 1]
            train_X_success: Successful trial parameters [M, D]
            train_Y_objective: Objective values for successful trials [M, 1]
            n_inducing_points: Number of inducing points for classifier
        """
        # Convergence classifier (trained on all trials)
        self.convergence_model = VariationalGPClassifier(
            train_X_all, train_Y_converged, n_inducing_points=n_inducing_points
        )
        self.likelihood = BernoulliLikelihood()

        # Objective regression model (trained on successful trials only)
        self.objective_model = None
        if train_X_success is not None and train_Y_objective is not None:
            self.objective_model = SingleTaskGP(train_X_success, train_Y_objective)

        self._is_trained = False

    def train_models(
        self,
        train_X_all: torch.Tensor,
        train_Y_converged: torch.Tensor,
        n_epochs: int = 500,
        learning_rate: float = 0.05,
        verbose: bool = False,
    ) -> None:
        """
        Train both models.

        Args:
            train_X_all: All trial parameters [N, D]
            train_Y_converged: Binary convergence labels [N, 1]
            n_epochs: Number of epochs for classifier training
            learning_rate: Learning rate for classifier
            verbose: Whether to print progress
        """
        # Train convergence classifier
        if verbose:
            print("Training convergence classifier...")

        self.convergence_model, self.likelihood = train_variational_gp_classifier(
            self.convergence_model,
            self.likelihood,
            train_X_all,
            train_Y_converged,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        # Train objective model if available
        if self.objective_model is not None:
            if verbose:
                print("Training objective regression model...")

            self.objective_model.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.objective_model.likelihood, self.objective_model
            )

            # Use BoTorch's fit_gpytorch_mll for hyperparameter optimization
            from botorch.fit import fit_gpytorch_mll

            fit_gpytorch_mll(mll)

            self.objective_model.eval()

        self._is_trained = True

    def predict_convergence(
        self, X: torch.Tensor, n_samples: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict convergence probability.

        Args:
            X: Test inputs [N, D]
            n_samples: Number of Monte Carlo samples

        Returns:
            probs: Convergence probabilities [N]
            std: Predictive standard deviation [N]
        """
        return predict_convergence_probability(self.convergence_model, self.likelihood, X, n_samples)

    def predict_convergence_with_grad(
        self, X: torch.Tensor, n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict convergence probability with gradients enabled (for acquisition optimization).

        Args:
            X: Test inputs [N, D]
            n_samples: Number of Monte Carlo samples

        Returns:
            probs: Convergence probabilities [N]
            std: Predictive standard deviation [N]
        """
        X = X.to(dtype=torch.float64)

        self.convergence_model.eval()
        self.likelihood.eval()

        # Get latent function distribution (with gradients)
        latent_dist = self.convergence_model(X)

        # Sample from latent distribution
        latent_samples = latent_dist.rsample(torch.Size([n_samples]))

        # Transform through likelihood (sigmoid)
        prob_samples = torch.sigmoid(latent_samples)

        # Compute mean and std
        probs = prob_samples.mean(dim=0)
        std = prob_samples.std(dim=0)

        return probs, std

    def predict_objective(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict objective value (for converged trials).

        Args:
            X: Test inputs [N, D]

        Returns:
            mean: Predicted objective values [N]
            std: Predictive standard deviation [N]
        """
        if self.objective_model is None:
            raise ValueError("Objective model not initialized")

        self.objective_model.eval()

        with torch.no_grad():
            posterior = self.objective_model.posterior(X)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.sqrt().squeeze(-1)

        return mean, std

    def get_botorch_model_list(self) -> ModelListGP:
        """
        Get BoTorch ModelListGP for use with acquisition functions.

        Returns:
            ModelListGP containing [objective_model, convergence_model_wrapper]
        """
        if self.objective_model is None:
            raise ValueError("Objective model not initialized")

        # For BoTorch compatibility, we need to wrap the convergence model
        # This is a simplified version - in practice, you'd create a proper BoTorch Model wrapper
        return ModelListGP(self.objective_model)

    @property
    def is_trained(self) -> bool:
        """Check if models have been trained."""
        return self._is_trained

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DualModel(convergence_model={self.convergence_model.__class__.__name__}, "
            f"objective_model={self.objective_model.__class__.__name__ if self.objective_model else None}, "
            f"trained={self._is_trained})"
        )
