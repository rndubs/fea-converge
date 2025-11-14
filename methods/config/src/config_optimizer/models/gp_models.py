"""
Gaussian Process models for objective and constraints.

Implements GP surrogates using BoTorch/GPyTorch with Matérn-5/2 kernels.
"""

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
import numpy as np
from typing import Optional, Tuple


class ObjectiveGP:
    """
    Gaussian Process model for the objective function.
    
    Uses Matérn-5/2 kernel with ARD (Automatic Relevance Determination)
    for anisotropic length scales.
    """
    
    def __init__(
        self,
        bounds: np.ndarray,
        noise_prior: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize Objective GP.
        
        Args:
            bounds: Parameter bounds array of shape (n_params, 2)
            noise_prior: Optional (concentration, rate) for noise Gamma prior
        """
        self.bounds = torch.tensor(bounds, dtype=torch.float64)
        self.n_params = bounds.shape[0]
        self.noise_prior = noise_prior
        self.model = None
        self.mll = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit GP model to training data using MLE.
        
        Args:
            X: Training inputs of shape (n, n_params)
            y: Training objectives of shape (n,)
        """
        # Convert to tensors
        train_X = torch.tensor(X, dtype=torch.float64)
        train_y = torch.tensor(y, dtype=torch.float64).unsqueeze(-1)
        
        # Normalize to [0, 1]
        train_X_normalized = (train_X - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )
        
        # Standardize y
        self.y_mean = train_y.mean()
        self.y_std = train_y.std() if train_y.std() > 1e-6 else 1.0
        train_y_standardized = (train_y - self.y_mean) / self.y_std
        
        # Create GP model with Matérn-5/2 kernel
        self.model = SingleTaskGP(
            train_X_normalized,
            train_y_standardized,
        )
        
        # Set Matérn-5/2 kernel with ARD
        self.model.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=self.n_params,
                lengthscale_prior=GammaPrior(3.0, 6.0)
            ),
            outputscale_prior=GammaPrior(2.0, 0.15)
        )
        
        # Fit via MLE
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)
        
        # Store training data
        self.train_X = train_X
        self.train_y = train_y
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation at test points.
        
        Args:
            X: Test inputs of shape (n, n_params)
            
        Returns:
            (mean, std) arrays of shape (n,)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        # Convert and normalize
        test_X = torch.tensor(X, dtype=torch.float64)
        test_X_normalized = (test_X - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(test_X_normalized)
            mean_standardized = posterior.mean.squeeze(-1)
            variance_standardized = posterior.variance.squeeze(-1)
        
        # Unstandardize
        mean = mean_standardized * self.y_std + self.y_mean
        std = torch.sqrt(variance_standardized) * self.y_std
        
        return mean.numpy(), std.numpy()
    
    def posterior(self, X: torch.Tensor):
        """
        Get posterior distribution at X (for BoTorch acquisition functions).
        
        Args:
            X: Test inputs as torch tensor
            
        Returns:
            Posterior distribution
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        # Normalize
        X_normalized = (X - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )
        
        return self.model.posterior(X_normalized)


class ConstraintGP:
    """
    Gaussian Process model for a constraint function.
    
    Uses Matérn-5/2 kernel with ARD, similar to ObjectiveGP.
    """
    
    def __init__(
        self,
        bounds: np.ndarray,
        constraint_name: str = "constraint",
        noise_prior: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize Constraint GP.
        
        Args:
            bounds: Parameter bounds array of shape (n_params, 2)
            constraint_name: Name of this constraint
            noise_prior: Optional (concentration, rate) for noise Gamma prior
        """
        self.bounds = torch.tensor(bounds, dtype=torch.float64)
        self.n_params = bounds.shape[0]
        self.constraint_name = constraint_name
        self.noise_prior = noise_prior
        self.model = None
        self.mll = None
        
    def fit(self, X: np.ndarray, y_constraint: np.ndarray):
        """
        Fit GP model to constraint data.
        
        Args:
            X: Training inputs of shape (n, n_params)
            y_constraint: Constraint values of shape (n,)
                         Negative = satisfied, Positive = violated
        """
        # Convert to tensors
        train_X = torch.tensor(X, dtype=torch.float64)
        train_y = torch.tensor(y_constraint, dtype=torch.float64).unsqueeze(-1)
        
        # Normalize to [0, 1]
        train_X_normalized = (train_X - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )
        
        # Standardize y
        self.y_mean = train_y.mean()
        self.y_std = train_y.std() if train_y.std() > 1e-6 else 1.0
        train_y_standardized = (train_y - self.y_mean) / self.y_std
        
        # Create GP model with Matérn-5/2 kernel
        self.model = SingleTaskGP(
            train_X_normalized,
            train_y_standardized,
        )
        
        # Set Matérn-5/2 kernel with ARD
        self.model.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=self.n_params,
                lengthscale_prior=GammaPrior(3.0, 6.0)
            ),
            outputscale_prior=GammaPrior(2.0, 0.15)
        )
        
        # Fit via MLE
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)
        
        # Store training data
        self.train_X = train_X
        self.train_y = train_y
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation for constraint.
        
        Args:
            X: Test inputs of shape (n, n_params)
            
        Returns:
            (mean, std) arrays of shape (n,)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        # Convert and normalize
        test_X = torch.tensor(X, dtype=torch.float64)
        test_X_normalized = (test_X - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(test_X_normalized)
            mean_standardized = posterior.mean.squeeze(-1)
            variance_standardized = posterior.variance.squeeze(-1)
        
        # Unstandardize
        mean = mean_standardized * self.y_std + self.y_mean
        std = torch.sqrt(variance_standardized) * self.y_std
        
        return mean.numpy(), std.numpy()
    
    def posterior(self, X: torch.Tensor):
        """
        Get posterior distribution at X.
        
        Args:
            X: Test inputs as torch tensor
            
        Returns:
            Posterior distribution
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        # Normalize
        X_normalized = (X - self.bounds[:, 0]) / (
            self.bounds[:, 1] - self.bounds[:, 0]
        )
        
        return self.model.posterior(X_normalized)
    
    def compute_lcb(self, X: np.ndarray, beta: float) -> np.ndarray:
        """
        Compute Lower Confidence Bound for constraint.
        
        LCB = μ(x) - β^(1/2) * σ(x)
        
        Args:
            X: Test points
            beta: Confidence parameter
            
        Returns:
            LCB values (negative = likely feasible, positive = likely infeasible)
        """
        mean, std = self.predict(X)
        lcb = mean - np.sqrt(beta) * std
        return lcb
