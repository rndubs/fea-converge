"""Feature preprocessing utilities for SHEBO."""

from typing import Optional
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class FeatureNormalizer:
    """Normalizes features for neural network training.

    Uses StandardScaler to normalize features to zero mean and unit variance,
    which is critical for proper neural network training when features have
    vastly different scales (e.g., penalty: 1e6-1e10, tolerance: 1e-8-1e-4).
    """

    def __init__(self):
        """Initialize feature normalizer."""
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: np.ndarray) -> None:
        """Fit normalizer to data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
        """
        self.scaler.fit(X)
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features.

        Args:
            X: Feature matrix to transform

        Returns:
            Normalized feature matrix

        Raises:
            ValueError: If normalizer not fitted
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: Feature matrix

        Returns:
            Normalized feature matrix
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform normalized features back to original scale.

        Args:
            X: Normalized feature matrix

        Returns:
            Original scale feature matrix

        Raises:
            ValueError: If normalizer not fitted
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return self.scaler.inverse_transform(X)

    def to_tensor(self, X: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Transform and convert to tensor.

        Args:
            X: Feature matrix
            dtype: Torch data type

        Returns:
            Normalized feature tensor
        """
        X_normalized = self.transform(X)
        return torch.tensor(X_normalized, dtype=dtype)

    def from_tensor(self, X_tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy and inverse transform.

        Args:
            X_tensor: Normalized feature tensor

        Returns:
            Original scale feature matrix
        """
        X_np = X_tensor.detach().cpu().numpy()
        return self.inverse_transform(X_np)
