"""Neural network models for convergence prediction."""

from typing import List, Optional
import torch
import torch.nn as nn


class ConvergenceNN(nn.Module):
    """Neural network for convergence prediction.

    Architecture: fully connected network with dropout for regularization.
    Output: sigmoid activation for binary classification (converged/not converged).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2
    ):
        """Initialize convergence neural network.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Convergence probability of shape (batch_size, 1)
        """
        return self.network(x)


class PerformanceNN(nn.Module):
    """Neural network for performance prediction (iteration count, solve time).

    Regression model that predicts log-transformed performance metrics.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,  # [log(iters), log(time)]
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2
    ):
        """Initialize performance neural network.

        Args:
            input_dim: Dimension of input features
            output_dim: Number of output metrics (default: 2 for iterations and time)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer for regression (no activation for unbounded output)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Performance predictions of shape (batch_size, output_dim)
        """
        return self.network(x)


class MultiGeometryNN(nn.Module):
    """Multi-task neural network with geometry descriptors for transfer learning.

    Combines parameter inputs with geometric features to enable transfer learning
    across different geometries.
    """

    def __init__(
        self,
        param_dim: int,
        geom_feature_dim: int,
        output_dim: int = 1,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2
    ):
        """Initialize multi-geometry neural network.

        Args:
            param_dim: Dimension of parameter features
            geom_feature_dim: Dimension of geometry features
            output_dim: Dimension of output (1 for convergence probability)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super().__init__()

        # Geometry encoder
        self.geom_encoder = nn.Sequential(
            nn.Linear(geom_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Combined network
        combined_dim = param_dim + 16  # Parameters + encoded geometry
        layers: List[nn.Module] = []
        prev_dim = combined_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # For convergence probability

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        params: torch.Tensor,
        geom_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            params: Parameter tensor of shape (batch_size, param_dim)
            geom_features: Geometry features of shape (batch_size, geom_feature_dim)

        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        geom_encoded = self.geom_encoder(geom_features)
        combined = torch.cat([params, geom_encoded], dim=1)
        return self.network(combined)
