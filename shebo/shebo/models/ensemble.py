"""Ensemble models for uncertainty quantification."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from shebo.models.convergence_nn import ConvergenceNN, PerformanceNN


class ConvergenceEnsemble(pl.LightningModule):
    """Ensemble of convergence neural networks for uncertainty quantification.

    Trains multiple networks with different random initializations to
    quantify both aleatoric and epistemic uncertainty.
    """

    def __init__(
        self,
        input_dim: int,
        n_networks: int = 5,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        learning_rate: float = 1e-3
    ):
        """Initialize convergence ensemble.

        Args:
            input_dim: Dimension of input features
            n_networks: Number of networks in ensemble
            hidden_dims: Hidden layer dimensions for each network
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()

        self.networks = nn.ModuleList([
            ConvergenceNN(input_dim, hidden_dims, dropout)
            for _ in range(n_networks)
        ])
        self.learning_rate = learning_rate

        # Disable automatic optimization for manual training of each network
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from all networks.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            List of predictions from each network
        """
        predictions = [net(x) for net in self.networks]
        return predictions

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step with independent network updates.

        Args:
            batch: Tuple of (features, labels)
            batch_idx: Batch index

        Returns:
            Average loss across all networks
        """
        x, y = batch

        # Get optimizers (one per network)
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        # Train each network independently with separate backward passes
        total_loss = 0.0
        for network, optimizer in zip(self.networks, optimizers):
            optimizer.zero_grad()

            # Forward pass for this network only
            pred = network(x)
            loss = nn.BCELoss()(pred, y)

            # Backward pass for this network only
            self.manual_backward(loss)
            optimizer.step()

            total_loss += loss.detach()

        # Average loss for logging
        avg_loss = total_loss / len(self.networks)
        self.log('train_loss', avg_loss, prog_bar=True)
        return avg_loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Tuple of (features, labels)
            batch_idx: Batch index

        Returns:
            Validation loss
        """
        x, y = batch
        predictions = self(x)

        losses = []
        for pred in predictions:
            loss = nn.BCELoss()(pred, y)
            losses.append(loss)

        val_loss = sum(losses) / len(losses)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure separate optimizer for each network.

        Returns:
            List of optimizers, one per network for independent training
        """
        return [
            torch.optim.Adam(network.parameters(), lr=self.learning_rate)
            for network in self.networks
        ]

    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Predict with uncertainty quantification.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Dictionary containing:
                - mean: Ensemble mean prediction
                - epistemic_uncertainty: Model uncertainty (variance)
                - aleatoric_uncertainty: Inherent randomness (entropy)
                - total_uncertainty: Sum of epistemic and aleatoric
        """
        self.eval()
        predictions = []

        with torch.no_grad():
            for network in self.networks:
                network.eval()  # Ensure each network is in eval mode
                pred = network(x)
                predictions.append(pred)

        predictions_tensor = torch.stack(predictions)  # (n_networks, batch_size, 1)

        # Ensemble mean
        mean = predictions_tensor.mean(dim=0)

        # Epistemic uncertainty (model uncertainty)
        epistemic_unc = predictions_tensor.var(dim=0)

        # Aleatoric uncertainty (average predictive entropy)
        epsilon = 1e-8  # Numerical stability
        entropy = -(
            predictions_tensor * torch.log(predictions_tensor + epsilon) +
            (1 - predictions_tensor) * torch.log(1 - predictions_tensor + epsilon)
        )
        aleatoric_unc = entropy.mean(dim=0)

        return {
            'mean': mean,
            'epistemic_uncertainty': epistemic_unc,
            'aleatoric_uncertainty': aleatoric_unc,
            'total_uncertainty': epistemic_unc + aleatoric_unc
        }


class PerformanceEnsemble(pl.LightningModule):
    """Ensemble of performance neural networks for regression.

    Predicts log-transformed performance metrics with uncertainty.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,  # Changed from 2 to 1 by default
        n_networks: int = 5,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        learning_rate: float = 1e-3
    ):
        """Initialize performance ensemble.

        Args:
            input_dim: Dimension of input features
            output_dim: Number of outputs (default: 1 for single performance metric)
            n_networks: Number of networks in ensemble
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        self.networks = nn.ModuleList([
            PerformanceNN(input_dim, output_dim, hidden_dims, dropout)
            for _ in range(n_networks)
        ])
        self.learning_rate = learning_rate

        # Disable automatic optimization for manual training
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from all networks."""
        predictions = [net(x) for net in self.networks]
        return predictions

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step with independent network updates."""
        x, y = batch

        # Get optimizers (one per network)
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        # Train each network independently
        total_loss = 0.0
        for network, optimizer in zip(self.networks, optimizers):
            optimizer.zero_grad()

            pred = network(x)
            loss = nn.MSELoss()(pred, y)

            self.manual_backward(loss)
            optimizer.step()

            total_loss += loss.detach()

        avg_loss = total_loss / len(self.networks)
        self.log('train_loss', avg_loss, prog_bar=True)
        return avg_loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        predictions = self(x)

        losses = []
        for pred in predictions:
            loss = nn.MSELoss()(pred, y)
            losses.append(loss)

        val_loss = sum(losses) / len(losses)
        self.log('val_loss', val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure separate optimizer for each network.

        Returns:
            List of optimizers, one per network for independent training
        """
        return [
            torch.optim.Adam(network.parameters(), lr=self.learning_rate)
            for network in self.networks
        ]

    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Predict with uncertainty quantification.

        Returns:
            Dictionary with mean and uncertainty estimates
        """
        self.eval()
        predictions = []

        with torch.no_grad():
            for network in self.networks:
                network.eval()  # Ensure each network is in eval mode
                pred = network(x)
                predictions.append(pred)

        predictions_tensor = torch.stack(predictions)

        mean = predictions_tensor.mean(dim=0)
        uncertainty = predictions_tensor.var(dim=0)

        return {
            'mean': mean,
            'uncertainty': uncertainty
        }
