"""Surrogate manager for coordinating multiple models."""

from typing import Any, Dict, Optional, List
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from shebo.models.ensemble import ConvergenceEnsemble, PerformanceEnsemble


class SurrogateManager:
    """Manages multiple surrogate models with asynchronous updates.

    Coordinates convergence, performance, and constraint surrogates,
    updating them on different schedules based on computational cost.
    """

    def __init__(
        self,
        input_dim: int,
        n_networks: int = 5,
        convergence_update_freq: int = 10,
        performance_update_freq: int = 10,
        constraint_update_freq: int = 50,
        max_epochs: int = 500,
        early_stop_patience: int = 20,
        device: Optional[str] = None
    ):
        """Initialize surrogate manager.

        Args:
            input_dim: Dimension of input features
            n_networks: Number of networks in each ensemble
            convergence_update_freq: Retrain convergence model every N samples
            performance_update_freq: Retrain performance model every N samples
            constraint_update_freq: Retrain constraint models every N samples
            max_epochs: Maximum training epochs
            early_stop_patience: Early stopping patience
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.input_dim = input_dim
        self.n_networks = n_networks
        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience

        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize models
        self.models: Dict[str, Any] = {
            'convergence': ConvergenceEnsemble(input_dim, n_networks),
            'performance': PerformanceEnsemble(input_dim, n_networks=n_networks),
            'constraints': {}  # Dynamically added as discovered
        }

        # Update schedules
        self.update_schedules = {
            'convergence': convergence_update_freq,
            'performance': performance_update_freq,
            'constraints': constraint_update_freq
        }

        self.sample_count = 0
        self.training_history: Dict[str, List[float]] = {}

    def add_constraint(
        self,
        name: str,
        constraint_type: str = 'binary'
    ) -> None:
        """Add new constraint model when failure mode discovered.

        Args:
            name: Name of the constraint
            constraint_type: Type of constraint ('binary' or 'continuous')
        """
        if constraint_type == 'binary':
            model = ConvergenceEnsemble(self.input_dim, self.n_networks)
        else:
            model = PerformanceEnsemble(self.input_dim, n_networks=self.n_networks)

        self.models['constraints'][name] = model
        print(f"New constraint discovered and added: {name}")

    def should_update(self, model_name: str) -> bool:
        """Check if model should be updated based on sample count.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model should be updated
        """
        if model_name == 'constraints':
            return False  # Constraints checked individually

        freq = self.update_schedules[model_name]
        return self.sample_count % freq == 0 and self.sample_count > 0

    def update_models(
        self,
        X: torch.Tensor,
        y_convergence: torch.Tensor,
        y_performance: Optional[torch.Tensor] = None,
        y_constraints: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: int = 32,
        val_split: float = 0.2
    ) -> None:
        """Update models based on new data and schedules.

        Args:
            X: Input features tensor of shape (n_samples, input_dim)
            y_convergence: Convergence labels of shape (n_samples, 1)
            y_performance: Performance targets of shape (n_samples, 2), optional
            y_constraints: Dictionary of constraint labels, optional
            batch_size: Batch size for training
            val_split: Validation split fraction
        """
        self.sample_count += len(X)

        # Update convergence model
        if self.should_update('convergence'):
            print(f"Updating convergence model at sample {self.sample_count}")
            self._train_model(
                self.models['convergence'],
                X,
                y_convergence,
                batch_size,
                val_split,
                'convergence'
            )

        # Update performance model
        if self.should_update('performance') and y_performance is not None:
            # Only train on successful convergences
            success_mask = y_convergence.squeeze() == 1
            if success_mask.sum() > 10:  # Need at least 10 successful samples
                X_success = X[success_mask]
                y_perf_success = y_performance[success_mask]

                print(f"Updating performance model at sample {self.sample_count}")
                self._train_model(
                    self.models['performance'],
                    X_success,
                    y_perf_success,
                    batch_size,
                    val_split,
                    'performance'
                )

        # Update constraint models
        if y_constraints is not None:
            for con_name, con_labels in y_constraints.items():
                if con_name in self.models['constraints']:
                    if self.sample_count % self.update_schedules['constraints'] == 0:
                        print(f"Updating constraint model '{con_name}' at sample {self.sample_count}")
                        self._train_model(
                            self.models['constraints'][con_name],
                            X,
                            con_labels,
                            batch_size,
                            val_split,
                            f'constraint_{con_name}'
                        )

    def _train_model(
        self,
        model: pl.LightningModule,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        val_split: float,
        model_name: str
    ) -> None:
        """Train a single model.

        Args:
            model: PyTorch Lightning module to train
            X: Input features
            y: Target labels
            batch_size: Batch size
            val_split: Validation split
            model_name: Name for logging
        """
        # Split into train and validation
        n_samples = len(X)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val

        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Create datasets
        train_dataset = TensorDataset(X[train_indices], y[train_indices])
        val_dataset = TensorDataset(X[val_indices], y[val_indices])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stop_patience,
            verbose=False,
            mode='min'
        )

        checkpoint = ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            verbose=False
        )

        # Trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[early_stop, checkpoint],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1
        )

        # Train
        trainer.fit(model, train_loader, val_loader)

        # Store training history
        if model_name not in self.training_history:
            self.training_history[model_name] = []
        self.training_history[model_name].append(checkpoint.best_model_score.item())

    def predict(
        self,
        x: torch.Tensor,
        model_name: str = 'convergence'
    ) -> Dict[str, torch.Tensor]:
        """Get predictions from specific model.

        Args:
            x: Input features
            model_name: Name of model ('convergence', 'performance', or constraint name)

        Returns:
            Dictionary with predictions and uncertainty
        """
        # Move to device
        x = x.to(self.device)

        if model_name == 'convergence':
            model = self.models['convergence']
        elif model_name == 'performance':
            model = self.models['performance']
        elif model_name in self.models['constraints']:
            model = self.models['constraints'][model_name]
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = model.to(self.device)
        return model.predict_with_uncertainty(x)

    def get_model(self, model_name: str) -> pl.LightningModule:
        """Get a model by name.

        Args:
            model_name: Name of the model

        Returns:
            The model
        """
        if model_name == 'convergence':
            return self.models['convergence']
        elif model_name == 'performance':
            return self.models['performance']
        elif model_name in self.models['constraints']:
            return self.models['constraints'][model_name]
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def save_models(self, save_dir: str) -> None:
        """Save all models to directory.

        Args:
            save_dir: Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Save convergence model
        torch.save(
            self.models['convergence'].state_dict(),
            os.path.join(save_dir, 'convergence.pth')
        )

        # Save performance model
        torch.save(
            self.models['performance'].state_dict(),
            os.path.join(save_dir, 'performance.pth')
        )

        # Save constraint models
        for con_name, con_model in self.models['constraints'].items():
            torch.save(
                con_model.state_dict(),
                os.path.join(save_dir, f'constraint_{con_name}.pth')
            )

        print(f"Models saved to {save_dir}")

    def load_models(self, save_dir: str) -> None:
        """Load all models from directory.

        Args:
            save_dir: Directory containing saved models
        """
        import os

        # Load convergence model
        conv_path = os.path.join(save_dir, 'convergence.pth')
        if os.path.exists(conv_path):
            self.models['convergence'].load_state_dict(torch.load(conv_path))

        # Load performance model
        perf_path = os.path.join(save_dir, 'performance.pth')
        if os.path.exists(perf_path):
            self.models['performance'].load_state_dict(torch.load(perf_path))

        # Load constraint models
        for con_name in self.models['constraints'].keys():
            con_path = os.path.join(save_dir, f'constraint_{con_name}.pth')
            if os.path.exists(con_path):
                self.models['constraints'][con_name].load_state_dict(torch.load(con_path))

        print(f"Models loaded from {save_dir}")
