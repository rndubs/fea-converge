"""Surrogate manager for coordinating multiple models."""

from typing import Any, Dict, Optional, List
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import logging

from shebo.models.ensemble import ConvergenceEnsemble, PerformanceEnsemble
from shebo.utils.preprocessing import FeatureNormalizer

# Configure logger
logger = logging.getLogger(__name__)

# Constants
MIN_SAMPLES_FOR_TRAINING = 10
MIN_SAMPLES_PER_CLASS = 3
SEVERE_IMBALANCE_RATIO = 10


class SurrogateManager:
    """Manages multiple surrogate models with asynchronous updates.

    Coordinates convergence, performance, and constraint surrogates,
    updating them on different schedules based on computational cost.
    Includes feature normalization and comprehensive data validation.
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
            convergence_update_freq: Retrain convergence model every N iterations
            performance_update_freq: Retrain performance model every N iterations
            constraint_update_freq: Retrain constraint models every N iterations
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

        logger.info(f"SurrogateManager using device: {self.device}")

        # Initialize feature normalizer
        self.normalizer = FeatureNormalizer()

        # Initialize models
        self.models: Dict[str, Any] = {
            'convergence': ConvergenceEnsemble(input_dim, n_networks).to(self.device),
            'performance': PerformanceEnsemble(
                input_dim,
                output_dim=1,  # Single output by default
                n_networks=n_networks
            ).to(self.device),
            'constraints': {}  # Dynamically added as discovered
        }

        # Update schedules (iteration-based, not sample-based)
        self.update_schedules = {
            'convergence': convergence_update_freq,
            'performance': performance_update_freq,
            'constraints': constraint_update_freq
        }

        # Track last update iteration for each model
        self.last_update = {
            'convergence': 0,
            'performance': 0,
            'constraints': {}
        }

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

        # Move to device immediately
        model = model.to(self.device)
        self.models['constraints'][name] = model
        self.last_update['constraints'][name] = 0

        logger.info(f"New constraint discovered and added: {name}")

    def should_update(self, model_name: str, current_iteration: int) -> bool:
        """Check if model should be updated based on iteration count.

        Args:
            model_name: Name of the model to check
            current_iteration: Current optimization iteration

        Returns:
            True if model should be updated
        """
        if model_name == 'constraints':
            return False  # Constraints checked individually

        freq = self.update_schedules[model_name]
        iterations_since_last = current_iteration - self.last_update[model_name]

        return iterations_since_last >= freq and current_iteration > 0

    def update_models(
        self,
        X: torch.Tensor,
        y_convergence: torch.Tensor,
        y_performance: Optional[torch.Tensor] = None,
        y_constraints: Optional[Dict[str, torch.Tensor]] = None,
        current_iteration: int = 0,
        batch_size: int = 32,
        val_split: float = 0.2
    ) -> None:
        """Update models based on new data and schedules.

        Args:
            X: Input features tensor of shape (n_samples, input_dim)
            y_convergence: Convergence labels of shape (n_samples, 1)
            y_performance: Performance targets of shape (n_samples, 1), optional
            y_constraints: Dictionary of constraint labels, optional
            current_iteration: Current optimization iteration
            batch_size: Batch size for training
            val_split: Validation split fraction
        """
        # Fit normalizer on first call
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        if not self.normalizer.fitted:
            self.normalizer.fit(X_np)
            logger.info("Feature normalizer fitted to data")

        # Normalize features
        X_normalized = self.normalizer.transform(X_np)
        X = torch.tensor(X_normalized, dtype=torch.float32)

        # Update convergence model
        if self.should_update('convergence', current_iteration):
            self.last_update['convergence'] = current_iteration
            logger.info(f"Updating convergence model at iteration {current_iteration}")
            self._train_model(
                self.models['convergence'],
                X,
                y_convergence,
                batch_size,
                val_split,
                'convergence'
            )

        # Update performance model
        if self.should_update('performance', current_iteration) and y_performance is not None:
            # Only train on successful convergences
            success_mask = y_convergence.squeeze() == 1
            if success_mask.sum() > MIN_SAMPLES_FOR_TRAINING:
                X_success = X[success_mask]
                y_perf_success = y_performance[success_mask]

                self.last_update['performance'] = current_iteration
                logger.info(f"Updating performance model at iteration {current_iteration}")
                self._train_model(
                    self.models['performance'],
                    X_success,
                    y_perf_success,
                    batch_size,
                    val_split,
                    'performance'
                )
            else:
                logger.warning(
                    f"Insufficient successful samples ({success_mask.sum()}) "
                    f"to train performance model"
                )

        # Update constraint models
        if y_constraints is not None:
            for con_name, con_labels in y_constraints.items():
                if con_name in self.models['constraints']:
                    iterations_since = (current_iteration -
                                       self.last_update['constraints'].get(con_name, 0))

                    if iterations_since >= self.update_schedules['constraints']:
                        self.last_update['constraints'][con_name] = current_iteration
                        logger.info(
                            f"Updating constraint model '{con_name}' "
                            f"at iteration {current_iteration}"
                        )
                        self._train_model(
                            self.models['constraints'][con_name],
                            X,
                            con_labels,
                            batch_size,
                            val_split,
                            f'constraint_{con_name}'
                        )

    def _validate_training_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        model_name: str
    ) -> bool:
        """Validate training data quality.

        Args:
            X: Input features
            y: Target labels
            model_name: Name of model for logging

        Returns:
            True if data is valid for training
        """
        # Check for NaN/Inf in features
        if torch.isnan(X).any() or torch.isinf(X).any():
            logger.warning(f"NaN/Inf in features for {model_name}, skipping training")
            return False

        # Check for NaN/Inf in labels
        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.warning(f"NaN/Inf in labels for {model_name}, skipping training")
            return False

        # Check minimum samples
        n_samples = len(X)
        if n_samples < MIN_SAMPLES_FOR_TRAINING:
            logger.warning(
                f"Only {n_samples} samples for {model_name} "
                f"(min: {MIN_SAMPLES_FOR_TRAINING}), skipping training"
            )
            return False

        # For binary classification, check class balance
        if len(y.shape) == 2 and y.shape[1] == 1:
            n_positive = (y == 1).sum().item()
            n_negative = (y == 0).sum().item()

            if n_positive < MIN_SAMPLES_PER_CLASS or n_negative < MIN_SAMPLES_PER_CLASS:
                logger.warning(
                    f"Insufficient samples per class for {model_name} "
                    f"(pos: {n_positive}, neg: {n_negative}), skipping training"
                )
                return False

            # Warn about severe imbalance
            imbalance_ratio = max(n_positive, n_negative) / min(n_positive, n_negative)
            if imbalance_ratio > SEVERE_IMBALANCE_RATIO:
                logger.warning(
                    f"Severe class imbalance for {model_name} "
                    f"({imbalance_ratio:.1f}:1)"
                )

        return True

    def _train_model(
        self,
        model: pl.LightningModule,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        val_split: float,
        model_name: str
    ) -> None:
        """Train a single model with validation and error handling.

        Args:
            model: PyTorch Lightning module to train
            X: Input features (already normalized)
            y: Target labels
            batch_size: Batch size
            val_split: Validation split
            model_name: Name for logging
        """
        # Validate data
        if not self._validate_training_data(X, y, model_name):
            return

        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)

        # Ensure model is on correct device
        model = model.to(self.device)

        # Adjust validation split for small datasets
        n_samples = len(X)
        if n_samples < 50:
            # Ensure at least 5 validation samples
            val_split = max(0.1, min(0.2, 5 / n_samples))
            logger.info(f"Adjusted val_split to {val_split:.2f} for {n_samples} samples")

        # Split into train and validation
        n_val = max(1, int(n_samples * val_split))
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

        try:
            # Train
            trainer.fit(model, train_loader, val_loader)

            # Store training history
            if checkpoint.best_model_score is not None:
                if model_name not in self.training_history:
                    self.training_history[model_name] = []
                self.training_history[model_name].append(
                    checkpoint.best_model_score.item()
                )
                logger.info(
                    f"Model {model_name} trained. "
                    f"Best val loss: {checkpoint.best_model_score.item():.6f}"
                )
            else:
                logger.warning(f"Training completed but no best_model_score for {model_name}")

        except Exception as e:
            logger.error(f"Error training model {model_name}: {str(e)}")

    def predict(
        self,
        x: torch.Tensor,
        model_name: str = 'convergence'
    ) -> Dict[str, torch.Tensor]:
        """Get predictions from specific model.

        Args:
            x: Input features (will be normalized internally)
            model_name: Name of model ('convergence', 'performance', or constraint name)

        Returns:
            Dictionary with predictions and uncertainty
        """
        # Normalize input if normalizer is fitted
        if self.normalizer.fitted:
            x_np = x.detach().cpu().numpy() if x.is_cuda else x.numpy()
            x_normalized = self.normalizer.transform(x_np)
            x = torch.tensor(x_normalized, dtype=torch.float32)

        # Move to device
        x = x.to(self.device)

        # Get model
        if model_name == 'convergence':
            model = self.models['convergence']
        elif model_name == 'performance':
            model = self.models['performance']
        elif model_name in self.models['constraints']:
            model = self.models['constraints'][model_name]
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Ensure model is on correct device
        model = model.to(self.device)

        # Predict
        return model.predict_with_uncertainty(x)

    def get_model(self, model_name: str) -> pl.LightningModule:
        """Get a model by name.

        Args:
            model_name: Name of the model

        Returns:
            The model

        Raises:
            ValueError: If model name is unknown
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
        """Save all models and normalizer to directory.

        Args:
            save_dir: Directory to save models
        """
        import os
        import pickle

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

        # Save normalizer
        with open(os.path.join(save_dir, 'normalizer.pkl'), 'wb') as f:
            pickle.dump(self.normalizer, f)

        logger.info(f"Models saved to {save_dir}")

    def load_models(self, save_dir: str) -> None:
        """Load all models and normalizer from directory.

        Args:
            save_dir: Directory containing saved models
        """
        import os
        import pickle

        # Load convergence model
        conv_path = os.path.join(save_dir, 'convergence.pth')
        if os.path.exists(conv_path):
            self.models['convergence'].load_state_dict(
                torch.load(conv_path, map_location=self.device)
            )

        # Load performance model
        perf_path = os.path.join(save_dir, 'performance.pth')
        if os.path.exists(perf_path):
            self.models['performance'].load_state_dict(
                torch.load(perf_path, map_location=self.device)
            )

        # Load constraint models
        for con_name in self.models['constraints'].keys():
            con_path = os.path.join(save_dir, f'constraint_{con_name}.pth')
            if os.path.exists(con_path):
                self.models['constraints'][con_name].load_state_dict(
                    torch.load(con_path, map_location=self.device)
                )

        # Load normalizer
        normalizer_path = os.path.join(save_dir, 'normalizer.pkl')
        if os.path.exists(normalizer_path):
            with open(normalizer_path, 'rb') as f:
                self.normalizer = pickle.load(f)

        logger.info(f"Models loaded from {save_dir}")
