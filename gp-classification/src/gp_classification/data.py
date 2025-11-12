"""
Data management system for trial history and convergence labeling.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class SimulationTrial:
    """Individual simulation trial record."""

    trial_id: int
    parameters: Dict[str, float]
    converged: bool
    objective_value: Optional[float] = None  # Only for converged trials
    iteration_count: Optional[int] = None
    residual_norm: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    geometry_metadata: Optional[Dict[str, float]] = None


class TrialDatabase:
    """
    Database for managing simulation trials with binary convergence labels.

    Features:
    - Time-series storage of (parameters, convergence_status, performance_metrics)
    - Automatic binary labeling (y=1 converged, y=0 failed)
    - Filtering by convergence status
    - Schema versioning for parameter evolution
    """

    def __init__(self, parameter_bounds: Dict[str, Tuple[float, float]]):
        """
        Initialize trial database.

        Args:
            parameter_bounds: Dictionary mapping parameter names to (min, max) bounds
        """
        self.parameter_bounds = parameter_bounds
        self.parameter_names = list(parameter_bounds.keys())
        self.trials: List[SimulationTrial] = []
        self._next_trial_id = 0

    def add_trial(
        self,
        parameters: Dict[str, float],
        converged: bool,
        objective_value: Optional[float] = None,
        iteration_count: Optional[int] = None,
        residual_norm: Optional[float] = None,
        geometry_metadata: Optional[Dict[str, float]] = None,
    ) -> int:
        """
        Add a new trial to the database.

        Args:
            parameters: Dictionary of parameter values
            converged: Binary convergence outcome (True/False)
            objective_value: Performance metric (only for converged trials)
            iteration_count: Number of iterations taken
            residual_norm: Final residual norm
            geometry_metadata: Geometric features for transfer learning

        Returns:
            Trial ID
        """
        # Validate parameters
        for param, value in parameters.items():
            if param not in self.parameter_names:
                raise ValueError(f"Unknown parameter: {param}")
            min_val, max_val = self.parameter_bounds[param]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Parameter {param}={value} out of bounds [{min_val}, {max_val}]"
                )

        trial = SimulationTrial(
            trial_id=self._next_trial_id,
            parameters=parameters,
            converged=converged,
            objective_value=objective_value if converged else None,
            iteration_count=iteration_count,
            residual_norm=residual_norm,
            geometry_metadata=geometry_metadata,
        )

        self.trials.append(trial)
        self._next_trial_id += 1

        return trial.trial_id

    def get_all_trials(self) -> pd.DataFrame:
        """Get all trials as a pandas DataFrame."""
        if not self.trials:
            return pd.DataFrame()

        data = []
        for trial in self.trials:
            row = {
                "trial_id": trial.trial_id,
                "converged": trial.converged,
                "objective_value": trial.objective_value,
                "iteration_count": trial.iteration_count,
                "residual_norm": trial.residual_norm,
                "timestamp": trial.timestamp,
            }
            row.update(trial.parameters)
            if trial.geometry_metadata:
                row.update({f"geo_{k}": v for k, v in trial.geometry_metadata.items()})
            data.append(row)

        return pd.DataFrame(data)

    def get_training_data(
        self, converged_only: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get training data for GP models.

        Args:
            converged_only: If True, return only converged trials (for objective GP)
                          If False, return all trials (for convergence classifier)

        Returns:
            X: Parameter tensor [N, D]
            y_converged: Binary labels [N, 1] (0=failed, 1=converged)
            y_objective: Objective values [M, 1] (only converged trials, None if converged_only=False)
        """
        if not self.trials:
            raise ValueError("No trials in database")

        trials = [t for t in self.trials if t.converged] if converged_only else self.trials

        if not trials:
            raise ValueError("No trials match the filter criteria")

        # Extract parameters
        X = []
        y_converged = []
        y_objective = []

        for trial in trials:
            # Parameters in consistent order
            x = [trial.parameters[name] for name in self.parameter_names]
            X.append(x)
            y_converged.append(1.0 if trial.converged else 0.0)

            if trial.converged and trial.objective_value is not None:
                y_objective.append(trial.objective_value)

        X = torch.tensor(X, dtype=torch.float64)
        y_converged = torch.tensor(y_converged, dtype=torch.float64).unsqueeze(-1)

        if converged_only and y_objective:
            y_objective_tensor = torch.tensor(y_objective, dtype=torch.float64).unsqueeze(-1)
        else:
            y_objective_tensor = None

        return X, y_converged, y_objective_tensor

    def get_successful_trials(self) -> List[SimulationTrial]:
        """Get all trials where convergence was achieved."""
        return [t for t in self.trials if t.converged]

    def get_failed_trials(self) -> List[SimulationTrial]:
        """Get all trials where convergence failed."""
        return [t for t in self.trials if not t.converged]

    def get_convergence_rate(self) -> float:
        """Calculate overall convergence rate."""
        if not self.trials:
            return 0.0
        return sum(1 for t in self.trials if t.converged) / len(self.trials)

    def get_best_trial(self) -> Optional[SimulationTrial]:
        """Get the trial with the best (minimum) objective value among converged trials."""
        converged = self.get_successful_trials()
        if not converged:
            return None

        return min(
            (t for t in converged if t.objective_value is not None),
            key=lambda t: t.objective_value,
            default=None,
        )

    def save(self, filepath: Path) -> None:
        """Save database to CSV file."""
        df = self.get_all_trials()
        df.to_csv(filepath, index=False)

    @classmethod
    def load(cls, filepath: Path, parameter_bounds: Dict[str, Tuple[float, float]]) -> "TrialDatabase":
        """Load database from CSV file."""
        df = pd.read_csv(filepath)
        db = cls(parameter_bounds)

        param_cols = list(parameter_bounds.keys())

        for _, row in df.iterrows():
            parameters = {col: row[col] for col in param_cols}

            # Extract geometry metadata
            geo_metadata = {}
            for col in df.columns:
                if col.startswith("geo_"):
                    geo_metadata[col[4:]] = row[col]

            db.add_trial(
                parameters=parameters,
                converged=bool(row["converged"]),
                objective_value=row.get("objective_value"),
                iteration_count=row.get("iteration_count"),
                residual_norm=row.get("residual_norm"),
                geometry_metadata=geo_metadata if geo_metadata else None,
            )

        return db

    def get_statistics(self) -> Dict[str, float]:
        """Get summary statistics about the trial database."""
        if not self.trials:
            return {}

        converged = self.get_successful_trials()

        stats = {
            "total_trials": len(self.trials),
            "converged_trials": len(converged),
            "failed_trials": len(self.trials) - len(converged),
            "convergence_rate": self.get_convergence_rate(),
        }

        if converged:
            objectives = [t.objective_value for t in converged if t.objective_value is not None]
            if objectives:
                stats["best_objective"] = min(objectives)
                stats["mean_objective"] = np.mean(objectives)
                stats["std_objective"] = np.std(objectives)

        return stats

    def __len__(self) -> int:
        """Return number of trials in database."""
        return len(self.trials)

    def __repr__(self) -> str:
        """String representation of database."""
        stats = self.get_statistics()
        return (
            f"TrialDatabase(trials={stats.get('total_trials', 0)}, "
            f"converged={stats.get('converged_trials', 0)}, "
            f"rate={stats.get('convergence_rate', 0):.2%})"
        )
