"""
Utility functions for FR-BO optimization.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import json
import pickle
from pathlib import Path
import torch


def save_optimization_results(results: Dict, output_dir: str, prefix: str = "frbo"):
    """
    Save optimization results to disk.

    Args:
        results: Dictionary of optimization results
        output_dir: Output directory path
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON summary (without trials)
    summary = {
        "best_objective": results["best_objective"],
        "best_parameters": results["best_parameters"],
        "best_trial_number": results["best_trial_number"],
        "metrics": results["metrics"],
        "total_trials": results["total_trials"],
    }

    if results.get("parameter_importance") is not None:
        summary["parameter_importance"] = results["parameter_importance"].tolist()

    with open(output_path / f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full results with trials as pickle
    with open(output_path / f"{prefix}_full_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved to {output_path}")


def load_optimization_results(results_path: str) -> Dict:
    """
    Load optimization results from disk.

    Args:
        results_path: Path to pickle file

    Returns:
        Dictionary of results
    """
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    return results


def normalize_parameters(params: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Normalize parameters to [0, 1] based on bounds.

    Args:
        params: Parameter array
        bounds: (2, d) array of (lower, upper) bounds

    Returns:
        Normalized parameters
    """
    lower, upper = bounds[0], bounds[1]
    return (params - lower) / (upper - lower + 1e-10)


def denormalize_parameters(params_norm: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Denormalize parameters from [0, 1] to original scale.

    Args:
        params_norm: Normalized parameters
        bounds: (2, d) array of (lower, upper) bounds

    Returns:
        Denormalized parameters
    """
    lower, upper = bounds[0], bounds[1]
    return params_norm * (upper - lower) + lower


def compute_pairwise_distances(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute pairwise Euclidean distances.

    Args:
        X: (n, d) array
        Y: Optional (m, d) array (if None, compute distances within X)

    Returns:
        Distance matrix
    """
    if Y is None:
        Y = X

    from scipy.spatial.distance import cdist
    return cdist(X, Y, metric="euclidean")


def stratified_split(
    data: List[Any],
    labels: List[int],
    test_fraction: float = 0.2,
    random_seed: Optional[int] = None,
) -> tuple:
    """
    Perform stratified train-test split.

    Args:
        data: List of data samples
        labels: List of labels (0/1 for failure/success)
        test_fraction: Fraction for test set
        random_seed: Random seed

    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(
        data,
        labels,
        test_size=test_fraction,
        stratify=labels,
        random_state=random_seed,
    )


def compute_cross_validation_score(
    X: np.ndarray,
    y: np.ndarray,
    model_class: Any,
    n_folds: int = 5,
    random_seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute cross-validation scores.

    Args:
        X: Feature matrix
        y: Labels
        model_class: Model class to instantiate
        n_folds: Number of CV folds
        random_seed: Random seed

    Returns:
        Dictionary of scores
    """
    from sklearn.model_selection import cross_validate

    cv_results = cross_validate(
        model_class,
        X,
        y,
        cv=n_folds,
        scoring=["accuracy", "roc_auc"],
        return_train_score=True,
    )

    return {
        "test_accuracy_mean": np.mean(cv_results["test_accuracy"]),
        "test_accuracy_std": np.std(cv_results["test_accuracy"]),
        "test_auc_mean": np.mean(cv_results["test_roc_auc"]),
        "test_auc_std": np.std(cv_results["test_roc_auc"]),
    }


def set_random_seeds(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    """
    Format time duration as human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_parameter_grid(
    bounds: np.ndarray,
    n_points_per_dim: int = 10,
) -> np.ndarray:
    """
    Create regular grid in parameter space.

    Args:
        bounds: (2, d) array of (lower, upper) bounds
        n_points_per_dim: Number of grid points per dimension

    Returns:
        Grid points array
    """
    lower, upper = bounds[0], bounds[1]
    dim = len(lower)

    # Create 1D grids
    grids_1d = [
        np.linspace(lower[i], upper[i], n_points_per_dim)
        for i in range(dim)
    ]

    # Create meshgrid
    grids = np.meshgrid(*grids_1d, indexing="ij")

    # Flatten and stack
    grid_points = np.column_stack([g.ravel() for g in grids])

    return grid_points


def log_experiment(
    experiment_name: str,
    config: Dict,
    results: Dict,
    log_file: str = "experiments.log",
):
    """
    Log experiment configuration and results.

    Args:
        experiment_name: Name of experiment
        config: Configuration dictionary
        results: Results dictionary
        log_file: Path to log file
    """
    import datetime

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "config": config,
        "results": {
            "best_objective": results.get("best_objective"),
            "total_trials": results.get("total_trials"),
            "success_rate": results.get("metrics", {}).get("overall", {}).get("success_rate"),
        },
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
