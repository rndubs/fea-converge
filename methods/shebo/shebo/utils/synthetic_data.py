"""Synthetic data generation for testing SHEBO components."""

from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
from shebo.utils.black_box_solver import BlackBoxFEASolver


def generate_synthetic_dataset(
    n_samples: int = 200,
    n_params: int = 4,
    noise_level: float = 0.1,
    random_seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Generate synthetic dataset for training and testing.

    Args:
        n_samples: Number of samples to generate
        n_params: Number of parameters
        noise_level: Noise level for stochasticity
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with:
            - X: Parameter arrays of shape (n_samples, n_params)
            - y_convergence: Convergence labels of shape (n_samples, 1)
            - y_performance: Performance values of shape (n_samples, 1)
            - outputs: List of full output dictionaries
    """
    rng = np.random.RandomState(random_seed)

    # Define parameter bounds
    # Example: [penalty, tolerance, timestep, damping]
    bounds = np.array([
        [1e6, 1e10],    # penalty parameter
        [1e-8, 1e-4],   # tolerance
        [0.0, 1.0],     # timestep (normalized)
        [0.0, 1.0]      # damping (normalized)
    ])

    # Generate random samples
    X = rng.uniform(
        bounds[:n_params, 0],
        bounds[:n_params, 1],
        size=(n_samples, n_params)
    )

    # Create solver
    solver = BlackBoxFEASolver(noise_level=noise_level, random_seed=random_seed)

    # Evaluate all samples
    y_convergence = []
    y_performance = []
    outputs = []

    for i, params in enumerate(X):
        output = solver.solve(params)
        outputs.append(output)

        y_convergence.append(1.0 if output['convergence_status'] else 0.0)
        y_performance.append(output['iterations'])

    return {
        'X': X,
        'y_convergence': np.array(y_convergence).reshape(-1, 1),
        'y_performance': np.array(y_performance).reshape(-1, 1),
        'outputs': outputs
    }


def generate_train_val_test_split(
    n_train: int = 150,
    n_val: int = 30,
    n_test: int = 50,
    n_params: int = 4,
    noise_level: float = 0.1,
    random_seed: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Generate train, validation, and test datasets.

    Args:
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        n_params: Number of parameters
        noise_level: Noise level
        random_seed: Random seed

    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    total_samples = n_train + n_val + n_test

    # Generate full dataset
    full_data = generate_synthetic_dataset(
        n_samples=total_samples,
        n_params=n_params,
        noise_level=noise_level,
        random_seed=random_seed
    )

    # Split into train/val/test
    train_data = {
        'X': full_data['X'][:n_train],
        'y_convergence': full_data['y_convergence'][:n_train],
        'y_performance': full_data['y_performance'][:n_train],
        'outputs': full_data['outputs'][:n_train]
    }

    val_data = {
        'X': full_data['X'][n_train:n_train+n_val],
        'y_convergence': full_data['y_convergence'][n_train:n_train+n_val],
        'y_performance': full_data['y_performance'][n_train:n_train+n_val],
        'outputs': full_data['outputs'][n_train:n_train+n_val]
    }

    test_data = {
        'X': full_data['X'][n_train+n_val:],
        'y_convergence': full_data['y_convergence'][n_train+n_val:],
        'y_performance': full_data['y_performance'][n_train+n_val:],
        'outputs': full_data['outputs'][n_train+n_val:]
    }

    return train_data, val_data, test_data


def save_synthetic_dataset(
    data: Dict[str, np.ndarray],
    filepath: str
) -> None:
    """Save synthetic dataset to file.

    Args:
        data: Dataset dictionary
        filepath: Path to save file (.npz)
    """
    # Separate outputs (can't save dicts in npz easily)
    outputs = data.pop('outputs', None)

    # Save arrays
    np.savez(filepath, **data)

    # Restore outputs
    if outputs is not None:
        data['outputs'] = outputs

    print(f"Dataset saved to {filepath}")


def load_synthetic_dataset(filepath: str) -> Dict[str, np.ndarray]:
    """Load synthetic dataset from file.

    Args:
        filepath: Path to .npz file

    Returns:
        Dataset dictionary
    """
    loaded = np.load(filepath)
    data = {key: loaded[key] for key in loaded.files}
    print(f"Dataset loaded from {filepath}")
    return data
