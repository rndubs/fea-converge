"""
Sampling utilities for CONFIG initialization.
"""

import numpy as np
from typing import Tuple
from scipy.stats import qmc


def latin_hypercube_sampling(
    bounds: np.ndarray,
    n_samples: int,
    seed: int = None
) -> np.ndarray:
    """
    Generate Latin Hypercube samples for parameter space initialization.
    
    LHS provides better space-filling coverage than random sampling.
    
    Args:
        bounds: Array of shape (n_params, 2) with [lower, upper] bounds
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_samples, n_params) with sampled points
        
    Example:
        >>> bounds = np.array([[0, 1], [0, 1]])  # 2D unit square
        >>> samples = latin_hypercube_sampling(bounds, 10, seed=42)
        >>> print(samples.shape)
        (10, 2)
    """
    n_params = bounds.shape[0]
    
    # Create Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    
    # Generate samples in [0, 1]^d
    samples_unit = sampler.random(n=n_samples)
    
    # Scale to actual bounds
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    samples = qmc.scale(samples_unit, lower_bounds, upper_bounds)
    
    return samples


def sobol_sampling(
    bounds: np.ndarray,
    n_samples: int,
    seed: int = None
) -> np.ndarray:
    """
    Generate Sobol sequence samples for low-discrepancy coverage.
    
    Sobol sequences provide better uniformity than random or LHS
    for certain applications.
    
    Args:
        bounds: Array of shape (n_params, 2) with [lower, upper] bounds
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_samples, n_params) with sampled points
    """
    n_params = bounds.shape[0]
    
    # Create Sobol sampler
    sampler = qmc.Sobol(d=n_params, scramble=True, seed=seed)
    
    # Generate samples in [0, 1]^d
    samples_unit = sampler.random(n=n_samples)
    
    # Scale to actual bounds
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    samples = qmc.scale(samples_unit, lower_bounds, upper_bounds)
    
    return samples


def generate_candidate_set(
    bounds: np.ndarray,
    n_candidates: int = 1000,
    method: str = "sobol",
    seed: int = None
) -> np.ndarray:
    """
    Generate a candidate set for acquisition optimization.
    
    Args:
        bounds: Parameter bounds
        n_candidates: Number of candidate points
        method: Sampling method ('sobol', 'lhs', or 'random')
        seed: Random seed
        
    Returns:
        Candidate points array
    """
    if method == "sobol":
        return sobol_sampling(bounds, n_candidates, seed)
    elif method == "lhs":
        return latin_hypercube_sampling(bounds, n_candidates, seed)
    elif method == "random":
        rng = np.random.default_rng(seed)
        n_params = bounds.shape[0]
        samples = rng.uniform(
            low=bounds[:, 0],
            high=bounds[:, 1],
            size=(n_candidates, n_params)
        )
        return samples
    else:
        raise ValueError(f"Unknown sampling method: {method}")
