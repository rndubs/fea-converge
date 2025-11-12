"""
Beta schedule computation for CONFIG algorithm.

Implements the theoretical β schedule from CONFIG paper:
β_n = 2 log(π² n² / 6δ)
"""

import numpy as np
from typing import Union


def compute_beta(
    n: int,
    delta: float = 0.1,
    min_beta: float = 0.1
) -> float:
    """
    Compute theoretical β schedule for iteration n.
    
    The β parameter controls the width of confidence bounds in the LCB
    acquisition function. Higher β leads to wider bounds (more exploration).
    
    Formula: β_n = 2 log(π² n² / 6δ)
    
    Args:
        n: Current iteration number (must be >= 1)
        delta: Failure probability, typically 0.1 (10% chance of failure)
        min_beta: Minimum β value for numerical stability
        
    Returns:
        β value for iteration n
        
    Example:
        >>> beta = compute_beta(10, delta=0.1)
        >>> print(f"β at iteration 10: {beta:.2f}")
    """
    if n < 1:
        raise ValueError("Iteration n must be >= 1")
    if not (0 < delta < 1):
        raise ValueError("Delta must be in (0, 1)")
    
    beta = 2 * np.log(np.pi**2 * n**2 / (6 * delta))
    return max(beta, min_beta)


def compute_beta_schedule(
    n_iterations: int,
    delta: float = 0.1,
    min_beta: float = 0.1
) -> np.ndarray:
    """
    Compute β schedule for multiple iterations.
    
    Args:
        n_iterations: Number of iterations
        delta: Failure probability
        min_beta: Minimum β value
        
    Returns:
        Array of β values for iterations 1 to n_iterations
    """
    return np.array([
        compute_beta(i, delta, min_beta)
        for i in range(1, n_iterations + 1)
    ])


def adaptive_beta_adjustment(
    current_beta: float,
    feasible_set_size: float,
    target_size: float = 0.1,
    adjustment_factor: float = 1.5
) -> float:
    """
    Adaptively adjust β if optimistic feasible set is too small.
    
    If F_opt is empty or very small, we increase β to expand the
    optimistic feasible set, enabling continued exploration.
    
    Args:
        current_beta: Current β value
        feasible_set_size: Estimated size/volume of F_opt (0 to 1)
        target_size: Target minimum size for F_opt
        adjustment_factor: Multiplier for β when adjusting
        
    Returns:
        Adjusted β value
    """
    if feasible_set_size < target_size:
        return current_beta * adjustment_factor
    return current_beta
