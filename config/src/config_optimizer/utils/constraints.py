"""
Constraint formulation utilities for CONFIG.

Provides continuous constraint formulations suitable for GP modeling.
"""

import numpy as np
from typing import Dict, Any


def convergence_constraint(
    final_residual: float,
    tolerance: float,
    max_residual: float = 1e10
) -> float:
    """
    Compute continuous convergence constraint value.
    
    Formula: c(x) = log10(final_residual) - log10(tolerance)
    
    - c < 0: Converged (satisfied)
    - c > 0: Failed to converge (violated)
    - Magnitude indicates how far from convergence
    
    Args:
        final_residual: Final residual norm from simulation
        tolerance: Convergence tolerance threshold
        max_residual: Maximum residual value (for diverged cases)
        
    Returns:
        Constraint value (negative = satisfied, positive = violated)
        
    Example:
        >>> c = convergence_constraint(1e-10, 1e-8)
        >>> print(f"Constraint value: {c:.2f} (converged)")
        >>> c = convergence_constraint(1e-6, 1e-8)
        >>> print(f"Constraint value: {c:.2f} (violated)")
    """
    # Clip residual to avoid log(0) or log(inf)
    residual = np.clip(final_residual, 1e-20, max_residual)
    tolerance = max(tolerance, 1e-20)
    
    return np.log10(residual) - np.log10(tolerance)


def iteration_constraint(
    iterations: int,
    max_iterations: int
) -> float:
    """
    Compute iteration budget constraint.
    
    Formula: c(x) = iterations - max_iterations
    
    Args:
        iterations: Number of iterations used
        max_iterations: Maximum allowed iterations
        
    Returns:
        Constraint value (negative = within budget, positive = exceeded)
    """
    return float(iterations - max_iterations)


def penetration_constraint(
    max_penetration: float,
    penetration_limit: float
) -> float:
    """
    Compute physics validity constraint based on penetration.
    
    Formula: c(x) = max_penetration - penetration_limit
    
    Args:
        max_penetration: Maximum penetration observed
        penetration_limit: Acceptable penetration threshold
        
    Returns:
        Constraint value (negative = valid, positive = invalid)
    """
    return float(max_penetration - penetration_limit)


def compute_multiple_constraints(
    simulation_results: Dict[str, Any],
    constraint_configs: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Compute multiple constraint values from simulation results.
    
    Args:
        simulation_results: Dictionary with simulation outputs
            - 'final_residual': float
            - 'iterations': int
            - 'max_penetration': float (optional)
            - 'converged': bool
            
        constraint_configs: Configuration for each constraint type
            - 'convergence': {'tolerance': float}
            - 'iteration': {'max_iterations': int}
            - 'penetration': {'limit': float} (optional)
            
    Returns:
        Dictionary mapping constraint names to constraint values
        
    Example:
        >>> results = {
        ...     'final_residual': 1e-9,
        ...     'iterations': 50,
        ...     'converged': True
        ... }
        >>> configs = {
        ...     'convergence': {'tolerance': 1e-8},
        ...     'iteration': {'max_iterations': 100}
        ... }
        >>> constraints = compute_multiple_constraints(results, configs)
    """
    constraint_values = {}
    
    # Convergence constraint
    if 'convergence' in constraint_configs:
        config = constraint_configs['convergence']
        constraint_values['convergence'] = convergence_constraint(
            simulation_results.get('final_residual', 1e10),
            config['tolerance']
        )
    
    # Iteration constraint
    if 'iteration' in constraint_configs:
        config = constraint_configs['iteration']
        constraint_values['iteration'] = iteration_constraint(
            simulation_results.get('iterations', config['max_iterations'] + 1),
            config['max_iterations']
        )
    
    # Penetration constraint
    if 'penetration' in constraint_configs:
        config = constraint_configs['penetration']
        if 'max_penetration' in simulation_results:
            constraint_values['penetration'] = penetration_constraint(
                simulation_results['max_penetration'],
                config['limit']
            )
    
    return constraint_values


def constraint_violation_amount(constraint_value: float) -> float:
    """
    Compute the violation amount for a constraint.
    
    For cumulative violation tracking: V_t = Σ max(0, c(x_i))
    
    Args:
        constraint_value: Constraint value
        
    Returns:
        Violation amount (0 if satisfied, positive if violated)
    """
    return max(0.0, constraint_value)
