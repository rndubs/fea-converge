"""
Integration test for complete CONFIG optimizer.
"""

import pytest
import numpy as np
from src.config_optimizer.core.controller import CONFIGController, CONFIGConfig
from src.config_optimizer.solvers.black_box_solver import BlackBoxSolver
from src.config_optimizer.utils.constraints import convergence_constraint


def test_config_optimizer_branin():
    """Test CONFIG optimizer on Branin function."""
    # Create black box solver
    solver = BlackBoxSolver(problem_type="branin", noise_level=0.01, seed=42)
    
    # Define objective function
    def objective_function(x):
        result = solver.evaluate(x)
        return {
            'objective_value': result.objective_value,
            'final_residual': result.final_residual,
            'iterations': result.iterations,
            'converged': result.converged
        }
    
    # Create CONFIG config
    config = CONFIGConfig(
        bounds=solver.bounds,
        constraint_configs={
            'convergence': {'tolerance': 1e-8}
        },
        delta=0.1,
        n_init=10,
        n_max=30,
        acquisition_method="discrete",
        seed=42
    )
    
    # Create and run optimizer
    optimizer = CONFIGController(config, objective_function)
    results = optimizer.optimize()
    
    # Check results
    assert results['n_evaluations'] <= 30
    assert results['best_x'] is not None or results['n_feasible'] > 0
    assert 'violation_statistics' in results
    assert results['violation_bound_check']['status'] in ['OK', 'WARNING']
    
    print("\nOptimization Results:")
    print(f"Best feasible value: {results['best_y']:.4f}")
    print(f"Number of evaluations: {results['n_evaluations']}")
    print(f"Number feasible: {results['n_feasible']}")
    print(f"Violation statistics: {results['violation_statistics']}")


def test_config_optimizer_quadratic():
    """Test CONFIG optimizer on simple quadratic."""
    # Create black box solver
    solver = BlackBoxSolver(problem_type="quadratic", noise_level=0.01, seed=42)
    
    # Define objective function
    def objective_function(x):
        result = solver.evaluate(x)
        return {
            'objective_value': result.objective_value,
            'final_residual': result.final_residual,
            'iterations': result.iterations,
            'converged': result.converged
        }
    
    # Create CONFIG config
    config = CONFIGConfig(
        bounds=solver.bounds,
        constraint_configs={
            'convergence': {'tolerance': 1e-8}
        },
        delta=0.1,
        n_init=10,
        n_max=25,
        acquisition_method="discrete",
        seed=42
    )
    
    # Create and run optimizer
    optimizer = CONFIGController(config, objective_function)
    results = optimizer.optimize()
    
    # Check results
    assert results['best_x'] is not None
    assert results['n_feasible'] > 0
    
    # For quadratic, should find solution near origin
    if results['best_x'] is not None:
        distance_to_origin = np.linalg.norm(results['best_x'])
        print(f"\nDistance to origin: {distance_to_origin:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
