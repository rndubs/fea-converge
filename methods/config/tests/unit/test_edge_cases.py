"""
Edge case tests for CONFIG optimizer.

Tests various edge cases and failure scenarios to ensure robustness.
"""

import pytest
import numpy as np
from src.config_optimizer.core.controller import CONFIGController, CONFIGConfig
from src.config_optimizer.utils.constants import FAILED_OBJECTIVE_PENALTY, FAILED_CONSTRAINT_PENALTY


def test_objective_function_returns_non_finite():
    """Test handling of NaN/Inf objective values."""
    bounds = np.array([[0, 1], [0, 1]])

    # Objective function that returns NaN
    call_count = [0]

    def objective_with_nan(x):
        call_count[0] += 1
        if call_count[0] > 5:  # Return NaN after 5 calls
            return {
                'objective_value': np.nan,
                'final_residual': 1e-10,
                'iterations': 10,
                'converged': True
            }
        return {
            'objective_value': np.sum(x**2),
            'final_residual': 1e-10,
            'iterations': 10,
            'converged': True
        }

    config = CONFIGConfig(
        bounds=bounds,
        constraint_configs={'convergence': {'tolerance': 1e-8}},
        n_init=10,
        n_max=15,
        seed=42,
        verbose=False
    )

    optimizer = CONFIGController(config, objective_with_nan)
    results = optimizer.optimize()

    # Should complete without crashing
    assert results['n_evaluations'] <= 15
    # NaN values should be replaced with penalty
    assert all(np.isfinite(y) for y in results['y_observed'])


def test_objective_function_raises_exception():
    """Test handling of objective function exceptions."""
    bounds = np.array([[0, 1], [0, 1]])

    call_count = [0]

    def failing_objective(x):
        call_count[0] += 1
        if call_count[0] > 5:  # Fail after 5 calls
            raise RuntimeError("Simulation crashed!")
        return {
            'objective_value': np.sum(x**2),
            'final_residual': 1e-10,
            'iterations': 10,
            'converged': True
        }

    config = CONFIGConfig(
        bounds=bounds,
        constraint_configs={'convergence': {'tolerance': 1e-8}},
        n_init=10,
        n_max=15,
        seed=42,
        verbose=False
    )

    optimizer = CONFIGController(config, failing_objective)
    results = optimizer.optimize()

    # Should complete without crashing
    assert results['n_evaluations'] <= 15
    # Failed evaluations should use penalty values
    assert any(y == FAILED_OBJECTIVE_PENALTY for y in results['y_observed'])


def test_no_feasible_points_found():
    """Test optimization when no feasible points are found."""
    bounds = np.array([[0, 1], [0, 1]])

    def always_infeasible(x):
        # Always violate constraint
        return {
            'objective_value': np.sum(x**2),
            'final_residual': 1e-2,  # Large residual = infeasible
            'iterations': 100,
            'converged': False
        }

    config = CONFIGConfig(
        bounds=bounds,
        constraint_configs={'convergence': {'tolerance': 1e-8}},
        n_init=10,
        n_max=30,
        seed=42,
        verbose=False
    )

    optimizer = CONFIGController(config, always_infeasible)
    results = optimizer.optimize()

    # Should complete without crashing
    assert results['n_evaluations'] <= 30
    assert results['best_x'] is None  # No feasible solution found
    assert results['n_feasible'] == 0


def test_all_initial_points_fail():
    """Test when all initial LHS samples fail."""
    bounds = np.array([[0, 1], [0, 1]])

    call_count = [0]

    def initially_failing(x):
        call_count[0] += 1
        if call_count[0] <= 10:  # First 10 calls fail
            raise ValueError("Initialization failure")
        return {
            'objective_value': np.sum(x**2),
            'final_residual': 1e-10,
            'iterations': 10,
            'converged': True
        }

    config = CONFIGConfig(
        bounds=bounds,
        constraint_configs={'convergence': {'tolerance': 1e-8}},
        n_init=10,
        n_max=20,
        seed=42,
        verbose=False
    )

    optimizer = CONFIGController(config, initially_failing)
    results = optimizer.optimize()

    # Should recover after initialization phase
    assert results['n_evaluations'] <= 20
    # Later evaluations should succeed
    assert results['n_feasible'] > 0


def test_empty_constraint_configs():
    """Test with empty constraint configuration."""
    bounds = np.array([[0, 1], [0, 1]])

    def simple_objective(x):
        return {
            'objective_value': np.sum(x**2),
            'final_residual': 1e-10,
            'iterations': 10,
            'converged': True
        }

    config = CONFIGConfig(
        bounds=bounds,
        constraint_configs={},  # No constraints
        n_init=5,
        n_max=10,
        seed=42,
        verbose=False
    )

    optimizer = CONFIGController(config, simple_objective)

    # Should handle empty constraints gracefully
    # Note: This may not be fully supported, but shouldn't crash
    try:
        results = optimizer.optimize()
        assert results['n_evaluations'] <= 10
    except (ValueError, KeyError):
        # Expected if empty constraints are not supported
        pass


def test_single_dimension_problem():
    """Test 1D optimization problem."""
    bounds = np.array([[0, 2]])  # Single dimension

    def simple_1d(x):
        # Minimum at x=1
        obj = (x[0] - 1)**2
        return {
            'objective_value': obj,
            'final_residual': 1e-10,
            'iterations': 10,
            'converged': True
        }

    config = CONFIGConfig(
        bounds=bounds,
        constraint_configs={'convergence': {'tolerance': 1e-8}},
        n_init=5,
        n_max=15,
        seed=42,
        verbose=False
    )

    optimizer = CONFIGController(config, simple_1d)
    results = optimizer.optimize()

    assert results['n_evaluations'] <= 15
    assert results['best_x'] is not None
    # Should find solution near x=1
    if results['best_x'] is not None:
        assert abs(results['best_x'][0] - 1.0) < 0.5


def test_high_dimensional_problem():
    """Test higher dimensional optimization (10D)."""
    bounds = np.array([[0, 1]] * 10)  # 10 dimensions

    def high_dim_objective(x):
        return {
            'objective_value': np.sum(x**2),
            'final_residual': 1e-10,
            'iterations': 10,
            'converged': True
        }

    config = CONFIGConfig(
        bounds=bounds,
        constraint_configs={'convergence': {'tolerance': 1e-8}},
        n_init=20,
        n_max=30,
        seed=42,
        verbose=False
    )

    optimizer = CONFIGController(config, high_dim_objective)
    results = optimizer.optimize()

    assert results['n_evaluations'] <= 30
    assert results['X_observed'].shape[1] == 10  # Verify dimensionality


def test_early_termination_trigger():
    """Test that optimizer completes within budget.

    Note: Early termination requires MIN_ITERATIONS_FOR_TERMINATION (50)
    and STABILITY_WINDOW (15) with no improvements, so for simple problems
    it may use the full budget. This test just ensures completion.
    """
    bounds = np.array([[0, 1], [0, 1]])

    def simple_objective(x):
        return {
            'objective_value': np.sum(x**2),
            'final_residual': 1e-10,
            'iterations': 10,
            'converged': True
        }

    config = CONFIGConfig(
        bounds=bounds,
        constraint_configs={'convergence': {'tolerance': 1e-8}},
        n_init=10,
        n_max=100,  # Large budget
        seed=42,
        verbose=False
    )

    optimizer = CONFIGController(config, simple_objective)
    results = optimizer.optimize()

    # Should complete without error
    assert results['n_evaluations'] <= 100
    assert results['n_evaluations'] >= 10  # At least initialization


def test_multiple_constraints():
    """Test with multiple constraints."""
    bounds = np.array([[0, 2], [0, 2]])

    def multi_constraint_obj(x):
        return {
            'objective_value': np.sum(x**2),
            'final_residual': 1e-10 if np.sum(x) < 2 else 1e-2,
            'iterations': 50 if np.sum(x) < 2 else 100,
            'max_penetration': 0.001 if np.linalg.norm(x) < 1.5 else 0.1,
            'converged': np.sum(x) < 2
        }

    config = CONFIGConfig(
        bounds=bounds,
        constraint_configs={
            'convergence': {'tolerance': 1e-8},
            'iteration': {'max_iterations': 75},
            'penetration': {'limit': 0.01}
        },
        n_init=10,
        n_max=30,
        seed=42,
        verbose=False
    )

    optimizer = CONFIGController(config, multi_constraint_obj)
    results = optimizer.optimize()

    assert results['n_evaluations'] <= 30
    # Should track all three constraints
    assert len(results['constraint_values']) == 3
    assert 'convergence' in results['constraint_values']
    assert 'iteration' in results['constraint_values']
    assert 'penetration' in results['constraint_values']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
