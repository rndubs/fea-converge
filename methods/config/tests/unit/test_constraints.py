"""
Unit tests for constraint formulations.
"""

import pytest
import numpy as np
from src.config_optimizer.utils.constraints import (
    convergence_constraint,
    iteration_constraint,
    penetration_constraint,
    compute_multiple_constraints,
    constraint_violation_amount
)


def test_convergence_constraint_converged():
    """Test convergence constraint for converged case."""
    c = convergence_constraint(1e-10, 1e-8)
    assert c < 0, "Should be negative for converged case"


def test_convergence_constraint_failed():
    """Test convergence constraint for failed case."""
    c = convergence_constraint(1e-6, 1e-8)
    assert c > 0, "Should be positive for failed case"


def test_iteration_constraint():
    """Test iteration constraint."""
    c = iteration_constraint(50, 100)
    assert c < 0, "Should be negative when within budget"
    
    c = iteration_constraint(150, 100)
    assert c > 0, "Should be positive when over budget"


def test_penetration_constraint():
    """Test penetration constraint."""
    c = penetration_constraint(0.001, 0.01)
    assert c < 0, "Should be negative when valid"
    
    c = penetration_constraint(0.02, 0.01)
    assert c > 0, "Should be positive when invalid"


def test_compute_multiple_constraints():
    """Test computing multiple constraints."""
    results = {
        'final_residual': 1e-9,
        'iterations': 50,
        'converged': True
    }
    
    configs = {
        'convergence': {'tolerance': 1e-8},
        'iteration': {'max_iterations': 100}
    }
    
    constraints = compute_multiple_constraints(results, configs)
    
    assert 'convergence' in constraints
    assert 'iteration' in constraints
    assert constraints['convergence'] < 0
    assert constraints['iteration'] < 0


def test_constraint_violation_amount():
    """Test violation amount calculation."""
    assert constraint_violation_amount(-1.0) == 0.0
    assert constraint_violation_amount(2.0) == 2.0
    assert constraint_violation_amount(0.0) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
