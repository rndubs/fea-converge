"""
Shared fixtures and configuration for FR-BO tests.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


@pytest.fixture
def simple_bounds():
    """Simple 2D parameter bounds for testing."""
    return torch.tensor([[0.0, 1.0], [0.0, 1.0]])


@pytest.fixture
def sample_train_x():
    """Sample training data (10 points in 2D)."""
    torch.manual_seed(RANDOM_SEED)
    return torch.rand(10, 2)


@pytest.fixture
def sample_train_y():
    """Sample training objectives (10 values)."""
    torch.manual_seed(RANDOM_SEED)
    return torch.randn(10, 1)


@pytest.fixture
def sample_failure_labels():
    """Sample failure labels (10 binary labels)."""
    torch.manual_seed(RANDOM_SEED)
    # Mix of successes and failures
    labels = torch.zeros(10, 1)
    labels[[2, 5, 8]] = 1.0  # 3 failures out of 10
    return labels


@pytest.fixture
def test_points():
    """Test points for prediction (5 points in 2D)."""
    torch.manual_seed(RANDOM_SEED)
    return torch.rand(5, 2)


@pytest.fixture
def mock_simulation_result():
    """Mock simulation result for testing."""
    from dataclasses import dataclass

    @dataclass
    class MockResult:
        converged: bool
        final_residual: float
        iterations: int
        objective_value: float
        failed: bool

    return MockResult(
        converged=True,
        final_residual=1e-8,
        iterations=50,
        objective_value=0.5,
        failed=False
    )


@pytest.fixture
def mock_failed_result():
    """Mock failed simulation result."""
    from dataclasses import dataclass

    @dataclass
    class MockResult:
        converged: bool
        final_residual: float
        iterations: int
        objective_value: float
        failed: bool

    return MockResult(
        converged=False,
        final_residual=1.0,
        iterations=1000,
        objective_value=float('inf'),
        failed=True
    )


@pytest.fixture
def simple_test_function():
    """Simple 2D test function for optimization testing."""
    def test_func(x: torch.Tensor) -> torch.Tensor:
        """
        Branin function (modified for 2D testing).
        Has multiple local minima.
        """
        x1 = x[..., 0]
        x2 = x[..., 1]
        # Scale to appropriate range
        x1_scaled = 15 * x1 - 5
        x2_scaled = 15 * x2

        a = 1.0
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6.0
        s = 10.0
        t = 1 / (8 * np.pi)

        term1 = a * (x2_scaled - b * x1_scaled**2 + c * x1_scaled - r)**2
        term2 = s * (1 - t) * torch.cos(x1_scaled)
        term3 = s

        return term1 + term2 + term3

    return test_func


@pytest.fixture
def optimization_config():
    """Basic optimization configuration for testing."""
    from fr_bo.optimizer import OptimizationConfig

    return OptimizationConfig(
        n_sobol_trials=10,
        n_frbo_trials=20,
        random_seed=RANDOM_SEED,
        max_iterations=100,
        timeout=60.0,
        convergence_tolerance=1e-3,
        convergence_patience=5
    )


def approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    """Check if two floats are approximately equal."""
    return abs(a - b) < tol


def tensor_approx_equal(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-6) -> bool:
    """Check if two tensors are approximately equal."""
    return torch.allclose(a, b, atol=tol, rtol=tol)


@pytest.fixture
def trained_dual_gp(sample_train_x, sample_train_y, sample_failure_labels):
    """Create and train a dual GP system for testing."""
    from fr_bo.gp_models import DualGPSystem

    dual_gp = DualGPSystem(sample_train_x, sample_train_y, sample_failure_labels)
    dual_gp.train_models(gp_restarts=1, classifier_epochs=10)
    return dual_gp
