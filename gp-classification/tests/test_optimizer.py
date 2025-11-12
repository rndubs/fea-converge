"""
Tests for GP Classification optimizer.
"""

import pytest
from gp_classification.optimizer import GPClassificationOptimizer
from gp_classification.mock_solver import MockSmithSolver, get_default_parameter_bounds


@pytest.fixture
def mock_solver():
    """Create mock solver for testing."""
    return MockSmithSolver(random_seed=42, noise_level=0.05, difficulty="easy")


@pytest.fixture
def parameter_bounds():
    """Get default parameter bounds."""
    return get_default_parameter_bounds()


def test_optimizer_initialization(parameter_bounds, mock_solver):
    """Test optimizer initialization."""

    def simulator(params):
        return mock_solver.simulate(params)

    optimizer = GPClassificationOptimizer(
        parameter_bounds=parameter_bounds,
        simulator=simulator,
        n_initial_samples=10,
        verbose=False,
    )

    assert optimizer is not None
    assert len(optimizer.database) == 0
    assert optimizer.iteration == 0


def test_initial_sampling(parameter_bounds, mock_solver):
    """Test initial Sobol sampling."""

    def simulator(params):
        return mock_solver.simulate(params)

    optimizer = GPClassificationOptimizer(
        parameter_bounds=parameter_bounds,
        simulator=simulator,
        n_initial_samples=10,
        verbose=False,
    )

    # Run just initialization
    optimizer._initialize_with_sobol()

    assert len(optimizer.database) == 10
    assert optimizer.iteration == 10
    assert optimizer.database.get_convergence_rate() > 0  # Should have some successes


def test_model_update(parameter_bounds, mock_solver):
    """Test model updating after initial sampling."""

    def simulator(params):
        return mock_solver.simulate(params)

    optimizer = GPClassificationOptimizer(
        parameter_bounds=parameter_bounds,
        simulator=simulator,
        n_initial_samples=15,
        verbose=False,
    )

    optimizer._initialize_with_sobol()
    optimizer._update_models()

    assert optimizer.dual_model is not None
    assert optimizer.dual_model.is_trained


def test_optimization_loop(parameter_bounds, mock_solver):
    """Test full optimization loop."""

    def simulator(params):
        return mock_solver.simulate(params)

    optimizer = GPClassificationOptimizer(
        parameter_bounds=parameter_bounds,
        simulator=simulator,
        n_initial_samples=15,
        phase1_end=20,
        phase2_end=30,
        verbose=False,
    )

    # Run optimization for 35 iterations total
    best_params = optimizer.optimize(n_iterations=35)

    assert best_params is not None
    assert len(optimizer.database) == 35
    assert optimizer.iteration == 35

    # Check that we have a best trial
    best_trial = optimizer.database.get_best_trial()
    assert best_trial is not None
    assert best_trial.objective_value is not None


def test_phase_transitions(parameter_bounds, mock_solver):
    """Test that optimizer transitions through phases correctly."""

    def simulator(params):
        return mock_solver.simulate(params)

    optimizer = GPClassificationOptimizer(
        parameter_bounds=parameter_bounds,
        simulator=simulator,
        n_initial_samples=10,
        phase1_end=15,
        phase2_end=25,
        verbose=False,
    )

    # Phase 1
    optimizer.optimize(n_iterations=15)
    stats = optimizer.get_statistics()
    assert stats["phase"] == "exploration"

    # Phase 2
    optimizer.optimize(n_iterations=25)
    stats = optimizer.get_statistics()
    assert stats["phase"] == "boundary_refinement"

    # Phase 3
    optimizer.optimize(n_iterations=30)
    stats = optimizer.get_statistics()
    assert stats["phase"] == "exploitation"


def test_convergence_improvement(parameter_bounds, mock_solver):
    """Test that convergence rate improves over iterations."""

    def simulator(params):
        return mock_solver.simulate(params)

    optimizer = GPClassificationOptimizer(
        parameter_bounds=parameter_bounds,
        simulator=simulator,
        n_initial_samples=20,
        verbose=False,
    )

    # Get initial convergence rate
    optimizer._initialize_with_sobol()
    initial_rate = optimizer.database.get_convergence_rate()

    # Run more iterations
    optimizer.optimize(n_iterations=50)
    final_rate = optimizer.database.get_convergence_rate()

    # Convergence rate should improve (or at least not degrade significantly)
    # With easy difficulty, we expect good convergence
    assert final_rate >= initial_rate - 0.1  # Allow some variance
