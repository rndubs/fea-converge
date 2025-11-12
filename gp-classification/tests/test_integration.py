"""
Integration tests for complete workflow.
"""

import pytest
import torch

from gp_classification import (
    GPClassificationOptimizer,
    TrialDatabase,
    ParameterSuggester,
    PreSimulationValidator,
    OptimizationVisualizer,
)
from gp_classification.mock_solver import MockSmithSolver, get_default_parameter_bounds


@pytest.fixture
def mock_solver():
    """Create mock solver for testing."""
    return MockSmithSolver(random_seed=42, noise_level=0.05, difficulty="medium")


@pytest.fixture
def parameter_bounds():
    """Get default parameter bounds."""
    return get_default_parameter_bounds()


def test_end_to_end_optimization(parameter_bounds, mock_solver):
    """Test complete optimization workflow."""

    # Create simulator function
    def simulator(params):
        return mock_solver.simulate(params)

    # Create optimizer
    optimizer = GPClassificationOptimizer(
        parameter_bounds=parameter_bounds,
        simulator=simulator,
        n_initial_samples=20,
        phase1_end=25,
        phase2_end=35,
        verbose=False,
    )

    # Run optimization
    best_params = optimizer.optimize(n_iterations=40)

    # Verify results
    assert best_params is not None
    assert len(optimizer.database) == 40

    # Check that we found good parameters
    best_trial = optimizer.database.get_best_trial()
    assert best_trial is not None
    assert best_trial.converged
    assert best_trial.objective_value is not None

    # Convergence rate should be reasonable
    stats = optimizer.database.get_statistics()
    assert stats["convergence_rate"] > 0.3  # At least some successes


def test_parameter_suggestion_workflow(parameter_bounds, mock_solver):
    """Test parameter suggestion use case."""

    def simulator(params):
        return mock_solver.simulate(params)

    # Run optimization to build up database
    optimizer = GPClassificationOptimizer(
        parameter_bounds=parameter_bounds,
        simulator=simulator,
        n_initial_samples=25,
        verbose=False,
    )
    optimizer.optimize(n_iterations=30)

    # Create parameter suggester
    suggester = ParameterSuggester(
        database=optimizer.database,
        dual_model=optimizer.dual_model,
        n_clusters=3,
    )

    # Get suggestions
    suggestions = suggester.suggest_parameters()

    assert len(suggestions) > 0
    assert len(suggestions) <= 3  # Should have up to 3 clusters

    # Check suggestion structure
    for suggestion in suggestions:
        assert "parameters" in suggestion
        assert "convergence_probability" in suggestion
        assert "confidence" in suggestion
        assert suggestion["confidence"] in ["high", "medium", "low"]


def test_validation_workflow(parameter_bounds, mock_solver):
    """Test pre-simulation validation use case."""

    def simulator(params):
        return mock_solver.simulate(params)

    # Build database
    optimizer = GPClassificationOptimizer(
        parameter_bounds=parameter_bounds,
        simulator=simulator,
        n_initial_samples=25,
        verbose=False,
    )
    optimizer.optimize(n_iterations=30)

    # Create validator
    validator = PreSimulationValidator(
        database=optimizer.database,
        dual_model=optimizer.dual_model,
        min_convergence_prob=0.3,
    )

    # Test with good parameters (from best trial)
    best_trial = optimizer.database.get_best_trial()
    result = validator.validate(best_trial.parameters)

    assert "approved" in result
    assert "risk_score" in result
    assert "risk_level" in result
    assert result["risk_level"] in ["low", "moderate", "high"]
    assert "recommendation" in result

    # Good parameters should generally be approved or at least moderate risk
    assert result["risk_level"] in ["low", "moderate"]


def test_visualization_workflow(parameter_bounds, mock_solver):
    """Test visualization creation."""

    def simulator(params):
        return mock_solver.simulate(params)

    # Build database
    optimizer = GPClassificationOptimizer(
        parameter_bounds=parameter_bounds,
        simulator=simulator,
        n_initial_samples=20,
        verbose=False,
    )
    optimizer.optimize(n_iterations=30)

    # Create visualizer
    visualizer = OptimizationVisualizer(
        database=optimizer.database,
        dual_model=optimizer.dual_model,
        parameter_names=optimizer.parameter_names,
    )

    # Test convergence landscape plot
    fig = visualizer.plot_convergence_landscape_2d(
        param1_name="penalty_stiffness",
        param2_name="gap_tolerance",
        resolution=20,  # Low resolution for speed
    )
    assert fig is not None

    # Test optimization history
    fig = visualizer.plot_optimization_history()
    assert fig is not None

    # Test parameter importance
    fig = visualizer.plot_parameter_importance()
    assert fig is not None


def test_convergence_landscape_prediction(parameter_bounds, mock_solver):
    """Test convergence landscape prediction."""

    def simulator(params):
        return mock_solver.simulate(params)

    optimizer = GPClassificationOptimizer(
        parameter_bounds=parameter_bounds,
        simulator=simulator,
        n_initial_samples=20,
        verbose=False,
    )
    optimizer.optimize(n_iterations=30)

    # Get landscape
    X, Y, P = optimizer.predict_convergence_landscape(
        param1_name="penalty_stiffness",
        param2_name="gap_tolerance",
        resolution=20,
    )

    assert X.shape == (20, 20)
    assert Y.shape == (20, 20)
    assert P.shape == (20, 20)

    # Probabilities should be in [0, 1]
    assert torch.all((P >= 0) & (P <= 1))
