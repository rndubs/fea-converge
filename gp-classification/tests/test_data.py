"""
Tests for data management system.
"""

import pytest
import torch
from pathlib import Path
import tempfile

from gp_classification.data import TrialDatabase, SimulationTrial


@pytest.fixture
def parameter_bounds():
    """Standard parameter bounds for testing."""
    return {
        "penalty_stiffness": (1e3, 1e8),
        "gap_tolerance": (1e-9, 1e-6),
        "absolute_tolerance": (1e-12, 1e-8),
    }


@pytest.fixture
def sample_parameters():
    """Sample parameters within bounds."""
    return {
        "penalty_stiffness": 1e5,
        "gap_tolerance": 1e-7,
        "absolute_tolerance": 1e-10,
    }


def test_database_initialization(parameter_bounds):
    """Test database initialization."""
    db = TrialDatabase(parameter_bounds)

    assert len(db) == 0
    assert db.parameter_names == list(parameter_bounds.keys())
    assert db.get_convergence_rate() == 0.0


def test_add_trial(parameter_bounds, sample_parameters):
    """Test adding trials to database."""
    db = TrialDatabase(parameter_bounds)

    # Add successful trial
    trial_id = db.add_trial(
        parameters=sample_parameters,
        converged=True,
        objective_value=25.5,
        iteration_count=25,
    )

    assert trial_id == 0
    assert len(db) == 1
    assert db.get_convergence_rate() == 1.0

    # Add failed trial
    trial_id = db.add_trial(
        parameters=sample_parameters,
        converged=False,
    )

    assert trial_id == 1
    assert len(db) == 2
    assert db.get_convergence_rate() == 0.5


def test_parameter_validation(parameter_bounds):
    """Test parameter bound validation."""
    db = TrialDatabase(parameter_bounds)

    # Invalid parameter name
    with pytest.raises(ValueError, match="Unknown parameter"):
        db.add_trial(
            parameters={"invalid_param": 1.0},
            converged=True,
        )

    # Out of bounds
    with pytest.raises(ValueError, match="out of bounds"):
        db.add_trial(
            parameters={
                "penalty_stiffness": 1e10,  # Too high
                "gap_tolerance": 1e-7,
                "absolute_tolerance": 1e-10,
            },
            converged=True,
        )


def test_get_training_data(parameter_bounds, sample_parameters):
    """Test training data extraction."""
    db = TrialDatabase(parameter_bounds)

    # Add multiple trials
    for i in range(5):
        converged = i % 2 == 0
        obj = float(i * 10) if converged else None
        db.add_trial(
            parameters=sample_parameters,
            converged=converged,
            objective_value=obj,
        )

    # Get all trials
    X_all, y_converged, _ = db.get_training_data(converged_only=False)
    assert X_all.shape == (5, 3)
    assert y_converged.shape == (5, 1)

    # Get only successful trials
    X_success, y_conv, y_obj = db.get_training_data(converged_only=True)
    assert X_success.shape[0] == 3  # 3 converged trials (0, 2, 4)
    assert y_obj.shape == (3, 1)


def test_best_trial(parameter_bounds, sample_parameters):
    """Test finding best trial."""
    db = TrialDatabase(parameter_bounds)

    # No trials yet
    assert db.get_best_trial() is None

    # Add trials with different objectives
    db.add_trial(parameters=sample_parameters, converged=True, objective_value=30.0)
    db.add_trial(parameters=sample_parameters, converged=True, objective_value=20.0)
    db.add_trial(parameters=sample_parameters, converged=True, objective_value=25.0)

    best = db.get_best_trial()
    assert best is not None
    assert best.objective_value == 20.0


def test_save_load(parameter_bounds, sample_parameters):
    """Test database save/load functionality."""
    db = TrialDatabase(parameter_bounds)

    # Add some trials
    db.add_trial(parameters=sample_parameters, converged=True, objective_value=25.0)
    db.add_trial(parameters=sample_parameters, converged=False)

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_db.csv"
        db.save(filepath)

        # Load back
        db_loaded = TrialDatabase.load(filepath, parameter_bounds)

        assert len(db_loaded) == len(db)
        assert db_loaded.get_convergence_rate() == db.get_convergence_rate()


def test_statistics(parameter_bounds, sample_parameters):
    """Test statistics computation."""
    db = TrialDatabase(parameter_bounds)

    # Add trials
    for i in range(10):
        converged = i < 7  # 70% convergence rate
        obj = float(20 + i) if converged else None
        db.add_trial(parameters=sample_parameters, converged=converged, objective_value=obj)

    stats = db.get_statistics()

    assert stats["total_trials"] == 10
    assert stats["converged_trials"] == 7
    assert stats["failed_trials"] == 3
    assert stats["convergence_rate"] == 0.7
    assert stats["best_objective"] == 20.0
    assert "mean_objective" in stats
    assert "std_objective" in stats
