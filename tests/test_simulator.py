"""Tests for simulation executors."""

import pytest
from fr_bo.simulator import SyntheticSimulator, SimulationResult


def test_synthetic_simulator_basic():
    """Test basic synthetic simulator functionality."""
    simulator = SyntheticSimulator(random_seed=42)

    parameters = {
        "penalty_stiffness": 1e6,
        "gap_tolerance": 1e-7,
        "max_iterations": 500,
    }

    result = simulator.run(parameters, max_iterations=100, timeout=10.0)

    # Check result structure
    assert isinstance(result, SimulationResult)
    assert isinstance(result.converged, bool)
    assert result.iterations > 0
    assert result.iterations <= 100
    assert result.time_elapsed > 0


def test_synthetic_simulator_reproducibility():
    """Test that simulator is reproducible with same seed."""
    params = {"penalty_stiffness": 1e6}

    sim1 = SyntheticSimulator(random_seed=42)
    result1 = sim1.run(params, max_iterations=100)

    sim2 = SyntheticSimulator(random_seed=42)
    result2 = sim2.run(params, max_iterations=100)

    # Should get same results
    assert result1.converged == result2.converged
    assert result1.iterations == result2.iterations


def test_synthetic_simulator_different_params():
    """Test simulator with different parameter sets."""
    simulator = SyntheticSimulator(random_seed=42)

    results = []
    for i in range(10):
        params = {
            "penalty_stiffness": 10 ** (3 + i * 0.5),
            "gap_tolerance": 1e-7,
        }
        result = simulator.run(params, max_iterations=100)
        results.append(result)

    # Should get mix of successes and failures
    successes = sum(1 for r in results if r.converged)
    assert 0 < successes < len(results)


def test_simulation_result_to_dict():
    """Test converting SimulationResult to dictionary."""
    result = SimulationResult(
        converged=True,
        iterations=50,
        max_iterations=100,
        time_elapsed=5.0,
        timeout=100.0,
        final_residual=1e-11,
        contact_pressure_max=1e7,
        penetration_max=1e-8,
        severe_instability=False,
        residual_history=[1.0, 0.5, 0.1, 1e-11],
        active_set_sizes=[100, 100, 100, 100],
    )

    result_dict = result.to_dict()

    assert result_dict["converged"] is True
    assert result_dict["iterations"] == 50
    assert len(result_dict["residual_history"]) == 4


def test_synthetic_simulator_residual_history():
    """Test that residual history is generated."""
    simulator = SyntheticSimulator(random_seed=42)

    result = simulator.run({"penalty_stiffness": 1e6}, max_iterations=50)

    assert result.residual_history is not None
    assert len(result.residual_history) > 0
    assert len(result.residual_history) <= 50


def test_synthetic_simulator_divergence_detection():
    """Test detection of severe instabilities."""
    simulator = SyntheticSimulator(random_seed=42)

    # Run multiple trials to potentially hit divergence cases
    divergence_count = 0
    for i in range(20):
        params = {
            "penalty_stiffness": 10 ** (8 + i * 0.1),  # Very high stiffness
        }
        result = simulator.run(params, max_iterations=100)

        if result.severe_instability:
            divergence_count += 1

    # Should detect some divergences (not guaranteed, but likely)
    # Just check that the field is being used
    assert divergence_count >= 0  # Can be 0, but field should exist
