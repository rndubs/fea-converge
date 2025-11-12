"""Tests for objective function."""

import pytest
from fr_bo.objective import ObjectiveFunction, extract_objective_from_result, compute_success_metrics


def test_objective_function_converged():
    """Test objective function for converged simulation."""
    obj_func = ObjectiveFunction()

    # Fast convergence should have low objective
    objective = obj_func.compute(
        converged=True,
        iterations=20,
        max_iterations=100,
        time_elapsed=1.0,
        timeout=100.0,
        severe_instability=False,
    )

    assert objective > 0
    assert objective < 1.0  # Should be relatively low


def test_objective_function_failed():
    """Test objective function for failed simulation."""
    obj_func = ObjectiveFunction()

    # First add a successful trial
    obj_success = obj_func.compute(
        converged=True,
        iterations=50,
        max_iterations=100,
        time_elapsed=5.0,
        timeout=100.0,
    )

    # Then test failed trial
    objective = obj_func.compute(
        converged=False,
        iterations=100,
        max_iterations=100,
        time_elapsed=10.0,
        timeout=100.0,
        severe_instability=False,
    )

    # Failed objective should be much larger
    assert objective > obj_success
    assert objective > 10.0  # Convergence penalty dominates


def test_objective_function_severe_instability():
    """Test objective function with severe instability."""
    obj_func = ObjectiveFunction()

    # Add successful trial first
    obj_func.compute(
        converged=True,
        iterations=50,
        max_iterations=100,
        time_elapsed=5.0,
        timeout=100.0,
    )

    # Failed with severe instability
    obj_severe = obj_func.compute(
        converged=False,
        iterations=100,
        max_iterations=100,
        time_elapsed=10.0,
        timeout=100.0,
        severe_instability=True,
    )

    # Failed without severe instability
    obj_normal = obj_func.compute(
        converged=False,
        iterations=100,
        max_iterations=100,
        time_elapsed=10.0,
        timeout=100.0,
        severe_instability=False,
    )

    # Severe instability should have higher penalty
    assert obj_severe > obj_normal


def test_objective_early_success_reward():
    """Test reward for early successful convergence."""
    obj_func = ObjectiveFunction()

    # Very fast convergence
    obj_fast = obj_func.compute(
        converged=True,
        iterations=10,
        max_iterations=100,
        time_elapsed=0.5,
        timeout=100.0,
    )

    # Slower convergence
    obj_slow = obj_func.compute(
        converged=True,
        iterations=80,
        max_iterations=100,
        time_elapsed=8.0,
        timeout=100.0,
    )

    # Fast should be better
    assert obj_fast < obj_slow


def test_extract_objective_from_result():
    """Test extracting objective from result dict."""
    result = {
        "converged": True,
        "iterations": 30,
        "max_iterations": 100,
        "time_elapsed": 3.0,
        "timeout": 100.0,
        "severe_instability": False,
    }

    objective = extract_objective_from_result(result)

    assert isinstance(objective, float)
    assert objective > 0


def test_compute_success_metrics():
    """Test computing success metrics."""
    results = [
        {
            "converged": True,
            "iterations": 30,
            "time_elapsed": 3.0,
        },
        {
            "converged": True,
            "iterations": 40,
            "time_elapsed": 4.0,
        },
        {
            "converged": False,
            "iterations": 100,
            "time_elapsed": 10.0,
        },
    ]

    metrics = compute_success_metrics(results)

    assert metrics["success_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["total_trials"] == 3
    assert metrics["successful_trials"] == 2
    assert metrics["mean_iterations"] == pytest.approx(35.0)


def test_compute_success_metrics_empty():
    """Test metrics with empty results."""
    metrics = compute_success_metrics([])

    assert metrics["success_rate"] == 0.0
    assert metrics["total_trials"] == 0
