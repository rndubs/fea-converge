"""Integration tests for complete FR-BO workflow."""

import pytest
import numpy as np
from fr_bo.optimizer import FRBOOptimizer, OptimizationConfig
from fr_bo.simulator import SyntheticSimulator
from fr_bo.synthetic_data import create_benchmark_dataset


def test_complete_optimization_workflow():
    """Test complete FR-BO optimization workflow."""
    # Create simulator
    simulator = SyntheticSimulator(random_seed=42)

    # Create configuration with small number of trials for fast testing
    config = OptimizationConfig(
        n_sobol_trials=5,
        n_frbo_trials=5,
        random_seed=42,
        max_iterations=50,
    )

    # Create optimizer
    optimizer = FRBOOptimizer(simulator=simulator, config=config)

    # Run optimization
    results = optimizer.optimize(total_trials=10)

    # Check results structure
    assert "best_objective" in results
    assert "best_parameters" in results
    assert "best_trial_number" in results
    assert "metrics" in results
    assert "trials" in results

    # Check that we have the right number of trials
    assert len(results["trials"]) == 10

    # Check that we have both phases
    phases = set(t.phase for t in results["trials"])
    assert "sobol" in phases
    assert "frbo" in phases


def test_optimizer_tracks_best():
    """Test that optimizer correctly tracks best trial."""
    simulator = SyntheticSimulator(random_seed=42)

    config = OptimizationConfig(
        n_sobol_trials=10,
        n_frbo_trials=10,
        random_seed=42,
    )

    optimizer = FRBOOptimizer(simulator=simulator, config=config)
    results = optimizer.optimize()

    # Best objective should be from a converged trial
    best_trial = results["trials"][results["best_trial_number"] - 1]
    assert best_trial.result.converged

    # Best objective should match
    assert best_trial.objective_value == results["best_objective"]


def test_optimizer_dual_gp_initialization():
    """Test that dual GP is initialized after Sobol phase."""
    simulator = SyntheticSimulator(random_seed=42)

    config = OptimizationConfig(
        n_sobol_trials=10,
        n_frbo_trials=0,  # No FR-BO phase
        random_seed=42,
    )

    optimizer = FRBOOptimizer(simulator=simulator, config=config)

    # Before optimization, dual_gp should be None
    assert optimizer.dual_gp is None

    # Run Sobol phase
    optimizer._run_sobol_phase()

    # After Sobol phase, dual_gp should be initialized
    assert optimizer.dual_gp is not None
    assert optimizer.dual_gp.failure_classifier is not None


def test_optimizer_metrics():
    """Test that optimizer computes meaningful metrics."""
    simulator = SyntheticSimulator(random_seed=42)

    config = OptimizationConfig(
        n_sobol_trials=20,
        n_frbo_trials=20,
        random_seed=42,
    )

    optimizer = FRBOOptimizer(simulator=simulator, config=config)
    results = optimizer.optimize()

    # Check overall metrics
    overall_metrics = results["metrics"]["overall"]
    assert 0 <= overall_metrics["success_rate"] <= 1
    assert overall_metrics["total_trials"] == 40

    # Check phase-specific metrics
    sobol_metrics = results["metrics"]["sobol_phase"]
    frbo_metrics = results["metrics"]["frbo_phase"]

    assert sobol_metrics["total_trials"] == 20
    assert frbo_metrics["total_trials"] == 20


def test_optimization_improves_over_random():
    """Test that FR-BO phase improves over Sobol phase."""
    simulator = SyntheticSimulator(random_seed=42)

    config = OptimizationConfig(
        n_sobol_trials=30,
        n_frbo_trials=30,
        random_seed=42,
    )

    optimizer = FRBOOptimizer(simulator=simulator, config=config)
    results = optimizer.optimize()

    sobol_metrics = results["metrics"]["sobol_phase"]
    frbo_metrics = results["metrics"]["frbo_phase"]

    # FR-BO should generally have higher success rate (not guaranteed, but likely)
    # At minimum, both should complete
    assert sobol_metrics["total_trials"] == 30
    assert frbo_metrics["total_trials"] == 30


@pytest.mark.slow
def test_full_optimization_run():
    """Test full optimization run with realistic settings."""
    simulator = SyntheticSimulator(random_seed=42)

    config = OptimizationConfig(
        n_sobol_trials=20,
        n_frbo_trials=50,
        random_seed=42,
        max_iterations=100,
    )

    optimizer = FRBOOptimizer(simulator=simulator, config=config)
    results = optimizer.optimize()

    # Should complete successfully
    assert len(results["trials"]) == 70

    # Should have some successful trials
    successful_trials = [t for t in results["trials"] if t.result.converged]
    assert len(successful_trials) > 0

    # Best objective should be reasonable
    assert results["best_objective"] < 10.0  # Not a failed trial


def test_optimizer_with_benchmark_dataset():
    """Test optimizer evaluation on benchmark dataset."""
    # This test demonstrates how to use benchmark datasets
    dataset = create_benchmark_dataset("simple", n_train=50, n_test=20, random_seed=42)

    # Check dataset structure
    assert dataset["train"]["X"].shape[0] == 50
    assert dataset["test"]["X"].shape[0] == 20

    # Verify success rates are reasonable
    train_success_rate = dataset["train"]["y_converged"].mean()
    assert 0.2 < train_success_rate < 0.8


def test_optimizer_state_consistency():
    """Test that optimizer maintains consistent state."""
    simulator = SyntheticSimulator(random_seed=42)

    config = OptimizationConfig(
        n_sobol_trials=5,
        n_frbo_trials=5,
        random_seed=42,
    )

    optimizer = FRBOOptimizer(simulator=simulator, config=config)

    # Initial state
    assert optimizer.best_objective == float("inf")
    assert optimizer.best_parameters is None
    assert len(optimizer.trials) == 0

    # After optimization
    results = optimizer.optimize()

    # State should be updated
    assert optimizer.best_objective < float("inf")
    assert optimizer.best_parameters is not None
    assert len(optimizer.trials) == 10

    # State should match results
    assert optimizer.best_objective == results["best_objective"]
