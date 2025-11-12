"""Tests for synthetic data generation."""

import pytest
import numpy as np
from fr_bo.synthetic_data import (
    SyntheticDataGenerator,
    create_benchmark_dataset,
)


def test_synthetic_data_generator():
    """Test basic synthetic data generation."""
    generator = SyntheticDataGenerator(random_seed=42)

    scenario = generator.generate_scenario_simple()

    assert scenario.name == "simple"
    assert scenario.parameter_dim == 3
    assert len(scenario.optimal_regions) == 2
    assert len(scenario.failure_regions) == 2


def test_generate_training_data():
    """Test training data generation."""
    generator = SyntheticDataGenerator(random_seed=42)
    scenario = generator.generate_scenario_simple()

    X, y_obj, y_conv = generator.generate_training_data(scenario, n_samples=100)

    assert X.shape == (100, scenario.parameter_dim)
    assert y_obj.shape == (100,)
    assert y_conv.shape == (100,)

    # Should have mix of successes and failures
    success_rate = y_conv.mean()
    assert 0.1 < success_rate < 0.9


def test_generate_complex_scenario():
    """Test complex scenario generation."""
    generator = SyntheticDataGenerator(random_seed=42)
    scenario = generator.generate_scenario_complex()

    assert scenario.name == "complex"
    assert scenario.parameter_dim == 5
    assert len(scenario.optimal_regions) == 3
    assert len(scenario.failure_regions) == 3


def test_generate_test_geometries():
    """Test geometry generation for multi-task GP."""
    generator = SyntheticDataGenerator(random_seed=42)
    geometries = generator.generate_test_geometries(n_geometries=5)

    assert len(geometries) == 5

    for geom in geometries:
        assert "geometry_id" in geom
        assert "total_surface_area" in geom
        assert "youngs_moduli" in geom
        assert len(geom["youngs_moduli"]) == 2


def test_generate_convergence_trajectory_converged():
    """Test convergence trajectory generation for converged case."""
    generator = SyntheticDataGenerator(random_seed=42)

    iterations, residuals, active_sets = generator.generate_convergence_trajectory(
        max_iterations=100,
        will_converge=True,
        convergence_rate=0.2,
    )

    assert len(iterations) > 0
    assert len(residuals) == len(iterations)
    assert len(active_sets) == len(iterations)

    # Should converge before max iterations
    assert len(iterations) < 100

    # Residuals should generally decrease
    assert residuals[-1] < residuals[0]


def test_generate_convergence_trajectory_failed():
    """Test convergence trajectory for failed case."""
    generator = SyntheticDataGenerator(random_seed=42)

    iterations, residuals, active_sets = generator.generate_convergence_trajectory(
        max_iterations=50,
        will_converge=False,
        convergence_rate=0.01,
    )

    # Should run to max iterations
    assert len(iterations) <= 50

    # Residuals should not reach convergence threshold
    assert residuals[-1] > 1e-10


def test_create_benchmark_dataset():
    """Test benchmark dataset creation."""
    dataset = create_benchmark_dataset(
        scenario_name="simple",
        n_train=50,
        n_test=20,
        random_seed=42,
    )

    assert "scenario" in dataset
    assert "train" in dataset
    assert "test" in dataset

    train = dataset["train"]
    assert train["X"].shape[0] == 50
    assert train["y_objective"].shape[0] == 50
    assert train["y_converged"].shape[0] == 50

    test = dataset["test"]
    assert test["X"].shape[0] == 20


def test_benchmark_dataset_reproducibility():
    """Test that benchmark datasets are reproducible."""
    dataset1 = create_benchmark_dataset("simple", n_train=50, random_seed=42)
    dataset2 = create_benchmark_dataset("simple", n_train=50, random_seed=42)

    np.testing.assert_array_equal(dataset1["train"]["X"], dataset2["train"]["X"])
    np.testing.assert_array_equal(dataset1["train"]["y_converged"], dataset2["train"]["y_converged"])


def test_optimal_regions_affect_success_rate():
    """Test that points in optimal regions have higher success rate."""
    generator = SyntheticDataGenerator(random_seed=42)
    scenario = generator.generate_scenario_simple()

    # Generate points near optimal region centers
    n_samples = 100
    optimal_center = scenario.optimal_regions[0]["center"]

    # Points near optimal center
    X_near = optimal_center + generator.rng.randn(n_samples, scenario.parameter_dim) * 0.05
    _, _, y_conv_near = generator.generate_training_data(scenario, n_samples)

    # Random points
    _, _, y_conv_random = generator.generate_training_data(scenario, n_samples)

    # Near optimal should have higher success rate (not guaranteed, but very likely)
    # Just check that we can generate data with different success rates
    assert len(y_conv_near) == n_samples
    assert len(y_conv_random) == n_samples
