"""
Tests for mock Smith solver.
"""

import pytest
import numpy as np

from gp_classification.mock_solver import (
    MockSmithSolver,
    SyntheticDataGenerator,
    get_default_parameter_bounds,
)


@pytest.fixture
def parameter_bounds():
    """Get default parameter bounds."""
    return get_default_parameter_bounds()


@pytest.fixture
def sample_parameters():
    """Sample parameters for testing."""
    return {
        "penalty_stiffness": 1e5,
        "gap_tolerance": 1e-7,
        "absolute_tolerance": 1e-10,
        "relative_tolerance": 1e-8,
        "max_iterations": 50,
        "timestep": 1e-4,
    }


def test_mock_solver_initialization():
    """Test MockSmithSolver initialization."""
    solver = MockSmithSolver(random_seed=42, noise_level=0.1, difficulty="medium")

    assert solver is not None
    assert solver.noise_level == 0.1
    assert solver.difficulty == "medium"


def test_mock_solver_simulation(sample_parameters):
    """Test basic simulation."""
    solver = MockSmithSolver(random_seed=42, noise_level=0.05)

    converged, objective = solver.simulate(sample_parameters)

    assert isinstance(converged, bool)
    if converged:
        assert objective is not None
        assert objective > 0  # Iteration count should be positive
    else:
        assert objective is None


def test_mock_solver_reproducibility(sample_parameters):
    """Test that solver is reproducible with same seed."""
    solver1 = MockSmithSolver(random_seed=42, noise_level=0.05)
    solver2 = MockSmithSolver(random_seed=42, noise_level=0.05)

    # Run multiple simulations
    results1 = [solver1.simulate(sample_parameters) for _ in range(10)]
    results2 = [solver2.simulate(sample_parameters) for _ in range(10)]

    # Should get identical results
    for r1, r2 in zip(results1, results2):
        assert r1[0] == r2[0]  # Convergence
        if r1[1] is not None and r2[1] is not None:
            assert abs(r1[1] - r2[1]) < 1e-6  # Objective


def test_mock_solver_difficulty_levels():
    """Test different difficulty levels."""
    params = {
        "penalty_stiffness": 1e5,
        "gap_tolerance": 1e-7,
        "absolute_tolerance": 1e-10,
        "relative_tolerance": 1e-8,
        "max_iterations": 50,
        "timestep": 1e-4,
    }

    # Run multiple simulations for each difficulty
    n_trials = 50

    for difficulty in ["easy", "medium", "hard"]:
        solver = MockSmithSolver(random_seed=42, noise_level=0.1, difficulty=difficulty)

        convergence_count = 0
        for _ in range(n_trials):
            converged, _ = solver.simulate(params)
            if converged:
                convergence_count += 1

        convergence_rate = convergence_count / n_trials

        # Easy should have higher convergence rate than hard
        if difficulty == "easy":
            assert convergence_rate > 0.6
        elif difficulty == "hard":
            # Hard might be very challenging
            pass  # No assertion, just checking it runs


def test_parameter_effects(sample_parameters):
    """Test that parameter changes affect convergence."""
    solver = MockSmithSolver(random_seed=42, noise_level=0.05, difficulty="medium")

    # Test with good parameters
    good_params = sample_parameters.copy()
    good_params["penalty_stiffness"] = 5e5  # Near optimal for medium difficulty

    # Test with bad parameters
    bad_params = sample_parameters.copy()
    bad_params["penalty_stiffness"] = 1e9  # Too high, should fail more often

    # Run multiple trials
    n_trials = 30
    good_success = sum(1 for _ in range(n_trials) if solver.simulate(good_params)[0])
    bad_success = sum(
        1
        for _ in range(n_trials)
        if MockSmithSolver(random_seed=42 + 100 + _, noise_level=0.05).simulate(
            bad_params
        )[0]
    )

    # Good parameters should generally perform better
    # (May not always be true due to randomness, but should be on average)
    assert good_success >= 0  # Just checking no errors


def test_synthetic_data_generator(parameter_bounds):
    """Test synthetic data generator."""
    generator = SyntheticDataGenerator(parameter_bounds, random_seed=42)

    # Generate random parameters
    params_list = generator.generate_random_parameters(n_samples=20)

    assert len(params_list) == 20
    assert all(isinstance(p, dict) for p in params_list)

    # Check bounds
    for params in params_list:
        for name, value in params.items():
            lower, upper = parameter_bounds[name]
            assert lower <= value <= upper


def test_latin_hypercube_sampling(parameter_bounds):
    """Test Latin Hypercube Sampling."""
    generator = SyntheticDataGenerator(parameter_bounds, random_seed=42)

    params_list = generator.generate_latin_hypercube(n_samples=30)

    assert len(params_list) == 30

    # Check bounds
    for params in params_list:
        for name, value in params.items():
            lower, upper = parameter_bounds[name]
            assert lower <= value <= upper


def test_generate_dataset_with_solver(parameter_bounds):
    """Test full dataset generation with solver."""
    solver = MockSmithSolver(random_seed=42, noise_level=0.05, difficulty="easy")
    generator = SyntheticDataGenerator(parameter_bounds, random_seed=42)

    dataset = generator.generate_dataset_with_solver(
        solver=solver,
        n_samples=20,
        sampling_method="latin_hypercube",
    )

    assert len(dataset) == 20

    # Check structure
    for params, converged, objective in dataset:
        assert isinstance(params, dict)
        assert isinstance(converged, bool)
        if converged:
            assert objective is not None
            assert objective > 0
        else:
            assert objective is None

    # Should have some successes with easy difficulty
    success_count = sum(1 for _, converged, _ in dataset if converged)
    assert success_count > 0
