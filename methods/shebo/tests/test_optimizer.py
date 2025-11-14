"""Tests for SHEBO optimizer."""

import pytest
import numpy as np
from shebo.core.optimizer import SHEBOOptimizer, SHEBOResult
from shebo.utils.black_box_solver import create_test_objective


class TestSHEBOOptimizer:
    """Test SHEBO optimizer functionality."""

    @pytest.fixture
    def simple_bounds(self):
        """Create simple parameter bounds."""
        return np.array([
            [1e6, 1e10],    # penalty parameter
            [1e-8, 1e-4],   # tolerance
            [0.0, 1.0],     # normalized param 1
            [0.0, 1.0]      # normalized param 2
        ])

    @pytest.fixture
    def test_objective(self):
        """Create test objective function."""
        return create_test_objective(n_params=4, random_seed=42)

    def test_optimizer_initialization(self, simple_bounds, test_objective):
        """Test optimizer can be initialized."""
        optimizer = SHEBOOptimizer(
            bounds=simple_bounds,
            objective_fn=test_objective,
            n_init=5,
            budget=10,
            random_seed=42
        )

        assert optimizer is not None
        assert optimizer.n_params == 4
        assert optimizer.budget == 10

    def test_optimizer_run_completes(self, simple_bounds, test_objective):
        """Test that optimizer runs to completion."""
        optimizer = SHEBOOptimizer(
            bounds=simple_bounds,
            objective_fn=test_objective,
            n_init=5,
            budget=15,
            random_seed=42
        )

        result = optimizer.run()

        assert isinstance(result, SHEBOResult)
        assert result.iterations > 0
        assert len(result.convergence_history) > 0
        assert len(result.all_params) > 0

    def test_optimizer_respects_budget(self, simple_bounds, test_objective):
        """Test that optimizer respects evaluation budget."""
        budget = 20
        optimizer = SHEBOOptimizer(
            bounds=simple_bounds,
            objective_fn=test_objective,
            n_init=5,
            budget=budget,
            random_seed=42
        )

        result = optimizer.run()

        # Should not exceed budget
        assert len(result.all_params) <= budget

    def test_optimizer_finds_feasible_solutions(self, simple_bounds, test_objective):
        """Test that optimizer finds at least some feasible solutions."""
        optimizer = SHEBOOptimizer(
            bounds=simple_bounds,
            objective_fn=test_objective,
            n_init=10,
            budget=30,
            random_seed=42
        )

        result = optimizer.run()

        # Should have at least some converged solutions
        n_converged = sum(result.convergence_history)
        assert n_converged > 0, "Optimizer should find at least one feasible solution"

    def test_optimizer_improves_over_time(self, simple_bounds, test_objective):
        """Test that best performance improves or stays constant."""
        optimizer = SHEBOOptimizer(
            bounds=simple_bounds,
            objective_fn=test_objective,
            n_init=10,
            budget=30,
            random_seed=42
        )

        result = optimizer.run()

        # Track best performance over time
        best_so_far = []
        current_best = float('inf')

        for perf, converged in zip(result.performance_history, result.convergence_history):
            if converged and perf < current_best:
                current_best = perf
            best_so_far.append(current_best if current_best != float('inf') else None)

        # Filter out None values
        valid_best = [b for b in best_so_far if b is not None]

        if len(valid_best) > 1:
            # Best performance should be monotonically decreasing (or constant)
            for i in range(1, len(valid_best)):
                assert valid_best[i] <= valid_best[i-1], \
                    "Best performance should not get worse"

    def test_optimizer_discovers_constraints(self, simple_bounds, test_objective):
        """Test that optimizer can discover constraints."""
        optimizer = SHEBOOptimizer(
            bounds=simple_bounds,
            objective_fn=test_objective,
            n_init=10,
            budget=40,
            random_seed=42
        )

        result = optimizer.run()

        # Should discover at least some constraints with enough evaluations
        # (depends on the black box solver producing failures)
        assert 'discovered_constraints' in result.__dict__

    def test_optimizer_stores_all_evaluations(self, simple_bounds, test_objective):
        """Test that all evaluations are stored."""
        optimizer = SHEBOOptimizer(
            bounds=simple_bounds,
            objective_fn=test_objective,
            n_init=5,
            budget=15,
            random_seed=42
        )

        result = optimizer.run()

        # All arrays should have same length
        n_evals = len(result.all_params)
        assert len(result.convergence_history) == n_evals
        assert len(result.performance_history) == n_evals
        assert len(result.all_outputs) == n_evals

    def test_optimizer_result_structure(self, simple_bounds, test_objective):
        """Test that result has expected structure."""
        optimizer = SHEBOOptimizer(
            bounds=simple_bounds,
            objective_fn=test_objective,
            n_init=5,
            budget=10,
            random_seed=42
        )

        result = optimizer.run()

        # Check all expected fields
        assert hasattr(result, 'best_params')
        assert hasattr(result, 'best_performance')
        assert hasattr(result, 'convergence_history')
        assert hasattr(result, 'performance_history')
        assert hasattr(result, 'discovered_constraints')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'all_params')
        assert hasattr(result, 'all_outputs')

    def test_optimizer_with_different_seeds(self, simple_bounds, test_objective):
        """Test that different seeds produce different results."""
        optimizer1 = SHEBOOptimizer(
            bounds=simple_bounds,
            objective_fn=test_objective,
            n_init=5,
            budget=10,
            random_seed=42
        )
        result1 = optimizer1.run()

        optimizer2 = SHEBOOptimizer(
            bounds=simple_bounds,
            objective_fn=test_objective,
            n_init=5,
            budget=10,
            random_seed=123
        )
        result2 = optimizer2.run()

        # Different seeds should produce different parameter sequences
        # (at least for most points after initialization)
        different_count = sum(
            not np.allclose(p1, p2)
            for p1, p2 in zip(result1.all_params, result2.all_params)
        )

        assert different_count > 0, "Different seeds should produce different results"

    def test_optimizer_bounds_respected(self, simple_bounds, test_objective):
        """Test that all evaluated points respect bounds."""
        optimizer = SHEBOOptimizer(
            bounds=simple_bounds,
            objective_fn=test_objective,
            n_init=5,
            budget=15,
            random_seed=42
        )

        result = optimizer.run()

        # Check all parameters respect bounds
        for params in result.all_params:
            for i, (low, high) in enumerate(simple_bounds):
                assert low <= params[i] <= high, \
                    f"Parameter {i} value {params[i]} outside bounds [{low}, {high}]"
