"""Tests for black box FEA solver."""

import pytest
import numpy as np
from shebo.utils.black_box_solver import BlackBoxFEASolver, create_test_objective


class TestBlackBoxFEASolver:
    """Test black box solver functionality."""

    def test_solver_initialization(self):
        """Test solver can be initialized."""
        solver = BlackBoxFEASolver(random_seed=42)
        assert solver is not None
        assert solver.noise_level == 0.1

    def test_solver_output_structure(self):
        """Test that solver output has correct structure."""
        solver = BlackBoxFEASolver(random_seed=42)
        params = np.array([1e8, 1e-6, 0.5, 0.5])

        output = solver.solve(params)

        # Check required keys
        assert 'convergence_status' in output
        assert 'residual_history' in output
        assert 'iterations' in output
        assert 'solve_time' in output
        assert 'penetration_max' in output
        assert 'jacobian_min' in output
        assert 'contact_pairs' in output
        assert 'all_values' in output

    def test_solver_convergence_variability(self):
        """Test that solver produces both convergence and failures."""
        solver = BlackBoxFEASolver(random_seed=42)

        results = []
        for _ in range(20):
            # Use random parameters
            params = np.random.uniform([1e6, 1e-8, 0, 0], [1e10, 1e-4, 1, 1])
            output = solver.solve(params)
            results.append(output['convergence_status'])

        # Should have mix of convergence and failures
        n_converged = sum(results)
        assert 0 < n_converged < 20, "Solver should produce mix of outcomes"

    def test_residual_history_length(self):
        """Test residual history is non-empty."""
        solver = BlackBoxFEASolver(random_seed=42)
        params = np.array([1e8, 1e-6, 0.5, 0.5])

        output = solver.solve(params)

        assert len(output['residual_history']) > 0
        assert all(r >= 0 for r in output['residual_history'])

    def test_convergence_probability_computation(self):
        """Test convergence probability is in valid range."""
        solver = BlackBoxFEASolver(random_seed=42)

        # Test multiple parameter sets
        for _ in range(10):
            params = np.random.uniform([1e6, 1e-8, 0, 0], [1e10, 1e-4, 1, 1])
            prob = solver._compute_convergence_probability(params)

            assert 0 <= prob <= 1, f"Probability {prob} out of range"

    def test_create_test_objective(self):
        """Test test objective function creation."""
        objective = create_test_objective(n_params=4, random_seed=42)

        params = np.array([1e8, 1e-6, 0.5, 0.5])
        result = objective(params)

        assert 'output' in result
        assert 'performance' in result
        assert isinstance(result['performance'], (int, float))

    def test_deterministic_with_seed(self):
        """Test that solver is deterministic with same seed."""
        params = np.array([1e8, 1e-6, 0.5, 0.5])

        solver1 = BlackBoxFEASolver(random_seed=42)
        output1 = solver1.solve(params)

        solver2 = BlackBoxFEASolver(random_seed=42)
        output2 = solver2.solve(params)

        # Should produce same convergence status
        assert output1['convergence_status'] == output2['convergence_status']

    def test_different_failure_modes(self):
        """Test that different failure modes can occur."""
        solver = BlackBoxFEASolver(random_seed=42)

        # Collect many samples to see different failure modes
        failure_modes = set()

        for _ in range(50):
            params = np.random.uniform([1e6, 1e-8, 0, 0], [1e10, 1e-4, 1, 1])
            output = solver.solve(params)

            if not output['convergence_status']:
                # Try to infer failure mode from output
                if output['penetration_max'] > 1e-3:
                    failure_modes.add('excessive_penetration')
                if output['jacobian_min'] <= 0:
                    failure_modes.add('mesh_distortion')
                if output['contact_pairs'] == 0:
                    failure_modes.add('contact_detection_failure')

        # Should observe at least some failures
        assert len(failure_modes) > 0
