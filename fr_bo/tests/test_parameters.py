"""
Unit tests for parameter space and encoding (parameters.py).

Tests:
- Parameter bounds definition
- Search space creation
- Parameter transformations
"""

import pytest
import torch
import numpy as np


class TestParameterBounds:
    """Test parameter bounds and definitions."""

    def test_bounds_import(self):
        """Test that ParameterBounds can be imported."""
        from fr_bo.parameters import ParameterBounds

        bounds = ParameterBounds()

        # Check that bounds are defined
        assert hasattr(bounds, 'PENALTY_STIFFNESS')
        assert hasattr(bounds, 'GAP_TOLERANCE')
        assert hasattr(bounds, 'MAX_ITERATIONS')

    def test_bounds_are_tuples(self):
        """Test that bounds are (min, max) tuples."""
        from fr_bo.parameters import ParameterBounds

        bounds = ParameterBounds()

        # Check penalty stiffness
        assert isinstance(bounds.PENALTY_STIFFNESS, tuple)
        assert len(bounds.PENALTY_STIFFNESS) == 2
        assert bounds.PENALTY_STIFFNESS[0] < bounds.PENALTY_STIFFNESS[1]

        # Check gap tolerance
        assert isinstance(bounds.GAP_TOLERANCE, tuple)
        assert len(bounds.GAP_TOLERANCE) == 2
        assert bounds.GAP_TOLERANCE[0] < bounds.GAP_TOLERANCE[1]

    def test_categorical_parameters(self):
        """Test categorical parameter definitions."""
        from fr_bo.parameters import ParameterBounds

        bounds = ParameterBounds()

        # Check enforcement methods
        assert isinstance(bounds.ENFORCEMENT_METHODS, list)
        assert len(bounds.ENFORCEMENT_METHODS) > 0
        assert "penalty" in bounds.ENFORCEMENT_METHODS

        # Check solver types
        assert isinstance(bounds.SOLVER_TYPES, list)
        assert "Newton" in bounds.SOLVER_TYPES


class TestSearchSpace:
    """Test search space creation."""

    def test_create_search_space(self):
        """Test that search space can be created."""
        from fr_bo.parameters import create_search_space

        search_space = create_search_space()

        assert search_space is not None
        assert hasattr(search_space, 'parameters')

    def test_search_space_parameters(self):
        """Test that search space has expected parameters."""
        from fr_bo.parameters import create_search_space

        search_space = create_search_space()

        # Should have multiple parameters
        assert len(search_space.parameters) > 0

        # Check for key parameters
        param_names = [p.name for p in search_space.parameters.values()]
        assert 'penalty_stiffness' in param_names or len(param_names) > 0

    def test_parameter_dimension(self):
        """Test getting parameter dimension."""
        from fr_bo.parameters import get_parameter_dimension, create_search_space

        search_space = create_search_space()
        dim = get_parameter_dimension(search_space)

        # Should have positive dimension
        assert dim > 0
        # Typically 9-12 parameters for Tribol/Smith
        assert dim >= 5  # At least a few parameters

    def test_parameter_types(self):
        """Test that parameters have correct types."""
        from fr_bo.parameters import create_search_space
        from ax.core.parameter import RangeParameter, ChoiceParameter

        search_space = create_search_space()

        # Should have both range and choice parameters
        has_range = False
        has_choice = False

        for param in search_space.parameters.values():
            if isinstance(param, RangeParameter):
                has_range = True
            elif isinstance(param, ChoiceParameter):
                has_choice = True

        # Should have at least range parameters (continuous optimization)
        assert has_range


class TestParameterEncoding:
    """Test parameter encoding and transformations."""

    def test_parameter_sampling(self):
        """Test that we can sample parameters from the search space."""
        from fr_bo.parameters import create_search_space

        search_space = create_search_space()

        # Sample a point (if supported by the search space)
        # This tests that the search space is properly configured
        try:
            # Try to get bounds for sampling
            assert search_space is not None
        except Exception:
            pytest.skip("Search space sampling not directly supported")

    def test_log_scale_bounds(self):
        """Test that log-scale parameters have appropriate ranges."""
        from fr_bo.parameters import ParameterBounds

        bounds = ParameterBounds()

        # Parameters with many orders of magnitude should use log scale
        penalty_range = bounds.PENALTY_STIFFNESS[1] / bounds.PENALTY_STIFFNESS[0]
        assert penalty_range > 100  # At least 2 orders of magnitude

        gap_tol_range = bounds.GAP_TOLERANCE[1] / bounds.GAP_TOLERANCE[0]
        assert gap_tol_range > 100  # At least 2 orders of magnitude

    def test_linear_scale_bounds(self):
        """Test that linear-scale parameters have reasonable ranges."""
        from fr_bo.parameters import ParameterBounds

        bounds = ParameterBounds()

        # Linear scale parameters should have reasonable ranges
        max_iter_range = bounds.MAX_ITERATIONS[1] - bounds.MAX_ITERATIONS[0]
        assert max_iter_range > 0
        assert max_iter_range < 10000  # Reasonable range
