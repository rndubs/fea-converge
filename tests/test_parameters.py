"""Tests for parameter space definition and encoding."""

import pytest
import numpy as np
from fr_bo.parameters import (
    create_search_space,
    encode_parameters,
    decode_parameters,
    get_parameter_dimension,
    ParameterBounds,
)


def test_create_search_space():
    """Test search space creation."""
    search_space = create_search_space()

    # Check that all expected parameters are present
    param_names = set(search_space.parameters.keys())
    expected_names = {
        "penalty_stiffness",
        "gap_tolerance",
        "projection_tolerance",
        "abs_tolerance",
        "rel_tolerance",
        "search_expansion",
        "max_iterations",
        "line_search_iters",
        "trust_region_scaling",
        "enforcement_method",
        "solver_type",
        "linear_solver",
    }

    assert param_names == expected_names


def test_parameter_dimension():
    """Test parameter dimension calculation."""
    dim = get_parameter_dimension()

    # 9 continuous + 3 enforcement + 4 solver + 2 linear_solver
    expected_dim = 9 + 3 + 4 + 2
    assert dim == expected_dim


def test_encode_decode_parameters():
    """Test parameter encoding and decoding."""
    params = {
        "penalty_stiffness": 1e6,
        "gap_tolerance": 1e-7,
        "projection_tolerance": 1e-8,
        "abs_tolerance": 1e-10,
        "rel_tolerance": 1e-9,
        "search_expansion": 1.3,
        "max_iterations": 500,
        "line_search_iters": 10,
        "trust_region_scaling": 0.2,
        "enforcement_method": "penalty",
        "solver_type": "NewtonLineSearch",
        "linear_solver": "GMRES",
    }

    # Encode
    encoded = encode_parameters(params)

    # Check shape
    assert encoded.shape == (get_parameter_dimension(),)

    # Decode
    decoded = decode_parameters(encoded)

    # Check that categorical parameters are preserved
    assert decoded["enforcement_method"] in ParameterBounds.ENFORCEMENT_METHODS
    assert decoded["solver_type"] in ParameterBounds.SOLVER_TYPES
    assert decoded["linear_solver"] in ParameterBounds.LINEAR_SOLVERS


def test_parameter_bounds():
    """Test parameter bounds."""
    # Check that bounds are reasonable
    assert ParameterBounds.PENALTY_STIFFNESS[0] > 0
    assert ParameterBounds.PENALTY_STIFFNESS[1] > ParameterBounds.PENALTY_STIFFNESS[0]

    assert 0 < ParameterBounds.GAP_TOLERANCE[0] < ParameterBounds.GAP_TOLERANCE[1]
    assert ParameterBounds.MAX_ITERATIONS[0] < ParameterBounds.MAX_ITERATIONS[1]


def test_encode_categorical():
    """Test one-hot encoding of categorical variables."""
    params = {
        "penalty_stiffness": 1e6,
        "gap_tolerance": 1e-7,
        "projection_tolerance": 1e-8,
        "abs_tolerance": 1e-10,
        "rel_tolerance": 1e-9,
        "search_expansion": 1.3,
        "max_iterations": 500,
        "line_search_iters": 10,
        "trust_region_scaling": 0.2,
        "enforcement_method": "penalty",
        "solver_type": "Newton",
        "linear_solver": "SuperLU",
    }

    encoded = encode_parameters(params)

    # Check that exactly one of each categorical group is 1.0
    # Enforcement method: indices 9:12
    enforcement_part = encoded[9:12]
    assert np.sum(enforcement_part) == 1.0
    assert np.sum(enforcement_part == 1.0) == 1

    # Solver type: indices 12:16
    solver_part = encoded[12:16]
    assert np.sum(solver_part) == 1.0

    # Linear solver: indices 16:18
    linear_part = encoded[16:18]
    assert np.sum(linear_part) == 1.0
