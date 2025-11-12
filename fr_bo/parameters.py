"""
Parameter space definition and encoding for Tribol and Smith solver parameters.

This module defines the search space for contact convergence optimization,
including parameter transformations and encoding schemes.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
from ax.core.parameter import ParameterType, RangeParameter, ChoiceParameter
from ax.core.search_space import SearchSpace


@dataclass
class ParameterBounds:
    """Parameter bounds and metadata for FEA optimization."""

    # Tribol contact parameters
    PENALTY_STIFFNESS = (1e3, 1e8)  # Log-scale
    GAP_TOLERANCE = (1e-9, 1e-6)  # Log-scale
    PROJECTION_TOLERANCE = (1e-9, 1e-6)  # Log-scale
    SEARCH_EXPANSION = (1.1, 1.5)  # Linear

    # Smith solver parameters
    MAX_ITERATIONS = (10, 1000)  # Linear
    ABS_TOLERANCE = (1e-14, 1e-8)  # Log-scale
    REL_TOLERANCE = (1e-14, 1e-8)  # Log-scale
    LINE_SEARCH_ITERS = (0, 20)  # Linear
    TRUST_REGION_SCALING = (0.05, 0.5)  # Linear

    # Categorical parameters
    ENFORCEMENT_METHODS = ["penalty", "mortar", "augmented_lagrange"]
    SOLVER_TYPES = ["Newton", "NewtonLineSearch", "TrustRegion", "L-BFGS"]
    LINEAR_SOLVERS = ["SuperLU", "GMRES"]


def create_search_space() -> SearchSpace:
    """
    Create the Ax SearchSpace for FR-BO optimization.

    Returns:
        SearchSpace: Ax search space with all parameters
    """
    parameters = []

    # Continuous parameters with log-scale transformation
    parameters.append(
        RangeParameter(
            name="penalty_stiffness",
            parameter_type=ParameterType.FLOAT,
            lower=ParameterBounds.PENALTY_STIFFNESS[0],
            upper=ParameterBounds.PENALTY_STIFFNESS[1],
            log_scale=True,
        )
    )

    parameters.append(
        RangeParameter(
            name="gap_tolerance",
            parameter_type=ParameterType.FLOAT,
            lower=ParameterBounds.GAP_TOLERANCE[0],
            upper=ParameterBounds.GAP_TOLERANCE[1],
            log_scale=True,
        )
    )

    parameters.append(
        RangeParameter(
            name="projection_tolerance",
            parameter_type=ParameterType.FLOAT,
            lower=ParameterBounds.PROJECTION_TOLERANCE[0],
            upper=ParameterBounds.PROJECTION_TOLERANCE[1],
            log_scale=True,
        )
    )

    parameters.append(
        RangeParameter(
            name="abs_tolerance",
            parameter_type=ParameterType.FLOAT,
            lower=ParameterBounds.ABS_TOLERANCE[0],
            upper=ParameterBounds.ABS_TOLERANCE[1],
            log_scale=True,
        )
    )

    parameters.append(
        RangeParameter(
            name="rel_tolerance",
            parameter_type=ParameterType.FLOAT,
            lower=ParameterBounds.REL_TOLERANCE[0],
            upper=ParameterBounds.REL_TOLERANCE[1],
            log_scale=True,
        )
    )

    # Continuous parameters with linear scale
    parameters.append(
        RangeParameter(
            name="search_expansion",
            parameter_type=ParameterType.FLOAT,
            lower=ParameterBounds.SEARCH_EXPANSION[0],
            upper=ParameterBounds.SEARCH_EXPANSION[1],
        )
    )

    parameters.append(
        RangeParameter(
            name="max_iterations",
            parameter_type=ParameterType.INT,
            lower=ParameterBounds.MAX_ITERATIONS[0],
            upper=ParameterBounds.MAX_ITERATIONS[1],
        )
    )

    parameters.append(
        RangeParameter(
            name="line_search_iters",
            parameter_type=ParameterType.INT,
            lower=ParameterBounds.LINE_SEARCH_ITERS[0],
            upper=ParameterBounds.LINE_SEARCH_ITERS[1],
        )
    )

    parameters.append(
        RangeParameter(
            name="trust_region_scaling",
            parameter_type=ParameterType.FLOAT,
            lower=ParameterBounds.TRUST_REGION_SCALING[0],
            upper=ParameterBounds.TRUST_REGION_SCALING[1],
        )
    )

    # Categorical parameters
    parameters.append(
        ChoiceParameter(
            name="enforcement_method",
            parameter_type=ParameterType.STRING,
            values=ParameterBounds.ENFORCEMENT_METHODS,
        )
    )

    parameters.append(
        ChoiceParameter(
            name="solver_type",
            parameter_type=ParameterType.STRING,
            values=ParameterBounds.SOLVER_TYPES,
        )
    )

    parameters.append(
        ChoiceParameter(
            name="linear_solver",
            parameter_type=ParameterType.STRING,
            values=ParameterBounds.LINEAR_SOLVERS,
        )
    )

    return SearchSpace(parameters=parameters)


def encode_parameters(params: Dict[str, Any]) -> np.ndarray:
    """
    Encode parameters for GP training (one-hot encoding for categorical).

    Args:
        params: Dictionary of parameter values

    Returns:
        Encoded parameter vector as numpy array
    """
    encoded = []

    # Continuous parameters (already normalized by Ax)
    continuous_keys = [
        "penalty_stiffness", "gap_tolerance", "projection_tolerance",
        "abs_tolerance", "rel_tolerance", "search_expansion",
        "max_iterations", "line_search_iters", "trust_region_scaling"
    ]

    for key in continuous_keys:
        encoded.append(params[key])

    # One-hot encode enforcement method
    for method in ParameterBounds.ENFORCEMENT_METHODS:
        encoded.append(1.0 if params["enforcement_method"] == method else 0.0)

    # One-hot encode solver type
    for solver in ParameterBounds.SOLVER_TYPES:
        encoded.append(1.0 if params["solver_type"] == solver else 0.0)

    # One-hot encode linear solver
    for linear_solver in ParameterBounds.LINEAR_SOLVERS:
        encoded.append(1.0 if params["linear_solver"] == linear_solver else 0.0)

    return np.array(encoded)


def decode_parameters(encoded: np.ndarray) -> Dict[str, Any]:
    """
    Decode parameter vector back to dictionary format.

    Args:
        encoded: Encoded parameter vector

    Returns:
        Dictionary of parameter values
    """
    params = {}
    idx = 0

    # Continuous parameters
    continuous_keys = [
        "penalty_stiffness", "gap_tolerance", "projection_tolerance",
        "abs_tolerance", "rel_tolerance", "search_expansion",
        "max_iterations", "line_search_iters", "trust_region_scaling"
    ]

    for key in continuous_keys:
        params[key] = encoded[idx]
        idx += 1

    # Decode enforcement method (one-hot)
    enforcement_idx = np.argmax(encoded[idx:idx+len(ParameterBounds.ENFORCEMENT_METHODS)])
    params["enforcement_method"] = ParameterBounds.ENFORCEMENT_METHODS[enforcement_idx]
    idx += len(ParameterBounds.ENFORCEMENT_METHODS)

    # Decode solver type (one-hot)
    solver_idx = np.argmax(encoded[idx:idx+len(ParameterBounds.SOLVER_TYPES)])
    params["solver_type"] = ParameterBounds.SOLVER_TYPES[solver_idx]
    idx += len(ParameterBounds.SOLVER_TYPES)

    # Decode linear solver (one-hot)
    linear_solver_idx = np.argmax(encoded[idx:idx+len(ParameterBounds.LINEAR_SOLVERS)])
    params["linear_solver"] = ParameterBounds.LINEAR_SOLVERS[linear_solver_idx]

    return params


def get_parameter_dimension() -> int:
    """Get the total dimension of the encoded parameter space."""
    return (
        9 +  # 9 continuous parameters
        len(ParameterBounds.ENFORCEMENT_METHODS) +
        len(ParameterBounds.SOLVER_TYPES) +
        len(ParameterBounds.LINEAR_SOLVERS)
    )
