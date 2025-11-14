"""
Configuration constants for CONFIG optimizer.

This module defines all magic numbers and configuration constants
used throughout the CONFIG implementation.
"""

# Beta schedule constants
DEFAULT_DELTA = 0.1  # Confidence level (10% failure probability)
MIN_BETA = 0.1  # Minimum beta value for numerical stability

# Acquisition optimization constants
DEFAULT_N_RESTARTS = 20  # Number of random restarts for acquisition optimization
DEFAULT_N_CANDIDATES = 1000  # Number of candidates for discrete optimization
MAX_BETA_RECURSION = 3  # Maximum recursion depth for beta adjustment
BETA_ADJUSTMENT_FACTOR = 1.5  # Multiplier when expanding optimistic feasible set

# Phase detection constants
UNCERTAINTY_THRESHOLD = 0.3  # Max constraint uncertainty for switching from boundary exploration
BOUNDARY_PROXIMITY_SCALE = 5.0  # Scale factor for boundary proximity weighting

# Initialization constants
DEFAULT_N_INIT = 20  # Default number of initial LHS samples
FEASIBILITY_DISCOVERY_BUFFER = 20  # Extra iterations for feasibility discovery

# Termination constants
MIN_ITERATIONS_FOR_TERMINATION = 50  # Minimum iterations before early termination check
STABILITY_WINDOW = 15  # Number of iterations to check for stability
STABILITY_TOLERANCE = 1e-3  # Tolerance for considering solutions equivalent

# Penalty values for failed evaluations
FAILED_OBJECTIVE_PENALTY = 1e10  # Large penalty for objective when evaluation fails
FAILED_CONSTRAINT_PENALTY = 1e3  # Large positive (violated) for constraints when evaluation fails

# GP hyperparameters
LENGTHSCALE_PRIOR_CONCENTRATION = 3.0
LENGTHSCALE_PRIOR_RATE = 6.0
OUTPUTSCALE_PRIOR_CONCENTRATION = 2.0
OUTPUTSCALE_PRIOR_RATE = 0.15
MIN_STANDARDIZATION_STD = 1e-6  # Minimum std for standardization

# Constraint computation
MIN_RESIDUAL_VALUE = 1e-20  # Minimum residual to avoid log(0)
MAX_RESIDUAL_VALUE = 1e10  # Maximum residual for diverged cases

# Optimization tolerances
SCIPY_FTOL = 1e-6  # Function tolerance for scipy optimizer
SCIPY_MAX_ITER = 100  # Maximum iterations for scipy optimizer
DIFFERENTIAL_EVOLUTION_MAX_ITER = 100  # Maximum iterations for differential evolution

# Constraint penalty for differential evolution
CONSTRAINT_PENALTY_WEIGHT = 100.0  # Weight for constraint violation penalty
