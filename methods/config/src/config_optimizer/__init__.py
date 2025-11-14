"""
CONFIG (Constrained Efficient Global Optimization) Package

This package implements the CONFIG algorithm for Bayesian optimization
with unknown constraints, providing rigorous theoretical guarantees on
both convergence and constraint violations.
"""

from .core.controller import CONFIGController
from .models.gp_models import ObjectiveGP, ConstraintGP
from .acquisition.config_acquisition import CONFIGAcquisition
from .monitoring.violation_monitor import ViolationMonitor
from .solvers.black_box_solver import BlackBoxSolver

__version__ = "0.1.0"

__all__ = [
    "CONFIGController",
    "ObjectiveGP",
    "ConstraintGP",
    "CONFIGAcquisition",
    "ViolationMonitor",
    "BlackBoxSolver",
]
