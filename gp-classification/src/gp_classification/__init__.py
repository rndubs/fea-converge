"""
GP Classification for FEA Contact Convergence Optimization

This package implements Gaussian Process Classification for optimizing
finite element contact simulation parameters using variational inference
and constrained Bayesian optimization.
"""

__version__ = "0.1.0"

from .models import VariationalGPClassifier, DualModel
from .acquisition import ConstrainedEI, EntropyAcquisition, AdaptiveAcquisition
from .optimizer import GPClassificationOptimizer
from .data import TrialDatabase, SimulationTrial
from .use_cases import ParameterSuggester, PreSimulationValidator, RealTimeEstimator
from .visualization import OptimizationVisualizer

__all__ = [
    "VariationalGPClassifier",
    "DualModel",
    "ConstrainedEI",
    "EntropyAcquisition",
    "AdaptiveAcquisition",
    "GPClassificationOptimizer",
    "TrialDatabase",
    "SimulationTrial",
    "ParameterSuggester",
    "PreSimulationValidator",
    "RealTimeEstimator",
    "OptimizationVisualizer",
]
