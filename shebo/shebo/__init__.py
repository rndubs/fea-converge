"""SHEBO - Surrogate Optimization with Hidden Constraints"""

__version__ = "0.1.0"

from shebo.core.optimizer import SHEBOOptimizer
from shebo.models.ensemble import ConvergenceEnsemble
from shebo.core.surrogate_manager import SurrogateManager
from shebo.core.constraint_discovery import ConstraintDiscovery
from shebo.core.acquisition import AdaptiveAcquisition

__all__ = [
    "SHEBOOptimizer",
    "ConvergenceEnsemble",
    "SurrogateManager",
    "ConstraintDiscovery",
    "AdaptiveAcquisition",
]
