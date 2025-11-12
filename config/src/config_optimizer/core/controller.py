"""
Main CONFIG controller implementing the complete algorithm.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import pickle
import json

from ..models.gp_models import ObjectiveGP, ConstraintGP
from ..acquisition.config_acquisition import CONFIGAcquisition
from ..monitoring.violation_monitor import ViolationMonitor
from ..utils.beta_schedule import compute_beta
from ..utils.constraints import compute_multiple_constraints
from ..utils.sampling import latin_hypercube_sampling


@dataclass
class CONFIGConfig:
    """Configuration for CONFIG optimizer."""
    bounds: np.ndarray
    constraint_configs: Dict[str, Dict[str, float]]
    delta: float = 0.1
    n_init: int = 20
    n_max: int = 100
    acquisition_method: str = "discrete"
    n_restarts: int = 20
    seed: Optional[int] = None


class CONFIGController:
    """
    Main CONFIG optimizer controller.
    
    Implements the complete CONFIG algorithm with:
    - Multi-phase optimization strategy
    - GP model management
    - Acquisition optimization
    - Violation monitoring
    """
    
    def __init__(
        self,
        config: CONFIGConfig,
        objective_function: Callable[[np.ndarray], Dict[str, Any]]
    ):
        """
        Initialize CONFIG controller.
        
        Args:
            config: CONFIG configuration
            objective_function: Black-box function to optimize
                Takes x (np.ndarray) and returns dict with:
                - 'objective_value': float
                - simulation results for constraint computation
        """
        self.config = config
        self.objective_function = objective_function
        
        # Initialize data storage
        self.X_observed = []
        self.y_observed = []
        self.constraint_values = {name: [] for name in config.constraint_configs.keys()}
        self.iteration = 0
        
        # Initialize GPs
        self.objective_gp = ObjectiveGP(config.bounds)
        self.constraint_gps = {
            name: ConstraintGP(config.bounds, constraint_name=name)
            for name in config.constraint_configs.keys()
        }
        
        # Initialize monitoring
        self.violation_monitor = ViolationMonitor()
        
        # Track best feasible solution
        self.best_feasible_x = None
        self.best_feasible_y = float('inf')
        
        # Set random seed
        if config.seed is not None:
            np.random.seed(config.seed)
    
    def initialize(self):
        """
        Phase 1: Initialization with Latin Hypercube Sampling.
        """
        print(f"Initializing with {self.config.n_init} LHS samples...")
        
        # Generate initial samples
        X_init = latin_hypercube_sampling(
            self.config.bounds,
            self.config.n_init,
            seed=self.config.seed
        )
        
        # Evaluate all initial points
        for i, x in enumerate(X_init):
            print(f"  Evaluating initial point {i+1}/{self.config.n_init}...", end='\r')
            self._evaluate_and_update(x)
        
        print(f"\nInitialization complete. {self._count_feasible()} feasible points found.")
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run the complete CONFIG optimization.
        
        Returns:
            Results dictionary with best solution and statistics
        """
        # Initialize if not done
        if self.iteration == 0:
            self.initialize()
        
        # Main optimization loop
        while self.iteration < self.config.n_max:
            print(f"\n=== Iteration {self.iteration + 1}/{self.config.n_max} ===")
            
            # Determine phase
            phase = self._determine_phase()
            print(f"Phase: {phase}")
            
            # Fit GP models
            self._fit_models()
            
            # Compute beta
            beta = compute_beta(self.iteration + 1, self.config.delta)
            print(f"Beta: {beta:.3f}")
            
            # Propose next point
            if phase == "constrained_optimization":
                next_x = self._propose_next_config(beta)
            elif phase == "feasibility_discovery":
                next_x = self._propose_boundary_exploration(beta)
            else:  # initialization phase continued
                next_x = self._propose_random()
            
            # Evaluate
            self._evaluate_and_update(next_x)
            
            # Report status
            self._print_status()
            
            # Check termination
            if self._check_termination():
                print("\nTermination criterion met.")
                break
        
        return self._get_results()
    
    def _evaluate_and_update(self, x: np.ndarray):
        """
        Evaluate point and update all data structures.
        
        Args:
            x: Point to evaluate
        """
        # Evaluate objective function
        result = self.objective_function(x)
        
        # Extract objective value
        obj_value = result['objective_value']
        
        # Compute constraints
        constraints = compute_multiple_constraints(
            result,
            self.config.constraint_configs
        )
        
        # Update data
        self.X_observed.append(x)
        self.y_observed.append(obj_value)
        for name, value in constraints.items():
            self.constraint_values[name].append(value)
        
        # Update violation monitor (use first constraint for simplicity)
        # In practice, could track all constraints
        first_constraint_value = list(constraints.values())[0]
        self.violation_monitor.add_violation(first_constraint_value)
        
        # Update best feasible if applicable
        all_feasible = all(v <= 0 for v in constraints.values())
        if all_feasible and obj_value < self.best_feasible_y:
            self.best_feasible_x = x
            self.best_feasible_y = obj_value
            print(f"  New best feasible: {obj_value:.4f}")
        
        self.iteration += 1
    
    def _fit_models(self):
        """Fit all GP models."""
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # Fit objective GP
        self.objective_gp.fit(X, y)
        
        # Fit constraint GPs
        for name, gp in self.constraint_gps.items():
            y_constraint = np.array(self.constraint_values[name])
            gp.fit(X, y_constraint)
    
    def _propose_next_config(self, beta: float) -> np.ndarray:
        """
        Propose next point using CONFIG acquisition.
        
        Args:
            beta: Current beta value
            
        Returns:
            Next point to evaluate
        """
        acqf = CONFIGAcquisition(
            self.objective_gp,
            list(self.constraint_gps.values()),
            beta,
            self.config.bounds
        )
        
        next_x, _ = acqf.optimize(
            n_restarts=self.config.n_restarts,
            method=self.config.acquisition_method
        )
        
        return next_x
    
    def _propose_boundary_exploration(self, beta: float) -> np.ndarray:
        """
        Propose point for boundary exploration (active learning).
        
        Targets high uncertainty near constraint boundary.
        
        Args:
            beta: Current beta value
            
        Returns:
            Next point to evaluate
        """
        from ..utils.sampling import generate_candidate_set
        
        # Generate candidates
        candidates = generate_candidate_set(
            self.config.bounds,
            n_candidates=1000,
            method="sobol"
        )
        
        # Compute uncertainty and boundary proximity
        scores = []
        for gp in self.constraint_gps.values():
            mean, std = gp.predict(candidates)
            lcb = mean - np.sqrt(beta) * std
            
            # Score = uncertainty × boundary proximity
            boundary_proximity = np.exp(-5 * lcb**2)
            score = std * boundary_proximity
            scores.append(score)
        
        # Select point with highest score
        total_score = np.sum(scores, axis=0)
        best_idx = np.argmax(total_score)
        
        return candidates[best_idx]
    
    def _propose_random(self) -> np.ndarray:
        """Propose random point (for continued initialization)."""
        return np.random.uniform(
            self.config.bounds[:, 0],
            self.config.bounds[:, 1]
        )
    
    def _determine_phase(self) -> str:
        """
        Determine current optimization phase.
        
        Returns:
            Phase name
        """
        n_feasible = self._count_feasible()
        
        if self.iteration < self.config.n_init:
            return "initialization"
        
        if n_feasible == 0 and self.iteration < self.config.n_init + 20:
            return "feasibility_discovery"
        
        # Check constraint uncertainty
        max_uncertainty = self._get_max_constraint_uncertainty()
        if max_uncertainty > 0.3:
            return "feasibility_discovery"
        
        return "constrained_optimization"
    
    def _count_feasible(self) -> int:
        """Count number of feasible evaluations."""
        count = 0
        for i in range(len(self.X_observed)):
            all_feasible = all(
                self.constraint_values[name][i] <= 0
                for name in self.constraint_values.keys()
            )
            if all_feasible:
                count += 1
        return count
    
    def _get_max_constraint_uncertainty(self) -> float:
        """Get maximum constraint uncertainty across recent evaluations."""
        if len(self.X_observed) < 5:
            return float('inf')

        # Check if models are fitted
        if any(gp.model is None for gp in self.constraint_gps.values()):
            return float('inf')

        # Sample recent evaluations
        recent_X = np.array(self.X_observed[-10:])

        max_uncertainty = 0.0
        for gp in self.constraint_gps.values():
            _, std = gp.predict(recent_X)
            max_uncertainty = max(max_uncertainty, np.max(std))

        return max_uncertainty
    
    def _check_termination(self) -> bool:
        """Check if termination criteria are met."""
        # Budget exhausted
        if self.iteration >= self.config.n_max:
            return True
        
        # Best solution stable for many iterations
        if self.best_feasible_x is not None and self.iteration >= 50:
            # Check if best hasn't changed in 15 iterations
            recent_feasible_count = 0
            for i in range(max(0, self.iteration - 15), self.iteration):
                all_feasible = all(
                    self.constraint_values[name][i] <= 0
                    for name in self.constraint_values.keys()
                )
                if all_feasible and self.y_observed[i] < self.best_feasible_y + 1e-3:
                    recent_feasible_count += 1
            
            if recent_feasible_count == 0:
                return True
        
        return False
    
    def _print_status(self):
        """Print current optimization status."""
        stats = self.violation_monitor.get_statistics()
        print(f"  Feasible: {self._count_feasible()}/{self.iteration}")
        print(f"  Best feasible: {self.best_feasible_y:.6f}" if self.best_feasible_x is not None else "  Best feasible: None")
        print(f"  Cumulative violations: {stats['cumulative_violation']:.4f}")
        
        # Check theoretical bound
        bound_check = self.violation_monitor.check_theoretical_bound(self.iteration)
        print(f"  Violation bound status: {bound_check['status']}")
    
    def _get_results(self) -> Dict[str, Any]:
        """
        Get final results.
        
        Returns:
            Results dictionary
        """
        return {
            'best_x': self.best_feasible_x,
            'best_y': self.best_feasible_y,
            'n_evaluations': self.iteration,
            'n_feasible': self._count_feasible(),
            'X_observed': np.array(self.X_observed),
            'y_observed': np.array(self.y_observed),
            'constraint_values': {k: np.array(v) for k, v in self.constraint_values.items()},
            'violation_statistics': self.violation_monitor.get_statistics(),
            'violation_bound_check': self.violation_monitor.check_theoretical_bound(self.iteration)
        }
    
    def save(self, filepath: str):
        """Save optimizer state."""
        state = {
            'config': self.config,
            'X_observed': self.X_observed,
            'y_observed': self.y_observed,
            'constraint_values': self.constraint_values,
            'iteration': self.iteration,
            'best_feasible_x': self.best_feasible_x,
            'best_feasible_y': self.best_feasible_y
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"State saved to {filepath}")
