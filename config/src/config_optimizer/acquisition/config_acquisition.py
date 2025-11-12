"""
CONFIG acquisition function using Lower Confidence Bound.
"""

import torch
import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import List, Optional, Tuple, Callable
from ..models.gp_models import ObjectiveGP, ConstraintGP
from ..utils.beta_schedule import compute_beta


class CONFIGAcquisition:
    """
    CONFIG acquisition function using LCB for objective and constraints.
    
    Solves: minimize LCB_obj(x) subject to x ∈ F_opt
    where F_opt = {x : LCB_constraint(x) ≤ 0 for all constraints}
    """
    
    def __init__(
        self,
        objective_gp: ObjectiveGP,
        constraint_gps: List[ConstraintGP],
        beta: float,
        bounds: np.ndarray
    ):
        """
        Initialize CONFIG acquisition.
        
        Args:
            objective_gp: Fitted GP for objective
            constraint_gps: List of fitted GPs for constraints
            beta: Confidence parameter
            bounds: Parameter bounds
        """
        self.objective_gp = objective_gp
        self.constraint_gps = constraint_gps
        self.beta = beta
        self.bounds = bounds
        
    def compute_objective_lcb(self, X: np.ndarray) -> np.ndarray:
        """
        Compute LCB for objective: LCB = μ(x) - β^(1/2) * σ(x)
        
        Args:
            X: Test points of shape (n, d)
            
        Returns:
            LCB values of shape (n,)
        """
        mean, std = self.objective_gp.predict(X)
        lcb = mean - np.sqrt(self.beta) * std
        return lcb
    
    def compute_constraint_lcbs(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Compute LCB for all constraints.
        
        Args:
            X: Test points
            
        Returns:
            List of LCB arrays
        """
        lcbs = []
        for gp in self.constraint_gps:
            lcb = gp.compute_lcb(X, self.beta)
            lcbs.append(lcb)
        return lcbs
    
    def is_in_optimistic_set(self, X: np.ndarray) -> np.ndarray:
        """
        Check if points are in optimistic feasible set F_opt.
        
        Args:
            X: Test points of shape (n, d)
            
        Returns:
            Boolean array of shape (n,) indicating membership in F_opt
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        constraint_lcbs = self.compute_constraint_lcbs(X)
        
        # Point is in F_opt if ALL constraint LCBs <= 0
        in_F_opt = np.all([lcb <= 0 for lcb in constraint_lcbs], axis=0)
        
        return in_F_opt
    
    def optimize(
        self,
        n_restarts: int = 20,
        n_candidates: int = 1000,
        method: str = "scipy"
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize CONFIG acquisition function.
        
        Args:
            n_restarts: Number of random restarts for local optimization
            n_candidates: Number of candidates for discrete optimization
            method: 'scipy' for gradient-based, 'discrete' for candidate set
            
        Returns:
            (best_x, best_lcb_value)
        """
        if method == "scipy":
            return self._optimize_scipy(n_restarts)
        elif method == "discrete":
            return self._optimize_discrete(n_candidates)
        elif method == "differential_evolution":
            return self._optimize_differential_evolution()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_scipy(self, n_restarts: int, max_recursion: int = 3) -> Tuple[np.ndarray, float]:
        """
        Optimize using scipy.minimize with SLSQP.

        Args:
            n_restarts: Number of random restarts
            max_recursion: Maximum recursion depth for beta adjustment
        """
        best_x = None
        best_val = float('inf')

        def objective(x):
            X = x.reshape(1, -1)
            return self.compute_objective_lcb(X)[0]

        def constraint_func(x, con_idx):
            X = x.reshape(1, -1)
            lcb = self.constraint_gps[con_idx].compute_lcb(X, self.beta)[0]
            return -lcb  # scipy wants g(x) >= 0 for feasibility

        constraints = [
            {'type': 'ineq', 'fun': constraint_func, 'args': (i,)}
            for i in range(len(self.constraint_gps))
        ]

        # Try multiple random starts
        for _ in range(n_restarts):
            x0 = np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1]
            )

            try:
                result = minimize(
                    objective,
                    x0,
                    bounds=[(b[0], b[1]) for b in self.bounds],
                    constraints=constraints,
                    method='SLSQP',
                    options={'ftol': 1e-6, 'maxiter': 100}
                )

                if result.success and result.fun < best_val:
                    best_val = result.fun
                    best_x = result.x
            except (ValueError, RuntimeError) as e:
                # Optimization failed for this restart, try next one
                continue

        # If no feasible solution found, try adaptive beta
        if best_x is None and max_recursion > 0:
            # Increase beta to expand F_opt
            self.beta *= 1.5
            return self._optimize_scipy(max(n_restarts // 2, 5), max_recursion - 1)
        
        return best_x, best_val
    
    def _optimize_discrete(self, n_candidates: int) -> Tuple[np.ndarray, float]:
        """
        Optimize over discrete candidate set (more robust).
        """
        from ..utils.sampling import generate_candidate_set
        
        # Generate candidates
        candidates = generate_candidate_set(
            self.bounds,
            n_candidates,
            method="sobol"
        )
        
        # Compute LCBs
        obj_lcbs = self.compute_objective_lcb(candidates)
        in_F_opt = self.is_in_optimistic_set(candidates)
        
        # Find best feasible candidate
        feasible_indices = np.where(in_F_opt)[0]
        
        if len(feasible_indices) == 0:
            # No feasible candidates, return most likely feasible
            # (smallest maximum constraint LCB)
            constraint_lcbs = self.compute_constraint_lcbs(candidates)
            max_constraint_lcbs = np.max(constraint_lcbs, axis=0)
            best_idx = np.argmin(max_constraint_lcbs)
        else:
            # Select best among feasible
            feasible_lcbs = obj_lcbs[feasible_indices]
            best_feasible_idx = np.argmin(feasible_lcbs)
            best_idx = feasible_indices[best_feasible_idx]
        
        return candidates[best_idx], obj_lcbs[best_idx]
    
    def _optimize_differential_evolution(self) -> Tuple[np.ndarray, float]:
        """
        Optimize using differential evolution (global optimizer).
        """
        def objective(x):
            X = x.reshape(1, -1)
            lcb_obj = self.compute_objective_lcb(X)[0]
            
            # Add penalty for constraint violations
            constraint_lcbs = self.compute_constraint_lcbs(X)
            penalty = sum(max(0, lcb)**2 for lcb in [c[0] for c in constraint_lcbs])
            
            return lcb_obj + 100 * penalty
        
        bounds_list = [(b[0], b[1]) for b in self.bounds]
        
        result = differential_evolution(
            objective,
            bounds_list,
            maxiter=100,
            seed=42,
            workers=1
        )
        
        return result.x, result.fun
