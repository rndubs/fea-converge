"""
Violation monitoring for CONFIG algorithm.

Tracks cumulative violations and validates theoretical bounds.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


class ViolationMonitor:
    """
    Tracks and validates constraint violations against theoretical bounds.
    """
    
    def __init__(self):
        """Initialize violation monitor."""
        self.violations = []
        self.constraint_values = []
        
    def add_violation(self, constraint_value: float):
        """
        Add a constraint evaluation result.
        
        Args:
            constraint_value: Constraint value (positive = violation)
        """
        violation = max(0.0, constraint_value)
        self.violations.append(violation)
        self.constraint_values.append(constraint_value)
    
    def cumulative_violation(self) -> float:
        """
        Compute cumulative violations: V_t = Σ max(0, c(x_i))
        
        Returns:
            Total cumulative violation
        """
        return sum(self.violations)
    
    def violation_rate(self) -> float:
        """
        Compute most recent violation rate.
        
        Returns:
            Last violation amount
        """
        if len(self.violations) < 1:
            return 0.0
        return self.violations[-1]
    
    def theoretical_bound(self, t: int, gamma_t: Optional[float] = None) -> float:
        """
        Compute theoretical violation bound.
        
        Theory: V_t = O(√(t γ_t log t))
        For Matérn kernels: γ_t ≈ log^(d+1) t
        
        Args:
            t: Current iteration
            gamma_t: Information gain (if None, uses log^5 t approximation)
            
        Returns:
            Theoretical violation bound
        """
        if t < 2:
            return 0.0
        
        if gamma_t is None:
            # Assume Matérn kernel with d=2: γ_t ≈ log^3 t
            gamma_t = np.log(t) ** 3
        
        bound = np.sqrt(t * gamma_t * np.log(t))
        return bound
    
    def check_theoretical_bound(
        self,
        t: int,
        gamma_t: Optional[float] = None,
        tolerance_factor: float = 2.0
    ) -> Dict[str, any]:
        """
        Check if violations are within theoretical bounds.
        
        Args:
            t: Current iteration
            gamma_t: Information gain
            tolerance_factor: Safety margin (default 2x)
            
        Returns:
            Status dictionary
        """
        V_t = self.cumulative_violation()
        bound = self.theoretical_bound(t, gamma_t)
        
        if V_t > tolerance_factor * bound:
            return {
                'status': 'WARNING',
                'message': 'Violations exceed theoretical bound',
                'V_t': V_t,
                'bound': bound,
                'ratio': V_t / bound if bound > 0 else float('inf')
            }
        else:
            return {
                'status': 'OK',
                'V_t': V_t,
                'bound': bound,
                'ratio': V_t / bound if bound > 0 else 0.0
            }
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get violation statistics.
        
        Returns:
            Dictionary with statistics
        """
        if len(self.violations) == 0:
            return {
                'total_evaluations': 0,
                'num_violations': 0,
                'violation_rate': 0.0,
                'cumulative_violation': 0.0,
                'mean_violation': 0.0,
                'max_violation': 0.0
            }
        
        violations_array = np.array(self.violations)
        num_violations = np.sum(violations_array > 0)
        
        return {
            'total_evaluations': len(self.violations),
            'num_violations': int(num_violations),
            'violation_rate': num_violations / len(self.violations),
            'cumulative_violation': self.cumulative_violation(),
            'mean_violation': np.mean(violations_array[violations_array > 0]) if num_violations > 0 else 0.0,
            'max_violation': np.max(violations_array) if len(violations_array) > 0 else 0.0
        }
    
    def plot_violations(
        self,
        gamma_t: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate violation trajectory plot vs theoretical bound.
        
        Args:
            gamma_t: Information gain array (if None, uses approximation)
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if len(self.violations) == 0:
            raise ValueError("No violations recorded yet")
        
        t = np.arange(1, len(self.violations) + 1)
        V_t = np.cumsum(self.violations)
        
        # Compute theoretical bound
        if gamma_t is None:
            theoretical = np.array([self.theoretical_bound(i) for i in t])
        else:
            theoretical = np.array([
                np.sqrt(i * gamma_t[i-1] * np.log(max(i, 2)))
                for i in t
            ])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Cumulative violations
        ax1.plot(t, V_t, 'b-', linewidth=2, label='Actual V_t')
        ax1.plot(t, theoretical, 'r--', linewidth=2, label='Theoretical O(√(t γ_t log t))')
        ax1.fill_between(t, 0, theoretical, alpha=0.2, color='red', label='Allowed region')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cumulative Violations')
        ax1.set_title('CONFIG Cumulative Violations vs Theoretical Bound')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Per-iteration violations
        ax2.bar(t, self.violations, color='orange', alpha=0.7, label='ΔV_t per iteration')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Per-Iteration Violation')
        ax2.set_title('Violation Rate (should decrease over time)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
