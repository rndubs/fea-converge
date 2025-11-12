#!/usr/bin/env python3
"""
Basic example of using CONFIG optimizer.

This example demonstrates how to use CONFIG to optimize a synthetic
test function with convergence constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
from config_optimizer.core.controller import CONFIGController, CONFIGConfig
from config_optimizer.solvers.black_box_solver import BlackBoxSolver


def main():
    print("="*60)
    print("CONFIG Optimizer - Basic Example")
    print("="*60)
    
    # 1. Create a black box solver (simulates FEA)
    print("\n1. Setting up black box solver...")
    solver = BlackBoxSolver(
        problem_type="branin",  # 2D test function
        noise_level=0.01,
        seed=42
    )
    print(f"   Problem: {solver.problem_type}")
    print(f"   Dimension: {solver.dim}D")
    print(f"   Bounds: {solver.bounds}")
    
    # 2. Define objective function
    def objective_function(x):
        """Evaluate the black box function."""
        result = solver.evaluate(x)
        return {
            'objective_value': result.objective_value,
            'final_residual': result.final_residual,
            'iterations': result.iterations,
            'converged': result.converged
        }
    
    # 3. Configure CONFIG optimizer
    print("\n2. Configuring CONFIG optimizer...")
    config = CONFIGConfig(
        bounds=solver.bounds,
        constraint_configs={
            'convergence': {'tolerance': 1e-8}
        },
        delta=0.1,           # Confidence parameter
        n_init=15,           # Initial LHS samples
        n_max=50,            # Maximum evaluations
        acquisition_method="discrete",  # Robust optimization method
        seed=42
    )
    
    # 4. Create optimizer
    optimizer = CONFIGController(config, objective_function)
    
    # 5. Run optimization
    print("\n3. Running optimization...")
    print("-"*60)
    results = optimizer.optimize()
    
    # 6. Print results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best feasible value: {results['best_y']:.6f}")
    print(f"Best feasible point: {results['best_x']}")
    print(f"Total evaluations: {results['n_evaluations']}")
    print(f"Feasible evaluations: {results['n_feasible']}")
    print(f"Success rate: {results['n_feasible']/results['n_evaluations']*100:.1f}%")
    
    print("\nViolation Statistics:")
    stats = results['violation_statistics']
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTheoretical Bound Check:")
    bound_check = results['violation_bound_check']
    print(f"  Status: {bound_check['status']}")
    print(f"  Actual violations: {bound_check['V_t']:.4f}")
    print(f"  Theoretical bound: {bound_check['bound']:.4f}")
    print(f"  Ratio: {bound_check['ratio']:.4f}")
    
    # 7. Plot violations
    print("\n4. Generating violation plot...")
    try:
        fig = optimizer.violation_monitor.plot_violations(
            save_path="examples/violation_plot.png"
        )
        print("   Saved to: examples/violation_plot.png")
    except Exception as e:
        print(f"   Could not generate plot: {e}")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)


if __name__ == "__main__":
    main()
