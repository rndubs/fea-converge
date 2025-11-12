"""Simple example demonstrating SHEBO optimization.

This example shows how to use SHEBO to optimize FEA contact convergence
parameters using the black box solver.
"""

import numpy as np
import matplotlib.pyplot as plt

from shebo import SHEBOOptimizer
from shebo.utils.black_box_solver import create_test_objective
from shebo.visualization.plots import (
    plot_convergence_history,
    plot_performance_history,
    plot_constraint_timeline,
    plot_parameter_space_2d
)


def main():
    """Run simple SHEBO optimization example."""
    print("=" * 70)
    print("SHEBO Optimization Example")
    print("=" * 70)
    print()

    # Define parameter bounds
    # [penalty, tolerance, timestep, damping]
    bounds = np.array([
        [1e6, 1e10],    # penalty parameter
        [1e-8, 1e-4],   # tolerance
        [0.0, 1.0],     # timestep (normalized)
        [0.0, 1.0]      # damping (normalized)
    ])

    # Create test objective function
    print("Creating test objective function...")
    objective = create_test_objective(n_params=4, noise_level=0.1, random_seed=42)

    # Initialize SHEBO optimizer
    print("Initializing SHEBO optimizer...")
    optimizer = SHEBOOptimizer(
        bounds=bounds,
        objective_fn=objective,
        n_init=20,          # Initial space-filling samples
        budget=100,         # Total evaluation budget
        n_networks=5,       # Ensemble size
        random_seed=42
    )

    # Run optimization
    print("\nRunning optimization...")
    result = optimizer.run()

    # Print results
    print("\n" + "=" * 70)
    print("Optimization Results")
    print("=" * 70)
    print(f"Total evaluations: {len(result.all_params)}")
    print(f"Successful convergences: {sum(result.convergence_history)}")
    print(f"Success rate: {sum(result.convergence_history)/len(result.convergence_history)*100:.1f}%")
    print(f"Best performance: {result.best_performance:.2f} iterations")
    print()

    if result.best_params is not None and len(result.best_params) > 0:
        print("Best parameters found:")
        param_names = ['Penalty', 'Tolerance', 'Timestep', 'Damping']
        for i, (name, val) in enumerate(zip(param_names, result.best_params)):
            print(f"  {name:12s}: {val:.6e}")
    print()

    # Constraint discovery summary
    if result.discovered_constraints.get('total_constraints', 0) > 0:
        print("Discovered constraints:")
        for name, info in result.discovered_constraints.get('constraints', {}).items():
            print(f"  - {name:30s}: frequency={info['frequency']:3d}, "
                  f"severity={info['severity']}")
    else:
        print("No constraints discovered")
    print()

    # Create visualizations
    print("Creating visualizations...")

    # Convergence history
    fig1 = plot_convergence_history(result.convergence_history)
    fig1.savefig('convergence_history.png', dpi=150, bbox_inches='tight')
    print("  - convergence_history.png")

    # Performance history
    fig2 = plot_performance_history(
        result.performance_history,
        result.convergence_history
    )
    fig2.savefig('performance_history.png', dpi=150, bbox_inches='tight')
    print("  - performance_history.png")

    # Constraint timeline
    if result.discovered_constraints.get('total_constraints', 0) > 0:
        fig3 = plot_constraint_timeline(
            result.discovered_constraints.get('constraints', {}),
            result.iterations
        )
        fig3.savefig('constraint_timeline.png', dpi=150, bbox_inches='tight')
        print("  - constraint_timeline.png")

    # Parameter space (first two parameters)
    fig4 = plot_parameter_space_2d(
        result.all_params,
        result.convergence_history,
        param_indices=(0, 1),
        param_names=['Penalty', 'Tolerance', 'Timestep', 'Damping']
    )
    fig4.savefig('parameter_space.png', dpi=150, bbox_inches='tight')
    print("  - parameter_space.png")

    print("\nVisualization complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
