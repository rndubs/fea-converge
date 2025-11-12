"""
Basic FR-BO Optimization Example

This example demonstrates how to use FR-BO for a simple optimization problem
with a synthetic simulator that can fail.
"""

import numpy as np
import torch
from fr_bo.optimizer import FRBOOptimizer, OptimizationConfig
from fr_bo.simulator import SyntheticSimulator


def main():
    """Run a basic FR-BO optimization."""

    print("=" * 70)
    print("FR-BO Basic Optimization Example")
    print("=" * 70)
    print()

    # Configure optimization
    print("1. Configuring optimizer...")
    config = OptimizationConfig(
        n_sobol_trials=20,      # Initial space-filling samples
        n_frbo_trials=50,       # FR-BO iterations
        random_seed=42,         # For reproducibility
        max_iterations=1000,    # Max simulation iterations
        timeout=300.0,          # 5 minute timeout per simulation
        convergence_tolerance=1e-3,  # Convergence threshold
        convergence_patience=10      # Patience for early stopping
    )

    # Create synthetic simulator
    # In practice, this would be your FEA simulator
    print("2. Creating simulator...")
    simulator = SyntheticSimulator(
        random_seed=42,
        failure_rate=0.15,  # 15% of simulations fail
        noise_level=0.1     # 10% noise in objectives
    )

    # Initialize optimizer
    print("3. Initializing FR-BO optimizer...")
    optimizer = FRBOOptimizer(
        simulator=simulator,
        config=config
    )

    # Run optimization
    print("4. Running optimization...")
    print(f"   - Phase 1: {config.n_sobol_trials} Sobol initialization trials")
    print(f"   - Phase 2: {config.n_frbo_trials} FR-BO adaptive trials")
    print()

    try:
        results = optimizer.optimize()

        # Print results
        print()
        print("=" * 70)
        print("Optimization Results")
        print("=" * 70)

        print(f"\nTotal trials: {len(optimizer.trials)}")

        # Count successes and failures
        n_success = sum(1 for t in optimizer.trials if not t.result.failed)
        n_failure = sum(1 for t in optimizer.trials if t.result.failed)

        print(f"Successful trials: {n_success}")
        print(f"Failed trials: {n_failure}")
        print(f"Failure rate: {n_failure / len(optimizer.trials) * 100:.1f}%")

        # Best result
        print(f"\nBest objective value: {optimizer.best_objective:.6f}")
        print(f"Best trial number: {optimizer.best_trial_number}")
        print(f"\nBest parameters:")
        for key, value in optimizer.best_parameters.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6e}")
            else:
                print(f"  {key}: {value}")

        # Convergence history
        print(f"\nConvergence history (first 10 successful trials):")
        successful_trials = [t for t in optimizer.trials if not t.result.failed][:10]
        for i, trial in enumerate(successful_trials):
            print(f"  Trial {trial.trial_number:3d}: {trial.objective_value:.6f} "
                  f"({trial.phase})")

        print("\n" + "=" * 70)
        print("Optimization completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during optimization: {e}")
        print("\nNote: FR-BO is still in development (v0.1.0)")
        print("Some features may not be fully implemented yet.")
        raise


def plot_convergence(optimizer):
    """
    Plot convergence history (optional).

    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt

        # Extract successful trials
        trials = [t for t in optimizer.trials if not t.result.failed]
        if len(trials) == 0:
            print("No successful trials to plot")
            return

        trial_numbers = [t.trial_number for t in trials]
        objectives = [t.objective_value for t in trials]

        # Compute running best
        running_best = []
        current_best = float('inf')
        for obj in objectives:
            current_best = min(current_best, obj)
            running_best.append(current_best)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: All objectives
        ax1.scatter(trial_numbers, objectives, alpha=0.6, label='Observed')
        ax1.plot(trial_numbers, running_best, 'r-', linewidth=2, label='Best so far')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('FR-BO Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Success/Failure
        all_trials = optimizer.trials
        success = [t.trial_number for t in all_trials if not t.result.failed]
        failures = [t.trial_number for t in all_trials if t.result.failed]

        ax2.scatter(success, [1] * len(success), c='green', marker='o',
                   s=50, label='Success', alpha=0.6)
        ax2.scatter(failures, [0] * len(failures), c='red', marker='x',
                   s=50, label='Failure', alpha=0.6)
        ax2.set_xlabel('Trial Number')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Failure', 'Success'])
        ax2.set_title('Trial Outcomes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('frbo_convergence.png', dpi=150)
        print("\nConvergence plot saved to: frbo_convergence.png")

    except ImportError:
        print("\nmatplotlib not available for plotting")
    except Exception as e:
        print(f"\nError creating plot: {e}")


if __name__ == "__main__":
    main()

    # Optionally plot results
    # Uncomment if you want to generate plots:
    # from fr_bo.optimizer import FRBOOptimizer
    # plot_convergence(optimizer)  # Note: requires running main() first
