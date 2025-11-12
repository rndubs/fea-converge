"""
Basic GP Classification optimization example.

This script demonstrates:
1. Setting up the optimizer with a mock solver
2. Running the three-phase optimization
3. Generating suggestions for new simulations
4. Validating parameters before simulation
5. Creating visualizations
"""

from pathlib import Path

from gp_classification import (
    GPClassificationOptimizer,
    ParameterSuggester,
    PreSimulationValidator,
    OptimizationVisualizer,
)
from gp_classification.mock_solver import MockSmithSolver, get_default_parameter_bounds


def main():
    """Run basic optimization example."""
    print("=" * 80)
    print("GP Classification Optimization Example")
    print("=" * 80)

    # 1. Set up parameter bounds
    parameter_bounds = get_default_parameter_bounds()

    print("\nParameter Bounds:")
    for name, (lower, upper) in parameter_bounds.items():
        print(f"  {name:25s}: [{lower:.2e}, {upper:.2e}]")

    # 2. Create mock solver (replace with real Smith solver in production)
    print("\nCreating mock solver (difficulty: medium)...")
    mock_solver = MockSmithSolver(random_seed=42, noise_level=0.1, difficulty="medium")

    def simulator(params):
        """Wrapper for mock solver."""
        return mock_solver.simulate(params)

    # 3. Create optimizer
    print("\nInitializing GP Classification Optimizer...")
    optimizer = GPClassificationOptimizer(
        parameter_bounds=parameter_bounds,
        simulator=simulator,
        n_initial_samples=20,
        phase1_end=30,
        phase2_end=50,
        n_inducing_points=50,
        verbose=True,
    )

    # 4. Run optimization
    print("\nStarting optimization...")
    best_params = optimizer.optimize(n_iterations=60)

    # 5. Display results
    print("\n" + "=" * 80)
    print("Optimization Results")
    print("=" * 80)

    stats = optimizer.get_statistics()
    print(f"\nTotal Trials: {stats['total_trials']}")
    print(f"Converged: {stats['converged_trials']} ({stats['convergence_rate']:.1%})")
    print(f"Failed: {stats['failed_trials']}")

    if "best_objective" in stats:
        print(f"\nBest Objective Value: {stats['best_objective']:.4f}")
        print("\nBest Parameters:")
        for name, value in best_params.items():
            print(f"  {name:25s}: {value:.6e}")

    # 6. Parameter suggestions for new geometry
    print("\n" + "=" * 80)
    print("Parameter Suggestions for New Geometry")
    print("=" * 80)

    suggester = ParameterSuggester(
        database=optimizer.database,
        dual_model=optimizer.dual_model,
        n_clusters=3,
    )

    suggestions = suggester.suggest_parameters()

    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"\nSuggestion {i}:")
        print(f"  Convergence Probability: {suggestion['convergence_probability']:.2%}")
        print(f"  Confidence: {suggestion['confidence']}")
        if suggestion["expected_objective"] is not None:
            print(f"  Expected Iterations: {suggestion['expected_objective']:.1f}")
        print(f"  Similar Trials: {suggestion['n_similar_trials']}")

    # 7. Pre-simulation validation
    print("\n" + "=" * 80)
    print("Pre-Simulation Validation Example")
    print("=" * 80)

    validator = PreSimulationValidator(
        database=optimizer.database,
        dual_model=optimizer.dual_model,
        min_convergence_prob=0.3,
    )

    # Validate best parameters
    print("\nValidating best parameters...")
    result = validator.validate(best_params)

    print(f"\nValidation Result:")
    print(f"  Approved: {result['approved']}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Risk Score: {result['risk_score']:.2f}")
    print(f"  ML Prediction: {result['ml_prediction']:.2%}")
    print(f"  Recommendation: {result['recommendation']}")

    if result["physics_violations"]:
        print(f"  Physics Violations: {result['physics_violations']}")

    # 8. Create visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    visualizer = OptimizationVisualizer(
        database=optimizer.database,
        dual_model=optimizer.dual_model,
        parameter_names=list(parameter_bounds.keys()),
    )

    print("\nCreating convergence landscape...")
    fig = visualizer.plot_convergence_landscape_2d(
        param1_name="penalty_stiffness",
        param2_name="gap_tolerance",
        resolution=50,
        save_path=output_dir / "convergence_landscape.png",
    )
    print(f"  Saved to: {output_dir / 'convergence_landscape.png'}")

    print("\nCreating uncertainty map...")
    fig = visualizer.plot_uncertainty_map(
        param1_name="penalty_stiffness",
        param2_name="gap_tolerance",
        resolution=50,
        save_path=output_dir / "uncertainty_map.png",
    )
    print(f"  Saved to: {output_dir / 'uncertainty_map.png'}")

    print("\nCreating optimization history...")
    fig = visualizer.plot_optimization_history(
        save_path=output_dir / "optimization_history.png"
    )
    print(f"  Saved to: {output_dir / 'optimization_history.png'}")

    print("\nCreating parameter importance plot...")
    fig = visualizer.plot_parameter_importance(
        save_path=output_dir / "parameter_importance.png"
    )
    print(f"  Saved to: {output_dir / 'parameter_importance.png'}")

    print("\nCreating calibration plot...")
    fig = visualizer.plot_calibration(save_path=output_dir / "calibration.png")
    print(f"  Saved to: {output_dir / 'calibration.png'}")

    print("\nCreating summary dashboard...")
    fig = visualizer.create_summary_dashboard(save_path=output_dir / "dashboard.png")
    print(f"  Saved to: {output_dir / 'dashboard.png'}")

    # 9. Save database
    print("\nSaving trial database...")
    optimizer.database.save(output_dir / "trial_database.csv")
    print(f"  Saved to: {output_dir / 'trial_database.csv'}")

    print("\n" + "=" * 80)
    print("Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
