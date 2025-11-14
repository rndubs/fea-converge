"""
Smith FEA Integration Example for FR-BO

This example demonstrates how to integrate FR-BO with the Smith/Serac/Tribol
finite element solver framework for contact convergence optimization.

Note: This example requires Smith to be built and available. In Claude Code
web environments, Smith cannot be built due to network restrictions.
"""

import sys
from pathlib import Path


# Placeholder for Smith integration
class SmithTribolExecutor:
    """
    Executor for Smith/Serac/Tribol contact simulations.

    This class wraps the Smith FEA solver and provides a clean interface
    for FR-BO optimization.

    Attributes:
        mesh_file: Path to mesh file
        material_props: Material properties dictionary
        contact_config: Contact configuration dictionary
        solver_config: Solver configuration dictionary
    """

    def __init__(
        self,
        mesh_file: str,
        material_props: dict,
        contact_config: dict = None,
        solver_config: dict = None
    ):
        """
        Initialize Smith/Tribol executor.

        Args:
            mesh_file: Path to Exodus mesh file
            material_props: Material properties (E, nu, rho, etc.)
            contact_config: Contact method configuration
            solver_config: Nonlinear solver configuration
        """
        self.mesh_file = mesh_file
        self.material_props = material_props
        self.contact_config = contact_config or {}
        self.solver_config = solver_config or {}

        # Check if Smith is available
        try:
            # In a real implementation, you would:
            # import serac
            # import tribol
            # self.serac_available = True
            self.serac_available = False
            print("Warning: Smith/Serac not available. Using mock simulator.")
        except ImportError:
            self.serac_available = False

    def run(self, parameters: dict):
        """
        Run Smith simulation with given parameters.

        Args:
            parameters: Dictionary of solver/contact parameters
                - penalty_stiffness: Contact penalty stiffness
                - gap_tolerance: Gap tolerance for contact
                - projection_tolerance: Projection tolerance
                - max_iterations: Max nonlinear iterations
                - abs_tolerance: Absolute convergence tolerance
                - rel_tolerance: Relative convergence tolerance
                - solver_type: Nonlinear solver type
                - etc.

        Returns:
            SimulationResult with convergence info
        """
        from fr_bo.simulator import SimulationResult

        if not self.serac_available:
            # Mock implementation for demonstration
            return self._mock_simulation(parameters)

        # Real Smith/Serac implementation would go here:
        """
        # 1. Configure Tribol contact
        tribol.configure_contact(
            enforcement=parameters.get('enforcement_method', 'penalty'),
            penalty_stiffness=parameters.get('penalty_stiffness', 1e5),
            gap_tolerance=parameters.get('gap_tolerance', 1e-8),
            ...
        )

        # 2. Configure Serac solver
        serac.configure_solver(
            solver_type=parameters.get('solver_type', 'Newton'),
            max_iterations=int(parameters.get('max_iterations', 100)),
            abs_tolerance=parameters.get('abs_tolerance', 1e-8),
            rel_tolerance=parameters.get('rel_tolerance', 1e-6),
            ...
        )

        # 3. Run simulation
        try:
            result = serac.solve(
                mesh=self.mesh_file,
                materials=self.material_props,
                time_steps=...
            )

            # 4. Extract convergence metrics
            converged = result.converged
            final_residual = result.final_residual
            iterations = result.num_iterations
            objective_value = self._compute_objective(result)

            return SimulationResult(
                converged=converged,
                final_residual=final_residual,
                iterations=iterations,
                objective_value=objective_value,
                failed=not converged,
                parameters=parameters
            )

        except Exception as e:
            # Simulation crashed/failed
            return SimulationResult(
                converged=False,
                final_residual=1.0,
                iterations=0,
                objective_value=float('inf'),
                failed=True,
                parameters=parameters,
                error_message=str(e)
            )
        """

    def _mock_simulation(self, parameters: dict):
        """Mock simulation for demonstration."""
        from fr_bo.simulator import SimulationResult
        import numpy as np

        # Simulate Smith-like behavior
        penalty = parameters.get('penalty_stiffness', 1e5)
        gap_tol = parameters.get('gap_tolerance', 1e-8)
        max_iter = int(parameters.get('max_iterations', 100))

        # Mock objective (lower is better)
        # Penalty too low or too high -> convergence issues
        log_penalty = np.log10(penalty)
        optimal_log_penalty = 5.5  # ~3e5
        penalty_term = (log_penalty - optimal_log_penalty)**2

        log_gap = np.log10(gap_tol)
        optimal_log_gap = -8  # 1e-8
        gap_term = (log_gap - optimal_log_gap)**2

        objective = penalty_term + gap_term + np.random.randn() * 0.1

        # Mock convergence
        converged = objective < 2.0 and max_iter >= 50
        final_residual = 10**(-6 - objective) if converged else 1e-2
        iterations = min(int(50 + objective * 10), max_iter)

        return SimulationResult(
            converged=converged,
            final_residual=final_residual,
            iterations=iterations,
            objective_value=objective,
            failed=not converged,
            parameters=parameters
        )

    def _compute_objective(self, smith_result):
        """
        Compute optimization objective from Smith results.

        This could combine multiple factors:
        - Number of nonlinear iterations
        - Final residual norm
        - Contact penetration violations
        - Computational time
        """
        # Example multi-objective formulation:
        # obj = w1 * iterations + w2 * log(residual) - w3 * penetration_quality
        pass


def main():
    """Example Smith/FR-BO optimization."""

    print("=" * 70)
    print("FR-BO with Smith/Tribol Integration Example")
    print("=" * 70)
    print()

    # Check if Smith is available
    try:
        # import serac
        # import tribol
        smith_available = False
    except:
        smith_available = False

    if not smith_available:
        print("WARNING: Smith/Serac not available in this environment.")
        print("Running with mock simulator for demonstration.\n")

    # Configure Smith executor
    print("1. Configuring Smith/Tribol executor...")

    executor = SmithTribolExecutor(
        mesh_file="contact_patch.e",  # Exodus mesh
        material_props={
            "youngs_modulus": 200e9,  # Pa (steel)
            "poissons_ratio": 0.3,
            "density": 7850.0  # kg/m^3
        },
        contact_config={
            "method": "penalty",
            "auto_contact": True
        },
        solver_config={
            "solver_type": "Newton",
            "line_search": True
        }
    )

    # Configure FR-BO
    print("2. Configuring FR-BO optimizer...")
    from fr_bo.optimizer import FRBOOptimizer, OptimizationConfig

    config = OptimizationConfig(
        n_sobol_trials=30,     # More initial trials for FEA
        n_frbo_trials=70,      # FR-BO iterations
        random_seed=42,
        max_iterations=1000,
        timeout=3600.0,        # 1 hour per simulation
        enable_early_termination=True,  # Stop failing sims early
        convergence_tolerance=1e-2,
        convergence_patience=15
    )

    # Initialize optimizer
    print("3. Initializing optimizer...")
    optimizer = FRBOOptimizer(
        simulator=executor,
        config=config
    )

    # Run optimization
    print("4. Running FR-BO optimization...")
    print("   This will optimize Smith/Tribol parameters for contact convergence")
    print()

    try:
        results = optimizer.optimize()

        # Print results
        print("\n" + "=" * 70)
        print("Optimization Results")
        print("=" * 70)

        print(f"\nTotal trials: {len(optimizer.trials)}")

        n_success = sum(1 for t in optimizer.trials if not t.result.failed)
        n_failure = sum(1 for t in optimizer.trials if t.result.failed)

        print(f"Successful simulations: {n_success}")
        print(f"Failed simulations: {n_failure}")
        print(f"Failure rate: {n_failure / len(optimizer.trials) * 100:.1f}%")

        print(f"\nBest objective value: {optimizer.best_objective:.6f}")
        print(f"\nOptimal Smith/Tribol parameters:")
        for key, value in optimizer.best_parameters.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6e}")
            else:
                print(f"  {key}: {value}")

        print("\n" + "=" * 70)
        print("Use these parameters in your Smith/Tribol simulations!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during optimization: {e}")
        print("\nNote: This example requires Smith/Serac to be built.")
        raise


if __name__ == "__main__":
    # Check if running in appropriate environment
    print("\nNOTE: Smith/Serac cannot be built in Claude Code web environments")
    print("due to network restrictions. Run this example in a local environment")
    print("with Smith installed.\n")

    main()
