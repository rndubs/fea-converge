"""
Example: Integrating CONFIG with Smith FEA Solver

This example demonstrates how to use CONFIG optimizer to tune
Smith FEA solver parameters for optimal convergence.

NOTE: This is a template/example. Actual Smith integration requires:
1. Built Smith executable
2. Lua input template with parameter placeholders
3. Network access for building Smith (see README.md)
"""

import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any
import re

from config_optimizer.core.controller import CONFIGController, CONFIGConfig


class SmithFEAOptimizer:
    """
    Wrapper for optimizing Smith FEA solver parameters using CONFIG.

    This class handles:
    - Generating Lua input files with parameters
    - Running Smith simulations
    - Parsing convergence metrics from output
    - Interfacing with CONFIG optimizer
    """

    def __init__(
        self,
        smith_executable: str,
        lua_template: str,
        work_dir: str = "./smith_opt_runs",
        mpi_procs: int = 1
    ):
        """
        Initialize Smith FEA optimizer.

        Args:
            smith_executable: Path to Smith executable
            lua_template: Path to Lua template file
            work_dir: Working directory for optimization runs
            mpi_procs: Number of MPI processes
        """
        self.smith_executable = Path(smith_executable)
        self.lua_template = Path(lua_template)
        self.work_dir = Path(work_dir)
        self.mpi_procs = mpi_procs
        self.work_dir.mkdir(parents=True, exist_ok=True)

        if not self.smith_executable.exists():
            raise FileNotFoundError(
                f"Smith executable not found: {smith_executable}\n"
                f"Please build Smith first. See README.md for instructions."
            )

        if not self.lua_template.exists():
            raise FileNotFoundError(
                f"Lua template not found: {lua_template}\n"
                f"Create a template with parameter placeholders like: {{ solver_tolerance }}"
            )

    def create_lua_input(
        self,
        parameters: Dict[str, float],
        output_path: Path
    ) -> None:
        """
        Create Lua input file from template with parameters.

        Args:
            parameters: Dictionary of parameter name → value
            output_path: Path to output Lua file
        """
        # Read template
        with open(self.lua_template, 'r') as f:
            lua_content = f.read()

        # Replace placeholders
        for param_name, param_value in parameters.items():
            placeholder = f"{{{{ {param_name} }}}}"
            lua_content = lua_content.replace(placeholder, str(param_value))

        # Write output
        with open(output_path, 'w') as f:
            f.write(lua_content)

    def run_smith_simulation(
        self,
        lua_input: Path,
        timeout: int = 600
    ) -> Dict[str, Any]:
        """
        Run Smith simulation and extract metrics.

        Args:
            lua_input: Path to Lua input file
            timeout: Timeout in seconds

        Returns:
            Dictionary with simulation metrics
        """
        cmd = [
            'mpirun', '-np', str(self.mpi_procs),
            str(self.smith_executable),
            str(lua_input)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.work_dir
            )

            # Parse output
            metrics = self.parse_smith_output(result.stdout, result.stderr)
            metrics['success'] = (result.returncode == 0)
            metrics['returncode'] = result.returncode

            return metrics

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'converged': False,
                'final_residual': 1e10,
                'iterations': 99999,
                'solve_time': float('inf'),
                'error': 'timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'converged': False,
                'final_residual': 1e10,
                'iterations': 99999,
                'solve_time': float('inf'),
                'error': str(e)
            }

    def parse_smith_output(
        self,
        stdout: str,
        stderr: str
    ) -> Dict[str, Any]:
        """
        Parse Smith output to extract convergence metrics.

        NOTE: This is a template. Actual parsing depends on Smith's output format.

        Args:
            stdout: Standard output from Smith
            stderr: Standard error from Smith

        Returns:
            Dictionary of parsed metrics
        """
        metrics = {
            'converged': False,
            'final_residual': 1e10,
            'iterations': 0,
            'solve_time': 0.0
        }

        # Example parsing patterns (adjust based on actual Smith output)

        # Look for convergence
        if re.search(r'Converged|CONVERGED|Solution converged', stdout):
            metrics['converged'] = True

        # Look for residual
        residual_match = re.search(r'[Rr]esidual[:\s]+([0-9.eE+-]+)', stdout)
        if residual_match:
            metrics['final_residual'] = float(residual_match.group(1))

        # Look for iterations
        iter_match = re.search(r'[Ii]terations?[:\s]+([0-9]+)', stdout)
        if iter_match:
            metrics['iterations'] = int(iter_match.group(1))

        # Look for solve time
        time_match = re.search(r'[Ss]olve [Tt]ime[:\s]+([0-9.]+)', stdout)
        if time_match:
            metrics['solve_time'] = float(time_match.group(1))

        return metrics

    def objective_function(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Objective function for CONFIG optimizer.

        Takes parameter vector x and returns simulation metrics.

        Args:
            x: Parameter vector [solver_tol, penalty_param, max_iters, ...]

        Returns:
            Dictionary with 'objective_value' and constraint metrics
        """
        # Map array to parameter names
        parameters = {
            'solver_tolerance': x[0],
            'penalty_parameter': x[1],
            'max_iterations': int(x[2]) if len(x) > 2 else 1000,
        }

        # Create unique run directory
        run_id = abs(hash(tuple(x))) % 10000000
        run_dir = self.work_dir / f"run_{run_id}"
        run_dir.mkdir(exist_ok=True)

        # Create Lua input
        lua_input = run_dir / "input.lua"
        self.create_lua_input(parameters, lua_input)

        # Run simulation
        metrics = self.run_smith_simulation(lua_input)

        # Compute objective (minimize solve time + iteration penalty)
        if metrics.get('converged', False):
            objective = metrics['solve_time'] + 0.01 * metrics['iterations']
        else:
            # Large penalty for non-convergence
            objective = 1e6 + metrics['iterations']

        return {
            'objective_value': objective,
            'final_residual': metrics['final_residual'],
            'iterations': metrics['iterations'],
            'converged': metrics['converged'],
            'solve_time': metrics.get('solve_time', 0.0)
        }


def main():
    """
    Main optimization workflow.
    """
    print("CONFIG + Smith FEA Integration Example")
    print("=" * 60)

    # Configuration
    SMITH_EXECUTABLE = "../smith/build/src/smith"
    LUA_TEMPLATE = "../smith/examples/contact_template.lua"

    # Check if Smith is available
    if not Path(SMITH_EXECUTABLE).exists():
        print("ERROR: Smith executable not found!")
        print(f"Expected location: {SMITH_EXECUTABLE}")
        print("\nSmith must be built first. See README.md for build instructions.")
        print("Note: Smith cannot be built in Claude Code web environment due to network restrictions.")
        return

    # Initialize optimizer wrapper
    try:
        smith_opt = SmithFEAOptimizer(
            smith_executable=SMITH_EXECUTABLE,
            lua_template=LUA_TEMPLATE,
            work_dir="./smith_optimization"
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Define parameter bounds
    # [solver_tolerance, penalty_parameter, max_iterations]
    bounds = np.array([
        [1e-10, 1e-4],  # solver_tolerance (log scale)
        [1e4, 1e8],     # penalty_parameter (log scale)
        [100, 5000]     # max_iterations
    ])

    # Configure CONFIG optimizer
    config = CONFIGConfig(
        bounds=bounds,
        constraint_configs={
            'convergence': {'tolerance': 1e-8},
            'iteration': {'max_iterations': 2000}
        },
        delta=0.1,
        n_init=20,
        n_max=100,
        acquisition_method="discrete",
        seed=42,
        verbose=True
    )

    # Create CONFIG controller
    controller = CONFIGController(config, smith_opt.objective_function)

    # Run optimization
    print("\nStarting CONFIG optimization...")
    results = controller.optimize()

    # Display results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    if results['best_x'] is not None:
        print(f"\nBest Parameters Found:")
        print(f"  Solver Tolerance:   {results['best_x'][0]:.2e}")
        print(f"  Penalty Parameter:  {results['best_x'][1]:.2e}")
        if len(results['best_x']) > 2:
            print(f"  Max Iterations:     {int(results['best_x'][2])}")
        print(f"\nBest Objective Value: {results['best_y']:.6f}")
    else:
        print("\nNo feasible solution found!")
        print("Consider:")
        print("  - Relaxing constraint tolerances")
        print("  - Expanding parameter bounds")
        print("  - Increasing optimization budget")

    print(f"\nTotal Evaluations: {results['n_evaluations']}")
    print(f"Feasible Points:   {results['n_feasible']}")

    # Violation statistics
    stats = results['violation_statistics']
    print(f"\nViolation Statistics:")
    print(f"  Cumulative Violations: {stats['cumulative_violation']:.4f}")
    print(f"  Violation Rate:        {stats['violation_rate']:.2%}")
    print(f"  Status:                {results['violation_bound_check']['status']}")


if __name__ == "__main__":
    main()
