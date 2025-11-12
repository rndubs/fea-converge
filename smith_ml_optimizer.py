#!/usr/bin/env python3
"""
Smith FEA Parameter Optimization with Ax/Botorch

This script demonstrates how to optimize Smith FEA solver parameters
using Bayesian optimization with the Ax platform and BoTorch.
"""

import subprocess
import json
import os
from pathlib import Path
from typing import Dict, Any


class SmithOptimizer:
    """Wrapper for optimizing Smith FEA parameters."""
    
    def __init__(self, smith_executable: str, template_lua: str, work_dir: str = "./smith_runs"):
        """
        Initialize the Smith optimizer.
        
        Args:
            smith_executable: Path to Smith executable
            template_lua: Path to Lua input template
            work_dir: Working directory for Smith runs
        """
        self.smith_executable = Path(smith_executable)
        self.template_lua = Path(template_lua)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.smith_executable.exists():
            raise FileNotFoundError(f"Smith executable not found: {smith_executable}")
        if not self.template_lua.exists():
            raise FileNotFoundError(f"Template file not found: {template_lua}")
    
    def create_input_deck(self, parameters: Dict[str, float], output_file: Path) -> None:
        """
        Create a Lua input deck with specified parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
            output_file: Path to output Lua file
        """
        # Read template
        with open(self.template_lua, 'r') as f:
            template = f.read()
        
        # Replace parameters in template
        # This assumes parameters are marked as {{ param_name }} in the template
        lua_content = template
        for param_name, param_value in parameters.items():
            placeholder = f"{{{{ {param_name} }}}}"
            lua_content = lua_content.replace(placeholder, str(param_value))
        
        # Write output
        with open(output_file, 'w') as f:
            f.write(lua_content)
    
    def run_smith(self, input_deck: Path, num_procs: int = 1) -> Dict[str, Any]:
        """
        Run Smith with the given input deck.
        
        Args:
            input_deck: Path to Lua input file
            num_procs: Number of MPI processes
            
        Returns:
            Dictionary containing performance metrics
        """
        # Run Smith
        cmd = ['mpirun', '-np', str(num_procs), str(self.smith_executable), str(input_deck)]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=self.work_dir
            )
            
            # Parse output
            metrics = self.parse_output(result.stdout, result.stderr)
            metrics['success'] = (result.returncode == 0)
            metrics['returncode'] = result.returncode
            
            return metrics
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'timeout',
                'iterations': float('inf'),
                'solve_time': float('inf')
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'iterations': float('inf'),
                'solve_time': float('inf')
            }
    
    def parse_output(self, stdout: str, stderr: str) -> Dict[str, float]:
        """
        Parse Smith output to extract performance metrics.
        
        Args:
            stdout: Standard output from Smith
            stderr: Standard error from Smith
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Example parsing (adjust based on actual Smith output format)
        for line in stdout.split('\n'):
            # Look for solver iterations
            if 'iterations' in line.lower():
                try:
                    metrics['iterations'] = float(line.split()[-1])
                except:
                    pass
            
            # Look for solve time
            if 'solve time' in line.lower():
                try:
                    metrics['solve_time'] = float(line.split()[-1])
                except:
                    pass
            
            # Look for convergence
            if 'converged' in line.lower():
                metrics['converged'] = True
            
            # Look for residual
            if 'residual' in line.lower():
                try:
                    metrics['final_residual'] = float(line.split()[-1])
                except:
                    pass
        
        return metrics
    
    def evaluate(self, parameters: Dict[str, float]) -> float:
        """
        Evaluate a parameter set and return objective value.
        
        Args:
            parameters: Dictionary of parameters to evaluate
            
        Returns:
            Objective value (lower is better)
        """
        # Create unique run directory
        run_id = hash(frozenset(parameters.items())) & 0x7FFFFFFF
        run_dir = self.work_dir / f"run_{run_id}"
        run_dir.mkdir(exist_ok=True)
        
        # Create input deck
        input_deck = run_dir / "input.lua"
        self.create_input_deck(parameters, input_deck)
        
        # Run simulation
        metrics = self.run_smith(input_deck)
        
        # Compute objective (example: minimize solve time * iterations)
        if not metrics.get('success', False):
            return float('inf')
        
        solve_time = metrics.get('solve_time', float('inf'))
        iterations = metrics.get('iterations', float('inf'))
        
        # Combined objective
        objective = solve_time + 0.1 * iterations
        
        # Save results
        with open(run_dir / "results.json", 'w') as f:
            json.dump({'parameters': parameters, 'metrics': metrics, 'objective': objective}, f, indent=2)
        
        return objective


def optimize_with_ax(optimizer: SmithOptimizer, parameter_space: Dict[str, tuple], n_trials: int = 20):
    """
    Optimize Smith parameters using Ax.
    
    Args:
        optimizer: SmithOptimizer instance
        parameter_space: Dictionary of {param_name: (min, max)} ranges
        n_trials: Number of optimization trials
        
    Returns:
        Best parameters and optimization results
    """
    try:
        from ax import optimize
    except ImportError:
        print("ERROR: Ax is not installed. Install with: pip install ax-platform")
        return None
    
    # Create parameter list for Ax
    parameters = [
        {
            "name": name,
            "type": "range",
            "bounds": bounds,
            "value_type": "float",
            "log_scale": True if "tolerance" in name.lower() else False,
        }
        for name, bounds in parameter_space.items()
    ]
    
    # Define evaluation function
    def evaluation_function(parameterization):
        return optimizer.evaluate(parameterization)
    
    # Run optimization
    print(f"Starting Bayesian optimization with {n_trials} trials...")
    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=evaluation_function,
        objective_name="solve_cost",
        minimize=True,
        total_trials=n_trials,
    )
    
    print("\nOptimization complete!")
    print(f"Best parameters: {best_parameters}")
    print(f"Best objective value: {values[0]['solve_cost']}")
    
    return best_parameters, values, experiment, model


if __name__ == "__main__":
    # Example usage
    print("Smith FEA Parameter Optimizer")
    print("=" * 50)
    
    # Configuration
    SMITH_EXECUTABLE = "./smith/build/src/smith"
    TEMPLATE_LUA = "./smith/examples/template.lua"  # Create this template
    
    # Check if files exist
    if not os.path.exists(SMITH_EXECUTABLE):
        print(f"ERROR: Smith executable not found at {SMITH_EXECUTABLE}")
        print("Please build Smith first using build_smith.sh")
        exit(1)
    
    if not os.path.exists(TEMPLATE_LUA):
        print(f"WARNING: Template file not found at {TEMPLATE_LUA}")
        print("Please create a Lua template with parameter placeholders")
        print("Example: solver_tolerance = {{ solver_tolerance }}")
        exit(1)
    
    # Initialize optimizer
    optimizer = SmithOptimizer(
        smith_executable=SMITH_EXECUTABLE,
        template_lua=TEMPLATE_LUA,
        work_dir="./optimization_runs"
    )
    
    # Define parameter space
    parameter_space = {
        "solver_tolerance": (1e-10, 1e-4),
        "penalty_parameter": (1e4, 1e8),
        "max_iterations": (100, 5000),
        "penalty_coefficient": (0.5, 2.0),
    }
    
    # Run optimization
    results = optimize_with_ax(
        optimizer=optimizer,
        parameter_space=parameter_space,
        n_trials=30
    )
    
    if results:
        best_params, values, experiment, model = results
        print("\nRecommended Smith configuration:")
        for param, value in best_params.items():
            print(f"  {param}: {value:.6e}")
