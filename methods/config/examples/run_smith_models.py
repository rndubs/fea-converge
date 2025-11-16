"""
CONFIG Method - Smith Models Workflow

This script demonstrates how to use the CONFIG optimizer with Smith contact
models to understand contact convergence characteristics.

Workflow:
1. Build and run all Smith models
2. Collect convergence metrics
3. Analyze convergence patterns
4. Use results to inform CONFIG optimization strategy
"""

import sys
from pathlib import Path

# Add parent directory to path to import smith_runner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from smith_runner import SmithModelRunner, BuildMode
import json
import numpy as np
from typing import Dict, List


def analyze_convergence_metrics(results: Dict) -> Dict:
    """
    Analyze convergence metrics from Smith model results.

    Args:
        results: Dictionary of model_name -> SmithModelResult

    Returns:
        Analysis summary
    """
    analysis = {
        'total_models': len(results),
        'successful': 0,
        'converged': 0,
        'failed': 0,
        'avg_iterations': 0,
        'avg_timesteps': 0,
        'models_by_status': {
            'success': [],
            'failure': [],
            'converged': [],
            'non_converged': []
        },
        'iteration_stats': {},
        'timestep_stats': {}
    }

    iterations = []
    timesteps = []

    for model_name, result in results.items():
        # Count successes and convergence
        if result.success:
            analysis['successful'] += 1
            analysis['models_by_status']['success'].append(model_name)

        if result.converged:
            analysis['converged'] += 1
            analysis['models_by_status']['converged'].append(model_name)
        else:
            analysis['models_by_status']['non_converged'].append(model_name)

        if not result.success:
            analysis['failed'] += 1
            analysis['models_by_status']['failure'].append(model_name)

        # Collect metrics
        if result.iterations > 0:
            iterations.append(result.iterations)

        if result.timesteps_completed > 0:
            timesteps.append(result.timesteps_completed)

    # Compute statistics
    if iterations:
        analysis['avg_iterations'] = np.mean(iterations)
        analysis['iteration_stats'] = {
            'min': int(np.min(iterations)),
            'max': int(np.max(iterations)),
            'mean': float(np.mean(iterations)),
            'std': float(np.std(iterations))
        }

    if timesteps:
        analysis['avg_timesteps'] = np.mean(timesteps)
        analysis['timestep_stats'] = {
            'min': int(np.min(timesteps)),
            'max': int(np.max(timesteps)),
            'mean': float(np.mean(timesteps)),
            'std': float(np.std(timesteps))
        }

    return analysis


def print_results_summary(results: Dict, analysis: Dict):
    """Print summary of Smith model results."""
    print("\n" + "=" * 70)
    print("SMITH MODELS EXECUTION SUMMARY")
    print("=" * 70)

    print(f"\nTotal Models: {analysis['total_models']}")
    print(f"Successful Runs: {analysis['successful']}/{analysis['total_models']}")
    print(f"Converged Simulations: {analysis['converged']}/{analysis['total_models']}")
    print(f"Failed Runs: {analysis['failed']}/{analysis['total_models']}")

    if analysis['iteration_stats']:
        print(f"\nIteration Statistics:")
        print(f"  Min: {analysis['iteration_stats']['min']}")
        print(f"  Max: {analysis['iteration_stats']['max']}")
        print(f"  Mean: {analysis['iteration_stats']['mean']:.1f}")
        print(f"  Std: {analysis['iteration_stats']['std']:.1f}")

    if analysis['timestep_stats']:
        print(f"\nTimestep Statistics:")
        print(f"  Min: {analysis['timestep_stats']['min']}")
        print(f"  Max: {analysis['timestep_stats']['max']}")
        print(f"  Mean: {analysis['timestep_stats']['mean']:.1f}")
        print(f"  Std: {analysis['timestep_stats']['std']:.1f}")

    print("\n" + "=" * 70)
    print("INDIVIDUAL MODEL RESULTS")
    print("=" * 70)

    for model_name, result in results.items():
        status = "✓" if result.success else "✗"
        conv = "CONV" if result.converged else "FAIL"

        print(f"\n{status} {model_name} ({conv})")
        print(f"  Iterations: {result.iterations}")
        print(f"  Timesteps: {result.timesteps_completed}")
        if result.final_residual:
            print(f"  Final Residual: {result.final_residual:.2e}")
        if result.solve_time:
            print(f"  Solve Time: {result.solve_time:.2f} s")
        if result.error_message:
            print(f"  Error: {result.error_message}")
        if result.output_files:
            print(f"  Output Files: {len(result.output_files)}")


def main():
    """Main workflow for running Smith models with CONFIG method."""
    print("=" * 70)
    print("CONFIG Method - Smith Models Workflow")
    print("=" * 70)

    # Initialize Smith model runner
    print("\n1. Initializing Smith model runner...")
    runner = SmithModelRunner(verbose=True)

    print(f"\nRepository root: {runner.repo_root}")
    print(f"Models directory: {runner.models_dir}")
    print(f"Available models: {len(runner.list_models())}")

    # List available models
    print("\n2. Available Smith models:")
    for model in runner.list_models():
        exists = "✓" if runner.model_exists(model) else "✗"
        print(f"  {exists} {model}")

    # Run all models
    print("\n3. Running all Smith models...")
    print("   This may take several minutes depending on your system.\n")

    results = runner.run_all_models(clean=False, timeout=600)

    # Analyze results
    print("\n4. Analyzing convergence metrics...")
    analysis = analyze_convergence_metrics(results)

    # Print summary
    print_results_summary(results, analysis)

    # Save results to JSON
    output_dir = Path(__file__).parent / "smith_results"
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "all_models_results.json"
    analysis_file = output_dir / "convergence_analysis.json"

    # Convert results to dict format for JSON
    results_dict = {
        model_name: result.to_dict()
        for model_name, result in results.items()
    }

    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Results saved to:")
    print(f"  {results_file}")
    print(f"  {analysis_file}")
    print("=" * 70)

    # Print next steps
    print("\n" + "=" * 70)
    print("NEXT STEPS FOR CONFIG OPTIMIZATION")
    print("=" * 70)
    print("""
The Smith model results provide baseline convergence data. You can now:

1. Use these results to identify challenging contact scenarios
2. Extract parameter ranges from successful runs
3. Set up CONFIG optimizer to explore parameter space
4. Define constraints based on convergence requirements
5. Run optimization to find robust parameter settings

See the CONFIG documentation for optimization setup:
  - examples/basic_example.py
  - examples/smith_integration_example.py
  - README.md
    """)


if __name__ == "__main__":
    main()
