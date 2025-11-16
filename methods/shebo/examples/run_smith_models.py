"""
SHEBO Method - Smith Models Workflow

This script demonstrates how to use SHEBO (Surrogate Optimization with Hidden
Constraints) with Smith contact models for constraint discovery.

SHEBO is designed for complex problems with hidden constraints that emerge
during simulation, making it well-suited for contact convergence.

Workflow:
1. Build and run all Smith models
2. Collect convergence and constraint violation data
3. Discover hidden constraints
4. Train ensemble neural network surrogates
"""

import sys
from pathlib import Path

# Add parent directory to path to import smith_runner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from smith_runner import SmithModelRunner, BuildMode
import json
import numpy as np
from typing import Dict, List


def discover_hidden_constraints(results: Dict) -> Dict:
    """
    Discover hidden constraints from Smith model results.

    SHEBO's constraint discovery engine identifies implicit constraints
    that aren't known a priori.

    Args:
        results: Dictionary of model_name -> SmithModelResult

    Returns:
        Constraint discovery analysis
    """
    analysis = {
        'total_models': len(results),
        'successful': 0,
        'failed': 0,
        'discovered_constraints': [],
        'constraint_violations': [],
        'safe_region': [],
        'unsafe_region': [],
        'constraint_metrics': {
            'iteration_threshold': None,
            'residual_threshold': None,
            'timestep_completion': None
        }
    }

    # Collect metrics from successful runs
    successful_iterations = []
    successful_residuals = []
    successful_timesteps = []

    for model_name, result in results.items():
        if result.success and result.converged:
            analysis['successful'] += 1
            analysis['safe_region'].append(model_name)

            if result.iterations > 0:
                successful_iterations.append(result.iterations)
            if result.final_residual:
                successful_residuals.append(result.final_residual)
            if result.timesteps_completed > 0:
                successful_timesteps.append(result.timesteps_completed)
        else:
            analysis['failed'] += 1
            analysis['unsafe_region'].append(model_name)

            # Record constraint violation
            violation = {
                'model': model_name,
                'type': 'unknown',
                'details': {}
            }

            if not result.success:
                violation['type'] = 'execution_failure'
                violation['details']['error'] = result.error_message
            elif not result.converged:
                violation['type'] = 'convergence_failure'
                violation['details']['iterations'] = result.iterations
                violation['details']['final_residual'] = result.final_residual

            analysis['constraint_violations'].append(violation)

    # Discover constraints from successful runs
    if successful_iterations:
        # Constraint: iterations must be below threshold
        max_iter = np.max(successful_iterations)
        mean_iter = np.mean(successful_iterations)
        std_iter = np.std(successful_iterations)

        analysis['constraint_metrics']['iteration_threshold'] = {
            'max': int(max_iter),
            'mean': float(mean_iter),
            'std': float(std_iter),
            'suggested_limit': int(mean_iter + 2 * std_iter)
        }

        analysis['discovered_constraints'].append({
            'type': 'iteration_limit',
            'description': f'Iterations should be < {int(mean_iter + 2 * std_iter)}',
            'confidence': 'high' if len(successful_iterations) >= 3 else 'low'
        })

    if successful_residuals:
        # Constraint: residual must be below threshold
        min_res = np.min(successful_residuals)
        max_res = np.max(successful_residuals)
        mean_res = np.mean(successful_residuals)

        analysis['constraint_metrics']['residual_threshold'] = {
            'min': float(min_res),
            'max': float(max_res),
            'mean': float(mean_res),
            'suggested_tolerance': float(max_res * 10)  # 10x safety margin
        }

        analysis['discovered_constraints'].append({
            'type': 'residual_tolerance',
            'description': f'Final residual should be < {max_res * 10:.2e}',
            'confidence': 'high' if len(successful_residuals) >= 3 else 'low'
        })

    if successful_timesteps:
        # Constraint: must complete minimum timesteps
        min_steps = np.min(successful_timesteps)

        analysis['constraint_metrics']['timestep_completion'] = {
            'min_required': int(min_steps),
            'max_observed': int(np.max(successful_timesteps)),
            'mean': float(np.mean(successful_timesteps))
        }

        analysis['discovered_constraints'].append({
            'type': 'timestep_completion',
            'description': f'Must complete at least {int(min_steps)} timesteps',
            'confidence': 'high' if len(successful_timesteps) >= 3 else 'low'
        })

    return analysis


def print_shebo_analysis(results: Dict, analysis: Dict):
    """Print SHEBO specific analysis."""
    print("\n" + "=" * 70)
    print("SHEBO CONSTRAINT DISCOVERY ANALYSIS")
    print("=" * 70)

    print(f"\nOverall Statistics:")
    print(f"  Total Models: {analysis['total_models']}")
    print(f"  Successful (Safe Region): {analysis['successful']}")
    print(f"  Failed (Unsafe Region): {analysis['failed']}")

    print(f"\nSafe Region Models:")
    for model in analysis['safe_region']:
        print(f"  ✓ {model}")

    if analysis['unsafe_region']:
        print(f"\nUnsafe Region Models:")
        for model in analysis['unsafe_region']:
            print(f"  ✗ {model}")

    print("\n" + "=" * 70)
    print("DISCOVERED HIDDEN CONSTRAINTS")
    print("=" * 70)

    if analysis['discovered_constraints']:
        print(f"\nFound {len(analysis['discovered_constraints'])} potential constraints:\n")

        for i, constraint in enumerate(analysis['discovered_constraints'], 1):
            print(f"{i}. {constraint['type'].upper()}")
            print(f"   Description: {constraint['description']}")
            print(f"   Confidence: {constraint['confidence']}")
            print()

        # Print detailed metrics
        if analysis['constraint_metrics']['iteration_threshold']:
            print("Iteration Constraint Details:")
            metrics = analysis['constraint_metrics']['iteration_threshold']
            print(f"  Max observed: {metrics['max']}")
            print(f"  Mean: {metrics['mean']:.1f}")
            print(f"  Std: {metrics['std']:.1f}")
            print(f"  Suggested limit: {metrics['suggested_limit']}")
            print()

        if analysis['constraint_metrics']['residual_threshold']:
            print("Residual Constraint Details:")
            metrics = analysis['constraint_metrics']['residual_threshold']
            print(f"  Min: {metrics['min']:.2e}")
            print(f"  Max: {metrics['max']:.2e}")
            print(f"  Mean: {metrics['mean']:.2e}")
            print(f"  Suggested tolerance: {metrics['suggested_tolerance']:.2e}")
            print()

        if analysis['constraint_metrics']['timestep_completion']:
            print("Timestep Constraint Details:")
            metrics = analysis['constraint_metrics']['timestep_completion']
            print(f"  Min required: {metrics['min_required']}")
            print(f"  Max observed: {metrics['max_observed']}")
            print(f"  Mean: {metrics['mean']:.1f}")
            print()
    else:
        print("\nNo constraints could be discovered (insufficient successful runs)")

    print("=" * 70)
    print("CONSTRAINT VIOLATIONS")
    print("=" * 70)

    if analysis['constraint_violations']:
        for violation in analysis['constraint_violations']:
            print(f"\n✗ {violation['model']}")
            print(f"  Type: {violation['type']}")
            for key, value in violation['details'].items():
                print(f"  {key}: {value}")
    else:
        print("\nNo constraint violations detected (all models successful)")

    print("\n" + "=" * 70)
    print("SHEBO SUITABILITY ASSESSMENT")
    print("=" * 70)

    if len(analysis['discovered_constraints']) >= 2:
        print(f"""
✓ EXCELLENT for SHEBO

Discovered {len(analysis['discovered_constraints'])} hidden constraints.
SHEBO is well-suited for this problem because:

1. Multiple implicit constraints exist
2. Constraint discovery can guide optimization
3. Ensemble neural networks can model complex interactions
4. Feature normalization handles different parameter scales

SHEBO will:
- Train ensemble of neural networks as surrogates
- Use discovered constraints to define safe regions
- Balance exploration vs exploitation with constraint awareness
- Provide checkpointing for crash recovery
        """)
    elif len(analysis['discovered_constraints']) == 1:
        print(f"""
✓ MODERATE for SHEBO

Discovered {len(analysis['discovered_constraints'])} constraint. SHEBO can
still be beneficial, though simpler constraint-aware BO methods might
also work well.

Consider SHEBO if:
- Problem dimensionality is high (>10D)
- Parameter scales vary widely (e.g., 1e-8 to 1e10)
- Constraint boundaries are complex/nonlinear
        """)
    else:
        print(f"""
⚠ LIMITED for SHEBO

No hidden constraints discovered. This could mean:
1. Insufficient successful runs to establish patterns
2. Problem has simple/explicit constraints
3. All tested configurations are in safe region

SHEBO can still be used but may not provide advantages over
standard constraint-aware Bayesian optimization.

Recommendation: Collect more data or consider other methods.
        """)


def main():
    """Main workflow for running Smith models with SHEBO."""
    print("=" * 70)
    print("SHEBO Method - Smith Models Workflow")
    print("=" * 70)

    # Initialize Smith model runner
    print("\n1. Initializing Smith model runner...")
    runner = SmithModelRunner(verbose=True)

    print(f"\nRepository root: {runner.repo_root}")
    print(f"Available models: {len(runner.list_models())}")

    # Run all models
    print("\n2. Running all Smith models...")
    print("   Discovering hidden constraints for SHEBO...\n")

    results = runner.run_all_models(clean=False, timeout=600)

    # Analyze for constraint discovery
    print("\n3. Discovering hidden constraints...")
    analysis = discover_hidden_constraints(results)

    # Print SHEBO analysis
    print_shebo_analysis(results, analysis)

    # Save results
    output_dir = Path(__file__).parent / "smith_results"
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "all_models_results.json"
    analysis_file = output_dir / "shebo_constraint_discovery.json"

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
    print("NEXT STEPS FOR SHEBO OPTIMIZATION")
    print("=" * 70)
    print("""
The constraint discovery provides inputs for SHEBO optimization. Next steps:

1. Use discovered constraints to define constraint functions

2. Train ensemble of neural network surrogates:
   - Each network trains independently for diversity
   - Feature normalization for parameter scale handling
   - Model both objective and constraints

3. Run SHEBO optimization with:
   - Adaptive acquisition balancing exploration/exploitation
   - Constraint-aware sampling
   - Checkpointing for recovery

4. Analyze results to understand:
   - Safe parameter regions
   - Constraint boundary structure
   - Optimal configurations

See SHEBO documentation:
  - examples/simple_optimization.py
  - DEVELOPMENT.md
  - README.md
    """)


if __name__ == "__main__":
    main()
