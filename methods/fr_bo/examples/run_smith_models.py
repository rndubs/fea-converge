"""
FR-BO Method - Smith Models Workflow

This script demonstrates how to use the FR-BO optimizer with Smith contact
models to understand contact convergence and failure patterns.

FR-BO is specifically designed for scenarios with frequent simulation failures,
making it well-suited for contact convergence problems.

Workflow:
1. Build and run all Smith models
2. Collect convergence and failure metrics
3. Analyze failure patterns
4. Use results to train failure-aware GP models
"""

import sys
from pathlib import Path

# Add parent directory to path to import smith_runner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from smith_runner import SmithModelRunner, BuildMode
import json
import numpy as np
from typing import Dict


def analyze_failure_patterns(results: Dict) -> Dict:
    """
    Analyze failure patterns from Smith model results.

    This is particularly relevant for FR-BO which models both
    convergence and failure probability.

    Args:
        results: Dictionary of model_name -> SmithModelResult

    Returns:
        Failure analysis summary
    """
    analysis = {
        'total_models': len(results),
        'successful': 0,
        'failed': 0,
        'converged': 0,
        'non_converged': 0,
        'failure_rate': 0.0,
        'convergence_rate': 0.0,
        'failure_modes': {
            'timeout': 0,
            'build_error': 0,
            'runtime_error': 0,
            'non_convergence': 0,
            'unknown': 0
        },
        'models_by_category': {
            'successful_converged': [],
            'successful_non_converged': [],
            'failed': []
        },
        'metrics_by_model': {}
    }

    for model_name, result in results.items():
        # Categorize results
        if result.success:
            analysis['successful'] += 1
            if result.converged:
                analysis['converged'] += 1
                analysis['models_by_category']['successful_converged'].append(
                    model_name
                )
            else:
                analysis['non_converged'] += 1
                analysis['models_by_category']['successful_non_converged'].append(
                    model_name
                )
        else:
            analysis['failed'] += 1
            analysis['models_by_category']['failed'].append(model_name)

            # Classify failure mode
            if result.error_message:
                if 'timeout' in result.error_message.lower():
                    analysis['failure_modes']['timeout'] += 1
                elif 'build' in result.error_message.lower():
                    analysis['failure_modes']['build_error'] += 1
                elif 'not found' in result.error_message.lower():
                    analysis['failure_modes']['runtime_error'] += 1
                else:
                    analysis['failure_modes']['unknown'] += 1
            elif not result.converged:
                analysis['failure_modes']['non_convergence'] += 1

        # Store per-model metrics for FR-BO training
        analysis['metrics_by_model'][model_name] = {
            'success': result.success,
            'converged': result.converged,
            'failed': not result.success,
            'iterations': result.iterations,
            'timesteps': result.timesteps_completed,
            'final_residual': result.final_residual,
            'solve_time': result.solve_time
        }

    # Compute rates
    if analysis['total_models'] > 0:
        analysis['failure_rate'] = (
            analysis['failed'] / analysis['total_models']
        )
        analysis['convergence_rate'] = (
            analysis['converged'] / analysis['total_models']
        )

    return analysis


def print_frbo_analysis(results: Dict, analysis: Dict):
    """Print FR-BO specific analysis."""
    print("\n" + "=" * 70)
    print("FR-BO FAILURE-AWARE ANALYSIS")
    print("=" * 70)

    print(f"\nOverall Statistics:")
    print(f"  Total Models: {analysis['total_models']}")
    print(f"  Successful Runs: {analysis['successful']}")
    print(f"  Failed Runs: {analysis['failed']}")
    print(f"  Failure Rate: {analysis['failure_rate']:.1%}")
    print(f"  Converged: {analysis['converged']}")
    print(f"  Non-Converged: {analysis['non_converged']}")
    print(f"  Convergence Rate: {analysis['convergence_rate']:.1%}")

    print(f"\nFailure Mode Breakdown:")
    for mode, count in analysis['failure_modes'].items():
        if count > 0:
            print(f"  {mode}: {count}")

    print(f"\nModel Categories:")
    print(f"  Successful + Converged: {len(analysis['models_by_category']['successful_converged'])}")
    for model in analysis['models_by_category']['successful_converged']:
        print(f"    ✓ {model}")

    if analysis['models_by_category']['successful_non_converged']:
        print(f"  Successful + Non-Converged: {len(analysis['models_by_category']['successful_non_converged'])}")
        for model in analysis['models_by_category']['successful_non_converged']:
            print(f"    ⚠ {model}")

    if analysis['models_by_category']['failed']:
        print(f"  Failed: {len(analysis['models_by_category']['failed'])}")
        for model in analysis['models_by_category']['failed']:
            print(f"    ✗ {model}")

    print("\n" + "=" * 70)
    print("FR-BO RELEVANCE ASSESSMENT")
    print("=" * 70)

    if analysis['failure_rate'] > 0.1:
        print(f"""
✓ HIGH RELEVANCE for FR-BO

With a failure rate of {analysis['failure_rate']:.1%}, this problem is well-suited
for FR-BO's failure-aware optimization approach. FR-BO will:

1. Train a dual GP model (convergence + failure probability)
2. Use failure-aware acquisition: EI(x) × (1 - P_failure(x))
3. Avoid parameter regions likely to cause failures
4. Balance exploration vs exploitation accounting for failure risk
        """)
    elif analysis['failure_rate'] > 0.05:
        print(f"""
✓ MODERATE RELEVANCE for FR-BO

With a failure rate of {analysis['failure_rate']:.1%}, FR-BO's failure modeling
may provide some benefit, though standard BO might also work well.
        """)
    else:
        print(f"""
⚠ LOW RELEVANCE for FR-BO

With a failure rate of {analysis['failure_rate']:.1%}, standard Bayesian
optimization may be sufficient. FR-BO is most beneficial when failure
rates exceed 10%.

However, FR-BO can still be used and won't hurt performance.
        """)


def main():
    """Main workflow for running Smith models with FR-BO method."""
    print("=" * 70)
    print("FR-BO Method - Smith Models Workflow")
    print("=" * 70)

    # Initialize Smith model runner
    print("\n1. Initializing Smith model runner...")
    runner = SmithModelRunner(verbose=True)

    print(f"\nRepository root: {runner.repo_root}")
    print(f"Available models: {len(runner.list_models())}")

    # Run all models
    print("\n2. Running all Smith models...")
    print("   Collecting convergence and failure data for FR-BO...\n")

    results = runner.run_all_models(clean=False, timeout=600)

    # Analyze with FR-BO lens
    print("\n3. Analyzing failure patterns...")
    analysis = analyze_failure_patterns(results)

    # Print FR-BO specific analysis
    print_frbo_analysis(results, analysis)

    # Save results
    output_dir = Path(__file__).parent / "smith_results"
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "all_models_results.json"
    analysis_file = output_dir / "frbo_failure_analysis.json"

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
    print("NEXT STEPS FOR FR-BO OPTIMIZATION")
    print("=" * 70)
    print("""
The failure analysis provides data for FR-BO optimization. Next steps:

1. Use failure data to train dual GP models:
   - Convergence GP: models successful convergence behavior
   - Failure GP: models probability of simulation failure

2. Set up FR-BO optimizer with failure-aware acquisition function

3. Run optimization to find robust parameters that:
   - Maximize convergence quality
   - Minimize failure probability

4. Validate results on held-out test cases

See FR-BO documentation:
  - examples/basic_optimization.py
  - examples/smith_integration_example.py
  - README.md
    """)


if __name__ == "__main__":
    main()
