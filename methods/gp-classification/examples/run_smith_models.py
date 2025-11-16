"""
GP-Classification Method - Smith Models Workflow

This script demonstrates how to use GP Classification with Smith contact
models for binary convergence prediction.

GP-Classification is ideal for interpreting convergence boundaries and
understanding which parameter regions lead to success vs failure.

Workflow:
1. Build and run all Smith models
2. Collect binary convergence labels
3. Analyze decision boundaries
4. Train variational GP classifier
"""

import sys
from pathlib import Path

# Add parent directory to path to import smith_runner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from smith_runner import SmithModelRunner, BuildMode
import json
import numpy as np
from typing import Dict


def analyze_binary_convergence(results: Dict) -> Dict:
    """
    Analyze binary convergence patterns for GP Classification.

    Args:
        results: Dictionary of model_name -> SmithModelResult

    Returns:
        Binary classification analysis
    """
    analysis = {
        'total_models': len(results),
        'converged_class': [],  # Class 1: Converged
        'failed_class': [],     # Class 0: Failed/Non-converged
        'class_balance': 0.0,
        'class_distribution': {
            'converged': 0,
            'failed': 0
        },
        'convergence_characteristics': {
            'converged_models': {},
            'failed_models': {}
        },
        'decision_boundary_insights': []
    }

    for model_name, result in results.items():
        # Binary classification: converged (1) vs non-converged (0)
        if result.converged and result.success:
            analysis['converged_class'].append(model_name)
            analysis['class_distribution']['converged'] += 1

            # Store characteristics of converged runs
            analysis['convergence_characteristics']['converged_models'][model_name] = {
                'iterations': result.iterations,
                'timesteps': result.timesteps_completed,
                'final_residual': result.final_residual,
                'solve_time': result.solve_time
            }
        else:
            analysis['failed_class'].append(model_name)
            analysis['class_distribution']['failed'] += 1

            # Store characteristics of failed runs
            analysis['convergence_characteristics']['failed_models'][model_name] = {
                'iterations': result.iterations,
                'timesteps': result.timesteps_completed,
                'error': result.error_message,
                'converged': result.converged,
                'success': result.success
            }

    # Compute class balance
    if analysis['total_models'] > 0:
        analysis['class_balance'] = (
            analysis['class_distribution']['converged'] /
            analysis['total_models']
        )

    # Assess if problem is suitable for GP Classification
    if 0.2 <= analysis['class_balance'] <= 0.8:
        analysis['decision_boundary_insights'].append(
            "Good class balance for GP Classification"
        )
    elif analysis['class_balance'] < 0.2:
        analysis['decision_boundary_insights'].append(
            "Imbalanced: Few converged cases - consider oversampling"
        )
    else:
        analysis['decision_boundary_insights'].append(
            "Imbalanced: Few failed cases - consider undersampling"
        )

    return analysis


def print_gp_classification_analysis(results: Dict, analysis: Dict):
    """Print GP Classification specific analysis."""
    print("\n" + "=" * 70)
    print("GP CLASSIFICATION ANALYSIS")
    print("=" * 70)

    print(f"\nBinary Classification Setup:")
    print(f"  Total Samples: {analysis['total_models']}")
    print(f"  Class 1 (Converged): {analysis['class_distribution']['converged']}")
    print(f"  Class 0 (Failed): {analysis['class_distribution']['failed']}")
    print(f"  Class Balance: {analysis['class_balance']:.2%}")

    print(f"\nConverged Models (Class 1):")
    for model in analysis['converged_class']:
        chars = analysis['convergence_characteristics']['converged_models'][model]
        print(f"  ✓ {model}")
        print(f"      Iterations: {chars['iterations']}, Timesteps: {chars['timesteps']}")
        if chars['final_residual']:
            print(f"      Final Residual: {chars['final_residual']:.2e}")

    print(f"\nFailed Models (Class 0):")
    for model in analysis['failed_class']:
        chars = analysis['convergence_characteristics']['failed_models'][model]
        print(f"  ✗ {model}")
        if chars['error']:
            print(f"      Error: {chars['error']}")
        elif not chars['converged']:
            print(f"      Non-converged (ran but didn't converge)")

    print("\n" + "=" * 70)
    print("DECISION BOUNDARY INSIGHTS")
    print("=" * 70)

    for insight in analysis['decision_boundary_insights']:
        print(f"  • {insight}")

    print("\n" + "=" * 70)
    print("GP CLASSIFICATION SUITABILITY")
    print("=" * 70)

    if 0.2 <= analysis['class_balance'] <= 0.8:
        print(f"""
✓ EXCELLENT for GP Classification

Class balance of {analysis['class_balance']:.1%} is ideal for learning
convergence decision boundaries. GP Classification will:

1. Train variational GP classifier on binary convergence labels
2. Learn interpretable decision boundary
3. Provide uncertainty estimates for predictions
4. Use three-phase optimization strategy:
   - Phase 1: Entropy-based boundary discovery
   - Phase 2: Boundary refinement
   - Phase 3: Constrained exploitation with CEI
        """)
    else:
        print(f"""
⚠ USABLE but with caveats

Class balance of {analysis['class_balance']:.1%} suggests class imbalance.
Recommendations:

1. Apply SMOTE or other resampling techniques
2. Use class weights in GP training
3. Focus on boundary discovery in imbalanced region
4. Consider collecting more data in minority class region

GP Classification can still provide valuable insights into convergence
boundary structure.
        """)


def main():
    """Main workflow for running Smith models with GP Classification."""
    print("=" * 70)
    print("GP Classification Method - Smith Models Workflow")
    print("=" * 70)

    # Initialize Smith model runner
    print("\n1. Initializing Smith model runner...")
    runner = SmithModelRunner(verbose=True)

    print(f"\nRepository root: {runner.repo_root}")
    print(f"Available models: {len(runner.list_models())}")

    # Run all models
    print("\n2. Running all Smith models...")
    print("   Collecting binary convergence labels for GP Classification...\n")

    results = runner.run_all_models(clean=False, timeout=600)

    # Analyze for GP Classification
    print("\n3. Analyzing binary convergence patterns...")
    analysis = analyze_binary_convergence(results)

    # Print GP Classification analysis
    print_gp_classification_analysis(results, analysis)

    # Save results
    output_dir = Path(__file__).parent / "smith_results"
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "all_models_results.json"
    analysis_file = output_dir / "gp_classification_analysis.json"

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
    print("NEXT STEPS FOR GP CLASSIFICATION")
    print("=" * 70)
    print("""
The binary convergence data is ready for GP Classification. Next steps:

1. Train Variational GP Classifier on convergence labels

2. Visualize decision boundary:
   - Identify parameter regions that separate converged/failed
   - Plot uncertainty in boundary location
   - Highlight high-uncertainty regions for exploration

3. Run three-phase optimization:
   - Phase 1 (1-20): Entropy-based boundary discovery
   - Phase 2 (21-50): Boundary refinement
   - Phase 3 (51+): Constrained exploitation with CEI

4. Interpret learned boundary for physical insights

See GP Classification documentation:
  - examples/basic_optimization.py
  - src/gp_classification/use_cases.py
  - README.md
    """)


if __name__ == "__main__":
    main()
