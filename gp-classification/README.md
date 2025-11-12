# GP Classification for FEA Contact Convergence Optimization

This package implements Gaussian Process Classification for optimizing finite element contact simulation parameters using variational inference and constrained Bayesian optimization.

## Overview

**GP Classification** models binary convergence outcomes directly as probabilistic predictions using variational GP classifiers. This enables constrained optimization where feasibility itself becomes a probabilistic constraint integrated into acquisition functions.

### Key Features

- **Variational GP Classification**: Direct prediction of P(converged|x) with non-Gaussian posterior approximations
- **Dual Model Architecture**: Combined convergence classifier and objective regression
- **Three-Phase Exploration Strategy**:
  - Phase 1 (Iterations 1-20): Entropy-based boundary discovery
  - Phase 2 (Iterations 21-50): Boundary refinement near P(converge)=0.5
  - Phase 3 (Iterations 51+): Constrained exploitation with CEI
- **Interpretable Outputs**: Probability estimates with confidence intervals
- **Natural Constraint Handling**: Multiplicative CEI formulation
- **Active Learning**: Entropy-driven acquisition for efficient boundary mapping

### Best For

- Interpretable probabilistic reasoning
- Risk-aware parameter suggestions
- Rapid boundary learning through entropy-driven acquisition
- Problems where understanding failure modes is important

## Quick Start

### Installation

\`\`\`bash
cd gp-classification
uv sync --extra dev
\`\`\`

### Basic Usage

\`\`\`python
from gp_classification import GPClassificationOptimizer
from gp_classification.mock_solver import MockSmithSolver, get_default_parameter_bounds

# Set up parameter bounds
parameter_bounds = get_default_parameter_bounds()

# Create simulator
solver = MockSmithSolver(random_seed=42, difficulty="medium")

def simulator(params):
    return solver.simulate(params)

# Create optimizer
optimizer = GPClassificationOptimizer(
    parameter_bounds=parameter_bounds,
    simulator=simulator,
    n_initial_samples=20,
    verbose=True,
)

# Run optimization
best_params = optimizer.optimize(n_iterations=60)
\`\`\`

## Running Tests

\`\`\`bash
uv run pytest
\`\`\`

## Running Examples

\`\`\`bash
cd examples
uv run python basic_optimization.py
\`\`\`

## Documentation

- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)**: Detailed technical specification
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Development setup and guidelines

## Related Methods

This package is part of a suite of four ML optimization approaches. See the main repository for other methods.
