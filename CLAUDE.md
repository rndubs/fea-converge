# Claude Code Context

## Project Overview

This repository is developing machine learning optimization approaches for resolving contact convergence failures in finite element simulations using the LLNL Tribol contact library and Smith/Serac solver framework.

**Implementation Status:** One method fully implemented (CONFIG), three methods in planning phase (FR-BO, GP-Classification, SHEBO).

## GP Method Directories

### [config/](config/) - Constrained Efficient Global Optimization (CONFIG) ✅ **IMPLEMENTED**
Provides rigorous theoretical guarantees with sublinear regret and bounded cumulative violations. Uses optimistic feasible sets and lower confidence bound acquisition for provable convergence. Best for safety-critical applications requiring formal guarantees.

**Status:** Fully functional with ~2000 lines of code, comprehensive tests, and documentation. Ready for use.

### [fr-bo/](fr-bo/) - Failure-Robust Bayesian Optimization 🚧 **PLANNED**
Treats simulation failures as informative constraints rather than nuisances. Learns failure boundaries through dual Gaussian processes (one for objective, one for failure probability) with failure-aware acquisition functions. Best for rapid convergence when limited violations are acceptable.

**Status:** Detailed implementation plan available (IMPLEMENTATION_PLAN.md). Code not yet developed.

### [gp-classification/](gp-classification/) - GP Classification with Sign-Based Convergence 🚧 **PLANNED**
Models binary convergence outcomes directly as probabilistic predictions using variational GP classifiers. Enables constrained optimization with entropy-based acquisitions for boundary refinement. Best for interpretable, risk-aware parameter suggestions with confidence intervals.

**Status:** Detailed implementation plan available (IMPLEMENTATION_PLAN.md). Code not yet developed.

### [shebo/](shebo/) - Surrogate Optimization with Hidden Constraints (SHEBO) 🚧 **PLANNED**
Combines ensemble surrogate modeling with constraint discovery to identify unknown failure modes. Uses neural network ensembles for epistemic uncertainty quantification and active learning for boundary mapping. Best for complex multi-constraint problems with large datasets.

**Status:** Detailed implementation plan available (IMPLEMENTATION_PLAN.md). Code not yet developed.

## Smith Build System

The `./smith` directory contains submodules and build scripts for the Smith/Serac finite element solver framework.

**Important Build Limitation**: The Smith build system **cannot be built in Claude Code for the Web environments** due to network access restrictions. Building Smith requires:
- Network access to download Spack dependencies
- Access to external package repositories
- Ability to fetch TPL (Third-Party Library) sources

If you need to build Smith, use a local development environment or a containerized environment with network access enabled. The `build_smith.sh` script handles the build process but will fail in sandboxed/network-restricted environments.

## Additional Resources

- [RESEARCH.md](RESEARCH.md) - Comprehensive technical documentation of all four ML systems
- [README.md](README.md) - Project setup and usage instructions
- [CRITICAL_REVIEW.md](CRITICAL_REVIEW.md) - Comprehensive code review and identified issues
- `config/` - The only fully implemented optimizer (CONFIG method)
- `smith_ml_optimizer.py` - Basic Ax/BoTorch wrapper (separate from the four methods)
