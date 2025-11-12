# Claude Code Context

## Project Overview

This repository provides **CONFIG**, a production-ready Bayesian optimization algorithm for resolving contact convergence failures in finite element simulations using the LLNL Tribol contact library and Smith/Serac solver framework.

**Project Scope:** CONFIG optimizer (fully implemented) + research plans for future methods

## Primary Deliverable: CONFIG Optimizer

### [config/](config/) - Constrained Efficient Global Optimization (CONFIG) ⭐ **PRODUCTION READY**

The main deliverable of this project - a fully functional, tested, and documented optimization algorithm.

**Implementation Status:**
- ✅ 2000+ lines of production code
- ✅ 22 comprehensive tests (all passing)
- ✅ Professional logging and error handling
- ✅ Named constants (no magic numbers)
- ✅ Visualization utilities
- ✅ Smith FEA integration example
- ✅ Edge case handling
- ✅ Complete documentation

**Key Features:**
- Rigorous theoretical guarantees (sublinear regret, bounded violations)
- Multi-phase optimization strategy
- GP-based surrogate modeling
- LCB acquisition function
- Violation monitoring with theoretical bounds
- Best for safety-critical applications

**Location:** `config/` directory
**Documentation:** [config/README.md](config/README.md)
**Examples:** [config/examples/](config/examples/)

## Future Research Methods

### [future_methods/](future_methods/) - Research Plans 📚 **NOT IMPLEMENTED**

Three additional methods exist as comprehensive research plans for future development:

### [fr-bo/](future_methods/fr-bo/) - Failure-Robust Bayesian Optimization 🚧 **PLAN ONLY**
Treats simulation failures as informative constraints. Dual Gaussian processes with failure-aware acquisition functions.

**Status:** 20,000-word implementation plan. Code not developed.
**Implementation Effort:** ~10 weeks
**Best For:** Rapid convergence when limited violations acceptable

### [gp-classification/](future_methods/gp-classification/) - GP Classification 🚧 **PLAN ONLY**
Models binary convergence outcomes directly using variational GP classifiers.

**Status:** 26,000-word implementation plan. Code not developed.
**Implementation Effort:** ~10 weeks
**Best For:** Interpretable, risk-aware parameter suggestions

### [shebo/](future_methods/shebo/) - Surrogate Optimization with Hidden Constraints 🚧 **PLAN ONLY**
Ensemble surrogate modeling with automatic constraint discovery.

**Status:** 50,000-word implementation plan. Code not developed.
**Implementation Effort:** ~14 weeks
**Best For:** Complex multi-constraint problems, large datasets

**Total Future Work:** ~34 weeks (8+ months) to implement all three

## Smith Build System

The `./smith` directory contains submodules and build scripts for the Smith/Serac finite element solver framework.

### Build Prerequisites (Now Installed)

The following dependencies are now installed and verified:
- **CMake 3.28.3** - Build system generator
- **Python 3.11.14** - Required for uberenv build scripts
- **GCC 13.3.0** - C/C++ compiler
- **gfortran 13.3.0** - Fortran compiler
- **MPICH 4.2.0** - MPI implementation
- **Clang 18.1.3** - Alternative compiler (optional)

The smith submodule has been initialized with all recursive submodules.

### Build Status

✅ **Fixed Issues:**
- Installed missing MPI (MPICH) package
- Initialized smith git submodule and all nested submodules
- Verified all build prerequisites
- Enhanced build script with better error handling and diagnostics

⚠️ **Remaining Limitation:**

The Smith build system **cannot complete in Claude Code for the Web environments** due to network access restrictions. Building Smith requires:
- Network access to download Spack dependencies
- Access to external package repositories
- Ability to fetch TPL (Third-Party Library) sources

The `build_smith.sh` script will successfully check all prerequisites and begin the build process, but will fail when uberenv attempts to clone Spack repositories and download dependencies.

### Build Documentation

See `SMITH_BUILD_STATUS.md` for:
- Complete list of fixed issues
- Current system configuration
- Alternative build approaches for restricted environments
- Instructions for using Spack mirrors or pre-built TPLs

### Running the Build

To attempt the build (will fail at network access stage):
```bash
./build_smith.sh
```

To use an alternative Spack configuration:
```bash
SPACK_CONFIG=./smith/scripts/spack/configs/linux_ubuntu_24/spack.yaml ./build_smith.sh
```

## Additional Resources

- **[README.md](README.md)** - Main project README (CONFIG-focused)
- **[PROJECT_SCOPE.md](PROJECT_SCOPE.md)** - Project scoping decision and rationale
- **[CRITICAL_REVIEW.md](CRITICAL_REVIEW.md)** - Comprehensive code review and improvements
- **[RESEARCH.md](RESEARCH.md)** - Technical documentation of all methods (current + future)
- **[config/](config/)** - The fully implemented CONFIG optimizer ⭐
- **[future_methods/](future_methods/)** - Research plans for FR-BO, GP-Classification, SHEBO
- **[smith_ml_optimizer.py](smith_ml_optimizer.py)** - Basic Ax/BoTorch wrapper (separate utility)

## Quick Links

**Want to use the optimizer?** → [config/README.md](config/README.md)
**Want to understand the theory?** → [RESEARCH.md](RESEARCH.md)
**Want to implement future methods?** → [future_methods/](future_methods/)
**Want to review code quality?** → [CRITICAL_REVIEW.md](CRITICAL_REVIEW.md)
**Want to understand project scope?** → [PROJECT_SCOPE.md](PROJECT_SCOPE.md)


      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.
