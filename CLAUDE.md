# Claude Code Context

## Project Overview

This repository implements four distinct machine learning optimization approaches for resolving contact convergence failures in finite element simulations using the LLNL Tribol contact library and Smith/Serac solver framework.

## GP Method Directories

### [fr-bo/](fr-bo/) - Failure-Robust Bayesian Optimization
Treats simulation failures as informative constraints rather than nuisances. Learns failure boundaries through dual Gaussian processes (one for objective, one for failure probability) with failure-aware acquisition functions. Best for rapid convergence when limited violations are acceptable.

### [gp-classification/](gp-classification/) - GP Classification with Sign-Based Convergence
Models binary convergence outcomes directly as probabilistic predictions using variational GP classifiers. Enables constrained optimization with entropy-based acquisitions for boundary refinement. Best for interpretable, risk-aware parameter suggestions with confidence intervals.

### [config/](config/) - Constrained Efficient Global Optimization (CONFIG)
Provides rigorous theoretical guarantees with sublinear regret and bounded cumulative violations. Uses optimistic feasible sets and lower confidence bound acquisition for provable convergence. Best for safety-critical applications requiring formal guarantees.

### [shebo/](shebo/) - Surrogate Optimization with Hidden Constraints (SHEBO)
Combines ensemble surrogate modeling with constraint discovery to identify unknown failure modes. Uses neural network ensembles for epistemic uncertainty quantification and active learning for boundary mapping. Best for complex multi-constraint problems with large datasets.

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

- [RESEARCH.md](RESEARCH.md) - Comprehensive technical documentation of all four ML systems
- [README.md](README.md) - Project setup and usage instructions
- `smith_ml_optimizer.py` - Main optimizer implementation
