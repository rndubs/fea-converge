# Claude Code Context

## Project Overview

This repository provides **four distinct Bayesian optimization methods** for resolving contact convergence failures in finite element simulations using the LLNL Tribol contact library and Smith/Serac solver framework.

**Project Scope:** Four parallel implementations (3 production-ready, 1 in progress)

## Implementation Status Summary

| **Method** | **Status** | **Code** | **Tests** | **Examples** | **Docs** |
|------------|-----------|----------|-----------|--------------|----------|
| **CONFIG** | ✅ Production | 2.3K LOC | 22 passing | 2 scripts | Complete |
| **GP-Classification** | ✅ Production | 2.8K LOC | 32 passing | 1 script | Complete |
| **SHEBO** | ✅ Production | 3.4K LOC | Full suite | 1 script | Complete |
| **FR-BO** | ⚠️ Near-Production | 4.1K LOC | 58 tests (5 files) | 2 scripts | Complete |

**Total:** 12,600+ lines of implementation code

---

## Method 1: CONFIG - Constrained Efficient Global Optimization ⭐

**Location:** `config/`

**Implementation Status:**
- ✅ 2,310 lines of production code (19 source files)
- ✅ 22 comprehensive tests (4 test files, all passing)
- ✅ 2 complete examples (basic + Smith integration)
- ✅ Professional logging and error handling
- ✅ Named constants (no magic numbers)
- ✅ Visualization utilities
- ✅ Edge case handling
- ✅ Complete documentation (README, IMPLEMENTATION_PLAN, CONTRIBUTING)

**Key Features:**
- Rigorous theoretical guarantees (sublinear regret, bounded violations)
- GP-based surrogate modeling with RBF kernel
- LCB (Lower Confidence Bound) acquisition function
- Multi-phase optimization strategy
- Violation monitoring with theoretical bounds
- Best for safety-critical applications requiring formal guarantees

**Algorithm:** Optimistic auxiliary problem with LCB-based acquisition over optimistic feasible set

**Documentation:** [config/README.md](config/README.md)

---

## Method 2: GP-Classification - Gaussian Process Classification ⭐

**Location:** `gp-classification/`

**Implementation Status:**
- ✅ 2,813 lines of production code (8 source files)
- ✅ 32 comprehensive tests (5 test files, 100% pass rate)
- ✅ 1 complete example (basic_optimization.py)
- ✅ Variational inference for probabilistic predictions
- ✅ Robust BoTorch compatibility with fallback mechanisms
- ✅ Three-phase optimization strategy
- ✅ Complete documentation (README, STATUS, IMPLEMENTATION_PLAN, CONTRIBUTING)

**Key Features:**
- Variational GP classifier for binary convergence outcomes
- Direct probability modeling: P(converged | parameters)
- Dual-model architecture (classifier + regression fallback)
- Three-phase exploration: Sobol → Entropy → CEI
- Interpretable risk scores and decision boundaries
- Automatic hyperparameter optimization
- Best for binary outcomes and risk-aware decision making

**Algorithm:** Three-phase strategy combining space-filling, entropy maximization, and constrained expected improvement

**Documentation:** [gp-classification/README.md](gp-classification/README.md)
**Status Report:** [gp-classification/STATUS.md](gp-classification/STATUS.md)

---

## Method 3: SHEBO - Surrogate Optimization with Hidden Constraints ⭐

**Location:** `shebo/`

**Implementation Status:**
- ✅ 3,351 lines of production code (15 source files)
- ✅ Full test suite (6 test files, comprehensive coverage)
- ✅ 1 complete example (simple_optimization.py)
- ✅ Ensemble neural network surrogates
- ✅ Automatic constraint discovery
- ✅ GPU/CPU support with device auto-selection
- ✅ Checkpointing for crash recovery
- ✅ Complete documentation (README, DEVELOPMENT, CRITICAL_REVIEW, CONTRIBUTING, FIXES)

**Key Features:**
- Ensemble neural network surrogates (5 models) with uncertainty quantification
- Automatic constraint discovery using anomaly detection
- Adaptive acquisition function balancing multiple objectives
- GPU acceleration for large-scale problems
- Checkpoint system for long-running optimizations
- Comprehensive visualization tools
- Best for complex multi-constraint problems and large datasets

**Algorithm:** Ensemble-based surrogate modeling with automatic constraint discovery via clustering

**Documentation:** [shebo/README.md](shebo/README.md)
**Developer Guide:** [shebo/DEVELOPMENT.md](shebo/DEVELOPMENT.md)

---

## Method 4: FR-BO - Failure-Robust Bayesian Optimization ⚠️

**Location:** `fr_bo/` (note: underscore, not hyphen)

**Implementation Status:**
- ✅ 4,062 lines of production code (13 source files)
- ✅ 58 comprehensive tests (5 test files: test_gp_models.py, test_acquisition.py, test_parameters.py, test_integration.py, test_optimizer.py)
- ✅ 2 complete examples (basic_optimization.py, smith_integration_example.py)
- ✅ Complete documentation (README.md, CONTRIBUTING.md, TEST_FIXES.md, TEST_VALIDATION_SUMMARY.md)
- ✅ Version 0.2.0 with proper pyproject.toml
- ⚠️ 7 modules need additional test coverage (visualization, early_termination, risk_scoring, multi_task, synthetic_data, utils, objective)
- ⚠️ Uses print() instead of logging module (35 occurrences)
- ⚠️ Missing dependencies in pyproject.toml (plotly, scikit-learn)
- ⚠️ Parameter decoding bug in optimizer.py:379 needs fixing

**Implemented Components:**
- `optimizer.py` - Main FR-BO optimizer with three-phase strategy
- `gp_models.py` - Dual GP system (objective + failure prediction)
- `acquisition.py` - Failure-aware acquisition functions
- `multi_task.py` - Multi-task GP for transfer learning (needs tests)
- `early_termination.py` - Trajectory monitoring (needs tests)
- `risk_scoring.py` - Pre-simulation risk assessment (needs tests)
- `visualization.py` - Comprehensive plotting tools (needs tests)
- `synthetic_data.py` - Test data generation (needs tests)
- `simulator.py` - Simulation executor wrapper with timeout
- `objective.py`, `parameters.py`, `utils.py` - Supporting modules

**Key Features:**
- Dual GP system: convergence objective + failure prediction
- Failure-aware acquisition functions (EI × success probability)
- Early termination monitoring of simulation trajectories
- Multi-task transfer learning across geometries
- Risk scoring for pre-simulation assessment
- Three-phase optimization: Sobol → Exploitation → Exploration
- Comprehensive type hints and error handling
- Best for rapid convergence when limited violations acceptable

**Algorithm:** Dual Gaussian processes with failure-aware acquisition: α(x) = EI(x) × (1 - P_failure(x))

**Status:** ~70% production-ready. Remaining work:
- [ ] Replace print() statements with logging module (35 occurrences)
- [ ] Fix parameter decoding bug (optimizer.py:379)
- [ ] Add plotly and scikit-learn to dependencies
- [ ] Add tests for 7 untested modules (target: 25-30 new tests)
- [ ] Replace 52 magic numbers with named constants
- [ ] Improve edge case handling (all failures, timeouts, empty history)
- [ ] Add 2-3 advanced examples
- **Estimated Effort:** 1-2 weeks to production parity

**Documentation:** [fr_bo/README.md](fr_bo/README.md)
**Test Status:** [fr_bo/TEST_VALIDATION_SUMMARY.md](fr_bo/TEST_VALIDATION_SUMMARY.md)

**Note:** The `/future_methods/fr-bo/` directory contains old planning documents and should be disregarded in favor of the actual implementation in `/fr_bo/`.

---

## Smith Build System

The `./smith` directory contains submodules and build scripts for the Smith/Serac finite element solver framework.

### Build Prerequisites (Installed)

The following dependencies are installed and verified:
- **CMake 3.28.3** - Build system generator
- **Python 3.11.14** - Required for uberenv build scripts
- **GCC 13.3.0** - C/C++ compiler
- **gfortran 13.3.0** - Fortran compiler
- **MPICH 4.2.0** - MPI implementation
- **Clang 18.1.3** - Alternative compiler (optional)

### Build Status

✅ **Prerequisites installed and verified**

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

---

## Repository Structure

```
fea-converge/
├── config/                     # ✅ CONFIG (production-ready)
│   ├── src/config_optimizer/
│   ├── tests/                  # 22 passing tests
│   ├── examples/               # 2 complete examples
│   └── README.md
│
├── gp-classification/          # ✅ GP-Classification (production-ready)
│   ├── src/gp_classification/
│   ├── tests/                  # 32 passing tests
│   ├── examples/               # 1 complete example
│   ├── README.md
│   └── STATUS.md
│
├── shebo/                      # ✅ SHEBO (production-ready)
│   ├── src/shebo/
│   ├── tests/                  # Full test suite
│   ├── examples/               # 1 example script
│   ├── README.md
│   └── DEVELOPMENT.md
│
├── fr_bo/                      # ⚠️ FR-BO (partial implementation)
│   ├── optimizer.py
│   ├── gp_models.py
│   ├── acquisition.py
│   └── ... (13 files total)   # Tests/examples/docs needed
│
├── future_methods/             # 📚 OLD PLANNING DOCS (mostly superseded)
│   ├── fr-bo/                  # Old FR-BO plan (use /fr_bo/ instead)
│   ├── gp-classification/      # Old plan (implemented in /gp-classification/)
│   └── shebo/                  # Old plan (implemented in /shebo/)
│
├── smith/                      # Smith FEA submodule
├── README.md                   # Main project README (4-method overview)
├── PROJECT_SCOPE.md            # Project scoping and evolution
├── RESEARCH.md                 # Technical documentation (all methods)
├── CRITICAL_REVIEW.md          # Code review across implementations
└── smith_ml_optimizer.py       # Basic Ax/BoTorch wrapper (separate utility)
```

---

## Quick Links

### Production-Ready Methods

- **CONFIG:** [config/README.md](config/README.md) - Safety-critical, theoretical guarantees
- **GP-Classification:** [gp-classification/README.md](gp-classification/README.md) - Binary outcomes, risk-aware
- **SHEBO:** [shebo/README.md](shebo/README.md) - Complex constraints, ensemble modeling

### Work in Progress

- **FR-BO:** [fr_bo/](fr_bo/) - Implementation exists, needs tests/docs/examples

### Cross-Method Resources

- **[README.md](README.md)** - Main project README with method comparison
- **[RESEARCH.md](RESEARCH.md)** - Comprehensive technical documentation for all methods
- **[CRITICAL_REVIEW.md](CRITICAL_REVIEW.md)** - Code quality analysis
- **[PROJECT_SCOPE.md](PROJECT_SCOPE.md)** - Project evolution and rationale

---

## Method Selection Guide

**Choose your method based on your needs:**

| **Your Need** | **Recommended Method** |
|---------------|------------------------|
| Formal safety guarantees | CONFIG |
| Binary convergence modeling | GP-Classification |
| Multiple unknown constraints | SHEBO |
| Learning from failures | FR-BO (when complete) |
| GPU acceleration | SHEBO |
| Interpretable risk scores | GP-Classification |
| Theoretical convergence proofs | CONFIG |
| Large-scale problems | SHEBO |

---

## Development Priorities

### Immediate (FR-BO Completion)

1. **Replace print() with logging** - 35 occurrences need professional logging
2. **Fix parameter decoding bug** - optimizer.py:379 uses random sampling instead of optimized values
3. **Add missing dependencies** - plotly and scikit-learn to pyproject.toml
4. **Expand test coverage** - Add 25-30 tests for 7 untested modules
5. **Replace magic numbers** - 52 hardcoded values need named constants
6. **Improve edge case handling** - All failures, timeouts, empty history scenarios

**Estimated Effort:** 1-2 weeks

### Future Enhancements (All Methods)

- Transfer learning across geometries
- Multi-fidelity optimization
- Batch/parallel evaluation
- Real-time monitoring dashboards
- Hybrid methods combining multiple approaches
- AutoML for automatic method selection

---

## Important Notes

1. **FR-BO Status Update:** The `fr_bo/` directory is ~70% production-ready with 58 tests, 2 examples, and complete documentation. Main gaps: logging module (uses print()), parameter decoding bug, missing dependencies (plotly/sklearn), and test coverage for 7 modules.

2. **All Methods Have Issues:** Code review identified 50+ issues in CONFIG, 60+ in SHEBO, 40+ in GP-Classification, and 60+ in FR-BO. Most are quality/testing gaps, not functionality bugs. See detailed reviews for each method.

3. **Old Planning Docs:** The `future_methods/` directory contains old implementation plans. For GP-Classification and SHEBO, use the actual implementations in `gp-classification/` and `shebo/`. For FR-BO, use the implementation in `fr_bo/`, not the old plan in `future_methods/fr-bo/`.

4. **Method Maturity:** CONFIG, GP-Classification, and SHEBO are functional but need quality improvements (tests, magic numbers, error handling). FR-BO is close to parity but needs critical bug fixes first.

5. **Smith Integration:** All methods are designed for Smith FEA integration, but Smith cannot be built in web environments due to network restrictions. Use local development for actual Smith integration testing.
