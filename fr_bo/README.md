# FR-BO: Failure-Robust Bayesian Optimization

## ⚠️ Status: Partial Implementation

**Version:** 0.1.0
**Status:** Core implementation exists, but **NOT production-ready**

This directory contains a partial implementation of FR-BO (Failure-Robust Bayesian Optimization) for FEA convergence optimization. The core algorithms are implemented but lack tests, examples, and documentation needed for production use.

---

## Current State

### ✅ Implemented (4,062 lines of code)

| **Component** | **File** | **Lines** | **Description** |
|---------------|----------|-----------|-----------------|
| **Main Optimizer** | `optimizer.py` | 15,266 | Core FR-BO optimization loop |
| **Dual GP Models** | `gp_models.py` | 10,569 | Convergence + failure prediction GPs |
| **Acquisition Functions** | `acquisition.py` | 8,525 | Failure-aware acquisition: α(x) = EI(x) × (1 - P_failure(x)) |
| **Multi-Task GP** | `multi_task.py` | 12,004 | Transfer learning across geometries |
| **Early Termination** | `early_termination.py` | 11,218 | Trajectory monitoring and failure prediction |
| **Risk Scoring** | `risk_scoring.py` | 12,912 | Pre-simulation risk assessment |
| **Visualization** | `visualization.py` | 15,602 | Plotting tools for optimization progress |
| **Synthetic Data** | `synthetic_data.py` | 11,438 | Generate synthetic training data |
| **Simulator Interface** | `simulator.py` | 12,023 | Simulation executor wrapper |
| **Objective Functions** | `objective.py` | 6,108 | Objective function definitions |
| **Parameter Space** | `parameters.py` | 7,639 | Parameter encoding and bounds |
| **Utilities** | `utils.py` | 6,803 | Helper functions |
| **Package Init** | `__init__.py` | - | Module definition (version 0.1.0) |

**Total:** 13 files, 4,062 lines of implementation code

### ❌ Missing (Blocking Production Use)

- **No tests** - 0 test files
- **No examples** - 0 example scripts
- **No documentation** - No user guide, API reference, or tutorials
- **No CONTRIBUTING.md** - No developer guidelines
- **Unvalidated** - Not tested against real FEA problems

---

## Algorithm Overview

FR-BO uses a **dual Gaussian process system** to simultaneously model:
1. **Convergence objective:** f_convergence(x) → minimize
2. **Failure probability:** P_failure(x) → predict crashes

### Key Innovation: Failure-Aware Acquisition

Standard Bayesian optimization uses:
```
α(x) = EI(x)  (Expected Improvement)
```

FR-BO uses:
```
α(x) = EI_convergence(x) × (1 - P_failure(x))
```

This **penalizes high-risk regions** while still exploring for good solutions.

### Additional Features

- **Early Termination:** Monitor simulation trajectories, terminate failing runs early
- **Risk Scoring:** Pre-assess risk without running full simulation
- **Multi-Task Learning:** Transfer knowledge across similar problems
- **Synthetic Data:** Bootstrap from artificial failures when real data is scarce

---

## Comparison to Other Methods

| **Feature** | **FR-BO** | **CONFIG** | **GP-Classification** | **SHEBO** |
|-------------|-----------|------------|----------------------|-----------|
| **Status** | ⚠️ Partial | ✅ Production | ✅ Production | ✅ Production |
| **Tests** | ❌ None | 22 passing | 32 passing | Full suite |
| **Examples** | ❌ None | 2 scripts | 1 script | 1 script |
| **Docs** | ❌ Missing | Complete | Complete | Complete |
| **Core Code** | ✅ 4.1K LOC | ✅ 2.3K LOC | ✅ 2.8K LOC | ✅ 3.4K LOC |
| **Best For** | Failure tolerance | Safety-critical | Binary outcomes | Complex constraints |

**Recommendation:** Use **CONFIG**, **GP-Classification**, or **SHEBO** for production work. Use FR-BO only for research/development until testing and documentation are complete.

---

## Roadmap to Production

### Phase 1: Testing (1 week)

**Goal:** Add comprehensive test suite (target: 20+ tests)

**Tasks:**
- [ ] Unit tests for dual GP models (`gp_models.py`)
- [ ] Unit tests for acquisition functions (`acquisition.py`)
- [ ] Unit tests for early termination (`early_termination.py`)
- [ ] Unit tests for risk scoring (`risk_scoring.py`)
- [ ] Integration tests for full optimization loop
- [ ] Edge case tests (NaN, failures, no feasible points)
- [ ] Multi-dimensional problem tests (1D to 10D)
- [ ] Comparison tests vs. standard BO

**Deliverable:** `tests/` directory with pytest suite

### Phase 2: Examples (3-4 days)

**Goal:** Create working examples demonstrating FR-BO usage

**Tasks:**
- [ ] Basic optimization example (simple test function)
- [ ] Smith FEA integration example
- [ ] Comparison with CONFIG/GP-Classification/SHEBO
- [ ] Early termination demonstration
- [ ] Risk scoring demonstration

**Deliverable:** `examples/` directory with 2-3 complete scripts

### Phase 3: Documentation (3-4 days)

**Goal:** Write user and developer documentation

**Tasks:**
- [ ] Expand this README with quick start guide
- [ ] Add API reference documentation (docstrings)
- [ ] Write CONTRIBUTING.md for developers
- [ ] Create tutorial notebooks (optional)
- [ ] Document known limitations and caveats

**Deliverable:** Complete README.md, CONTRIBUTING.md, docstrings

### Phase 4: Validation (2-3 days)

**Goal:** Validate FR-BO on real problems

**Tasks:**
- [ ] Test on standard BO benchmark problems
- [ ] Validate against Smith FEA convergence problems
- [ ] Performance benchmarking vs. other methods
- [ ] Edge case verification
- [ ] Failure mode analysis

**Deliverable:** Validation report, performance comparison

### Phase 5: Production Release (1 day)

**Goal:** Package for production use

**Tasks:**
- [ ] Set version to 1.0.0
- [ ] Add logging and error handling
- [ ] Create release notes
- [ ] Update main repo README/CLAUDE.md/PROJECT_SCOPE.md
- [ ] Tag release in git

**Deliverable:** FR-BO v1.0.0 production release

---

## Estimated Effort

**Total Time to Production:** 2-3 weeks

| **Phase** | **Effort** | **Priority** |
|-----------|------------|--------------|
| Testing | 1 week | 🔴 Critical |
| Examples | 3-4 days | 🔴 Critical |
| Documentation | 3-4 days | 🟡 High |
| Validation | 2-3 days | 🟡 High |
| Production Release | 1 day | 🟢 Medium |

---

## Current Usage (For Developers Only)

⚠️ **Warning:** This code is **untested** and **undocumented**. Use at your own risk.

### Installation

```bash
cd /path/to/fea-converge/fr_bo
uv sync  # If pyproject.toml exists
# or
pip install -e .  # If setup.py exists
```

### Basic Usage (Untested)

```python
from fr_bo.optimizer import FRBOOptimizer
from fr_bo.gp_models import DualGPModels
from fr_bo.acquisition import FailureAwareAcquisition

# Define your simulator (that can fail/crash)
def my_simulator(x):
    # Your FEA simulation
    try:
        result = run_fea_simulation(x)
        return {
            'objective': result.convergence_metric,
            'failed': False
        }
    except SimulationFailure:
        return {
            'objective': None,
            'failed': True
        }

# Configure FR-BO (untested API)
optimizer = FRBOOptimizer(
    bounds=[[0, 1], [0, 1]],
    n_init=20,
    n_max=100,
    seed=42
)

# Run optimization
results = optimizer.optimize(my_simulator)
```

**Note:** This API is **speculative** based on the code structure. Actual usage may differ.

---

## Why FR-BO?

### Problem: Simulation Failures

In FEA optimization, simulations often **fail or crash** due to:
- Non-physical parameter combinations
- Solver divergence
- Mesh distortion
- Memory/numerical errors

Standard Bayesian optimization treats failures as:
- ❌ Missing data (wastes information)
- ❌ Infeasible constraints (ignores failure patterns)
- ❌ Poor parameter choices (abandons promising regions)

### Solution: Learn from Failures

FR-BO treats failures as **informative signals**:
- ✅ Model failure probability: P(fail | x)
- ✅ Avoid high-risk regions proactively
- ✅ Early termination of likely failures
- ✅ Transfer learning from past failures

### When to Use FR-BO

**✅ Good fit:**
- Simulations frequently fail (>10% failure rate)
- Failures are expensive (wasted compute time)
- Failure regions are structured (not random)
- You want rapid convergence despite failures

**❌ Poor fit:**
- Simulations never fail (use CONFIG/SHEBO instead)
- Failures are completely random (no pattern to learn)
- You need formal safety guarantees (use CONFIG)
- You need production-ready code NOW (use CONFIG/GP-Classification/SHEBO)

---

## Key Differences from Other Methods

### vs. CONFIG
- **CONFIG:** Formal guarantees, bounded violations, conservative
- **FR-BO:** Learns from failures, rapid convergence, more aggressive

### vs. GP-Classification
- **GP-Classification:** Binary pass/fail classification only
- **FR-BO:** Dual GPs for both objective and failure, early termination

### vs. SHEBO
- **SHEBO:** Constraint discovery, ensemble surrogates, deterministic
- **FR-BO:** Explicit failure modeling, probabilistic, stochastic

---

## Contributing

**Want to help complete FR-BO?**

Priority contributions:
1. **Write tests** - Most critical need
2. **Create examples** - Show how to use it
3. **Write docs** - Make it understandable
4. **Validate** - Test on real problems
5. **Fix bugs** - Improve code quality

See the [Roadmap to Production](#roadmap-to-production) for specific tasks.

---

## References

### FR-BO Research

- Letham et al., "Constrained Bayesian Optimization with Noisy Experiments" (2019)
- McLeod et al., "Optimization Under Unknown Constraints" (2018)
- Gramacy et al., "Modeling an Augmented Lagrangian for Blackbox Constrained Optimization" (2016)

### Related Methods in This Repo

- **[CONFIG](../config/README.md)** - Constrained optimization with guarantees
- **[GP-Classification](../gp-classification/README.md)** - Binary convergence modeling
- **[SHEBO](../shebo/README.md)** - Ensemble surrogates with constraint discovery

---

## License

Part of the fea-converge project. See repository root for license information.

---

## Contact

For questions about FR-BO implementation:
- Check the [main README](../README.md)
- Review [PROJECT_SCOPE.md](../PROJECT_SCOPE.md)
- See [RESEARCH.md](../RESEARCH.md) for technical details

**Production-ready alternatives:**
- Use **CONFIG** for safety-critical applications
- Use **GP-Classification** for binary convergence modeling
- Use **SHEBO** for complex constraint problems

---

**Status:** Partial Implementation (v0.1.0)
**Next Milestone:** Add test suite (Phase 1)
**Production Ready:** Estimated 2-3 weeks
**Last Updated:** 2025-11-12
