# Failure-Robust Bayesian Optimization (FR-BO)

## Status: 🚧 NOT YET IMPLEMENTED

This method is currently in the **planning phase**. Only the implementation plan exists at this time.

## Overview

FR-BO treats simulation failures as informative constraints rather than nuisances, jointly modeling convergence feasibility and performance objectives through dual Gaussian processes with failure-aware acquisition functions.

**Key Innovation**: FREI(θ) = EI(θ) × (1 - P_fail(θ)) naturally balances optimization potential with feasibility likelihood.

**Expected Performance**: 3-8x convergence speedup versus standard BO in simulation-heavy applications.

**Best For**: Rapid convergence when limited violations are acceptable.

## Current Documentation

- ✅ [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Detailed 10-week implementation plan
- ❌ Source code - Not yet developed
- ❌ Tests - Not yet developed
- ❌ Examples - Not yet developed

## Implementation Phases

The implementation plan outlines:
1. **Phase 1-2:** Foundation and dual GP system
2. **Phase 3:** FR-BO acquisition function
3. **Phase 4:** Three-phase workflow
4. **Phase 5:** Early termination system
5. **Phase 6-7:** Use case features and visualization
6. **Phase 8:** Testing and validation

**Estimated Implementation Time:** 10 weeks

## How to Use (When Implemented)

```python
# Future usage example
from frbo import FRBOOptimizer

optimizer = FRBOOptimizer(
    bounds=bounds,
    objective_function=my_simulator,
    n_init=20,
    n_max=200
)

results = optimizer.optimize()
```

## Alternative: Use CONFIG

While FR-BO is not yet implemented, consider using the **CONFIG** optimizer which is fully functional:

```bash
cd ../config/
# See config/README.md for usage
```

## Contributing

Interested in implementing FR-BO? See the implementation plan and:
1. Review the theoretical foundation
2. Set up development environment (Ax/BoTorch)
3. Follow the phased implementation checklist
4. Submit pull requests with tests

## References

- Implementation plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- Project research: [../RESEARCH.md](../RESEARCH.md)
- Theoretical papers: See implementation plan references

---

**Last Updated:** 2025-11-12
**Maintainers:** fea-converge contributors
