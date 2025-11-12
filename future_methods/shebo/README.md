# Surrogate Optimization with Hidden Constraints (SHEBO)

## Status: 🚧 NOT YET IMPLEMENTED

This method is currently in the **planning phase**. Only the implementation plan exists at this time.

## Overview

SHEBO combines surrogate modeling with constraint discovery, using ensemble approaches for robust feasibility prediction and adaptive sampling to map unknown convergence boundaries. This method excels when constraints are completely unknown a priori and when building reusable surrogate models across geometry families.

**Key Innovation**: Multiple surrogates (neural networks or GPs) for objectives and discovered constraints, using ensemble disagreement to quantify epistemic uncertainty and active learning to reveal hidden failure modes.

**Best For**: Production systems with many simulations, complex multi-constraint problems, transfer learning across geometries, and discovering unexpected failure modes.

## Current Documentation

- ✅ [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Detailed 14-week implementation plan
- ❌ Source code - Not yet developed
- ❌ Tests - Not yet developed
- ❌ Examples - Not yet developed

## Implementation Phases

The implementation plan outlines:
1. **Phase 1-2:** Foundation and neural network ensemble architecture
2. **Phase 3:** Surrogate manager with multiple models
3. **Phase 4:** Constraint discovery module
4. **Phase 5:** Adaptive acquisition function
5. **Phase 6:** Main workflow orchestration
6. **Phase 7:** Transfer learning and multi-fidelity
7. **Phase 8-9:** Use case features and visualization
8. **Phase 10:** Testing and production deployment

**Estimated Implementation Time:** 14 weeks

## How to Use (When Implemented)

```python
# Future usage example
from shebo import SHEBOOptimizer

optimizer = SHEBOOptimizer(
    bounds=bounds,
    objective_function=my_simulator,
    ensemble_size=5,
    enable_constraint_discovery=True
)

results = optimizer.optimize()

# View discovered constraints
print(f"Discovered {len(results['discovered_constraints'])} constraints")

# Use surrogate for fast prediction
pred = optimizer.predict(x_new)  # <10ms inference
```

## Alternative: Use CONFIG

While SHEBO is not yet implemented, consider using the **CONFIG** optimizer which is fully functional:

```bash
cd ../config/
# See config/README.md for usage
```

## Key Features (Planned)

- **Neural network ensembles** for uncertainty quantification
- **Automatic constraint discovery** from simulation failures
- **Fast surrogate inference** (<10ms for real-time use)
- **Transfer learning** across geometry families
- **Multi-fidelity optimization** (cheap pre-screening + expensive refinement)
- **Production deployment** with model serving

## Contributing

Interested in implementing SHEBO? See the implementation plan and:
1. Review ensemble learning and constraint discovery theory
2. Set up development environment (PyTorch/PyTorch Lightning)
3. Follow the phased implementation checklist
4. Implement database backend (PostgreSQL or MongoDB)
5. Submit pull requests with tests

## References

- Implementation plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- Project research: [../RESEARCH.md](../RESEARCH.md)
- Related papers: See implementation plan for constraint discovery references

---

**Last Updated:** 2025-11-12
**Maintainers:** fea-converge contributors
