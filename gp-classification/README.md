# GP Classification with Sign-Based Convergence

## Status: 🚧 NOT YET IMPLEMENTED

This method is currently in the **planning phase**. Only the implementation plan exists at this time.

## Overview

GP Classification models the binary convergence outcome directly as a probabilistic prediction, enabling constrained optimization where feasibility itself becomes a probabilistic constraint integrated into acquisition functions.

**Key Innovation**: Directly predicts P(converged|x) using a GP classifier with non-Gaussian posterior approximations, enabling risk-aware decision making and soft constraint handling.

**Best For**: Interpretable probabilistic reasoning, risk-aware parameter suggestions, and rapid boundary learning through entropy-driven acquisition.

## Current Documentation

- ✅ [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Detailed 10-week implementation plan
- ❌ Source code - Not yet developed
- ❌ Tests - Not yet developed
- ❌ Examples - Not yet developed

## Implementation Phases

The implementation plan outlines:
1. **Phase 1-2:** Foundation and GP classification model
2. **Phase 3:** Dual model architecture (objective + classifier)
3. **Phase 4:** Constrained acquisition functions (CEI, entropy-based)
4. **Phase 5:** Three-phase exploration strategy
5. **Phase 6-7:** Use case features and visualization
6. **Phase 8:** Testing and deployment

**Estimated Implementation Time:** 10 weeks

## How to Use (When Implemented)

```python
# Future usage example
from gp_classification import GPClassificationOptimizer

optimizer = GPClassificationOptimizer(
    bounds=bounds,
    objective_function=my_simulator,
    convergence_threshold=1e-8,
    n_init=20
)

results = optimizer.optimize()
# Get convergence probability for new point
prob = optimizer.predict_convergence_probability(x_new)
```

## Alternative: Use CONFIG

While GP Classification is not yet implemented, consider using the **CONFIG** optimizer which is fully functional:

```bash
cd ../config/
# See config/README.md for usage
```

## Key Features (Planned)

- **Probabilistic convergence prediction** with confidence intervals
- **Entropy-based active learning** for boundary refinement
- **Real-time feasibility assessment** (<100ms latency)
- **Interpretable decision boundaries**
- **Risk-aware parameter recommendations**

## Contributing

Interested in implementing GP Classification? See the implementation plan and:
1. Review variational GP classifier theory
2. Set up development environment (BoTorch/GPyTorch)
3. Follow the phased implementation checklist
4. Submit pull requests with tests

## References

- Implementation plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- Project research: [../RESEARCH.md](../RESEARCH.md)
- GPyTorch classification: https://docs.gpytorch.ai/en/latest/examples/01_Exact_GPs/GP_Regression_Classification.html

---

**Last Updated:** 2025-11-12
**Maintainers:** fea-converge contributors
