"""
Unit tests for beta schedule computation.
"""

import pytest
import numpy as np
from src.config_optimizer.utils.beta_schedule import (
    compute_beta,
    compute_beta_schedule,
    adaptive_beta_adjustment
)


def test_compute_beta_basic():
    """Test basic beta computation."""
    beta = compute_beta(10, delta=0.1)
    assert beta > 0
    assert isinstance(beta, float)


def test_compute_beta_monotonic():
    """Test that beta increases with iteration number."""
    betas = [compute_beta(i, delta=0.1) for i in range(1, 20)]
    for i in range(len(betas) - 1):
        assert betas[i+1] > betas[i], "Beta should be monotonically increasing"


def test_compute_beta_invalid():
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        compute_beta(0)  # n must be >= 1
    
    with pytest.raises(ValueError):
        compute_beta(10, delta=1.5)  # delta must be in (0, 1)


def test_compute_beta_schedule():
    """Test beta schedule for multiple iterations."""
    schedule = compute_beta_schedule(10, delta=0.1)
    assert len(schedule) == 10
    assert np.all(np.diff(schedule) > 0), "Schedule should be monotonic"


def test_adaptive_beta_adjustment():
    """Test adaptive beta adjustment for small feasible sets."""
    current_beta = 1.0
    
    # Small feasible set should increase beta
    adjusted = adaptive_beta_adjustment(current_beta, feasible_set_size=0.05)
    assert adjusted > current_beta
    
    # Large feasible set should keep beta unchanged
    adjusted = adaptive_beta_adjustment(current_beta, feasible_set_size=0.5)
    assert adjusted == current_beta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
