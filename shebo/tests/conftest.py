"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import torch


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def sample_bounds():
    """Standard parameter bounds for testing."""
    return np.array([
        [1e6, 1e10],    # penalty parameter
        [1e-8, 1e-4],   # tolerance
        [0.0, 1.0],     # normalized param 1
        [0.0, 1.0]      # normalized param 2
    ])
