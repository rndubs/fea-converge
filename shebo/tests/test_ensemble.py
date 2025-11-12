"""Tests for ensemble models."""

import pytest
import torch
import numpy as np
from shebo.models.ensemble import ConvergenceEnsemble, PerformanceEnsemble


class TestConvergenceEnsemble:
    """Test convergence ensemble functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        torch.manual_seed(42)
        n_samples = 100
        n_features = 4

        X = torch.randn(n_samples, n_features)
        y = torch.randint(0, 2, (n_samples, 1)).float()

        return X, y

    def test_ensemble_initialization(self):
        """Test ensemble can be initialized."""
        ensemble = ConvergenceEnsemble(input_dim=4, n_networks=3)

        assert ensemble is not None
        assert len(ensemble.networks) == 3

    def test_ensemble_forward(self, sample_data):
        """Test forward pass produces correct output."""
        X, _ = sample_data
        ensemble = ConvergenceEnsemble(input_dim=4, n_networks=3)

        predictions = ensemble(X[:10])

        assert len(predictions) == 3  # 3 networks
        assert all(p.shape == (10, 1) for p in predictions)

    def test_predict_with_uncertainty(self, sample_data):
        """Test uncertainty quantification."""
        X, _ = sample_data
        ensemble = ConvergenceEnsemble(input_dim=4, n_networks=5)

        result = ensemble.predict_with_uncertainty(X[:10])

        assert 'mean' in result
        assert 'epistemic_uncertainty' in result
        assert 'aleatoric_uncertainty' in result
        assert 'total_uncertainty' in result

        # Check shapes
        assert result['mean'].shape == (10, 1)
        assert result['epistemic_uncertainty'].shape == (10, 1)
        assert result['aleatoric_uncertainty'].shape == (10, 1)
        assert result['total_uncertainty'].shape == (10, 1)

        # Check values are in valid ranges
        assert torch.all(result['mean'] >= 0) and torch.all(result['mean'] <= 1)
        assert torch.all(result['epistemic_uncertainty'] >= 0)
        assert torch.all(result['aleatoric_uncertainty'] >= 0)

    def test_ensemble_predictions_vary(self, sample_data):
        """Test that different networks give different predictions."""
        X, _ = sample_data
        ensemble = ConvergenceEnsemble(input_dim=4, n_networks=5)

        predictions = ensemble(X[:10])

        # Convert to numpy for comparison
        pred_arrays = [p.detach().numpy() for p in predictions]

        # Networks should give different predictions (not all identical)
        for i in range(len(pred_arrays) - 1):
            assert not np.allclose(pred_arrays[i], pred_arrays[i+1])


class TestPerformanceEnsemble:
    """Test performance ensemble functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        torch.manual_seed(42)
        n_samples = 100
        n_features = 4

        X = torch.randn(n_samples, n_features)
        y = torch.randn(n_samples, 2)  # 2 outputs (iterations, time)

        return X, y

    def test_ensemble_initialization(self):
        """Test ensemble can be initialized."""
        ensemble = PerformanceEnsemble(input_dim=4, output_dim=2, n_networks=3)

        assert ensemble is not None
        assert len(ensemble.networks) == 3

    def test_ensemble_forward(self, sample_data):
        """Test forward pass produces correct output."""
        X, _ = sample_data
        ensemble = PerformanceEnsemble(input_dim=4, output_dim=2, n_networks=3)

        predictions = ensemble(X[:10])

        assert len(predictions) == 3  # 3 networks
        assert all(p.shape == (10, 2) for p in predictions)

    def test_predict_with_uncertainty(self, sample_data):
        """Test uncertainty quantification."""
        X, _ = sample_data
        ensemble = PerformanceEnsemble(input_dim=4, output_dim=2, n_networks=5)

        result = ensemble.predict_with_uncertainty(X[:10])

        assert 'mean' in result
        assert 'uncertainty' in result

        # Check shapes
        assert result['mean'].shape == (10, 2)
        assert result['uncertainty'].shape == (10, 2)

        # Check uncertainty is non-negative
        assert torch.all(result['uncertainty'] >= 0)
