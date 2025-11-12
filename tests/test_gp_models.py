"""Tests for GP models."""

import pytest
import torch
import numpy as np
from fr_bo.gp_models import ObjectiveGP, FailureClassifier, DualGPSystem


def test_objective_gp_basic():
    """Test basic ObjectiveGP functionality."""
    # Generate synthetic data
    train_X = torch.randn(20, 5)
    train_Y = torch.randn(20, 1)

    # Create and train GP
    gp = ObjectiveGP(train_X, train_Y)
    gp.optimize_hyperparameters(num_restarts=2)

    # Make predictions
    test_X = torch.randn(5, 5)
    mean, variance = gp.predict(test_X)

    assert mean.shape == (5,)
    assert variance.shape == (5,)
    assert torch.all(variance > 0)


def test_objective_gp_lengthscales():
    """Test that lengthscales can be extracted."""
    train_X = torch.randn(20, 5)
    train_Y = torch.randn(20, 1)

    gp = ObjectiveGP(train_X, train_Y)
    gp.optimize_hyperparameters(num_restarts=1)

    lengthscales = gp.get_lengthscales()

    assert len(lengthscales) == 5
    assert np.all(lengthscales > 0)


def test_failure_classifier_basic():
    """Test basic FailureClassifier functionality."""
    # Generate synthetic data
    train_X = torch.randn(30, 5)
    train_Y = torch.randint(0, 2, (30,)).float()

    # Create and train classifier
    classifier = FailureClassifier(train_X, train_Y)
    classifier.train_model(num_epochs=50, lr=0.1)

    # Make predictions
    test_X = torch.randn(10, 5)
    failure_prob, uncertainty = classifier.predict_failure_probability(test_X)

    assert failure_prob.shape == (10,)
    assert uncertainty.shape == (10,)
    assert torch.all(failure_prob >= 0)
    assert torch.all(failure_prob <= 1)


def test_dual_gp_system():
    """Test DualGPSystem with mixed success/failure data."""
    # Generate synthetic data
    n_samples = 50
    train_X = torch.randn(n_samples, 5)
    train_Y = torch.randn(n_samples, 1)

    # Create failure labels (30% failures)
    failure_labels = torch.zeros(n_samples)
    failure_labels[:15] = 1.0  # First 15 are failures

    # For failed trials, set Y to NaN (will be handled by system)
    train_Y[:15] = float("nan")

    # Replace NaN for initialization
    train_Y_clean = torch.nan_to_num(train_Y, nan=0.0)

    # Create dual GP system
    dual_gp = DualGPSystem(train_X, train_Y_clean, failure_labels)
    dual_gp.train(num_restarts=2)

    # Make predictions
    test_X = torch.randn(5, 5)
    obj_mean, obj_var, fail_prob, fail_unc = dual_gp.predict(test_X)

    # Check objective predictions (should be available since we have successes)
    if obj_mean is not None:
        assert obj_mean.shape == (5,)
        assert obj_var.shape == (5,)

    # Check failure predictions
    assert fail_prob.shape == (5,)
    assert torch.all(fail_prob >= 0)
    assert torch.all(fail_prob <= 1)


def test_dual_gp_parameter_importance():
    """Test parameter importance extraction."""
    n_samples = 40
    train_X = torch.randn(n_samples, 5)
    train_Y = torch.randn(n_samples, 1)
    failure_labels = torch.zeros(n_samples)
    failure_labels[:10] = 1.0

    dual_gp = DualGPSystem(train_X, train_Y, failure_labels)
    dual_gp.train(num_restarts=1)

    importance = dual_gp.get_parameter_importance()

    if importance is not None:
        assert len(importance) == 5
        assert np.abs(importance.sum() - 1.0) < 1e-6  # Should sum to 1
        assert np.all(importance >= 0)


def test_dual_gp_all_failures():
    """Test DualGPSystem when all trials fail."""
    n_samples = 20
    train_X = torch.randn(n_samples, 5)
    train_Y = torch.zeros(n_samples, 1)  # Dummy values
    failure_labels = torch.ones(n_samples)  # All failures

    # This should still work (classifier trains, objective GP is None)
    dual_gp = DualGPSystem(train_X, train_Y, failure_labels)
    dual_gp.train(num_restarts=1)

    # Objective GP should be None
    assert dual_gp.objective_gp is None

    # But failure classifier should work
    test_X = torch.randn(5, 5)
    obj_mean, obj_var, fail_prob, fail_unc = dual_gp.predict(test_X)

    assert obj_mean is None
    assert obj_var is None
    assert fail_prob is not None
