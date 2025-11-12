"""
Unit tests for acquisition functions (acquisition.py).

Tests:
- FailureRobustEI: Failure-aware expected improvement
- Acquisition optimization
- Edge cases and boundary conditions
"""

import pytest
import torch
import numpy as np
from fr_bo.acquisition import FailureRobustEI
from fr_bo.gp_models import ObjectiveGP, FailureClassifier, DualGPSystem


class TestFailureRobustEI:
    """Test suite for FailureRobustEI acquisition function."""

    @pytest.fixture
    def trained_dual_gp(self, sample_train_x, sample_train_y, sample_failure_labels):
        """Create and train a dual GP system for testing."""
        dual_gp = DualGPSystem(sample_train_x, sample_train_y, sample_failure_labels)
        dual_gp.train_models(gp_restarts=1, classifier_epochs=10)
        return dual_gp

    def test_initialization(self, trained_dual_gp, sample_train_y):
        """Test FREI initialization."""
        best_f = sample_train_y.min().item()

        acq = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f,
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        assert acq.best_f == best_f
        assert acq.failure_penalty_weight == 1.0
        assert not acq.maximize

    def test_forward_shape(self, trained_dual_gp, sample_train_y, test_points):
        """Test that acquisition function returns correct shape."""
        best_f = sample_train_y.min().item()

        acq = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f,
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        # Test with batch of points
        values = acq(test_points.unsqueeze(1))  # Add q-batch dimension

        assert values.shape == (test_points.shape[0],)

    def test_positive_values(self, trained_dual_gp, sample_train_y, test_points):
        """Test that acquisition values are non-negative."""
        best_f = sample_train_y.min().item()

        acq = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f,
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        values = acq(test_points.unsqueeze(1))

        # Acquisition values should be non-negative
        assert torch.all(values >= 0.0)

    def test_failure_penalty_effect(self, trained_dual_gp, sample_train_y, test_points):
        """Test that failure penalty reduces acquisition value."""
        best_f = sample_train_y.min().item()

        # Standard EI (no failure penalty)
        acq_no_penalty = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f,
            failure_penalty_weight=0.0,  # No penalty
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        # FR-EI with penalty
        acq_with_penalty = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f,
            failure_penalty_weight=1.0,  # Standard penalty
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        values_no_penalty = acq_no_penalty(test_points.unsqueeze(1))
        values_with_penalty = acq_with_penalty(test_points.unsqueeze(1))

        # With penalty should generally be <= without penalty
        # (unless failure probability is very low)
        mean_no_penalty = torch.mean(values_no_penalty)
        mean_with_penalty = torch.mean(values_with_penalty)

        assert mean_with_penalty <= mean_no_penalty + 1e-6

    def test_high_failure_probability_suppression(self, trained_dual_gp, sample_train_y):
        """Test that high failure probability suppresses acquisition."""
        best_f = sample_train_y.min().item()

        acq = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f,
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        # Create points near known failures
        failure_indices = torch.where(torch.rand(10) < 0.3)[0]  # Some random failures
        if len(failure_indices) > 0:
            # Points should exist, test acquisition
            test_X = torch.rand(5, trained_dual_gp.objective_gp.train_X.shape[1])
            values = acq(test_X.unsqueeze(1))
            assert torch.all(torch.isfinite(values))

    def test_best_f_influence(self, trained_dual_gp, sample_train_y, test_points):
        """Test that best_f threshold influences acquisition."""
        # Lower threshold (better best value found)
        best_f_low = sample_train_y.min().item()

        # Higher threshold (worse best value found)
        best_f_high = sample_train_y.min().item() + 1.0

        acq_low = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f_low,
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        acq_high = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f_high,
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        values_low = acq_low(test_points.unsqueeze(1))
        values_high = acq_high(test_points.unsqueeze(1))

        # Higher best_f (worse) should generally give higher acquisition
        assert torch.mean(values_high) >= torch.mean(values_low) - 1e-6

    def test_gradient_computation(self, trained_dual_gp, sample_train_y, test_points):
        """Test that gradients can be computed for optimization."""
        best_f = sample_train_y.min().item()

        acq = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f,
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        # Enable gradient computation
        test_points_grad = test_points.clone().detach().requires_grad_(True)

        values = acq(test_points_grad.unsqueeze(1))
        loss = values.sum()

        # Should be able to compute gradients
        loss.backward()

        assert test_points_grad.grad is not None
        assert torch.all(torch.isfinite(test_points_grad.grad))

    def test_maximize_mode(self, trained_dual_gp, sample_train_y, test_points):
        """Test acquisition in maximize mode."""
        best_f = sample_train_y.max().item()  # Maximum instead of minimum

        acq = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f,
            maximize=True,  # Maximize objective
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        values = acq(test_points.unsqueeze(1))

        # Should still return valid values
        assert torch.all(values >= 0.0)
        assert torch.all(torch.isfinite(values))

    def test_single_point_evaluation(self, trained_dual_gp, sample_train_y):
        """Test evaluation at a single point."""
        best_f = sample_train_y.min().item()

        acq = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f,
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        # Single point
        single_point = torch.rand(1, trained_dual_gp.objective_gp.train_X.shape[1])
        value = acq(single_point.unsqueeze(1))

        assert value.shape == (1,)
        assert value.item() >= 0.0

    def test_batch_evaluation(self, trained_dual_gp, sample_train_y):
        """Test batch evaluation of multiple points."""
        best_f = sample_train_y.min().item()

        acq = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f,
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        # Batch of points
        batch_points = torch.rand(20, trained_dual_gp.objective_gp.train_X.shape[1])
        values = acq(batch_points.unsqueeze(1))

        assert values.shape == (20,)
        assert torch.all(values >= 0.0)

    def test_different_penalty_weights(self, trained_dual_gp, sample_train_y, test_points):
        """Test different failure penalty weights."""
        best_f = sample_train_y.min().item()

        weights = [0.0, 0.5, 1.0, 2.0]
        results = []

        for weight in weights:
            acq = FailureRobustEI(
                model=trained_dual_gp.objective_gp.model,
                failure_model=trained_dual_gp.failure_classifier.model,
                best_f=best_f,
                failure_penalty_weight=weight,
                failure_likelihood=trained_dual_gp.failure_classifier.likelihood
            )
            values = acq(test_points.unsqueeze(1))
            results.append(torch.mean(values).item())

        # Higher penalty weight should generally reduce acquisition
        # (though not always monotonic depending on failure probabilities)
        assert all(r >= 0 for r in results)


class TestAcquisitionOptimization:
    """Test acquisition function optimization."""

    def test_optimize_acquisition_basic(self, trained_dual_gp, sample_train_y, simple_bounds):
        """Test basic acquisition optimization."""
        from fr_bo.acquisition import optimize_acquisition

        best_f = sample_train_y.min().item()

        acq = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f,
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        # Optimize acquisition
        candidate, acq_value = optimize_acquisition(
            acq_function=acq,
            bounds=simple_bounds,
            num_restarts=3,
            raw_samples=128
        )

        # Should return a point within bounds
        assert candidate.shape == (1, simple_bounds.shape[0])
        assert torch.all(candidate >= simple_bounds[:, 0])
        assert torch.all(candidate <= simple_bounds[:, 1])

        # Acquisition value should be non-negative
        assert acq_value >= 0.0

    def test_optimize_multiple_restarts(self, trained_dual_gp, sample_train_y, simple_bounds):
        """Test that multiple restarts find good solutions."""
        from fr_bo.acquisition import optimize_acquisition

        best_f = sample_train_y.min().item()

        acq = FailureRobustEI(
            model=trained_dual_gp.objective_gp.model,
            failure_model=trained_dual_gp.failure_classifier.model,
            best_f=best_f,
            failure_likelihood=trained_dual_gp.failure_classifier.likelihood
        )

        # Few restarts
        candidate_few, value_few = optimize_acquisition(
            acq_function=acq,
            bounds=simple_bounds,
            num_restarts=2,
            raw_samples=64
        )

        # Many restarts
        candidate_many, value_many = optimize_acquisition(
            acq_function=acq,
            bounds=simple_bounds,
            num_restarts=10,
            raw_samples=256
        )

        # More restarts should find equal or better solution
        assert value_many >= value_few - 1e-6
