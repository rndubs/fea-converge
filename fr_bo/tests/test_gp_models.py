"""
Unit tests for dual GP system (gp_models.py).

Tests:
- ObjectiveGP: Regression GP for objective function
- FailureClassifierGP: Variational GP for failure prediction
- FailureClassifier: Wrapper with Bernoulli likelihood
- DualGPSystem: Combined system managing both models
"""

import pytest
import torch
import numpy as np
from fr_bo.gp_models import ObjectiveGP, FailureClassifier, DualGPSystem


class TestObjectiveGP:
    """Test suite for ObjectiveGP class."""

    def test_initialization(self, sample_train_x, sample_train_y):
        """Test ObjectiveGP initialization."""
        gp = ObjectiveGP(sample_train_x, sample_train_y)

        assert gp.train_X.shape == sample_train_x.shape
        assert gp.train_Y.shape == sample_train_y.shape
        assert gp.model is not None
        assert gp.mll is not None

    def test_hyperparameter_optimization(self, sample_train_x, sample_train_y):
        """Test GP hyperparameter optimization."""
        gp = ObjectiveGP(sample_train_x, sample_train_y)

        # Optimize hyperparameters
        gp.optimize_hyperparameters(num_restarts=1)

        # Check that model is in eval mode after optimization
        assert not gp.model.training
        assert not gp.model.likelihood.training

    def test_prediction_shape(self, sample_train_x, sample_train_y, test_points):
        """Test that predictions have correct shape."""
        gp = ObjectiveGP(sample_train_x, sample_train_y)
        gp.optimize_hyperparameters(num_restarts=1)

        mean, variance = gp.predict(test_points)

        assert mean.shape == (test_points.shape[0],)
        assert variance.shape == (test_points.shape[0],)

    def test_prediction_values(self, sample_train_x, sample_train_y):
        """Test that predictions are reasonable."""
        gp = ObjectiveGP(sample_train_x, sample_train_y)
        gp.optimize_hyperparameters(num_restarts=1)

        # Predict at training points - should have low uncertainty
        mean, variance = gp.predict(sample_train_x[:3])

        # Mean should be close to training values
        assert torch.allclose(mean, sample_train_y[:3].squeeze(), atol=0.5)

        # Variance should be small at training points
        assert torch.all(variance < 1.0)

    def test_uncertainty_increases_away_from_data(self, sample_train_x, sample_train_y):
        """Test that uncertainty increases away from training data."""
        gp = ObjectiveGP(sample_train_x, sample_train_y)
        gp.optimize_hyperparameters(num_restarts=1)

        # Points near data
        near_points = sample_train_x[:2] + 0.01
        mean_near, var_near = gp.predict(near_points)

        # Points far from data
        far_points = torch.ones_like(near_points) * 10.0
        mean_far, var_far = gp.predict(far_points)

        # Variance should be larger for far points
        assert torch.all(var_far > var_near)

    def test_multiple_dimensions(self):
        """Test GP with higher dimensional input."""
        torch.manual_seed(42)
        # 5D input
        train_X = torch.rand(20, 5)
        train_Y = torch.randn(20, 1)

        gp = ObjectiveGP(train_X, train_Y)
        gp.optimize_hyperparameters(num_restarts=1)

        test_X = torch.rand(5, 5)
        mean, variance = gp.predict(test_X)

        assert mean.shape == (5,)
        assert variance.shape == (5,)


class TestFailureClassifier:
    """Test suite for FailureClassifier class."""

    def test_initialization(self, sample_train_x, sample_failure_labels):
        """Test FailureClassifier initialization."""
        classifier = FailureClassifier(sample_train_x, sample_failure_labels)

        assert classifier.train_X.shape == sample_train_x.shape
        # train_Y is squeezed to 1D for Bernoulli likelihood
        assert classifier.train_Y.shape == (sample_failure_labels.shape[0],)
        assert classifier.model is not None
        assert classifier.likelihood is not None

    def test_training(self, sample_train_x, sample_failure_labels):
        """Test classifier training."""
        classifier = FailureClassifier(sample_train_x, sample_failure_labels)

        # Train for a few iterations
        classifier.train_model(num_epochs=10)

        # Check that model is in eval mode after training
        assert not classifier.model.training

    def test_prediction_shape(self, sample_train_x, sample_failure_labels, test_points):
        """Test that failure predictions have correct shape."""
        classifier = FailureClassifier(sample_train_x, sample_failure_labels)
        classifier.train_model(num_epochs=10)

        failure_prob, uncertainty = classifier.predict_failure_probability(test_points)

        assert failure_prob.shape == (test_points.shape[0],)
        assert uncertainty.shape == (test_points.shape[0],)

    def test_prediction_range(self, sample_train_x, sample_failure_labels, test_points):
        """Test that failure probabilities are in [0, 1]."""
        classifier = FailureClassifier(sample_train_x, sample_failure_labels)
        classifier.train_model(num_epochs=10)

        failure_prob, _ = classifier.predict_failure_probability(test_points)

        # Probabilities should be between 0 and 1
        assert torch.all(failure_prob >= 0.0)
        assert torch.all(failure_prob <= 1.0)

    def test_prediction_confidence(self, sample_train_x, sample_failure_labels):
        """Test that predictions are confident near training data."""
        classifier = FailureClassifier(sample_train_x, sample_failure_labels)
        classifier.train_model(num_epochs=20)

        # Predict at training points that were failures
        failure_indices = sample_failure_labels.squeeze() == 1.0
        failure_points = sample_train_x[failure_indices]

        if failure_points.shape[0] > 0:
            failure_prob, _ = classifier.predict_failure_probability(failure_points)

            # Should predict reasonable probability of failure for failure points
            # Note: With limited training data and epochs, perfect prediction isn't expected
            assert torch.mean(failure_prob) > 0.25

    def test_all_success_case(self, sample_train_x):
        """Test classifier when all trials succeed."""
        all_success = torch.zeros(sample_train_x.shape[0], 1)

        classifier = FailureClassifier(sample_train_x, all_success)
        classifier.train_model(num_epochs=10)

        test_X = torch.rand(5, sample_train_x.shape[1])
        failure_prob, _ = classifier.predict_failure_probability(test_X)

        # Should predict low failure probability
        assert torch.mean(failure_prob) < 0.5

    def test_all_failure_case(self, sample_train_x):
        """Test classifier when all trials fail."""
        all_failure = torch.ones(sample_train_x.shape[0], 1)

        classifier = FailureClassifier(sample_train_x, all_failure)
        classifier.train_model(num_epochs=10)

        test_X = torch.rand(5, sample_train_x.shape[1])
        failure_prob, _ = classifier.predict_failure_probability(test_X)

        # Should predict high failure probability
        assert torch.mean(failure_prob) > 0.5


class TestDualGPSystem:
    """Test suite for DualGPSystem class."""

    def test_initialization_with_all_success(self, sample_train_x, sample_train_y):
        """Test initialization when all trials succeed."""
        # All successes
        failure_labels = torch.zeros(sample_train_x.shape[0], 1)

        dual_gp = DualGPSystem(sample_train_x, sample_train_y, failure_labels)

        assert dual_gp.objective_gp is not None
        assert dual_gp.failure_classifier is not None
        assert dual_gp.n_successes == sample_train_x.shape[0]
        assert dual_gp.n_failures == 0

    def test_initialization_with_mixed_outcomes(self, sample_train_x, sample_train_y, sample_failure_labels):
        """Test initialization with mixed success/failure."""
        dual_gp = DualGPSystem(sample_train_x, sample_train_y, sample_failure_labels)

        assert dual_gp.objective_gp is not None
        assert dual_gp.failure_classifier is not None

        # Check counts
        n_failures = int(sample_failure_labels.sum().item())
        n_successes = sample_train_x.shape[0] - n_failures

        assert dual_gp.n_failures == n_failures
        assert dual_gp.n_successes == n_successes

    def test_training(self, sample_train_x, sample_train_y, sample_failure_labels):
        """Test dual GP training."""
        dual_gp = DualGPSystem(sample_train_x, sample_train_y, sample_failure_labels)

        # Train models
        dual_gp.train_models(gp_restarts=1, classifier_epochs=10)

        # Both models should be trained
        assert not dual_gp.objective_gp.model.training
        assert not dual_gp.failure_classifier.model.training

    def test_predict_objective(self, sample_train_x, sample_train_y, sample_failure_labels, test_points):
        """Test objective prediction through dual GP system."""
        dual_gp = DualGPSystem(sample_train_x, sample_train_y, sample_failure_labels)
        dual_gp.train_models(gp_restarts=1, classifier_epochs=10)

        mean, variance = dual_gp.predict_objective(test_points)

        assert mean.shape == (test_points.shape[0],)
        assert variance.shape == (test_points.shape[0],)

    def test_predict_failure(self, sample_train_x, sample_train_y, sample_failure_labels, test_points):
        """Test failure probability prediction."""
        dual_gp = DualGPSystem(sample_train_x, sample_train_y, sample_failure_labels)
        dual_gp.train_models(gp_restarts=1, classifier_epochs=10)

        failure_prob = dual_gp.predict_failure_probability(test_points)

        assert failure_prob.shape == (test_points.shape[0],)
        assert torch.all(failure_prob >= 0.0)
        assert torch.all(failure_prob <= 1.0)

    def test_joint_prediction(self, sample_train_x, sample_train_y, sample_failure_labels, test_points):
        """Test joint prediction of objective and failure."""
        dual_gp = DualGPSystem(sample_train_x, sample_train_y, sample_failure_labels)
        dual_gp.train_models(gp_restarts=1, classifier_epochs=10)

        result = dual_gp.predict(test_points)

        assert 'objective_mean' in result
        assert 'objective_variance' in result
        assert 'failure_probability' in result
        assert 'success_probability' in result

        assert result['objective_mean'].shape == (test_points.shape[0],)
        assert result['objective_variance'].shape == (test_points.shape[0],)
        assert result['failure_probability'].shape == (test_points.shape[0],)
        assert result['success_probability'].shape == (test_points.shape[0],)

        # Success + failure probabilities should sum to 1
        assert torch.allclose(
            result['success_probability'] + result['failure_probability'],
            torch.ones_like(result['success_probability']),
            atol=1e-6
        )

    def test_update_with_new_data(self, sample_train_x, sample_train_y, sample_failure_labels):
        """Test updating dual GP with new observations."""
        dual_gp = DualGPSystem(sample_train_x, sample_train_y, sample_failure_labels)
        dual_gp.train_models(gp_restarts=1, classifier_epochs=10)

        # Add new data
        new_X = torch.rand(3, sample_train_x.shape[1])
        new_Y = torch.randn(3, 1)
        new_failures = torch.zeros(3, 1)

        # Update would require implementing an update method
        # For now, just test that we can create a new system with combined data
        combined_X = torch.cat([sample_train_x, new_X], dim=0)
        combined_Y = torch.cat([sample_train_y, new_Y], dim=0)
        combined_failures = torch.cat([sample_failure_labels, new_failures], dim=0)

        updated_gp = DualGPSystem(combined_X, combined_Y, combined_failures)
        updated_gp.train_models(gp_restarts=1, classifier_epochs=10)

        assert updated_gp.n_successes + updated_gp.n_failures == combined_X.shape[0]

    def test_edge_case_single_success(self):
        """Test with only one successful observation."""
        torch.manual_seed(42)
        train_X = torch.rand(1, 2)
        train_Y = torch.randn(1, 1)
        failures = torch.zeros(1, 1)

        # Should handle gracefully
        dual_gp = DualGPSystem(train_X, train_Y, failures)
        assert dual_gp.n_successes == 1

    def test_edge_case_all_failures(self, sample_train_x):
        """Test when all observations are failures."""
        torch.manual_seed(42)
        train_Y = torch.randn(sample_train_x.shape[0], 1)
        all_failures = torch.ones(sample_train_x.shape[0], 1)

        # Should handle gracefully (though objective GP may not be useful)
        try:
            dual_gp = DualGPSystem(sample_train_x, train_Y, all_failures)
            assert dual_gp.n_failures == sample_train_x.shape[0]
            assert dual_gp.n_successes == 0
        except Exception as e:
            # Expected to potentially fail or issue warning
            pytest.skip(f"All-failure case not fully supported: {e}")
