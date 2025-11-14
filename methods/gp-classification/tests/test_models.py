"""
Tests for GP models.
"""

import pytest
import torch
from gpytorch.likelihoods import BernoulliLikelihood

from gp_classification.models import (
    VariationalGPClassifier,
    DualModel,
    train_variational_gp_classifier,
    predict_convergence_probability,
)


@pytest.fixture
def synthetic_classification_data():
    """Generate synthetic binary classification data."""
    torch.manual_seed(42)

    # Generate data with clear decision boundary
    X = torch.randn(100, 3, dtype=torch.float64)
    # Simple linear decision boundary: converged if x0 + x1 > 0
    y = ((X[:, 0] + X[:, 1]) > 0).float().unsqueeze(-1)

    return X, y


@pytest.fixture
def synthetic_regression_data():
    """Generate synthetic regression data."""
    torch.manual_seed(42)

    # Converged samples only
    X = torch.randn(50, 3, dtype=torch.float64)
    # Objective = simple function of inputs
    y = (X[:, 0] ** 2 + X[:, 1]).unsqueeze(-1)

    return X, y


def test_variational_gp_classifier_initialization(synthetic_classification_data):
    """Test VariationalGPClassifier initialization."""
    X, y = synthetic_classification_data

    model = VariationalGPClassifier(
        train_X=X,
        train_Y=y,
        n_inducing_points=20,
    )

    assert model is not None
    assert model.mean_module is not None
    assert model.covar_module is not None


def test_variational_gp_classifier_training(synthetic_classification_data):
    """Test training of VariationalGPClassifier."""
    X, y = synthetic_classification_data

    model = VariationalGPClassifier(
        train_X=X,
        train_Y=y,
        n_inducing_points=20,
    )

    likelihood = BernoulliLikelihood()

    # Train for a few epochs
    model_trained, likelihood_trained = train_variational_gp_classifier(
        model=model,
        likelihood=likelihood,
        train_X=X,
        train_Y=y,
        n_epochs=50,
        learning_rate=0.05,
        verbose=False,
    )

    assert model_trained is not None
    assert likelihood_trained is not None


def test_convergence_prediction(synthetic_classification_data):
    """Test convergence probability prediction."""
    X, y = synthetic_classification_data

    model = VariationalGPClassifier(
        train_X=X,
        train_Y=y,
        n_inducing_points=20,
    )
    likelihood = BernoulliLikelihood()

    # Train
    model, likelihood = train_variational_gp_classifier(
        model=model,
        likelihood=likelihood,
        train_X=X,
        train_Y=y,
        n_epochs=50,
        verbose=False,
    )

    # Predict on test data
    X_test = torch.randn(10, 3, dtype=torch.float64)
    probs, stds = predict_convergence_probability(
        model=model,
        likelihood=likelihood,
        X=X_test,
        n_samples=100,
    )

    assert probs.shape == (10,)
    assert stds.shape == (10,)
    assert torch.all((probs >= 0) & (probs <= 1))
    assert torch.all(stds >= 0)


def test_dual_model_initialization(
    synthetic_classification_data, synthetic_regression_data
):
    """Test DualModel initialization."""
    X_all, y_converged = synthetic_classification_data
    X_success, y_objective = synthetic_regression_data

    dual_model = DualModel(
        train_X_all=X_all,
        train_Y_converged=y_converged,
        train_X_success=X_success,
        train_Y_objective=y_objective,
        n_inducing_points=20,
    )

    assert dual_model is not None
    assert dual_model.convergence_model is not None
    assert dual_model.objective_model is not None
    assert not dual_model.is_trained


def test_dual_model_training(synthetic_classification_data, synthetic_regression_data):
    """Test DualModel training."""
    X_all, y_converged = synthetic_classification_data
    X_success, y_objective = synthetic_regression_data

    dual_model = DualModel(
        train_X_all=X_all,
        train_Y_converged=y_converged,
        train_X_success=X_success,
        train_Y_objective=y_objective,
        n_inducing_points=20,
    )

    # Train both models
    dual_model.train_models(
        train_X_all=X_all,
        train_Y_converged=y_converged,
        n_epochs=50,
        verbose=False,
    )

    assert dual_model.is_trained


def test_dual_model_predictions(
    synthetic_classification_data, synthetic_regression_data
):
    """Test predictions from DualModel."""
    X_all, y_converged = synthetic_classification_data
    X_success, y_objective = synthetic_regression_data

    dual_model = DualModel(
        train_X_all=X_all,
        train_Y_converged=y_converged,
        train_X_success=X_success,
        train_Y_objective=y_objective,
        n_inducing_points=20,
    )

    dual_model.train_models(
        train_X_all=X_all,
        train_Y_converged=y_converged,
        n_epochs=50,
        verbose=False,
    )

    # Test data
    X_test = torch.randn(5, 3, dtype=torch.float64)

    # Convergence prediction
    probs, stds = dual_model.predict_convergence(X_test)
    assert probs.shape == (5,)
    assert stds.shape == (5,)

    # Objective prediction
    obj_mean, obj_std = dual_model.predict_objective(X_test)
    assert obj_mean.shape == (5,)
    assert obj_std.shape == (5,)
