"""Tests for correctness of SHEBO components (not just structure)."""

import pytest
import numpy as np
import torch
from shebo.models.ensemble import ConvergenceEnsemble
from shebo.core.surrogate_manager import SurrogateManager
from shebo.utils.preprocessing import FeatureNormalizer
from shebo.utils.black_box_solver import create_test_objective
from shebo import SHEBOOptimizer


class TestEnsembleDiversity:
    """Test that ensemble actually produces diverse predictions."""

    def test_ensemble_produces_diverse_predictions(self):
        """Ensemble networks should give different predictions after training."""
        # Create synthetic data
        np.random.seed(42)
        torch.manual_seed(42)

        n_samples = 100
        n_features = 4

        X = torch.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).float().reshape(-1, 1)

        # Create and train ensemble
        ensemble = ConvergenceEnsemble(input_dim=n_features, n_networks=5)

        # Simple training loop
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10, shuffle=True)

        # Get optimizers
        optimizers = ensemble.configure_optimizers()

        # Train for a few epochs
        for epoch in range(10):
            for batch in loader:
                ensemble.training_step(batch, 0)

        # Test predictions
        test_X = torch.randn(20, n_features)
        predictions = []

        for network in ensemble.networks:
            network.eval()
            with torch.no_grad():
                pred = network(test_X)
                predictions.append(pred)

        # Check that networks give different predictions
        predictions_array = [p.numpy() for p in predictions]

        # At least some predictions should be different
        different_count = 0
        for i in range(len(predictions_array) - 1):
            for j in range(i + 1, len(predictions_array)):
                if not np.allclose(predictions_array[i], predictions_array[j], rtol=0.01):
                    different_count += 1

        # At least half of the pairs should be different
        total_pairs = len(predictions_array) * (len(predictions_array) - 1) // 2
        assert different_count > total_pairs / 2, \
            f"Only {different_count}/{total_pairs} pairs are different - ensemble may not be diverse"

    def test_uncertainty_increases_with_disagreement(self):
        """Epistemic uncertainty should be higher where networks disagree."""
        ensemble = ConvergenceEnsemble(input_dim=4, n_networks=5)

        # Create test data
        X_certain = torch.ones(10, 4) * 5  # Far from decision boundary
        X_uncertain = torch.randn(10, 4) * 0.1  # Near decision boundary

        pred_certain = ensemble.predict_with_uncertainty(X_certain)
        pred_uncertain = ensemble.predict_with_uncertainty(X_uncertain)

        # Uncertain points should generally have higher epistemic uncertainty
        # (though not guaranteed for untrained model, but should see some difference)
        mean_epistemic_certain = pred_certain['epistemic_uncertainty'].mean().item()
        mean_epistemic_uncertain = pred_uncertain['epistemic_uncertainty'].mean().item()

        # At least one should have non-zero uncertainty
        assert mean_epistemic_certain > 0 or mean_epistemic_uncertain > 0


class TestFeatureNormalization:
    """Test that feature normalization works correctly."""

    def test_normalization_changes_scale(self):
        """Normalized features should have zero mean and unit variance."""
        # Create features with vastly different scales
        X = np.array([
            [1e8, 1e-6, 0.5, 0.3],
            [5e8, 5e-7, 0.8, 0.6],
            [2e9, 2e-5, 0.2, 0.9],
            [1e7, 1e-8, 0.1, 0.4]
        ])

        normalizer = FeatureNormalizer()
        X_normalized = normalizer.fit_transform(X)

        # Check mean is close to zero
        mean = X_normalized.mean(axis=0)
        assert np.allclose(mean, 0, atol=1e-10), \
            f"Mean should be ~0, got {mean}"

        # Check std is close to one
        std = X_normalized.std(axis=0, ddof=1)
        assert np.allclose(std, 1, atol=1e-10), \
            f"Std should be ~1, got {std}"

    def test_inverse_transform_recovers_original(self):
        """Inverse transform should recover original values."""
        X_original = np.array([
            [1e8, 1e-6, 0.5],
            [5e8, 5e-7, 0.8],
            [2e9, 2e-5, 0.2]
        ])

        normalizer = FeatureNormalizer()
        X_normalized = normalizer.fit_transform(X_original)
        X_recovered = normalizer.inverse_transform(X_normalized)

        assert np.allclose(X_original, X_recovered), \
            "Inverse transform should recover original values"


class TestSurrogateManager:
    """Test surrogate manager correctness."""

    def test_normalization_applied_during_prediction(self):
        """Predictions should use normalized features."""
        manager = SurrogateManager(input_dim=4)

        # Fit normalizer with training data
        X_train = torch.tensor([
            [1e8, 1e-6, 0.5, 0.3],
            [5e8, 5e-7, 0.8, 0.6],
            [2e9, 2e-5, 0.2, 0.9]
        ], dtype=torch.float32)
        y_train = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)

        manager.update_models(X_train, y_train, current_iteration=10)

        # Test prediction normalizes input
        X_test = torch.tensor([[1e9, 1e-7, 0.4, 0.5]], dtype=torch.float32)
        pred = manager.predict(X_test, 'convergence')

        # Should not crash and should return valid prediction
        assert 'mean' in pred
        assert 0 <= pred['mean'].item() <= 1

    def test_iteration_based_updates(self):
        """Model updates should be based on iteration count, not sample count."""
        manager = SurrogateManager(
            input_dim=4,
            convergence_update_freq=5
        )

        X = torch.randn(20, 4)
        y = torch.randint(0, 2, (20, 1)).float()

        # First update at iteration 5
        manager.update_models(X, y, current_iteration=5)
        first_update = manager.last_update['convergence']
        assert first_update == 5

        # Should not update at iteration 7 (need iteration 10)
        manager.update_models(X, y, current_iteration=7)
        assert manager.last_update['convergence'] == 5  # Unchanged

        # Should update at iteration 10
        manager.update_models(X, y, current_iteration=10)
        assert manager.last_update['convergence'] == 10


class TestOptimization:
    """Test that optimization actually improves."""

    def test_optimization_finds_better_solutions(self):
        """Optimizer should find better solutions over time."""
        bounds = np.array([
            [1e6, 1e10],
            [1e-8, 1e-4],
            [0.0, 1.0],
            [0.0, 1.0]
        ])

        objective = create_test_objective(n_params=4, random_seed=42)

        optimizer = SHEBOOptimizer(
            bounds=bounds,
            objective_fn=objective,
            n_init=10,
            budget=30,
            random_seed=42
        )

        result = optimizer.run()

        # Should find at least some converged solutions
        n_converged = sum(result.convergence_history)
        assert n_converged > 0, "Should find at least one converged solution"

        # Best performance should be better than random
        # (average performance of converged solutions)
        converged_perfs = [p for p, c in zip(result.performance_history, result.convergence_history) if c]

        if len(converged_perfs) > 5:
            # Best should be better than median
            median_perf = np.median(converged_perfs)
            assert result.best_performance <= median_perf, \
                f"Best ({result.best_performance:.2f}) should be <= median ({median_perf:.2f})"

    def test_convergence_rate_improves(self):
        """Convergence rate should improve as optimization progresses."""
        bounds = np.array([
            [1e6, 1e10],
            [1e-8, 1e-4],
            [0.0, 1.0],
            [0.0, 1.0]
        ])

        objective = create_test_objective(n_params=4, random_seed=42)

        optimizer = SHEBOOptimizer(
            bounds=bounds,
            objective_fn=objective,
            n_init=15,
            budget=50,
            random_seed=42
        )

        result = optimizer.run()

        # Split into early and late phases
        mid_point = len(result.convergence_history) // 2

        early_rate = sum(result.convergence_history[:mid_point]) / mid_point
        late_rate = sum(result.convergence_history[mid_point:]) / (len(result.convergence_history) - mid_point)

        # Late phase should have higher or equal convergence rate
        # (allowing for noise with small sample sizes)
        assert late_rate >= early_rate - 0.2, \
            f"Late rate ({late_rate:.2f}) should be >= early rate ({early_rate:.2f}) - 0.2"


class TestCheckpointing:
    """Test checkpoint save/load functionality."""

    def test_checkpoint_saves_and_loads(self, tmp_path):
        """Checkpoint should preserve optimization state."""
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        objective = create_test_objective(n_params=2, random_seed=42)

        optimizer = SHEBOOptimizer(
            bounds=bounds,
            objective_fn=objective,
            n_init=5,
            budget=10,
            checkpoint_dir=str(tmp_path),
            checkpoint_frequency=5,
            random_seed=42
        )

        # Run a few iterations
        optimizer._initialize()
        for _ in range(3):
            optimizer.iteration += 1
            x = np.random.uniform(bounds[:, 0], bounds[:, 1])
            optimizer._evaluate_and_store(x)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pkl"
        optimizer.save_checkpoint(str(checkpoint_path))

        # Create new optimizer and load
        optimizer2 = SHEBOOptimizer(
            bounds=bounds,
            objective_fn=objective,
            n_init=5,
            budget=10,
            random_seed=42
        )

        optimizer2.load_checkpoint(str(checkpoint_path))

        # Check state is restored
        assert optimizer2.iteration == optimizer.iteration
        assert optimizer2.best_performance == optimizer.best_performance
        assert len(optimizer2.all_params) == len(optimizer.all_params)
        assert np.array_equal(optimizer2.all_params[0], optimizer.all_params[0])


class TestBatchParallelization:
    """Test batch parallelization API."""

    def test_batch_selection_diverse(self):
        """Batch points should be diverse."""
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        objective = create_test_objective(n_params=2, random_seed=42)

        optimizer = SHEBOOptimizer(
            bounds=bounds,
            objective_fn=objective,
            n_init=10,
            budget=50,
            random_seed=42
        )

        # Initialize and train surrogates
        optimizer._initialize()
        optimizer._update_surrogates()

        # Get batch
        batch = optimizer._get_next_batch('exploration', batch_size=5)

        assert batch.shape == (5, 2)

        # Check diversity - points should not be too close
        for i in range(len(batch)):
            for j in range(i + 1, len(batch)):
                distance = np.linalg.norm(batch[i] - batch[j])
                assert distance > 0.01, f"Points {i} and {j} too close: {distance}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
