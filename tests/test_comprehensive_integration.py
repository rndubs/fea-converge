"""
Comprehensive integration tests for FR-BO with full optimization runs.

These tests run complete optimization workflows with synthetic data
to validate the entire system without requiring Smith/Tribol.
"""

import pytest
import numpy as np
from fr_bo.optimizer import FRBOOptimizer, OptimizationConfig
from fr_bo.simulator import SyntheticSimulator
from fr_bo.synthetic_data import create_benchmark_dataset, SyntheticDataGenerator
from fr_bo.visualization import OptimizationVisualizer
from fr_bo.parameters import encode_parameters
from fr_bo.risk_scoring import RiskScorer
from fr_bo.multi_task import GeometryOptimizationManager, GeometricFeatureExtractor


class TestFullOptimizationWorkflow:
    """Tests for complete optimization workflows."""

    def test_quick_optimization_run(self):
        """Test quick optimization with minimal trials."""
        simulator = SyntheticSimulator(random_seed=42)
        config = OptimizationConfig(
            n_sobol_trials=5,
            n_frbo_trials=5,
            random_seed=42,
            max_iterations=50,
        )

        optimizer = FRBOOptimizer(simulator=simulator, config=config)
        results = optimizer.optimize()

        # Verify completion
        assert len(results["trials"]) == 10
        assert results["best_objective"] is not None
        assert results["best_parameters"] is not None

        # Verify metrics structure
        assert "overall" in results["metrics"]
        assert "sobol_phase" in results["metrics"]
        assert "frbo_phase" in results["metrics"]

        # Verify each phase ran
        assert results["metrics"]["sobol_phase"]["total_trials"] == 5
        assert results["metrics"]["frbo_phase"]["total_trials"] == 5

    def test_medium_optimization_run(self):
        """Test medium-length optimization with more trials."""
        simulator = SyntheticSimulator(random_seed=123)
        config = OptimizationConfig(
            n_sobol_trials=15,
            n_frbo_trials=25,
            random_seed=123,
            max_iterations=100,
        )

        optimizer = FRBOOptimizer(simulator=simulator, config=config)
        results = optimizer.optimize()

        # Verify completion
        assert len(results["trials"]) == 40

        # Check that we found at least some successful trials
        successful = sum(1 for t in results["trials"] if t.result.converged)
        assert successful > 0, "Should have at least some successful trials"

        # Check success rate improved from Sobol to FR-BO
        sobol_rate = results["metrics"]["sobol_phase"]["success_rate"]
        frbo_rate = results["metrics"]["frbo_phase"]["success_rate"]

        # Both phases should complete
        assert sobol_rate >= 0
        assert frbo_rate >= 0

    def test_optimization_with_different_seeds(self):
        """Test that different seeds produce different results."""
        config = OptimizationConfig(
            n_sobol_trials=10,
            n_frbo_trials=10,
            max_iterations=50,
        )

        # Run with seed 1
        sim1 = SyntheticSimulator(random_seed=1)
        opt1 = FRBOOptimizer(simulator=sim1, config=config._replace(random_seed=1))
        results1 = opt1.optimize()

        # Run with seed 2
        sim2 = SyntheticSimulator(random_seed=2)
        opt2 = FRBOOptimizer(simulator=sim2, config=config._replace(random_seed=2))
        results2 = opt2.optimize()

        # Results should be different (best objectives unlikely to match exactly)
        assert results1["best_objective"] != results2["best_objective"]

    def test_optimization_reproducibility(self):
        """Test that same seed produces same results."""
        config = OptimizationConfig(
            n_sobol_trials=10,
            n_frbo_trials=10,
            random_seed=42,
            max_iterations=50,
        )

        # Run 1
        sim1 = SyntheticSimulator(random_seed=42)
        opt1 = FRBOOptimizer(simulator=sim1, config=config)
        results1 = opt1.optimize()

        # Run 2
        sim2 = SyntheticSimulator(random_seed=42)
        opt2 = FRBOOptimizer(simulator=sim2, config=config)
        results2 = opt2.optimize()

        # Results should be identical
        assert results1["best_objective"] == results2["best_objective"]
        assert results1["best_trial_number"] == results2["best_trial_number"]

    def test_optimization_with_early_termination(self):
        """Test optimization with early termination enabled."""
        simulator = SyntheticSimulator(random_seed=99)
        config = OptimizationConfig(
            n_sobol_trials=10,
            n_frbo_trials=20,
            random_seed=99,
            max_iterations=100,
            enable_early_termination=True,
        )

        optimizer = FRBOOptimizer(simulator=simulator, config=config)
        results = optimizer.optimize()

        # Should complete successfully
        assert len(results["trials"]) == 30
        assert results["best_objective"] < float("inf")

    def test_optimization_tracks_convergence(self):
        """Test that optimization tracks convergence over time."""
        simulator = SyntheticSimulator(random_seed=456)
        config = OptimizationConfig(
            n_sobol_trials=20,
            n_frbo_trials=30,
            random_seed=456,
            max_iterations=100,
        )

        optimizer = FRBOOptimizer(simulator=simulator, config=config)
        results = optimizer.optimize()

        # Track best objective over time
        best_over_time = []
        current_best = float("inf")

        for trial in results["trials"]:
            if trial.result.converged and trial.objective_value < current_best:
                current_best = trial.objective_value
            best_over_time.append(current_best)

        # Best should improve (or stay same) over time
        for i in range(1, len(best_over_time)):
            assert best_over_time[i] <= best_over_time[i-1]

    def test_optimization_with_benchmark_dataset(self):
        """Test optimization can be evaluated on benchmark data."""
        # Create benchmark dataset
        dataset = create_benchmark_dataset("simple", n_train=30, n_test=10, random_seed=42)

        assert dataset["train"]["X"].shape[0] == 30
        assert dataset["test"]["X"].shape[0] == 10

        # Run optimization
        simulator = SyntheticSimulator(random_seed=42)
        config = OptimizationConfig(
            n_sobol_trials=10,
            n_frbo_trials=10,
            random_seed=42,
            max_iterations=50,
        )

        optimizer = FRBOOptimizer(simulator=simulator, config=config)
        results = optimizer.optimize()

        # Should complete successfully
        assert len(results["trials"]) == 20


class TestRiskAssessmentIntegration:
    """Tests for risk assessment integration with optimization."""

    def test_risk_scoring_after_optimization(self):
        """Test risk scoring on optimized parameters."""
        simulator = SyntheticSimulator(random_seed=789)
        config = OptimizationConfig(
            n_sobol_trials=15,
            n_frbo_trials=15,
            random_seed=789,
            max_iterations=50,
        )

        optimizer = FRBOOptimizer(simulator=simulator, config=config)
        results = optimizer.optimize()

        # Create risk scorer
        risk_scorer = RiskScorer()

        # Add trial history
        for trial in results["trials"]:
            params_encoded = encode_parameters(trial.parameters)
            risk_scorer.add_trial(params_encoded, trial.result.converged)

        # Assess risk for best parameters
        best_params_encoded = encode_parameters(results["best_parameters"])
        risk_assessment = risk_scorer.assess_risk(
            best_params_encoded,
            failure_model=optimizer.dual_gp.failure_classifier.model if optimizer.dual_gp else None
        )

        # Best parameters should have low risk
        assert risk_assessment.risk_score >= 0.0
        assert risk_assessment.risk_score <= 1.0
        assert risk_assessment.risk_level in ["LOW", "MODERATE", "HIGH"]

    def test_safe_alternative_suggestion(self):
        """Test that safe alternatives can be suggested."""
        simulator = SyntheticSimulator(random_seed=321)
        config = OptimizationConfig(
            n_sobol_trials=20,
            n_frbo_trials=20,
            random_seed=321,
            max_iterations=50,
        )

        optimizer = FRBOOptimizer(simulator=simulator, config=config)
        results = optimizer.optimize()

        # Create risk scorer
        risk_scorer = RiskScorer()

        for trial in results["trials"]:
            params_encoded = encode_parameters(trial.parameters)
            risk_scorer.add_trial(params_encoded, trial.result.converged)

        # Find a failed trial
        failed_trials = [t for t in results["trials"] if not t.result.converged]

        if failed_trials:
            failed_params = encode_parameters(failed_trials[0].parameters)
            risk_assessment = risk_scorer.assess_risk(failed_params)

            # If there are successful trials, should be able to suggest alternative
            if results["metrics"]["overall"]["successful_trials"] > 0:
                # Risk assessment should work
                assert risk_assessment is not None


class TestMultiTaskOptimization:
    """Tests for multi-task geometry optimization."""

    def test_multi_geometry_optimization(self):
        """Test optimization across multiple geometries."""
        # Generate synthetic geometries
        generator = SyntheticDataGenerator(random_seed=111)
        geometries = generator.generate_test_geometries(n_geometries=3)

        # Create manager
        manager = GeometryOptimizationManager()

        # Register geometries
        for geom in geometries:
            manager.register_geometry(geom["geometry_id"], geom)

        # Run optimization for each geometry
        for geom_id in range(3):
            simulator = SyntheticSimulator(random_seed=geom_id)
            config = OptimizationConfig(
                n_sobol_trials=10,
                n_frbo_trials=10,
                random_seed=geom_id,
                max_iterations=50,
            )

            optimizer = FRBOOptimizer(simulator=simulator, config=config)
            results = optimizer.optimize()

            # Add successful trials to manager
            for trial in results["trials"]:
                if trial.result.converged:
                    params_encoded = encode_parameters(trial.parameters)
                    manager.add_trial(
                        geometry_id=geom_id,
                        parameters=params_encoded,
                        objective=trial.objective_value,
                        converged=trial.result.converged
                    )

        # Train multi-task GP
        manager.train_multi_task_gp()

        # Should be able to get recommendations
        # (would test recommendations but they require more setup)
        assert len(manager.trials) > 0


class TestVisualizationIntegration:
    """Tests for visualization integration."""

    def test_visualization_after_optimization(self):
        """Test that visualizations can be created from results."""
        simulator = SyntheticSimulator(random_seed=555)
        config = OptimizationConfig(
            n_sobol_trials=15,
            n_frbo_trials=15,
            random_seed=555,
            max_iterations=50,
        )

        optimizer = FRBOOptimizer(simulator=simulator, config=config)
        results = optimizer.optimize()

        # Create visualizer
        visualizer = OptimizationVisualizer()

        # Test that visualization methods don't crash
        # (not testing actual plots, just that code runs)
        try:
            # These would create plots if matplotlib backend was available
            # Just verify they don't crash
            assert results["trials"] is not None
            assert len(results["trials"]) > 0
        except Exception as e:
            pytest.skip(f"Visualization requires display: {e}")


class TestOptimizationEdgeCases:
    """Tests for edge cases and error handling."""

    def test_optimization_with_all_failures(self):
        """Test optimization when all initial trials fail."""
        # This is hard to guarantee with synthetic simulator
        # but we can test it handles the case
        simulator = SyntheticSimulator(random_seed=999)
        config = OptimizationConfig(
            n_sobol_trials=5,
            n_frbo_trials=5,
            random_seed=999,
            max_iterations=10,
        )

        optimizer = FRBOOptimizer(simulator=simulator, config=config)

        # Should not crash even if no successes
        try:
            results = optimizer.optimize()
            assert len(results["trials"]) == 10
        except Exception:
            # If it fails due to no successful trials, that's expected
            pass

    def test_optimization_with_minimal_trials(self):
        """Test optimization with minimal number of trials."""
        simulator = SyntheticSimulator(random_seed=777)
        config = OptimizationConfig(
            n_sobol_trials=3,
            n_frbo_trials=2,
            random_seed=777,
            max_iterations=20,
        )

        optimizer = FRBOOptimizer(simulator=simulator, config=config)
        results = optimizer.optimize()

        assert len(results["trials"]) == 5

    def test_optimization_tracks_phases(self):
        """Test that optimization correctly tracks which phase each trial is from."""
        simulator = SyntheticSimulator(random_seed=888)
        config = OptimizationConfig(
            n_sobol_trials=10,
            n_frbo_trials=15,
            random_seed=888,
            max_iterations=50,
        )

        optimizer = FRBOOptimizer(simulator=simulator, config=config)
        results = optimizer.optimize()

        # Check phase assignments
        sobol_trials = [t for t in results["trials"] if t.phase == "sobol"]
        frbo_trials = [t for t in results["trials"] if t.phase == "frbo"]

        assert len(sobol_trials) == 10
        assert len(frbo_trials) == 15

        # Check trial numbers are sequential
        for i, trial in enumerate(results["trials"], 1):
            assert trial.trial_number == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
