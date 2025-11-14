"""
Use case implementations for GP Classification system.

Includes:
- Initial parameter suggestions via clustering
- Real-time convergence probability estimation
- Pre-simulation validation pipeline
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .data import SimulationTrial, TrialDatabase
from .models import DualModel


class ParameterSuggester:
    """
    Suggests initial parameters for new geometries using clustering and transfer learning.
    """

    def __init__(
        self,
        database: TrialDatabase,
        dual_model: DualModel,
        n_clusters: int = 5,
        n_neighbors: int = 5,
    ):
        """
        Initialize parameter suggester.

        Args:
            database: Trial database with historical data
            dual_model: Trained dual model for convergence prediction
            n_clusters: Number of clusters for parameter grouping
            n_neighbors: Number of neighbors for similarity matching
        """
        self.database = database
        self.dual_model = dual_model
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors

    def suggest_parameters(
        self, geometry_features: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest initial parameters for a new geometry.

        Args:
            geometry_features: Optional geometric features for similarity matching
                (e.g., contact_area, mesh_size, gap_mean)

        Returns:
            List of suggestions, each containing:
                - parameters: Dict[str, float]
                - convergence_probability: float
                - confidence: str ('high', 'medium', 'low')
                - expected_objective: Optional[float]
        """
        # Get successful trials
        successful_trials = self.database.get_successful_trials()

        if not successful_trials:
            raise ValueError("No successful trials in database for suggestion")

        # If geometry features provided, filter by similarity
        if geometry_features is not None:
            similar_trials = self._find_similar_geometries(successful_trials, geometry_features)
            if len(similar_trials) < 3:
                # Fall back to all successful trials if too few similar ones
                similar_trials = successful_trials
        else:
            similar_trials = successful_trials

        # Extract parameters from similar trials
        param_matrix = []
        for trial in similar_trials:
            params = [trial.parameters[name] for name in self.database.parameter_names]
            param_matrix.append(params)

        param_matrix = np.array(param_matrix)

        # Cluster parameters
        n_clusters = min(self.n_clusters, len(similar_trials))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(param_matrix)

        # Generate suggestions from cluster centers
        suggestions = []
        for i, center in enumerate(kmeans.cluster_centers_):
            # Convert to parameter dict
            parameters = {
                name: float(center[j]) for j, name in enumerate(self.database.parameter_names)
            }

            # Predict convergence probability
            param_tensor = torch.tensor(center, dtype=torch.float64).unsqueeze(0)
            prob, std = self.dual_model.predict_convergence(param_tensor)

            prob_val = prob.item()
            std_val = std.item()

            # Determine confidence level
            if std_val < 0.15:
                confidence = "high"
            elif std_val < 0.3:
                confidence = "medium"
            else:
                confidence = "low"

            # Predict expected objective if model available
            expected_objective = None
            if self.dual_model.objective_model is not None:
                obj_mean, obj_std = self.dual_model.predict_objective(param_tensor)
                expected_objective = obj_mean.item()

            suggestions.append(
                {
                    "cluster_id": i,
                    "parameters": parameters,
                    "convergence_probability": prob_val,
                    "convergence_uncertainty": std_val,
                    "confidence": confidence,
                    "expected_objective": expected_objective,
                    "n_similar_trials": int(np.sum(kmeans.labels_ == i)),
                }
            )

        # Sort by convergence probability (descending)
        suggestions.sort(key=lambda x: x["convergence_probability"], reverse=True)

        return suggestions

    def _find_similar_geometries(
        self, trials: List[SimulationTrial], target_features: Dict[str, float]
    ) -> List[SimulationTrial]:
        """
        Find trials from similar geometries using k-NN.

        Args:
            trials: List of trials to search
            target_features: Target geometry features

        Returns:
            k most similar trials
        """
        # Extract geometry features from trials
        trials_with_features = [t for t in trials if t.geometry_metadata is not None]

        if not trials_with_features:
            return trials  # No geometry metadata, return all

        # Get feature names from target
        feature_names = sorted(target_features.keys())

        # Build feature matrix
        feature_matrix = []
        valid_trials = []

        for trial in trials_with_features:
            # Check if trial has all required features
            if all(name in trial.geometry_metadata for name in feature_names):
                features = [trial.geometry_metadata[name] for name in feature_names]
                feature_matrix.append(features)
                valid_trials.append(trial)

        if not valid_trials:
            return trials

        feature_matrix = np.array(feature_matrix)
        target_vector = np.array([target_features[name] for name in feature_names])

        # Normalize features
        mean = feature_matrix.mean(axis=0)
        std = feature_matrix.std(axis=0) + 1e-8
        feature_matrix = (feature_matrix - mean) / std
        target_vector = (target_vector - mean) / std

        # Find k nearest neighbors
        k = min(self.n_neighbors, len(valid_trials))
        knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        knn.fit(feature_matrix)

        distances, indices = knn.kneighbors(target_vector.reshape(1, -1))

        similar_trials = [valid_trials[i] for i in indices[0]]

        return similar_trials


class PreSimulationValidator:
    """
    Multi-stage validation pipeline for parameter checking before simulation.
    """

    def __init__(
        self,
        database: TrialDatabase,
        dual_model: DualModel,
        min_convergence_prob: float = 0.3,
        n_neighbors: int = 10,
    ):
        """
        Initialize pre-simulation validator.

        Args:
            database: Trial database
            dual_model: Trained dual model
            min_convergence_prob: Minimum acceptable convergence probability
            n_neighbors: Number of neighbors for local success rate
        """
        self.database = database
        self.dual_model = dual_model
        self.min_convergence_prob = min_convergence_prob
        self.n_neighbors = n_neighbors

    def validate(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate parameters through multi-stage pipeline.

        Args:
            parameters: Parameter dictionary to validate

        Returns:
            Validation result containing:
                - approved: bool
                - risk_score: float (0-1, higher is riskier)
                - risk_level: str ('low', 'moderate', 'high')
                - ml_prediction: float (convergence probability)
                - ml_uncertainty: float
                - physics_violations: List[str]
                - local_success_rate: float
                - recommendation: str
                - safe_alternative: Optional[Dict[str, float]]
        """
        # Stage 1: ML Prediction
        param_tensor = torch.tensor(
            [parameters[name] for name in self.database.parameter_names], dtype=torch.float64
        ).unsqueeze(0)

        prob, std = self.dual_model.predict_convergence(param_tensor)
        ml_prob = prob.item()
        ml_std = std.item()

        # Stage 2: Physics Rules
        violations = self._check_physics_rules(parameters)

        physics_penalty = len(violations) / 10.0  # Normalize

        # Stage 3: Historical Comparison
        local_success_rate = self._compute_local_success_rate(parameters)

        # Aggregate into composite risk score
        risk_score = (
            0.4 * (1 - ml_prob) + 0.3 * physics_penalty + 0.3 * (1 - local_success_rate)
        )

        risk_score = np.clip(risk_score, 0.0, 1.0)

        # Determine risk level and action
        if risk_score > 0.7:
            risk_level = "high"
            approved = False
            recommendation = "REJECT: High risk of convergence failure"
            # Try to find safe alternative
            safe_alternative = self._find_safe_alternative(parameters)
        elif risk_score > 0.4:
            risk_level = "moderate"
            approved = True  # Allow with warning
            recommendation = "WARNING: Moderate risk - consider adjustments"
            safe_alternative = self._find_safe_alternative(parameters)
        else:
            risk_level = "low"
            approved = True
            recommendation = "APPROVED: Low risk of convergence failure"
            safe_alternative = None

        return {
            "approved": approved,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "ml_prediction": ml_prob,
            "ml_uncertainty": ml_std,
            "physics_violations": violations,
            "local_success_rate": local_success_rate,
            "recommendation": recommendation,
            "safe_alternative": safe_alternative,
        }

    def _check_physics_rules(self, parameters: Dict[str, float]) -> List[str]:
        """
        Check physics-based parameter constraints.

        Returns list of violation descriptions.
        """
        violations = []

        # Example physics rules (adapt to your specific problem)

        # Rule 1: Penalty stiffness bounds
        if "penalty_stiffness" in parameters:
            penalty = parameters["penalty_stiffness"]
            if penalty < 1e3:
                violations.append("Penalty stiffness too low (< 1e3)")
            if penalty > 1e8:
                violations.append("Penalty stiffness too high (> 1e8)")

        # Rule 2: Gap tolerance bounds
        if "gap_tolerance" in parameters:
            gap_tol = parameters["gap_tolerance"]
            if gap_tol < 1e-9:
                violations.append("Gap tolerance too tight (< 1e-9)")
            if gap_tol > 1e-6:
                violations.append("Gap tolerance too loose (> 1e-6)")

        # Rule 3: Timestep vs contact timescale
        if "timestep" in parameters:
            dt = parameters["timestep"]
            if dt < 1e-6:
                violations.append("Timestep too small (< 1e-6)")
            if dt > 1e-2:
                violations.append("Timestep too large (> 1e-2)")

        # Rule 4: Tolerance consistency
        if "absolute_tolerance" in parameters and "relative_tolerance" in parameters:
            abs_tol = parameters["absolute_tolerance"]
            rel_tol = parameters["relative_tolerance"]
            if abs_tol > rel_tol * 100:
                violations.append("Absolute tolerance too large relative to relative tolerance")

        return violations

    def _compute_local_success_rate(self, parameters: Dict[str, float]) -> float:
        """
        Compute success rate of k-nearest neighbors in parameter space.
        """
        if len(self.database) < self.n_neighbors:
            return 0.5  # Not enough data

        # Get all trials
        X_all, y_converged, _ = self.database.get_training_data(converged_only=False)

        # Convert parameters to tensor
        param_tensor = torch.tensor(
            [parameters[name] for name in self.database.parameter_names], dtype=torch.float64
        )

        # Compute distances
        distances = torch.norm(X_all - param_tensor, dim=1)

        # Find k nearest neighbors
        k = min(self.n_neighbors, len(X_all))
        _, indices = torch.topk(distances, k, largest=False)

        # Compute success rate
        neighbor_outcomes = y_converged[indices]
        success_rate = neighbor_outcomes.mean().item()

        return success_rate

    def _find_safe_alternative(self, parameters: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Find nearest safe parameters using constrained optimization.

        Returns safe parameters if found, None otherwise.
        """
        from scipy.optimize import minimize

        # Objective: minimize distance to proposed parameters
        param_vector = np.array([parameters[name] for name in self.database.parameter_names])

        def objective(x):
            return np.sum((x - param_vector) ** 2)

        # Constraint: P(converge) >= 0.8
        def convergence_constraint(x):
            x_tensor = torch.tensor(x, dtype=torch.float64).unsqueeze(0)
            prob, _ = self.dual_model.predict_convergence(x_tensor)
            return prob.item() - 0.8  # >= 0 means feasible

        # Bounds
        bounds = [self.database.parameter_bounds[name] for name in self.database.parameter_names]

        # Optimize
        try:
            result = minimize(
                objective,
                param_vector,
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": convergence_constraint},
                options={"maxiter": 100},
            )

            if result.success:
                safe_params = {
                    name: float(result.x[i]) for i, name in enumerate(self.database.parameter_names)
                }
                return safe_params
            else:
                return None

        except Exception:
            return None


class RealTimeEstimator:
    """
    Real-time convergence probability estimation for interactive exploration.
    """

    def __init__(self, dual_model: DualModel, parameter_names: List[str]):
        """
        Initialize real-time estimator.

        Args:
            dual_model: Trained dual model
            parameter_names: Ordered list of parameter names
        """
        self.dual_model = dual_model
        self.parameter_names = parameter_names

    def estimate(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """
        Estimate convergence probability and expected objective in real-time.

        Args:
            parameters: Parameter dictionary

        Returns:
            Dictionary with convergence_prob, uncertainty, expected_objective
        """
        param_tensor = torch.tensor(
            [parameters[name] for name in self.parameter_names], dtype=torch.float64
        ).unsqueeze(0)

        # Convergence prediction
        prob, std = self.dual_model.predict_convergence(param_tensor)

        result = {
            "convergence_probability": prob.item(),
            "convergence_uncertainty": std.item(),
        }

        # Objective prediction if available
        if self.dual_model.objective_model is not None:
            obj_mean, obj_std = self.dual_model.predict_objective(param_tensor)
            result["expected_objective"] = obj_mean.item()
            result["objective_uncertainty"] = obj_std.item()
        else:
            result["expected_objective"] = None
            result["objective_uncertainty"] = None

        return result

    def estimate_batch(
        self, parameter_list: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        Estimate for batch of parameters (vectorized for efficiency).

        Args:
            parameter_list: List of parameter dictionaries

        Returns:
            List of estimation results
        """
        # Convert to tensor
        param_tensors = []
        for params in parameter_list:
            tensor = torch.tensor(
                [params[name] for name in self.parameter_names], dtype=torch.float64
            )
            param_tensors.append(tensor)

        param_batch = torch.stack(param_tensors)

        # Batch prediction
        probs, stds = self.dual_model.predict_convergence(param_batch)

        results = []
        for i in range(len(parameter_list)):
            result = {
                "convergence_probability": probs[i].item(),
                "convergence_uncertainty": stds[i].item(),
            }

            if self.dual_model.objective_model is not None:
                obj_means, obj_stds = self.dual_model.predict_objective(param_batch)
                result["expected_objective"] = obj_means[i].item()
                result["objective_uncertainty"] = obj_stds[i].item()
            else:
                result["expected_objective"] = None
                result["objective_uncertainty"] = None

            results.append(result)

        return results
