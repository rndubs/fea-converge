"""
Multi-Task Gaussian Process for uncertainty quantification across geometries.

Enables transfer learning across different FEA geometries by sharing
statistical strength through task descriptors (geometric features).
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from botorch.models import MultiTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


class GeometricFeatureExtractor:
    """
    Extract geometric features from FEA problems for task descriptors.

    Features include:
    - Contact area ratio
    - Mesh density
    - Material property contrast
    - Gap distribution statistics
    """

    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = [
            "contact_area_ratio",
            "mesh_density",
            "material_contrast",
            "gap_mean",
            "gap_std",
            "num_contact_pairs",
        ]

    def extract(self, geometry_data: Dict) -> np.ndarray:
        """
        Extract features from geometry data.

        Args:
            geometry_data: Dictionary containing geometric information

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Contact area ratio
        total_area = geometry_data.get("total_surface_area", 1.0)
        contact_area = geometry_data.get("potential_contact_area", 0.5)
        features.append(contact_area / total_area)

        # Mesh density (elements per unit volume)
        volume = geometry_data.get("volume", 1.0)
        num_elements = geometry_data.get("num_elements", 1000)
        features.append(num_elements / volume)

        # Material property contrast (E_max / E_min)
        E_values = geometry_data.get("youngs_moduli", [1e9, 1e9])
        features.append(max(E_values) / min(E_values))

        # Gap distribution statistics
        gaps = geometry_data.get("initial_gaps", [1e-3])
        features.append(np.mean(gaps))
        features.append(np.std(gaps))

        # Number of contact pairs
        features.append(geometry_data.get("num_contact_pairs", 100))

        return np.array(features)

    def extract_synthetic(self, geometry_id: int, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate synthetic geometric features for testing.

        Args:
            geometry_id: Unique geometry identifier
            random_seed: Random seed

        Returns:
            Synthetic feature vector
        """
        rng = np.random.RandomState(random_seed if random_seed is not None else geometry_id)

        features = [
            rng.uniform(0.1, 0.9),  # contact_area_ratio
            rng.uniform(100, 10000),  # mesh_density
            rng.uniform(1.0, 10.0),  # material_contrast
            rng.uniform(1e-4, 1e-2),  # gap_mean
            rng.uniform(1e-5, 1e-3),  # gap_std
            rng.uniform(10, 1000),  # num_contact_pairs
        ]

        return np.array(features)


class MultiTaskGPOptimizer:
    """
    Multi-Task GP for optimization across multiple geometries.

    Enables transfer learning by sharing information across similar
    geometric configurations.
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        task_indices: torch.Tensor,
        task_features: Optional[torch.Tensor] = None,
    ):
        """
        Initialize Multi-Task GP.

        Args:
            train_X: Training inputs (parameters)
            train_Y: Training outputs (objectives)
            task_indices: Task index for each training point
            task_features: Optional task feature vectors (geometric features)
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.task_indices = task_indices
        self.task_features = task_features

        # Combine parameters with task indices
        train_X_full = torch.cat([train_X, task_indices.unsqueeze(-1).float()], dim=-1)

        # Create Multi-Task GP
        self.model = MultiTaskGP(
            train_X=train_X_full,
            train_Y=train_Y,
            task_feature=task_indices.long(),
            output_tasks=[int(idx.item()) for idx in torch.unique(task_indices)],
            outcome_transform=Standardize(m=1),
        )

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def optimize_hyperparameters(self, num_restarts: int = 3):
        """
        Optimize GP hyperparameters.

        Args:
            num_restarts: Number of random restarts
        """
        self.model.train()
        self.model.likelihood.train()

        best_loss = float("inf")
        best_state = None

        for restart in range(num_restarts):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

            for i in range(100):
                optimizer.zero_grad()
                train_X_full = torch.cat(
                    [self.train_X, self.task_indices.unsqueeze(-1).float()], dim=-1
                )
                output = self.model(train_X_full)
                loss = -self.mll(output, self.train_Y.squeeze())
                loss.backward()
                optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = self.model.state_dict()

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.model.likelihood.eval()

    def predict_for_task(
        self, test_X: torch.Tensor, task_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for a specific task.

        Args:
            test_X: Test parameters
            task_idx: Task index to predict for

        Returns:
            Tuple of (mean, variance)
        """
        self.model.eval()
        self.model.likelihood.eval()

        # Add task index to test inputs
        task_indices = torch.full((test_X.shape[0], 1), task_idx, dtype=torch.float32)
        test_X_full = torch.cat([test_X, task_indices], dim=-1)

        with torch.no_grad():
            posterior = self.model.posterior(test_X_full)
            mean = posterior.mean
            variance = posterior.variance

        return mean, variance

    def recommend_parameters_ucb(
        self,
        task_idx: int,
        candidate_X: torch.Tensor,
        beta: float = 2.0,
        top_k: int = 3,
    ) -> List[Dict[str, float]]:
        """
        Recommend parameters for new geometry using UCB.

        Args:
            task_idx: Task index for new geometry
            candidate_X: Candidate parameter sets
            beta: UCB exploration parameter
            top_k: Number of recommendations to return

        Returns:
            List of recommended parameter configurations with scores
        """
        mean, variance = self.predict_for_task(candidate_X, task_idx)
        std = torch.sqrt(variance)

        # UCB score (lower is better for minimization)
        ucb_scores = mean.squeeze() - beta * std.squeeze()

        # Get top-k recommendations
        top_k_indices = torch.argsort(ucb_scores)[:top_k]

        recommendations = []
        for idx in top_k_indices:
            idx_val = idx.item()
            recommendations.append({
                "parameters": candidate_X[idx_val].numpy(),
                "predicted_objective": mean[idx_val].item(),
                "uncertainty": std[idx_val].item(),
                "ucb_score": ucb_scores[idx_val].item(),
                "success_probability": self._estimate_success_prob(
                    mean[idx_val], variance[idx_val]
                ),
            })

        return recommendations

    def _estimate_success_prob(self, mean: torch.Tensor, variance: torch.Tensor) -> float:
        """
        Estimate success probability based on predicted objective.

        Args:
            mean: Predicted mean objective
            variance: Predicted variance

        Returns:
            Estimated success probability
        """
        # Heuristic: success if objective < 1.0
        # P(obj < 1.0) assuming Gaussian
        threshold = 1.0
        std = torch.sqrt(variance)

        z = (threshold - mean) / (std + 1e-6)
        prob = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))

        return prob.item()


class GeometryOptimizationManager:
    """
    Manages optimization across multiple geometries with transfer learning.
    """

    def __init__(self, feature_extractor: Optional[GeometricFeatureExtractor] = None):
        """
        Initialize geometry optimization manager.

        Args:
            feature_extractor: Feature extractor for geometric properties
        """
        self.feature_extractor = feature_extractor or GeometricFeatureExtractor()
        self.geometries: Dict[int, Dict] = {}
        self.trials: List[Dict] = []
        self.mtgp: Optional[MultiTaskGPOptimizer] = None

    def register_geometry(self, geometry_id: int, geometry_data: Dict):
        """
        Register a new geometry for optimization.

        Args:
            geometry_id: Unique geometry identifier
            geometry_data: Dictionary containing geometric information
        """
        features = self.feature_extractor.extract(geometry_data)
        self.geometries[geometry_id] = {
            "data": geometry_data,
            "features": features,
        }

    def add_trial(
        self,
        geometry_id: int,
        parameters: np.ndarray,
        objective: float,
        converged: bool,
    ):
        """
        Add trial result to database.

        Args:
            geometry_id: Geometry identifier
            parameters: Parameter vector
            objective: Objective value
            converged: Whether simulation converged
        """
        self.trials.append({
            "geometry_id": geometry_id,
            "parameters": parameters,
            "objective": objective,
            "converged": converged,
        })

    def train_multi_task_gp(self):
        """Train multi-task GP on all accumulated trials."""
        if not self.trials:
            raise ValueError("No trials available for training")

        # Prepare training data
        train_X_list = []
        train_Y_list = []
        task_indices_list = []

        for trial in self.trials:
            if trial["converged"]:  # Only use successful trials
                train_X_list.append(trial["parameters"])
                train_Y_list.append(trial["objective"])
                task_indices_list.append(trial["geometry_id"])

        if not train_X_list:
            raise ValueError("No successful trials for training")

        # Convert to tensors
        train_X = torch.tensor(np.array(train_X_list), dtype=torch.float32)
        train_Y = torch.tensor(train_Y_list, dtype=torch.float32).unsqueeze(-1)
        task_indices = torch.tensor(task_indices_list, dtype=torch.long)

        # Train Multi-Task GP
        self.mtgp = MultiTaskGPOptimizer(train_X, train_Y, task_indices)
        self.mtgp.optimize_hyperparameters()

    def recommend_for_new_geometry(
        self,
        geometry_id: int,
        candidate_parameters: np.ndarray,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Recommend parameters for a new geometry.

        Args:
            geometry_id: Geometry identifier
            candidate_parameters: Array of candidate parameter sets
            top_k: Number of recommendations

        Returns:
            List of recommendations with confidence estimates
        """
        if self.mtgp is None:
            raise ValueError("Multi-Task GP not trained yet")

        candidate_X = torch.tensor(candidate_parameters, dtype=torch.float32)

        recommendations = self.mtgp.recommend_parameters_ucb(
            task_idx=geometry_id,
            candidate_X=candidate_X,
            top_k=top_k,
        )

        return recommendations
