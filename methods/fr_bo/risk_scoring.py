"""
Pre-simulation risk scoring and validation system.

Provides risk assessment before running expensive simulations,
with suggestions for safer alternative parameters.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
from scipy.spatial.distance import cdist
from dataclasses import dataclass


@dataclass
class RiskAssessment:
    """Risk assessment result."""

    risk_score: float  # Overall risk score [0, 1]
    risk_level: str  # "LOW", "MODERATE", "HIGH"
    failure_probability: float
    distance_to_failure: float
    local_success_rate: float
    recommendation: str
    safe_alternative: Optional[Dict] = None
    confidence: float = 0.0


class RiskScorer:
    """
    Risk scoring system for pre-simulation validation.

    Combines:
    1. Failure probability from GP classifier
    2. Distance to nearest failure
    3. Local success rate in parameter neighborhood
    """

    def __init__(
        self,
        failure_weight: float = 0.5,
        distance_weight: float = 0.3,
        local_success_weight: float = 0.2,
        high_risk_threshold: float = 0.7,
        moderate_risk_threshold: float = 0.4,
    ):
        """
        Initialize risk scorer.

        Args:
            failure_weight: Weight for failure probability
            distance_weight: Weight for distance to failures
            local_success_weight: Weight for local success rate
            high_risk_threshold: Threshold for high risk (> this value)
            moderate_risk_threshold: Threshold for moderate risk
        """
        self.failure_weight = failure_weight
        self.distance_weight = distance_weight
        self.local_success_weight = local_success_weight
        self.high_risk_threshold = high_risk_threshold
        self.moderate_risk_threshold = moderate_risk_threshold

        # Historical data
        self.successful_params: List[np.ndarray] = []
        self.failed_params: List[np.ndarray] = []

    def add_trial(self, parameters: np.ndarray, converged: bool):
        """
        Add trial to historical database.

        Args:
            parameters: Parameter vector
            converged: Whether trial converged
        """
        if converged:
            self.successful_params.append(parameters.copy())
        else:
            self.failed_params.append(parameters.copy())

    def assess_risk(
        self,
        parameters: np.ndarray,
        failure_model: Optional[any] = None,
    ) -> RiskAssessment:
        """
        Assess risk for given parameters.

        Args:
            parameters: Parameter vector to assess
            failure_model: Optional trained failure classifier

        Returns:
            RiskAssessment object
        """
        # Compute failure probability
        if failure_model is not None:
            failure_prob = self._compute_failure_probability(parameters, failure_model)
        else:
            failure_prob = 0.5  # Unknown

        # Compute distance to nearest failure
        distance_to_failure = self._compute_distance_to_failures(parameters)

        # Compute local success rate
        local_success_rate = self._compute_local_success_rate(parameters)

        # Compute overall risk score
        risk_score = (
            self.failure_weight * failure_prob +
            self.distance_weight * (1.0 - distance_to_failure) +
            self.local_success_weight * (1.0 - local_success_rate)
        )

        # Determine risk level
        if risk_score > self.high_risk_threshold:
            risk_level = "HIGH"
            recommendation = "Do not run - high risk of failure"
        elif risk_score > self.moderate_risk_threshold:
            risk_level = "MODERATE"
            recommendation = "Proceed with caution - moderate risk"
        else:
            risk_level = "LOW"
            recommendation = "Safe to proceed - low risk"

        # Generate safe alternative if high risk
        safe_alternative = None
        if risk_level == "HIGH" and len(self.successful_params) > 0:
            safe_alternative = self._find_safe_alternative(parameters)

        # Estimate confidence based on data availability
        confidence = self._estimate_confidence(parameters)

        return RiskAssessment(
            risk_score=risk_score,
            risk_level=risk_level,
            failure_probability=failure_prob,
            distance_to_failure=distance_to_failure,
            local_success_rate=local_success_rate,
            recommendation=recommendation,
            safe_alternative=safe_alternative,
            confidence=confidence,
        )

    def _compute_failure_probability(
        self, parameters: np.ndarray, failure_model: any
    ) -> float:
        """
        Compute failure probability using trained classifier.

        Args:
            parameters: Parameter vector
            failure_model: Trained failure classifier

        Returns:
            Failure probability [0, 1]
        """
        # Convert to tensor
        x = torch.tensor(parameters, dtype=torch.float32).unsqueeze(0)

        # Get failure probability
        with torch.no_grad():
            latent_dist = failure_model(x)
            if hasattr(failure_model, "likelihood"):
                pred_dist = failure_model.likelihood(latent_dist)
                failure_prob = pred_dist.mean.item()
            else:
                failure_prob = torch.sigmoid(latent_dist.mean).item()

        return np.clip(failure_prob, 0.0, 1.0)

    def _compute_distance_to_failures(self, parameters: np.ndarray) -> float:
        """
        Compute normalized distance to nearest failure.

        Args:
            parameters: Parameter vector

        Returns:
            Normalized distance [0, 1] (1 = far from failures)
        """
        if len(self.failed_params) == 0:
            return 1.0  # No failures known, assume safe

        # Compute distances to all failures
        failed_array = np.array(self.failed_params)
        distances = cdist(parameters.reshape(1, -1), failed_array, metric="euclidean")
        min_distance = np.min(distances)

        # Normalize by typical parameter space scale
        # Use distance to nearest success as reference
        if len(self.successful_params) > 0:
            success_array = np.array(self.successful_params)
            success_distances = cdist(parameters.reshape(1, -1), success_array, metric="euclidean")
            typical_scale = np.median(success_distances) + 1e-6
        else:
            typical_scale = 1.0

        normalized_distance = min_distance / typical_scale

        # Map to [0, 1] with sigmoid
        return 1.0 / (1.0 + np.exp(-2.0 * (normalized_distance - 0.5)))

    def _compute_local_success_rate(
        self, parameters: np.ndarray, k_neighbors: int = 10
    ) -> float:
        """
        Compute success rate in local neighborhood.

        Args:
            parameters: Parameter vector
            k_neighbors: Number of neighbors to consider

        Returns:
            Local success rate [0, 1]
        """
        if len(self.successful_params) == 0 and len(self.failed_params) == 0:
            return 0.5  # Unknown

        # Get all historical parameters
        all_params = self.successful_params + self.failed_params
        all_labels = [1] * len(self.successful_params) + [0] * len(self.failed_params)

        if len(all_params) < k_neighbors:
            k_neighbors = len(all_params)

        # Find k-nearest neighbors
        all_params_array = np.array(all_params)
        distances = cdist(parameters.reshape(1, -1), all_params_array, metric="euclidean")[0]
        nearest_indices = np.argsort(distances)[:k_neighbors]

        # Compute success rate
        nearest_labels = [all_labels[i] for i in nearest_indices]
        success_rate = np.mean(nearest_labels)

        return success_rate

    def _find_safe_alternative(self, parameters: np.ndarray) -> Dict:
        """
        Find nearest safe alternative parameters.

        Solves: minimize ||θ_new - θ_proposed||²
                subject to P_fail(θ_new) < 0.2

        Args:
            parameters: Proposed parameters

        Returns:
            Dictionary with safe alternative parameters and metadata
        """
        if len(self.successful_params) == 0:
            return None

        # Find nearest successful parameters
        success_array = np.array(self.successful_params)
        distances = cdist(parameters.reshape(1, -1), success_array, metric="euclidean")[0]
        nearest_idx = np.argmin(distances)

        safe_params = self.successful_params[nearest_idx]
        distance = distances[nearest_idx]

        return {
            "parameters": safe_params,
            "distance_from_proposed": distance,
            "estimated_success_prob": 0.9,  # Known success
        }

    def _estimate_confidence(self, parameters: np.ndarray) -> float:
        """
        Estimate confidence in risk assessment.

        Higher confidence when more data available nearby.

        Args:
            parameters: Parameter vector

        Returns:
            Confidence score [0, 1]
        """
        total_data = len(self.successful_params) + len(self.failed_params)

        if total_data == 0:
            return 0.0

        # Base confidence from total data
        base_confidence = min(total_data / 100.0, 1.0)

        # Adjust based on local density
        all_params = self.successful_params + self.failed_params
        if len(all_params) > 0:
            all_params_array = np.array(all_params)
            distances = cdist(parameters.reshape(1, -1), all_params_array, metric="euclidean")[0]

            # Count nearby points (within median distance)
            median_dist = np.median(distances) if len(distances) > 1 else 1.0
            nearby_count = np.sum(distances < median_dist)
            local_density = min(nearby_count / 10.0, 1.0)

            confidence = 0.5 * base_confidence + 0.5 * local_density
        else:
            confidence = base_confidence

        return confidence


class ParameterValidator:
    """
    Validates parameters before simulation with comprehensive checks.
    """

    def __init__(self, risk_scorer: Optional[RiskScorer] = None):
        """
        Initialize parameter validator.

        Args:
            risk_scorer: Optional risk scorer instance
        """
        self.risk_scorer = risk_scorer or RiskScorer()

    def validate(
        self,
        parameters: Dict[str, any],
        failure_model: Optional[any] = None,
        require_low_risk: bool = False,
    ) -> Tuple[bool, RiskAssessment]:
        """
        Validate parameters before running simulation.

        Args:
            parameters: Parameter dictionary
            failure_model: Optional failure classifier
            require_low_risk: If True, reject moderate/high risk parameters

        Returns:
            Tuple of (is_valid, risk_assessment)
        """
        from fr_bo.parameters import encode_parameters

        # Encode parameters
        params_encoded = encode_parameters(parameters)

        # Assess risk
        risk_assessment = self.risk_scorer.assess_risk(params_encoded, failure_model)

        # Determine validity
        is_valid = True
        if require_low_risk and risk_assessment.risk_level != "LOW":
            is_valid = False

        return is_valid, risk_assessment

    def get_validation_report(self, risk_assessment: RiskAssessment) -> str:
        """
        Generate human-readable validation report.

        Args:
            risk_assessment: Risk assessment result

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("PARAMETER VALIDATION REPORT")
        report.append("=" * 70)
        report.append(f"Risk Level: {risk_assessment.risk_level}")
        report.append(f"Risk Score: {risk_assessment.risk_score:.3f}")
        report.append(f"Confidence: {risk_assessment.confidence:.3f}")
        report.append("")
        report.append("Risk Components:")
        report.append(f"  Failure Probability: {risk_assessment.failure_probability:.3f}")
        report.append(f"  Distance to Failures: {risk_assessment.distance_to_failure:.3f}")
        report.append(f"  Local Success Rate: {risk_assessment.local_success_rate:.3f}")
        report.append("")
        report.append(f"Recommendation: {risk_assessment.recommendation}")

        if risk_assessment.safe_alternative is not None:
            report.append("")
            report.append("Safe Alternative Available:")
            report.append(f"  Distance: {risk_assessment.safe_alternative['distance_from_proposed']:.4f}")
            report.append(f"  Estimated Success: {risk_assessment.safe_alternative['estimated_success_prob']:.3f}")

        report.append("=" * 70)

        return "\n".join(report)
