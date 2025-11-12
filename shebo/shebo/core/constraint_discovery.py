"""Constraint discovery module for identifying hidden failure modes."""

from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class Violation:
    """Represents a constraint violation."""
    type: str
    severity: str  # 'low', 'medium', 'high'
    description: str
    value: Optional[float] = None


class ConstraintDiscovery:
    """Discovers hidden constraints through anomaly detection.

    Monitors simulation outputs for unexpected behaviors and failure patterns,
    automatically creating new constraint surrogates for discovered modes.
    """

    def __init__(self):
        """Initialize constraint discovery."""
        self.discovered_constraints: Dict[str, Dict[str, Any]] = {}
        self.expected_behaviors = {
            'residual_monotonic': 'Residual should decrease monotonically',
            'penetration_bounded': 'Penetration should remain < 1e-3',
            'no_nan_inf': 'No NaN or Inf values',
            'mesh_quality': 'Jacobian determinant > 0',
            'contact_detection': 'Contact pairs should be detected',
        }

    def check_simulation_output(
        self,
        output: Dict[str, Any]
    ) -> List[Violation]:
        """Analyze simulation output for anomalies.

        Args:
            output: Dictionary with simulation results containing:
                - residual_history: List of residual norms
                - penetration_max: Maximum penetration
                - convergence_status: Boolean
                - jacobian_min: Minimum Jacobian determinant
                - contact_pairs: Number of contact pairs detected
                - expected_contact: Whether contact is expected
                - all_values: All numerical values for NaN/Inf check

        Returns:
            List of discovered constraint violations
        """
        violations: List[Violation] = []

        # Check residual pattern
        violations.extend(self._check_residual_pattern(output))

        # Check for NaN/Inf
        violations.extend(self._check_numerical_stability(output))

        # Check penetration bounds
        violations.extend(self._check_penetration(output))

        # Check mesh quality
        violations.extend(self._check_mesh_quality(output))

        # Check contact detection
        violations.extend(self._check_contact_detection(output))

        return violations

    def _check_residual_pattern(
        self,
        output: Dict[str, Any]
    ) -> List[Violation]:
        """Check residual convergence pattern."""
        violations = []

        residual_history = output.get('residual_history', [])
        if not residual_history or len(residual_history) < 5:
            return violations

        # Check for oscillation (non-monotonic decrease after initial iterations)
        if len(residual_history) > 10:
            recent = residual_history[-10:]
            oscillations = sum(
                1 for i in range(1, len(recent))
                if recent[i] > recent[i-1]
            )

            if oscillations > 3:  # More than 3 increases in last 10 iterations
                violations.append(Violation(
                    type='residual_oscillation',
                    severity='medium',
                    description='Residual oscillating/increasing',
                    value=float(oscillations)
                ))

        # Check for stagnation
        if len(residual_history) > 5:
            recent = residual_history[-5:]
            if max(recent) / (min(recent) + 1e-15) < 1.01:  # Less than 1% change
                violations.append(Violation(
                    type='residual_stagnation',
                    severity='medium',
                    description='Residual stagnated',
                    value=float(max(recent))
                ))

        # Check for divergence
        if len(residual_history) > 3:
            if residual_history[-1] > 10 * residual_history[0]:
                violations.append(Violation(
                    type='residual_divergence',
                    severity='high',
                    description='Residual diverging',
                    value=float(residual_history[-1])
                ))

        return violations

    def _check_numerical_stability(
        self,
        output: Dict[str, Any]
    ) -> List[Violation]:
        """Check for NaN or Inf values."""
        violations = []

        all_values = output.get('all_values', [])
        if all_values:
            has_nan = any(np.isnan(val) for val in all_values if isinstance(val, (int, float)))
            has_inf = any(np.isinf(val) for val in all_values if isinstance(val, (int, float)))

            if has_nan or has_inf:
                violations.append(Violation(
                    type='numerical_instability',
                    severity='high',
                    description='NaN or Inf detected in simulation outputs'
                ))

        return violations

    def _check_penetration(
        self,
        output: Dict[str, Any]
    ) -> List[Violation]:
        """Check penetration bounds."""
        violations = []

        penetration_max = output.get('penetration_max', 0.0)
        threshold = output.get('penetration_threshold', 1e-3)

        if penetration_max > threshold:
            violations.append(Violation(
                type='excessive_penetration',
                severity='high',
                description=f'Penetration {penetration_max:.2e} exceeds limit {threshold:.2e}',
                value=float(penetration_max)
            ))

        return violations

    def _check_mesh_quality(
        self,
        output: Dict[str, Any]
    ) -> List[Violation]:
        """Check mesh quality (Jacobian determinant)."""
        violations = []

        jacobian_min = output.get('jacobian_min', 1.0)

        if jacobian_min <= 0:
            violations.append(Violation(
                type='mesh_distortion',
                severity='high',
                description='Inverted elements detected (negative Jacobian)',
                value=float(jacobian_min)
            ))
        elif jacobian_min < 0.1:
            violations.append(Violation(
                type='mesh_quality_degradation',
                severity='medium',
                description=f'Poor mesh quality (min Jacobian: {jacobian_min:.2e})',
                value=float(jacobian_min)
            ))

        return violations

    def _check_contact_detection(
        self,
        output: Dict[str, Any]
    ) -> List[Violation]:
        """Check contact detection."""
        violations = []

        contact_pairs = output.get('contact_pairs', 0)
        expected_contact = output.get('expected_contact', True)

        if expected_contact and contact_pairs == 0:
            violations.append(Violation(
                type='contact_detection_failure',
                severity='medium',
                description='No contact pairs found when contact expected',
                value=0.0
            ))

        return violations

    def update_discovered_constraints(
        self,
        violations: List[Violation],
        iteration: int,
        surrogate_manager: Any
    ) -> None:
        """Update discovered constraints and add surrogates for new failure modes.

        Args:
            violations: List of violations found
            iteration: Current iteration number
            surrogate_manager: Surrogate manager to add new constraint models
        """
        for violation in violations:
            con_type = violation.type

            if con_type not in self.discovered_constraints:
                # New constraint discovered
                self.discovered_constraints[con_type] = {
                    'first_seen': iteration,
                    'frequency': 1,
                    'severity': violation.severity,
                    'description': violation.description
                }

                # Add surrogate model to manager
                surrogate_manager.add_constraint(con_type, constraint_type='binary')

                print(f"[Iteration {iteration}] New constraint discovered: {con_type}")
                print(f"  Severity: {violation.severity}")
                print(f"  Description: {violation.description}")
            else:
                # Existing constraint, increment frequency
                self.discovered_constraints[con_type]['frequency'] += 1

    def get_constraint_labels(
        self,
        outputs: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Extract constraint labels from simulation outputs.

        Args:
            outputs: List of simulation output dictionaries

        Returns:
            Dictionary mapping constraint names to binary labels
        """
        constraint_labels: Dict[str, List[int]] = {
            con_name: [] for con_name in self.discovered_constraints.keys()
        }

        for output in outputs:
            violations = self.check_simulation_output(output)
            violation_types = {v.type for v in violations}

            # Create labels for all discovered constraints
            for con_name in self.discovered_constraints.keys():
                label = 1 if con_name in violation_types else 0
                constraint_labels[con_name].append(label)

        # Convert to numpy arrays
        return {
            name: np.array(labels).reshape(-1, 1)
            for name, labels in constraint_labels.items()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of discovered constraints.

        Returns:
            Summary dictionary
        """
        return {
            'total_constraints': len(self.discovered_constraints),
            'constraints': self.discovered_constraints,
            'constraint_types': list(self.discovered_constraints.keys())
        }

    def has_new_constraints_since(self, iteration: int) -> bool:
        """Check if new constraints discovered since given iteration.

        Args:
            iteration: Iteration to check from

        Returns:
            True if new constraints found since iteration
        """
        for con_info in self.discovered_constraints.values():
            if con_info['first_seen'] > iteration:
                return True
        return False
