"""Tests for constraint discovery module."""

import pytest
import numpy as np
from shebo.core.constraint_discovery import ConstraintDiscovery, Violation


class TestConstraintDiscovery:
    """Test constraint discovery functionality."""

    def test_initialization(self):
        """Test constraint discovery can be initialized."""
        discovery = ConstraintDiscovery()
        assert discovery is not None
        assert len(discovery.discovered_constraints) == 0

    def test_check_residual_oscillation(self):
        """Test detection of residual oscillation."""
        discovery = ConstraintDiscovery()

        # Create output with oscillating residuals
        residuals = [1.0, 0.5, 0.4, 0.5, 0.3, 0.4, 0.2, 0.3, 0.1, 0.2, 0.1, 0.15]
        output = {
            'residual_history': residuals,
            'convergence_status': False,
            'all_values': residuals
        }

        violations = discovery.check_simulation_output(output)

        # Should detect oscillation
        violation_types = [v.type for v in violations]
        assert 'residual_oscillation' in violation_types

    def test_check_numerical_instability(self):
        """Test detection of NaN/Inf values."""
        discovery = ConstraintDiscovery()

        output = {
            'residual_history': [1.0, 0.5, 0.1],
            'all_values': [1.0, 0.5, np.nan, 0.1],
            'convergence_status': False
        }

        violations = discovery.check_simulation_output(output)

        violation_types = [v.type for v in violations]
        assert 'numerical_instability' in violation_types

    def test_check_excessive_penetration(self):
        """Test detection of excessive penetration."""
        discovery = ConstraintDiscovery()

        output = {
            'residual_history': [1.0, 0.5],
            'penetration_max': 0.01,  # Exceeds 1e-3 threshold
            'all_values': [1.0, 0.5]
        }

        violations = discovery.check_simulation_output(output)

        violation_types = [v.type for v in violations]
        assert 'excessive_penetration' in violation_types

    def test_check_mesh_distortion(self):
        """Test detection of mesh distortion."""
        discovery = ConstraintDiscovery()

        output = {
            'residual_history': [1.0, 0.5],
            'jacobian_min': -0.1,  # Negative jacobian
            'all_values': [1.0, 0.5]
        }

        violations = discovery.check_simulation_output(output)

        violation_types = [v.type for v in violations]
        assert 'mesh_distortion' in violation_types

    def test_check_contact_detection_failure(self):
        """Test detection of contact detection failure."""
        discovery = ConstraintDiscovery()

        output = {
            'residual_history': [1.0, 0.5],
            'contact_pairs': 0,
            'expected_contact': True,
            'all_values': [1.0, 0.5]
        }

        violations = discovery.check_simulation_output(output)

        violation_types = [v.type for v in violations]
        assert 'contact_detection_failure' in violation_types

    def test_update_discovered_constraints(self):
        """Test updating discovered constraints."""
        # Mock surrogate manager
        class MockSurrogateManager:
            def __init__(self):
                self.constraints_added = []

            def add_constraint(self, name, constraint_type):
                self.constraints_added.append((name, constraint_type))

        discovery = ConstraintDiscovery()
        manager = MockSurrogateManager()

        # Create violation
        violations = [
            Violation(
                type='residual_oscillation',
                severity='medium',
                description='Residual oscillating'
            )
        ]

        discovery.update_discovered_constraints(violations, iteration=10, surrogate_manager=manager)

        # Check constraint was discovered
        assert 'residual_oscillation' in discovery.discovered_constraints
        assert discovery.discovered_constraints['residual_oscillation']['first_seen'] == 10
        assert discovery.discovered_constraints['residual_oscillation']['frequency'] == 1

        # Check surrogate was added
        assert ('residual_oscillation', 'binary') in manager.constraints_added

    def test_constraint_frequency_increment(self):
        """Test that constraint frequency increments."""
        class MockSurrogateManager:
            def add_constraint(self, name, constraint_type):
                pass

        discovery = ConstraintDiscovery()
        manager = MockSurrogateManager()

        violations = [
            Violation(
                type='residual_oscillation',
                severity='medium',
                description='Residual oscillating'
            )
        ]

        # First occurrence
        discovery.update_discovered_constraints(violations, iteration=10, surrogate_manager=manager)
        assert discovery.discovered_constraints['residual_oscillation']['frequency'] == 1

        # Second occurrence
        discovery.update_discovered_constraints(violations, iteration=20, surrogate_manager=manager)
        assert discovery.discovered_constraints['residual_oscillation']['frequency'] == 2

    def test_get_constraint_labels(self):
        """Test getting constraint labels from outputs."""
        discovery = ConstraintDiscovery()

        # Create some outputs
        outputs = [
            {'residual_history': [1.0, 0.5], 'all_values': [1.0, 0.5]},
            {'residual_history': [1.0, 0.5, 0.4, 0.5], 'all_values': [1.0, 0.5]},  # Will oscillate
            {'residual_history': [1.0, 0.5], 'penetration_max': 0.01, 'all_values': [1.0, 0.5]}
        ]

        # First discover constraints
        class MockSurrogateManager:
            def add_constraint(self, name, constraint_type):
                pass

        manager = MockSurrogateManager()

        for i, output in enumerate(outputs):
            violations = discovery.check_simulation_output(output)
            discovery.update_discovered_constraints(violations, iteration=i, surrogate_manager=manager)

        # Get labels
        labels = discovery.get_constraint_labels(outputs)

        assert isinstance(labels, dict)
        for con_name, con_labels in labels.items():
            assert isinstance(con_labels, np.ndarray)
            assert len(con_labels) == len(outputs)
            assert all(label in [0, 1] for label in con_labels.flatten())

    def test_has_new_constraints_since(self):
        """Test checking for new constraints."""
        class MockSurrogateManager:
            def add_constraint(self, name, constraint_type):
                pass

        discovery = ConstraintDiscovery()
        manager = MockSurrogateManager()

        # Add constraint at iteration 10
        violations = [Violation(type='test_constraint', severity='medium', description='test')]
        discovery.update_discovered_constraints(violations, iteration=10, surrogate_manager=manager)

        # Check
        assert discovery.has_new_constraints_since(5) == True  # New since iteration 5
        assert discovery.has_new_constraints_since(15) == False  # None since iteration 15

    def test_get_summary(self):
        """Test getting summary of discovered constraints."""
        discovery = ConstraintDiscovery()

        class MockSurrogateManager:
            def add_constraint(self, name, constraint_type):
                pass

        manager = MockSurrogateManager()

        # Add some constraints
        violations = [
            Violation(type='constraint1', severity='high', description='test1'),
            Violation(type='constraint2', severity='low', description='test2')
        ]

        for v in violations:
            discovery.update_discovered_constraints([v], iteration=10, surrogate_manager=manager)

        summary = discovery.get_summary()

        assert summary['total_constraints'] == 2
        assert 'constraint1' in summary['constraints']
        assert 'constraint2' in summary['constraints']
        assert len(summary['constraint_types']) == 2
