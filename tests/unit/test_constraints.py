"""Tests for parameter constraints system.

This module tests the constraint configuration, pattern matching,
position windows, and constraint application logic.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from peakfit.core.fitting.constraints import (
    ConstraintResolver,
    ParameterConfig,
    ParameterConstraint,
    ParameterDefaults,
    PeakConstraints,
    PositionWindowConfig,
    ResolvedConstraint,
    apply_constraints,
    constraints_from_cli,
)
from peakfit.core.fitting.parameters import (
    ParameterId,
    Parameters,
)


class TestParameterConstraint:
    """Tests for ParameterConstraint model."""

    def test_empty_constraint(self) -> None:
        """Test that empty constraint reports as empty."""
        constraint = ParameterConstraint()
        assert constraint.is_empty()

    def test_non_empty_constraint_value(self) -> None:
        """Test constraint with value is not empty."""
        constraint = ParameterConstraint(value=1.0)
        assert not constraint.is_empty()

    def test_non_empty_constraint_bounds(self) -> None:
        """Test constraint with bounds is not empty."""
        constraint = ParameterConstraint(min=0.0, max=10.0)
        assert not constraint.is_empty()

    def test_non_empty_constraint_vary(self) -> None:
        """Test constraint with vary is not empty."""
        constraint = ParameterConstraint(vary=False)
        assert not constraint.is_empty()

    def test_full_constraint(self) -> None:
        """Test constraint with all fields."""
        constraint = ParameterConstraint(
            value=5.0,
            min=0.0,
            max=10.0,
            vary=True,
        )
        assert constraint.value == 5.0
        assert constraint.min == 0.0
        assert constraint.max == 10.0
        assert constraint.vary is True


class TestPositionWindowConfig:
    """Tests for PositionWindowConfig model."""

    def test_empty_config(self) -> None:
        """Test empty position window config."""
        config = PositionWindowConfig()
        assert config.F1 is None
        assert config.F2 is None
        assert config.F3 is None
        assert config.F4 is None

    def test_get_method(self) -> None:
        """Test get method for axis access."""
        config = PositionWindowConfig(F2=0.5, F3=0.05)
        assert config.get("F2") == 0.5
        assert config.get("F3") == 0.05
        assert config.get("F1") is None

    def test_dict_access(self) -> None:
        """Test dict-like access."""
        config = PositionWindowConfig(F2=0.5)
        assert config["F2"] == 0.5
        assert config["F1"] is None


class TestPeakConstraints:
    """Tests for PeakConstraints model."""

    def test_basic_peak_constraints(self) -> None:
        """Test basic peak constraints."""
        constraints = PeakConstraints(
            position_window=0.1,
        )
        assert constraints.position_window == 0.1

    def test_peak_with_parameter_constraints(self) -> None:
        """Test peak constraints with parameter-level constraints."""
        constraints = PeakConstraints(
            parameters={
                "F2.cs": ParameterConstraint(vary=False),
                "F3.lw": ParameterConstraint(min=10.0, max=50.0),
            }
        )
        assert "F2.cs" in constraints.parameters
        assert constraints.parameters["F2.cs"].vary is False
        assert constraints.parameters["F3.lw"].min == 10.0

    def test_inline_constraint_parsing(self) -> None:
        """Test that inline constraints are collected properly."""
        # Simulate TOML-style dict with inline constraints
        data = {
            "position_window": 0.1,
            "F2.cs": {"value": 120.5},
            "F3.lw": {"min": 10.0},
        }
        constraints = PeakConstraints.model_validate(data)
        assert constraints.position_window == 0.1
        assert "F2.cs" in constraints.parameters
        assert constraints.parameters["F2.cs"].value == 120.5


class TestParameterConfig:
    """Tests for ParameterConfig model."""

    def test_empty_config(self) -> None:
        """Test empty parameter config."""
        config = ParameterConfig()
        assert config.position_window is None
        assert len(config.peaks) == 0
        assert config.from_file is None

    def test_global_position_window(self) -> None:
        """Test global position window."""
        config = ParameterConfig(position_window=0.1)
        assert config.position_window == 0.1

    def test_per_axis_windows(self) -> None:
        """Test per-axis position windows."""
        config = ParameterConfig(
            position_windows=PositionWindowConfig(F2=0.5, F3=0.05)
        )
        assert config.position_windows.F2 == 0.5
        assert config.position_windows.F3 == 0.05

    def test_peak_constraints(self) -> None:
        """Test per-peak constraints."""
        config = ParameterConfig(
            peaks={
                "2N-H": PeakConstraints(position_window=0.02),
                "G45N-HN": PeakConstraints(
                    position_windows=PositionWindowConfig(F2=1.0, F3=0.03)
                ),
            }
        )
        assert config.peaks["2N-H"].position_window == 0.02
        assert config.peaks["G45N-HN"].position_windows.F2 == 1.0


class TestResolvedConstraint:
    """Tests for ResolvedConstraint."""

    def test_merge_from(self) -> None:
        """Test merging constraints."""
        resolved = ResolvedConstraint(value=1.0, source="default")

        # Merge constraint with new value
        constraint = ParameterConstraint(value=2.0)
        resolved.merge_from(constraint, "new_source")

        assert resolved.value == 2.0
        assert resolved.source == "new_source"

    def test_merge_preserves_none(self) -> None:
        """Test that None values don't overwrite."""
        resolved = ResolvedConstraint(value=1.0, min=0.0, source="default")

        # Merge constraint with only max set
        constraint = ParameterConstraint(max=10.0)
        resolved.merge_from(constraint, "new_source")

        assert resolved.value == 1.0  # Unchanged
        assert resolved.min == 0.0  # Unchanged
        assert resolved.max == 10.0  # Updated
        assert resolved.source == "new_source"


class TestConstraintResolver:
    """Tests for ConstraintResolver."""

    def test_pattern_matching(self) -> None:
        """Test glob pattern matching."""
        assert ConstraintResolver._matches_pattern("2N-H.F2.cs", "*.*.cs")
        assert ConstraintResolver._matches_pattern("2N-H.F2.cs", "*.F2.*")
        assert ConstraintResolver._matches_pattern("2N-H.F2.cs", "2N-H.*.*")
        assert not ConstraintResolver._matches_pattern("2N-H.F2.cs", "*.F3.*")
        assert not ConstraintResolver._matches_pattern("2N-H.F2.cs", "*.*.lw")

    def test_resolve_with_global_window(self) -> None:
        """Test resolution with global position window."""
        config = ParameterConfig(position_window=0.1)
        resolver = ConstraintResolver(config)

        resolved = resolver.resolve(
            param_name="2N-H.F2.cs",
            peak_name="2N-H",
            axis="F2",
            param_type="cs",
            current_value=120.5,
        )

        assert resolved.min == pytest.approx(120.4)
        assert resolved.max == pytest.approx(120.6)
        assert resolved.source == "global_position_window"

    def test_resolve_with_axis_window(self) -> None:
        """Test resolution with per-axis window."""
        config = ParameterConfig(
            position_window=0.1,  # Global
            position_windows=PositionWindowConfig(F2=0.5),  # Override for F2
        )
        resolver = ConstraintResolver(config)

        resolved = resolver.resolve(
            param_name="2N-H.F2.cs",
            peak_name="2N-H",
            axis="F2",
            param_type="cs",
            current_value=120.5,
        )

        assert resolved.min == pytest.approx(120.0)
        assert resolved.max == pytest.approx(121.0)
        assert resolved.source == "position_windows.F2"

    def test_resolve_with_peak_window(self) -> None:
        """Test resolution with per-peak window."""
        config = ParameterConfig(
            position_window=0.1,
            peaks={
                "2N-H": PeakConstraints(position_window=0.02),
            },
        )
        resolver = ConstraintResolver(config)

        resolved = resolver.resolve(
            param_name="2N-H.F2.cs",
            peak_name="2N-H",
            axis="F2",
            param_type="cs",
            current_value=120.5,
        )

        assert resolved.min == pytest.approx(120.48)
        assert resolved.max == pytest.approx(120.52)
        assert resolved.source == "peaks.2N-H.position_window"

    def test_resolve_with_peak_axis_window(self) -> None:
        """Test resolution with per-peak per-axis window."""
        config = ParameterConfig(
            position_window=0.1,
            peaks={
                "2N-H": PeakConstraints(
                    position_window=0.02,  # Per-peak global
                    position_windows=PositionWindowConfig(F2=0.5),  # Per-peak F2
                ),
            },
        )
        resolver = ConstraintResolver(config)

        resolved = resolver.resolve(
            param_name="2N-H.F2.cs",
            peak_name="2N-H",
            axis="F2",
            param_type="cs",
            current_value=120.5,
        )

        # Per-peak per-axis should take priority
        assert resolved.min == pytest.approx(120.0)
        assert resolved.max == pytest.approx(121.0)
        assert resolved.source == "peaks.2N-H.position_windows.F2"

    def test_resolve_pattern_defaults(self) -> None:
        """Test resolution with pattern-based defaults."""
        config = ParameterConfig(
            defaults=ParameterDefaults(
                patterns={
                    "*.*.lw": ParameterConstraint(min=5.0, max=100.0),
                }
            )
        )
        resolver = ConstraintResolver(config)

        resolved = resolver.resolve(
            param_name="2N-H.F2.lw",
            peak_name="2N-H",
            axis="F2",
            param_type="lw",
            current_value=25.0,
        )

        assert resolved.min == pytest.approx(5.0)
        assert resolved.max == pytest.approx(100.0)

    def test_resolve_with_vary_constraint(self) -> None:
        """Test resolution with vary constraint."""
        config = ParameterConfig(
            peaks={
                "2N-H": PeakConstraints(
                    parameters={
                        "F2.cs": ParameterConstraint(vary=False),
                    }
                ),
            }
        )
        resolver = ConstraintResolver(config)

        resolved = resolver.resolve(
            param_name="2N-H.F2.cs",
            peak_name="2N-H",
            axis="F2",
            param_type="cs",
            current_value=120.5,
        )

        assert resolved.vary is False

    def test_load_from_json_file(self) -> None:
        """Test loading parameter values from JSON file."""
        # Create a mock fit_summary.json
        fit_summary = {
            "clusters": [
                {
                    "lineshape_parameters": [
                        {"name": "2N-H.F2.cs", "value": 120.6},
                        {"name": "2N-H.F2.lw", "value": 28.5},
                    ],
                    "amplitudes": [
                        {"peak_name": "2N-H", "plane_index": 0, "value": 1000.0},
                    ],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(fit_summary, f)
            temp_path = Path(f.name)

        try:
            config = ParameterConfig(from_file=temp_path)
            resolver = ConstraintResolver(config)

            # The value from file should be used
            resolved = resolver.resolve(
                param_name="2N-H.F2.cs",
                peak_name="2N-H",
                axis="F2",
                param_type="cs",
                current_value=120.5,  # Original value
            )

            assert resolved.value == pytest.approx(120.6)  # From file
            assert resolved.source == "from_file"
        finally:
            temp_path.unlink()


class TestApplyConstraints:
    """Tests for apply_constraints function."""

    def _create_test_params(self) -> Parameters:
        """Create test parameters."""
        params = Parameters()

        # Add position parameters
        pos_id = ParameterId.position("2N-H", "F2")
        params.add(pos_id, value=120.5, min=119.5, max=121.5)

        pos_id2 = ParameterId.position("2N-H", "F3")
        params.add(pos_id2, value=8.45, min=8.35, max=8.55)

        # Add linewidth parameters
        lw_id = ParameterId.linewidth("2N-H", "F2")
        params.add(lw_id, value=25.0, min=0.1, max=200.0)

        return params

    def test_apply_position_window(self) -> None:
        """Test applying position window constraints."""
        params = self._create_test_params()

        config = ParameterConfig(position_window=0.1)
        apply_constraints(params, config)

        # Check that position bounds were updated
        cs_param = params["2N-H.F2.cs"]
        assert cs_param.min == pytest.approx(120.4)
        assert cs_param.max == pytest.approx(120.6)

    def test_apply_vary_constraint(self) -> None:
        """Test applying vary=False constraint."""
        params = self._create_test_params()

        config = ParameterConfig(
            peaks={
                "2N-H": PeakConstraints(
                    parameters={
                        "F2.cs": ParameterConstraint(vary=False),
                    }
                )
            }
        )
        apply_constraints(params, config)

        # Check that vary was set to False
        cs_param = params["2N-H.F2.cs"]
        assert cs_param.vary is False

        # Other parameters should still vary
        lw_param = params["2N-H.F2.lw"]
        assert lw_param.vary is True

    def test_value_clamping(self) -> None:
        """Test that values are clamped to bounds."""
        params = Parameters()
        pos_id = ParameterId.position("2N-H", "F2")
        params.add(pos_id, value=120.5, min=119.0, max=122.0)

        # Apply tight window that excludes current value
        config = ParameterConfig(
            peaks={
                "2N-H": PeakConstraints(
                    parameters={
                        "F2.cs": ParameterConstraint(min=121.0, max=122.0),
                    }
                )
            }
        )
        apply_constraints(params, config)

        # Value should be clamped to min
        cs_param = params["2N-H.F2.cs"]
        assert cs_param.value == pytest.approx(121.0)


class TestConstraintsFromCLI:
    """Tests for constraints_from_cli function."""

    def test_basic_cli_constraints(self) -> None:
        """Test creating constraints from CLI options."""
        config = constraints_from_cli(
            position_window=0.1,
            position_window_f2=0.5,
            position_window_f3=0.05,
        )

        assert config.position_window == 0.1
        assert config.position_windows.F2 == 0.5
        assert config.position_windows.F3 == 0.05

    def test_fix_patterns(self) -> None:
        """Test CLI fix patterns."""
        config = constraints_from_cli(
            fix_patterns=["*.*.cs", "*.*.eta"],
        )

        assert "*.*.cs" in config.defaults.patterns
        assert config.defaults.patterns["*.*.cs"].vary is False
        assert "*.*.eta" in config.defaults.patterns
        assert config.defaults.patterns["*.*.eta"].vary is False

    def test_vary_patterns(self) -> None:
        """Test CLI vary patterns."""
        config = constraints_from_cli(
            vary_patterns=["*.*.lw"],
        )

        assert "*.*.lw" in config.defaults.patterns
        assert config.defaults.patterns["*.*.lw"].vary is True

    def test_from_file(self) -> None:
        """Test CLI from_file option."""
        config = constraints_from_cli(
            from_file=Path("/some/path/fit_summary.json"),
        )

        assert config.from_file == Path("/some/path/fit_summary.json")
