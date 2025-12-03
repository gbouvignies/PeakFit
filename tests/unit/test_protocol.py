"""Tests for multi-step fitting protocol system.

This module tests the protocol configuration, step execution,
and pattern-based parameter control.
"""

from __future__ import annotations

import pytest

from peakfit.core.fitting.parameters import (
    ParameterId,
    Parameters,
)
from peakfit.core.fitting.protocol import (
    FitProtocol,
    FitStep,
    apply_step_constraints,
    create_protocol_from_config,
)


class TestFitStep:
    """Tests for FitStep model."""

    def test_default_step(self) -> None:
        """Test default step values."""
        step = FitStep()
        assert step.name == ""
        assert step.fix == []
        assert step.vary == []
        assert step.iterations == 1
        assert step.description == ""

    def test_step_with_patterns(self) -> None:
        """Test step with fix/vary patterns."""
        step = FitStep(
            name="fix_positions",
            fix=["*.*.cs"],
            vary=["*.*.lw"],
            iterations=2,
            description="Fix positions, vary linewidths",
        )
        assert step.name == "fix_positions"
        assert "*.*.cs" in step.fix
        assert "*.*.lw" in step.vary
        assert step.iterations == 2

    def test_step_validation(self) -> None:
        """Test step validation."""
        # iterations must be >= 1
        with pytest.raises(ValueError):
            FitStep(iterations=0)


class TestFitProtocol:
    """Tests for FitProtocol model."""

    def test_empty_protocol(self) -> None:
        """Test empty protocol."""
        protocol = FitProtocol()
        assert protocol.is_empty()
        assert len(protocol.steps) == 0

    def test_protocol_with_steps(self) -> None:
        """Test protocol with steps."""
        protocol = FitProtocol(
            steps=[
                FitStep(name="step1"),
                FitStep(name="step2"),
            ]
        )
        assert not protocol.is_empty()
        assert len(protocol.steps) == 2

    def test_default_protocol(self) -> None:
        """Test default protocol creation."""
        protocol = FitProtocol.default(refine_iterations=3)
        assert len(protocol.steps) == 1
        assert protocol.steps[0].name == "default"
        assert protocol.steps[0].iterations == 3
        assert "*" in protocol.steps[0].vary

    def test_positions_then_full_protocol(self) -> None:
        """Test predefined positions-then-full protocol."""
        protocol = FitProtocol.positions_then_full()
        assert len(protocol.steps) == 2

        # First step fixes positions
        assert protocol.steps[0].name == "fix_positions"
        assert "*.*.cs" in protocol.steps[0].fix

        # Second step varies all
        assert protocol.steps[1].name == "full_optimization"
        assert "*" in protocol.steps[1].vary


class TestApplyStepConstraints:
    """Tests for apply_step_constraints function."""

    def _create_test_params(self) -> Parameters:
        """Create test parameters."""
        params = Parameters()

        # Add various parameters for peak "2N-H"
        params.add(
            ParameterId.position("2N-H", "F2"),
            value=120.5,
            min=119.5,
            max=121.5,
        )
        params.add(
            ParameterId.position("2N-H", "F3"),
            value=8.45,
            min=8.35,
            max=8.55,
        )
        params.add(
            ParameterId.linewidth("2N-H", "F2"),
            value=25.0,
            min=0.1,
            max=200.0,
        )
        params.add(
            ParameterId.linewidth("2N-H", "F3"),
            value=15.0,
            min=0.1,
            max=200.0,
        )
        params.add(
            ParameterId.fraction("2N-H", "F2"),
            value=0.5,
            min=0.0,
            max=1.0,
        )

        return params

    def test_fix_all_cs(self) -> None:
        """Test fixing all chemical shift parameters."""
        params = self._create_test_params()

        step = FitStep(fix=["*.*.cs"])
        apply_step_constraints(params, step)

        # CS parameters should be fixed
        assert params["2N-H.F2.cs"].vary is False
        assert params["2N-H.F3.cs"].vary is False

        # Other parameters should still vary
        assert params["2N-H.F2.lw"].vary is True
        assert params["2N-H.F3.lw"].vary is True
        assert params["2N-H.F2.eta"].vary is True

    def test_vary_only_lw(self) -> None:
        """Test varying only linewidth parameters."""
        params = self._create_test_params()

        # Fix everything, then vary linewidths
        step = FitStep(
            fix=["*"],
            vary=["*.*.lw"],
        )
        apply_step_constraints(params, step)

        # LW parameters should vary
        assert params["2N-H.F2.lw"].vary is True
        assert params["2N-H.F3.lw"].vary is True

        # Other parameters should be fixed
        assert params["2N-H.F2.cs"].vary is False
        assert params["2N-H.F3.cs"].vary is False
        assert params["2N-H.F2.eta"].vary is False

    def test_fix_specific_axis(self) -> None:
        """Test fixing parameters for specific axis."""
        params = self._create_test_params()

        step = FitStep(fix=["*.F2.*"])
        apply_step_constraints(params, step)

        # F2 parameters should be fixed
        assert params["2N-H.F2.cs"].vary is False
        assert params["2N-H.F2.lw"].vary is False
        assert params["2N-H.F2.eta"].vary is False

        # F3 parameters should still vary
        assert params["2N-H.F3.cs"].vary is True
        assert params["2N-H.F3.lw"].vary is True

    def test_vary_overrides_fix(self) -> None:
        """Test that vary patterns override fix patterns."""
        params = self._create_test_params()

        # Fix all, then vary CS
        step = FitStep(
            fix=["*"],
            vary=["*.*.cs"],
        )
        apply_step_constraints(params, step)

        # CS should vary (vary applied after fix)
        assert params["2N-H.F2.cs"].vary is True
        assert params["2N-H.F3.cs"].vary is True

        # Others should be fixed
        assert params["2N-H.F2.lw"].vary is False
        assert params["2N-H.F2.eta"].vary is False

    def test_computed_params_unchanged(self) -> None:
        """Test that computed parameters are not modified."""
        params = Parameters()

        # Add a computed parameter (like amplitude)
        params.add(
            ParameterId.amplitude("2N-H", "F1", 0),
            value=1000.0,
            vary=False,
            computed=True,
        )

        step = FitStep(vary=["*"])
        apply_step_constraints(params, step)

        # Computed parameter should remain not varying
        assert params["2N-H.F1.I0"].vary is False


class TestCreateProtocolFromConfig:
    """Tests for create_protocol_from_config function."""

    def test_with_explicit_steps(self) -> None:
        """Test protocol creation with explicit steps."""
        steps = [
            FitStep(name="step1", iterations=1),
            FitStep(name="step2", iterations=2),
        ]
        protocol = create_protocol_from_config(steps=steps)

        assert len(protocol.steps) == 2
        assert protocol.steps[0].name == "step1"
        assert protocol.steps[1].name == "step2"

    def test_with_legacy_refine_iterations(self) -> None:
        """Test protocol creation with legacy refine_iterations."""
        protocol = create_protocol_from_config(
            steps=None,
            refine_iterations=3,
            fixed=False,
        )

        assert len(protocol.steps) == 1
        assert protocol.steps[0].iterations == 4  # refine_iterations + 1

    def test_with_legacy_fixed(self) -> None:
        """Test protocol creation with legacy --fixed flag."""
        protocol = create_protocol_from_config(
            steps=None,
            refine_iterations=2,
            fixed=True,
        )

        assert len(protocol.steps) == 1
        assert "*.*.cs" in protocol.steps[0].fix
        assert protocol.steps[0].iterations == 3  # refine_iterations + 1


class TestProtocolIntegration:
    """Integration tests for protocol system."""

    def test_multi_step_protocol_simulation(self) -> None:
        """Simulate a multi-step protocol execution."""
        params = Parameters()

        # Add parameters
        params.add(
            ParameterId.position("Peak1", "F2"),
            value=120.0,
            vary=True,
        )
        params.add(
            ParameterId.linewidth("Peak1", "F2"),
            value=25.0,
            vary=True,
        )

        # Define protocol
        protocol = FitProtocol(
            steps=[
                FitStep(
                    name="fix_positions",
                    fix=["*.*.cs"],
                    iterations=1,
                ),
                FitStep(
                    name="full",
                    vary=["*"],
                    iterations=1,
                ),
            ]
        )

        # Step 1: Fix positions
        apply_step_constraints(params, protocol.steps[0])
        assert params["Peak1.F2.cs"].vary is False
        assert params["Peak1.F2.lw"].vary is True

        # Step 2: Vary all
        apply_step_constraints(params, protocol.steps[1])
        assert params["Peak1.F2.cs"].vary is True
        assert params["Peak1.F2.lw"].vary is True

    def test_protocol_toml_like_config(self) -> None:
        """Test protocol from TOML-like configuration dict."""
        # Simulate what would come from TOML parsing
        steps_config = [
            {
                "name": "linewidths_only",
                "fix": ["*.*.cs", "*.*.eta"],
                "vary": ["*.*.lw"],
                "iterations": 1,
            },
            {
                "name": "all_params",
                "vary": ["*"],
                "iterations": 2,
            },
        ]

        # Parse into FitStep objects
        steps = [FitStep.model_validate(s) for s in steps_config]
        protocol = FitProtocol(steps=steps)

        assert len(protocol.steps) == 2
        assert protocol.steps[0].name == "linewidths_only"
        assert "*.*.cs" in protocol.steps[0].fix
        assert protocol.steps[1].iterations == 2
