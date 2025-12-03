"""Test Pydantic models."""

from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from peakfit.core.domain.config import (
    ClusterConfig,
    FitConfig,
    FitResult,
    LineshapeName,
    OutputConfig,
    OutputFormat,
    PeakData,
    PeakFitConfig,
)

VALID_LINESHAPES: tuple[LineshapeName, ...] = (
    "auto",
    "gaussian",
    "lorentzian",
    "pvoigt",
    "sp1",
    "sp2",
    "no_apod",
)

VALID_FORMATS: list[OutputFormat] = ["csv", "json", "txt"]


class TestFitConfig:
    """Tests for FitConfig model."""

    def test_default_values(self):
        """FitConfig should have sensible defaults."""
        config = FitConfig()
        assert config.lineshape == "auto"
        assert config.refine_iterations == 1
        assert config.fix_positions is False
        assert config.max_iterations == 1000

    def test_valid_lineshapes(self):
        """FitConfig should accept valid lineshape values."""
        for lineshape in VALID_LINESHAPES:
            config = FitConfig(lineshape=lineshape)
            assert config.lineshape == lineshape

    def test_invalid_lineshape(self):
        """FitConfig should reject invalid lineshape values."""
        with pytest.raises(ValidationError):
            FitConfig(lineshape=cast("LineshapeName", "invalid"))

    def test_refine_iterations_bounds(self):
        """FitConfig should validate refine_iterations bounds."""
        # Valid range
        config = FitConfig(refine_iterations=0)
        assert config.refine_iterations == 0

        config = FitConfig(refine_iterations=20)
        assert config.refine_iterations == 20

        # Invalid range
        with pytest.raises(ValidationError):
            FitConfig(refine_iterations=-1)

        with pytest.raises(ValidationError):
            FitConfig(refine_iterations=21)

    def test_tolerance_must_be_positive(self):
        """FitConfig tolerance must be positive."""
        with pytest.raises(ValidationError):
            FitConfig(tolerance=0)

        with pytest.raises(ValidationError):
            FitConfig(tolerance=-1e-8)

    def test_extra_fields_forbidden(self):
        """FitConfig should reject unknown fields."""
        with pytest.raises(ValidationError):
            FitConfig.model_validate({"unknown_field": "value"})


class TestClusterConfig:
    """Tests for ClusterConfig model."""

    def test_default_values(self):
        """ClusterConfig should have sensible defaults."""
        config = ClusterConfig()
        assert config.contour_factor == 5.0
        assert config.contour_level is None

    def test_contour_factor_positive(self):
        """ClusterConfig contour_factor must be positive."""
        with pytest.raises(ValidationError):
            ClusterConfig(contour_factor=0)

        with pytest.raises(ValidationError):
            ClusterConfig(contour_factor=-1.0)


class TestOutputConfig:
    """Tests for OutputConfig model."""

    def test_default_values(self):
        """OutputConfig should have sensible defaults."""
        config = OutputConfig()
        assert config.directory == Path("Fits")
        # Default formats include all structured outputs plus legacy txt
        assert config.formats == ["json", "csv", "txt"]
        assert config.save_simulated is True

    def test_valid_formats(self):
        """OutputConfig should accept valid format values."""
        config = OutputConfig(formats=VALID_FORMATS)
        assert config.formats == VALID_FORMATS

    def test_invalid_format(self):
        """OutputConfig should reject invalid format values."""
        with pytest.raises(ValidationError):
            OutputConfig(formats=cast("list[OutputFormat]", ["invalid"]))


class TestPeakFitConfig:
    """Tests for main PeakFitConfig model."""

    def test_default_values(self):
        """PeakFitConfig should have sensible defaults."""
        config = PeakFitConfig()
        assert isinstance(config.fitting, FitConfig)
        assert isinstance(config.clustering, ClusterConfig)
        assert isinstance(config.output, OutputConfig)
        assert config.noise_level is None
        assert config.exclude_planes == []

    def test_nested_config(self):
        """PeakFitConfig should properly nest configs."""
        config = PeakFitConfig(
            fitting=FitConfig(lineshape="gaussian", refine_iterations=3),
            clustering=ClusterConfig(contour_factor=10.0),
            output=OutputConfig(directory=Path("MyResults")),
        )
        assert config.fitting.lineshape == "gaussian"
        assert config.fitting.refine_iterations == 3
        assert config.clustering.contour_factor == 10.0
        assert config.output.directory == Path("MyResults")

    def test_exclude_planes_validation(self):
        """PeakFitConfig should validate exclude_planes."""
        config = PeakFitConfig(exclude_planes=[0, 5, 10])
        assert config.exclude_planes == [0, 5, 10]

        # Should sort and deduplicate
        config = PeakFitConfig(exclude_planes=[10, 5, 5, 0])
        assert config.exclude_planes == [0, 5, 10]

        # Should reject negative indices
        with pytest.raises(ValidationError):
            PeakFitConfig(exclude_planes=[-1, 0, 5])

    def test_noise_level_positive(self):
        """PeakFitConfig noise_level must be positive if set."""
        config = PeakFitConfig(noise_level=100.0)
        assert config.noise_level == 100.0

        with pytest.raises(ValidationError):
            PeakFitConfig(noise_level=-10.0)

        with pytest.raises(ValidationError):
            PeakFitConfig(noise_level=0)


class TestPeakData:
    """Tests for PeakData model."""

    def test_basic_peak(self):
        """PeakData should store peak information."""
        peak = PeakData(name="Peak1", positions=[8.5, 120.5])
        assert peak.name == "Peak1"
        assert peak.positions[0] == 8.5
        assert peak.positions[1] == 120.5
        assert peak.cluster_id is None

    def test_peak_with_z(self):
        """PeakData should handle 3D peak positions."""
        peak = PeakData(name="Peak1", positions=[8.5, 120.5, 10.0])
        assert peak.positions[2] == 10.0

    def test_peak_with_cluster(self):
        """PeakData should track cluster assignment."""
        peak = PeakData(name="Peak1", positions=[8.5, 120.5], cluster_id=3)
        assert peak.cluster_id == 3


class TestFitResult:
    """Tests for FitResult model."""

    def test_successful_fit(self):
        """FitResult should store successful fit information."""
        result = FitResult(
            cluster_id=1,
            peaks=[],
            residual_norm=0.05,
            n_iterations=25,
            success=True,
            message="Optimization converged",
        )
        assert result.cluster_id == 1
        assert result.success is True
        assert result.n_iterations == 25

    def test_failed_fit(self):
        """FitResult should handle failed fits."""
        result = FitResult(
            cluster_id=1,
            peaks=[],
            residual_norm=float("inf"),
            n_iterations=1000,
            success=False,
            message="Maximum iterations reached",
        )
        assert result.success is False
        assert "Maximum iterations" in result.message
