"""Edge case tests for PeakFit modernization.

Tests edge cases, error conditions, and boundary conditions.
"""

import pytest

import numpy as np

from peakfit.core.fitting.parameters import Parameter, Parameters, ParameterType
from peakfit.core.lineshapes import gaussian, lorentzian, pvoigt


class TestParameterEdgeCases:
    """Test edge cases for Parameter and Parameters classes."""

    def test_parameter_at_lower_bound(self):
        """Test parameter at lower boundary."""
        p = Parameter("test", value=0.0, min=0.0, max=10.0)
        assert p.is_at_boundary()
        assert p.relative_position() == 0.0

    def test_parameter_at_upper_bound(self):
        """Test parameter at upper boundary."""
        p = Parameter("test", value=10.0, min=0.0, max=10.0)
        assert p.is_at_boundary()
        assert p.relative_position() == 1.0

    def test_parameter_infinite_bounds(self):
        """Test parameter with infinite bounds."""
        p = Parameter("test", value=100.0, min=-np.inf, max=np.inf)
        assert not p.is_at_boundary()
        # With infinite bounds, relative_position returns 0.5 (middle)
        assert p.relative_position() == 0.5

    def test_parameter_negative_values(self):
        """Test parameter with negative values."""
        p = Parameter("test", value=-5.0, min=-10.0, max=0.0)
        assert -10.0 <= p.value <= 0.0

    def test_parameter_very_small_range(self):
        """Test parameter with very small range."""
        p = Parameter("test", value=1e-10, min=0.0, max=1e-9, vary=True)
        assert 0.0 <= p.value <= 1e-9

    def test_parameters_empty_collection(self):
        """Test empty Parameters collection."""
        params = Parameters()
        assert len(params) == 0
        assert params.get_vary_names() == []
        assert len(params.get_vary_values()) == 0

    def test_parameters_all_fixed(self):
        """Test Parameters with all fixed parameters."""
        params = Parameters()
        params.add("p1", value=1.0, vary=False)
        params.add("p2", value=2.0, vary=False)
        params.add("p3", value=3.0, vary=False)

        assert len(params) == 3
        assert len(params.get_vary_names()) == 0

    def test_parameters_duplicate_name(self):
        """Test adding parameter with duplicate name."""
        params = Parameters()
        params.add("test", value=1.0)
        params.add("test", value=2.0)  # Should overwrite
        assert params["test"].value == 2.0
        assert len(params) == 1

    def test_parameter_with_all_types(self):
        """Test creating parameters with all NMR types."""
        # Use appropriate values for each type
        type_values = {
            ParameterType.POSITION: 100.0,
            ParameterType.FWHM: 10.0,
            ParameterType.FRACTION: 0.5,  # Must be in [0, 1]
            ParameterType.PHASE: 90.0,
            ParameterType.JCOUPLING: 10.0,  # Must be in [0, 20]
            ParameterType.AMPLITUDE: 1000.0,
            ParameterType.GENERIC: 100.0,  # No specific bounds
        }

        for param_type in ParameterType:
            value = type_values.get(param_type, 100.0)  # Default to 100.0
            p = Parameter(f"test_{param_type.name}", value=value, param_type=param_type)
            assert p.param_type == param_type
            # Parameter type is set correctly
            assert isinstance(p.param_type, ParameterType)


class TestLineshapeEdgeCases:
    """Test edge cases for lineshape functions."""

    @pytest.fixture(autouse=True)
    def reset_backend(self):
        """Backend selection removed - this fixture is now a no-op."""
        yield

    def test_gaussian_zero_fwhm(self):
        """Test Gaussian with zero FWHM (should raise or return delta function)."""
        x = np.array([0.0, 0.1, 1.0])
        # Zero FWHM might cause division by zero
        with pytest.warns(RuntimeWarning):  # NumPy warns about division by zero
            y = gaussian(x, fwhm=0.0)
            assert np.any(np.isnan(y)) or np.any(np.isinf(y))

    def test_gaussian_negative_fwhm(self):
        """Test Gaussian with negative FWHM (physically meaningless)."""
        x = np.linspace(-5, 5, 100)
        # Negative FWHM squared is still positive, so this might work
        y = gaussian(x, fwhm=-2.0)
        assert np.all(np.isfinite(y))

    def test_gaussian_very_large_fwhm(self):
        """Test Gaussian with very large FWHM."""
        x = np.linspace(-10, 10, 100)
        y = gaussian(x, fwhm=1000.0)
        # Should be almost flat
        assert np.allclose(y, y[0], atol=0.01)

    def test_gaussian_very_small_fwhm(self):
        """Test Gaussian with very small FWHM."""
        x = np.linspace(-0.001, 0.001, 1000)  # Need to be very close to center
        y = gaussian(x, fwhm=1e-6)
        # Should be very peaked, but might underflow to zero
        # Just check it's finite and non-negative
        assert np.all(np.isfinite(y))
        assert np.all(y >= 0.0)

    def test_lorentzian_zero_fwhm(self):
        """Test Lorentzian with zero FWHM."""
        x = np.array([0.0, 0.1, 1.0])
        with pytest.warns(RuntimeWarning):  # Division by zero
            y = lorentzian(x, fwhm=0.0)
            # Division by zero produces NaN or Inf
            assert np.any(np.isnan(y)) or np.any(np.isinf(y))

    def test_pvoigt_eta_boundaries(self):
        """Test Pseudo-Voigt with eta at boundaries."""
        x = np.linspace(-5, 5, 100)

        # eta = 0 (pure Gaussian)
        y0 = pvoigt(x, fwhm=2.0, eta=0.0)
        y_gauss = gaussian(x, fwhm=2.0)
        assert np.allclose(y0, y_gauss)

        # eta = 1 (pure Lorentzian)
        y1 = pvoigt(x, fwhm=2.0, eta=1.0)
        y_lorentz = lorentzian(x, fwhm=2.0)
        assert np.allclose(y1, y_lorentz)

    def test_pvoigt_eta_out_of_range(self):
        """Test Pseudo-Voigt with eta outside [0, 1]."""
        x = np.linspace(-5, 5, 100)

        # eta < 0
        y_neg = pvoigt(x, fwhm=2.0, eta=-0.5)
        assert np.all(np.isfinite(y_neg))

        # eta > 1
        y_large = pvoigt(x, fwhm=2.0, eta=1.5)
        assert np.all(np.isfinite(y_large))

    def test_lineshape_with_empty_array(self):
        """Test lineshapes with empty array."""
        x = np.array([])
        y_gauss = gaussian(x, fwhm=2.0)
        y_lorentz = lorentzian(x, fwhm=2.0)
        y_pv = pvoigt(x, fwhm=2.0, eta=0.5)

        assert len(y_gauss) == 0
        assert len(y_lorentz) == 0
        assert len(y_pv) == 0

    def test_lineshape_with_single_point(self):
        """Test lineshapes with single point."""
        x = np.array([0.0])
        y_gauss = gaussian(x, fwhm=2.0)
        y_lorentz = lorentzian(x, fwhm=2.0)
        y_pv = pvoigt(x, fwhm=2.0, eta=0.5)

        assert len(y_gauss) == 1
        assert len(y_lorentz) == 1
        assert len(y_pv) == 1
        assert y_gauss[0] > 0.99
        assert y_lorentz[0] > 0.99

    def test_lineshape_with_nan_input(self):
        """Test lineshapes with NaN input."""
        x = np.array([0.0, np.nan, 1.0])
        y_gauss = gaussian(x, fwhm=2.0)
        assert np.isnan(y_gauss[1])
        assert not np.isnan(y_gauss[0])
        assert not np.isnan(y_gauss[2])

    def test_lineshape_with_inf_input(self):
        """Test lineshapes with infinite input."""
        x = np.array([0.0, np.inf, 1.0])
        y_gauss = gaussian(x, fwhm=2.0)
        assert y_gauss[1] == 0.0  # Gaussian goes to zero at infinity

        y_lorentz = lorentzian(x, fwhm=2.0)
        assert y_lorentz[1] == 0.0  # Lorentzian also goes to zero


class TestBackendEdgeCases:
    """Test edge cases for backend selection - DEPRECATED."""

    def test_backend_removed(self):
        """Backend selection has been removed."""
        # Backend selection is now deprecated
        # This test just verifies lineshapes still work
        import numpy as np

        x = np.linspace(-5, 5, 100)
        y = gaussian(x, fwhm=2.0)
        assert np.all(np.isfinite(y))


class TestConfigEdgeCases:
    """Test edge cases for configuration system."""

    def test_negative_refine_iterations(self):
        """Test config with negative refine iterations."""
        from pydantic import ValidationError

        from peakfit.core.domain.config import FitConfig

        with pytest.raises(ValidationError):
            FitConfig(refine_iterations=-1)

    def test_negative_contour_factor(self):
        """Test config with negative contour factor."""
        from pydantic import ValidationError

        from peakfit.core.domain.config import ClusterConfig

        with pytest.raises(ValidationError):
            ClusterConfig(contour_factor=-1.0)

    def test_empty_output_formats(self):
        """Test config with empty output formats."""
        from peakfit.core.domain.config import OutputConfig

        config = OutputConfig(formats=[])
        assert config.formats == []

    def test_exclude_planes_negative_rejected(self):
        """Test config rejects negative exclude planes."""
        from pydantic import ValidationError

        from peakfit.core.domain.config import PeakFitConfig

        # Negative indices should be rejected by validation
        with pytest.raises(ValidationError, match="non-negative"):
            PeakFitConfig(exclude_planes=[999, 1000, -1])

        # But large positive indices are allowed (validated when reading spectra)
        config = PeakFitConfig(exclude_planes=[999, 1000])
        assert 999 in config.exclude_planes

    def test_config_round_trip(self):
        """Test config can be saved and loaded."""
        import tempfile
        from pathlib import Path

        from peakfit.core.domain.config import PeakFitConfig
        from peakfit.io.config import load_config, save_config

        config = PeakFitConfig(noise_level=1.5, exclude_planes=[1, 2, 3])
        config.fitting.lineshape = "lorentzian"
        config.fitting.refine_iterations = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.toml"
            save_config(config, config_path)

            loaded = load_config(config_path)
            assert loaded.noise_level == 1.5
            assert loaded.exclude_planes == [1, 2, 3]
            assert loaded.fitting.lineshape == "lorentzian"
            assert loaded.fitting.refine_iterations == 3


class TestNumericStability:
    """Test numeric stability and edge cases."""

    def test_very_large_numbers(self):
        """Test functions with very large numbers."""
        x = np.array([1e10, 1e11, 1e12])
        y = gaussian(x, fwhm=2.0)
        assert np.all(y >= 0.0)
        assert np.all(y < 1e-10)  # Should be essentially zero

    def test_very_small_numbers(self):
        """Test functions with very small numbers."""
        x = np.array([1e-10, 1e-11, 1e-12])
        y = gaussian(x, fwhm=2.0)
        assert np.all(y > 0.99)  # All very close to peak

    def test_mixed_scale_numbers(self):
        """Test functions with mixed scale numbers."""
        x = np.array([1e-6, 1.0, 1e6])
        y = gaussian(x, fwhm=2.0)
        assert np.all(np.isfinite(y))
        # 1e-6 is very close to center (0), so should be ~1.0
        # 1.0 is away from center, so should be lower
        # 1e6 is very far, so should be ~0
        assert y[0] > y[1] > y[2]


class TestErrorMessages:
    """Test that error messages are informative."""

    def test_parameter_keyerror(self):
        """Test accessing non-existent parameter."""
        params = Parameters()
        with pytest.raises(KeyError):
            _ = params["nonexistent"]

    def test_parameter_bounds_message(self):
        """Test parameter summary shows bound warnings."""
        params = Parameters()
        # Add parameter at boundary
        params.add("at_bound", value=10.0, min=0.0, max=10.0)
        # Add normal parameter
        params.add("normal", value=5.0, min=0.0, max=10.0)

        boundary_params = params.get_boundary_params()
        assert "at_bound" in boundary_params
        assert "normal" not in boundary_params


class TestTypeHints:
    """Test that type hints are correctly used."""

    def test_parameter_type_enum(self):
        """Test ParameterType enum values."""
        types = list(ParameterType)
        assert ParameterType.POSITION in types
        assert ParameterType.FWHM in types
        assert ParameterType.FRACTION in types
        assert ParameterType.PHASE in types
        assert ParameterType.JCOUPLING in types
        assert ParameterType.AMPLITUDE in types

    def test_parameter_unit_attribute(self):
        """Test parameter unit attribute."""
        p1 = Parameter("test", value=100.0, unit="Hz")
        assert p1.unit == "Hz"

        p2 = Parameter("test", value=5.0, unit="ppm")
        assert p2.unit == "ppm"

        p3 = Parameter("test", value=90.0, unit="deg")
        assert p3.unit == "deg"
