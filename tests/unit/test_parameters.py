"""Test Parameter and Parameters classes."""

import pytest

import numpy as np


class TestParameter:
    """Tests for Parameter class."""

    def test_basic_parameter(self):
        """Should create a basic parameter."""
        from peakfit.core.fitting.parameters import Parameter

        param = Parameter("x0", 10.0, min=0.0, max=20.0, vary=True)
        assert param.name == "x0"
        assert param.value == 10.0
        assert param.min == 0.0
        assert param.max == 20.0
        assert param.vary is True

    def test_invalid_bounds_raises(self):
        """Should raise error when min > max."""
        from peakfit.core.fitting.parameters import Parameter

        with pytest.raises(ValueError, match=r"min.*>.*max"):
            Parameter("bad", 10.0, min=20.0, max=10.0)

    def test_value_outside_bounds_raises(self):
        """Should raise error when value outside bounds."""
        from peakfit.core.fitting.parameters import Parameter

        with pytest.raises(ValueError, match="outside bounds"):
            Parameter("bad", 30.0, min=0.0, max=20.0)

    def test_repr(self):
        """Should have readable string representation."""
        from peakfit.core.fitting.parameters import Parameter

        param = Parameter("fwhm", 25.5, min=1.0, max=100.0, vary=True)
        repr_str = repr(param)
        assert "fwhm" in repr_str
        assert "25.5" in repr_str
        assert "vary" in repr_str

    def test_is_at_boundary_lower(self):
        """Should detect parameter at lower boundary."""
        from peakfit.core.fitting.parameters import Parameter

        param = Parameter("x", 0.0, min=0.0, max=10.0)
        assert param.is_at_boundary() is True

    def test_is_at_boundary_upper(self):
        """Should detect parameter at upper boundary."""
        from peakfit.core.fitting.parameters import Parameter

        param = Parameter("x", 10.0, min=0.0, max=10.0)
        assert param.is_at_boundary() is True

    def test_not_at_boundary(self):
        """Should detect parameter not at boundary."""
        from peakfit.core.fitting.parameters import Parameter

        param = Parameter("x", 5.0, min=0.0, max=10.0)
        assert param.is_at_boundary() is False


class TestParameters:
    """Tests for Parameters collection class."""

    def test_add_and_get(self):
        """Should add and retrieve parameters."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, min=0.0, max=20.0)
        params.add("fwhm", value=25.0, min=1.0, max=100.0)

        assert params["x0"].value == 10.0
        assert params["fwhm"].value == 25.0

    def test_contains(self):
        """Should check parameter existence."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0)

        assert "x0" in params
        assert "nonexistent" not in params

    def test_len(self):
        """Should return number of parameters."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        assert len(params) == 0

        params.add("x0", value=10.0)
        assert len(params) == 1

        params.add("fwhm", value=25.0)
        assert len(params) == 2

    def test_repr(self):
        """Should have readable string representation."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("fixed", value=0.0, vary=False)

        repr_str = repr(params)
        assert "2 total" in repr_str
        assert "1 varying" in repr_str

    def test_valuesdict(self):
        """Should return dictionary of values."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0)
        params.add("fwhm", value=25.0)

        # The previous `valuesdict` helper was removed. Build an equivalent
        # mapping using the public `items()` API and Parameter.value.
        values = {name: p.value for name, p in params.items()}
        assert values == {"x0": 10.0, "fwhm": 25.0}

    def test_summary(self):
        """Should generate formatted summary."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.5, min=0.0, max=20.0, vary=True)
        params.add("fixed", value=1.0, vary=False)

        summary = params.summary()
        assert "Parameters:" in summary
        assert "x0" in summary
        assert "fixed" in summary
        assert "vary" in summary

    def test_get_boundary_params(self):
        """Should identify parameters at boundaries."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("at_min", value=0.0, min=0.0, max=10.0, vary=True)
        params.add("at_max", value=10.0, min=0.0, max=10.0, vary=True)
        params.add("not_at_bound", value=5.0, min=0.0, max=10.0, vary=True)
        params.add("fixed_at_bound", value=0.0, min=0.0, max=10.0, vary=False)

        boundary = params.get_boundary_params()
        assert "at_min" in boundary
        assert "at_max" in boundary
        assert "not_at_bound" not in boundary
        # Fixed params are excluded
        assert "fixed_at_bound" not in boundary

    def test_freeze_all(self):
        """Should freeze all parameters."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("fwhm", value=25.0, vary=True)

        params.freeze()

        assert params["x0"].vary is False
        assert params["fwhm"].vary is False

    def test_freeze_specific(self):
        """Should freeze specific parameters."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("fwhm", value=25.0, vary=True)

        params.freeze(["x0"])

        assert params["x0"].vary is False
        assert params["fwhm"].vary is True

    def test_unfreeze_all(self):
        """Should unfreeze all parameters."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=False)
        params.add("fwhm", value=25.0, vary=False)

        params.unfreeze()

        assert params["x0"].vary is True
        assert params["fwhm"].vary is True

    def test_copy(self):
        """Should create independent copy."""
        from peakfit.core.fitting.parameters import Parameters

        original = Parameters()
        original.add("x0", value=10.0, min=0.0, max=20.0, vary=True)

        copy = original.copy()
        copy["x0"].value = 15.0

        assert original["x0"].value == 10.0
        assert copy["x0"].value == 15.0

    def test_update(self):
        """Should update parameters from another collection."""
        from peakfit.core.fitting.parameters import Parameters

        params1 = Parameters()
        params1.add("x0", value=10.0)

        params2 = Parameters()
        params2.add("x0", value=15.0)
        params2.add("fwhm", value=25.0)

        params1.update(params2)

        assert params1["x0"].value == 15.0
        assert params1["fwhm"].value == 25.0

    def test_get_vary_names(self):
        """Should return names of varying parameters."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("fixed", value=0.0, vary=False)
        params.add("fwhm", value=25.0, vary=True)

        vary_names = params.get_vary_names()
        assert "x0" in vary_names
        assert "fwhm" in vary_names
        assert "fixed" not in vary_names

    def test_get_vary_values(self):
        """Should return array of varying parameter values."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("fixed", value=0.0, vary=False)
        params.add("fwhm", value=25.0, vary=True)

        values = params.get_vary_values()
        np.testing.assert_array_equal(values, [10.0, 25.0])

    def test_set_vary_values(self):
        """Should set varying parameter values from array."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("fixed", value=0.0, vary=False)
        params.add("fwhm", value=25.0, vary=True)

        params.set_vary_values(np.array([12.0, 30.0]))

        assert params["x0"].value == 12.0
        assert params["fixed"].value == 0.0  # Unchanged
        assert params["fwhm"].value == 30.0

    def test_get_vary_bounds(self):
        """Should return bounds for varying parameters."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, min=0.0, max=20.0, vary=True)
        params.add("fwhm", value=25.0, min=1.0, max=100.0, vary=True)

        lower, upper = params.get_vary_bounds()

        np.testing.assert_array_equal(lower, [0.0, 1.0])
        np.testing.assert_array_equal(upper, [20.0, 100.0])


class TestNMRSpecificParameters:
    """Tests for NMR-specific parameter features."""

    def test_parameter_type_enum(self):
        """Should have all expected NMR parameter types."""
        from peakfit.core.fitting.parameters import ParameterType

        assert ParameterType.POSITION.value == "position"
        assert ParameterType.FWHM.value == "fwhm"
        assert ParameterType.FRACTION.value == "fraction"
        assert ParameterType.PHASE.value == "phase"
        assert ParameterType.JCOUPLING.value == "jcoupling"
        assert ParameterType.AMPLITUDE.value == "amplitude"
        assert ParameterType.GENERIC.value == "generic"

    def test_parameter_with_type(self):
        """Should create parameter with NMR type."""
        from peakfit.core.fitting.parameters import Parameter, ParameterType

        param = Parameter(
            "peak1_fwhm",
            25.0,
            min=0.1,
            max=200.0,
            vary=True,
            param_type=ParameterType.FWHM,
            unit="Hz",
        )
        assert param.param_type == ParameterType.FWHM
        assert param.unit == "Hz"

    def test_parameter_type_default_bounds_fwhm(self):
        """Should apply default bounds for FWHM type."""
        from peakfit.core.fitting.parameters import Parameter, ParameterType

        # Only provide value and type - should get default bounds
        param = Parameter("lw", 25.0, param_type=ParameterType.FWHM)
        assert param.min == 0.1
        assert param.max == 200.0

    def test_parameter_type_default_bounds_fraction(self):
        """Should apply default bounds for FRACTION type."""
        from peakfit.core.fitting.parameters import Parameter, ParameterType

        param = Parameter("eta", 0.5, param_type=ParameterType.FRACTION)
        assert param.min == 0.0
        assert param.max == 1.0

    def test_parameter_type_default_bounds_phase(self):
        """Should apply default bounds for PHASE type."""
        from peakfit.core.fitting.parameters import Parameter, ParameterType

        param = Parameter("ph", 0.0, param_type=ParameterType.PHASE)
        assert param.min == -180.0
        assert param.max == 180.0

    def test_parameter_type_default_bounds_jcoupling(self):
        """Should apply default bounds for JCOUPLING type."""
        from peakfit.core.fitting.parameters import Parameter, ParameterType

        param = Parameter("j", 7.0, param_type=ParameterType.JCOUPLING)
        assert param.min == 0.0
        assert param.max == 20.0

    def test_parameter_explicit_bounds_override_defaults(self):
        """Explicit bounds should override type defaults."""
        from peakfit.core.fitting.parameters import Parameter, ParameterType

        # Explicitly set bounds should be used
        param = Parameter("lw", 25.0, min=5.0, max=50.0, param_type=ParameterType.FWHM)
        assert param.min == 5.0
        assert param.max == 50.0

    def test_parameters_add_with_type(self):
        """Should add parameter with type and unit."""
        from peakfit.core.fitting.parameters import Parameters, ParameterType

        params = Parameters()
        params.add(
            "peak1_fwhm", value=25.0, min=1.0, max=100.0, param_type=ParameterType.FWHM, unit="Hz"
        )

        assert params["peak1_fwhm"].param_type == ParameterType.FWHM
        assert params["peak1_fwhm"].unit == "Hz"

    def test_parameters_copy_preserves_type(self):
        """Copy should preserve param_type and unit."""
        from peakfit.core.fitting.parameters import Parameters, ParameterType

        original = Parameters()
        original.add("x0", value=10.0, param_type=ParameterType.POSITION, unit="ppm")
        original.add("fwhm", value=25.0, param_type=ParameterType.FWHM, unit="Hz")

        copy = original.copy()

        assert copy["x0"].param_type == ParameterType.POSITION
        assert copy["x0"].unit == "ppm"
        assert copy["fwhm"].param_type == ParameterType.FWHM
        assert copy["fwhm"].unit == "Hz"

    def test_parameter_repr_includes_unit(self):
        """String representation should include unit."""
        from peakfit.core.fitting.parameters import Parameter, ParameterType

        param = Parameter("lw", 25.5, min=1.0, max=100.0, param_type=ParameterType.FWHM, unit="Hz")
        repr_str = repr(param)
        assert "Hz" in repr_str
        assert "lw" in repr_str

    def test_parameter_relative_position(self):
        """Should calculate relative position within bounds."""
        from peakfit.core.fitting.parameters import Parameter

        # At minimum
        param1 = Parameter("x", 0.0, min=0.0, max=10.0)
        assert param1.relative_position() == 0.0

        # At maximum
        param2 = Parameter("x", 10.0, min=0.0, max=10.0)
        assert param2.relative_position() == 1.0

        # In middle
        param3 = Parameter("x", 5.0, min=0.0, max=10.0)
        assert param3.relative_position() == 0.5

        # At 25%
        param4 = Parameter("x", 2.5, min=0.0, max=10.0)
        assert param4.relative_position() == 0.25

    def test_parameter_relative_position_infinite_bounds(self):
        """Relative position with infinite bounds should return 0.5."""
        from peakfit.core.fitting.parameters import Parameter

        param = Parameter("x", 100.0)  # Default unbounded
        assert param.relative_position() == 0.5


class TestComputedParameters:
    """Tests for computed parameter functionality."""

    def test_computed_parameter_creation(self):
        """Should create a computed parameter with vary=False."""
        from peakfit.core.fitting.parameters import Parameter, ParameterType

        param = Parameter(
            "I_peak1[0]",
            1000.0,
            vary=False,
            param_type=ParameterType.AMPLITUDE,
            computed=True,
        )
        assert param.computed is True
        assert param.vary is False
        assert param.param_type == ParameterType.AMPLITUDE

    def test_computed_with_vary_raises(self):
        """Should raise error when computed=True and vary=True."""
        from peakfit.core.fitting.parameters import Parameter

        with pytest.raises(ValueError, match="computed=True requires vary=False"):
            Parameter("bad", 10.0, vary=True, computed=True)

    def test_parameter_repr_computed(self):
        """String representation should show 'computed' status."""
        from peakfit.core.fitting.parameters import Parameter

        param = Parameter("I_peak[0]", 1000.0, vary=False, computed=True)
        repr_str = repr(param)
        assert "computed" in repr_str

    def test_parameters_add_computed(self):
        """Should add computed parameter via add()."""
        from peakfit.core.fitting.parameters import Parameters, ParameterType

        params = Parameters()
        params.add(
            "I_peak1[0]",
            value=1000.0,
            vary=False,
            param_type=ParameterType.AMPLITUDE,
            computed=True,
        )

        assert params["I_peak1[0]"].computed is True
        assert params["I_peak1[0]"].vary is False

    def test_get_computed_names(self):
        """Should return names of computed parameters."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("I_peak1[0]", value=1000.0, vary=False, computed=True)
        params.add("I_peak2[0]", value=2000.0, vary=False, computed=True)
        params.add("fixed", value=0.0, vary=False)

        computed_names = params.get_computed_names()
        assert "I_peak1[0]" in computed_names
        assert "I_peak2[0]" in computed_names
        assert "x0" not in computed_names
        assert "fixed" not in computed_names

    def test_get_fitted_names(self):
        """Should return names of all fitted parameters (vary or computed)."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("I_peak1[0]", value=1000.0, vary=False, computed=True)
        params.add("fixed", value=0.0, vary=False)

        fitted_names = params.get_fitted_names()
        assert "x0" in fitted_names
        assert "I_peak1[0]" in fitted_names
        assert "fixed" not in fitted_names

    def test_get_n_fitted_params(self):
        """Should count both varying and computed parameters."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("fwhm", value=25.0, vary=True)
        params.add("I_peak1[0]", value=1000.0, vary=False, computed=True)
        params.add("I_peak2[0]", value=2000.0, vary=False, computed=True)
        params.add("fixed", value=0.0, vary=False)

        # 2 varying + 2 computed = 4 fitted
        assert params.get_n_fitted_params() == 4

    def test_copy_preserves_computed(self):
        """Copy should preserve computed flag."""
        from peakfit.core.fitting.parameters import Parameters

        original = Parameters()
        original.add("x0", value=10.0, vary=True)
        original.add("I_peak1[0]", value=1000.0, vary=False, computed=True)

        copy = original.copy()

        assert copy["I_peak1[0]"].computed is True
        assert copy["x0"].computed is False

    def test_repr_with_computed(self):
        """Parameters repr should show computed count."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("I_peak1[0]", value=1000.0, vary=False, computed=True)

        repr_str = repr(params)
        assert "1 computed" in repr_str

    def test_summary_shows_computed(self):
        """Summary should show 'computed' status."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("I_peak1[0]", value=1000.0, vary=False, computed=True)

        summary = params.summary()
        assert "(computed)" in summary
        assert "(vary)" in summary

    def test_get_vary_names_excludes_computed(self):
        """get_vary_names should exclude computed parameters."""
        from peakfit.core.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("I_peak1[0]", value=1000.0, vary=False, computed=True)

        vary_names = params.get_vary_names()
        assert "x0" in vary_names
        assert "I_peak1[0]" not in vary_names
