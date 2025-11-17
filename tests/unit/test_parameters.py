"""Test Parameter and Parameters classes."""

import numpy as np
import pytest


class TestParameter:
    """Tests for Parameter class."""

    def test_basic_parameter(self):
        """Should create a basic parameter."""
        from peakfit.core.fitting import Parameter

        param = Parameter("x0", 10.0, min=0.0, max=20.0, vary=True)
        assert param.name == "x0"
        assert param.value == 10.0
        assert param.min == 0.0
        assert param.max == 20.0
        assert param.vary is True

    def test_invalid_bounds_raises(self):
        """Should raise error when min > max."""
        from peakfit.core.fitting import Parameter

        with pytest.raises(ValueError, match="min.*>.*max"):
            Parameter("bad", 10.0, min=20.0, max=10.0)

    def test_value_outside_bounds_raises(self):
        """Should raise error when value outside bounds."""
        from peakfit.core.fitting import Parameter

        with pytest.raises(ValueError, match="outside bounds"):
            Parameter("bad", 30.0, min=0.0, max=20.0)

    def test_repr(self):
        """Should have readable string representation."""
        from peakfit.core.fitting import Parameter

        param = Parameter("fwhm", 25.5, min=1.0, max=100.0, vary=True)
        repr_str = repr(param)
        assert "fwhm" in repr_str
        assert "25.5" in repr_str
        assert "vary" in repr_str

    def test_is_at_boundary_lower(self):
        """Should detect parameter at lower boundary."""
        from peakfit.core.fitting import Parameter

        param = Parameter("x", 0.0, min=0.0, max=10.0)
        assert param.is_at_boundary() is True

    def test_is_at_boundary_upper(self):
        """Should detect parameter at upper boundary."""
        from peakfit.core.fitting import Parameter

        param = Parameter("x", 10.0, min=0.0, max=10.0)
        assert param.is_at_boundary() is True

    def test_not_at_boundary(self):
        """Should detect parameter not at boundary."""
        from peakfit.core.fitting import Parameter

        param = Parameter("x", 5.0, min=0.0, max=10.0)
        assert param.is_at_boundary() is False


class TestParameters:
    """Tests for Parameters collection class."""

    def test_add_and_get(self):
        """Should add and retrieve parameters."""
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("x0", value=10.0, min=0.0, max=20.0)
        params.add("fwhm", value=25.0, min=1.0, max=100.0)

        assert params["x0"].value == 10.0
        assert params["fwhm"].value == 25.0

    def test_contains(self):
        """Should check parameter existence."""
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("x0", value=10.0)

        assert "x0" in params
        assert "nonexistent" not in params

    def test_len(self):
        """Should return number of parameters."""
        from peakfit.core.fitting import Parameters

        params = Parameters()
        assert len(params) == 0

        params.add("x0", value=10.0)
        assert len(params) == 1

        params.add("fwhm", value=25.0)
        assert len(params) == 2

    def test_repr(self):
        """Should have readable string representation."""
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("fixed", value=0.0, vary=False)

        repr_str = repr(params)
        assert "2 total" in repr_str
        assert "1 varying" in repr_str

    def test_valuesdict(self):
        """Should return dictionary of values."""
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("x0", value=10.0)
        params.add("fwhm", value=25.0)

        values = params.valuesdict()
        assert values == {"x0": 10.0, "fwhm": 25.0}

    def test_summary(self):
        """Should generate formatted summary."""
        from peakfit.core.fitting import Parameters

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
        from peakfit.core.fitting import Parameters

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
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("fwhm", value=25.0, vary=True)

        params.freeze()

        assert params["x0"].vary is False
        assert params["fwhm"].vary is False

    def test_freeze_specific(self):
        """Should freeze specific parameters."""
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("fwhm", value=25.0, vary=True)

        params.freeze(["x0"])

        assert params["x0"].vary is False
        assert params["fwhm"].vary is True

    def test_unfreeze_all(self):
        """Should unfreeze all parameters."""
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=False)
        params.add("fwhm", value=25.0, vary=False)

        params.unfreeze()

        assert params["x0"].vary is True
        assert params["fwhm"].vary is True

    def test_copy(self):
        """Should create independent copy."""
        from peakfit.core.fitting import Parameters

        original = Parameters()
        original.add("x0", value=10.0, min=0.0, max=20.0, vary=True)

        copy = original.copy()
        copy["x0"].value = 15.0

        assert original["x0"].value == 10.0
        assert copy["x0"].value == 15.0

    def test_update(self):
        """Should update parameters from another collection."""
        from peakfit.core.fitting import Parameters

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
        from peakfit.core.fitting import Parameters

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
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("x0", value=10.0, vary=True)
        params.add("fixed", value=0.0, vary=False)
        params.add("fwhm", value=25.0, vary=True)

        values = params.get_vary_values()
        np.testing.assert_array_equal(values, [10.0, 25.0])

    def test_set_vary_values(self):
        """Should set varying parameter values from array."""
        from peakfit.core.fitting import Parameters

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
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("x0", value=10.0, min=0.0, max=20.0, vary=True)
        params.add("fwhm", value=25.0, min=1.0, max=100.0, vary=True)

        lower, upper = params.get_vary_bounds()

        np.testing.assert_array_equal(lower, [0.0, 1.0])
        np.testing.assert_array_equal(upper, [20.0, 100.0])
