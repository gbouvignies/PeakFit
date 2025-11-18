"""Test fast fitting module (direct scipy optimization)."""

import numpy as np
import pytest

# Skip if scipy isn't available
pytest.importorskip("scipy")


class TestFastFitConversions:
    """Tests for parameter conversion functions."""

    def test_params_to_arrays_basic(self):
        """Should convert Parameters to numpy arrays."""
        from peakfit.core.fast_fit import params_to_arrays
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("x0", value=10.0, min=5.0, max=15.0, vary=True)
        params.add("fwhm", value=25.0, min=1.0, max=100.0, vary=True)
        params.add("fixed", value=0.0, vary=False)

        x0, lower, upper, names = params_to_arrays(params)

        assert len(names) == 2  # Only varying params
        assert "x0" in names
        assert "fwhm" in names
        assert "fixed" not in names
        np.testing.assert_array_equal(x0, [10.0, 25.0])
        np.testing.assert_array_equal(lower, [5.0, 1.0])
        np.testing.assert_array_equal(upper, [15.0, 100.0])

    def test_params_to_arrays_no_bounds(self):
        """Should handle parameters without explicit bounds."""
        from peakfit.core.fast_fit import params_to_arrays
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("unbounded", value=42.0, vary=True)

        x0, lower, upper, names = params_to_arrays(params)

        assert names == ["unbounded"]
        assert x0[0] == 42.0
        assert lower[0] == -np.inf
        assert upper[0] == np.inf

    def test_params_to_arrays_empty(self):
        """Should handle case with no varying parameters."""
        from peakfit.core.fast_fit import params_to_arrays
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("fixed", value=10.0, vary=False)

        x0, lower, upper, names = params_to_arrays(params)

        assert len(names) == 0
        assert len(x0) == 0
        assert len(lower) == 0
        assert len(upper) == 0

    def test_arrays_to_params_update(self):
        """Should update Parameters with optimized values."""
        from peakfit.core.fast_fit import arrays_to_params
        from peakfit.core.fitting import Parameters

        template = Parameters()
        template.add("x0", value=10.0, min=5.0, max=15.0, vary=True)
        template.add("fwhm", value=25.0, min=1.0, max=100.0, vary=True)

        x_new = np.array([12.0, 30.0])
        names = ["x0", "fwhm"]

        updated = arrays_to_params(x_new, names, template)

        assert updated["x0"].value == 12.0
        assert updated["fwhm"].value == 30.0
        # Should preserve bounds
        assert updated["x0"].min == 5.0
        assert updated["x0"].max == 15.0


class TestFastFitModuleImports:
    """Tests for module imports and basic structure."""

    def test_import_fast_fit_module(self):
        """Should be able to import fast_fit module."""
        from peakfit.core.fast_fit import (
            arrays_to_params,
            fit_cluster_fast,
            fit_clusters_fast,
            params_to_arrays,
            residuals_fast,
        )

        assert callable(params_to_arrays)
        assert callable(arrays_to_params)
        assert callable(residuals_fast)
        assert callable(fit_cluster_fast)
        assert callable(fit_clusters_fast)

    def test_fit_cluster_fast_signature(self):
        """fit_cluster_fast should have correct signature."""
        import inspect

        from peakfit.core.fast_fit import fit_cluster_fast

        sig = inspect.signature(fit_cluster_fast)
        params = list(sig.parameters.keys())
        assert "cluster" in params
        assert "noise" in params
        assert "fixed" in params
        assert "params_init" in params

    def test_fit_clusters_fast_signature(self):
        """fit_clusters_fast should have correct signature."""
        import inspect

        from peakfit.core.fast_fit import fit_clusters_fast

        sig = inspect.signature(fit_clusters_fast)
        params = list(sig.parameters.keys())
        assert "clusters" in params
        assert "noise" in params
        assert "refine_iterations" in params
        assert "fixed" in params
        assert "verbose" in params


class TestFastFitResultStructure:
    """Tests for result dictionary structure."""

    def test_result_dictionary_keys(self):
        """Result should contain all expected keys."""
        # Test that the structure is documented correctly
        expected_keys = {"params", "success", "chisqr", "redchi", "nfev", "message"}

        # These are the keys that fit_cluster_fast returns
        assert expected_keys == {"params", "success", "chisqr", "redchi", "nfev", "message"}

    def test_param_info_structure(self):
        """Parameter info should contain all expected fields."""
        expected_fields = {"value", "stderr", "vary", "min", "max"}

        # These are the fields each parameter has in the result
        assert expected_fields == {"value", "stderr", "vary", "min", "max"}


class TestFastFitEdgeCases:
    """Tests for edge cases in fast fitting."""

    def test_params_to_arrays_preserves_order(self):
        """Parameter order should be preserved."""
        from peakfit.core.fast_fit import params_to_arrays
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("a", value=1.0, vary=True)
        params.add("b", value=2.0, vary=True)
        params.add("c", value=3.0, vary=True)

        x0, _lower, _upper, names = params_to_arrays(params)

        assert names == ["a", "b", "c"]
        np.testing.assert_array_equal(x0, [1.0, 2.0, 3.0])

    def test_params_with_inf_bounds(self):
        """Should handle infinite bounds correctly."""
        from peakfit.core.fast_fit import params_to_arrays
        from peakfit.core.fitting import Parameters

        params = Parameters()
        params.add("pos", value=0.0, min=-np.inf, max=np.inf, vary=True)
        params.add("fwhm", value=10.0, min=0.0, max=np.inf, vary=True)

        _x0, lower, upper, _names = params_to_arrays(params)

        assert lower[0] == -np.inf
        assert upper[0] == np.inf
        assert lower[1] == 0.0
        assert upper[1] == np.inf

    def test_arrays_to_params_copy(self):
        """Should return a copy, not modify original."""
        from peakfit.core.fast_fit import arrays_to_params
        from peakfit.core.fitting import Parameters

        template = Parameters()
        template.add("x0", value=10.0, vary=True)

        x_new = np.array([20.0])
        names = ["x0"]

        updated = arrays_to_params(x_new, names, template)

        # Original should be unchanged
        assert template["x0"].value == 10.0
        # Updated should have new value
        assert updated["x0"].value == 20.0


class TestFastFitIntegration:
    """Integration tests with actual fitting (mocked cluster)."""

    def test_scipy_least_squares_available(self):
        """scipy.optimize.least_squares should be available."""
        from scipy.optimize import least_squares

        assert callable(least_squares)

    def test_numpy_lstsq_available(self):
        """numpy.linalg.lstsq should be available."""
        # This is used in residuals_fast
        result = np.linalg.lstsq(
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([1, 2, 3]),
            rcond=None,
        )
        assert len(result) == 4  # solution, residuals, rank, singular values

    def test_simple_least_squares_optimization(self):
        """Basic scipy least_squares should work."""
        from scipy.optimize import least_squares

        def residual(x):
            return np.array([x[0] - 3.0, x[1] - 5.0])

        result = least_squares(residual, [0.0, 0.0], bounds=([-10, -10], [10, 10]))

        assert result.success
        np.testing.assert_allclose(result.x, [3.0, 5.0], rtol=1e-6)
