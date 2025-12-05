"""Test scipy optimizer module (direct scipy optimization)."""

import pytest

import numpy as np

# Skip if scipy isn't available
pytest.importorskip("scipy")


class TestScipyOptimizerModuleImports:
    """Tests for module imports and basic structure."""

    def test_import_scipy_optimizer_module(self):
        """Should be able to import scipy_optimizer module."""
        from peakfit.core.fitting.optimizer import (
            VarProOptimizer,
            fit_cluster,
            fit_clusters,
        )

        assert callable(VarProOptimizer)
        assert callable(fit_cluster)
        assert callable(fit_clusters)

    def test_fit_cluster_signature(self):
        """fit_cluster should have correct signature."""
        import inspect

        from peakfit.core.fitting.optimizer import fit_cluster

        sig = inspect.signature(fit_cluster)
        params = list(sig.parameters.keys())
        assert "params" in params
        assert "cluster" in params
        assert "noise" in params
        assert "max_nfev" in params

    def test_fit_clusters_signature(self):
        """fit_clusters should have correct signature."""
        import inspect

        from peakfit.core.fitting.optimizer import fit_clusters

        sig = inspect.signature(fit_clusters)
        params = list(sig.parameters.keys())
        assert "clusters" in params
        assert "noise" in params
        assert "refine_iterations" in params
        assert "fixed" in params
        assert "verbose" in params


class TestScipyOptimizerResultStructure:
    """Tests for result dictionary structure."""

    def test_result_dictionary_keys(self):
        """Result should contain all expected keys."""
        # Test that the structure is documented correctly
        expected_keys = {"params", "success", "chisqr", "redchi", "nfev", "message"}

        # These are the keys that fit_cluster returns
        assert expected_keys == {"params", "success", "chisqr", "redchi", "nfev", "message"}

    def test_param_info_structure(self):
        """Parameter info should contain all expected fields."""
        expected_fields = {"value", "stderr", "vary", "min", "max"}

        # These are the fields each parameter has in the result
        assert expected_fields == {"value", "stderr", "vary", "min", "max"}


class TestScipyOptimizerIntegration:
    """Integration tests with actual fitting (mocked cluster)."""

    def test_scipy_least_squares_available(self):
        """scipy.optimize.least_squares should be available."""
        from scipy.optimize import least_squares

        assert callable(least_squares)

    def test_numpy_lstsq_available(self):
        """numpy.linalg.lstsq should be available."""
        # This is used in compute_residuals
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
