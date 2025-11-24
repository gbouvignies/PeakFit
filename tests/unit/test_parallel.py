"""Parallel module removal test.

This test ensures that the parallelization module has been removed and is
no longer importable.
"""

import pytest


def test_parallel_module_removed():
    """Importing `peakfit.fitting.parallel` should fail now that it's removed."""
    with pytest.raises(ModuleNotFoundError):
        __import__("peakfit.fitting.parallel")


class TestOptimizedFunctions:
    """Tests for optimized lineshape functions."""

    def test_import_optimized_module(self):
        """Should be able to import optimized module."""
        from peakfit.lineshapes.functions import gaussian_jit, lorentzian_jit, pvoigt_jit

        assert callable(gaussian_jit)
        assert callable(lorentzian_jit)
        assert callable(pvoigt_jit)

    def test_optimized_gaussian_correctness(self):
        """Optimized Gaussian should give same results as original."""
        import numpy as np

        from peakfit.lineshapes import gaussian
        from peakfit.lineshapes.functions import gaussian_jit

        dx = np.linspace(-50, 50, 101)
        fwhm = 10.0

        original = gaussian(dx, fwhm)
        optimized = gaussian_jit(dx, fwhm)

        np.testing.assert_allclose(original, optimized, rtol=1e-10)

    def test_optimized_lorentzian_correctness(self):
        """Optimized Lorentzian should give same results as original."""
        import numpy as np

        from peakfit.lineshapes import lorentzian
        from peakfit.lineshapes.functions import lorentzian_jit

        dx = np.linspace(-50, 50, 101)
        fwhm = 10.0

        original = lorentzian(dx, fwhm)
        optimized = lorentzian_jit(dx, fwhm)

        np.testing.assert_allclose(original, optimized, rtol=1e-10)

    def test_optimized_pvoigt_correctness(self):
        """Optimized Pseudo-Voigt should give same results as original."""
        import numpy as np

        from peakfit.lineshapes import pvoigt
        from peakfit.lineshapes.functions import pvoigt_jit

        dx = np.linspace(-50, 50, 101)
        fwhm = 10.0
        eta = 0.5

        original = pvoigt(dx, fwhm, eta)
        optimized = pvoigt_jit(dx, fwhm, eta)

        np.testing.assert_allclose(original, optimized, rtol=1e-10)

    def test_optimized_no_apod_correctness(self):
        """Optimized NoApod should give same results as original."""
        import numpy as np

        from peakfit.lineshapes import no_apod
        from peakfit.lineshapes.functions import no_apod_jit

        dx = np.linspace(-50, 50, 101)
        r2 = 5.0  # Hz
        aq = 0.1  # seconds
        phase = 10.0  # degrees

        original = no_apod(dx, r2, aq, phase)
        optimized = no_apod_jit(dx, r2, aq, phase)

        np.testing.assert_allclose(original, optimized, rtol=1e-10)

    def test_optimized_sp1_correctness(self):
        """Optimized SP1 should give same results as original."""
        import numpy as np

        from peakfit.lineshapes import sp1
        from peakfit.lineshapes.functions import sp1_jit

        dx = np.linspace(-50, 50, 101)
        r2 = 5.0  # Hz
        aq = 0.1  # seconds
        end = 1.0
        off = 0.5
        phase = 10.0  # degrees

        original = sp1(dx, r2, aq, end, off, phase)
        optimized = sp1_jit(dx, r2, aq, end, off, phase)

        np.testing.assert_allclose(original, optimized, rtol=1e-10)

    def test_optimized_sp2_correctness(self):
        """Optimized SP2 should give same results as original."""
        import numpy as np

        from peakfit.lineshapes import sp2
        from peakfit.lineshapes.functions import sp2_jit

        dx = np.linspace(-50, 50, 101)
        r2 = 5.0  # Hz
        aq = 0.1  # seconds
        end = 1.0
        off = 0.5
        phase = 10.0  # degrees

        original = sp2(dx, r2, aq, end, off, phase)
        optimized = sp2_jit(dx, r2, aq, end, off, phase)

        np.testing.assert_allclose(original, optimized, rtol=1e-10)

    def test_optimization_info(self):
        """Should provide optimization info."""
        from peakfit.lineshapes.functions import get_optimization_info

        info = get_optimization_info()
        assert "numba_available" in info
        assert "jit_enabled" in info
        assert "optimizations" in info
        assert isinstance(info["optimizations"], list)

    def test_check_numba_available(self):
        """Should report numba availability."""
        from peakfit.lineshapes.functions import check_numba_available

        result = check_numba_available()
        assert isinstance(result, bool)


class TestFittingEngine:
    """Tests for the new fitting engine."""

    def test_import_fitting_module(self):
        """Should be able to import fitting module."""
        from peakfit.fitting.parameters import FitResult, Parameter, Parameters, fit_cluster

        assert Parameter is not None
        assert Parameters is not None
        assert FitResult is not None
        assert callable(fit_cluster)

    def test_parameter_creation(self):
        """Parameter should store values and bounds correctly."""
        from peakfit.fitting.parameters import Parameter

        param = Parameter("test", 10.0, min=5.0, max=15.0, vary=True)
        assert param.name == "test"
        assert param.value == 10.0
        assert param.min == 5.0
        assert param.max == 15.0
        assert param.vary is True

    def test_parameter_bounds_validation(self):
        """Parameter should validate bounds."""
        from peakfit.fitting.parameters import Parameter

        # Value outside bounds should raise
        with pytest.raises(ValueError, match=r"min \("):
            Parameter("test", 20.0, min=5.0, max=15.0)

        # Min > max should raise
        with pytest.raises(ValueError, match=r"min \("):
            Parameter("test", 10.0, min=15.0, max=5.0)

    def test_parameters_collection(self):
        """Parameters collection should work correctly."""
        from peakfit.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", 10.0, min=5.0, max=15.0)
        params.add("fwhm", 25.0, min=1.0, max=100.0)

        assert "x0" in params
        assert "fwhm" in params
        assert params["x0"].value == 10.0
        assert params["fwhm"].value == 25.0

    def test_parameters_valuesdict(self):
        """Parameters should provide values dictionary."""
        from peakfit.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", 10.0)
        params.add("fwhm", 25.0)

        vdict = params.valuesdict()
        assert vdict == {"x0": 10.0, "fwhm": 25.0}

    def test_parameters_vary_methods(self):
        """Parameters should track varying parameters."""
        from peakfit.fitting.parameters import Parameters

        params = Parameters()
        params.add("x0", 10.0, vary=True)
        params.add("fwhm", 25.0, vary=True)
        params.add("fixed", 0.0, vary=False)

        vary_names = params.get_vary_names()
        assert "x0" in vary_names
        assert "fwhm" in vary_names
        assert "fixed" not in vary_names

        vary_values = params.get_vary_values()
        assert len(vary_values) == 2

    def test_fit_result_chisqr(self):
        """FitResult should compute chi-squared."""
        import numpy as np

        from peakfit.fitting.parameters import FitResult, Parameters

        params = Parameters()
        residual = np.array([1.0, 2.0, 3.0])

        result = FitResult(
            params=params,
            residual=residual,
            cost=7.0,
            nfev=10,
            njev=5,
            success=True,
            message="Converged",
        )

        assert result.chisqr == 14.0  # 1 + 4 + 9
