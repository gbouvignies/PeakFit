"""Tests for backend selection and management - DEPRECATED."""



# Backend selection has been deprecated - these tests verify lineshapes still work
class TestBackendDeprecated:
    """Backend selection has been deprecated."""

    def test_lineshapes_still_work(self):
        """Test that lineshapes still work without backend selection."""
        import numpy as np

        from peakfit.core.lineshapes import gaussian, lorentzian, pvoigt

        x = np.linspace(-5, 5, 100)
        y_gauss = gaussian(x, fwhm=2.0)
        y_lorentz = lorentzian(x, fwhm=2.0)
        y_pv = pvoigt(x, fwhm=2.0, eta=0.5)

        assert np.all(np.isfinite(y_gauss))
        assert np.all(np.isfinite(y_lorentz))
        assert np.all(np.isfinite(y_pv))
