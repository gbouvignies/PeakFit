"""Test spectra reading and processing."""

from unittest.mock import MagicMock, patch

import numpy as np


class TestSpectraExcludePlanes:
    """Tests for plane exclusion functionality."""

    @patch("peakfit.spectra.guess_udic")
    def test_exclude_planes_removes_data(self, mock_udic):
        """Should exclude specified planes from data."""
        from peakfit.spectra import Spectra

        # Mock guess_udic to return freq=False (no pseudo-dim needed)
        mock_udic.return_value = {0: {"freq": False}}

        # Create test data with 5 planes
        dic = {"FDF2QUADFLAG": 0.0}
        rng = np.random.default_rng()
        data = rng.standard_normal((5, 10, 10)).astype(np.float32)
        z_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        spectra = Spectra(dic, data, z_values)

        # Exclude planes 1 and 3 (0-indexed)
        spectra.exclude_planes([1, 3])

        # Should have 3 planes remaining
        assert spectra.data.shape[0] == 3
        assert len(spectra.z_values) == 3
        np.testing.assert_array_equal(spectra.z_values, [1.0, 3.0, 5.0])

    @patch("peakfit.spectra.guess_udic")
    def test_exclude_planes_none_does_nothing(self, mock_udic):
        """Should not modify data when exclude_list is None."""
        from peakfit.spectra import Spectra

        mock_udic.return_value = {0: {"freq": False}}

        dic = {"FDF2QUADFLAG": 0.0}
        rng = np.random.default_rng()
        data = rng.standard_normal((5, 10, 10)).astype(np.float32)
        z_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        spectra = Spectra(dic, data, z_values)
        original_shape = spectra.data.shape
        original_z = spectra.z_values.copy()

        spectra.exclude_planes(None)

        assert spectra.data.shape == original_shape
        np.testing.assert_array_equal(spectra.z_values, original_z)

    @patch("peakfit.spectra.guess_udic")
    def test_exclude_planes_empty_list_does_nothing(self, mock_udic):
        """Should not modify data when exclude_list is empty."""
        from peakfit.spectra import Spectra

        mock_udic.return_value = {0: {"freq": False}}

        dic = {"FDF2QUADFLAG": 0.0}
        rng = np.random.default_rng()
        data = rng.standard_normal((5, 10, 10)).astype(np.float32)
        z_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        spectra = Spectra(dic, data, z_values)
        original_shape = spectra.data.shape

        spectra.exclude_planes([])

        assert spectra.data.shape == original_shape

    @patch("peakfit.spectra.guess_udic")
    @patch("peakfit.spectra.read")
    def test_read_spectra_returns_correct_object(self, mock_read, mock_udic):
        """Should return the spectra object with exclusions applied (bug fix test)."""
        from pathlib import Path

        from peakfit.spectra import read_spectra

        # Mock guess_udic to return freq=False (no pseudo-dim needed)
        mock_udic.return_value = {0: {"freq": False}}

        # Mock NMRPipe read
        dic = {"FDF2QUADFLAG": 0.0}
        rng = np.random.default_rng()
        data = rng.standard_normal((5, 10, 10)).astype(np.float32)
        mock_read.return_value = (dic, data)

        # Create temp z_values file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1.0\n2.0\n3.0\n4.0\n5.0\n")
            z_path = Path(f.name)

        try:
            # Read with exclusions
            spectra = read_spectra(
                Path("fake_spectrum.ft3"),
                z_path,
                exclude_list=[0, 2, 4],  # Exclude first, third, fifth planes
            )

            # Critical: The returned object should have exclusions applied
            assert spectra.data.shape[0] == 2  # Only 2 planes remaining
            assert len(spectra.z_values) == 2
            np.testing.assert_array_equal(spectra.z_values, [2.0, 4.0])
        finally:
            z_path.unlink()


class TestSpectraPostInit:
    """Tests for Spectra post-initialization."""

    def test_pseudo_dim_added_for_2d_spectrum(self):
        """Should add pseudo dimension for 2D spectra."""
        from peakfit.spectra import Spectra

        # Simulate 2D spectrum (no pseudo-3D dimension)
        # We need to mock guess_udic to return freq=True for first dimension
        dic = {"FDF2QUADFLAG": 0.0, "FDF1QUADFLAG": 0.0}
        rng = np.random.default_rng()
        data = rng.standard_normal((100, 100)).astype(np.float32)
        z_values = np.array([])

        with patch("peakfit.spectra.guess_udic") as mock_udic:
            mock_udic.return_value = {0: {"freq": True}, 1: {"freq": True}}
            spectra = Spectra(dic, data, z_values)

        assert spectra.pseudo_dim_added is True
        assert spectra.data.ndim == 3
        assert spectra.data.shape[0] == 1

    @patch("peakfit.spectra.guess_udic")
    def test_empty_z_values_generates_indices(self, mock_udic):
        """Should generate z_values from indices when not provided."""
        from peakfit.spectra import Spectra

        mock_udic.return_value = {0: {"freq": False}}

        dic = {"FDF2QUADFLAG": 0.0}
        rng = np.random.default_rng()
        data = rng.standard_normal((5, 10, 10)).astype(np.float32)
        z_values = np.array([])

        spectra = Spectra(dic, data, z_values)

        assert len(spectra.z_values) == 5
        np.testing.assert_array_equal(spectra.z_values, np.arange(5))


class TestGetShapeNames:
    """Tests for shape name determination."""

    def test_pvoigt_flag(self):
        """Should return pvoigt when flag is set."""
        from peakfit.spectra import get_shape_names

        clargs = MagicMock()
        clargs.pvoigt = True
        clargs.lorentzian = False
        clargs.gaussian = False

        spectra = MagicMock()
        spectra.data.ndim = 3

        result = get_shape_names(clargs, spectra)
        assert result == ["pvoigt", "pvoigt"]

    def test_lorentzian_flag(self):
        """Should return lorentzian when flag is set."""
        from peakfit.spectra import get_shape_names

        clargs = MagicMock()
        clargs.pvoigt = False
        clargs.lorentzian = True
        clargs.gaussian = False

        spectra = MagicMock()
        spectra.data.ndim = 3

        result = get_shape_names(clargs, spectra)
        assert result == ["lorentzian", "lorentzian"]

    def test_gaussian_flag(self):
        """Should return gaussian when flag is set."""
        from peakfit.spectra import get_shape_names

        clargs = MagicMock()
        clargs.pvoigt = False
        clargs.lorentzian = False
        clargs.gaussian = True

        spectra = MagicMock()
        spectra.data.ndim = 3

        result = get_shape_names(clargs, spectra)
        assert result == ["gaussian", "gaussian"]


class TestDetermineShapeName:
    """Tests for automatic shape determination."""

    def test_sp1_apodization(self):
        """Should detect SP1 apodization."""
        from peakfit.spectra import determine_shape_name

        params = MagicMock()
        params.apocode = 1.0
        params.apodq3 = 1.0

        result = determine_shape_name(params)
        assert result == "sp1"

    def test_sp2_apodization(self):
        """Should detect SP2 apodization."""
        from peakfit.spectra import determine_shape_name

        params = MagicMock()
        params.apocode = 1.0
        params.apodq3 = 2.0

        result = determine_shape_name(params)
        assert result == "sp2"

    def test_no_apod(self):
        """Should detect no apodization."""
        from peakfit.spectra import determine_shape_name

        params = MagicMock()
        params.apocode = 0.0

        result = determine_shape_name(params)
        assert result == "no_apod"

    def test_default_pvoigt(self):
        """Should default to pvoigt for unknown cases."""
        from peakfit.spectra import determine_shape_name

        params = MagicMock()
        params.apocode = 99.0  # Unknown

        result = determine_shape_name(params)
        assert result == "pvoigt"
