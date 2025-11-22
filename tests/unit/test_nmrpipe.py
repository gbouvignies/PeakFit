"""Tests for NMRPipe spectral parameters module."""

import numpy as np
import pytest

from peakfit.data.spectrum import SpectralParameters, read_spectral_parameters


class TestSpectralParameters:
    """Tests for SpectralParameters dataclass."""

    @pytest.fixture
    def sample_params(self):
        """Create sample spectral parameters."""
        return SpectralParameters(
            size=1024,
            sw=10000.0,  # Hz
            obs=600.0,  # MHz
            car=4.7 * 600.0,  # Hz (water frequency)
            aq_time=0.1024,  # seconds
            apocode=0.0,
            apodq1=0.5,
            apodq2=0.98,
            apodq3=0.0,
            p180=False,
            direct=True,
            ft=True,
        )

    def test_spectral_parameters_initialization(self, sample_params):
        """Test that SpectralParameters initializes correctly."""
        assert sample_params.size == 1024
        assert sample_params.sw == 10000.0
        assert sample_params.obs == 600.0

    def test_delta_calculation(self, sample_params):
        """Test delta (ppm per point) calculation."""
        # delta = -sw / (size * obs)
        expected_delta = -10000.0 / (1024 * 600.0)
        assert sample_params.delta == pytest.approx(expected_delta)

    def test_first_calculation(self, sample_params):
        """Test first (center frequency) calculation."""
        # first = car/obs - delta * size / 2
        expected_first = (4.7 * 600.0) / 600.0 - sample_params.delta * 1024 / 2
        assert sample_params.first == pytest.approx(expected_first)

    def test_hz2pts_delta(self, sample_params):
        """Test Hz to points (delta only) conversion."""
        hz_value = 100.0
        pts = sample_params.hz2pts_delta(hz_value)
        # Should be hz / (obs * delta)
        expected = hz_value / (sample_params.obs * sample_params.delta)
        assert pts == pytest.approx(expected)

    def test_pts2hz_delta(self, sample_params):
        """Test points to Hz (delta only) conversion."""
        pts = 10.0
        hz_value = sample_params.pts2hz_delta(pts)
        expected = pts * sample_params.obs * sample_params.delta
        assert hz_value == pytest.approx(expected)

    def test_hz2pts_pts2hz_roundtrip(self, sample_params):
        """Test Hz <-> points roundtrip conversion."""
        original_hz = 100.0
        pts = sample_params.hz2pts(original_hz)
        recovered_hz = sample_params.pts2hz(pts)
        assert recovered_hz == pytest.approx(original_hz, rel=1e-10)

    def test_ppm2pts_pts2ppm_roundtrip(self, sample_params):
        """Test ppm <-> points roundtrip conversion."""
        original_ppm = 8.5
        pts = sample_params.ppm2pts(original_ppm)
        recovered_ppm = sample_params.pts2ppm(pts)
        assert recovered_ppm == pytest.approx(original_ppm, rel=1e-10)

    def test_hz2pt_i_returns_integer(self, sample_params):
        """Test Hz to points returns integer index."""
        hz_value = 2850.0
        pt_i = sample_params.hz2pt_i(hz_value)
        assert isinstance(pt_i, int)
        assert 0 <= pt_i < sample_params.size

    def test_ppm2pt_i_returns_integer(self, sample_params):
        """Test ppm to points returns integer index."""
        ppm_value = 4.75
        pt_i = sample_params.ppm2pt_i(ppm_value)
        assert isinstance(pt_i, int)
        assert 0 <= pt_i < sample_params.size

    def test_hz2ppm_conversion(self, sample_params):
        """Test Hz to ppm conversion."""
        hz_value = 600.0  # 1 ppm at 600 MHz
        ppm = sample_params.hz2ppm(hz_value)
        assert ppm == pytest.approx(1.0)

    def test_array_conversions(self, sample_params):
        """Test that conversions work with arrays."""
        hz_array = np.array([100.0, 200.0, 300.0])
        pts_array = sample_params.hz2pts_delta(hz_array)

        assert isinstance(pts_array, np.ndarray)
        assert pts_array.shape == hz_array.shape

    def test_zero_obs_handling(self):
        """Test that zero obs frequency doesn't crash."""
        params = SpectralParameters(
            size=1024,
            sw=10000.0,
            obs=0.0,
            car=0.0,
            aq_time=0.1,
            apocode=0.0,
            apodq1=0.5,
            apodq2=0.98,
            apodq3=0.0,
            p180=False,
            direct=True,
            ft=True,
        )
        # Should handle gracefully
        assert params.delta == 0.0
        assert params.first == 0.0


class TestReadSpectralParameters:
    """Tests for read_spectral_parameters function."""

    def test_read_1d_parameters(self):
        """Test reading parameters for 1D spectrum."""
        data = np.zeros(1024)
        dic = {
            "FDDIMORDER": [1],
            "FDF1FTFLAG": 1.0,
            "FDF1SW": 10000.0,
            "FDF1OBS": 600.0,
            "FDF1ORIG": -5000.0,
            "FDF1APOD": 1024.0,
            "FDF1APODCODE": 0.0,
            "FDF1APODQ1": 0.5,
            "FDF1APODQ2": 0.98,
            "FDF1APODQ3": 0.0,
            "FDF1P1": 0.0,
        }

        params_list = read_spectral_parameters(dic, data)

        assert len(params_list) == 1
        assert params_list[0].size == 1024
        assert params_list[0].sw == 10000.0
        assert params_list[0].obs == 600.0
        assert params_list[0].direct is True
        assert params_list[0].ft is True

    def test_read_2d_parameters(self):
        """Test reading parameters for 2D spectrum."""
        data = np.zeros((128, 512))
        dic = {
            "FDDIMORDER": [1, 2],
            "FDF1FTFLAG": 1.0,
            "FDF1SW": 2000.0,
            "FDF1OBS": 150.0,
            "FDF1ORIG": -1000.0,
            "FDF1APOD": 128.0,
            "FDF1APODCODE": 0.0,
            "FDF1APODQ1": 0.5,
            "FDF1APODQ2": 0.98,
            "FDF1APODQ3": 0.0,
            "FDF1P1": 0.0,
            "FDF2FTFLAG": 1.0,
            "FDF2SW": 10000.0,
            "FDF2OBS": 600.0,
            "FDF2ORIG": -5000.0,
            "FDF2APOD": 512.0,
            "FDF2APODCODE": 0.0,
            "FDF2APODQ1": 0.5,
            "FDF2APODQ2": 0.98,
            "FDF2APODQ3": 0.0,
            "FDF2P1": 0.0,
        }

        params_list = read_spectral_parameters(dic, data)

        assert len(params_list) == 2
        assert params_list[0].size == 128
        assert params_list[1].size == 512
        assert params_list[1].direct is True
        assert params_list[0].direct is False

    def test_non_ft_dimension(self):
        """Test reading parameters for non-Fourier transformed dimension."""
        data = np.zeros(256)
        dic = {
            "FDDIMORDER": [1],
            "FDF1FTFLAG": 0.0,  # Not FT'd
        }

        params_list = read_spectral_parameters(dic, data)

        assert len(params_list) == 1
        assert params_list[0].ft is False
        assert params_list[0].sw == 1.0  # Defaults
        assert params_list[0].obs == 1.0

    def test_p180_detection(self):
        """Test detection of 180-degree pulse."""
        data = np.zeros(1024)
        dic = {
            "FDDIMORDER": [1],
            "FDF1FTFLAG": 1.0,
            "FDF1SW": 10000.0,
            "FDF1OBS": 600.0,
            "FDF1ORIG": -5000.0,
            "FDF1APOD": 1024.0,
            "FDF1APODCODE": 0.0,
            "FDF1APODQ1": 0.5,
            "FDF1APODQ2": 0.98,
            "FDF1APODQ3": 0.0,
            "FDF1P1": 180.0,  # 180 pulse
        }

        params_list = read_spectral_parameters(dic, data)

        assert params_list[0].p180 is True

    def test_missing_optional_parameters(self):
        """Test that function handles missing optional parameters."""
        data = np.zeros(512)
        dic = {
            "FDDIMORDER": [1],
            "FDF1FTFLAG": 1.0,
            # Missing some optional parameters
        }

        params_list = read_spectral_parameters(dic, data)

        assert len(params_list) == 1
        # Should use defaults
        assert params_list[0].sw == 1.0
        assert params_list[0].obs == 1.0
