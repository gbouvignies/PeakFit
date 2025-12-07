"""Unit tests for PipelineRunner."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from peakfit.services.fit.runner import PipelineRunner, FitArguments
from peakfit.core.domain.config import (
    PeakFitConfig,
    OutputConfig,
    FitConfig,
    ParameterConfig,
    ClusterConfig,
)


@pytest.fixture
def mock_config():
    return PeakFitConfig(
        output=OutputConfig(directory=Path("test_out")),
        fitting=FitConfig(),
        parameters=ParameterConfig(),
        clustering=ClusterConfig(),
    )


@pytest.fixture
def mock_clargs():
    return FitArguments(
        path_spectra=Path("test.ft2"),
        path_list=Path("test.list"),
    )


@pytest.fixture
def runner(mock_config, mock_clargs):
    return PipelineRunner(mock_config, mock_clargs)


def test_runner_initialization(runner, mock_config, mock_clargs):
    """Test that runner initializes correctly."""
    assert runner.config == mock_config
    assert runner.clargs == mock_clargs


@patch("peakfit.services.fit.runner.read_spectra")
def test_load_data(mock_read, runner):
    """Test data loading delegation."""
    runner.load_data()
    mock_read.assert_called_once_with(
        runner.clargs.path_spectra, runner.clargs.path_z_values, runner.clargs.exclude
    )


@patch("peakfit.services.fit.runner.read_list")
def test_load_peaks(mock_read, runner):
    """Test peak loading delegation."""
    spectra = MagicMock()
    shape_names = ["Lorentzian"]
    runner.load_peaks(spectra, shape_names)
    mock_read.assert_called_once_with(spectra, shape_names, runner.clargs)


@patch("peakfit.services.fit.runner.prepare_noise_level")
def test_estimate_noise(mock_prep, runner):
    """Test noise estimation logic."""
    spectra = MagicMock()
    mock_prep.return_value = 100.0

    val, provided = runner.estimate_noise(spectra)

    assert val == 100.0
    assert provided is False  # Default is None in clargs
    assert runner.clargs.noise == 100.0


@patch("peakfit.services.fit.runner.create_clusters")
def test_cluster_peaks(mock_cluster, runner):
    """Test clustering logic."""
    spectra = MagicMock()
    peaks = []
    runner.clargs.noise = 10.0
    # contour_level is None initially

    runner.cluster_peaks(spectra, peaks)

    assert runner.clargs.contour_level == 50.0  # 5 * 10.0
    mock_cluster.assert_called_once()
