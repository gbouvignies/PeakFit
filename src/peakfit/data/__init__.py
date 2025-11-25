"""Data structures and I/O for NMR spectra and peaks.

This module provides data structures for representing NMR spectra, peaks,
and clusters, along with utilities for reading peak lists and estimating noise.
"""

# Spectrum data structures
# Noise estimation
from peakfit.core.algorithms.noise import estimate_noise
from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak, create_params
from peakfit.core.domain.peaks_io import (
    read_csv_list,
    read_excel_list,
    read_json_list,
    read_list,
    read_sparky_list,
)
from peakfit.core.domain.spectrum import Spectra, SpectralParameters

__all__ = [
    # Spectrum
    "Spectra",
    "SpectralParameters",
    # Peaks
    "Peak",
    "create_params",
    "read_list",
    "read_sparky_list",
    "read_csv_list",
    "read_json_list",
    "read_excel_list",
    # Clustering
    "Cluster",
    # Noise
    "estimate_noise",
]
