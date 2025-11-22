"""Data structures and I/O for NMR spectra and peaks.

This module provides data structures for representing NMR spectra, peaks,
and clusters, along with utilities for reading peak lists and estimating noise.
"""

# Spectrum data structures
from peakfit.data.spectrum import Spectra, SpectralParameters

# Peak data structures and I/O
from peakfit.data.peaks import (
    Peak,
    create_params,
    read_csv_list,
    read_excel_list,
    read_json_list,
    read_list,
    read_sparky_list,
)

# Clustering
from peakfit.data.clustering import Cluster

# Noise estimation
from peakfit.data.noise import estimate_noise

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
