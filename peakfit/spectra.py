from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import nmrglue as ng
import numpy as np

from peakfit.nmrpipe import SpectralParameters, read_spectral_parameters


@dataclass
class Spectra:
    data: np.ndarray
    z_values: np.ndarray
    params: list[SpectralParameters]


def read_spectra(
    paths_spectra: Sequence[Path],
    paths_z_values: Sequence[Path],
    exclude_list: Sequence[int] = None,
) -> Spectra:
    """Read NMRPipe spectra and return a Spectra object."""
    data_list = [ng.fileio.pipe.read(str(path))[1] for path in paths_spectra]
    data = np.concatenate(data_list, axis=0)

    z_values_list = [np.genfromtxt(path, dtype=None) for path in paths_z_values]
    z_values = np.concatenate(z_values_list)

    if exclude_list:
        data, z_values = exclude_planes(data, z_values, exclude_list)

    dic = ng.fileio.pipe.read(str(paths_spectra[0]))[0]
    params = read_spectral_parameters(dic, data)

    return Spectra(data, z_values, params)


def exclude_planes(
    data: np.ndarray, z_values: np.ndarray, exclude_list: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Exclude specified planes from data and z_values."""
    mask = np.ones(len(z_values), dtype=bool)
    mask[exclude_list] = False
    return data[mask], z_values[mask]


def get_shape_names(clargs: Namespace, spectra: Spectra) -> list[str]:
    """Get the shape names from the command line arguments."""
    if clargs.pvoigt:
        return ["pvoigt"] * (spectra.data.ndim - 1)
    if clargs.lorentzian:
        return ["lorentzian"] * (spectra.data.ndim - 1)
    if clargs.gaussian:
        return ["gaussian"] * (spectra.data.ndim - 1)

    shape_names = []
    for dim_params in spectra.params[1:]:
        shape_name = determine_shape_name(dim_params)
        shape_names.append(shape_name)
    return shape_names


def determine_shape_name(dim_params: SpectralParameters) -> str:
    """Determine the shape name based on spectral parameters."""
    if dim_params.apocode == 1.0:
        if dim_params.apodq3 == 1.0:
            return "sp1"
        if dim_params.apodq3 == 2.0:
            return "sp2"
    if dim_params.apocode in {0.0, 2.0}:
        return "no_apod"
    return "pvoigt"
