from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from nmrglue.fileio.pipe import guess_udic, read

from peakfit.cli import Arguments
from peakfit.nmrpipe import SpectralParameters, read_spectral_parameters
from peakfit.typing import FloatArray


@dataclass
class Spectra:
    dic: dict
    data: FloatArray
    z_values: np.ndarray
    pseudo_dim_added: bool = False

    def __post_init__(self) -> None:
        udic = guess_udic(self.dic, self.data)
        no_pseudo_dim = udic[0]["freq"]
        if no_pseudo_dim:
            self.data = np.expand_dims(self.data, axis=0)
            self.pseudo_dim_added = True
        if self.z_values.size == 0:
            self.z_values = np.arange(self.data.shape[0])

    @cached_property
    def params(self) -> list[SpectralParameters]:
        return read_spectral_parameters(self.dic, self.data)

    def exclude_planes(self, exclude_list: Sequence[int] | None) -> None:
        if exclude_list is None:
            return
        mask = ~np.isin(range(self.data.shape[0]), exclude_list)
        self.data, self.z_values = self.data[mask], self.z_values[mask]


def read_spectra(
    path_spectra: Path,
    path_z_values: Path | None = None,
    exclude_list: Sequence[int] | None = None,
) -> Spectra:
    """Read NMRPipe spectra and z-values, returning a Spectra object."""
    dic, data = read(path_spectra)
    data = data.astype(np.float32)

    if path_z_values is not None:
        z_values = np.genfromtxt(path_z_values, dtype=None, encoding="utf-8")
    else:
        z_values = np.array([])

    spectra = Spectra(dic, data, z_values)
    spectra.exclude_planes(exclude_list)

    return Spectra(dic, data, z_values)


def get_shape_names(clargs: Arguments, spectra: Spectra) -> list[str]:
    """Determine the shape names for fitting based on command line arguments or spectral parameters."""
    if clargs.pvoigt:
        shape = "pvoigt"
    elif clargs.lorentzian:
        shape = "lorentzian"
    elif clargs.gaussian:
        shape = "gaussian"
    else:
        return [determine_shape_name(param) for param in spectra.params[1:]]

    return [shape] * (spectra.data.ndim - 1)


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
