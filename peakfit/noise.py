import numpy as np
from lmfit.models import GaussianModel

from peakfit.cli import Arguments
from peakfit.messages import print_estimated_noise
from peakfit.spectra import Spectra
from peakfit.typing import FloatArray


def prepare_noise_level(clargs: Arguments, spectra: Spectra) -> float:
    """Prepare the noise level for fitting."""
    if clargs.noise is not None and clargs.noise < 0.0:
        clargs.noise = None

    if clargs.noise is None:
        clargs.noise = estimate_noise(spectra.data)
        print_estimated_noise(clargs.noise)

    return clargs.noise


def estimate_noise(data: FloatArray) -> float:
    """Estimate the noise level in the data."""
    std = np.std(data)
    truncated_data = data[np.abs(data) < std]
    y, x = np.histogram(truncated_data.flatten(), bins=100)
    x = (x[1:] + x[:-1]) / 2
    model = GaussianModel()
    pars = model.guess(y, x=x)
    pars["center"].set(value=0.0, vary=False)
    out = model.fit(y, pars, x=x)
    return out.best_values["sigma"]
