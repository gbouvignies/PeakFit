from collections.abc import Sequence

import lmfit as lf
import numpy as np
from lmfit.models import GaussianModel
from numpy.typing import NDArray

from peakfit.clustering import Cluster
from peakfit.computing import calculate_shape_heights, residuals, simulate_data
from peakfit.spectra import Spectra

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


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


def extract_noise_spectra(
    mc: Sequence[str], spectra: Spectra
) -> tuple[int, FloatArray]:
    """Extract noise spectra based on monte carlo parameters."""
    x1, x2, y1, y2, n_iter = mc
    x1, x2, y1, y2 = (float(val.replace("ppm", "")) for val in (x1, x2, y1, y2))
    n_iter = int(n_iter)
    x1_pt, x2_pt = sorted(
        (spectra.params[2].ppm2pt_i(x1), spectra.params[2].ppm2pt_i(x2))
    )
    y1_pt, y2_pt = sorted(
        (spectra.params[1].ppm2pt_i(y1), spectra.params[1].ppm2pt_i(y2))
    )

    noise = spectra.data[:, y1_pt:y2_pt, x1_pt:x2_pt]
    return n_iter, noise


def get_some_noise(
    spectra: Spectra, noise: FloatArray, x_pt: IntArray, y_pt: IntArray
) -> FloatArray:
    """Retrieve some noise from spectra data."""
    rng = np.random.default_rng()
    rng.shuffle(noise)
    nz_noise, ny_noise, nx_noise = noise.shape

    x_off, y_off = rng.integers(0, nx_noise - 1), rng.integers(0, ny_noise - 1)
    x_noise, y_noise = (x_pt + x_off) % nx_noise, (y_pt + y_off) % ny_noise

    return spectra.data[:, y_noise, x_noise].reshape((nz_noise, x_noise.size)).T


def calc_err_from_mc(
    params_list: Sequence[dict], heights_list: FloatArray
) -> tuple[dict, FloatArray]:
    """Calculate errors from monte carlo simulations."""
    params_dict = {}

    for params_mc in params_list:
        for name, param in params_mc.items():
            params_dict.setdefault(name, []).append(param.value)

    params_err = {name: np.std(values) for name, values in params_dict.items()}
    height_err = np.std(heights_list, axis=0, ddof=1)

    return params_err, height_err


def monte_carlo(
    mc: Sequence[str],
    spectra: Spectra,
    cluster: Cluster,
    result: lf.minimizer.MinimizerResult,
    noise: float,
) -> tuple[dict, FloatArray]:
    """Perform monte carlo simulation for error estimation."""
    n_iter, spectra_noise = extract_noise_spectra(mc, spectra)
    y_pt, x_pt = cluster.positions

    data_sim = simulate_data(result.params, cluster)
    mc_list = []

    for _ in range(n_iter):
        data_noise = get_some_noise(spectra, spectra_noise, x_pt, y_pt)
        data_mc = data_sim + data_noise
        cluster_mc = Cluster(cluster.peaks, cluster.positions, data_mc)

        params_mc = lf.minimize(
            residuals,
            result.params,
            args=(cluster_mc, noise),
            method="least_squares",
        ).params

        _, heights = calculate_shape_heights(params_mc, cluster_mc)
        mc_list.append((params_mc, heights))

    params_list, heights_list = zip(*mc_list, strict=False)
    params_err, height_err = calc_err_from_mc(params_list, heights_list)

    return params_err, height_err
