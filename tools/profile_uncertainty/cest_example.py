"""The bundled CEST example, fitted as PeakFit fits it: set-up shared by the CEST studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import nmrglue as ng
import numpy as np

from peakfit.engine.algorithms.clustering import prepare_clusters
from peakfit.engine.algorithms.varpro import VarProOptimizer, fit_cluster
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.constraints import apply_constraints
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.spectrum import Spectra, get_shape_names
from peakfit.io.readers.peaks import read_list
from peakfit.io.readers.spectrum import read_noise_model, read_spectra

if TYPE_CHECKING:
    from peakfit.engine.algorithms.noise_correlation import NoiseModel
    from peakfit.engine.domain.cluster import Cluster

DATA = Path(__file__).resolve().parents[2] / "examples" / "04-uncertainty-analysis" / "data"
SPECTRUM = DATA / "pseudo3d.ft2"
OFFSETS = DATA / "b1_offsets.txt"
NOISE_ONLY = 4.5  # a ¹⁵N column is noise only when every point is below this many σ


@dataclass(frozen=True)
class Example:
    """The spectrum, its clusters and its processing-derived noise model."""

    spectra: Spectra
    config: PeakFitConfig
    noise: float
    clusters: list[Cluster]
    model: NoiseModel
    offsets: np.ndarray


def load() -> Example:
    """Read the example and cluster it with the default configuration."""
    spectra = read_spectra(SPECTRUM, OFFSETS)
    config = PeakFitConfig()
    shapes = get_shape_names(config.fitting, spectra)  # sp2 on both axes, from the header
    peaks = read_list(DATA / "pseudo3d.list", spectra, shapes, config.fitting)
    noise, _, clusters = prepare_clusters(spectra, peaks, config)
    model = read_noise_model(spectra.dic, spectra.data.shape)
    return Example(spectra, config, noise, clusters, model, np.genfromtxt(OFFSETS))


def fit_ols(example: Example, cluster: Cluster) -> VarProOptimizer:
    """Fit one cluster by PeakFit's OLS VarPro; the optimizer is left at the optimum."""
    params = apply_constraints(Parameters.from_peaks(cluster.peaks), example.config.parameters)
    fitted = fit_cluster(params, cluster, example.noise, correlation=None).params
    optimizer = VarProOptimizer(
        cluster=cluster,
        names=fitted.get_vary_names(),
        params_template=fitted,
        noise=example.noise,
    )
    optimizer.compute_residuals(fitted.get_vary_values())
    return optimizer


def whitener(r: np.ndarray, alpha: float) -> np.ndarray:
    """W^½ for W = R_α⁻¹, R_α = (1 − α)R + αI; a pseudo-inverse where R is singular."""
    r_alpha = (1 - alpha) * r + alpha * np.eye(len(r))
    lam, vec = np.linalg.eigh(r_alpha)
    keep = lam > 1e-12 * lam.max()
    return (vec[:, keep] / np.sqrt(lam[keep])).T


def noise_only_columns() -> np.ndarray:
    """Every ¹⁵N column (all planes, all ¹H points) holding noise only."""
    _, data = ng.pipe.read(str(SPECTRUM))
    x = np.asarray(data, dtype=float)
    sigma = 1.4826 * np.median(np.abs(x - np.median(x)))
    columns = x.transpose(0, 2, 1).reshape(-1, x.shape[1])
    return columns[np.abs(columns).max(axis=1) < NOISE_ONLY * sigma]


def measured_15n_correlation() -> np.ndarray:
    """Circular ¹⁵N noise autocorrelation measured on the noise-only columns."""
    columns = noise_only_columns()
    power = (np.abs(np.fft.fft(columns, axis=1)) ** 2).mean(axis=0)
    autocorrelation = np.real(np.fft.ifft(power))
    return autocorrelation / autocorrelation[0]


def spread(values) -> str:
    """Median, 10th–90th percentile and maximum."""
    v = np.asarray(values, dtype=float)
    return (
        f"{np.median(v):.3g} ({np.percentile(v, 10):.3g}–{np.percentile(v, 90):.3g}), "
        f"max {v.max():.3g}"
    )


def _array(value) -> np.ndarray:
    if value is None:
        msg = "The optimizer has not been evaluated."
        raise RuntimeError(msg)
    return value


def data(opt: VarProOptimizer) -> np.ndarray:
    """The cluster's data, (points, planes)."""
    return _array(opt._data_matrix)


def shapes(opt: VarProOptimizer) -> np.ndarray:
    """Shapes at the optimizer's current point, (peaks, points)."""
    return _array(opt._shapes)


def amplitudes(opt: VarProOptimizer) -> np.ndarray:
    """Amplitudes at the optimizer's current point, (peaks, planes)."""
    return _array(opt._amplitudes)


def residuals(opt: VarProOptimizer) -> np.ndarray:
    """Residuals at the optimizer's current point, (points, planes)."""
    return _array(opt._residuals)


def derivatives(opt: VarProOptimizer) -> tuple[np.ndarray, np.ndarray]:
    """Shape derivatives (parameters, points) and the peak owning each parameter."""
    return _array(opt._d_matrix).copy(), _array(opt._param_peak_indices)
