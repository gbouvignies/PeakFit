"""Can a resolution-based measure flag biased or under-covering fits of overlapping peaks?

Study only: nothing here is used by PeakFit. See ``docs/validation/resolution-flag.md``
for the pre-registered definitions and pass criteria, and the results.

Each realization is a 2D pseudo-3D pair drawn with PeakFit's own lineshapes, plus
noise from the exact processing operators of ``validate_sandwich_errors.py``. It is
fitted by ``run_pipeline`` with the noise model ``read_noise_model`` reads from the
header, as PeakFit publishes it. The sweep crosses the separation (in units of the
apodized peak width), its 2D geometry, the linewidth, the amplitude ratio, the noise
level, the lineshape and the start (the truth, or a jittered peak-list start).

For every fit the tool records the published errors and warning, the errors of
the positions against the truth, and the candidate measures:
- fitted separation over the fitted apodized width (Euclidean over axes, and the
  largest single axis);
- the overlap (cosine) of the two fitted 2D shapes;
- the largest amplitude correlation and the scaled Jacobian condition number;
- the separation significance: the fitted separation over its published (sandwich)
  standard error, as a Mahalanobis distance over both axes;
- the profile nonlinearity: the profile χ² of the separation, refitted at the
  edges of its published 95% interval, against the local quadratic prediction;
- alternative-start disagreement: refits from a collapsed and a spread start.

    uv run python tools/validate_resolution_flag.py fit --output /tmp/resolution-fits.json.gz
    uv run python tools/validate_resolution_flag.py analyze /tmp/resolution-fits.json.gz \
        --output /tmp/resolution-flag.json
"""

# ruff: noqa: T201, PLR2004

from __future__ import annotations

import argparse
import gzip
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import cache
from multiprocessing import get_context
from pathlib import Path
from typing import cast

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))

from validate_sandwich_errors import HEIGHTS, NO_APOD, PLANES, SP1, SP2, Axis

from peakfit.engine.algorithms.noise_correlation import cluster_correlation
from peakfit.engine.algorithms.varpro import VarProOptimizer, identifiable_covariance
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, LineshapeName, PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.scientific.sampling import sampling_problem
from peakfit.fit.pipeline import run_pipeline
from peakfit.io.readers.spectrum import read_noise_model

SEED = 20261105
CHAINS: dict[str, tuple[Axis, Axis]] = {"sp2": SP2, "sp1": SP1, "no_apod": NO_APOD}
CENTRE = (64, 128)  # grid point of the pair's midpoint (¹⁵N, ¹H)
DEFAULT_LW = 20.0  # PeakFit's starting linewidth for phased lineshapes, Hz
JITTER = 0.2  # peak-list start: position error SD, in apodized widths
Z95 = 1.96


@dataclass(frozen=True)
class Config:
    """One sweep cell. ``separation`` is Euclidean, in apodized widths at the truth."""

    lineshape: LineshapeName
    geometry: str  # "1H", "15N", "diagonal" or "none" (coincident)
    separation: float
    lw: float  # intrinsic linewidth on both axes, Hz
    ratio: float  # height of the second peak over the first
    sigma: float  # noise SD; the larger peak's height is 1
    start: str  # "truth" or "picked"
    points: tuple[float, float] | None = None  # explicit separation in points
    window: tuple[int, int] | None = None  # explicit half window in points

    @property
    def name(self) -> str:
        """A unique, readable label; it also seeds the realizations."""
        if self.points is not None:
            return f"reference near-coalescent pair {self.points}"
        return (
            f"{self.lineshape} {self.geometry} d={self.separation:g} lw={self.lw:g} "
            f"ratio={self.ratio:g} sigma={self.sigma:g} start={self.start}"
        )


def sweep() -> list[Config]:
    """The pre-registered sweep (``docs/validation/resolution-flag.md``)."""
    separations = (0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5)
    cells = [("none", 0.0)] + [(g, d) for g in ("1H", "15N", "diagonal") for d in separations]
    configs = []
    for lineshape, lws, starts in (
        ("sp2", (10.0, 20.0, 40.0), ("picked",)),
        ("sp2", (20.0,), ("truth",)),
        ("sp1", (20.0,), ("picked",)),
        ("no_apod", (20.0,), ("picked",)),
    ):
        configs += [
            Config(lineshape, geometry, d, lw, ratio, sigma, start)
            for start in starts
            for lw in lws
            for geometry, d in cells
            for ratio in (1.0, 0.3)
            for sigma in (0.01, 0.03, 0.1)
        ]
    # The near-coalescent case of validate_sandwich_errors.py, for continuity.
    configs.append(Config("sp2", "diagonal", 0.0, 20.0, 1.0, 0.02, "truth", (0.4, 0.4), (5, 6)))
    return configs


# --- Spectra, shapes and apodized widths ------------------------------------------


@cache
def spectra_for(lineshape: str) -> Spectra:
    """An empty pseudo-3D spectrum whose header records the lineshape's processing."""
    first, second = CHAINS[lineshape]
    dic = {"FDDIMORDER": [2.0, 1.0, 3.0, 4.0], **first.header("FDF1"), **second.header("FDF2")}
    pseudo = SpectralParameters(
        size=PLANES, sw=1.0, obs=1.0, car=0.0, aq_time=1.0, apocode=0, apodq1=0, apodq2=0,
        apodq3=0, p180=False, direct=False, ft=False,
    )  # fmt: skip
    shape = (PLANES, first.processing.transform, second.processing.transform)
    return Spectra(
        dic=dic,
        data=np.zeros(shape),
        z_values=np.arange(float(PLANES)),
        params=[pseudo, first.parameters(False), second.parameters(True)],
    )


def ppm_per_point(lineshape: str) -> np.ndarray:
    """The ppm spacing of the grid points, per axis."""
    sp = spectra_for(lineshape).spectral_params
    return np.array([abs(float(sp[i].pts2ppm(1.0) - sp[i].pts2ppm(0.0))) for i in range(2)])


def make_peak(lineshape: str, name: str, points: np.ndarray) -> Peak:
    """A peak at grid ``points``; its position bounds centre on it, as in a peak list."""
    spectra = spectra_for(lineshape)
    position = [float(spectra.spectral_params[i].pts2ppm(points[i])) for i in range(2)]
    config = FitConfig(lineshape=cast("LineshapeName", lineshape))
    shapes = create_shapes(spectra, config, name, position, [lineshape] * 2)
    peak = Peak(name, np.array(position), shapes)
    peak.set_cluster_id(1)
    return peak


def parameters(peaks: list[Peak], lw: dict[str, float]) -> Parameters:
    """The peaks' parameters, with the given linewidths."""
    params = Parameters.from_peaks(peaks)
    for name, value in lw.items():
        params[name].value = value
    return params


def grid_around(centre: np.ndarray, half: np.ndarray) -> list[np.ndarray]:
    """The grid indices of a rectangle of half-width ``half`` points around ``centre``."""
    rows = [np.arange(round(c) - h, round(c) + h + 1) for c, h in zip(centre, half, strict=True)]
    return [index.ravel() for index in np.meshgrid(*rows, indexing="ij")]


@cache
def width_table(lineshape: str, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Apodized FWHM in points of the lineshape along ``axis``, against linewidth (Hz).

    The profile is sampled every 0.05 point by moving the peak across the grid.
    """
    lws = np.geomspace(0.1, 200.0, 61)
    widths = []
    for lw in lws:
        offsets, values = [], []
        for shift in np.arange(20) / 20:
            points = np.array(CENTRE, dtype=float)
            points[axis] += shift
            peak = make_peak(lineshape, "P", points)
            params = parameters([peak], {f"P.F{axis + 2}.lw": float(lw)})
            half = np.array([0, 0])
            half[axis] = 40
            grid = grid_around(np.array(CENTRE, dtype=float), half)
            shape = Cluster(1, [peak], grid, np.zeros((grid[0].size, PLANES))).evaluate(params)[0]
            offsets.append(grid[axis] - points[axis])
            values.append(shape)
        order = np.argsort(np.concatenate(offsets))
        x, y = np.concatenate(offsets)[order], np.concatenate(values)[order]
        y = y / y.max()
        top = int(np.argmax(y))
        right = top + int(np.argmax(y[top:] < 0.5))
        left = top - int(np.argmax(y[top::-1] < 0.5))
        x_right = np.interp(0.5, [y[right], y[right - 1]], [x[right], x[right - 1]])
        x_left = np.interp(0.5, [y[left], y[left + 1]], [x[left], x[left + 1]])
        widths.append(x_right - x_left)
    return lws, np.array(widths)


def apodized_width(lineshape: str, axis: int, lw: float) -> float:
    """The apodized FWHM, in points, at linewidth ``lw`` (Hz)."""
    lws, widths = width_table(lineshape, axis)
    return float(np.interp(np.log(max(lw, 0.1)), np.log(lws), widths))


# --- One realization ---------------------------------------------------------------


def _true_layout(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Peak A and B grid points, and the half window."""
    widths = np.array([apodized_width(config.lineshape, i, config.lw) for i in range(2)])
    if config.points is not None:
        delta = np.array(config.points)
    else:
        direction = {
            "1H": (0.0, 1.0),
            "15N": (1.0, 0.0),
            "diagonal": (np.sqrt(0.5), np.sqrt(0.5)),
            "none": (0.0, 0.0),
        }[config.geometry]
        delta = config.separation * np.array(direction) * widths
    centre = np.array(CENTRE, dtype=float)
    # Peak A at the midpoint minus half the separation, as in validate_sandwich_errors.
    a, b = centre - delta / 2, centre + delta / 2
    if config.window is not None:
        half = np.array(config.window)
    else:
        half = np.ceil(np.abs(delta) / 2 + 1.25 * widths).astype(int)
    return a, b, half


def _noise(config: Config, rows: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    first, second = CHAINS[config.lineshape]
    left, right = first.noise_operator(rows[0]), second.noise_operator(rows[1])
    noise = [
        (left @ rng.standard_normal((left.shape[1], right.shape[1])) @ right.T).ravel()
        for _ in range(PLANES)
    ]
    return config.sigma * np.stack(noise, axis=1)


def _fit(config: Config, grid, data, model, start_points, lw: dict[str, float]):
    """PeakFit's fit from peaks at ``start_points`` (their bounds centre there)."""
    peaks = [make_peak(config.lineshape, n, p) for n, p in zip("AB", start_points, strict=True)]
    params = parameters(peaks, lw)
    fit_config = PeakFitConfig(fitting=FitConfig(lineshape=config.lineshape))
    return run_pipeline(
        fit_config, [Cluster(1, peaks, grid, data)], config.sigma, params, noise_model=model
    ).completed_fit


def _lw_names() -> list[str]:
    return [f"{p}.F{a}.lw" for p in "AB" for a in (2, 3)]


def _profile(optimizer: VarProOptimizer, names: list[str], x: np.ndarray, scale: float):
    """Minimum χ² with the separation vector fixed at ``scale`` times the fitted one."""
    index = {n: j for j, n in enumerate(names)}
    a = [index[f"A.F{i}.cs"] for i in (2, 3)]
    b = [index[f"B.F{i}.cs"] for i in (2, 3)]
    lws = [index[n] for n in _lw_names()]
    delta = x[b] - x[a]
    y0 = np.concatenate([(x[a] + x[b]) / 2, x[lws]])
    transform = np.zeros((len(names), len(y0)))
    for k in range(2):
        transform[a[k], k] = transform[b[k], k] = 1.0
    for k, j in enumerate(lws):
        transform[j, 2 + k] = 1.0

    def full(y):
        z = x.copy()
        z[a] = y[:2] - scale * delta / 2
        z[b] = y[:2] + scale * delta / 2
        z[lws] = y[2:]
        return z

    lower = np.concatenate([[-np.inf, -np.inf], np.full(4, 0.1)])
    upper = np.concatenate([[np.inf, np.inf], np.full(4, 200.0)])
    result = least_squares(
        lambda y: optimizer.compute_residuals(full(y)),
        np.clip(y0, lower + 1e-9, upper - 1e-9),
        jac=lambda y: optimizer.compute_jacobian(full(y)) @ transform,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=200,
    )
    return float(np.sum(result.fun**2))


def realization(config: Config, index: int) -> dict:  # noqa: PLR0915
    """Fit one realization as PeakFit publishes it, with every candidate measure."""
    key = zlib.crc32(config.name.encode())
    noise_rng = np.random.default_rng([SEED, key, index, 0])
    start_rng = np.random.default_rng([SEED, key, index, 1])
    if config.points is not None:  # validate_sandwich_errors.py's own seeds
        noise_rng = np.random.default_rng([20261104, index, 6])
    lineshape = config.lineshape
    spectra = spectra_for(lineshape)
    widths = np.array([apodized_width(lineshape, i, config.lw) for i in range(2)])
    a, b, half = _true_layout(config)
    grid = grid_around(np.array(CENTRE, dtype=float), half)
    rows = [np.unique(g) for g in grid]
    true_peaks = [make_peak(lineshape, "A", a), make_peak(lineshape, "B", b)]
    true_lw = dict.fromkeys(_lw_names(), config.lw)
    truth = parameters(true_peaks, true_lw)
    empty = Cluster(1, true_peaks, grid, np.zeros((grid[0].size, PLANES)))
    shapes = empty.evaluate(truth)
    heights = np.array([HEIGHTS[0], HEIGHTS[1] * config.ratio])
    amplitudes = heights / shapes.max(axis=1)[:, None]
    clean = shapes.T @ amplitudes
    data = clean + _noise(config, rows, noise_rng)

    model = read_noise_model(spectra.dic, spectra.data.shape)
    correlation = cluster_correlation(model, grid)
    if config.start == "truth":
        start_points, start_lw = (a, b), true_lw
    else:
        jitter = start_rng.normal(0.0, JITTER, (2, 2)) * widths
        start_points, start_lw = (
            (a + jitter[0], b + jitter[1]),
            dict.fromkeys(_lw_names(), DEFAULT_LW),
        )
    completed = _fit(config, grid, data, model, start_points, start_lw)
    outcome = completed.outcome.cluster(1)
    row: dict = {"config": config.name, "index": index}
    if not outcome.usable or outcome.analytical_evaluation is None:
        return row | {"usable": False}
    problem = sampling_problem(completed, 1)
    names = list(problem.names)
    # Unbounded: the profile and the alternative refits may leave the fit's bounds.
    loose = problem.parameters.copy()
    for name in names:
        loose[name].min, loose[name].max = -np.inf, np.inf
    optimizer = VarProOptimizer(problem.cluster, names, loose, config.sigma)
    x = np.asarray(problem.start, dtype=float)
    true_x = np.array([truth[n].value for n in names])
    evaluation = outcome.analytical_evaluation

    # Published errors and the full sandwich covariance at the fit.
    published = np.array([problem.parameters[n].stderr for n in names], dtype=float)
    jacobian = optimizer.compute_jacobian(x)
    sandwich = identifiable_covariance(jacobian, correlation)
    white = identifiable_covariance(jacobian)
    chi2 = float(np.sum(optimizer.compute_residuals(x) ** 2))
    scales = np.linalg.norm(jacobian, axis=0)
    singular = np.linalg.svd(jacobian / scales, compute_uv=False)

    # The precision the experiment supports: the sandwich SE at the truth, noise-free.
    exact = VarProOptimizer(Cluster(1, true_peaks, grid, clean), names, truth, config.sigma)
    true_covariance = identifiable_covariance(exact.compute_jacobian(true_x), correlation)

    index_of = {n: j for j, n in enumerate(names)}
    d_a = [index_of[f"A.F{i}.cs"] for i in (2, 3)]
    d_b = [index_of[f"B.F{i}.cs"] for i in (2, 3)]
    selector = np.zeros((2, len(names)))
    selector[[0, 1], d_b] = 1.0
    selector[[0, 1], d_a] = -1.0

    def significance(vector, covariance):
        block = selector @ covariance @ selector.T
        if not np.all(np.isfinite(block)):
            return float("nan")
        return float(np.sqrt(vector @ np.linalg.solve(block, vector)))

    delta_ppm = selector @ x
    true_delta_ppm = selector @ true_x
    per_point = ppm_per_point(lineshape)
    fitted_widths = np.array(
        [
            np.mean([apodized_width(lineshape, i, x[index_of[f"{p}.F{i + 2}.lw"]]) for p in "AB"])
            for i in range(2)
        ]
    )
    in_widths = np.abs(delta_ppm) / per_point / fitted_widths

    # The overlap (cosine) of the two fitted 2D shapes, over a window holding both.
    midpoint = np.array(
        [float(spectra.spectral_params[i].ppm2pts(np.mean(x[[d_a[i], d_b[i]]]))) for i in range(2)]
    )
    wide = grid_around(midpoint, np.ceil(np.abs(delta_ppm) / per_point / 2 + 4 * fitted_widths))
    fitted_shapes = Cluster(
        1, problem.cluster.peaks, wide, np.zeros((wide[0].size, PLANES))
    ).evaluate(problem.parameters)
    norms = np.linalg.norm(fitted_shapes, axis=1)
    overlap = float(fitted_shapes[0] @ fitted_shapes[1] / (norms[0] * norms[1]))

    amplitude_covariance = evaluation.amplitude_covariance
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(np.diag(amplitude_covariance))
        rho = float(abs(amplitude_covariance[0, 1] / (sd[0] * sd[1])))

    z_sep = significance(delta_ppm, sandwich)
    z_white = significance(delta_ppm, white)
    profile = {}
    if np.isfinite(z_sep) and z_sep > 0:
        for label, scale in (
            ("inward", max(0.0, 1.0 - Z95 / z_sep)),
            ("outward", 1.0 + Z95 / z_sep),
        ):
            actual = _profile(optimizer, names, x, scale) - chi2
            predicted = (1.0 - scale) ** 2 * z_white**2
            profile[label] = [actual, predicted]

    # Alternative starts: the pair collapsed to a quarter, and spread to twice, the
    # fitted separation about the fitted midpoint, at the fitted linewidths.
    fitted_lw = {n: float(x[index_of[n]]) for n in _lw_names()}
    fitted_a = np.array([float(spectra.spectral_params[i].ppm2pts(x[d_a[i]])) for i in range(2)])
    fitted_b = np.array([float(spectra.spectral_params[i].ppm2pts(x[d_b[i]])) for i in range(2)])
    alternatives = {}
    for label, scale in (("collapsed", 0.25), ("spread", 2.0)):
        middle, half_delta = (fitted_a + fitted_b) / 2, (fitted_b - fitted_a) / 2
        start = (middle - scale * half_delta, middle + scale * half_delta)
        alternative = _fit(config, grid, data, model, start, fitted_lw)
        other = alternative.outcome.cluster(1)
        if not other.usable or other.analytical_evaluation is None:
            alternatives[label] = None
            continue
        alt_problem = sampling_problem(alternative, 1)
        alt_x = np.array([alt_problem.parameters[n].value for n in names])
        alt_chi2 = float(np.sum(optimizer.compute_residuals(alt_x) ** 2))  # same data, same metric
        alternatives[label] = {"estimates": alt_x.tolist(), "chi2": alt_chi2}

    return row | {
        "usable": True,
        "names": names,
        "warned": evaluation.uncertainty_warning is not None,
        "estimates": x.tolist(),
        "truth": true_x.tolist(),
        "published_se": published.tolist(),
        "true_se": np.sqrt(np.diag(true_covariance)).tolist(),
        "chi2": chi2,
        "true_separation_widths": float(
            np.linalg.norm(np.abs(true_delta_ppm) / per_point / widths)
        ),
        "true_significance": significance(true_delta_ppm, true_covariance),
        "fitted_separation_widths": float(np.linalg.norm(in_widths)),
        "fitted_separation_max_axis": float(np.max(in_widths)),
        "fitted_separation_ratio": float(
            np.linalg.norm(delta_ppm / per_point) / np.linalg.norm(true_delta_ppm / per_point)
        )
        if np.any(true_delta_ppm)
        else float("nan"),
        "shape_overlap": overlap,
        "amplitude_correlation": rho,
        "log10_condition": float(np.log10(singular[0] / singular[-1])),
        "separation_significance": z_sep,
        "profile": profile,
        "alternatives": alternatives,
    }


def _run(job: tuple[dict, int]) -> dict:
    config, index = job
    return realization(Config(**config), index)


def fit(args: argparse.Namespace) -> None:
    """Fit every realization of the sweep and write the records."""
    configs = sweep()
    if args.match:
        configs = [c for c in configs if all(m in c.name for m in args.match)]
    jobs = [
        (asdict(c), i)
        for c in configs
        for i in range(150 if c.points is not None else args.realizations)
    ]
    print(f"{len(configs)} configurations, {len(jobs)} realizations", file=sys.stderr)
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        rows = []
        for k, row in enumerate(pool.map(_run, jobs, chunksize=16)):
            rows.append(row)
            if (k + 1) % 2000 == 0:
                print(f"  {k + 1}/{len(jobs)}", file=sys.stderr)
    document = {
        "configs": [asdict(c) | {"name": c.name} for c in configs],
        "apodized_widths_points": {
            f"{ls} {'15N' if axis == 0 else '1H'}": {
                f"{lw:g} Hz": round(apodized_width(ls, axis, lw), 3) for lw in (10.0, 20.0, 40.0)
            }
            for ls in dict.fromkeys(c.lineshape for c in configs)
            for axis in (0, 1)
        },
        "fits": rows,
    }
    with gzip.open(args.output, "wt") as handle:
        json.dump(document, handle)


# --- Analysis (pre-registered in docs/validation/resolution-flag.md) ---------------

MISS = 3.0  # a problematic fit: a published position interval missed by > 3 SE
MIN_FITS = 20  # fewest unwarned fits for a configuration's calibration to be judged
TARGET_FP = 0.05  # false-positive budget when the threshold is chosen
GROSS_95 = 0.85  # after flagging, no configuration's unflagged fits may cover < 0.85


def _nonlinearity(profile: dict) -> float:
    """Largest |ln(actual / quadratic)| profile Δχ² over the two interval edges."""
    if not profile:
        return float("nan")
    worst = 0.0
    for actual, predicted in profile.values():
        if predicted <= 0:
            continue
        if actual <= 0:
            return float("inf")
        worst = max(worst, abs(np.log(actual / predicted)))
    return worst


def _disagreement(row: dict) -> float:
    """Largest position shift, in published SEs, to a refit no worse by Δχ² > 1."""
    x = np.array(row["estimates"])
    se = np.array(row["published_se"])
    positions = [j for j, n in enumerate(row["names"]) if n.endswith(".cs")]
    shift = 0.0
    for alternative in row["alternatives"].values():
        if alternative is None or alternative["chi2"] > row["chi2"] + 1.0:
            continue
        moved = np.abs(np.array(alternative["estimates"]) - x)[positions] / se[positions]
        shift = max(shift, float(np.max(moved)))
    return shift


# name: (row field or function, True when large values flag)
MEASURES = {
    "fitted separation / apodized width": ("fitted_separation_widths", False),
    "fitted separation / apodized width, largest axis": ("fitted_separation_max_axis", False),
    "fitted shape overlap": ("shape_overlap", True),
    "amplitude correlation": ("amplitude_correlation", True),
    "log10 scaled Jacobian condition": ("log10_condition", True),
    "separation significance": ("separation_significance", False),
    "profile nonlinearity": (lambda r: _nonlinearity(r["profile"]), True),
    "alternative-start disagreement": (_disagreement, True),
}
ORACLES = {  # not available to a user: the truth, to test resolution itself
    "true separation / apodized width (oracle)": ("true_separation_widths", False),
    "true separation significance (oracle)": ("true_significance", False),
}


def _table(document: dict) -> dict:
    """Per-fit arrays for usable fits, and per-configuration calibration labels."""
    configs = {c["name"]: c for c in document["configs"]}
    rows = [r for r in document["fits"] if r["usable"]]
    out: dict = {
        "config": np.array([r["config"] for r in rows]),
        "index": np.array([r["index"] for r in rows]),
        "warned": np.array([r["warned"] for r in rows]),
    }
    z_pub, z_true = [], []
    for r in rows:
        positions = [j for j, n in enumerate(r["names"]) if n.endswith(".cs")]
        error = np.abs(np.array(r["estimates"]) - np.array(r["truth"]))[positions]
        z_pub.append(error / np.array(r["published_se"], dtype=float)[positions])
        z_true.append(error / np.array(r["true_se"], dtype=float)[positions])
    out["z_published"] = np.array(z_pub)
    out["z_true"] = np.array(z_true)
    with np.errstate(invalid="ignore"):
        out["problematic"] = np.nanmax(out["z_published"], axis=1) > MISS
        out["biased"] = np.max(out["z_true"], axis=1) > MISS  # NaN at coincidence: False
    for name, (field, _) in (MEASURES | ORACLES).items():
        getter = field if callable(field) else (lambda r, f=field: r[f])
        out[name] = np.array([getter(r) for r in rows], dtype=float)
    out["fitted_separation_ratio"] = np.array([r["fitted_separation_ratio"] for r in rows])
    calibrated, cells = {}, {}
    for name in configs:
        keep = (out["config"] == name) & ~out["warned"]
        cell = _coverage(out["z_published"][keep])
        total = int(np.sum(out["config"] == name))
        cell |= {
            "fits": sum(r["config"] == name for r in document["fits"]),
            "usable": total,
            "warned": int(np.sum((out["config"] == name) & out["warned"])),
            "problematic_unwarned": int(np.sum(keep & out["problematic"])),
            "biased_unwarned": int(np.sum(keep & out["biased"])),
            "median_fitted_over_true_separation": _median(
                out["fitted_separation_ratio"][out["config"] == name]
            ),
        }
        cells[name] = cell
        calibrated[name] = cell["calibrated"]
    out["calibrated"] = np.array([calibrated[c] for c in out["config"]], dtype=object)
    out["resolved"] = np.array(
        [configs[c]["separation"] >= 1.0 and configs[c]["points"] is None for c in out["config"]]
    )
    return {"fits": out, "cells": cells, "configs": configs}


def _median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def _coverage(z: np.ndarray) -> dict:
    """Pooled position coverage; calibrated unless below nominal by > 3 binomial SE.

    n is the number of fits, not of positions: the positions of one fit are
    correlated, so this is the conservative count. Over-coverage is not a failure.
    """
    n = len(z)
    if n < MIN_FITS:
        return {"unwarned": n, "coverage_68_95": None, "calibrated": None}
    low, high = float(np.mean(z < 1.0)), float(np.mean(z < Z95))
    ok = low >= 0.68 - 3 * np.sqrt(0.68 * 0.32 / n) and high >= 0.95 - 3 * np.sqrt(0.95 * 0.05 / n)
    return {"unwarned": n, "coverage_68_95": [low, high], "calibrated": bool(ok)}


def _flags(values: np.ndarray, threshold: float, large: bool) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        flagged = values >= threshold if large else values <= threshold
    return flagged | ~np.isfinite(values) & ~np.isnan(values)


def _rates(fits: dict, flagged: np.ndarray, keep: np.ndarray) -> dict:
    unwarned = keep & ~fits["warned"]
    problematic = unwarned & fits["problematic"]
    calibrated = unwarned & (fits["calibrated"] == True)  # noqa: E712
    clean = unwarned & ~fits["problematic"]
    resolved = unwarned & fits["resolved"] & (fits["calibrated"] == True)  # noqa: E712

    def share(mask):
        return float(np.mean(flagged[mask])) if mask.any() else None

    return {
        "sensitivity": share(problematic),
        "sensitivity_biased": share(unwarned & fits["biased"]),
        "false_positive_calibrated": share(calibrated),
        "false_positive_not_problematic": share(clean),
        "false_positive_resolved_calibrated": share(resolved),
        "flagged_unwarned": share(unwarned),
        "problematic_fits": int(problematic.sum()),
        "calibrated_fits": int(calibrated.sum()),
    }


def _after_flagging(fits: dict, cells: dict, flagged: np.ndarray, keep: np.ndarray) -> dict:
    """Calibration of the unwarned, unflagged fits, configuration by configuration."""
    failing, gross, judged = [], [], 0
    pooled = []
    for name, cell in cells.items():
        mask = keep & (fits["config"] == name) & ~fits["warned"] & ~flagged
        result = _coverage(fits["z_published"][mask])
        if result["calibrated"] is None:
            continue
        judged += 1
        if not result["calibrated"]:
            failing.append([name, result["coverage_68_95"], result["unwarned"]])
        if result["coverage_68_95"][1] < GROSS_95:
            gross.append([name, result["coverage_68_95"], result["unwarned"]])
        if cell["calibrated"] is False:
            pooled.append(fits["z_published"][mask])
    previously = _coverage(np.concatenate(pooled)) if pooled else None
    return {
        "configurations_judged": judged,
        "failing": failing,
        "grossly_under_covering": gross,
        "pooled_over_previously_miscalibrated": previously,
    }


def _choose(values, large, fits, keep) -> float | None:
    """The threshold of most sensitivity with ≤ 5% of calibrated fits flagged."""
    unwarned = keep & ~fits["warned"]
    finite = values[unwarned & np.isfinite(values)]
    if not finite.size:
        return None
    candidates = np.unique(np.quantile(finite, np.linspace(0, 1, 801)))
    best, best_sensitivity = None, -1.0
    for threshold in candidates:
        rates = _rates(fits, _flags(values, threshold, large), keep)
        fp = rates["false_positive_calibrated"]
        if fp is not None and fp <= TARGET_FP and rates["sensitivity"] > best_sensitivity:
            best, best_sensitivity = float(threshold), rates["sensitivity"]
    return best


def _roc(values, large, fits, keep) -> dict:
    """Sensitivity at fixed false-positive rates, and the AUC, on ``keep``."""
    unwarned = keep & ~fits["warned"]
    positive = values[unwarned & fits["problematic"]]
    negative = values[unwarned & (fits["calibrated"] == True)]  # noqa: E712
    if not large:
        positive, negative = -positive, -negative
    positive = np.nan_to_num(positive, nan=-np.inf, posinf=np.inf)
    negative = np.nan_to_num(negative, nan=-np.inf, posinf=np.inf)
    out = {}
    for fp in (0.01, 0.05, 0.1, 0.2):
        threshold = np.quantile(negative, 1 - fp)
        out[f"sensitivity_at_fp_{fp:g}"] = float(np.mean(positive > threshold))
    ranks = np.searchsorted(np.sort(negative), positive, side="left") + 0.5 * (
        np.searchsorted(np.sort(negative), positive, side="right")
        - np.searchsorted(np.sort(negative), positive, side="left")
    )
    out["auc"] = float(np.mean(ranks) / negative.size)
    return out


def _breakdown(table: dict) -> dict:
    """Where the unwarned problematic fits are, by each sweep factor."""
    fits, configs = table["fits"], table["configs"]
    unwarned = ~fits["warned"]
    out = {}
    for factor in ("lineshape", "geometry", "separation", "lw", "ratio", "sigma", "start"):
        levels: dict = {}
        for name, config in configs.items():
            if config["points"] is not None:
                continue
            level = str(config[factor])
            mask = (fits["config"] == name) & unwarned
            entry = levels.setdefault(level, [0, 0, 0])
            entry[0] += int(mask.sum())
            entry[1] += int((mask & fits["problematic"]).sum())
            entry[2] += int(np.sum((fits["config"] == name) & fits["warned"]))
        out[factor] = {
            k: {"unwarned": v[0], "problematic": v[1], "warned": v[2]} for k, v in levels.items()
        }
    return out


def analyze(args: argparse.Namespace) -> None:
    """Evaluate every measure against the pre-registered criteria."""
    with gzip.open(args.fits, "rt") as handle:
        document = json.load(handle)
    table = _table(document)
    fits, cells = table["fits"], table["cells"]
    dev = fits["index"] % 2 == 0
    test = ~dev
    everything = np.ones_like(dev)
    no_flag = np.zeros_like(dev)
    report: dict = {
        "fits": len(document["fits"]),
        "usable": len(fits["warned"]),
        "warned": int(fits["warned"].sum()),
        "unwarned_problematic": int((~fits["warned"] & fits["problematic"]).sum()),
        "unwarned_biased": int((~fits["warned"] & fits["biased"]).sum()),
        "configurations": len(cells),
        "configurations_calibrated": sum(c["calibrated"] is True for c in cells.values()),
        "configurations_miscalibrated": sum(c["calibrated"] is False for c in cells.values()),
        "without_a_flag": _after_flagging(fits, cells, no_flag, test),
        # Context, not a candidate: flag every fit of a miscalibrated configuration.
        # Problematic fits in calibrated configurations are beyond any such flag.
        "configuration_oracle_test": _rates(fits, fits["calibrated"] == False, test),  # noqa: E712
        "breakdown": _breakdown(table),
        "apodized_widths_points": document["apodized_widths_points"],
        "measures": {},
    }
    for name, (_, large) in (MEASURES | ORACLES).items():
        values = fits[name]
        threshold = _choose(values, large, fits, dev)
        entry: dict = {
            "flags_when": "at or above" if large else "at or below",
            "roc_all": _roc(values, large, fits, everything),
            "threshold_from_development_half": threshold,
        }
        if threshold is not None:
            flagged = _flags(values, threshold, large)
            entry["development"] = _rates(fits, flagged, dev)
            entry["test"] = _rates(fits, flagged, test)
            entry["test_after_flagging"] = _after_flagging(fits, cells, flagged, test)
            t = entry["test"]
            after = entry["test_after_flagging"]
            fp = t["false_positive_calibrated"]
            n_fp = t["calibrated_fits"]
            entry["verdict"] = {
                "sensitivity_at_least_0.8": t["sensitivity"] >= 0.8,
                "false_positive_at_most_0.05": fp <= TARGET_FP + 3 * np.sqrt(0.05 * 0.95 / n_fp),
                "no_gross_under_coverage_after_flagging": not after["grossly_under_covering"],
                "previously_miscalibrated_now_calibrated": bool(
                    after["pooled_over_previously_miscalibrated"]
                    and after["pooled_over_previously_miscalibrated"]["calibrated"]
                ),
            }
            entry["verdict"]["passes"] = all(entry["verdict"].values())
        report["measures"][name] = entry
    report["cells"] = cells
    text = json.dumps(report, indent=1, default=float)
    if args.output:
        args.output.write_text(text + "\n")
    summary = {
        name: {
            "threshold": m["threshold_from_development_half"],
            "auc": round(m["roc_all"]["auc"], 3),
            "test": {k: v for k, v in m.get("test", {}).items() if "fits" not in k},
            "passes": m.get("verdict", {}).get("passes"),
        }
        for name, m in report["measures"].items()
    }
    print(json.dumps(summary, indent=1, default=float))


def main() -> None:
    """Fit the sweep, or analyze recorded fits."""
    parser = argparse.ArgumentParser(description="Resolution-based review flag study.")
    commands = parser.add_subparsers(required=True)
    run = commands.add_parser("fit", help="fit the sweep and write every fit's record")
    run.add_argument("--realizations", type=int, default=80)
    run.add_argument("--workers", type=int, default=16)
    run.add_argument("--match", nargs="*", help="only configurations whose names hold all")
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(func=fit)
    study = commands.add_parser("analyze", help="evaluate the measures on the recorded fits")
    study.add_argument("fits", type=Path)
    study.add_argument("--output", type=Path)
    study.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
