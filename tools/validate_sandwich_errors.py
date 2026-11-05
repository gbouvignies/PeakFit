"""Sandwich errors validated with PeakFit's own fitting engine (correlated noise, step 2).

See ``docs/correlated-noise.md`` §11. Each case is a 2D pseudo-3D cluster (four
planes) drawn with PeakFit's own ``sp2`` lineshapes, plus noise from the exact,
separable processing operators that the NMRPipe header describes:
- ¹⁵N: 32 → 128 points, sine bell² (off 0.4), P1 180°;
- ¹H: 64 → 256 points, sine bell² (off 0.3), first point 0.5.

``read_noise_model`` reads that header, and ``cluster_correlation`` gives R. Every
realization is fitted by ``run_pipeline``, starting at the truth. The tool then
compares, over realizations:
- **Nonlinear errors:** σ-only (as published) and sandwich SEs against the scatter
  of the fitted positions and linewidths, with 68% and 95% coverage.
- **Conditional amplitude errors:** amplitudes solved at the true lineshape, so the
  comparison isolates the conditional error.
- **Total against conditional:** the scatter of the jointly fitted amplitudes, over
  the conditional sandwich SD.
- **Withholding:** σ-only and sandwich covariances must be withheld in exactly the
  same realizations.

The white-noise control (no zero fill, no apodization, so R = I) must reproduce
σ-only to round-off.

**Pass criteria, fixed in advance:**
- For the isolated and overlapping cases, sandwich coverage within 3 binomial SE of
  68% and 95%, and scatter / sandwich SD within 0.9–1.1.
- No withholding mismatch in any case.
- R = I reproduces σ-only to 10⁻¹⁰.

    uv run python tools/validate_sandwich_errors.py --output /tmp/sandwich.json
"""

# ruff: noqa: T201

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from peakfit.engine.algorithms.linear_algebra import (
    correlated_amplitude_covariance,
    factorize_shapes,
)
from peakfit.engine.algorithms.noise_correlation import (
    AxisProcessing,
    cluster_correlation,
)
from peakfit.engine.algorithms.varpro import (
    VarProOptimizer,
    _calculate_errors,
    identifiable_covariance,
)
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, LineshapeName, PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.scientific.sampling import sampling_problem
from peakfit.fit.pipeline import run_pipeline
from peakfit.io.readers.spectrum import read_noise_model

PLANES = 4
SIGMA = 0.02
HEIGHTS = np.array([[1.0, 0.8, 0.6, 0.9], [0.7, 0.9, 1.0, 0.5]])  # peak heights per plane
HALF_WINDOW = (5, 6)  # ¹⁵N, ¹H points either side of the cluster centre
LINEWIDTH = 20.0  # true linewidth of every line, Hz
Z95 = 1.96


@dataclass(frozen=True)
class Axis:
    """One spectral axis: its processing (as NMRPipe records it) and calibration."""

    processing: AxisProcessing
    sw: float
    obs: float

    def header(self, fdf: str) -> dict[str, float]:
        """The NMRPipe header fields that describe this processing."""
        p = self.processing
        return {
            f"{fdf}FTFLAG": 1.0,
            f"{fdf}TDSIZE": float(p.acquired),
            f"{fdf}FTSIZE": float(p.transform),
            f"{fdf}APOD": float(p.acquired),
            f"{fdf}APODCODE": 1.0 if p.window == "sine_bell" else 0.0,
            f"{fdf}APODQ1": p.offset,
            f"{fdf}APODQ2": p.end,
            f"{fdf}APODQ3": p.power,
            f"{fdf}C1": p.first_point - 1.0,
            f"{fdf}P1": p.phase,
        }

    def parameters(self, direct: bool) -> SpectralParameters:
        """The calibration PeakFit's ``sp2`` lineshape is built from."""
        p = self.processing
        sine = p.window == "sine_bell"
        return SpectralParameters(
            size=p.transform,
            sw=self.sw,
            obs=self.obs,
            car=0.0,
            aq_time=p.acquired / self.sw,
            apocode=1.0 if sine else 0.0,
            apodq1=p.offset if sine else 0.0,
            apodq2=p.end if sine else 0.0,
            apodq3=p.power if sine else 0.0,
            p180=False,
            direct=direct,
            ft=True,
            td_size=p.acquired,
            ft_size=p.transform,
        )

    def noise_operator(self, rows: np.ndarray) -> np.ndarray:
        """Real processed noise at ``rows`` from 2N real white inputs, unit variance."""
        p = self.processing
        lags = np.arange(p.transform)[rows]
        fourier = np.exp(-2j * np.pi * np.outer(lags, np.arange(p.acquired)) / p.transform)
        # NMRPipe's sign: its P1 ramp over stored points is e^{−iφk/M} in index k.
        phase = np.exp(-1j * np.deg2rad(p.phase) * lags / p.transform)
        operator = phase[:, None] * fourier * p.weights()[None, :]
        real = np.hstack([operator.real, -operator.imag])
        return real / np.sqrt(np.sum(p.weights() ** 2))


def _chain(power: float | None) -> tuple[Axis, Axis]:
    """¹⁵N and ¹H processed with a sine bell to ``power``, or without apodization."""
    if power is None:
        return (
            Axis(AxisProcessing(32, 128, "none", phase=180.0), 1500.0, 81.0),
            Axis(AxisProcessing(64, 256, "none", first_point=0.5), 4000.0, 800.0),
        )
    return (
        Axis(AxisProcessing(32, 128, "sine_bell", 0.4, 0.98, power, phase=180.0), 1500.0, 81.0),
        Axis(
            AxisProcessing(64, 256, "sine_bell", 0.3, 0.98, power, first_point=0.5), 4000.0, 800.0
        ),
    )


SP2, SP1, NO_APOD = _chain(2.0), _chain(1.0), _chain(None)
WHITE = (
    Axis(AxisProcessing(128, 128, "none"), 1500.0, 81.0),
    Axis(AxisProcessing(256, 256, "none"), 4000.0, 800.0),
)
# Each case: separation in (¹⁵N, ¹H) points (None: a single peak), the lineshape, the
# processing of the signal (and of PeakFit's lineshape), and the processing the header
# records and the noise follows. Only the white control differs between the two.
CASES: dict[
    str, tuple[tuple[float, float] | None, LineshapeName, tuple[Axis, Axis], tuple[Axis, Axis]]
] = {
    "isolated peak": (None, "sp2", SP2, SP2),
    "overlapping pair": ((2.0, 2.0), "sp2", SP2, SP2),
    "isolated peak, sp1": (None, "sp1", SP1, SP1),
    "overlapping pair, sp1": ((2.0, 2.0), "sp1", SP1, SP1),
    "isolated peak, no_apod": (None, "no_apod", NO_APOD, NO_APOD),
    "overlapping pair, no_apod": ((2.0, 2.0), "no_apod", NO_APOD, NO_APOD),
    "near-coalescent pair": ((0.4, 0.4), "sp2", SP2, SP2),
    "duplicated peak": ((0.0, 0.0), "sp2", SP2, SP2),
    "white-noise control (R = I)": ((2.0, 2.0), "sp2", SP2, WHITE),
}
CALIBRATED = [case for case in CASES if case.startswith(("isolated", "overlapping"))]


def build(case: str, rng: np.random.Generator | None):
    """The spectrum, cluster, true parameters and amplitudes of one realization."""
    separation, lineshape, signal, (first, second) = CASES[case]
    dic = {
        "FDDIMORDER": [2.0, 1.0, 3.0, 4.0],
        "FDDMXVAL": 0.0,  # NMRPipe headers always record the digital-filter state
        "FDDMXFLAG": 0.0,
        **first.header("FDF1"),
        **second.header("FDF2"),
    }
    pseudo = SpectralParameters(
        size=PLANES, sw=1.0, obs=1.0, car=0.0, aq_time=1.0, apocode=0, apodq1=0, apodq2=0,
        apodq3=0, p180=False, direct=False, ft=False,
    )  # fmt: skip
    shape = (PLANES, first.processing.transform, second.processing.transform)
    spectra = Spectra(
        dic=dic,
        data=np.zeros(shape),
        z_values=np.arange(float(PLANES)),
        params=[pseudo, signal[0].parameters(False), signal[1].parameters(True)],
    )
    centre = (shape[1] / 2, shape[2] / 2)
    offsets = [(0.0, 0.0)] if separation is None else [
        (-separation[0] / 2, -separation[1] / 2),
        (separation[0] / 2, separation[1] / 2),
    ]  # fmt: skip
    peaks = []
    for name, (d1, d2) in zip("AB", offsets, strict=False):
        position = [
            float(spectra.spectral_params[0].pts2ppm(centre[0] + d1)),
            float(spectra.spectral_params[1].pts2ppm(centre[1] + d2)),
        ]
        config = FitConfig(lineshape=lineshape)
        shapes = create_shapes(spectra, config, name, position, [lineshape] * 2)
        peak = Peak(name, np.array(position), shapes)
        peak.set_cluster_id(1)
        peaks.append(peak)
    rows = [
        np.arange(round(c) - h, round(c) + h + 1) for c, h in zip(centre, HALF_WINDOW, strict=True)
    ]
    grid = [index.ravel() for index in np.meshgrid(*rows, indexing="ij")]
    params = Parameters.from_peaks(peaks)
    for name in params.get_vary_names():
        if name.endswith(".lw"):
            params[name].value = LINEWIDTH
    empty = Cluster(1, peaks, grid, np.zeros((grid[0].size, PLANES)))
    shapes = empty.evaluate(params)
    # PeakFit's shapes are not height-normalized: scale each amplitude to its height.
    amplitudes = HEIGHTS[: len(peaks)] / shapes.max(axis=1)[:, None]
    data = shapes.T @ amplitudes
    if rng is not None:
        left, right = first.noise_operator(rows[0]), second.noise_operator(rows[1])
        noise = [
            (left @ rng.standard_normal((left.shape[1], right.shape[1])) @ right.T).ravel()
            for _ in range(PLANES)
        ]
        data = data + SIGMA * np.stack(noise, axis=1)
    return spectra, Cluster(1, peaks, grid, data), params, amplitudes


def realization(case: str, index: int) -> dict:
    """Fit one realization as PeakFit publishes it; return its errors and estimates."""
    rng = np.random.default_rng([20261104, index, list(CASES).index(case)])
    spectra, cluster, truth, amplitudes = build(case, rng)
    model = read_noise_model(spectra.dic, spectra.data.shape)
    if not model.modelled:
        raise ValueError(model.reason)
    correlation = cluster_correlation(model, cluster.grid_indices)
    config = PeakFitConfig(fitting=FitConfig(lineshape=CASES[case][1]))
    completed = run_pipeline(
        config, [cluster], SIGMA, truth.copy(), noise_model=model
    ).completed_fit
    outcome = completed.outcome.cluster(1)
    if not outcome.usable or outcome.analytical_evaluation is None:
        return {"usable": False}
    problem = sampling_problem(completed, 1)
    optimizer = VarProOptimizer(problem.cluster, list(problem.names), problem.parameters, SIGMA)
    x = problem.start
    independent = _calculate_errors(optimizer, x, None)
    sandwich = _calculate_errors(optimizer, x, correlation)
    jacobian = optimizer.compute_jacobian(x)
    covariance_withheld = [
        bool(np.isnan(identifiable_covariance(jacobian, None)).any()),
        bool(np.isnan(identifiable_covariance(jacobian, correlation)).any()),
    ]
    published = np.array([problem.parameters[n].stderr for n in problem.names])
    # Conditional amplitudes: solved at the true lineshape.
    exact = factorize_shapes(problem.cluster.evaluate(truth))
    conditional = exact.solve(problem.data)
    # The fit's own amplitude covariance at θ̂, for the withholding comparison.
    at_fit = factorize_shapes(problem.cluster.evaluate(problem.parameters))
    evaluation = outcome.analytical_evaluation
    published_amplitude = SIGMA * np.sqrt(
        np.diag(correlated_amplitude_covariance(at_fit, correlation))
    )
    covariance = evaluation.amplitude_covariance
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(np.diag(covariance))
        rho = covariance / np.outer(sd, sd)
    off_diagonal = rho[~np.eye(len(sd), dtype=bool)]
    return {
        "usable": True,
        "names": list(problem.names),
        "estimates": x.tolist(),
        "truth": [truth[n].value for n in problem.names],
        # PeakFit publishes the sandwich: nonlinear errors and conditional amplitudes.
        "published_matches_sandwich": bool(
            np.allclose(published, sandwich, rtol=1e-10, equal_nan=True)
            and np.allclose(
                evaluation.amplitude_standard_errors,
                published_amplitude,
                rtol=1e-10,
                equal_nan=True,
            )
        ),
        "warned": evaluation.uncertainty_warning is not None,
        "max_abs_amplitude_correlation": (
            float(np.max(np.abs(off_diagonal))) if off_diagonal.size else 0.0
        ),
        "independent_se": independent.tolist(),
        "sandwich_se": sandwich.tolist(),
        "conditional": conditional.tolist(),
        "fitted_amplitudes": evaluation.amplitudes.tolist(),
        "amplitudes": amplitudes.tolist(),
        "conditional_independent_sd": (SIGMA * np.sqrt(np.diag(exact.covariance))).tolist(),
        "conditional_sandwich_sd": (
            SIGMA * np.sqrt(np.diag(correlated_amplitude_covariance(exact, correlation)))
        ).tolist(),
        "amplitude_withheld": [
            bool(np.isnan(at_fit.covariance).any()),
            bool(np.isnan(correlated_amplitude_covariance(at_fit, correlation)).any()),
        ],
        "covariance_withheld": covariance_withheld,
        "nonlinear_withheld": [np.isnan(independent).tolist(), np.isnan(sandwich).tolist()],
    }


def _stats(errors: np.ndarray, sd: np.ndarray) -> dict:
    """Scatter / SD for each quantity (column), and coverage pooled through z.

    Columns are kept apart because their scales differ (¹⁵N and ¹H positions in ppm
    differ about eightfold): pooling raw errors would compare unlike quantities.
    """
    ratios = [
        float(np.std(errors[ok, c]) / np.median(sd[ok, c]))
        for c in range(errors.shape[1])
        if (ok := np.isfinite(sd[:, c])).sum() > 1
    ]
    finite = np.isfinite(sd)
    z = np.abs(errors[finite]) / sd[finite]
    return {
        "n": int(finite.sum()),
        "scatter_over_sd_per_quantity": ratios,
        "coverage_68_95": [float(np.mean(z < 1.0)), float(np.mean(z < Z95))] if z.size else None,
    }


def summarize(case: str, rows: list[dict]) -> dict:
    """Coverage and scatter per kind of quantity, and the withholding comparison."""
    usable = [r for r in rows if r["usable"]]
    names = usable[0]["names"]
    estimates = np.array([r["estimates"] for r in usable])
    truth = np.array(usable[0]["truth"])
    independent = np.array([r["independent_se"] for r in usable])
    sandwich = np.array([r["sandwich_se"] for r in usable])
    out: dict = {
        "realizations": len(rows),
        "usable": len(usable),
        "published_matches_sandwich": all(r["published_matches_sandwich"] for r in usable),
        "warned": sum(r["warned"] for r in usable),
    }
    for kind in ("cs", "lw"):
        columns = [j for j, n in enumerate(names) if n.endswith(f".{kind}")]
        errors = (estimates - truth)[:, columns]
        out[kind] = {
            "sigma_only": _stats(errors, independent[:, columns]),
            "sandwich": _stats(errors, sandwich[:, columns]),
        }
    conditional = np.array([r["conditional"] for r in usable])  # (n, peaks, planes)
    n, _, planes = conditional.shape
    errors = (conditional - np.array(usable[0]["amplitudes"])).reshape(n, -1)

    def per_point(key):
        values = np.array([r[key] for r in usable])[:, :, None]
        return np.repeat(values, planes, axis=2).reshape(n, -1)

    out["conditional_amplitude"] = {
        "sigma_only": _stats(errors, per_point("conditional_independent_sd")),
        "sandwich": _stats(errors, per_point("conditional_sandwich_sd")),
    }
    fitted = np.array([r["fitted_amplitudes"] for r in usable]).reshape(n, -1)
    sd = per_point("conditional_sandwich_sd")
    if np.isfinite(sd).all():
        out["total_over_conditional_amplitude_sd"] = float(
            np.median(fitted.std(axis=0) / np.median(sd, axis=0))
        )
    out["withholding"] = {
        "covariance_withheld": sum(r["covariance_withheld"][0] for r in usable),
        "covariance_mismatches": sum(
            r["covariance_withheld"][0] != r["covariance_withheld"][1] for r in usable
        ),
        "amplitude_covariance_withheld": sum(r["amplitude_withheld"][0] for r in usable),
        "amplitude_mismatches": sum(
            r["amplitude_withheld"][0] != r["amplitude_withheld"][1] for r in usable
        ),
        # The bound-crossing rule applies to the published (sandwich) SE, so a larger
        # SE crosses a bound more often: expected, and reported, not a mismatch.
        "bound_rule_withheld_sigma_only": int(
            sum(np.sum(r["nonlinear_withheld"][0]) for r in usable)
        ),
        "bound_rule_withheld_sandwich": int(
            sum(np.sum(r["nonlinear_withheld"][1]) for r in usable)
        ),
    }
    # Positions of unwarned realizations, by the largest amplitude |ρ| at the fit:
    # what a reader sees presented without a caution.
    rho = np.array([r["max_abs_amplitude_correlation"] for r in usable])
    warned = np.array([r["warned"] for r in usable])
    positions = [j for j, n in enumerate(names) if n.endswith(".cs")]
    for label, keep in (
        ("unwarned", ~warned),
        ("unwarned_rho_below_0.9", ~warned & (rho < 0.9)),  # noqa: PLR2004
        ("unwarned_rho_at_least_0.9", ~warned & (rho >= 0.9)),  # noqa: PLR2004
    ):
        if keep.any():
            out[f"positions_{label}"] = _stats(
                (estimates - truth)[keep][:, positions], sandwich[keep][:, positions]
            ) | {"realizations": int(keep.sum())}
    out["max_abs_amplitude_correlation_median"] = float(np.median(rho))
    if case.startswith("white"):
        ratio = sandwich / independent
        finite = np.isfinite(ratio)
        out["max_relative_difference_sandwich_vs_sigma_only"] = float(
            np.max(np.abs(ratio[finite] - 1.0))
        )
    return out


def verdict(report: dict) -> dict:
    """The pre-registered pass criteria, case by case."""
    checks = {}
    for case, s in report.items():
        withholding = s["withholding"]
        result = {
            "withholding_identical": withholding["amplitude_mismatches"] == 0
            and withholding["covariance_mismatches"] == 0,
            "published_is_the_sandwich": s["published_matches_sandwich"],
        }
        if case in CALIBRATED:
            for quantity in ("cs", "lw", "conditional_amplitude"):
                stats = s[quantity]["sandwich"]
                n = stats["n"]
                low, high = stats["coverage_68_95"]
                result[quantity] = bool(
                    abs(low - 0.68) <= 3 * np.sqrt(0.68 * 0.32 / n)
                    and abs(high - 0.95) <= 3 * np.sqrt(0.95 * 0.05 / n)
                    and all(0.9 <= r <= 1.1 for r in stats["scatter_over_sd_per_quantity"])  # noqa: PLR2004
                )
        if case.startswith("white"):
            result["sandwich_equals_sigma_only"] = (
                s["max_relative_difference_sandwich_vs_sigma_only"] < 1e-10  # noqa: PLR2004
            )
        checks[case] = result
    return checks


def main() -> None:
    """Run every case in parallel and report against the pass criteria."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--realizations", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cases", nargs="*", choices=list(CASES), default=list(CASES))
    parser.add_argument(
        "--first-index", type=int, default=0, help="seed index of the first realization"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    counts = {case: args.realizations if case in CALIBRATED else 150 for case in args.cases}
    jobs = [
        (case, i)
        for case in args.cases
        for i in range(args.first_index, args.first_index + counts[case])
    ]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(realization, *zip(*jobs, strict=True), chunksize=4))
    report = {
        case: summarize(case, [r for (c, _), r in zip(jobs, rows, strict=True) if c == case])
        for case in args.cases
    }
    result = {"cases": report, "verdict": verdict(report)}
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
