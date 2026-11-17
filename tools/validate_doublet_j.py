"""Evidence that doublet fits estimate J with a calibrated σ(J) (validation/doublet-j.md).

A 1D doublet over three planes (profile 1 : 0.6 : 0.3) is fitted by the VarPro
optimizer from a J start 20 % below the truth, across noise realizations. Every case
is scaled to a stated first-plane peak signal-to-noise ratio, so lineshapes with
different kernel heights are compared at equal information.

Each case reports how far J moved, the bias and scatter of the estimates, how often
σ(J) was withheld, and how often the ±σ and ±1.96σ intervals cover the true J,
counted only where σ(J) was reported. Cases span resolved, partly resolved and
unresolved doublets and a ¹⁵N-like one-bond coupling, at high and low SNR, for a
real-valued (Lorentzian) and a phased, apodized (SP1) lineshape.

    uv run python tools/validate_doublet_j.py --output docs/validation/doublet-j.json

Run on a commit without the J derivative, the same tool records the former behaviour.
"""

# ruff: noqa: T201

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from peakfit.engine.algorithms.varpro import fit_cluster
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, LineshapeName
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.lineshapes.create import create_shapes

REALIZATIONS = 200
NOISE = 2.0
PROFILE = np.array([[1.0, 0.6, 0.3]])
# name: (true J in Hz, true linewidth in Hz, first-plane peak height / noise)
REGIMES = {
    "resolved (J = 5 lw)": (40.0, 8.0, 50.0),
    "partly resolved (J = 1.2 lw)": (12.0, 10.0, 50.0),
    "unresolved (J = 0.25 lw)": (5.0, 20.0, 50.0),
    "one-bond coupling (J = 92 Hz)": (92.0, 15.0, 50.0),
    "partly resolved, low SNR": (12.0, 10.0, 8.0),
    "unresolved, low SNR": (5.0, 20.0, 8.0),
}
LINESHAPES: tuple[LineshapeName, ...] = ("lorentzian_doublet", "sp1_doublet")


def _axis(size: int, *, direct: bool, sw: float, obs: float, label: str) -> SpectralParameters:
    return SpectralParameters(
        size=size,
        sw=sw,
        obs=obs,
        car=0.0,
        aq_time=size / sw,
        apocode=1.0,
        apodq1=0.35,
        apodq2=0.98,
        apodq3=1.0,
        p180=False,
        direct=direct,
        ft=True,
        label=label,
    )


def _problem(lineshape: LineshapeName, j_hz: float, lw_hz: float):
    axis = _axis(512, direct=True, sw=3000.0, obs=600.0, label="F2")
    planes = _axis(3, direct=False, sw=1.0, obs=1.0, label="F1")
    spectra = Spectra(
        dic={}, data=np.zeros((3, 512)), z_values=np.arange(3.0), params=[planes, axis]
    )
    centre = float(axis.pts2ppm(256.0))
    phased = lineshape.startswith("sp")
    config = FitConfig(lineshape=lineshape, fit_phase=["F2"] if phased else [])
    peak = Peak("A", np.array([centre]), create_shapes(spectra, config, "A", [centre], [lineshape]))
    peak.set_cluster_id(1)
    truth = Parameters.from_peaks([peak])
    for name, parameter in truth.items():
        if name.endswith(".j"):
            parameter.min, parameter.max, parameter.value = 0.5, 150.0, j_hz
        elif name.endswith(".lw"):
            parameter.value = lw_hz
        elif name.endswith(".phase"):
            parameter.value = 4.0
    cluster = Cluster(1, [peak], [np.arange(512)], np.zeros((512, 3)))
    shape = cluster.evaluate(truth).T
    clean = shape @ PROFILE
    return cluster, truth, clean / np.max(np.abs(clean[:, 0]))


def _run(task: tuple[LineshapeName, str, int]) -> dict:
    lineshape, regime, seed = task
    j_hz, lw_hz, snr = REGIMES[regime]
    cluster, truth, clean = _problem(lineshape, j_hz, lw_hz)
    rng = np.random.default_rng(seed)
    cluster.data = snr * NOISE * clean + NOISE * rng.normal(size=clean.shape)
    params = truth.copy(deep=True)
    j_name = next(name for name in params if name.endswith(".j"))
    start = 0.8 * j_hz
    params[j_name].value = start
    result = fit_cluster(params, cluster, noise=NOISE, correlation=None)
    fitted = result.params[j_name]
    return {
        "lineshape": lineshape,
        "regime": regime,
        "estimate": float(fitted.value),
        "at_bound": [
            name
            for name, parameter in result.params.items()
            if parameter.vary and np.isclose(parameter.value, (parameter.min, parameter.max)).any()
        ],
        "stderr": float(fitted.stderr) if fitted.stderr is not None else float("nan"),
        "moved": not np.isclose(fitted.value, start, rtol=0, atol=1e-9),
        # The cluster's "Unreliable local uncertainties" warning: any varied nonlinear
        # parameter without a finite error (evaluation.py, same rule).
        "warned": any(
            parameter.vary and not np.isfinite(parameter.stderr)
            for parameter in result.params.values()
        ),
        "success": bool(result.success),
    }


def _summary(rows: list[dict], j_hz: float) -> dict:
    estimates = np.array([r["estimate"] for r in rows])
    errors = np.array([r["stderr"] for r in rows])
    finite = np.isfinite(errors)
    deviation = np.abs(estimates - j_hz)
    clear = finite & ~np.array([r["warned"] for r in rows])
    return {
        "true_j_hz": j_hz,
        "realizations": len(rows),
        "moved_from_start": float(np.mean([r["moved"] for r in rows])),
        "bias_hz": float(np.mean(estimates - j_hz)),
        "observed_sd_hz": float(np.std(estimates, ddof=1)),
        "median_reported_sigma_hz": float(np.median(errors[finite])) if finite.any() else None,
        "sigma_withheld": float(np.mean(~finite)),
        "some_parameter_at_bound": float(np.mean([bool(r["at_bound"]) for r in rows])),
        "coverage_1sigma": float(np.mean(deviation[finite] <= errors[finite]))
        if finite.any()
        else None,
        "coverage_1.96sigma": (
            float(np.mean(deviation[finite] <= 1.96 * errors[finite])) if finite.any() else None
        ),
        "cluster_warned": float(np.mean([r["warned"] for r in rows])),
        "unwarned_with_sigma": int(clear.sum()),
        "unwarned_coverage_1sigma": (
            float(np.mean(deviation[clear] <= errors[clear])) if clear.any() else None
        ),
        "unwarned_coverage_1.96sigma": (
            float(np.mean(deviation[clear] <= 1.96 * errors[clear])) if clear.any() else None
        ),
    }


def main() -> None:
    """Fit every case over the noise realizations and write the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--realizations", type=int, default=REALIZATIONS)
    args = parser.parse_args()
    tasks = [
        (lineshape, regime, seed)
        for lineshape in LINESHAPES
        for regime in REGIMES
        for seed in range(args.realizations)
    ]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(_run, tasks, chunksize=20))
    cases = {}
    for lineshape in LINESHAPES:
        for regime, (j_hz, _, _) in REGIMES.items():
            selected = [r for r in rows if r["lineshape"] == lineshape and r["regime"] == regime]
            cases[f"{lineshape}: {regime}"] = _summary(selected, j_hz)
    report = {
        "noise": NOISE,
        "profile": PROFILE.tolist(),
        "binomial_95_halfwidth_at_68pct": float(
            1.96 * np.sqrt(0.6827 * 0.3173 / args.realizations)
        ),
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, case in cases.items():
        print(name, json.dumps(case))


if __name__ == "__main__":
    main()
