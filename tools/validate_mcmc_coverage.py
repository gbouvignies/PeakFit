"""Coverage of MCMC intervals on one peak and on overlapping pairs (validation V6).

See ``docs/mcmc-formulation.md`` §10. Each scenario simulates a 1D Gaussian cluster
over four planes with independent noise, fits it with the ordinary pipeline, samples
its posterior and checks every interval that is reported against the truth.

- The intervals checked are the 68% and 95% intervals of positions, linewidths,
  per-peak amplitudes and summed amplitudes.
- Intervals reported with a caution (a peak not separately resolved, §6) are scored
  in their own "(flagged)" rows, never mixed with or dropped from the others.
- Withheld intervals (exchanged peaks) are counted, not scored.
- A provisional run contributes no intervals.

Near-coalescent pairs must produce flags rather than confident per-peak intervals.

    uv run python tools/validate_mcmc_coverage.py --output /tmp/mcmc-coverage.json
"""

# ruff: noqa: T201

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from peakfit.engine.algorithms.noise_correlation import not_modelled
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.scientific.sampler import SamplerSettings, sample_posterior
from peakfit.engine.scientific.sampling import sampling_problem
from peakfit.fit.pipeline import run_pipeline

NOISE = 0.05
POINTS = 101
AMPLITUDES = np.array([[2.0, -1.0, 3.0, 0.5], [-1.0, 3.0, 0.5, 2.0]])
# Separations in points; the Gaussian's default 25 Hz linewidth spans about 5 points.
SCENARIOS = {
    "one peak": None,
    "resolved pair (10 points)": 10.0,
    "partially overlapping pair (4 points)": 4.0,
    "near-coalescent pair (1.5 points)": 1.5,
}
SETTINGS = SamplerSettings(walkers=16, steps=2000)


def _axis(size: int, direct: bool) -> SpectralParameters:
    return SpectralParameters(
        size=size,
        sw=500,
        obs=500,
        car=0,
        aq_time=0.01,
        apocode=0,
        apodq1=0,
        apodq2=0,
        apodq3=0,
        p180=False,
        direct=direct,
        ft=True,
    )


def _cluster(separation: float | None, rng: np.random.Generator):
    """A noisy cluster and its true parameters, with the fit starting at the truth."""
    spectra = Spectra(
        dic={},
        data=np.zeros((4, POINTS)),
        z_values=np.arange(4.0),
        params=[_axis(4, False), _axis(POINTS, True)],
    )
    points = [50.0] if separation is None else [50 - separation / 2, 50 + separation / 2]
    peaks = []
    for name, point in zip("AB", points, strict=False):
        center = float(spectra.spectral_params[0].pts2ppm(point))
        config = FitConfig(lineshape="gaussian")
        peak = Peak(
            name, np.array([center]), create_shapes(spectra, config, name, [center], ["gaussian"])
        )
        peak.set_cluster_id(1)
        peaks.append(peak)
    params = Parameters.from_peaks(peaks)
    truth = {name: p.value for name, p in params.items()}
    amplitudes = AMPLITUDES[: len(peaks)]
    cluster = Cluster(1, peaks, [np.arange(POINTS)], np.zeros((POINTS, 4)))
    cluster.data = cluster.evaluate(params).T @ amplitudes + NOISE * rng.standard_normal(
        (POINTS, 4)
    )
    return cluster, params, truth, amplitudes


def _inside(value: float, quantiles) -> tuple[bool, bool]:
    low95, low68, _, high68, high95 = quantiles
    return bool(low68 <= value <= high68), bool(low95 <= value <= high95)


def realization(scenario: str, index: int) -> dict:
    """One simulated data set: its status, flags and the coverage of reported intervals."""
    rng = np.random.default_rng([20261104, index, list(SCENARIOS).index(scenario)])
    cluster, params, truth, amplitudes = _cluster(SCENARIOS[scenario], rng)
    config = PeakFitConfig(fitting=FitConfig(lineshape="gaussian"))
    completed = run_pipeline(
        config, [cluster], NOISE, params, noise_model=not_modelled("synthetic spectrum")
    ).completed_fit
    if not completed.outcome.cluster(1).usable:
        return {"status": "unusable fit", "flags": [], "covered": {}}
    posterior = sample_posterior(
        sampling_problem(completed, 1), SETTINGS, np.random.SeedSequence([7, index])
    )
    flags = []
    if posterior.exchanges:
        flags.append("exchange")
    if posterior.amplitudes.correlated:
        flags.append("correlated amplitudes")
    if posterior.amplitudes.undetermined:
        flags.append("undetermined amplitude")
    covered: dict[str, list] = defaultdict(list)
    if posterior.provisional:
        return {"status": "provisional", "flags": flags, "covered": covered}
    names = [p.name for p in cluster.peaks]

    def kind(base: str, peak: str) -> str:
        return f"{base} (flagged)" if posterior.caution(peak) else base

    for p in posterior.diagnosis.parameters:
        if posterior.withheld(p.name) is None:
            base = "position" if p.name.endswith(".cs") else "linewidth"
            covered[kind(base, p.name.rsplit(".", 2)[0])].append(
                _inside(truth[p.name], p.quantiles)
            )
    exchanged = {peak for e in posterior.exchanges for peak in (e.first, e.second)}
    for a in posterior.amplitudes.amplitudes:
        if a.peaks[0] not in exchanged:
            truth_value = amplitudes[names.index(a.peaks[0]), a.plane]
            covered[kind("amplitude", a.peaks[0])].append(_inside(truth_value, a.quantiles))
    for total in posterior.amplitudes.sums:
        truth_value = sum(amplitudes[names.index(peak), total.plane] for peak in total.peaks)
        covered["summed amplitude"].append(_inside(truth_value, total.quantiles))
    if any(posterior.caution(name) for name in names):
        flags.append("passed with a caution")
    return {"status": "passed", "flags": flags, "covered": covered}


def main() -> None:
    """Run every scenario in parallel and report coverage with binomial errors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--realizations", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    jobs = [(scenario, i) for scenario in SCENARIOS for i in range(args.realizations)]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        results = list(pool.map(realization, *zip(*jobs, strict=True), chunksize=4))
    scenarios: dict[str, dict] = {}
    report = {"noise": NOISE, "realizations": args.realizations, "scenarios": scenarios}
    for scenario in SCENARIOS:
        rows = [r for (s, _), r in zip(jobs, results, strict=True) if s == scenario]
        status = defaultdict(int)
        flags = defaultdict(int)
        coverage = defaultdict(list)
        for r in rows:
            status[r["status"]] += 1
            for flag in r["flags"]:
                flags[flag] += 1
            for kind, hits in r["covered"].items():
                coverage[kind].extend(hits)
        summary = {}
        for kind, hits in coverage.items():
            array = np.array(hits, dtype=float)
            n = len(array)
            summary[kind] = {
                "n": n,
                "coverage_68": float(array[:, 0].mean()),
                "coverage_95": float(array[:, 1].mean()),
                "binomial_se_68": float(np.sqrt(0.68 * 0.32 / n)),
                "binomial_se_95": float(np.sqrt(0.95 * 0.05 / n)),
            }
        scenarios[scenario] = {
            "status": dict(status),
            "flags": dict(flags),
            "coverage": summary,
        }
        print(f"\n{scenario}: {dict(status)} flags {dict(flags)}")
        for kind, s in summary.items():
            print(
                f"  {kind:<17} n={s['n']:5d}  68%: {s['coverage_68']:.3f} "
                f"(±{s['binomial_se_68']:.3f})  95%: {s['coverage_95']:.3f} "
                f"(±{s['binomial_se_95']:.3f})"
            )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
