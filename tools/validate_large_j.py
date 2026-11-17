"""Evidence that large J splittings fit inside one region (validation/large-j.md).

A pseudo-3D peak, a doublet on ¹⁵N F2 and a singlet on ¹H F3, is prepared and fitted
as the CLI and GUI do (``prepare_clusters``, then ``run_pipeline``) across noise
realizations. Each case is scaled to a stated first-plane peak signal-to-noise ratio
and clustered at a contour of 15 % of that peak, where each component of a one-bond
doublet is its own contour island, or at 5σ (PeakFit's default contour factor) where
that is higher, so that noise alone does not join the region. Cases: a Lorentzian ¹J
(92 Hz); an SP1 ¹J with a 10° phase on F2, fitted; a Lorentzian J of half its
linewidth at SNR 50 and 8.

**Geometry.** Three planes (profile 1 : 0.6 : 0.3). ¹⁵N: 256 points over 2000 Hz at
60.8 MHz (7.8 Hz per point), 32 ms acquired. ¹H: 256 points over 2400 Hz at 600 MHz
(9.4 Hz per point), 50 ms acquired. SP1 uses the sine bell 0.35 0.98 on both axes.
The truth sits at points (128.3, 120.6); each realization starts from it moved by a
normal error of 0.25 point per axis. J starts 20 % below the truth, with the default
bounds [start/2, 2·start]; linewidths start at the family defaults.

**Protocol (fixed before the first run; docs/development.md):**
- 1000 realizations per case, seed 2141117; realization i of case c uses the seed
  (2141117, c, i), so a larger run extends this one. Five disjoint blocks of 200.
- **Region gate**, every case: in every realization the peak forms one cluster that
  holds both components, the grid points nearest cs ± J/2 at the true values.
- **Calibration gate**, the two ¹J cases: for J, both positions, both linewidths and
  the SP1 phase, among fits that report σ, the coverage of ±1σ and ±1.96σ lies within
  3 SE of 0.6827 and 0.95 (binomial SE).
- **Qualification gate**: "below linewidth" is raised in at least 99 % of the J = ½ LW
  fits and in at most 1 % of the ¹J fits.
- **No misleading confidence**, every case: the fraction of all fits that report
  σ(J) and whose ±1.96σ interval misses the true J is at most 0.05 + 3 SE.
- Every rate gate is also reported over blocks 2–5 (a replication of block 1), with
  the χ² of the block values about their pooled value. A gated value within 0.5 SE of
  its limit calls for a larger confirmation run.
- **Reported, not gated:** the first-order ρ(J, LW) and ρ(J, phase) that the fit
  records (mean over fits that report it) beside the correlation of the estimates
  across those fits, with its 95 % interval; and, for comparison, whether the former
  rule, the centre point alone, would have put both components in the peak's region.

    uv run python tools/validate_large_j.py --output docs/validation/large-j.json

**Confirmation (fixed after the first run, before its own; development.md).** The first
run put two gated values within 0.5 SE of their limits: the SP1 phase's ±1.96σ
coverage (−2.9 SE) and the low-SNR J = ½ LW misleading rate (+2.5 SE). ``--confirm``
fits those two cases again on realizations 1000–4999 of the same seed stream, so
they extend the first run and share none of its draws, in five blocks of 800. Each
of the two gates passes if it holds on those 4000 realizations alone; every other
gate is recomputed there and reported, but the first run decides them. The record
keeps both runs.

    uv run python tools/validate_large_j.py --confirm \
        --output docs/validation/large-j-confirmation.json

It prints the record's tables, which docs/validation/large-j.md quotes and
tests/test_large_j_record.py checks.
"""

# ruff: noqa: T201

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import scipy.stats

from peakfit.engine.algorithms.clustering import prepare_clusters, segment_data
from peakfit.engine.algorithms.noise_correlation import not_modelled
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import (
    AxisModel,
    Doublet,
    FitConfig,
    LineshapeFamily,
    PeakFitConfig,
)
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters, axis_lineshapes
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.fit.pipeline import run_pipeline

SEED = 2141117
REALIZATIONS = 1000
BLOCKS = 5
GATE_SE = 3.0
NEAR_SE = 0.5
NOMINAL = {"1sigma": 0.6827, "1.96sigma": 0.95}
MISS_LIMIT = 0.05
QUALIFIED_AT_LEAST = 0.99  # J = ½ LW fits flagged below linewidth
QUALIFIED_AT_MOST = 0.01  # ¹J fits flagged below linewidth
NOISE = 1.0
CONTOUR = 0.15  # clustering contour, as a fraction of the first-plane peak height
CONTOUR_NOISE = 5.0  # but never below this many noise σ
START_JITTER = 0.25  # SD of the start position error, in points per axis
PROFILE = np.array([1.0, 0.6, 0.3])
CENTRE = (128.3, 120.6)  # true position, in points (F2, F3)
SIZES = (256, 256)
# name: (family, true J in Hz, true linewidths (F2, F3) in Hz, true F2 phase in degrees
# or None when not fitted, first-plane peak height / noise)
CASES: dict[str, tuple[LineshapeFamily, float, tuple[float, float], float | None, float]] = {
    "Lorentzian, one-bond coupling (J = 92 Hz)": ("lorentzian", 92.0, (20.0, 25.0), None, 50.0),
    "SP1, one-bond coupling, 10° phase": ("sp1", 92.0, (10.0, 20.0), 10.0, 50.0),
    "Lorentzian, J = ½ LW": ("lorentzian", 10.0, (20.0, 25.0), None, 50.0),
    "Lorentzian, J = ½ LW, low SNR": ("lorentzian", 10.0, (20.0, 25.0), None, 8.0),
}
CALIBRATED = ("Lorentzian, one-bond coupling (J = 92 Hz)", "SP1, one-bond coupling, 10° phase")
BELOW = ("Lorentzian, J = ½ LW", "Lorentzian, J = ½ LW, low SNR")
NAMES = list(CASES)
# The confirmation run: these cases, realizations FIRST to FIRST + N - 1, and the gates
# (case, gate) it decides.
CONFIRMATION_FIRST = 1000
CONFIRMATION_REALIZATIONS = 4000
CONFIRMED = (
    ("SP1, one-bond coupling, 10° phase", "coverage_1.96sigma:phase"),
    ("Lorentzian, J = ½ LW, low SNR", "misleading"),
)
# Recovered parameters: (label in the record, parameter name, factor to the reported
# unit). Positions are reported in hertz on their axis.
PARAMETERS = (
    ("J", "A.F2.j", 1.0),
    ("δ F2", "A.F2.cs", 60.8),
    ("δ F3", "A.F3.cs", 600.0),
    ("LW F2", "A.F2.lw", 1.0),
    ("LW F3", "A.F3.lw", 1.0),
    ("phase F2", "phase", 1.0),
)


def _axis(size: int, sw: float, obs: float, aq: float, label: str, *, apodized: bool):
    return SpectralParameters(
        size=size,
        sw=sw,
        obs=obs,
        car=0.0,
        aq_time=aq,
        apocode=1.0 if apodized else 0.0,
        apodq1=0.35 if apodized else 0.0,
        apodq2=0.98 if apodized else 0.0,
        apodq3=1.0 if apodized else 0.0,
        p180=False,
        direct=label == "F3",
        ft=True,
        label=label,
    )


def _spectra(*, apodized: bool) -> Spectra:
    params = [
        _axis(3, 1.0, 1.0, 1.0, "F1", apodized=False),
        _axis(SIZES[0], 2000.0, 60.8, 0.032, "F2", apodized=apodized),
        _axis(SIZES[1], 2400.0, 600.0, 0.05, "F3", apodized=apodized),
    ]
    return Spectra(dic={}, data=np.zeros((3, *SIZES)), z_values=np.arange(3.0), params=params)


def _peak(
    spectra: Spectra, family: LineshapeFamily, j_start: float, points, *, phase: bool
) -> Peak:
    axes = {"F2": AxisModel(doublet=Doublet(j=j_start), phase=phase)}
    lineshapes = axis_lineshapes(FitConfig(family=family, axes=axes), spectra)
    position = [float(a.pts2ppm(p)) for a, p in zip(spectra.spectral_params, points, strict=True)]
    return Peak("A", np.array(position), create_shapes(spectra, lineshapes, "A", position))


def _truth(case: str) -> tuple[Spectra, Parameters, np.ndarray]:
    """The noiseless spectrum of a case, scaled to its SNR, and the true parameters."""
    family, j_hz, widths, phase, snr = CASES[case]
    spectra = _spectra(apodized=family == "sp1")
    peak = _peak(spectra, family, j_hz, CENTRE, phase=phase is not None)
    truth = Parameters.from_peaks([peak])
    truth["A.F2.j"].value = j_hz
    truth["A.F2.lw"].value, truth["A.F3.lw"].value = widths
    if phase is not None:
        truth[next(name for name in truth if name.endswith(".phase"))].value = phase
    grid = [index.ravel() for index in np.indices(SIZES)]
    shape = Cluster.evaluate_peaks([peak], truth, grid)[0]
    clean = PROFILE[:, None] * shape / np.max(np.abs(shape)) * snr * NOISE
    return spectra, truth, clean.reshape(3, *SIZES)


def _components(spectra: Spectra, truth: Parameters) -> list[tuple[int, int]]:
    n15, h1 = spectra.spectral_params
    half = n15.hz2ppm(0.5 * truth["A.F2.j"].value)
    column = h1.ppm2pt_i(truth["A.F3.cs"].value)
    centre = truth["A.F2.cs"].value
    return [(n15.ppm2pt_i(centre + side * half), column) for side in (-1, 1)]


def realization(case_index: int, index: int, seed: int = SEED) -> dict:
    """Fit one noise realization of a case; its seed is (seed, case, index)."""
    case = NAMES[case_index]
    family, j_hz, _widths, phase, snr = CASES[case]
    spectra, truth, clean = _truth(case)
    rng = np.random.default_rng((seed, case_index, index))
    start = [c + START_JITTER * rng.normal() for c in CENTRE]
    spectra.data = clean + NOISE * rng.normal(size=clean.shape)
    peak = _peak(spectra, family, 0.8 * j_hz, start, phase=phase is not None)
    config = PeakFitConfig(noise_level=NOISE)
    config.clustering.contour_level = max(CONTOUR * snr, CONTOUR_NOISE) * NOISE
    _, _, clusters = prepare_clusters(spectra, [peak], config)
    components = _components(spectra, truth)
    region = set(zip(*(i.tolist() for i in clusters[0].grid_indices), strict=True))
    # The former rule: the centre point alone, dilated, joined to the contour.
    old = segment_data(spectra.data, config.clustering.contour_level, [peak.positions_i])
    old_label = old[tuple(peak.positions_i)]
    completed = run_pipeline(
        config,
        clusters,
        NOISE,
        Parameters.from_peaks([peak]),
        noise_model=not_modelled("white synthetic noise"),
    ).completed_fit
    [cluster] = completed.outcome.clusters
    # The cluster's phase is named after the cluster; "phase" stands for it here.
    fitted = {
        "phase" if p.name.endswith(".phase") else p.name: p
        for p in cluster.final_nonlinear_parameters
    }
    coupling = completed.couplings()["A.F2.j"]
    return {
        "one_cluster": len(clusters) == 1,
        "both_in_region": len(clusters) == 1 and all(c in region for c in components),
        "both_in_old_region": all(old[c] == old_label for c in components),
        "region_points": int(clusters[0].n_points),
        "estimates": {
            name: float(fitted[name].value) for _, name, _ in PARAMETERS if name in fitted
        },
        "errors": {
            name: float(fitted[name].standard_error) for _, name, _ in PARAMETERS if name in fitted
        },
        "below_linewidth": coupling.below_linewidth,
        "rho_lw": coupling.linewidth_correlation,
        "rho_phase": coupling.phase_correlation,
    }


def _rate(hits: np.ndarray, nominal: float, *, at_most: bool = False, at_least: bool = False):
    """A binomial rate against its nominal value, gated at GATE_SE standard errors."""
    n = int(hits.size)
    if n == 0:
        return {"n": 0, "rate": None, "se": None, "deviation_se": None, "passes": True}
    rate = float(hits.mean())
    se = float(np.sqrt(nominal * (1 - nominal) / n))
    deviation = (rate - nominal) / se
    if at_most:
        margin = GATE_SE - deviation
    elif at_least:
        margin = GATE_SE + deviation
    else:
        margin = GATE_SE - abs(deviation)
    return {
        "n": n,
        "rate": rate,
        "se": se,
        "deviation_se": deviation,
        "passes": bool(margin >= 0),
        "near_limit": bool(abs(margin) <= NEAR_SE),
    }


def _all(hits: np.ndarray) -> dict:
    """A rate that must be exactly 1: every realization passes."""
    n = int(hits.size)
    rate = float(hits.mean()) if n else None
    return {"n": n, "rate": rate, "se": None, "deviation_se": None, "passes": bool(hits.all())}


def _truths(case: str) -> dict[str, float]:
    _, truth, _ = _truth(case)
    values = {name: truth[name].value for _, name, _ in PARAMETERS if name in truth}
    phase = next((truth[name].value for name in truth if name.endswith(".phase")), None)
    if phase is not None:
        values["phase"] = phase
    return values


def _gates(rows: list[dict], case: str) -> dict:
    truths = _truths(case)
    gates = {"region": _all(np.array([r["both_in_region"] for r in rows]))}
    below = np.array([r["below_linewidth"] for r in rows])
    if case in BELOW:
        gates["qualified"] = _rate(below, QUALIFIED_AT_LEAST, at_least=True)
    else:
        gates["qualified"] = _rate(below, QUALIFIED_AT_MOST, at_most=True)
    estimates = np.array([r["estimates"]["A.F2.j"] for r in rows])
    errors = np.array([r["errors"]["A.F2.j"] for r in rows])
    misses = np.isfinite(errors) & ~(np.abs(estimates - truths["A.F2.j"]) <= 1.96 * errors)
    gates["misleading"] = _rate(misses, MISS_LIMIT, at_most=True)
    if case in CALIBRATED:
        for _, name, _ in PARAMETERS:
            if name not in truths:
                continue
            values = np.array([r["estimates"][name] for r in rows])
            sigma = np.array([r["errors"][name] for r in rows])
            reported = np.isfinite(sigma)
            deviation = np.abs(values - truths[name])[reported]
            for level, z in (("1sigma", 1.0), ("1.96sigma", 1.96)):
                covered = deviation <= z * sigma[reported]
                gates[f"coverage_{level}:{name}"] = _rate(covered, NOMINAL[level])
    return gates


def _blocks(rows: list[dict], case: str) -> dict:
    size = len(rows) // BLOCKS
    parts = [rows[k * size : (k + 1) * size] for k in range(BLOCKS)]
    per_block = [_gates(part, case) for part in parts]
    consistency = {}
    for name in per_block[0]:
        usable = [b[name] for b in per_block if b[name]["n"] and b[name]["se"]]
        if len(usable) < BLOCKS:
            continue
        rates = np.array([b["rate"] for b in usable])
        ses = np.array([b["se"] for b in usable])
        pooled = float(np.sum(rates / ses**2) / np.sum(1 / ses**2))
        chi2 = float(np.sum(((rates - pooled) / ses) ** 2))
        consistency[name] = {
            "rates": rates.tolist(),
            "chi2": chi2,
            "p_value": float(scipy.stats.chi2.sf(chi2, len(usable) - 1)),
        }
    replication = _gates([r for part in parts[1:] for r in part], case)
    return {"count": BLOCKS, "consistency": consistency, "replication": replication}


def _correlation(rows: list[dict], key: str, partner: str) -> dict | None:
    """The mean recorded ρ(J, partner) and the correlation of the estimates, 95 % CI."""
    kept = [r for r in rows if r[key] is not None]
    if len(kept) < 10:  # noqa: PLR2004
        return None
    j = np.array([r["estimates"]["A.F2.j"] for r in kept])
    other = np.array([r["estimates"][partner] for r in kept])
    empirical = float(np.corrcoef(j, other)[0, 1])
    half = 1.96 / np.sqrt(len(kept) - 3)
    low, high = np.tanh(np.arctanh(empirical) + np.array([-half, half]))
    return {
        "n": len(kept),
        "recorded_mean": float(np.mean([r[key] for r in kept])),
        "empirical": empirical,
        "empirical_ci95": [float(low), float(high)],
    }


def summarize(case_index: int, rows: list[dict]) -> dict:
    """The case's descriptive statistics, gates and block replication."""
    case = NAMES[case_index]
    family, j_hz, widths, phase, snr = CASES[case]
    truths = _truths(case)
    parameters = {}
    for label, name, factor in PARAMETERS:
        if name not in truths:
            continue
        values = np.array([r["estimates"][name] for r in rows])
        sigma = np.array([r["errors"][name] for r in rows])
        reported = np.isfinite(sigma)
        parameters[label] = {
            "name": name,
            "truth": truths[name],
            "bias": float(np.mean(values - truths[name]) * factor),
            "scatter": float(np.std(values, ddof=1) * factor),
            "rms_sigma": (
                float(np.sqrt(np.mean(sigma[reported] ** 2)) * factor) if reported.any() else None
            ),
            "withheld": float(np.mean(~reported)),
        }
    gates = _gates(rows, case)
    blocks = _blocks(rows, case)
    return {
        "case": case,
        "family": family,
        "true_j_hz": j_hz,
        "linewidths_hz": list(widths),
        "phase_deg": phase,
        "snr": snr,
        "calibration_gated": case in CALIBRATED,
        "realizations": len(rows),
        "one_cluster": float(np.mean([r["one_cluster"] for r in rows])),
        "region_points_mean": float(np.mean([r["region_points"] for r in rows])),
        "both_in_old_region": float(np.mean([r["both_in_old_region"] for r in rows])),
        "below_linewidth": float(np.mean([r["below_linewidth"] for r in rows])),
        "parameters": parameters,
        "rho_j_lw": _correlation(rows, "rho_lw", "A.F2.lw"),
        "rho_j_phase": _correlation(rows, "rho_phase", "phase") if phase is not None else None,
        "gates": gates,
        "blocks": blocks,
        "passes": all(g["passes"] for g in gates.values())
        and all(g["passes"] for g in blocks["replication"].values()),
        "near_limit": any(g.get("near_limit", False) for g in gates.values()),
    }


def _cell(gate: dict | None) -> str:
    if gate is None or gate["rate"] is None:
        return "—"
    return f"{gate['rate']:.3f} ({gate['deviation_se']:+.1f})"


def _number(value: float | None, digits: int = 3, sign: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def recovery_table(record: dict) -> str:
    """Recovery and coverage of every parameter in the calibrated cases."""
    lines = [
        "| Case | Parameter | Bias | Scatter | RMS σ | Withheld | ±1σ (SE) | ±1.96σ (SE) |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for case in record["cases"]:
        if not case["calibration_gated"]:
            continue
        for label, values in case["parameters"].items():
            name = values["name"]
            gates = case["gates"]
            lines.append(
                f"| {case['case']} | {label} | {_number(values['bias'], sign=True)} "
                f"| {_number(values['scatter'])} | {_number(values['rms_sigma'])} "
                f"| {values['withheld']:.3f} | {_cell(gates.get(f'coverage_1sigma:{name}'))} "
                f"| {_cell(gates.get(f'coverage_1.96sigma:{name}'))} |"
            )
    return "\n".join(lines)


def _rho(value: dict | None) -> str:
    if value is None:
        return "—"
    low, high = value["empirical_ci95"]
    return f"{value['recorded_mean']:+.2f} / {value['empirical']:+.2f} [{low:+.2f}, {high:+.2f}]"


def case_table(record: dict) -> str:
    """Region, qualification, J and correlations per case, with the verdict."""
    lines = [
        "| Case | Region (former rule) | Below LW | J bias / scatter (Hz) | σ(J) withheld "
        "| Misleading (SE) | ρ(J, LW) recorded / estimates [95 %] | ρ(J, phase) | Gate |",
        "|---|---|---:|---|---:|---|---|---|---|",
    ]
    for case in record["cases"]:
        j = case["parameters"]["J"]
        lines.append(
            f"| {case['case']} | {case['gates']['region']['rate']:.3f} "
            f"({case['both_in_old_region']:.3f}) | {case['below_linewidth']:.3f} "
            f"| {j['bias']:+.3f} / {j['scatter']:.3f} | {j['withheld']:.3f} "
            f"| {_cell(case['gates']['misleading'])} | {_rho(case['rho_j_lw'])} "
            f"| {_rho(case['rho_j_phase'])} | {'pass' if case['passes'] else 'FAIL'} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Run every case under the fixed protocol and write the record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--realizations", type=int, default=REALIZATIONS)
    parser.add_argument("--confirm", action="store_true", help="the confirmation run")
    args = parser.parse_args()
    if args.confirm:
        confirm(args.output)
        return
    n = args.realizations
    cases_ix = [case for case in range(len(NAMES)) for _ in range(n)]
    index_ix = [index for _ in range(len(NAMES)) for index in range(n)]
    with ProcessPoolExecutor(mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(realization, cases_ix, index_ix, chunksize=50))
    cases = [summarize(case, rows[case * n : (case + 1) * n]) for case in range(len(NAMES))]
    record = {
        "protocol": {
            "seed": SEED,
            "realizations": n,
            "blocks": BLOCKS,
            "gate_se": GATE_SE,
            "near_se": NEAR_SE,
            "nominal_coverage": NOMINAL,
            "misleading_limit": MISS_LIMIT,
            "qualified_at_least": QUALIFIED_AT_LEAST,
            "qualified_at_most": QUALIFIED_AT_MOST,
            "noise": NOISE,
            "contour": CONTOUR,
            "contour_noise": CONTOUR_NOISE,
            "start_jitter_points": START_JITTER,
            "profile": PROFILE.tolist(),
        },
        "cases": cases,
        "passes": all(case["passes"] for case in cases),
        "confirmation_needed": any(case["near_limit"] for case in cases),
    }
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(case_table(record))
    print()
    print(recovery_table(record))
    print(f"\nGate: {'pass' if record['passes'] else 'FAIL'}; ", end="")
    print(f"confirmation needed: {record['confirmation_needed']}")


def confirmation_table(record: dict) -> str:
    """The confirmed gates over the confirmation realizations alone."""
    lines = [
        "| Case | Gate | First run (SE) | Confirmation (SE) | Blocks p | Verdict |",
        "|---|---|---|---|---:|---|",
    ]
    lines.extend(
        f"| {gate['case']} | {gate['gate']} | {_cell(gate['first'])} "
        f"| {_cell(gate['confirmation'])} | {_number(gate['blocks_p'], 2)} "
        f"| {'pass' if gate['confirmation']['passes'] else 'FAIL'} |"
        for gate in record["gates"]
    )
    return "\n".join(lines)


def confirm(output: Path) -> None:
    """The confirmation run of the near-limit gates (see the docstring)."""
    first, n = CONFIRMATION_FIRST, CONFIRMATION_REALIZATIONS
    names = list(dict.fromkeys(case for case, _ in CONFIRMED))
    cases_ix = [NAMES.index(case) for case in names for _ in range(n)]
    index_ix = [first + index for _ in names for index in range(n)]
    with ProcessPoolExecutor(mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(realization, cases_ix, index_ix, chunksize=50))
    summaries = {
        case: summarize(NAMES.index(case), rows[k * n : (k + 1) * n])
        for k, case in enumerate(names)
    }
    previous = output.with_name("large-j.json")
    first_run = json.loads(previous.read_text()) if previous.exists() else {"cases": []}
    earlier = {case["case"]: case for case in first_run["cases"]}
    gates = []
    for case, name in CONFIRMED:
        summary = summaries[case]
        consistency = summary["blocks"]["consistency"].get(name)
        gates.append(
            {
                "case": case,
                "gate": name,
                "first": earlier.get(case, {}).get("gates", {}).get(name),
                "confirmation": summary["gates"][name],
                "blocks_p": consistency["p_value"] if consistency else None,
            }
        )
    record = {
        "protocol": {
            "seed": SEED,
            "first_index": first,
            "realizations": n,
            "blocks": BLOCKS,
            "gate_se": GATE_SE,
            "confirmed": [list(gate) for gate in CONFIRMED],
        },
        "gates": gates,
        "cases": list(summaries.values()),
        "passes": all(gate["confirmation"]["passes"] for gate in gates),
    }
    output.write_text(json.dumps(record, indent=1) + "\n")
    print(confirmation_table(record))
    print(f"\nConfirmation: {'pass' if record['passes'] else 'FAIL'}")


if __name__ == "__main__":
    main()
