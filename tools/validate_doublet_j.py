"""Evidence that doublet fits estimate J with a calibrated σ(J) (validation/doublet-j.md).

A 1D doublet over three planes (profile 1 : 0.6 : 0.3) is fitted by the VarPro
optimizer from a J start 20 % below the truth, across noise realizations. Every case
is scaled to a stated first-plane peak signal-to-noise ratio, so lineshapes with
different kernel heights are compared at equal information. Cases span resolved,
partly resolved and unresolved doublets and a ¹⁵N-like one-bond coupling, at high and
low SNR, for a real-valued (Lorentzian) and a phased, apodized (SP1) lineshape.

**Protocol (fixed before the confirmation run; docs/development.md):**
- 1000 realizations per case, seed 20261117; realization i of case c uses the seed
  (20261117, c, i), so a larger run extends this one. Five disjoint blocks of 200.
- **Calibration gate** on the resolved, partly resolved and one-bond cases: among
  fits that report σ(J), the coverage of ±1σ and ±1.96σ lies within 3 SE of 0.6827
  and 0.95 (binomial SE).
- **No misleading confidence** on every case: the fraction of all fits that report
  σ(J) and whose ±1.96σ interval misses the true J is at most 0.05 + 3 SE.
- Every gate is also reported over blocks 2–5 (a replication of block 1), with the
  χ² of the block values about their pooled value. A gated value within 0.5 SE of its
  limit calls for a larger confirmation run.

    uv run python tools/validate_doublet_j.py --output docs/validation/doublet-j.json

It prints the record's table, which docs/validation/doublet-j.md quotes and
tests/test_doublet_j_record.py checks.
"""

# ruff: noqa: T201

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import scipy.stats

from peakfit.engine.algorithms.evaluation import nonlinear_errors_withheld
from peakfit.engine.algorithms.varpro import fit_cluster
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import AxisModel, Doublet, FitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters, axis_lineshapes
from peakfit.engine.lineshapes.create import create_shapes

SEED = 20261117
REALIZATIONS = 1000
BLOCKS = 5
GATE_SE = 3.0
NEAR_SE = 0.5
NOMINAL = {"1sigma": 0.6827, "1.96sigma": 0.95}
MISS_LIMIT = 0.05
NOISE = 2.0
PROFILE = np.array([[1.0, 0.6, 0.3]])
# name: (true J in Hz, true linewidth in Hz, first-plane peak height / noise, gated
# on calibration)
REGIMES = {
    "resolved (J = 5 lw)": (40.0, 8.0, 50.0, True),
    "partly resolved (J = 1.2 lw)": (12.0, 10.0, 50.0, True),
    "unresolved (J = 0.25 lw)": (5.0, 20.0, 50.0, False),
    "one-bond coupling (J = 92 Hz)": (92.0, 15.0, 50.0, True),
    "partly resolved, low SNR": (12.0, 10.0, 8.0, True),
    "unresolved, low SNR": (5.0, 20.0, 8.0, False),
}
# Case labels as the record names them; each is a doublet of that family on F2.
LINESHAPES = ("lorentzian_doublet", "sp1_doublet")
CASES: list[tuple[str, str]] = [
    (lineshape, regime) for lineshape in LINESHAPES for regime in REGIMES
]


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


def _problem(lineshape: str, j_hz: float, lw_hz: float):
    axis = _axis(512, direct=True, sw=3000.0, obs=600.0, label="F2")
    planes = _axis(3, direct=False, sw=1.0, obs=1.0, label="F1")
    spectra = Spectra(
        dic={}, data=np.zeros((3, 512)), z_values=np.arange(3.0), params=[planes, axis]
    )
    centre = float(axis.pts2ppm(256.0))
    phased = lineshape.startswith("sp")
    # J's start and bounds are set below, so any valid doublet model serves here.
    model = AxisModel(doublet=Doublet(j=5.0), phase=phased)
    family = "lorentzian" if lineshape.startswith("lorentzian") else "sp1"
    config = FitConfig(family=family, axes={"F2": model})
    peak = Peak(
        "A",
        np.array([centre]),
        create_shapes(spectra, axis_lineshapes(config, spectra), "A", [centre]),
    )
    peak.set_cluster_id(1)
    truth = Parameters.from_peaks([peak])
    for parameter in truth.values():
        if parameter.is_coupling:
            parameter.min, parameter.max, parameter.value = 0.5, 150.0, j_hz
        elif parameter.name.endswith(".lw"):
            parameter.value = lw_hz
        elif parameter.name.endswith(".phase"):
            parameter.value = 4.0
    cluster = Cluster(1, [peak], [np.arange(512)], np.zeros((512, 3)))
    clean = cluster.evaluate(truth).T @ PROFILE
    return cluster, truth, clean / np.max(np.abs(clean[:, 0]))


def realization(case: int, index: int, seed: int = SEED) -> dict:
    """Fit one noise realization of case ``case``; its seed is (seed, case, index)."""
    lineshape, regime = CASES[case]
    j_hz, lw_hz, snr, _ = REGIMES[regime]
    cluster, truth, clean = _problem(lineshape, j_hz, lw_hz)
    rng = np.random.default_rng((seed, case, index))
    cluster.data = snr * NOISE * clean + NOISE * rng.normal(size=clean.shape)
    params = truth.copy(deep=True)
    j_name = next(name for name in params if params[name].is_coupling)
    start = 0.8 * j_hz
    params[j_name].value = start
    result = fit_cluster(params, cluster, noise=NOISE, correlation=None)
    fitted = result.params[j_name]
    return {
        "estimate": float(fitted.value),
        "stderr": float(fitted.stderr),
        "moved": bool(not np.isclose(fitted.value, start, rtol=0, atol=1e-9)),
        # The nonlinear part of the cluster's "Unreliable local uncertainties" rule.
        "warned": nonlinear_errors_withheld(result.params, list(result.params)),
    }


def _rate(hits: np.ndarray, nominal: float, *, at_most: bool = False) -> dict:
    """A binomial rate against its nominal value, gated at GATE_SE standard errors."""
    n = int(hits.size)
    if n == 0:
        return {"n": 0, "rate": None, "se": None, "deviation_se": None, "passes": True}
    rate = float(hits.mean())
    se = float(np.sqrt(nominal * (1 - nominal) / n))
    deviation = (rate - nominal) / se
    margin = GATE_SE - (deviation if at_most else abs(deviation))
    return {
        "n": n,
        "rate": rate,
        "se": se,
        "deviation_se": deviation,
        "passes": bool(margin >= 0),
        "near_limit": bool(abs(margin) <= NEAR_SE),
    }


def _gates(rows: list[dict], j_hz: float, calibrated: bool) -> dict:
    estimates = np.array([r["estimate"] for r in rows])
    errors = np.array([r["stderr"] for r in rows])
    reported = np.isfinite(errors)
    deviation = np.abs(estimates - j_hz)
    misses = reported & ~(deviation <= 1.96 * errors)
    gates = {"misleading": _rate(misses, MISS_LIMIT, at_most=True)}
    if calibrated:
        for name, z in (("1sigma", 1.0), ("1.96sigma", 1.96)):
            covered = deviation[reported] <= z * errors[reported]
            gates[f"coverage_{name}"] = _rate(covered, NOMINAL[name])
    return gates


def _blocks(rows: list[dict], j_hz: float, calibrated: bool) -> dict:
    size = len(rows) // BLOCKS
    parts = [rows[k * size : (k + 1) * size] for k in range(BLOCKS)]
    per_block = [_gates(part, j_hz, calibrated) for part in parts]
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
    replication = _gates([r for part in parts[1:] for r in part], j_hz, calibrated)
    return {"count": BLOCKS, "consistency": consistency, "replication": replication}


def summarize(case: int, rows: list[dict]) -> dict:
    """The case's descriptive statistics, gates and block replication."""
    lineshape, regime = CASES[case]
    j_hz, lw_hz, snr, calibrated = REGIMES[regime]
    estimates = np.array([r["estimate"] for r in rows])
    errors = np.array([r["stderr"] for r in rows])
    reported = np.isfinite(errors)
    gates = _gates(rows, j_hz, calibrated)
    blocks = _blocks(rows, j_hz, calibrated)
    return {
        "lineshape": lineshape,
        "regime": regime,
        "true_j_hz": j_hz,
        "linewidth_hz": lw_hz,
        "snr": snr,
        "calibration_gated": calibrated,
        "realizations": len(rows),
        "moved_from_start": float(np.mean([r["moved"] for r in rows])),
        "bias_hz": float(np.mean(estimates - j_hz)),
        "scatter_hz": float(np.std(estimates, ddof=1)),
        "rms_reported_sigma_hz": (
            float(np.sqrt(np.mean(errors[reported] ** 2))) if reported.any() else None
        ),
        "withheld": float(np.mean(~reported)),
        "cluster_warned": float(np.mean([r["warned"] for r in rows])),
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


def table(record: dict) -> str:
    """The record's results table, as docs/validation/doublet-j.md quotes it."""
    lines = [
        "| Lineshape, case | J / lw (Hz) | SNR | Bias | Scatter | RMS σ | Withheld "
        "| ±1σ (SE) | ±1.96σ (SE) | Misleading (SE) | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for case in record["cases"]:
        gates = case["gates"]
        name = "Lorentzian" if case["lineshape"].startswith("lorentzian") else "SP1"
        rms = case["rms_reported_sigma_hz"]
        lines.append(
            f"| {name}, {case['regime']} | {case['true_j_hz']:g} / {case['linewidth_hz']:g} "
            f"| {case['snr']:g} | {case['bias_hz']:+.3f} | {case['scatter_hz']:.3f} "
            f"| {'—' if rms is None else f'{rms:.3f}'} | {case['withheld']:.3f} "
            f"| {_cell(gates.get('coverage_1sigma'))} | {_cell(gates.get('coverage_1.96sigma'))} "
            f"| {_cell(gates['misleading'])} | {'pass' if case['passes'] else 'FAIL'} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Run every case under the fixed protocol and write the record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--realizations", type=int, default=REALIZATIONS)
    args = parser.parse_args()
    n = args.realizations
    cases_ix = [case for case in range(len(CASES)) for _ in range(n)]
    index_ix = [index for _ in range(len(CASES)) for index in range(n)]
    with ProcessPoolExecutor(mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(realization, cases_ix, index_ix, chunksize=50))
    cases = [summarize(case, rows[case * n : (case + 1) * n]) for case in range(len(CASES))]
    record = {
        "protocol": {
            "seed": SEED,
            "realizations": n,
            "blocks": BLOCKS,
            "gate_se": GATE_SE,
            "near_se": NEAR_SE,
            "nominal_coverage": NOMINAL,
            "misleading_limit": MISS_LIMIT,
            "noise": NOISE,
            "profile": PROFILE.ravel().tolist(),
        },
        "cases": cases,
        "passes": all(case["passes"] for case in cases),
        "confirmation_needed": any(case["near_limit"] for case in cases),
    }
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(table(record))
    print(f"\nGate: {'pass' if record['passes'] else 'FAIL'}; ", end="")
    print(f"confirmation needed: {record['confirmation_needed']}")


if __name__ == "__main__":
    main()
