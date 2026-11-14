"""T8: validation of the duplicate-plane check of ``peakfit replicates``.

``docs/profile-uncertainty.md`` §8, T8. Pseudo-3D series with duplicate planes (exactly
equal plane values) are drawn as in T3 (``profile_coverage.py``): PeakFit's own
``sp1``, ``sp2`` or ``no_apod`` lineshapes on both axes, complex white FID noise
through the exact processing operators, the engine fit from the truth. Each
realization is checked as a published run is, by ``duplicate_plane_check`` with the
correlated-profile flags derived from the fit.

**Profiles,** decays A·e^(−R·t) with R = 1 (peak A) and 2 (its partner B):
- two copies: t = 0, 0.1, 0.1, 0.3, 0.6, 0.6, 1.0 (two pairs, k = 2 per peak);
- three copies: t = 0, 0.2, 0.2, 0.2, 0.5, 1.0, 1.0, 1.0 (two triples, k = 4).

**Statistic.** Per realization, Σq/Σk over the peaks of the case's pool (both pools
unless stated); E[q] = k at first order. **Gate:** the mean of q/k over realizations
within ±3 SE of 1, the SE from the realizations' own spread.

**Analytical check** (every case): the Monte Carlo covariance of each peak's contrasts
Ca over realizations against the contrast covariance CΣCᵀ the stored factors state,
averaged over realizations; each variance ratio within ±3 SE of 1, SE √(2/(N − 1)).

**Cases:**
- isolated peaks, two and three copies, every lineshape for two copies;
- pairs 3 points apart (unflagged), two and three copies; with a cluster phase
  shared by both peaks (the shared parameter);
- a pair 2 points apart, flagged: the flagged pool;
- exclusions: an isolated peak checked with a coincident pair (both peaks at one
  position), whose covariance factors are withheld or whose contrast covariance
  cannot be whitened; the gate is on the unflagged pool, which the exclusions must
  leave calibrated.

**Recorded, not gated:**
- a near-coalescent pair, 1 point apart, beyond the first order (as in T3);
- unmodelled correlation between planes: the noise of the copies of each group
  correlated with ρ = +0.3 or −0.3, which the model omits. q/k is then about
  1 − ρ: the RMS falls below 1 for positive and rises above it for negative
  correlation. This illustrates unmodelled covariance in general; it is not a model
  of joint pseudo-3D NUS reconstruction.

    uv run python tools/profile_uncertainty/duplicate_check.py --realizations 400
    uv run python tools/profile_uncertainty/run_all.py duplicate_check   # the record
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/
from profile_coverage import CHAINS, SIGMA, _fit, build_cluster

from peakfit.engine.algorithms.covariance_factors import (
    peak_covariances_between_planes,
    profile_correlations,
)
from peakfit.engine.scientific.replicates import (
    RecordedCluster,
    duplicate_groups,
    duplicate_plane_check,
)

if TYPE_CHECKING:
    from peakfit.engine.domain.config import LineshapeName

RATES = (1.0, 2.0)
DELAYS = {
    "two copies": np.array([0.0, 0.1, 0.1, 0.3, 0.6, 0.6, 1.0]),
    "three copies": np.array([0.0, 0.2, 0.2, 0.2, 0.5, 1.0, 1.0, 1.0]),
}
GATE_SE = 3.0


@dataclass(frozen=True)
class Case:
    """One validation case.

    ``separation`` is None for an isolated peak. ``pool`` is the pool the statistic
    reads: "all", "unflagged" or "flagged". ``coincident`` checks a coincident pair
    alongside. ``copies_correlation`` correlates the noise of each group's copies.
    ``gated``: whether the case is part of the gate.
    """

    lineshape: LineshapeName
    copies: str
    separation: tuple[float, float] | None = None
    shared_phase: bool = False
    pool: str = "all"
    coincident: bool = False
    copies_correlation: float = 0.0
    gated: bool = True


CASES: dict[str, Case] = {
    "sp2, isolated, two copies": Case("sp2", "two copies"),
    "sp2, isolated, three copies": Case("sp2", "three copies"),
    "sp1, isolated, two copies": Case("sp1", "two copies"),
    "no_apod, isolated, two copies": Case("no_apod", "two copies"),
    "sp2, pair 3 points, two copies": Case("sp2", "two copies", (3.0, 3.0)),
    "sp2, pair 3 points, three copies": Case("sp2", "three copies", (3.0, 3.0)),
    "sp2, pair 3 points, shared phase, three copies": Case(
        "sp2", "three copies", (3.0, 3.0), shared_phase=True
    ),
    "sp2, pair 2 points (flagged), two copies": Case(
        "sp2", "two copies", (2.0, 2.0), pool="flagged"
    ),
    "sp2, isolated with a coincident pair (exclusions), two copies": Case(
        "sp2", "two copies", pool="unflagged", coincident=True
    ),
    "sp2, pair 1 point (near-coalescent), two copies": Case(
        "sp2", "two copies", (1.0, 1.0), gated=False
    ),
    "sp2, isolated, two copies, copies correlated +0.3": Case(
        "sp2", "two copies", copies_correlation=0.3, gated=False
    ),
    "sp2, isolated, two copies, copies correlated −0.3": Case(
        "sp2", "two copies", copies_correlation=-0.3, gated=False
    ),
    "sp2, pair 3 points, three copies, copies correlated +0.3": Case(
        "sp2", "three copies", (3.0, 3.0), copies_correlation=0.3, gated=False
    ),
    "sp2, pair 3 points, three copies, copies correlated −0.3": Case(
        "sp2", "three copies", (3.0, 3.0), copies_correlation=-0.3, gated=False
    ),
}


def copies_mixing(plane_values: np.ndarray, rho: float) -> np.ndarray:
    """The S × S matrix that correlates the noise of each group's copies with ``rho``.

    Applied to independent per-plane noise, it leaves every plane's variance unchanged
    and gives copies of one group the correlation ``rho``; other planes stay
    independent.
    """
    n = plane_values.size
    mixing = np.eye(n)
    for group in duplicate_groups(plane_values):
        m = len(group)
        correlation = np.full((m, m), rho) + (1.0 - rho) * np.eye(m)
        mixing[np.ix_(group, group)] = np.linalg.cholesky(correlation)
    return mixing


def _noisy_fit(lineshape, separation, heights, plane_values, rng, *, shared_phase, rho, names):
    """One noisy cluster fitted by the engine, as the duplicate check reads it."""
    cluster, start, _amplitudes, model = build_cluster(
        lineshape, separation, heights, shared_phase=shared_phase
    )
    first, second = CHAINS[lineshape]
    rows = [np.unique(index) for index in cluster.grid_indices]
    left = first.noise_operator(rows[0])
    right = second.noise_operator(rows[1])
    white = rng.standard_normal((plane_values.size, left.shape[1], right.shape[1]))
    white = np.einsum("st,tij->sij", copies_mixing(plane_values, rho), white)
    noise = [(left @ w @ right.T).ravel() for w in white]
    cluster.data = cluster.data + SIGMA * np.stack(noise, axis=1)
    evaluation = _fit(cluster, start.copy(deep=True), model)
    factors, amplitudes = evaluation.covariance_factors, evaluation.amplitudes
    flags = tuple(flag.flagged for flag in profile_correlations(factors, amplitudes))
    return RecordedCluster(names, plane_values, amplitudes, factors, flags)


def realization(case: Case, seed: tuple[int, ...]) -> dict:
    """One realization: the pools' Σq and Σk, exclusions, and each peak's contrasts."""
    rng = np.random.default_rng(seed)
    t = DELAYS[case.copies]
    peaks = 1 if case.separation is None else 2
    heights = np.exp(-np.outer(RATES[:peaks], t))
    clusters = [
        _noisy_fit(
            case.lineshape, case.separation, heights, t, rng,
            shared_phase=case.shared_phase, rho=case.copies_correlation,
            names=("A", "B")[:peaks],
        )
    ]  # fmt: skip
    if case.coincident:
        clusters.append(
            _noisy_fit(
                case.lineshape, (0.0, 0.0), np.exp(-np.outer(RATES, t)), t, rng,
                shared_phase=False, rho=case.copies_correlation, names=("C1", "C2"),
            )
        )  # fmt: skip
    check = duplicate_plane_check(clusters)
    contrasts = _contrast_matrix(t)
    main = clusters[0]
    assert main.amplitudes is not None
    assert main.factors is not None
    stated = peak_covariances_between_planes(main.factors, main.amplitudes)
    return {
        "pools": {
            name: [pool.q, pool.contrasts, pool.peaks]
            for name, pool in (("unflagged", check.unflagged), ("flagged", check.flagged))
        },
        "excluded": dict(check.excluded),
        "contrasts": (main.amplitudes @ contrasts.T).tolist(),
        "stated": np.einsum("is,lst,jt->lij", contrasts, stated, contrasts).tolist(),
        "flagged": list(main.flagged),
    }


def _contrast_matrix(plane_values: np.ndarray) -> np.ndarray:
    """Orthonormal within-group contrasts, stacked (k × S), as the check builds them."""
    rows = []
    for group in duplicate_groups(plane_values):
        for j in range(1, len(group)):
            row = np.zeros(plane_values.size)
            row[list(group[:j])] = 1.0
            row[group[j]] = -float(j)
            rows.append(row / np.sqrt(j * (j + 1.0)))
    return np.array(rows)


def q_over_k(rows: list[dict], pool: str) -> np.ndarray:
    """Σq/Σk per realization over the named pool ("all" pools both); NaN where empty."""
    names = ("unflagged", "flagged") if pool == "all" else (pool,)
    q = np.array([sum(r["pools"][n][0] for n in names) for r in rows], dtype=float)
    k = np.array([sum(r["pools"][n][1] for n in names) for r in rows], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(k > 0, q / np.where(k > 0, k, 1.0), np.nan)


def gate(values: np.ndarray) -> dict:
    """The mean of q/k over realizations, its SE from their spread, and the verdict."""
    values = values[np.isfinite(values)]
    n = values.size
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    return {
        "realizations_used": int(n),
        "mean_q_over_k": mean,
        "se": se,
        "deviation_se": (mean - 1.0) / se,
        "rms": float(np.sqrt(mean)),
        "passes": bool(abs(mean - 1.0) <= GATE_SE * se),
    }


def analytical(rows: list[dict]) -> dict:
    """The Monte Carlo covariance of each peak's contrasts against the stated CΣCᵀ."""
    contrasts = np.array([r["contrasts"] for r in rows])  # N × L × k
    stated = np.array([r["stated"] for r in rows]).mean(axis=0)  # L × k × k
    n = contrasts.shape[0]
    se = float(np.sqrt(2.0 / (n - 1)))
    ratios, correlation_gap = [], 0.0
    for peak in range(contrasts.shape[1]):
        observed = np.cov(contrasts[:, peak, :], rowvar=False).reshape(stated.shape[1:])
        ratios.extend((np.diag(observed) / np.diag(stated[peak])).tolist())
        scale = np.sqrt(np.outer(np.diag(stated[peak]), np.diag(stated[peak])))
        gap = np.abs(observed / scale - stated[peak] / scale)
        np.fill_diagonal(gap, 0.0)
        correlation_gap = max(correlation_gap, float(gap.max()))
    ratios_array = np.array(ratios)
    return {
        "variance_ratios": ratios,
        "se": se,
        "largest_deviation_se": float(np.max(np.abs(ratios_array - 1.0)) / se),
        "largest_covariance_gap": correlation_gap,
        "passes": bool(np.all(np.abs(ratios_array - 1.0) <= GATE_SE * se)),
    }


def summarize(case: Case, rows: list[dict]) -> dict:
    """The case's statistic, analytical check, flags and exclusions."""
    pools = {}
    for name in ("unflagged", "flagged"):
        values = q_over_k(rows, name)
        if np.isfinite(values).sum() > 1:
            pools[name] = gate(values) | {
                "peaks_per_realization": float(np.mean([r["pools"][name][2] for r in rows]))
            }
    reasons: dict[str, int] = {}
    for r in rows:
        for reason in set(r["excluded"].values()):
            reasons[reason] = reasons.get(reason, 0) + 1
    statistic = gate(q_over_k(rows, case.pool))
    check = analytical(rows)
    return {
        "realizations": len(rows),
        "pool": case.pool,
        "gated": case.gated,
        "copies_correlation": case.copies_correlation,
        "statistic": statistic,
        "pools": pools,
        "flagged_fraction": float(np.mean([any(r["flagged"]) for r in rows])),
        "realizations_with_exclusion": reasons,
        "analytical": check,
        "passes": (statistic["passes"] and check["passes"]) if case.gated else None,
    }


def _run(job: tuple[Case, tuple[int, ...]]) -> dict:
    return realization(*job)


def _rounded(value):
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in value]
    return value


def main() -> None:
    """Run every case and report the gate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--realizations", type=int, default=400)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    jobs = [
        (CASES[name], (args.seed, list(CASES).index(name), i))
        for name in args.cases
        for i in range(args.realizations)
    ]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(_run, jobs, chunksize=16))
    n = args.realizations
    report = {
        name: summarize(CASES[name], rows[k * n : (k + 1) * n]) for k, name in enumerate(args.cases)
    }
    failing = [name for name, s in report.items() if s["passes"] is False]
    result = {
        "realizations": n,
        "seed": args.seed,
        "noise": SIGMA,
        "criterion": "mean q/k within ±3 SE of 1 (SE from the realizations); analytical "
        "variance ratios within ±3 SE of 1, SE √(2/(N − 1))",
        "gate": {
            "cases": sum(s["gated"] for s in report.values()),
            "failing": failing,
            "passes": not failing,
        },
        "cases": report,
    }
    text = json.dumps(_rounded(result), indent=1, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    for name, s in report.items():
        st, an = s["statistic"], s["analytical"]
        verdict = {True: "pass", False: "FAIL", None: "recorded"}[s["passes"]]
        print(
            f"{name}: q/k {st['mean_q_over_k']:.4f} ± {st['se']:.4f} "
            f"({st['deviation_se']:+.1f} SE), RMS {st['rms']:.3f}; analytical worst "
            f"{an['largest_deviation_se']:.1f} SE; {verdict}"
        )
    print(json.dumps(_rounded(result["gate"]), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
