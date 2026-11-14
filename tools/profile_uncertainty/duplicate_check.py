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
unless stated), q = |Ca|²/c with c the peak's conditional variance; E[q] = k at first
order. **Gate:** the mean of q/k over realizations within ±3 SE of 1, the SE from the
realizations' own spread.

**Analytical check** (every case): the Monte Carlo covariance of each peak's contrasts
Ca over realizations against the contrast covariance the check states, cI (k × k),
averaged over realizations; each variance ratio within ±3 SE of 1, SE √(2/(N − 1)),
and the off-diagonal covariances, relative to the variances, recorded against 0.

**Cases:**
- isolated peaks, two and three copies, every lineshape for two copies;
- pairs 3 points apart (unflagged), two and three copies; with a cluster phase
  shared by both peaks (the shared parameter);
- a pair 2 points apart, flagged: the flagged pool;
- exclusions: an isolated peak checked with a coincident pair (both peaks at one
  position), whose conditional covariance the fit withholds where the shapes stay
  coincident; the gate is on the unflagged pool, which the exclusions must leave
  calibrated. Where the pair fits apart, its conditional covariance is published:
  with its lineshape factors withheld, its peaks are excluded for that; otherwise
  they join a pool, recorded by pool with their own Σq/Σk.

**What the lineshape exclusion hides** (every case): peaks excluded because their
lineshape covariance factors are withheld still have a finite conditional variance.
Their q and k are computed here from it, tool-side, and recorded by the pool they
would have joined (the flag as the run derives it: not assessed, so unset), with
the case's statistic as it would have been with them.

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
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/
from profile_coverage import CHAINS, SIGMA, _fit, build_cluster

from peakfit.engine.scientific.replicates import (
    DuplicatePool,
    ExclusionReason,
    RecordedCluster,
    duplicate_groups,
    duplicate_plane_check,
    orthonormal_contrasts,
)

if TYPE_CHECKING:
    from peakfit.engine.domain.config import LineshapeName

type Copies = Literal["two copies", "three copies"]
type PoolName = Literal["unflagged", "flagged"]
type CasePool = Literal["all", "unflagged", "flagged"]  # "all" pools both

RATES = (1.0, 2.0)
DELAYS: dict[Copies, np.ndarray] = {
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
    copies: Copies
    separation: tuple[float, float] | None = None
    shared_phase: bool = False
    pool: CasePool = "all"
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


def _noisy_fit(
    case: Case,
    separation: tuple[float, float] | None,
    heights: np.ndarray,
    rng: np.random.Generator,
    *,
    shared_phase: bool,
    names: tuple[str, ...],
) -> RecordedCluster:
    """One noisy cluster of ``case`` fitted by the engine, as the duplicate check reads it."""
    plane_values = DELAYS[case.copies]
    cluster, start, _amplitudes, model = build_cluster(
        case.lineshape, separation, heights, shared_phase=shared_phase
    )
    first, second = CHAINS[case.lineshape]
    rows = [np.unique(index) for index in cluster.grid_indices]
    left = first.noise_operator(rows[0])
    right = second.noise_operator(rows[1])
    white = rng.standard_normal((plane_values.size, left.shape[1], right.shape[1]))
    white = np.einsum("st,tij->sij", copies_mixing(plane_values, case.copies_correlation), white)
    noise = [(left @ w @ right.T).ravel() for w in white]
    cluster.data = cluster.data + SIGMA * np.stack(noise, axis=1)
    evaluation = _fit(cluster, start.copy(deep=True), model)
    return RecordedCluster.fitted(
        names, plane_values, evaluation.amplitudes, evaluation.covariance_factors
    )


@dataclass(frozen=True, slots=True)
class PoolSums:
    """One pool of one realization: Σq, Σk and the peaks pooled."""

    q: float
    contrasts: int
    peaks: int

    @classmethod
    def of(cls, pool: DuplicatePool) -> PoolSums:
        """The sums of one pool of the check."""
        return cls(pool.q, pool.contrasts, pool.peaks)


@dataclass(frozen=True, slots=True)
class Realization:
    """One realization's check, and the contrasts of its main cluster's peaks.

    ``contrasts`` (L × k) are the peaks' contrasts Ca; ``stated`` (L × k × k) their
    covariance as the check states it, cI. ``coincident`` holds the pools the peaks
    of the coincident pair joined, if any. ``withheld`` holds, by the pool they would
    have joined, the sums of the peaks excluded for withheld lineshape covariance.
    """

    pools: dict[PoolName, PoolSums]
    excluded: dict[str, ExclusionReason]
    contrasts: np.ndarray
    stated: np.ndarray
    flagged: tuple[bool, ...]
    coincident: dict[PoolName, PoolSums]
    withheld: dict[PoolName, PoolSums]


def withheld_sums(
    clusters: list[RecordedCluster], excluded: dict[str, ExclusionReason]
) -> dict[PoolName, PoolSums]:
    """The sums the peaks excluded for withheld lineshape covariance would have given.

    q = |Ca|²/c from each peak's conditional variance c, in the pool its flag, as the
    run derives it, would have put it in.
    """
    sums: dict[PoolName, PoolSums] = {}
    for cluster in clusters:
        for peak, name in enumerate(cluster.peak_names):
            if excluded.get(name) is not ExclusionReason.LINESHAPE_COVARIANCE_WITHHELD:
                continue
            assert cluster.amplitudes is not None
            assert cluster.factors is not None
            groups = duplicate_groups(cluster.plane_values)
            contrasts = orthonormal_contrasts(groups, cluster.plane_values.size)
            variance = float(cluster.factors.conditional[peak, peak])
            q = float(np.sum((contrasts @ cluster.amplitudes[peak]) ** 2)) / variance
            pool: PoolName = "flagged" if cluster.flagged[peak] else "unflagged"
            total = sums.get(pool, PoolSums(0.0, 0, 0))
            sums[pool] = PoolSums(
                total.q + q, total.contrasts + contrasts.shape[0], total.peaks + 1
            )
    return sums


def realization(case: Case, seed: tuple[int, ...]) -> Realization:
    """One realization: the pools' Σq and Σk, exclusions, and each peak's contrasts."""
    rng = np.random.default_rng(seed)
    t = DELAYS[case.copies]
    peaks = 1 if case.separation is None else 2
    heights = np.exp(-np.outer(RATES[:peaks], t))
    clusters = [
        _noisy_fit(
            case, case.separation, heights, rng,
            shared_phase=case.shared_phase, names=("A", "B")[:peaks],
        )
    ]  # fmt: skip
    if case.coincident:
        clusters.append(
            _noisy_fit(
                case, (0.0, 0.0), np.exp(-np.outer(RATES, t)), rng,
                shared_phase=False, names=("C1", "C2"),
            )
        )  # fmt: skip
    check = duplicate_plane_check(clusters)
    contrasts = orthonormal_contrasts(duplicate_groups(t), t.size)
    main = clusters[0]
    assert main.amplitudes is not None
    assert main.factors is not None
    variances = np.diag(main.factors.conditional)
    pooled: dict[PoolName, PoolSums] = {}
    if case.coincident:
        alone = duplicate_plane_check(clusters[1:])
        names: tuple[PoolName, ...] = ("unflagged", "flagged")
        for name, pool in zip(names, (alone.unflagged, alone.flagged), strict=True):
            if pool.peaks:
                pooled[name] = PoolSums.of(pool)
    return Realization(
        pools={"unflagged": PoolSums.of(check.unflagged), "flagged": PoolSums.of(check.flagged)},
        excluded=dict(check.excluded),
        contrasts=main.amplitudes @ contrasts.T,
        stated=variances[:, None, None] * np.eye(contrasts.shape[0]),
        flagged=main.flagged,
        coincident=pooled,
        withheld=withheld_sums(clusters, check.excluded),
    )


def q_over_k(rows: list[Realization], pool: CasePool, *, withheld: bool = False) -> np.ndarray:
    """Σq/Σk per realization over the named pool ("all" pools both); NaN where empty.

    With ``withheld``, the peaks excluded for withheld lineshape covariance count in
    the pool they would have joined, as before that exclusion.
    """
    names: tuple[PoolName, ...] = ("unflagged", "flagged") if pool == "all" else (pool,)
    empty = PoolSums(0.0, 0, 0)

    def extra(r: Realization, n: PoolName) -> PoolSums:
        return r.withheld.get(n, empty) if withheld else empty

    q = np.array([sum(r.pools[n].q + extra(r, n).q for n in names) for r in rows])
    k = np.array(
        [sum(r.pools[n].contrasts + extra(r, n).contrasts for n in names) for r in rows],
        dtype=float,
    )
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


def analytical(rows: list[Realization]) -> dict:
    """The Monte Carlo covariance of each peak's contrasts against the stated cI."""
    contrasts = np.array([r.contrasts for r in rows])  # N × L × k
    stated = np.array([r.stated for r in rows]).mean(axis=0)  # L × k × k
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


def blocks(values: np.ndarray, count: int) -> dict:
    """The statistic over ``count`` disjoint, equal blocks of consecutive realizations.

    Each block's mean q/k and its deviation in its own SE; the χ² of the block means
    about their pooled mean (count − 1 degrees of freedom) and its upper-tail p-value;
    and the gate applied to every block but the first, an independent replication of
    the first block's realizations.
    """
    parts = np.array_split(values, count)
    stats = [gate(part) for part in parts]
    means = np.array([s["mean_q_over_k"] for s in stats])
    ses = np.array([s["se"] for s in stats])
    weights = 1.0 / ses**2
    pooled = float(np.sum(weights * means) / np.sum(weights))
    chi2 = float(np.sum(((means - pooled) / ses) ** 2))
    return {
        "count": count,
        "deviation_se": [s["deviation_se"] for s in stats],
        "chi2": chi2,
        "p_value": float(scipy.stats.chi2.sf(chi2, count - 1)),
        "replication": gate(np.concatenate(parts[1:])),
    }


def summarize(case: Case, rows: list[Realization], block_count: int = 1) -> dict:
    """The case's statistic, analytical check, flags and exclusions."""
    pools = {}
    names: tuple[PoolName, ...] = ("unflagged", "flagged")
    for name in names:
        values = q_over_k(rows, name)
        if np.isfinite(values).sum() > 1:
            pools[name] = gate(values) | {
                "peaks_per_realization": float(np.mean([r.pools[name].peaks for r in rows]))
            }
    reasons: dict[str, int] = {}
    for r in rows:
        for reason in dict.fromkeys(r.excluded.values()):
            reasons[reason.value] = reasons.get(reason.value, 0) + 1
    coincident = {}
    for name in names:
        joined = [r.coincident[name] for r in rows if name in r.coincident]
        if joined:
            q, k = sum(p.q for p in joined), sum(p.contrasts for p in joined)
            coincident[name] = {
                "realizations": len(joined),
                "peaks": sum(p.peaks for p in joined),
                "pooled_q_over_k": q / k,
            }
    values = q_over_k(rows, case.pool)
    statistic = gate(values)
    withheld: dict = {"realizations": sum(bool(r.withheld) for r in rows), "peaks": {}}
    for r in rows:
        for peak, reason in r.excluded.items():
            if reason is ExclusionReason.LINESHAPE_COVARIANCE_WITHHELD:
                withheld["peaks"][peak] = withheld["peaks"].get(peak, 0) + 1
    for name in names:
        joined = [r.withheld[name] for r in rows if name in r.withheld]
        if joined:
            q, k = sum(p.q for p in joined), sum(p.contrasts for p in joined)
            withheld[f"would_join_{name}"] = {
                "realizations": len(joined),
                "peaks": sum(p.peaks for p in joined),
                "pooled_q_over_k": q / k,
            }
    if withheld["realizations"]:
        withheld["statistic_with_them"] = gate(q_over_k(rows, case.pool, withheld=True))
    check = analytical(rows)
    return {
        "realizations": len(rows),
        "pool": case.pool,
        "gated": case.gated,
        "copies_correlation": case.copies_correlation,
        "statistic": statistic,
        **({"blocks": blocks(values, block_count)} if block_count > 1 else {}),
        "pools": pools,
        "flagged_fraction": float(np.mean([any(r.flagged) for r in rows])),
        "realizations_with_exclusion": reasons,
        "lineshape_covariance_withheld": withheld,
        **({"coincident_pair_pooled": coincident} if case.coincident else {}),
        "analytical": check,
        "passes": (statistic["passes"] and check["passes"]) if case.gated else None,
    }


def _run(job: tuple[Case, tuple[int, ...]]) -> Realization:
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
    parser.add_argument("--blocks", type=int, default=1)
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
        name: summarize(CASES[name], rows[k * n : (k + 1) * n], args.blocks)
        for k, name in enumerate(args.cases)
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
        if "blocks" in s:
            b, rep = s["blocks"], s["blocks"]["replication"]
            print(
                f"  blocks: {' '.join(f'{d:+.1f}' for d in b['deviation_se'])} SE; "
                f"χ² {b['chi2']:.1f} (p {b['p_value']:.2f}); replication q/k "
                f"{rep['mean_q_over_k']:.4f} ({rep['deviation_se']:+.1f} SE)"
            )
        withheld = s["lineshape_covariance_withheld"]
        if withheld["realizations"]:
            joined = "; ".join(
                f"would join {pool}: {w['realizations']} realizations, {w['peaks']} peaks, "
                f"pooled q/k {w['pooled_q_over_k']:.3f}"
                for pool in ("unflagged", "flagged")
                if (w := withheld.get(f"would_join_{pool}"))
            )
            with_them = withheld["statistic_with_them"]
            print(
                f"  lineshape covariance withheld: {joined}; statistic with them "
                f"{with_them['mean_q_over_k']:.4f} ({with_them['deviation_se']:+.1f} SE)"
            )
    print(json.dumps(_rounded(result["gate"]), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
