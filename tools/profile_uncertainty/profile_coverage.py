"""T3: coverage of the profile uncertainties, and the correlated-profile thresholds.

The release gate of ``docs/profile-uncertainty.md`` (§8, T3). Every cluster is a 2D
pseudo-3D drawn with PeakFit's own ``sp1``, ``sp2`` or ``no_apod`` lineshapes, whose
equality with NMRPipe processing ``tests/test_lineshapes_equivalence.py`` checks, plus
noise through the exact processing operators of ``validate_sandwich_errors.py``:
complex white FID noise, sine bell (or none), first-point scaling, zero fill, Fourier
transform and P1, on both axes. The engine fits it (``fit_cluster`` from the truth,
``evaluate_analytical_model`` under the processing-derived correlation), as PeakFit
publishes it.

**Profiles** (two peaks with differing profiles, or one):
- 1 plane: only the marginal σ is defined;
- 3 planes, a decay: downstream, A·e^(−R·t) per peak, free scale A;
- 30 planes, CPMG-like: one reference and 29 ν_CPMG; R2eff; downstream, a ChemEx-like
  global fit with a free scale per profile and kex shared by both peaks;
- 30 planes, CEST-like: two reference planes and 28 offsets with one dip per peak;
  downstream, A·(1 − d·w²/(w² + (ω − ω₀)²)) per peak, free scale A.

**Quantities, scatter over realizations against the stated σ:**
- marginal σ, of every absolute amplitude;
- profile σ: Σ_rel = PΣP + c·uuᵀ, so the scatter of the amplitudes with their own
  profile direction removed, (I − uuᵀ)(a − a_true) with u the true unit profile, is
  compared with σ_profile² − c·û² of the fit;
- ratios to the reference planes and R2eff, through ``profiles.normalize`` with each
  peak's covariance between planes;
- the downstream parameters, fitted with the profile σ as independent per-plane errors,
  as ChemEx and other downstream programs read them. A fit with the joint covariance of
  both peaks is the control: it separates the downstream fit's own nonlinearity from
  the error model.

**Diagnostics of the correlated-profile flag** (§6), for each peak l:
- the largest correlated mode, λ_max(Σ_rel − cI)/c;
- the within-plane amplitude correlation with a cluster partner, max |ρ_lk| of the
  conditional covariance.

**Thresholds.** ``first_order`` maps each configuration of a dense sweep to the
diagnostics and to the exact first-order misstatement of every downstream parameter's
σ when the profile σ is used as independent errors: √(KΣKᵀ / (JᵀWJ)⁻¹), with Σ the
covariance between planes of both peaks (validated against the sandwich, T1). The
thresholds are the largest that keep every unflagged configuration within the
tolerance. The Monte Carlo then gates the reported σ in the unflagged cases.

The default, 400 realizations per case (20 s on 16 cores), checks the machinery; it
cannot resolve the 5% gate (SE of scatter / stated about 3.5%). The record uses 4000
(about 3 minutes), through ``run_all.py``:

    uv run python tools/profile_uncertainty/profile_coverage.py
    uv run python tools/profile_uncertainty/run_all.py profile_coverage
"""

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/
from validate_sandwich_errors import NO_APOD, SP1, SP2, Axis

from peakfit.engine.algorithms.covariance_factors import (
    CORRELATED_MODE_THRESHOLD,
    DOWNSTREAM_SIGMA_TOLERANCE,
    PARTNER_CORRELATION_THRESHOLD,
    covariance_between_planes,
    marginal_standard_errors,
    peak_covariances_between_planes,
    profile_standard_errors,
    relative_profile_covariance,
)
from peakfit.engine.algorithms.evaluation import (
    AnalyticalModelEvaluation,
    evaluate_analytical_model,
)
from peakfit.engine.algorithms.noise_correlation import NoiseModel, cluster_correlation
from peakfit.engine.algorithms.varpro import fit_cluster
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, LineshapeName
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.scientific.profiles import Normalized, normalize, r2eff

if TYPE_CHECKING:
    from collections.abc import Callable

SIGMA = 0.02  # noise SD; the larger peak's height is 1 in its reference plane
LINEWIDTH = 20.0  # Hz, every line
HALF_WINDOW = (5, 6)  # ¹⁵N, ¹H points either side of the pair, beyond half its separation
TOLERANCE = DOWNSTREAM_SIGMA_TOLERANCE  # largest misstatement of a downstream σ, unflagged
# The correlated-profile flag (§6): set from the sweep's boundary, rounded down for
# configurations between its points (see ``threshold_boundary``).
# The engine owns the constants; the record (``thresholds``) is tested against them.
MODE_THRESHOLD = CORRELATED_MODE_THRESHOLD  # largest correlated mode of Σ_rel, relative to c
PARTNER_THRESHOLD = PARTNER_CORRELATION_THRESHOLD  # within-plane |ρ| with a cluster partner
Z95 = 1.96
# The gate (§8.1, criterion (b) of §10 item 5): scatter / stated within SCATTER_GATE;
# 95% coverage within ±COVERAGE_SE_GATE SE for a single quantity, and within
# ±COVERAGE_POOLED_GATE absolute for quantities pooled over planes.
SCATTER_GATE = (0.95, 1.05)
COVERAGE_SE_GATE = 3.0
COVERAGE_POOLED_GATE = 0.01
CHAINS: dict[LineshapeName, tuple[Axis, Axis]] = {"sp1": SP1, "sp2": SP2, "no_apod": NO_APOD}


# ---------------------------------------------------------------- profiles


@dataclass(frozen=True)
class Profile:
    """Two peaks' true heights over the planes, their references and downstream model.

    ``model`` maps the downstream parameters to both peaks' amplitudes (2 × S), in
    height units; ``owners[l]`` are the parameters peak l's profile informs; ``scales``
    are the free scales, which no flag concerns.
    """

    heights: np.ndarray  # (2, S)
    reference: np.ndarray  # reference planes of the ratios; empty: no ratio
    model: Callable[[np.ndarray], np.ndarray] | None = None
    truth: np.ndarray = field(default_factory=lambda: np.zeros(0))
    names: tuple[str, ...] = ()
    owners: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    scales: tuple[int, ...] = ()
    cpmg_time: float | None = None  # T, s: R2eff on the planes outside the reference


DECAY_TIMES = np.array([0.0, 0.6, 1.2])


def _decay(p: np.ndarray) -> np.ndarray:
    return np.array([p[0] * np.exp(-p[1] * DECAY_TIMES), p[2] * np.exp(-p[3] * DECAY_TIMES)])


CPMG_TIME = 0.04
NU_CPMG = np.concatenate([[0.0], np.geomspace(25.0, 1000.0, 29)])


def _cpmg(p: np.ndarray) -> np.ndarray:
    """Fast-exchange (Luz–Meiboom) R2eff; the ν = 0 plane is the reference, no delay."""
    a1, a2, r20_1, r20_2, phi1, phi2, kex = p
    x = kex / (4.0 * np.maximum(NU_CPMG, 1e-9))
    dispersion = 1.0 - np.tanh(x) / x
    out = []
    for scale, r20, phi in ((a1, r20_1, phi1), (a2, r20_2, phi2)):
        profile = np.exp(-CPMG_TIME * (r20 + phi / kex * dispersion))
        profile[NU_CPMG == 0] = 1.0
        out.append(scale * profile)
    return np.array(out)


CEST_REFERENCE = 1.0e3  # far off resonance: the dips are below 10⁻⁷ there
CEST_OFFSETS = np.concatenate([[CEST_REFERENCE, CEST_REFERENCE], np.linspace(-1.0, 1.0, 28)])


def _cest(p: np.ndarray) -> np.ndarray:
    out = []
    for scale, depth, centre, width in p.reshape(2, 4):
        dip = width**2 / (width**2 + (CEST_OFFSETS - centre) ** 2)
        out.append(scale * (1.0 - depth * dip))
    return np.array(out)


def _profile(model, truth, names, owners, scales, reference, **kw) -> Profile:
    return Profile(
        model(truth), np.asarray(reference, int), model, truth, names, owners, scales, **kw
    )


PROFILES: dict[str, Profile] = {
    "1 plane": Profile(np.array([[1.0], [0.7]]), np.zeros(0, int)),
    "3-plane decay": _profile(
        _decay,
        np.array([1.0, 1.0, 0.8, 0.2]),
        ("A_A", "R_A", "A_B", "R_B"),
        ((0, 1), (2, 3)),
        (0, 2),
        [0],
    ),
    "30-plane CPMG": _profile(
        _cpmg,
        np.array([1.0, 0.8, 10.0, 12.0, 4.0e4, 1.0e4, 1500.0]),
        ("A_A", "A_B", "R20_A", "R20_B", "Phi_A", "Phi_B", "kex"),
        ((0, 2, 4, 6), (1, 3, 5, 6)),
        (0, 1),
        [0],
        cpmg_time=CPMG_TIME,
    ),
    "30-plane CEST": _profile(
        _cest,
        np.array([1.0, 0.8, -0.3, 0.15, 0.8, 0.6, 0.4, 0.1]),
        ("A_A", "depth_A", "offset_A", "width_A", "A_B", "depth_B", "offset_B", "width_B"),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        (0, 4),
        [0, 1],
    ),
}


def downstream_fit(
    profile: Profile,
    amplitudes: np.ndarray,
    peaks: int,
    errors: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The downstream parameters and their stated σ, fitted to ``amplitudes``.

    ``errors`` (peaks × S) are independent per-plane σ; ``covariance`` instead whitens
    the flattened amplitudes jointly. A single-peak cluster fits its own parameters
    only, the others held at the truth. Parameters not fitted are NaN.
    """
    model = profile.model
    assert model is not None
    free = np.array(sorted(set().union(*profile.owners[:peaks])))
    if errors is not None:
        whiten = np.diag(1.0 / errors.ravel())
    else:
        assert covariance is not None
        whiten = np.linalg.cholesky(np.linalg.inv(covariance)).T

    def residuals(x):
        p = profile.truth.copy()
        p[free] = x
        return whiten @ (model(p)[:peaks] - amplitudes).ravel()

    start = profile.truth[free]
    fit = least_squares(residuals, start, x_scale=np.abs(start), method="trf")
    values = np.full(profile.truth.size, np.nan)
    sigma = np.full(profile.truth.size, np.nan)
    values[free] = fit.x
    sigma[free] = np.sqrt(np.diag(np.linalg.pinv(fit.jac.T @ fit.jac)))
    return values, sigma


# ---------------------------------------------------------------- clusters


@dataclass(frozen=True)
class Case:
    """One cluster configuration: lineshape, geometry, profile and shared phase."""

    lineshape: LineshapeName
    separation: tuple[float, float] | None  # (¹⁵N, ¹H) points; None: one peak
    profile: str
    partner: float = 1.0  # height of the second peak relative to its profile
    shared_phase: bool = False  # a cluster phase on ¹H, shared by both peaks

    @property
    def peaks(self) -> int:
        """The number of peaks in the cluster."""
        return 1 if self.separation is None else 2


def build(case: Case) -> tuple[Cluster, Parameters, np.ndarray, NoiseModel]:
    """The noiseless cluster, its true parameters and amplitudes, and its noise model."""
    first, second = CHAINS[case.lineshape]
    profile = PROFILES[case.profile]
    planes = profile.heights.shape[1]
    pseudo = SpectralParameters(
        size=planes, sw=1.0, obs=1.0, car=0.0, aq_time=1.0, apocode=0, apodq1=0, apodq2=0,
        apodq3=0, p180=False, direct=False, ft=False,
    )  # fmt: skip
    shape = (planes, first.processing.transform, second.processing.transform)
    spectra = Spectra(
        dic={},
        data=np.zeros(shape),
        z_values=np.arange(float(planes)),
        params=[pseudo, first.parameters(False), second.parameters(True)],
    )
    centre = (shape[1] / 2, shape[2] / 2)
    separation = case.separation or (0.0, 0.0)
    offsets = [(-separation[0] / 2, -separation[1] / 2), (separation[0] / 2, separation[1] / 2)]
    config = FitConfig(lineshape=case.lineshape, fit_phase=["F2"] if case.shared_phase else [])
    peaks = []
    for name, (d1, d2) in zip("AB"[: case.peaks], offsets, strict=False):
        position = [
            float(spectra.spectral_params[0].pts2ppm(centre[0] + d1)),
            float(spectra.spectral_params[1].pts2ppm(centre[1] + d2)),
        ]
        shapes = create_shapes(spectra, config, name, position, [case.lineshape] * 2)
        peak = Peak(name, np.array(position), shapes)
        peak.set_cluster_id(1)
        peaks.append(peak)
    rows = [
        np.arange(round(c - h - abs(d) / 2), round(c + h + abs(d) / 2) + 1)
        for c, h, d in zip(centre, HALF_WINDOW, separation, strict=True)
    ]
    grid = [index.ravel() for index in np.meshgrid(*rows, indexing="ij")]
    params = Parameters.from_peaks(peaks)
    for name in params.get_vary_names():
        if name.endswith(".lw"):
            params[name].value = LINEWIDTH
    cluster = Cluster(1, peaks, grid, np.zeros((grid[0].size, planes)))
    shapes = cluster.evaluate(params)
    heights = profile.heights[: case.peaks] * np.array([1.0, case.partner])[: case.peaks, None]
    # PeakFit's shapes are not height-normalized: amplitudes in units of each height.
    amplitudes = heights / shapes.max(axis=1)[:, None]
    cluster.data = shapes.T @ amplitudes
    return cluster, params, amplitudes, NoiseModel(axes=(first.processing, second.processing))


def _truth(case: Case) -> np.ndarray:
    """The downstream parameters' truth, the partner's scale times its height."""
    profile = PROFILES[case.profile]
    truth = profile.truth.copy()
    truth[profile.scales[1]] *= case.partner
    return truth


def diagnostics(evaluation: AnalyticalModelEvaluation) -> tuple[np.ndarray, np.ndarray]:
    """Each peak's largest correlated mode / c, and its largest |ρ| with a partner."""
    factors, amplitudes = evaluation.covariance_factors, evaluation.amplitudes
    conditional = np.diag(factors.conditional)
    relative = relative_profile_covariance(factors, amplitudes)
    identity = np.eye(amplitudes.shape[1])
    modes = np.array(
        [
            np.linalg.eigvalsh(relative[l] - c * identity).max() / c
            for l, c in enumerate(conditional)
        ]
    )
    correlation = factors.conditional / np.sqrt(np.outer(conditional, conditional))
    np.fill_diagonal(correlation, 0.0)
    return modes, np.abs(correlation).max(axis=1)


def _fit(cluster: Cluster, params: Parameters, model: NoiseModel) -> AnalyticalModelEvaluation:
    fit_cluster(
        params, cluster, SIGMA, correlation=cluster_correlation(model, cluster.grid_indices)
    )
    evaluation = evaluate_analytical_model(cluster, params, SIGMA, model)
    assert isinstance(evaluation, AnalyticalModelEvaluation)
    return evaluation


# ---------------------------------------------------------------- first order


def first_order(case: Case) -> dict:
    """The flag diagnostics and the first-order misstatement of each downstream σ.

    Misstatement: the downstream σ's true first-order SD over the one stated by an
    independent-error fit with the profile σ, √(KΣKᵀ)/√((JᵀWJ)⁻¹); 1 when exact.
    """
    cluster, params, amplitudes, model = build(case)
    profile = PROFILES[case.profile]
    evaluation = _fit(cluster, params, model)
    factors = evaluation.covariance_factors
    modes, partner = diagnostics(evaluation)
    out = {"modes": modes.tolist(), "partner_correlation": partner.tolist()}
    if profile.model is None:
        return out
    heights = profile.heights[: case.peaks] * np.array([1.0, case.partner])[: case.peaks, None]
    unit = heights[:, :1] / amplitudes[:, :1]  # height per amplitude, by peak
    planes = amplitudes.shape[1]
    full = covariance_between_planes(factors, amplitudes).reshape(
        case.peaks * planes, case.peaks * planes
    )
    full = full * np.outer(unit.repeat(planes), unit.repeat(planes))
    errors = profile_standard_errors(factors, amplitudes) * unit
    truth = _truth(case)
    free = np.array(sorted(set().union(*profile.owners[: case.peaks])))
    step = 1e-6 * np.maximum(np.abs(truth[free]), 1e-3)
    jacobian = np.empty((case.peaks * planes, free.size))
    for j, (index, h) in enumerate(zip(free, step, strict=True)):
        up, down = truth.copy(), truth.copy()
        up[index] += h
        down[index] -= h
        jacobian[:, j] = (
            profile.model(up)[: case.peaks] - profile.model(down)[: case.peaks]
        ).ravel() / (2 * h)
    weights = np.diag(errors.ravel() ** -2.0)
    stated = np.linalg.inv(jacobian.T @ weights @ jacobian)
    response = stated @ jacobian.T @ weights
    scatter = response @ full @ response.T
    ratio = np.sqrt(np.diag(scatter) / np.diag(stated))
    out["misstatement"] = {
        profile.names[index]: float(r)
        for index, r in zip(free, ratio, strict=True)
        if index not in profile.scales
    }
    return out


def sweep_cases() -> list[Case]:
    """The configurations the thresholds are chosen from.

    Every downstream profile and lineshape; pairs from 1 to 6 points apart (per axis on
    the diagonal, the same Euclidean distance along one axis), partners of equal or
    0.3× height, and a cluster phase shared by both peaks.
    """
    separations = np.arange(1.0, 6.01, 0.125)
    geometries = (
        lambda d: (d, d),
        lambda d: (d * 2**0.5, 0.0),  # along ¹⁵N
        lambda d: (0.0, d * 2**0.5),  # along ¹H
    )
    downstream = [name for name, profile in PROFILES.items() if profile.model is not None]
    return [
        Case(lineshape, geometry(float(d)), name, partner=partner)
        for lineshape in CHAINS
        for name in downstream
        for geometry in geometries
        for d in separations
        for partner in (1.0, 0.3)
    ] + [
        Case(lineshape, (float(d), float(d)), name, shared_phase=True)
        for lineshape in CHAINS
        for name in downstream
        for d in separations
    ]


def _sweep_peaks(rows: list[tuple[Case, dict]]) -> list[dict]:
    """One entry per peak: its diagnostics, and its worst misstatement, own and shared.

    *Own* parameters are informed by this peak's profile alone (R, Φ, R20, CEST dip);
    *shared* ones by both peaks of the cluster (kex).
    """
    peaks = []
    for case, result in rows:
        profile = PROFILES[case.profile]
        misstatement = result["misstatement"]
        owners = profile.owners[: case.peaks]
        for l, own in enumerate(owners):
            others = set().union(*(o for k, o in enumerate(owners) if k != l))
            errors = {
                kind: max(
                    (
                        abs(misstatement[profile.names[i]] - 1.0)
                        for i in own
                        if i not in profile.scales and ((i in others) == (kind == "shared"))
                    ),
                    default=0.0,
                )
                for kind in ("own", "shared")
            }
            peaks.append(
                {
                    "mode": result["modes"][l],
                    "partner_correlation": result["partner_correlation"][l],
                    **errors,
                }
            )
    return peaks


def threshold_boundary(rows: list[tuple[Case, dict]], tolerance: float = TOLERANCE) -> dict:
    """Where the sweep's downstream σ first leave the tolerance, by mechanism.

    - Correlated modes misstate each peak's own parameters: the mode boundary is the
      smallest largest-mode / c among peaks whose own parameters leave the tolerance.
    - The covariance between partners misstates parameters shared by a joint fit: the
      partner boundary is the smallest |ρ| among peaks below the mode boundary whose
      shared parameters leave it.
    """
    peaks = _sweep_peaks(rows)
    mode = min((p["mode"] for p in peaks if p["own"] > tolerance), default=np.inf)
    partner = min(
        (
            p["partner_correlation"]
            for p in peaks
            if p["mode"] < mode and max(p["own"], p["shared"]) > tolerance
        ),
        default=np.inf,
    )
    return {"tolerance": tolerance, "mode": float(mode), "partner_correlation": float(partner)}


def flagged(modes, partner_correlation) -> np.ndarray:
    """The correlated-profile flag of each peak, from its two diagnostics."""
    return (np.asarray(modes) > MODE_THRESHOLD) | (
        np.asarray(partner_correlation) > PARTNER_THRESHOLD
    )


def unflagged_misstatement(rows: list[tuple[Case, dict]]) -> dict:
    """The chosen thresholds against the sweep: the worst misstatement left unflagged."""
    peaks = _sweep_peaks(rows)
    marks = flagged([p["mode"] for p in peaks], [p["partner_correlation"] for p in peaks])
    kept = [max(p["own"], p["shared"]) for p, f in zip(peaks, marks, strict=True) if not f]
    return {
        "peaks": len(peaks),
        "flagged": int(marks.sum()),
        "worst_unflagged": float(max(kept, default=0.0)),
        "unflagged_beyond_tolerance": int(sum(e > TOLERANCE for e in kept)),
    }


MODE_BINS = (0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.13, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, np.inf)
PARTNER_BINS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0)


def consequences(rows: list[tuple[Case, dict]]) -> dict:
    """The sweep, binned: the worst misstatement in each range of a diagnostic.

    Own parameters by largest mode / c, over every peak; shared parameters (kex) by
    |ρ| with the partner, over the peaks the mode threshold leaves unflagged.
    """
    peaks = _sweep_peaks(rows)

    def table(values, errors, edges):
        out = []
        for low, high in itertools.pairwise(edges):
            inside = [e for v, e in zip(values, errors, strict=True) if low <= v < high]
            if inside:
                upper = None if np.isinf(high) else high  # JSON has no infinity
                out.append({"from": low, "to": upper, "peaks": len(inside), "worst": max(inside)})
        return out

    low_mode = [p for p in peaks if p["mode"] <= MODE_THRESHOLD]
    return {
        "own_by_mode": table([p["mode"] for p in peaks], [p["own"] for p in peaks], MODE_BINS),
        "shared_by_partner_correlation": table(
            [p["partner_correlation"] for p in low_mode],
            [p["shared"] for p in low_mode],
            PARTNER_BINS,
        ),
    }


def gate_summary(report: dict) -> dict:
    """The gate over every unflagged case, under criterion (b), and the strict count.

    Criterion (b) of the specification's §10 item 5: scatter / stated within 0.95–1.05;
    coverage within ±3 SE for one quantity, ±0.01 for quantities pooled over planes.
    The stricter ±2 SE count is kept as information: each such test fails 4.6% of the
    time when the σ are exact, so among many quantities some exceed by chance.
    """
    failing = [
        f"{case}: {name}"
        for case, summary in report.items()
        if summary["passes"] is not None
        for name, q in summary["quantities"].items()
        if not name.endswith("(joint covariance)") and not q["passes"]
    ]
    families: dict[str, dict] = {}
    for case, summary in report.items():
        if summary["passes"] is None:
            continue
        for name, q in summary["quantities"].items():
            if name.endswith("(joint covariance)"):
                continue
            family = name if name in ("marginal", "profile", "ratio", "r2eff") else "downstream"
            entry = families.setdefault(
                family, {"quantities": 0, "scatter_outside": [], "coverage_outside": []}
            )
            entry["quantities"] += 1
            if not q["scatter_within_gate"]:
                entry["scatter_outside"].append(f"{case}: {name}")
            if not q["coverage_within_2se"]:
                z = (q["coverage_95"] - 0.95) / q["coverage_se"]
                entry["coverage_outside"].append(f"{case}: {name} ({z:+.1f} SE)")
    total = sum(f["quantities"] for f in families.values())
    return {
        "criterion": {
            "scatter": list(SCATTER_GATE),
            "coverage_single": f"±{COVERAGE_SE_GATE:g} SE",
            "coverage_pooled": f"±{COVERAGE_POOLED_GATE:g}",
        },
        "cases": sum(s["passes"] is not None for s in report.values()),
        "quantities": total,
        "failing": failing,
        "passes": not failing,
        "strict_2se": {
            "note": (
                "Information, not the gate: coverage within ±2 SE per quantity. R2eff "
                "over-covers by about 0.1–0.2 points: σ(R2eff) = σ_r/(r̂T) uses the "
                "fitted ratio, so the stated σ grows where R2eff errs high (first-order "
                "propagation through the logarithm)."
            ),
            "scatter_outside": sum(len(f["scatter_outside"]) for f in families.values()),
            "coverage_outside": sum(len(f["coverage_outside"]) for f in families.values()),
            "coverage_outside_by_chance": round(0.0455 * total, 1),
            "families": families,
        },
    }


# ---------------------------------------------------------------- Monte Carlo


def _ratios(case: Case, estimate, truth, covariances) -> dict:
    """Ratio and R2eff errors and stated σ, or why ``normalize`` gives none."""
    profile = PROFILES[case.profile]
    ratios, rates = [[], []], [[], []]
    for l in range(case.peaks):
        shown = normalize(estimate[l], covariances[l], profile.reference)
        if not isinstance(shown, Normalized):
            return {"ratio_unavailable": shown.reason.split(":")[0]}
        exact = truth[l] / truth[l, profile.reference].mean()
        keep = ~shown.in_reference if shown.exact else np.ones(exact.size, bool)
        ratios[0].append((shown.values - exact)[keep].tolist())
        ratios[1].append(shown.errors[keep].tolist())
        if profile.cpmg_time is not None:
            rate, sigma = r2eff(shown.values[keep], shown.errors[keep], profile.cpmg_time)
            rates[0].append((rate + np.log(exact[keep]) / profile.cpmg_time).tolist())
            rates[1].append(sigma.tolist())
    return {"ratio": ratios} | ({"r2eff": rates} if profile.cpmg_time is not None else {})


def realization(case: Case, seed: tuple[int, ...]) -> dict:
    """One noisy fit: every reported σ and estimate, the downstream fits, the diagnostics."""
    cluster, start, amplitudes, model = build(case)
    profile = PROFILES[case.profile]
    first, second = CHAINS[case.lineshape]
    rows = [np.unique(index) for index in cluster.grid_indices]
    left = first.noise_operator(rows[0])
    right = second.noise_operator(rows[1])
    rng = np.random.default_rng(seed)
    planes = amplitudes.shape[1]
    noise = [
        (left @ rng.standard_normal((left.shape[1], right.shape[1])) @ right.T).ravel()
        for _ in range(planes)
    ]
    cluster.data = cluster.data + SIGMA * np.stack(noise, axis=1)
    params = start.copy(deep=True)
    evaluation = _fit(cluster, params, model)
    factors, fitted = evaluation.covariance_factors, evaluation.amplitudes
    if not factors.available:
        return {"available": False, "reason": "lineshape factors withheld"}
    heights = profile.heights[: case.peaks] * np.array([1.0, case.partner])[: case.peaks, None]
    unit = heights[:, :1] / amplitudes[:, :1]  # everything in height units
    modes, partner = diagnostics(evaluation)
    conditional = np.diag(factors.conditional) * unit[:, 0] ** 2
    marginal = marginal_standard_errors(factors, fitted) * unit
    profile_sigma = profile_standard_errors(factors, fitted) * unit
    estimate, truth = fitted * unit, heights
    true_unit = truth / np.linalg.norm(truth, axis=1, keepdims=True)
    fitted_unit = estimate / np.linalg.norm(estimate, axis=1, keepdims=True)
    error = estimate - truth
    removed = error - true_unit * np.sum(true_unit * error, axis=1, keepdims=True)
    out = {
        "available": True,
        "modes": modes.tolist(),
        "partner_correlation": partner.tolist(),
        "marginal": [error.tolist(), marginal.tolist()],
    }
    if planes > 1:  # one plane has no profile: its free scale absorbs everything
        out["profile"] = [
            removed.tolist(),
            np.sqrt(
                np.clip(profile_sigma**2 - conditional[:, None] * fitted_unit**2, 0, None)
            ).tolist(),
        ]
    if profile.reference.size:
        covariances = peak_covariances_between_planes(factors, fitted) * unit[:, :, None] ** 2
        out.update(_ratios(case, estimate, truth, covariances))
    if profile.model is not None:
        truth_p = _truth(case)
        scaled = replace(profile, truth=truth_p)
        values, sigma = downstream_fit(scaled, estimate, case.peaks, errors=profile_sigma)
        joint = covariance_between_planes(factors, fitted).reshape(
            case.peaks * planes, case.peaks * planes
        ) * np.outer(unit.repeat(planes), unit.repeat(planes))
        control, control_sigma = downstream_fit(scaled, estimate, case.peaks, covariance=joint)
        out["downstream"] = [(values - truth_p).tolist(), sigma.tolist()]
        out["downstream_joint"] = [(control - truth_p).tolist(), control_sigma.tolist()]
    return out


def calibration(errors: np.ndarray, stated: np.ndarray) -> dict:
    """Scatter / stated σ per quantity, and 95% coverage with its standard error.

    ``errors`` and ``stated`` are (realizations, quantities). Scatter is the SD over
    realizations; stated, the median. Coverage pools the quantities; its SE comes from
    the per-realization covered fractions, so correlated quantities are not counted
    as independent.
    """
    errors, stated = np.asarray(errors, float), np.asarray(stated, float)
    keep = np.all(np.isfinite(stated), axis=0)
    errors, stated = errors[:, keep], stated[:, keep]
    ratio = errors.std(axis=0, ddof=1) / np.median(stated, axis=0)
    covered = (np.abs(errors) < Z95 * stated).mean(axis=1)
    n = covered.size
    coverage = float(covered.mean())
    # Binomial SE for one quantity; for several, the spread of per-realization fractions,
    # never below the binomial value of the pooled fraction for one quantity.
    single = errors.shape[1] == 1
    se = (
        float(np.sqrt(0.95 * 0.05 / n))
        if single
        else float(max(covered.std(ddof=1), 0.0) / np.sqrt(n))
    )
    deviation = abs(coverage - 0.95)
    scatter_ok = bool(np.all((ratio >= SCATTER_GATE[0]) & (ratio <= SCATTER_GATE[1])))
    coverage_ok = bool(
        deviation <= COVERAGE_SE_GATE * se if single else deviation <= COVERAGE_POOLED_GATE
    )
    return {
        "quantities": int(ratio.size),
        "min": float(ratio.min()),
        "max": float(ratio.max()),
        "bias_over_sd": float(np.max(np.abs(errors.mean(axis=0)) / errors.std(axis=0, ddof=1))),
        "coverage_95": coverage,
        "coverage_se": se,
        "coverage_within_2se": bool(deviation <= 2 * se),
        "coverage_within_criterion": coverage_ok,
        "scatter_within_gate": scatter_ok,
        "passes": scatter_ok and coverage_ok,
    }


def summarize(case: Case, rows: list[dict], predicted: dict) -> dict:
    """Every quantity's calibration, the flag the case raises, and its gate verdict.

    The case is gated where its noise-free diagnostics (``predicted``, from
    ``first_order``) raise no flag; otherwise it is reported. Ratios that
    ``normalize`` withholds are counted and left out of the ratio calibration only.
    """
    usable = [r for r in rows if r["available"]]
    profile = PROFILES[case.profile]
    marks = flagged(predicted["modes"], predicted["partner_correlation"])
    out: dict = {
        "realizations": len(rows),
        "factors_withheld": len(rows) - len(usable),
        "noise_free": predicted,
        "flagged": bool(marks.any()),
        "quantities": {},
    }
    if not usable:
        out["passes"] = None
        return out
    modes = np.array([r["modes"] for r in usable])
    partner = np.array([r["partner_correlation"] for r in usable])
    out["flagged_fraction"] = float(flagged(modes, partner).any(axis=1).mean())
    withheld: dict[str, int] = {}
    for r in usable:
        if "ratio_unavailable" in r:
            withheld[r["ratio_unavailable"]] = withheld.get(r["ratio_unavailable"], 0) + 1
    if withheld:
        out["ratios_withheld"] = withheld
    for key in ("marginal", "profile", "ratio", "r2eff"):
        present = [r for r in usable if key in r]
        if not present:
            continue
        errors = np.array([np.ravel(r[key][0]) for r in present])
        stated = np.array([np.ravel(r[key][1]) for r in present])
        out["quantities"][key] = calibration(errors, stated)
    for key in ("downstream", "downstream_joint"):
        if key not in usable[0]:
            continue
        errors = np.array([r[key][0] for r in usable])
        stated = np.array([r[key][1] for r in usable])
        for index, name in enumerate(profile.names):
            if index in profile.scales or not np.isfinite(stated[:, index]).all():
                continue
            label = name if key == "downstream" else f"{name} (joint covariance)"
            out["quantities"][label] = calibration(errors[:, [index]], stated[:, [index]])
    gated = [k for k in out["quantities"] if not k.endswith("(joint covariance)")]
    out["passes"] = None if out["flagged"] else all(out["quantities"][k]["passes"] for k in gated)
    return out


def _run(job: tuple[Case, tuple[int, ...]]) -> dict:
    return realization(*job)


def _first(case: Case) -> tuple[Case, dict]:
    return case, first_order(case)


MONTE_CARLO: dict[str, Case] = {}
for _lineshape in CHAINS:
    for _profile_name in PROFILES:
        MONTE_CARLO[f"{_lineshape}, {_profile_name}, isolated"] = Case(
            _lineshape, None, _profile_name
        )
        for _d in (4.0, 3.0, 2.5, 2.0):
            MONTE_CARLO[f"{_lineshape}, {_profile_name}, pair {_d:g} points"] = Case(
                _lineshape, (_d, _d), _profile_name
            )
for _profile_name in PROFILES:
    MONTE_CARLO[f"sp2, {_profile_name}, pair 3 points, shared phase"] = Case(
        "sp2", (3.0, 3.0), _profile_name, shared_phase=True
    )
    MONTE_CARLO[f"sp2, {_profile_name}, pair 1 point (near-coalescent)"] = Case(
        "sp2", (1.0, 1.0), _profile_name
    )


def _rounded(value):
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in value]
    return value


def main() -> None:
    """Sweep the thresholds, run the Monte Carlo, and report the gate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--realizations", type=int, default=400)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--cases", nargs="*", default=list(MONTE_CARLO))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    cases = [MONTE_CARLO[name] for name in args.cases]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        sweep = list(pool.map(_first, sweep_cases(), chunksize=8))
        predicted = [result for _, result in pool.map(_first, cases)]
        jobs = [
            (MONTE_CARLO[name], (args.seed, list(MONTE_CARLO).index(name), i))
            for name in args.cases
            for i in range(args.realizations)
        ]
        rows = list(pool.map(_run, jobs, chunksize=16))
    report = {
        name: summarize(
            case, rows[k * args.realizations : (k + 1) * args.realizations], predicted[k]
        )
        for k, (name, case) in enumerate(zip(args.cases, cases, strict=True))
    }
    gate = gate_summary(report)
    result = {
        "realizations": args.realizations,
        "seed": args.seed,
        "noise": SIGMA,
        "thresholds": {
            "tolerance": TOLERANCE,
            "mode": MODE_THRESHOLD,
            "partner_correlation": PARTNER_THRESHOLD,
            "boundary": threshold_boundary(sweep),
            "sweep": unflagged_misstatement(sweep),
            "consequences": consequences(sweep),
        },
        "gate": gate,
        "cases": report,
    }
    text = json.dumps(_rounded(result), indent=1, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    for name, summary in report.items():
        print(f"{name}: {'flagged' if summary['flagged'] else 'gated'}")
        for label, q in summary["quantities"].items():
            z = (q["coverage_95"] - 0.95) / q["coverage_se"]
            print(
                f"  {label:28s} scatter/stated {q['min']:.3f}–{q['max']:.3f}"
                f"  coverage {q['coverage_95']:.4f} ({z:+.1f} SE)"
            )
    brief = gate | {"strict_2se": {k: v for k, v in gate["strict_2se"].items() if k != "families"}}
    print(json.dumps(_rounded({"thresholds": result["thresholds"], "gate": brief}), indent=1))


if __name__ == "__main__":
    main()
