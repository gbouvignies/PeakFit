"""The factors' C_θ against the published lineshape covariance, on the CEST example.

Both are sandwiches under the processing-derived R, at the same optimum. They differ
only in the Jacobian of the lineshape parameters θ:
- **published** (``varpro._calculate_errors``): the exact Jacobian of the projected
  residual (Golub and Pereyra), which carries a term proportional to the residuals;
- **factors** (§3.1): the projected Jacobian [P⊥D_s] of the full-Jacobian sandwich
  over θ and all amplitudes (Kaufman), which does not.

For every cluster fitted as PeakFit fits it, the script reports:
- the SD ratio factors/published per free lineshape parameter;
- the range of that ratio over every linear combination of θ: the extreme generalized
  eigenvalues of (C_θ, C_published), square-rooted;
- what the difference does to the amplitudes: the marginal σ and the profile σ
  (§3.3) with C_θ, against the same with the published covariance in its place;
- the clusters where only one of the two is withheld.

Parameters at an active bound are excluded from both (they are not in θ).
"""

from dataclasses import replace

import numpy as np
from cest_example import load, spread
from scipy.linalg import eigh

from peakfit.engine.algorithms.covariance_factors import covariance_between_planes
from peakfit.engine.algorithms.evaluation import (
    AnalyticalModelEvaluation,
    evaluate_analytical_model,
)
from peakfit.engine.algorithms.noise_correlation import cluster_correlation
from peakfit.engine.algorithms.varpro import (
    VarProOptimizer,
    at_active_bound,
    fit_cluster,
    identifiable_covariance,
)
from peakfit.engine.domain.constraints import apply_constraints
from peakfit.engine.domain.params_scalar import Parameters


def per_peak_sigmas(covariance, conditional, amplitudes):
    """Marginal σ and profile σ, (L, S) each, from Cov (L, S, L, S)."""
    n_peaks, n_planes = amplitudes.shape
    marginal = np.empty((n_peaks, n_planes))
    profile = np.empty((n_peaks, n_planes))
    for peak in range(n_peaks):
        sigma = covariance[peak, :, peak, :]
        c = conditional[peak, peak]
        lineshape = sigma - c * np.eye(n_planes)
        u = amplitudes[peak] / np.linalg.norm(amplitudes[peak])
        p = np.eye(n_planes) - np.outer(u, u)
        marginal[peak] = np.sqrt(np.diag(sigma))
        profile[peak] = np.sqrt(np.diag(c * np.eye(n_planes) + p @ lineshape @ p))
    return marginal, profile


def rank(values: np.ndarray) -> np.ndarray:
    """Ranks, for a Spearman correlation."""
    return np.argsort(np.argsort(values))


def compare(example, cluster):
    """One cluster's comparison: a row, or a withholding outcome, and its bound count."""
    params = apply_constraints(Parameters.from_peaks(cluster.peaks), example.config.parameters)
    r = cluster_correlation(example.model, cluster.grid_indices)
    fitted = fit_cluster(params, cluster, example.noise, correlation=r).params
    names, x = fitted.get_vary_names(), fitted.get_vary_values()
    optimizer = VarProOptimizer(cluster, names, fitted, example.noise)
    published = identifiable_covariance(optimizer.compute_jacobian(x), r)
    evaluation = evaluate_analytical_model(cluster, fitted, example.noise, example.model)
    assert isinstance(evaluation, AnalyticalModelEvaluation)
    factors = evaluation.covariance_factors
    assert factors is not None
    free = ~at_active_bound(x, *fitted.get_vary_bounds())
    bound = int((~free).sum())
    published = published[np.ix_(free, free)]
    published_ok = bool(np.all(np.isfinite(published)))
    if not factors.available or not published_ok:
        outcome = (
            "both"
            if not (factors.available or published_ok)
            else ("published only" if published_ok else "factors only")
        )
        return outcome, bound
    theta = factors.parameter_covariance
    assert theta is not None
    sd_ratio = np.sqrt(np.diag(theta) / np.diag(published))
    low, high = np.sqrt(eigh(theta, published, eigvals_only=True)[[0, -1]])
    a = evaluation.amplitudes
    ours = covariance_between_planes(factors, a)
    swapped = covariance_between_planes(replace(factors, parameter_covariance=published), a)
    marginal, profile = per_peak_sigmas(ours, factors.conditional, a)
    marginal_p, profile_p = per_peak_sigmas(swapped, factors.conditional, a)
    free_names = [name for name, keep in zip(names, free, strict=True) if keep]
    worst = int(np.argmax(np.abs(sd_ratio - 1)))
    row = {
        "id": cluster.cluster_id,
        "peaks": len(cluster.peaks),
        "chi2r": evaluation.statistics.reduced_chi_squared,
        "residual": np.linalg.norm(evaluation.raw_residuals)
        / np.linalg.norm(evaluation.model_values),
        "worst": (free_names[worst], sd_ratio[worst]),
        "share": (marginal**2 / np.diag(factors.conditional)[:, None]).max(axis=1),
        "sd": sd_ratio,
        "range": (low, high),
        "gap": max(abs(low - 1), abs(high - 1)),
        "marginal": np.abs(marginal / marginal_p - 1).max(axis=1),
        "profile": np.abs(profile / profile_p - 1).max(axis=1),
    }
    return row, bound


def report(kind, found) -> None:
    """Print one group's summary."""
    sd = np.concatenate([row["sd"] for row in found])
    gap = np.array([row["gap"] for row in found])
    marginal = np.concatenate([row["marginal"] for row in found])
    profile = np.concatenate([row["profile"] for row in found])
    share = np.concatenate([row["share"] for row in found])
    flagged = marginal > 0.05
    print(f"{kind} n={len(found)} clusters, {sd.size} parameters, {marginal.size} peaks")
    print(f"  SD ratio per parameter: {spread(sd)}; min {sd.min():.3g}")
    print(f"  |SD ratio − 1|, worst linear combination per cluster: {spread(gap)}")
    print(f"  |Δ marginal σ| / marginal σ, worst plane per peak: {spread(marginal)}")
    print(f"  |Δ profile σ| / profile σ, worst plane per peak: {spread(profile)}")
    print(f"  peaks with |Δ marginal σ| > 5%: {flagged.sum()} of {marginal.size}")
    if flagged.any():
        values = ", ".join(f"{v:.3g}" for v in share[flagged])
        print(f"  marginal/conditional variance of those peaks (worst plane): {values}")
    print(
        "  largest differences (cluster, peaks, χ²ᵣ, ‖r‖/‖model‖, SD-ratio range over "
        "combinations, worst parameter):"
    )
    for row in sorted(found, key=lambda row: -row["gap"])[:5]:
        name, ratio = row["worst"]
        low, high = row["range"]
        print(
            f"    {row['id']:>4} {row['peaks']} {row['chi2r']:6.2f} {row['residual']:.4f} "
            f"{low:.3f}–{high:.3f}  {name} {ratio:.3f}"
        )
    for label in ("chi2r", "residual"):
        values = np.array([row[label] for row in found])
        rho = np.corrcoef(rank(values), rank(gap))[0, 1]
        print(f"  Spearman ρ({label}, worst |ratio − 1|): {rho:.2f}")


def main() -> None:
    """Fit every cluster and compare the two lineshape covariances."""
    example = load()
    rows = {"single": [], "multi": []}
    withheld = {"published only": 0, "factors only": 0, "both": 0}
    bound = 0
    for cluster in example.clusters:
        result, held = compare(example, cluster)
        bound += held
        if isinstance(result, str):
            withheld[result] += 1
        else:
            rows["single" if result["peaks"] == 1 else "multi"].append(result)
    print("C_θ (factors) against the published lineshape covariance, same optimum and R")
    print(f"parameters at an active bound, excluded from both: {bound}")
    print("withheld: " + ", ".join(f"{k} {v}" for k, v in withheld.items()))
    for kind, found in rows.items():
        if found:
            report(kind, found)


if __name__ == "__main__":
    main()
