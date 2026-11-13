"""First-order covariances on the bundled CEST example's real clusters (§2.4, §3).

Each cluster is fitted by OLS (PeakFit's VarPro). At that optimum, under the
processing-derived R, the script computes the first-order covariance of the
weighted estimator W = R_α⁻¹ (α = 1 OLS, 0.1 shrinkage, 0 GLS) and reports:
- GLS and shrinkage SDs relative to OLS: lineshape parameters, amplitudes, ratios;
- OLS total/conditional amplitude SD;
- the relative-profile covariance Σ_rel = cI + PLP of every peak: its largest
  correlated mode over c, and the profile σ over the conditional σ (§3.3);
- the within-plane amplitude correlation between the peaks of a cluster (§6a).
"""

import numpy as np
from cest_example import amplitudes, derivatives, fit_ols, load, shapes, spread, whitener

from peakfit.engine.algorithms.noise_correlation import cluster_correlation

ALPHAS = (1.0, 0.1, 0.0)


def first_order(phi, derivatives, owner, amps, r, w):
    """Lineshape SDs, conditional amplitude SDs and Cov(a_s, a_t) for weights w.

    phi (L, n) shapes, derivatives (P, n), owner (P,) the peak of each parameter,
    amps (L, S). All in noise units.
    """
    planes = amps.shape[1]
    m = np.linalg.solve(phi @ w @ phi.T, phi @ w)  # (L, n)
    w_perp = w - w @ phi.T @ m
    d = [(derivatives * amps[owner, s][:, None]).T for s in range(planes)]  # (n, P)
    h_inv = np.linalg.inv(sum(ds.T @ w_perp @ ds for ds in d))
    t = [h_inv @ ds.T @ w_perp for ds in d]  # (P, n)
    c_theta = sum(ts @ r @ ts.T for ts in t)
    mrm = m @ r @ m.T
    g = [m @ ds for ds in d]  # (L, P)

    def cov(s, u):
        c = -(m @ r @ t[s].T @ g[u].T) - (g[s] @ t[u] @ r @ m.T) + g[s] @ c_theta @ g[u].T
        return c + (mrm if s == u else 0)

    return np.sqrt(np.diag(c_theta)), np.sqrt(np.diag(mrm)), mrm, cov


def ratio_sd(cov, amps, peak, plane, ref=0):
    """SD of a_plane / a_ref for one peak, with the covariance between the planes."""
    a = amps[peak]
    gradient = np.array([1 / a[ref], -a[plane] / a[ref] ** 2])
    block = np.array(
        [
            [cov(plane, plane)[peak, peak], cov(plane, ref)[peak, peak]],
            [cov(ref, plane)[peak, peak], cov(ref, ref)[peak, peak]],
        ]
    )
    return float(np.sqrt(gradient @ block @ gradient))


def main() -> None:
    """Print the first-order comparisons of §2.4, §3.2–3.3 and §6a."""
    example = load()
    relative = {alpha: {"single": [], "multi": []} for alpha in ALPHAS[1:]}
    total_over_conditional, modes, profile_ratio, partner = [], {}, {}, []
    for cluster in example.clusters:
        opt = fit_ols(example, cluster)
        if opt._shared_derivatives:
            continue
        phi, amps = shapes(opt), amplitudes(opt)
        dmat, owner = derivatives(opt)
        r = cluster_correlation(example.model, cluster.grid_indices)
        out = {}
        for alpha in ALPHAS:
            wh = whitener(r, alpha)
            out[alpha] = first_order(phi, dmat, owner, amps, r, wh.T @ wh)
        peaks, planes = amps.shape
        kind = "single" if peaks == 1 else "multi"
        theta_ols, cond_ols, mrm_ols, cov_ols = out[1.0]
        for alpha in ALPHAS[1:]:
            theta, _, _, cov = out[alpha]
            amp0 = [np.sqrt(cov(0, 0)[k, k] / cov_ols(0, 0)[k, k]) for k in range(peaks)]
            ratio = [
                ratio_sd(cov, amps, k, planes // 2) / ratio_sd(cov_ols, amps, k, planes // 2)
                for k in range(peaks)
            ]
            relative[alpha][kind].append(
                (np.median(theta / theta_ols), np.median(amp0), np.median(ratio))
            )
        total_over_conditional += [np.sqrt(cov_ols(0, 0)[k, k]) / cond_ols[k] for k in range(peaks)]
        blocks = [[cov_ols(s, t) for t in range(planes)] for s in range(planes)]
        for k in range(peaks):
            sigma = np.array([[blocks[s][t][k, k] for t in range(planes)] for s in range(planes)])
            lineshape = sigma - np.eye(planes) * cond_ols[k] ** 2
            u = amps[k] / np.linalg.norm(amps[k])
            p = np.eye(planes) - np.outer(u, u)
            plp = p @ lineshape @ p
            modes.setdefault(kind, []).append(np.linalg.eigvalsh(plp).max() / cond_ols[k] ** 2)
            profile_ratio.setdefault(kind, []).append(
                np.sqrt(1 + np.clip(np.diag(plp), 0, None).max() / cond_ols[k] ** 2)
            )
        if peaks > 1:
            correlation = mrm_ols / np.outer(cond_ols, cond_ols)
            partner.append(np.abs(correlation[~np.eye(peaks, dtype=bool)]).max())

    print("GLS and shrinkage SDs over OLS, median per cluster: (10th–90th), max")
    for alpha, kinds in relative.items():
        for kind, rows in kinds.items():
            v = np.array(rows)
            print(
                f"  α={alpha:g} {kind:6s} n={len(v):3d}  lineshape {spread(v[:, 0])}"
                f" | amplitude, plane 1 {spread(v[:, 1])} | ratio to plane 1 {spread(v[:, 2])}"
            )
    print(f"OLS total/conditional amplitude SD, plane 1: {spread(total_over_conditional)}")
    for kind in ("single", "multi"):
        lam = np.array(modes[kind])
        print(
            f"{kind}: largest correlated mode of Σ_rel / c {spread(lam)};"
            f" > 0.1: {(lam > 0.1).sum()}, > 1: {(lam > 1).sum()} of {lam.size}"
        )
        print(f"{kind}: profile σ / conditional σ, worst plane {spread(profile_ratio[kind])}")
    rho = np.array(partner)
    print(
        f"within-plane |ρ| with a cluster partner, per multi-peak cluster: {spread(rho)};"
        f" > 0.3: {(rho > 0.3).sum()} of {rho.size}"
    )


if __name__ == "__main__":
    main()
