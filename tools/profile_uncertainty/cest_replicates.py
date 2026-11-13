"""CEST baseline planes as replicates, on the bundled example (§2.3, §5).

R is either the header's processing model ("hdr") or the ¹⁵N correlation measured on
noise-only columns ("meas", which accounts for the linear prediction). For OLS,
shrinkage α = 0.1 and GLS, the script reports:
- the whitened χ²ᵣ;
- the OLS residual excess, before and after removing one misfit shape per peak,
  proportional to its amplitude in each plane;
- on baseline planes (more than 400 Hz from the peak's own ¹⁵N offset, reference
  excluded), the robust SD of second differences of the relative intensities: its
  ratio to OLS's, and its ratio to the estimator's own stated conditional σ.
"""

import numpy as np
import scipy.optimize as so
from cest_example import (
    amplitudes,
    data,
    fit_ols,
    load,
    measured_15n_correlation,
    residuals,
    shapes,
    whitener,
)

from peakfit.engine.algorithms.noise_correlation import cluster_correlation

BASELINE_HZ = 400.0  # planes this far from the peak's own offset are baseline
MIN_BASELINE = 20
ESTIMATORS = (
    ("OLS", None, 1.0),
    ("α=0.1, header R", "hdr", 0.1),
    ("GLS, header R", "hdr", 0.0),
    ("α=0.1, measured R", "meas", 0.1),
    ("GLS, measured R", "meas", 0.0),
)


def robust_sd(v):
    """SD from the median absolute deviation."""
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def summary(values) -> str:
    """Median (10th–90th percentile)."""
    v = np.asarray(values, dtype=float)
    return f"{np.median(v):.2f} ({np.percentile(v, 10):.2f}–{np.percentile(v, 90):.2f})"


def main() -> None:  # noqa: PLR0915 - one study, reported in one place
    """Print the replicate comparison."""
    example = load()
    rho_15n = measured_15n_correlation()
    rho_1h = example.model.axes[1].correlation()
    carrier, observe = example.spectra.dic["FDF3CAR"], example.spectra.dic["FDF3OBS"]
    sizes = {c.cluster_id: len(c.peaks) for c in example.clusters}
    chi2 = {name: [] for name, *_ in ESTIMATORS}
    scatter = {name: {} for name, *_ in ESTIMATORS}
    calibration = {name: {"hdr": {}, "meas": {}} for name, *_ in ESTIMATORS}
    excess = {
        "hdr": [],
        "meas": [],
        "hdr, misfit shape removed": [],
        "meas, misfit shape removed": [],
    }

    for cluster in example.clusters:
        opt = fit_ols(example, cluster)
        params = opt.params_template
        x_ols = params.get_vary_values().copy()
        lower, upper = params.get_vary_bounds()
        y = data(opt)  # (n, S)
        planes = y.shape[1]
        g = cluster.grid_indices
        lag_15n = np.abs(g[0][:, None] - g[0][None, :])
        lag_1h = np.abs(g[1][:, None] - g[1][None, :])
        rs = {
            "hdr": cluster_correlation(example.model, g),
            "meas": rho_15n[lag_15n % rho_15n.size] * rho_1h[lag_1h],
        }
        phi, amps, res = shapes(opt), amplitudes(opt), residuals(opt)
        peaks = phi.shape[0]
        hat = phi.T @ np.linalg.pinv(phi.T)
        remainder = res - (res @ np.linalg.pinv(amps)) @ amps
        for key, r in rs.items():
            e = np.trace((np.eye(len(r)) - hat) @ r)
            excess[key].append(np.sum(res**2) / example.noise**2 / (planes * e))
            excess[f"{key}, misfit shape removed"].append(
                np.sum(remainder**2) / example.noise**2 / ((planes - peaks) * e)
            )
        baseline = {}
        for k, peak in enumerate(cluster.peaks):
            hz = (float(peak.positions[0]) - carrier) * observe
            far = np.abs(example.offsets - hz) > BASELINE_HZ
            baseline[k] = np.where((np.arange(planes) > 0) & far)[0]

        for name, key, alpha in ESTIMATORS:
            wh = whitener(rs[key] if key else np.eye(len(y)), alpha)
            yw = wh @ y

            def whitened(x, wh=wh, yw=yw, opt=opt):
                opt.compute_residuals(x)
                basis = shapes(opt) @ wh.T
                a, *_ = np.linalg.lstsq(basis.T, yw, rcond=None)
                return ((yw - basis.T @ a) / example.noise).ravel()

            x = x_ols
            if key is not None:
                x = so.least_squares(
                    whitened, x_ols, bounds=(lower, upper), method="trf", x_scale="jac"
                ).x
            f = whitened(x)
            chi2[name].append(float(f @ f / (f.size - len(x) - peaks * planes)))
            opt.compute_residuals(x)
            a = np.linalg.lstsq((shapes(opt) @ wh.T).T, yw, rcond=None)[0]  # (L, S)
            w = wh.T @ wh
            m = np.linalg.solve(shapes(opt) @ w @ shapes(opt).T, shapes(opt) @ w)
            for k in range(peaks):
                planes_k = baseline[k]
                if planes_k.size < MIN_BASELINE:
                    continue
                ab = a[k, planes_k]
                mean = np.median(ab)
                if mean <= 0:
                    continue
                d2 = (ab[:-2] - 2 * ab[1:-1] + ab[2:]) / np.sqrt(6)
                empirical = robust_sd(d2) / mean
                scatter[name][cluster.cluster_id, k] = empirical
                for rkey, r in rs.items():
                    stated = example.noise * np.sqrt((m @ r @ m.T)[k, k]) / mean
                    calibration[name][rkey][cluster.cluster_id, k] = empirical / stated

    print(f"clusters {len(example.clusters)}")
    for key, values in excess.items():
        print(f"OLS residual excess, {key}: {summary(values)}")
    ols = scatter["OLS"]
    print(f"OLS baseline relative scatter, median: {np.median(list(ols.values())):.4f}")
    for name, *_ in ESTIMATORS:
        keys = [k for k in scatter[name] if k in ols]
        multi = [k for k in keys if sizes[k[0]] > 1]
        ratio = [scatter[name][k] / ols[k] for k in keys]
        ratio_multi = [scatter[name][k] / ols[k] for k in multi]
        hdr, meas = calibration[name]["hdr"], calibration[name]["meas"]
        multi_ols = [hdr[k] for k in hdr if sizes[k[0]] > 1]
        print(
            f"{name:18s} whitened χ²ᵣ {summary(chi2[name])} | scatter / OLS {summary(ratio)}"
            f" (n={len(keys)}), multi-peak {summary(ratio_multi)} (n={len(multi)})"
            f" | empirical / stated: header R {summary(list(hdr.values()))},"
            f" measured R {summary(list(meas.values()))}, multi-peak (header R)"
            f" {summary(multi_ols)}"
        )


if __name__ == "__main__":
    main()
