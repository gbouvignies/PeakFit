"""Refit the bundled CEST example by GLS and shrinkage; compare with OLS (§2.5, §2.3).

Reports the residual excess at the OLS optimum, the whitened χ²ᵣ of each estimator
under the processing-derived R, and how far each moves the lineshape parameters, in
OLS sandwich SE. α = 1 refits OLS through the same code, as a check.
"""

import numpy as np
import scipy.optimize as so
from cest_example import data, fit_ols, load, residuals, shapes, spread, whitener

from peakfit.engine.algorithms.noise_correlation import cluster_correlation
from peakfit.engine.algorithms.varpro import identifiable_covariance

ALPHAS = (1.0, 0.1, 0.0)


def main() -> None:
    """Print the refit comparison."""
    example = load()
    rows = []
    for cluster in example.clusters:
        opt = fit_ols(example, cluster)
        params = opt.params_template
        x_ols = params.get_vary_values().copy()
        lower, upper = params.get_vary_bounds()
        r = cluster_correlation(example.model, cluster.grid_indices)
        y = data(opt)
        se = np.sqrt(np.diag(identifiable_covariance(opt.compute_jacobian(x_ols), r)))
        basis = shapes(opt).T
        hat = basis @ np.linalg.pinv(basis)
        expected = y.shape[1] * np.trace((np.eye(len(r)) - hat) @ r)
        excess = float(np.sum(residuals(opt) ** 2) / example.noise**2) / expected
        fits = {}
        for alpha in ALPHAS:
            wh = whitener(r, alpha)
            yw = wh @ y

            def whitened(x, wh=wh, yw=yw, opt=opt):
                opt.compute_residuals(x)
                phi = shapes(opt) @ wh.T
                a, *_ = np.linalg.lstsq(phi.T, yw, rcond=None)
                return ((yw - phi.T @ a) / example.noise).ravel()

            fit = so.least_squares(
                whitened, x_ols, bounds=(lower, upper), method="trf", x_scale="jac"
            )
            dof = fit.fun.size - len(x_ols) - shapes(opt).shape[0] * y.shape[1]
            fits[alpha] = (fit.x, float(fit.fun @ fit.fun / dof))
        shifts = {a: np.nanmax(np.abs(fits[a][0] - x_ols) / se) for a in ALPHAS}
        rows.append((excess, *(fits[a][1] for a in ALPHAS), *(shifts[a] for a in ALPHAS)))
    v = np.array(rows)
    print(f"clusters {len(v)}")
    print(f"check, α = 1 refit shift / SE: {spread(v[:, 4])}")
    print(f"OLS χ² / E[χ² | R]: {spread(v[:, 0])}")
    print(f"plain χ²ᵣ at the OLS optimum: {spread(v[:, 1])}")
    print(f"whitened χ²ᵣ, α = 0.1: {spread(v[:, 2])}")
    print(f"whitened χ²ᵣ, GLS: {spread(v[:, 3])}")
    print(f"largest shift / OLS SE, α = 0.1: {spread(v[:, 5])}")
    print(f"largest shift / OLS SE, GLS: {spread(v[:, 6])}")


if __name__ == "__main__":
    main()
