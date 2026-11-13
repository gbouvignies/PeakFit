"""Ratio errors over planes sharing one lineshape (§4): conditional, total and full covariance."""

import json
import sys

import numpy as np
from processed_lines import Chain, _complex_noise, lag_matrix, shapes
from scipy.optimize import least_squares

SNR = 30.0  # reference-plane height of line 1 / σ


def run(chain, index, theta, amps, rng, count, alpha, ref=0):  # noqa: PLR0915
    """amps: (L, S). Returns ratio diagnostics for every line and plane."""
    L, S = amps.shape
    n = index.size
    r = lag_matrix(chain.rho(), index)
    ra = (1 - alpha) * r + alpha * np.eye(n)
    lam, vec = np.linalg.eigh(ra)
    keep = lam > 1e-12 * lam.max()
    wh = (vec[:, keep] / np.sqrt(lam[keep])).T
    w = wh.T @ wh
    t = np.arange(chain.n)
    f0 = np.exp(2j * np.pi * theta[0] * t / chain.m - theta[1] * t / chain.n)
    s_fid = float(np.max(chain.process(f0))) / SNR / chain.noise_sd()
    sigma2 = (s_fid * chain.noise_sd()) ** 2
    phi0, _ = shapes(chain, index, theta)
    clean = phi0 @ amps  # (n, S)
    nth = theta.size

    def resid(x, y):
        phi, _ = shapes(chain, index, x[:nth])
        a = x[nth:].reshape(L, S)
        return (wh @ (phi @ a - y)).ravel(order="F")

    def jac_full(x):
        phi, dphi = shapes(chain, index, x[:nth])
        a = x[nth:].reshape(L, S)
        jt = np.zeros((n * S, nth))
        ja = np.zeros((n * S, L * S))
        for s in range(S):
            rows = slice(s * n, (s + 1) * n)
            for l in range(L):
                jt[rows, 2 * l] = a[l, s] * dphi[l][0]
                jt[rows, 2 * l + 1] = a[l, s] * dphi[l][1]
                ja[rows, l * S + s] = phi[:, l]
        return np.hstack([jt, ja])

    big_r = np.kron(np.eye(S), r)
    big_w = np.kron(np.eye(S), w)
    ratios, sd_cond, sd_tot, sd_full = [], [], [], []
    for _ in range(count):
        y = clean + chain.process(s_fid * _complex_noise(rng, (S, chain.n)))[:, index].T
        x0 = np.concatenate([theta, amps.ravel()])
        fit = least_squares(resid, x0, args=(y,), method="lm")
        x = fit.x
        j = jac_full(x)
        b = np.linalg.inv(j.T @ big_w @ j)
        cov = sigma2 * b @ j.T @ big_w @ big_r @ big_w @ j @ b
        ca = cov[nth:, nth:].reshape(L, S, L, S)
        a = x[nth:].reshape(L, S)
        # conditional: θ fixed at its estimate, per plane, weighted linear LS sandwich
        phi, _ = shapes(chain, index, x[:nth])
        bc = np.linalg.inv(phi.T @ w @ phi)
        cc = sigma2 * bc @ phi.T @ w @ r @ w @ phi @ bc  # (L, L), same for every plane
        rat = a / a[:, [ref]]
        g_cond, g_tot, g_full = (np.zeros((L, S)) for _ in range(3))
        for l in range(L):
            for s in range(S):
                q = rat[l, s]
                vs, v0 = ca[l, s, l, s], ca[l, ref, l, ref]
                c = ca[l, s, l, ref]
                d_s, d_0 = 1 / a[l, ref], -a[l, s] / a[l, ref] ** 2
                if s == ref:
                    g_cond[l, s] = g_tot[l, s] = g_full[l, s] = np.nan
                    continue
                g_cond[l, s] = np.sqrt(d_s**2 * cc[l, l] + d_0**2 * cc[l, l])
                g_tot[l, s] = np.sqrt(d_s**2 * vs + d_0**2 * v0)
                g_full[l, s] = np.sqrt(max(d_s**2 * vs + d_0**2 * v0 + 2 * d_s * d_0 * c, 0))
                del q
        ratios.append(rat)
        sd_cond.append(g_cond)
        sd_tot.append(g_tot)
        sd_full.append(g_full)
    ratios = np.array(ratios)
    truth = amps / amps[:, [ref]]
    out = {}
    for name, sds in (
        ("conditional, planes independent", sd_cond),
        ("total, planes independent", sd_tot),
        ("total, full covariance", sd_full),
    ):
        stated = np.array(sds)
        z = np.abs(ratios - truth) / stated
        scatter = ratios.std(axis=0)
        out[name] = {
            "scatter_over_stated (line × plane≠ref)": np.round(
                scatter / np.median(stated, axis=0), 2
            )[:, 1:].tolist(),
            "cov95": np.round((z < 1.96).mean(axis=0), 3)[:, 1:].tolist(),
        }
    out["ratio_sd (line × plane≠ref)"] = np.round(ratios.std(axis=0)[:, 1:], 4).tolist()
    return out


def main():
    rng = np.random.default_rng(7)
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    sb = Chain(92, 256, 0.4)
    profile = np.array([1.0, 0.8, 0.5, 0.3, 0.15])
    res = {}
    for alpha in (1.0, 0.0):
        res[f"single, alpha={alpha:g}"] = run(
            sb, np.arange(94, 107), np.array([100.0, 1.3]), profile[None], rng, count, alpha
        )
        pair = np.array([profile, 0.7 * np.array([1.0, 1.0, 0.95, 0.9, 0.9])])
        res[f"pair Δ=3, alpha={alpha:g}"] = run(
            sb, np.arange(91, 110), np.array([98.5, 1.3, 101.5, 1.3]), pair, rng, count, alpha
        )
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
