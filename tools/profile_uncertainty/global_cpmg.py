"""ChemEx-like global CPMG fit of an overlapping pair, from PeakFit-like cluster fits.

Cluster: two processed lines sharing lineshape parameters across S planes (OLS,
correlated noise). Planes: one reference (no CPMG delay) and ν_CPMG values.
Downstream (as ChemEx): I_l(ν) = A_l·exp(−T·R2eff_l(ν)), free A_l per profile,
fast-exchange Luz–Meiboom R2eff, kex shared by both peaks. Fitted with:
  marginal σ / profile σ / conditional σ as independent errors (what ChemEx does),
  per-peak covariance (block diagonal), full joint covariance (both peaks).
"""

import sys

import numpy as np
from processed_lines import Chain, _complex_noise, lag_matrix, shapes
from scipy.optimize import least_squares

chain = Chain(92, 256, 0.4)
T = 0.04
NU = np.array([0.0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000])
KEX = 1500.0
R20 = (10.0, 12.0)
PHI = (4.0e4, 1.0e4)


def r2eff(r20, phi, kex, nu):
    nu = np.maximum(nu, 1e-9)
    x = kex / (4 * nu)
    return r20 + phi / kex * (1 - np.tanh(x) / x)


def profile(r20, phi, kex):
    out = np.exp(-T * r2eff(r20, phi, kex, NU))
    out[NU == 0] = 1.0  # reference plane: no relaxation delay
    return out


def downstream(a, sig_or_cov, mode):
    """Fit (A1, A2, r20_1, r20_2, phi1, phi2, kex) to amplitudes a (2, S)."""
    if mode == "indep":
        w = [np.diag(1 / s) for s in sig_or_cov]

        def res(p):
            m = [p[0] * profile(p[2], p[4], p[6]), p[1] * profile(p[3], p[5], p[6])]
            return np.concatenate([w[l] @ (m[l] - a[l]) for l in range(2)])
    else:
        ch = np.linalg.cholesky(np.linalg.inv(sig_or_cov)).T  # whitening

        def res(p):
            m = np.concatenate([p[0] * profile(p[2], p[4], p[6]), p[1] * profile(p[3], p[5], p[6])])
            return ch @ (m - a.ravel())

    p0 = np.array([1.0, 0.8, *R20, *PHI, KEX])
    f = least_squares(res, p0, x_scale=np.abs(p0), method="trf")
    cov = np.linalg.pinv(f.jac.T @ f.jac)
    return f.x, np.sqrt(np.diag(cov))


def run(name, index, theta, snr, count, rng):  # noqa: PLR0915
    amps = np.array(
        [
            1.0 * profile(R20[0], PHI[0], KEX),
            0.8 * profile(R20[1], PHI[1], KEX),
        ]
    )
    L, S = amps.shape
    n, P = index.size, theta.size
    r = lag_matrix(chain.rho(), index)
    t = np.arange(chain.n)
    f0 = np.exp(2j * np.pi * theta[0] * t / chain.m - theta[1] * t / chain.n)
    s_fid = float(np.max(chain.process(f0))) / snr / chain.noise_sd()
    sigma2 = (s_fid * chain.noise_sd()) ** 2
    clean = shapes(chain, index, theta)[0] @ amps
    big_r = np.kron(np.eye(S), r)

    def resid(x, y):
        phi, _ = shapes(chain, index, x[:P])
        return (phi @ x[P:].reshape(L, S) - y).ravel(order="F")

    def jac(x):
        phi, dphi = shapes(chain, index, x[:P])
        a = x[P:].reshape(L, S)
        J = np.zeros((n * S, P + L * S))
        for s in range(S):
            rows = slice(s * n, (s + 1) * n)
            for l in range(L):
                J[rows, 2 * l] = a[l, s] * dphi[l][0]
                J[rows, 2 * l + 1] = a[l, s] * dphi[l][1]
                J[rows, P + l * S + s] = phi[:, l]
        return J

    keys = ["marginal σ", "profile σ", "conditional σ", "per-peak cov", "joint cov"]
    est = {k: [] for k in keys}
    sds = {k: [] for k in keys}
    cross = []
    for _ in range(count):
        y = clean + chain.process(s_fid * _complex_noise(rng, (S, chain.n)))[:, index].T
        x = least_squares(resid, np.concatenate([theta, amps.ravel()]), args=(y,), method="lm").x
        J = jac(x)
        B = np.linalg.inv(J.T @ J)
        cov = (sigma2 * B @ J.T @ big_r @ J @ B)[P:, P:]  # (L·S, L·S), peak-major
        a = x[P:].reshape(L, S)
        phi, _ = shapes(chain, index, x[:P])
        M = np.linalg.solve(phi.T @ phi, phi.T)
        cond = sigma2 * (M @ r @ M.T)
        marg, prof = [], []
        for l in range(L):
            blk = cov[l * S : (l + 1) * S, l * S : (l + 1) * S]
            line = blk - cond[l, l] * np.eye(S)
            u = a[l] / np.linalg.norm(a[l])
            pr = np.eye(S) - np.outer(u, u)
            marg.append(np.sqrt(np.diag(blk)))
            prof.append(np.sqrt(cond[l, l] + np.clip(np.diag(pr @ line @ pr), 0, None)))
        cross.append(cond[0, 1] / np.sqrt(cond[0, 0] * cond[1, 1]))
        block = cov.copy()
        block[:S, S:] = 0
        block[S:, :S] = 0
        inputs = {
            "marginal σ": ("indep", marg),
            "profile σ": ("indep", prof),
            "conditional σ": ("indep", [np.full(S, np.sqrt(cond[l, l])) for l in range(L)]),
            "per-peak cov": ("cov", block),
            "joint cov": ("cov", cov),
        }
        for k, (mode, s) in inputs.items():
            p, e = downstream(a, s, mode)
            est[k].append(p)
            sds[k].append(e)
    rho = np.median(cross)
    print(f"\n## {name}  (within-plane conditional amplitude correlation: {rho:+.2f})")
    best = np.array(est["joint cov"]).std(axis=0)
    for k in keys:
        e, s = np.array(est[k]), np.array(sds[k])
        out = []
        for j, label in ((6, "kex"), (4, "Φ1"), (5, "Φ2")):
            truth = KEX if j == 6 else PHI[j - 4]
            z = np.abs(e[:, j] - truth) / s[:, j]
            sd = e[:, j].std()
            out.append(
                f"{label}: SD/best {sd / best[j]:.2f} scat/stated {sd / np.median(s[:, j]):.2f} "
                f"cov95 {(z < 1.96).mean():.2f} bias/SD {(np.median(e[:, j]) - truth) / sd:+.2f}"
            )
        print(f"  {k:14s} " + " | ".join(out))


def main():
    rng = np.random.default_rng(5)
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    run(
        "isolated-like (Δ=6 pts)",
        np.arange(88, 113),
        np.array([97.0, 1.3, 103.0, 1.3]),
        150,
        count,
        rng,
    )
    run("pair Δ=3 pts", np.arange(91, 110), np.array([98.5, 1.3, 101.5, 1.3]), 150, count, rng)
    run("pair Δ=2 pts", np.arange(91, 110), np.array([99.0, 1.3, 101.0, 1.3]), 150, count, rng)


if __name__ == "__main__":
    main()
