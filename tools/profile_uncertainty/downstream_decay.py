"""Downstream test: a relaxation decay I_s = A·exp(−R·t_s) fitted to fitted amplitudes.

Cluster fits (OLS, shared lineshape, correlated noise) give amplitudes over S planes.
A downstream program fits (A, R) by weighted least squares with independent per-plane
σ_s. Which σ_s gives calibrated R errors?
- conditional σ;
- total σ (marginal, includes the shared-lineshape term);
- profile σ: conditional + lineshape term with its proportional (common-scale) part
  projected out;
- reference: GLS with the full total covariance (ideal, not what programs do).
"""

import sys

import numpy as np
from processed_lines import Chain, _complex_noise, lag_matrix, shapes
from scipy.optimize import least_squares

chain = Chain(92, 256, 0.4)
SNR = 30.0
TIMES = np.linspace(0, 1.5, 8)


def case(name, index, theta, rates, amp0, count, rng):
    L, S = len(rates), TIMES.size
    amps = np.array([a * np.exp(-k * TIMES) for a, k in zip(amp0, rates, strict=True)])
    n, P = index.size, theta.size
    r = lag_matrix(chain.rho(), index)
    t = np.arange(chain.n)
    f0 = np.exp(2j * np.pi * theta[0] * t / chain.m - theta[1] * t / chain.n)
    s_fid = float(np.max(chain.process(f0))) / SNR / chain.noise_sd()
    sigma2 = (s_fid * chain.noise_sd()) ** 2
    clean = shapes(chain, index, theta)[0] @ amps

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

    big_r = np.kron(np.eye(S), r)
    est = {k: [] for k in ("conditional", "total", "profile", "ideal")}
    ses = {k: [] for k in est}
    for _ in range(count):
        y = clean + chain.process(s_fid * _complex_noise(rng, (S, chain.n)))[:, index].T
        x0 = np.concatenate([theta, amps.ravel()])
        x = least_squares(resid, x0, args=(y,), method="lm").x
        J = jac(x)
        B = np.linalg.inv(J.T @ J)
        cov = (sigma2 * B @ J.T @ big_r @ J @ B)[P:, P:]
        phi, _ = shapes(chain, index, x[:P])
        M = np.linalg.solve(phi.T @ phi, phi.T)
        cond = sigma2 * (M @ r @ M.T)
        a = x[P:].reshape(L, S)
        for l in range(1):  # downstream fit of peak 0
            blk = cov[l * S : (l + 1) * S, l * S : (l + 1) * S]
            line = blk - np.eye(S) * cond[l, l]
            u = a[l] / np.linalg.norm(a[l])
            proj = np.eye(S) - np.outer(u, u)
            sig = {
                "conditional": np.full(S, np.sqrt(cond[l, l])),
                "total": np.sqrt(np.diag(blk)),
                "profile": np.sqrt(cond[l, l] + np.clip(np.diag(proj @ line @ proj), 0, None)),
            }
            for k, sd in sig.items():
                f = least_squares(
                    lambda p, y=a[l], sd=sd: (p[0] * np.exp(-p[1] * TIMES) - y) / sd,
                    [amp0[l], rates[l]],
                )
                jj = f.jac
                est[k].append(f.x[1])
                ses[k].append(np.sqrt(np.linalg.inv(jj.T @ jj)[1, 1]))
            w = np.linalg.inv(blk)
            ch = np.linalg.cholesky(w)
            f = least_squares(
                lambda p, y=a[l], ch=ch: ch.T @ (p[0] * np.exp(-p[1] * TIMES) - y),
                [amp0[l], rates[l]],
            )
            est["ideal"].append(f.x[1])
            ses["ideal"].append(np.sqrt(np.linalg.inv(f.jac.T @ f.jac)[1, 1]))
    print(f"\n## {name}  (R true {rates[0]})")
    for k, values in est.items():
        e, s = np.array(values), np.array(ses[k])
        z = np.abs(e - rates[0]) / s
        print(
            f"  {k:12s} R SD {e.std():.4f}  scatter/stated {e.std() / np.median(s):.2f}"
            f"  cov95 {(z < 1.96).mean():.3f}"
        )


def main():
    rng = np.random.default_rng(11)
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    case("isolated line", np.arange(94, 107), np.array([100.0, 1.3]), [1.0], [1.0], count, rng)
    case(
        "pair Δ=3, partner decays slower",
        np.arange(91, 110),
        np.array([98.5, 1.3, 101.5, 1.3]),
        [1.0, 0.2],
        [1.0, 0.8],
        count,
        rng,
    )
    case(
        "pair Δ=2, partner decays slower",
        np.arange(91, 110),
        np.array([99.0, 1.3, 101.0, 1.3]),
        [1.0, 0.2],
        [1.0, 0.8],
        count,
        rng,
    )


if __name__ == "__main__":
    main()
