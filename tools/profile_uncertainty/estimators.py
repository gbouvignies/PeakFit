"""OLS+sandwich vs GLS vs shrinkage-weighted LS on processed 1D lines (scratch study).

Reuses the processing chain of tools/validate_correlated_noise.py. A line set is a
sum of exponentially decaying complex lines, apodized, zero filled, FT'd, real part.
Estimator family: minimize (y-m)ᵀ W (y-m) with W = R_α⁻¹, R_α = (1-α)R + αI.
α = 1: OLS. α = 0: GLS. Stated covariance: general sandwich
B⁻¹ JᵀWRWJ B⁻¹ (B = JᵀWJ), which equals B⁻¹ for GLS.
"""

import json
import sys
from dataclasses import dataclass

import numpy as np
from processed_lines import Chain, _complex_noise, lag_matrix
from scipy.optimize import least_squares

SNR = 30.0


@dataclass
class Lines:
    """Lines on one processed axis, fitted over a window; optional misfit or wrong R."""

    chain: Chain
    truth: np.ndarray  # per line (amp, pos, decay); flattened
    index: np.ndarray
    gauss: float = 0.0  # unmodelled Gaussian decay (misfit), per acquisition time
    lb_true: float = 0.0  # unrecorded extra exponential window on data+noise
    model_chain: Chain | None = None

    def fid(self, theta, gauss=0.0):
        """Time-domain signal for (amplitude, position, decay) per line."""
        t = np.arange(self.chain.n)
        out = np.zeros(self.chain.n, complex)
        for a, p, d in theta.reshape(-1, 3):
            out += a * np.exp(2j * np.pi * p * t / self.chain.m - d * t / self.chain.n)
        return out * np.exp(-((gauss * t / self.chain.n) ** 2))

    def window_true(self):
        """Unrecorded exponential window applied to data and noise."""
        t = np.arange(self.chain.n)
        return np.exp(-self.lb_true * t / self.chain.n)

    def data_clean(self):
        """Noise-free processed data over the window."""
        return self.chain.process(self.fid(self.truth, self.gauss) * self.window_true())[self.index]

    def mchain(self):
        """The processing the model assumes."""
        return self.model_chain or self.chain

    def model(self, theta):
        """Processed model over the window."""
        return self.mchain().process(self.fid(theta))[self.index]

    def jac(self, theta):
        """Exact derivatives of the model, (points, parameters)."""
        t = np.arange(self.chain.n)
        cols = []
        for a, p, d in theta.reshape(-1, 3):
            f = a * np.exp(2j * np.pi * p * t / self.chain.m - d * t / self.chain.n)
            cols += [f / a, f * 2j * np.pi * t / self.chain.m, -f * t / self.chain.n]
        return np.stack([self.mchain().process(c)[self.index] for c in cols], axis=1)

    def noise(self, rng, count):
        """Processed noise realizations over the window."""
        s = self.noise_scale()
        z = s * _complex_noise(rng, (count, self.chain.n)) * self.window_true()
        return self.chain.process(z)[:, self.index]

    def noise_scale(self):
        """FID noise SD per component giving the target SNR."""
        h = float(np.max(self.chain.process(self.fid(self.truth[:3]))))
        return h / SNR / self.chain.noise_sd()

    def true_cov(self):
        """Exact covariance of the processed noise (incl. any unrecorded window)."""
        s = self.noise_scale()
        n = self.chain.n
        eye = np.eye(n) * self.window_true()
        # covariance of real(processed) for complex white noise of SD s per component
        x = np.zeros((n, self.chain.m), complex)
        x[:, :n] = eye * self.chain.window
        phase = np.exp(-1j * np.deg2rad(self.chain.p1) * np.arange(self.chain.m) / self.chain.m)
        f = (np.fft.fft(x, axis=-1) * phase)[:, self.index]  # rows: time impulses
        # real part of sum_t f_t z_t with z_t = s(u+iv): cov = s² (Re fᵀRe f + Im fᵀIm f)
        return s**2 * (f.real.T @ f.real + f.imag.T @ f.imag)


def assumed_r(lines: Lines, chain=None):
    c = chain or lines.mchain()
    return lag_matrix(c.rho(), lines.index)


def study(lines: Lines, rng, count, alphas=(1.0, 0.1, 0.01, 0.0), r_assumed=None):
    r = assumed_r(lines) if r_assumed is None else r_assumed
    rtrue = lines.true_cov()
    sigma2 = float(np.mean(np.diag(rtrue)))  # σ estimated from the spectrum
    clean = lines.data_clean()
    noises = lines.noise(rng, count)
    out = {
        "cond_R": float(np.linalg.cond(r)),
        "rank_R_1e-12": int((np.linalg.eigvalsh(r) > 1e-12).sum()),
        "n_points": int(lines.index.size),
    }
    est = {}
    for alpha in alphas:
        ra = (1 - alpha) * r + alpha * np.eye(r.shape[0])
        lam, vec = np.linalg.eigh(ra)
        keep = lam > 1e-12 * lam.max()  # pseudo-inverse where R is numerically singular
        whiten = (vec[:, keep] / np.sqrt(lam[keep])).T
        w = whiten.T @ whiten
        xs, sds = [], []
        for nz in noises:
            y = clean + nz
            fit = least_squares(
                lambda th, y=y, whiten=whiten: whiten @ (lines.model(th) - y),
                lines.truth,
                method="lm",
            )
            j = lines.jac(fit.x)
            b = np.linalg.inv(j.T @ w @ j)
            cov = sigma2 * b @ j.T @ w @ r @ w @ j @ b  # stated, under assumed R
            xs.append(fit.x)
            sds.append(np.sqrt(np.diag(cov)))
        xs, sds = np.array(xs), np.array(sds)
        # exact first-order SD under the true noise at the truth
        j0 = lines.jac(lines.truth)
        b0 = np.linalg.inv(j0.T @ w @ j0)
        exact = np.sqrt(np.diag(b0 @ j0.T @ w @ rtrue @ w @ j0 @ b0))
        est[alpha] = (xs, sds, exact, float(np.abs(whiten).sum(axis=1).max()))
    ols_sd = est[1.0][0].std(axis=0)
    for alpha, (xs, sds, exact, gain) in est.items():
        sd = xs.std(axis=0)
        bias = xs.mean(axis=0) - lines.truth
        rmse = np.sqrt(((xs - lines.truth) ** 2).mean(axis=0))
        z = np.abs(xs - lines.truth) / sds
        out[f"alpha={alpha:g}"] = {
            "sd_over_ols": np.round(sd / ols_sd, 3).tolist(),
            "rmse_over_ols_rmse": np.round(
                rmse / np.sqrt(((est[1.0][0] - lines.truth) ** 2).mean(axis=0)), 3
            ).tolist(),
            "first_order_sd_over_ols_first_order": np.round(exact / est[1.0][2], 3).tolist(),
            "bias_over_sd": np.round(bias / sd, 2).tolist(),
            "scatter_over_stated": np.round(sd / np.median(sds, axis=0), 3).tolist(),
            "cov68": np.round((z < 1).mean(axis=0), 3).tolist(),
            "cov95": np.round((z < 1.96).mean(axis=0), 3).tolist(),
            "whitening_max_abs_rowsum": round(gain, 1),
        }
    return out


def window(centre, half):
    c = round(centre)
    return np.arange(c - half, c + half + 1)


def main():
    rng = np.random.default_rng(20261112)
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    sb = Chain(92, 256, 0.4)  # sine-bell² off 0.4, zf ×2.8 (CEST 15N-like)
    sb5 = Chain(92, 256, 0.5)
    zf4 = Chain(92, 512, 0.5)
    res = {}
    res["single narrow, sb0.4"] = study(
        Lines(sb, np.array([1.0, 100.0, 1.3]), window(100, 6)), rng, count
    )
    res["single broad, sb0.4"] = study(
        Lines(sb, np.array([1.0, 100.0, 5.2]), window(100, 6)), rng, count
    )
    res["single narrow, sb0.4, wide window 25"] = study(
        Lines(sb, np.array([1.0, 100.0, 1.3]), window(100, 12)), rng, count
    )
    res["single narrow, sb0.5 zf×5.6"] = study(
        Lines(zf4, np.array([1.0, 200.0, 1.3]), window(200, 12)), rng, count
    )
    res["pair Δ=3 pts, sb0.4"] = study(
        Lines(sb, np.array([1.0, 98.5, 1.3, 0.7, 101.5, 1.3]), window(100, 9)), rng, count
    )
    res["pair Δ=1.5 pts, sb0.4"] = study(
        Lines(sb, np.array([1.0, 99.25, 1.3, 0.7, 100.75, 1.3]), window(100, 9)), rng, count
    )
    res["misfit: unmodelled Gaussian decay"] = study(
        Lines(sb, np.array([1.0, 100.0, 1.3]), window(100, 6), gauss=1.0), rng, count
    )
    res["misfit: model sb0.5, data sb0.4"] = study(
        Lines(sb, np.array([1.0, 100.0, 1.3]), window(100, 6), model_chain=sb5), rng, count
    )
    res["R wrong: unrecorded EM (lb·aq=2)"] = study(
        Lines(sb, np.array([1.0, 100.0, 1.3]), window(100, 6), lb_true=2.0), rng, count
    )
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
