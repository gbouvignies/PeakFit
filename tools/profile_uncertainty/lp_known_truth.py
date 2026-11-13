"""T7: known truth through linear-prediction processing (§2.3, §8).

Synthetic pseudo-2D data: planes × the LP axis, 47 acquired complex points extended
to the header's 92 by forward-backward LP, then the ¹⁵N chain of the example
(sine-bell² offset 0.4, zero fill to 256). One isolated line and one overlapping
pair whose partners decay at different rates across planes. Each is fitted by OLS
and by GLS (header R; measured R from LP of signal-free noise), with the lineshape
shared by every plane and the amplitudes linear per plane, as PeakFit fits.

Three effects of LP act on the same late-FID points. The study separates them with
the identity LP(s + n) = s + [LP(s) − s] + [LP(s + n) − LP(s)]:
- lineshape misfit: an unmodelled Gaussian decay, with and without LP;
- deterministic LP extrapolation error: the noise-free fit of LP(s);
- LP noise: the arm s + LP(n) carries the signal-free LP noise that noise-only
  columns measure; the arm s + LP(s+n) − LP(s) carries the signal-dependent LP noise
  of real data, without extrapolation error.

NMRPipe is not used: the LP is a NumPy implementation of NMRPipe's ``LP -fb``
(SVD least squares, root reflection, forward and backward filters averaged). Its
order, 16, reproduces the noise-power signature of the example's ¹⁵N axis (§2.2),
which the study prints first.

    uv run python tools/profile_uncertainty/lp_known_truth.py [realizations]
"""

import sys
from dataclasses import dataclass

import numpy as np
from processed_lines import Chain, lag_matrix
from scipy.optimize import least_squares

__all__ = [
    "Chain",
    "decay_rate",
    "fit_planes",
    "lag_matrix",
    "lp_extend",
    "planes_fid",
    "whitening",
]

FULL = 92  # points the header records; decays are per FULL points
LP_ORDER = 16


def _forward_coefficients(x: np.ndarray, order: int) -> np.ndarray:
    """Least-squares (SVD) forward filter: x[k] = Σ_j a_j x[k − 1 − j]."""
    rows = np.array([x[k : k + order][::-1] for k in range(x.size - order)])
    return np.linalg.lstsq(rows, x[order:], rcond=None)[0]


def _stable(roots: np.ndarray) -> np.ndarray:
    """Reflect growing roots inside the unit circle."""
    return np.where(np.abs(roots) > 1.0, 1.0 / np.conj(roots), roots)


def _filter(roots: np.ndarray) -> np.ndarray:
    return -np.poly(roots)[1:]


def lp_extend(x: np.ndarray, pred: int, order: int = LP_ORDER) -> np.ndarray:
    """Extend a complex FID by ``pred`` points with forward-backward LP.

    The forward filter and the backward filter (fitted on the reversed FID, its roots
    inverted into forward time) are each stabilized, averaged and stabilized again,
    as NMRPipe's ``LP -fb`` does.
    """
    forward = _stable(np.roots(np.r_[1.0, -_forward_coefficients(x, order)]))
    backward = _stable(1.0 / np.roots(np.r_[1.0, -_forward_coefficients(x[::-1], order)]))
    a = _filter(_stable(np.roots(np.r_[1.0, (_filter(forward) + _filter(backward)) / -2.0])))
    out = np.empty(x.size + pred, complex)
    out[: x.size] = x
    for k in range(x.size, out.size):
        out[k] = a @ out[k - order : k][::-1]
    return out


def planes_fid(theta: np.ndarray, amplitudes: np.ndarray, n: int, gauss: float = 0.0) -> np.ndarray:
    """FIDs (planes, n) of lines (position, decay) with per-plane amplitudes.

    ``gauss`` adds an unmodelled Gaussian decay, the lineshape misfit.
    """
    t = np.arange(n)
    lines = np.array(
        [np.exp(2j * np.pi * p * t / 256 - d * t / FULL) for p, d in theta.reshape(-1, 2)]
    )
    return (amplitudes @ lines) * np.exp(-((gauss * t / FULL) ** 2))


def whitening(r: np.ndarray) -> np.ndarray:
    """W^{1/2} with W the pseudo-inverse of R (eigenvalues below 1e-12·max dropped)."""
    lam, vec = np.linalg.eigh(r)
    keep = lam > 1e-12 * lam.max()
    return (vec[:, keep] / np.sqrt(lam[keep])).T


@dataclass(frozen=True)
class PlaneFit:
    """Shared lineshape, per-plane amplitudes and the weighted χ² of one fit."""

    theta: np.ndarray  # (lines, 2)
    amplitudes: np.ndarray  # (planes, lines)
    chi2: float


def _shapes(chain: Chain, index: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Processed unit lines over the window, (points, lines)."""
    return chain.process(planes_fid(theta, np.eye(theta.size // 2), chain.n))[:, index].T


def fit_planes(
    y: np.ndarray, chain: Chain, index: np.ndarray, theta0: np.ndarray, whiten: np.ndarray
) -> PlaneFit:
    """Variable projection: lineshape shared by every plane, amplitudes linear per plane.

    Minimizes Σ_k |S (Φ(θ) a_k − y_k)|² with S = ``whiten`` (OLS: S = I).
    """
    wy = y @ whiten.T  # (planes, whitened points)

    def amplitudes(theta):
        wphi = whiten @ _shapes(chain, index, theta)
        return wphi, np.linalg.lstsq(wphi, wy.T, rcond=None)[0].T

    def residuals(theta):
        wphi, a = amplitudes(theta)
        return (a @ wphi.T - wy).ravel()

    shape = np.shape(theta0)
    fit = least_squares(lambda th: residuals(th.reshape(shape)), np.ravel(theta0), x_scale="jac")
    theta = fit.x.reshape(shape)
    _, a = amplitudes(theta)
    return PlaneFit(theta, a, float(2.0 * fit.cost))


def decay_rate(tau: np.ndarray, profile: np.ndarray) -> float:
    """Rate of I₀ exp(−R τ) fitted to a profile with a free scale I₀, equal weights."""

    def residuals(rate):
        e = np.exp(-rate[0] * tau)
        return profile - (e @ profile) / (e @ e) * e

    start = -np.polyfit(tau, np.log(np.abs(profile) + 1e-12), 1)[0]
    return float(least_squares(residuals, [start], xtol=1e-14, ftol=1e-14).x[0])


# ---------------------------------------------------------------- the study

ACQUIRED = 47  # points actually acquired; LP extends them to FULL
SNR = 50.0  # height of line A in the first plane / σ, fully acquired
TAU = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8])  # plane delays
NOISE_DRAWS = 4000  # signal-free draws for σ² and the measured R
CHAIN_FULL = Chain(FULL, 256, 0.4)
CHAIN_ACQUIRED = Chain(ACQUIRED, 256, 0.4)


@dataclass(frozen=True)
class Case:
    """Lines (position, decay) on the LP axis and their decay rates across planes."""

    name: str
    theta: np.ndarray
    scale: np.ndarray  # amplitude in the first plane, per line
    rate: np.ndarray  # downstream rate, per line
    index: np.ndarray  # fitted window

    @property
    def amplitudes(self) -> np.ndarray:
        return self.scale * np.exp(-np.outer(TAU, self.rate))


CASES = (
    Case(
        "isolated",
        np.array([[100.0, 1.3]]),
        np.array([1.0]),
        np.array([1.5]),
        np.arange(91, 110),
    ),
    Case(
        "pair 3 points apart, differing profiles",
        np.array([[98.5, 1.3], [101.5, 1.3]]),
        np.array([1.0, 0.7]),
        np.array([1.0, 2.0]),
        np.arange(91, 110),
    ),
)
LINESHAPES = {"exact lineshape": 0.0, "unmodelled Gaussian decay": 1.0}


def lp(fids: np.ndarray) -> np.ndarray:
    """LP every FID from ACQUIRED to FULL points."""
    return np.array([lp_extend(f, FULL - ACQUIRED) for f in fids[:, :ACQUIRED]])


# Processing arms: (signal, FID noise), both over FULL points → (FID, chain).
# The LP arms split LP(s + n) = s + [LP(s) − s] + [LP(s + n) − LP(s)]: the
# extrapolation error and the LP noise. LP(n) is the noise of signal-free columns.
ARMS = {
    "acquired 92": lambda s, n: (s + n, CHAIN_FULL),
    "acquired 47, no LP": lambda s, n: ((s + n)[:, :ACQUIRED], CHAIN_ACQUIRED),
    "LP 47→92": lambda s, n: (lp(s + n), CHAIN_FULL),
    "s + LP(n)": lambda s, n: (s + lp(n), CHAIN_FULL),
    "s + LP(s+n) − LP(s)": lambda s, n: (s + lp(s + n) - lp(s), CHAIN_FULL),
}
LP_ARMS = ("LP 47→92", "s + LP(n)", "s + LP(s+n) − LP(s)")


def _complex(rng, shape) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def estimators(arm: str, chain: Chain, index: np.ndarray, s: float, rng) -> dict:
    """OLS and GLS under the header R; for the LP arms, also GLS under the measured R.

    Each value is (whitening, assumed R, σ²). σ² and the measured R come from
    signal-free noise processed like the arm, as from noise-only regions of a spectrum.
    """
    z = s * _complex(rng, (NOISE_DRAWS, FULL))
    noise = chain.process(lp(z) if arm in LP_ARMS else z[:, : chain.n])[:, index]
    cov = np.cov(noise, rowvar=False)
    sigma2 = float(np.mean(np.diag(cov)))
    header_r = lag_matrix(chain.rho(), index)
    out = {
        "OLS": (np.eye(index.size), header_r, sigma2),
        "GLS header R": (whitening(header_r), header_r, sigma2),
    }
    if arm in LP_ARMS:
        out["GLS measured R"] = (whitening(cov / sigma2), cov / sigma2, sigma2)
    return out


def expected_chi2(chain, index, theta, amplitudes, whiten, r, sigma2) -> float:
    """First-order E[χ²] over all planes for noise of variance σ² and correlation r."""
    planes, lines = amplitudes.shape
    phi = _shapes(chain, index, theta)
    n, p = index.size, theta.size
    jac = np.zeros((planes * n, p + planes * lines))
    for j in range(p):
        step = np.zeros(p)
        step[j] = 1e-6
        dphi = (_shapes(chain, index, theta + step.reshape(theta.shape)) - phi) / 1e-6
        jac[:, j] = (amplitudes @ dphi.T).ravel()
    for k in range(planes):
        jac[k * n : (k + 1) * n, p + k * lines : p + (k + 1) * lines] = phi
    s_full = np.kron(np.eye(planes), whiten)
    sj = s_full @ jac
    maker = np.eye(sj.shape[0]) - sj @ np.linalg.pinv(sj)
    cov = s_full @ np.kron(np.eye(planes), r) @ s_full.T
    return float(sigma2 * np.trace(maker @ cov @ maker.T))


def _robust_sd(x: np.ndarray) -> np.ndarray:
    return 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0)


def summarize(case: Case, fits: list[PlaneFit]) -> list[tuple[float, ...]]:
    """Per line: relative intensity (I_k/I_0) and downstream rate, bias and scatter.

    Bias is the median minus the truth and SD is the robust SD (1.4826 MAD), so the
    rare GLS fits that fail do not dominate. Relative intensity: bias (%) and bias/SD
    at the plane of largest |bias|, and the median SD (%) over planes. Rate: bias (%),
    SD (%) and bias/SD.
    """
    a = np.array([f.amplitudes for f in fits])  # (draws, planes, lines)
    rel = a[:, 1:, :] / a[:, :1, :]
    true_rel = np.exp(-np.outer(TAU[1:], case.rate))
    out = []
    for l in range(case.rate.size):
        bias = np.median(rel[:, :, l], axis=0) / true_rel[:, l] - 1.0
        sd = _robust_sd(rel[:, :, l]) / true_rel[:, l]
        k = int(np.argmax(np.abs(bias)))
        rates = np.array([decay_rate(TAU, f.amplitudes[:, l]) for f in fits])
        r_bias = np.median(rates) / case.rate[l] - 1.0
        r_sd = float(_robust_sd(rates)) / case.rate[l]
        out.append(
            (
                100 * bias[k],
                bias[k] / sd[k] if sd[k] > 0 else np.nan,
                100 * float(np.median(sd)),
                100 * r_bias,
                100 * r_sd,
                r_bias / r_sd if r_sd > 0 else np.nan,
            )
        )
    return out


ROW = "{:<22}{:<16}{:>4}{:>9}{:>7}{:>7}{:>9}{:>7}{:>7}{:>7}"
HEADER = ROW.format(
    "processing", "estimator", "ln", "I bias%", "b/SD", "I SD%", "R bias%", "R SD%", "b/SD", "χ²/E"
)


def _row(arm: str, est: str, l: int, values, chi: float | None) -> str:
    cells = [f"{v:.2f}" if np.isfinite(v) else "–" for v in values]
    return ROW.format(arm, est, "AB"[l], *cells, "" if chi is None else f"{chi:.2f}").rstrip()


def noise_power_signature(rng) -> None:
    """The LP's time-domain noise power against the header's prediction, as in §2.2."""
    chain = Chain(FULL, 256, 0.4, p1=180.0)  # the CEST ¹⁵N header

    def profile(fids):
        p = (np.abs(np.fft.ifft(chain.process(fids), axis=1)) ** 2).mean(axis=0)
        return p / p[:40].sum()

    z = _complex(rng, (NOISE_DRAWS, FULL))
    ratio = profile(lp(z)) / profile(z)
    points = (40, 45, 46, 47, 48, 56, 64, 80, 88)
    cest = (0.95, 0.94, 0.74, 0.51, 0.44, 0.21, 0.14, 0.27, 0.70)
    print(f"LP {ACQUIRED}→{FULL} points, order {LP_ORDER}, forward-backward.")
    print("Noise power / header prediction (§2.2):")
    print("t          " + "".join(f"{t:>6}" for t in points))
    print("this LP    " + "".join(f"{ratio[t]:6.2f}" for t in points))
    print("CEST ¹⁵N   " + "".join(f"{v:6.2f}" for v in cest))


def run_case(case: Case, gauss: float, count: int, rng) -> None:
    """Noise-free and Monte Carlo fits of one case through every arm and estimator."""
    truth = case.amplitudes
    height = float(np.max(CHAIN_FULL.process(planes_fid(case.theta[:1], np.ones((1, 1)), FULL))))
    s = height * case.scale[0] / SNR / CHAIN_FULL.noise_sd()
    signal = planes_fid(case.theta, truth, FULL, gauss)
    planes = TAU.size
    print("noise-free (deterministic bias):")
    print(HEADER)
    for arm in ("acquired 92", "acquired 47, no LP", "LP 47→92"):
        fid, chain = ARMS[arm](signal, np.zeros_like(signal))
        y = chain.process(fid)[:, case.index]
        for est, (whiten, _, _) in estimators(arm, chain, case.index, s, rng).items():
            fit = fit_planes(y, chain, case.index, case.theta, whiten)
            for l, values in enumerate(summarize(case, [fit])):
                print(
                    _row(
                        arm,
                        est,
                        l,
                        values[:1] + (np.nan, np.nan) + values[3:4] + (np.nan,) * 2,
                        None,
                    )
                )
    print(f"with noise ({count} realizations, the same noise in every arm):")
    print(HEADER)
    noise = s * _complex(rng, (count * planes, FULL))
    stacked = np.tile(signal, (count, 1))
    for arm, make in ARMS.items():
        fid, chain = make(stacked, noise)
        y = chain.process(fid)[:, case.index].reshape(count, planes, -1)
        for est, (whiten, r, sigma2) in estimators(arm, chain, case.index, s, rng).items():
            fits = [fit_planes(yk, chain, case.index, case.theta, whiten) for yk in y]
            chi = np.mean([f.chi2 for f in fits]) / expected_chi2(
                chain, case.index, case.theta, truth, whiten, r, sigma2
            )
            for l, values in enumerate(summarize(case, fits)):
                print(_row(arm, est, l, values, chi))


def main() -> None:
    """Print the LP signature, then bias and scatter for every case and lineshape."""
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(20261113)
    noise_power_signature(rng)
    print(
        f"\nSNR {SNR:g} (line A, first plane, 92 acquired points); "
        f"planes τ = {TAU.tolist()}; window 19 points"
    )
    print("I: relative intensity I_k/I_0; R: downstream rate, free-scale exponential fit.")
    print("I bias%, b/SD: at the plane of largest |bias|; I SD%: median over planes.")
    print("bias: median − truth; SD: robust SD (1.4826 MAD); b/SD: bias over SD,")
    print("      Monte Carlo SE ≈ 1.25/√count.")
    print("χ²/E: mean χ² over its first-order expectation under the estimator's own R")
    print("      and σ² from signal-free noise (OLS: the header R, as in §2.3).")
    for case in CASES:
        for shape, gauss in LINESHAPES.items():
            print(f"\n=== {case.name}; {shape}; rates {case.rate.tolist()}")
            run_case(case, gauss, count, rng)


if __name__ == "__main__":
    main()
