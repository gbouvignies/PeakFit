"""T7: known truth through NMRPipe linear-prediction processing (§2.3, §8).

Synthetic pseudo-2D data: planes × the LP axis, 47 acquired complex points extended
to the header's 92 by NMRPipe's ``LP -fb -ord 16``, then the example's ¹⁵N chain
(sine-bell² offset 0.4, zero fill to 256, FT). One isolated line and one overlapping
pair whose partners decay at different rates across planes. Each is fitted by OLS
and by GLS (header R; measured R from LP of signal-free noise), with the lineshape
shared by every plane and the amplitudes linear per plane, as PeakFit fits.

Three effects of LP act on the same late-FID points. The study separates them with
the identity LP(s + n) = s + [LP(s) − s] + [LP(s + n) − LP(s)], which needs only
the processing after LP to be linear:
- lineshape misfit: an unmodelled Gaussian decay, with and without LP;
- deterministic LP extrapolation error: the noise-free fit of LP(s);
- LP noise: the arm s + LP(n) carries the signal-free LP noise that noise-only
  columns measure; the arm s + LP(s+n) − LP(s) carries the signal-dependent LP noise
  of real data, without extrapolation error.

NMRPipe's outputs are committed fixtures (``nmrpipe_lp/``, written by
``make_lp_fixtures.py``), so the study runs without NMRPipe. A NumPy implementation
of ``LP -fb`` (``lp_extend``) gives a second LP arm for comparison. The LP order, 16,
is the one whose noise-power signature best matches the example's ¹⁵N axis (§2.2);
the example's processing script is not recorded.

    uv run python tools/profile_uncertainty/lp_known_truth.py
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from processed_lines import Chain, lag_matrix
from scipy.optimize import least_squares

__all__ = [
    "CASES",
    "CHAIN_FULL",
    "FULL",
    "TAU",
    "Chain",
    "decay_rate",
    "fit_planes",
    "lag_matrix",
    "lp",
    "lp_extend",
    "nmrpipe_fixture",
    "planes_fid",
    "realization_noise",
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
COUNT = 200  # noise realizations per case
SEED = 20261113
CHAIN_FULL = Chain(FULL, 256, 0.4)
CHAIN_ACQUIRED = Chain(ACQUIRED, 256, 0.4)
FIXTURES = Path(__file__).resolve().parent / "nmrpipe_lp"  # one .npz per case, under 500 kB


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

    def noise_sd(self) -> float:
        """FID noise SD per component giving SNR for line A, fully acquired."""
        unit = CHAIN_FULL.process(planes_fid(self.theta[:1], np.ones((1, 1)), FULL))
        return float(np.max(unit)) * self.scale[0] / SNR / CHAIN_FULL.noise_sd()

    def signal(self, gauss: float) -> np.ndarray:
        """Noise-free FIDs (planes, FULL)."""
        return planes_fid(self.theta, self.amplitudes, FULL, gauss)


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


def realization_noise(number: int) -> np.ndarray:
    """FID noise (COUNT · planes, FULL) of case ``number``; the fixtures use the same."""
    rng = np.random.default_rng([SEED, number])
    shape = (COUNT * TAU.size, FULL)
    return CASES[number].noise_sd() * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def lp(fids: np.ndarray) -> np.ndarray:
    """NumPy LP of every FID from ACQUIRED to FULL points."""
    return np.array([lp_extend(f, FULL - ACQUIRED) for f in fids[:, :ACQUIRED]])


def nmrpipe_fixture() -> dict[str, np.ndarray]:
    """NMRPipe's outputs over each case's window, in NumPy's frequency order.

    For case i: ``c{i}_lp_n`` is LP of its realization noise, ``c{i}_linear`` the
    noise-free signal processed without LP; for lineshape j, ``c{i}_g{j}_lp_sn`` is
    LP of signal plus noise and ``c{i}_g{j}_lp_s`` LP of the signal alone.
    ``power_ratio`` holds the noise-power signature for each LP order in ``orders``.
    """
    out = {}
    for path in sorted(FIXTURES.glob("*.npz")):
        with np.load(path) as data:
            out |= {key: data[key] for key in data.files}
    return out


def estimators(noise: np.ndarray | None, chain: Chain, index: np.ndarray, s: float) -> dict:
    """OLS and GLS under the header R; with LP noise, also GLS under the measured R.

    Each value is (whitening, assumed R, σ²). For an LP arm, σ² and the measured R
    come from LP of signal-free noise, as from noise-only regions of a spectrum.
    """
    header_r = lag_matrix(chain.rho(), index)
    if noise is None:
        sigma2 = (s * chain.noise_sd()) ** 2
    else:
        cov = np.cov(noise, rowvar=False)
        sigma2 = float(np.mean(np.diag(cov)))
    out = {
        "OLS": (np.eye(index.size), header_r, sigma2),
        "GLS header R": (whitening(header_r), header_r, sigma2),
    }
    if noise is not None:
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


ROW = "{:<24}{:<16}{:>4}{:>9}{:>7}{:>7}{:>9}{:>7}{:>7}{:>7}"
HEADER = ROW.format(
    "processing", "estimator", "ln", "I bias%", "b/SD", "I SD%", "R bias%", "R SD%", "b/SD", "χ²/E"
)


def _row(arm: str, est: str, l: int, values, chi: float | None) -> str:
    cells = [f"{v:.2f}" if np.isfinite(v) else "–" for v in values]
    return ROW.format(arm, est, "AB"[l], *cells, "" if chi is None else f"{chi:.2f}").rstrip()


def power_profile(spectra: np.ndarray) -> np.ndarray:
    """Mean time-domain power of processed spectra, normalized over the first 40 points."""
    p = (np.abs(np.fft.ifft(spectra, axis=1)) ** 2).mean(axis=0)
    return p / p[:40].sum()


def white_profile() -> np.ndarray:
    """The header's prediction: 92 acquired points of white noise."""
    rng = np.random.default_rng([SEED, 99])
    z = rng.standard_normal((4000, FULL)) + 1j * rng.standard_normal((4000, FULL))
    return power_profile(CHAIN_FULL.process(z))


def noise_power_signature(fixture: dict[str, np.ndarray]) -> None:
    """Each LP's time-domain noise power against the header's prediction (§2.2)."""
    points = (40, 45, 46, 47, 48, 56, 64, 80, 88)
    print(f"LP {ACQUIRED}→{FULL} points, forward-backward.")
    print("Noise power / header prediction (§2.2):")
    print("t                 " + "".join(f"{t:>6}" for t in points))
    for order, ratio in zip(fixture["orders"], fixture["power_ratio"], strict=True):
        print(f"NMRPipe, order {order:2d}  " + "".join(f"{ratio[t]:6.2f}" for t in points))
    rng = np.random.default_rng([SEED, 98])
    z = rng.standard_normal((2000, FULL)) + 1j * rng.standard_normal((2000, FULL))
    ratio = power_profile(CHAIN_FULL.process(lp(z))) / white_profile()
    print(f"NumPy, order {LP_ORDER}    " + "".join(f"{ratio[t]:6.2f}" for t in points))
    cest = (0.95, 0.94, 0.74, 0.51, 0.44, 0.21, 0.14, 0.27, 0.70)
    print("CEST ¹⁵N measured  " + "".join(f"{v:6.2f}" for v in cest))


def fit_arm(case, chain, y, noise, s, label, chi2=True) -> None:
    """Fit every realization of one arm with every estimator; print one row per line."""
    planes = TAU.size
    y = y.reshape(-1, planes, case.index.size)
    for est, (whiten, r, sigma2) in estimators(noise, chain, case.index, s).items():
        fits = [fit_planes(yk, chain, case.index, case.theta, whiten) for yk in y]
        chi = None
        if chi2:
            e = expected_chi2(chain, case.index, case.theta, case.amplitudes, whiten, r, sigma2)
            chi = float(np.mean([f.chi2 for f in fits])) / e
        for l, values in enumerate(summarize(case, fits)):
            if not chi2:  # one noise-free fit: bias only
                values = (values[0], np.nan, np.nan, values[3], np.nan, np.nan)  # noqa: PLW2901
            print(_row(label, est, l, values, chi))


def run_case(number: int, j: int, gauss: float, fixture: dict[str, np.ndarray]) -> None:
    """Noise-free and Monte Carlo fits of one case through every arm and estimator."""
    case = CASES[number]
    idx = case.index
    s = case.noise_sd()
    signal = case.signal(gauss)
    lp_s = fixture[f"c{number}_g{j}_lp_s"]
    lp_sn = fixture[f"c{number}_g{j}_lp_sn"]
    lp_n = fixture[f"c{number}_lp_n"]
    noise = realization_noise(number)

    print("noise-free (deterministic bias):")
    print(HEADER)
    fit_arm(case, CHAIN_FULL, CHAIN_FULL.process(signal)[:, idx], None, s, "acquired 92", False)
    y = CHAIN_ACQUIRED.process(signal[:, :ACQUIRED])[:, idx]
    fit_arm(case, CHAIN_ACQUIRED, y, None, s, "acquired 47, no LP", False)
    fit_arm(case, CHAIN_FULL, lp_s, lp_n, s, "LP 47→92", False)

    print(f"with noise ({COUNT} realizations, the same noise in every arm):")
    print(HEADER)
    data = np.tile(signal, (COUNT, 1)) + noise
    clean = np.tile(CHAIN_FULL.process(signal)[:, idx], (COUNT, 1))
    lp_s = np.tile(lp_s, (COUNT, 1))
    fit_arm(case, CHAIN_FULL, CHAIN_FULL.process(data)[:, idx], None, s, "acquired 92")
    y = CHAIN_ACQUIRED.process(data[:, :ACQUIRED])[:, idx]
    fit_arm(case, CHAIN_ACQUIRED, y, None, s, "acquired 47, no LP")
    fit_arm(case, CHAIN_FULL, lp_sn, lp_n, s, "LP 47→92")
    numpy_lp = CHAIN_FULL.process(lp(data))[:, idx]
    sigma = s * CHAIN_FULL.noise_sd()
    difference = np.abs(numpy_lp - lp_sn).max() / sigma
    print(f"NumPy LP against NMRPipe, max |difference| / σ: {difference:.1e}")
    fit_arm(case, CHAIN_FULL, clean + lp_n, lp_n, s, "s + LP(n)")
    fit_arm(case, CHAIN_FULL, clean + lp_sn - lp_s, lp_n, s, "s + LP(s+n) − LP(s)")


def main() -> None:
    """Print the LP signatures, then bias and scatter for every case and lineshape."""
    fixture = nmrpipe_fixture()
    noise_power_signature(fixture)
    print(
        f"\nLP order {int(fixture['order'])}; SNR {SNR:g} (line A, first plane, 92 acquired "
        f"points); planes τ = {TAU.tolist()}; window 19 points"
    )
    print("I: relative intensity I_k/I_0; R: downstream rate, free-scale exponential fit.")
    print("I bias%, b/SD: at the plane of largest |bias|; I SD%: median over planes.")
    print("bias: median − truth; SD: robust SD (1.4826 MAD); b/SD: bias over SD,")
    print("      Monte Carlo SE ≈ 1.25/√count.")
    print("χ²/E: mean χ² over its first-order expectation under the estimator's own R")
    print("      and σ² (LP: from LP of signal-free noise; OLS: the header R, as in §2.3).")
    print("Every LP arm is NMRPipe's; the NumPy LP is only compared with it.")
    for number, case in enumerate(CASES):
        for j, (shape, gauss) in enumerate(LINESHAPES.items()):
            print(f"\n=== {case.name}; {shape}; rates {case.rate.tolist()}")
            run_case(number, j, gauss, fixture)


if __name__ == "__main__":
    main()
