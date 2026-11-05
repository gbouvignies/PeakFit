"""Evidence for docs/correlated-noise.md: validations C1–C5 and the CEST comparison.

Everything synthetic is self-contained NumPy and SciPy with fixed seeds. A line is a
complex FID processed exactly as NMRPipe would: sine-bell apodization, first-point
scaling, zero fill and Fourier transform, keeping the real part. Its noise is
complex white noise in the FID, processed the same way. A fit uses a fixed 13-point
window around the line.

- **C1:** the processing-derived noise correlation against Monte Carlo, and the SD
  inflation of a line's amplitude.
- **C2:** coverage of σ-only and sandwich errors. It separates the *conditional*
  amplitude error (lineshape held at the truth) from the *total* errors of a joint
  fit of amplitude, position and R2.
- **C3:** twofold zero fill without apodization: the small, nonzero inflation it
  predicts, tested by Monte Carlo. (The exact regression test of today's σ-only path
  is R = I, in the implementation's unit tests.)
- **C4:** the χ² expectation and variance under correlated noise.
- **C5 (preliminary):** coverage of the exact grid posterior under the independence
  likelihood, and under the scalar-tempered likelihood. This is not the C5 gate,
  which needs the real sampler on overlapping peaks.
- **GLS:** generalized least squares under lineshape misfit, the evidence for
  rejecting it.

**CEST (optional, a comparison and not a validation):** the header self-consistency
check, the predicted against the measured noise autocorrelation, and the predicted
conditional sandwich factor for each peak.

    uv run python tools/validate_correlated_noise.py --output /tmp/correlated-noise.json
    uv run python tools/validate_correlated_noise.py --results RUN --spectrum FT2

With RUN a fit of the bundled CEST example and FT2 its ``pseudo3d.ft2``, the CEST
comparison is added.
"""

# ruff: noqa: T201

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

WINDOW = 13  # fitting points around the line
SNR = 30.0  # peak height / σ
Z68, Z95 = 1.0, 1.96


@dataclass(frozen=True)
class Chain:
    """One axis's processing: N acquired complex points, sine bell, zero fill to M."""

    n: int
    m: int
    offset: float | None  # sine-bell offset (π units); None: no apodization
    end: float = 0.98
    power: float = 2.0
    first: float = 1.0  # first-point scaling
    p1: float = 0.0  # first-order phase, degrees

    @property
    def window(self) -> np.ndarray:
        """The apodization over the acquired points, first point scaled."""
        if self.offset is None:
            w = np.ones(self.n)
        else:
            t = np.arange(self.n) / (self.n - 1)
            w = np.sin(np.pi * self.offset + np.pi * (self.end - self.offset) * t) ** self.power
        w = w.copy()
        w[0] *= self.first
        return w

    def process(self, fid: np.ndarray) -> np.ndarray:
        """Apodize, zero fill, Fourier transform, phase and keep the real part."""
        x = np.zeros((*fid.shape[:-1], self.m), complex)
        x[..., : self.n] = fid * self.window
        # NMRPipe stores frequency in decreasing order: its P1 ramp over stored points
        # is e^{−iφ₁k/M} in the transform index k (tests/data/nmrpipe).
        phase = np.exp(-1j * np.deg2rad(self.p1) * np.arange(self.m) / self.m)
        return np.real(np.fft.fft(x, axis=-1) * phase)

    def rho(self) -> np.ndarray:
        """Noise autocorrelation at every lag 0…M−1: Re[C(k)e^{−iφ₁k/M}] / C(0)."""
        w2 = self.window**2
        k = np.arange(self.m)[:, None]
        c = (w2 * np.exp(-2j * np.pi * k * np.arange(self.n) / self.m)).sum(axis=1)
        return np.real(c * np.exp(-1j * np.deg2rad(self.p1) * k[:, 0] / self.m)) / c[0].real

    def noise_sd(self, s: float = 1.0) -> float:
        """SD of a processed real point for FID noise of SD ``s`` per component."""
        return s * float(np.sqrt(np.sum(self.window**2)))


CEST_15N = Chain(92, 256, 0.4, first=1.0, p1=180.0)
CEST_1H = Chain(768, 2048, 0.3, first=0.5)
SYNTHETIC = Chain(92, 256, 0.4)  # the ¹⁵N chain, without the half-dwell phase


def lag_matrix(rho: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Correlation between the points at ``index`` of one axis."""
    return rho[np.abs(index[:, None] - index[None, :]) % rho.size]


def _complex_noise(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


# ---------------------------------------------------------------- C1


def c1(rng) -> dict:
    """The correlation formula against Monte Carlo, and the inflation table."""
    chains = {
        "CEST 15N (92->256, SP 0.4, P1 180)": CEST_15N,
        "CEST 1H (768->2048, SP 0.3, first point 0.5)": CEST_1H,
        "zero fill x2, no apodization": Chain(92, 184, None),
        "zero fill x4, no apodization": Chain(92, 368, None),
    }
    correlation = {}
    for name, chain in chains.items():
        noise = chain.process(_complex_noise(rng, (4000, chain.n)))
        noise /= noise.std()
        measured = [float(np.mean(noise * np.roll(noise, -k, axis=-1))) for k in range(6)]
        predicted = chain.rho()[:6]
        correlation[name] = {
            "predicted": [float(v) for v in predicted],
            "monte_carlo": measured,
            "max_abs_difference": float(np.max(np.abs(predicted - measured))),
        }
    table = {}
    for name, chain in {
        "no zero fill, sine-bell^2 (off 0.5)": Chain(92, 92, 0.5),
        "zero fill x2, no apodization": Chain(92, 184, None),
        "zero fill x2, sine-bell^2 (off 0.4)": Chain(92, 184, 0.4),
        "zero fill x2, sine-bell^2 (off 0.5)": Chain(92, 184, 0.5),
        "zero fill x4, no apodization": Chain(92, 368, None),
        "zero fill x4, sine-bell^2 (off 0.5)": Chain(92, 368, 0.5),
    }.items():
        r = lag_matrix(chain.rho(), np.arange(chain.m))
        row = {"rho1": float(chain.rho()[1])}
        for decay in (0.3, 1.3, 5.0):
            phi = chain.process(np.exp(-decay * np.arange(chain.n) / chain.n))
            row[f"inflation_r2aq_{decay}"] = float(np.sqrt(phi @ r @ phi / (phi @ phi)))
        table[name] = row
    # Monte Carlo of two rows: least-squares amplitude SD / σ-only SD, all points.
    checks = {}
    for name, chain in {
        "zero fill x2, sine-bell^2 (off 0.5), r2aq 1.3": Chain(92, 184, 0.5),
        "zero fill x4, no apodization, r2aq 5.0": Chain(92, 368, None),
        "zero fill x2, no apodization, r2aq 5.0": Chain(92, 184, None),
    }.items():
        decay = 1.3 if "1.3" in name else 5.0
        phi = chain.process(np.exp(-decay * np.arange(chain.n) / chain.n))
        noise = chain.process(_complex_noise(rng, (20000, chain.n)))
        amplitude = noise @ phi / (phi @ phi)
        checks[name] = float(amplitude.std() / (noise.std() / np.linalg.norm(phi)))
    # A global ESS factor summed over every lag is M·w0²/Σw²: the first point alone.
    ess = {
        f"first point {first}": float(Chain(92, 256, 0.4, first=first).rho().sum())
        for first in (1.0, 0.5)
    }
    return {
        "correlation": correlation,
        "inflation_table": table,
        "monte_carlo_rows": checks,
        "full_lag_ess_sum_15N_chain": ess,
    }


def c3_prediction(rng, draws: int = 50000) -> dict:
    """Twofold zero fill without apodization: the small predicted inflation, tested.

    Odd lags keep ρ = 1/N, so the inflation is small but not 1. Monte Carlo of the
    least-squares amplitude over all points checks the prediction; the SD of an SD
    ratio estimated from ``draws`` samples is about 1/√(2·draws).
    """
    chain = Chain(92, 184, None)
    r = lag_matrix(chain.rho(), np.arange(chain.m))
    noise = chain.process(_complex_noise(rng, (draws, chain.n)))
    rows = {}
    for decay in (0.3, 1.3, 5.0):
        phi = chain.process(np.exp(-decay * np.arange(chain.n) / chain.n))
        amplitude = noise @ phi / (phi @ phi)
        rows[f"r2aq_{decay}"] = {
            "predicted": float(np.sqrt(phi @ r @ phi / (phi @ phi))),
            "monte_carlo": float(amplitude.std() / (noise.std() / np.linalg.norm(phi))),
        }
    return {"rows": rows, "monte_carlo_se": float(1 / np.sqrt(2 * draws))}


# ---------------------------------------------------------------- the synthetic line


@dataclass(frozen=True)
class Line:
    """One synthetic line through a processing chain, fitted over a window."""

    chain: Chain
    decay: float  # R2 · acquisition time
    centre: float  # in processed points

    @property
    def index(self) -> np.ndarray:
        """The fitting window."""
        c = round(self.centre)
        return np.arange(c - WINDOW // 2, c + WINDOW // 2 + 1)

    def fid(self, theta, n=None) -> np.ndarray:
        """The time-domain signal for (amplitude, position, R2·aq)."""
        amplitude, position, decay = theta
        t = np.arange(self.chain.n if n is None else n)
        return amplitude * np.exp(
            2j * np.pi * position * t / self.chain.m - decay * t / self.chain.n
        )

    def model(self, theta) -> np.ndarray:
        """The processed line over the window."""
        return self.chain.process(self.fid(theta))[self.index]

    def jacobian(self, theta) -> np.ndarray:
        """Exact derivatives over the window, (points, 3)."""
        t = np.arange(self.chain.n)
        f = self.fid(theta)
        columns = [
            f / theta[0],
            f * 2j * np.pi * t / self.chain.m,
            -f * t / self.chain.n,
        ]
        return np.stack([self.chain.process(c)[self.index] for c in columns], axis=1)

    @property
    def truth(self) -> np.ndarray:
        """Unit amplitude at the line's centre and decay."""
        return np.array([1.0, self.centre, self.decay])

    def noise_scale(self) -> float:
        """FID noise SD per component that gives the target SNR."""
        height = float(np.max(self.chain.process(self.fid(self.truth))))
        return height / SNR / self.chain.noise_sd()


def _coverage(estimates, truth, sds) -> list[float]:
    z = np.abs(estimates - truth) / sds
    return [float(np.mean(z < Z68)), float(np.mean(z < Z95))]


def c2_c4(line: Line, rng, realizations: int) -> dict:
    """Conditional and joint-fit coverage (C2/C3) and the χ² distribution (C4)."""
    sigma = line.chain.noise_sd(line.noise_scale())
    r = lag_matrix(line.chain.rho(), line.index)
    noises = line.chain.process(
        line.noise_scale() * _complex_noise(rng, (realizations, line.chain.n))
    )[:, line.index]
    # Conditional amplitude: the lineshape held at the truth, a linear estimate.
    phi = line.model(line.truth) / line.truth[0]
    conditional = (line.model(line.truth) + noises) @ phi / (phi @ phi)
    independent_sd = sigma / np.linalg.norm(phi)
    sandwich_sd = sigma * np.sqrt(phi @ r @ phi) / (phi @ phi)
    result = {
        "conditional_amplitude": {
            "scatter_over_sigma_only": float(conditional.std() / independent_sd),
            "scatter_over_sandwich": float(conditional.std() / sandwich_sd),
            "sigma_only_coverage": _coverage(conditional, 1.0, independent_sd),
            "sandwich_coverage": _coverage(conditional, 1.0, sandwich_sd),
        }
    }
    # Joint fit of amplitude, position and R2: total errors, to first order.
    estimates, independent, sandwich, chi2 = [], [], [], []
    for noise in noises:
        y = line.model(line.truth) + noise
        fit = least_squares(lambda th, y=y: line.model(th) - y, line.truth, method="lm")
        j = line.jacobian(fit.x)
        inverse = np.linalg.inv(j.T @ j)
        estimates.append(fit.x)
        independent.append(sigma * np.sqrt(np.diag(inverse)))
        sandwich.append(sigma * np.sqrt(np.diag(inverse @ j.T @ r @ j @ inverse)))
        chi2.append(float(np.sum(fit.fun**2) / sigma**2))
    estimates, independent, sandwich = map(np.array, (estimates, independent, sandwich))
    joint = {}
    for k, name in enumerate(("amplitude", "position", "r2")):
        scatter = estimates[:, k].std()
        joint[name] = {
            "scatter_over_sigma_only": float(scatter / np.median(independent[:, k])),
            "scatter_over_sandwich": float(scatter / np.median(sandwich[:, k])),
            "sigma_only_coverage": _coverage(estimates[:, k], line.truth[k], independent[:, k]),
            "sandwich_coverage": _coverage(estimates[:, k], line.truth[k], sandwich[:, k]),
        }
    result["joint_fit"] = joint
    # Total (joint-fit) against conditional amplitude uncertainty, both sandwich.
    result["total_over_conditional_amplitude_sd"] = float(np.median(sandwich[:, 0]) / sandwich_sd)
    j = line.jacobian(line.truth)
    hat = j @ np.linalg.inv(j.T @ j) @ j.T
    residual = (np.eye(WINDOW) - hat) @ r
    result["chi_squared"] = {
        "nominal_degrees_of_freedom": WINDOW - 3,
        "expected_trace": float(np.trace(residual)),
        "observed_mean": float(np.mean(chi2)),
        "predicted_variance": float(2 * np.trace(residual @ residual)),
        "observed_variance": float(np.var(chi2)),
        "nominal_reduced_mean": float(np.mean(chi2) / (WINDOW - 3)),
    }
    return result


# ---------------------------------------------------------------- C5 (grid)


def c5(line: Line, rng, realizations: int) -> dict:
    """Exact grid posteriors of (position, R2), amplitude drawn conditionally.

    The independence likelihood is today's profile likelihood (the conditional
    Jeffreys prior makes it exact for one amplitude). The tempered likelihood raises
    it to the power k = p / tr(H_I V), with V the sandwich covariance at the truth.
    """
    sigma = line.chain.noise_sd(line.noise_scale())
    r = lag_matrix(line.chain.rho(), line.index)
    j = line.jacobian(line.truth)
    information = j.T @ j / sigma**2
    inverse = np.linalg.inv(j.T @ j)
    sandwich = sigma**2 * inverse @ j.T @ r @ j @ inverse
    k = 3 / float(np.trace(information @ sandwich))
    spread = 8 * np.sqrt(np.diag(sandwich))
    positions = line.centre + np.linspace(-spread[1], spread[1], 81)
    decays = line.decay + np.linspace(-spread[2], spread[2], 81)
    grid = [(p, d) for p in positions for d in decays]
    shapes = np.array([line.model((1.0, p, d)) for p, d in grid])  # (grid, window)
    norms = np.sum(shapes**2, axis=1)
    noises = line.chain.process(
        line.noise_scale() * _complex_noise(rng, (realizations, line.chain.n))
    )[:, line.index]
    hits = {"independence": [], "tempered": []}
    for noise in noises:
        y = line.model(line.truth) + noise
        projection = shapes @ y
        chi2 = (y @ y - projection**2 / norms) / sigma**2
        for name, power in (("independence", 1.0), ("tempered", k)):
            weights = np.exp(-0.5 * power * (chi2 - chi2.min()))
            weights /= weights.sum()
            draws = rng.choice(len(grid), size=4000, p=weights)
            mean = projection[draws] / norms[draws]
            amplitude = mean + sigma / np.sqrt(power * norms[draws]) * rng.standard_normal(4000)
            samples = {
                "amplitude": amplitude,
                "position": np.array([grid[i][0] for i in draws]),
                "r2": np.array([grid[i][1] for i in draws]),
            }
            hits[name].append(
                {
                    q: [
                        bool(np.quantile(v, 0.16) <= t <= np.quantile(v, 0.84)),
                        bool(np.quantile(v, 0.025) <= t <= np.quantile(v, 0.975)),
                    ]
                    for (q, v), t in zip(samples.items(), line.truth, strict=True)
                }
            )
    return {
        "tempering_k": k,
        **{
            name: {
                q: [float(np.mean([h[q][i] for h in rows])) for i in (0, 1)]
                for q in ("amplitude", "position", "r2")
            }
            for name, rows in hits.items()
        },
    }


# ---------------------------------------------------------------- GLS under misfit


def gls_under_misfit(line: Line, rng, realizations: int) -> dict:
    """Least squares and GLS fitting an unapodized Lorentzian to apodized data."""
    r = lag_matrix(line.chain.rho(), line.index)
    whiten = np.linalg.inv(np.linalg.cholesky(r))
    flat = Chain(line.chain.n, line.chain.m, None)
    wrong = Line(flat, line.decay, line.centre)

    def model(theta):  # amplitude scaled to the apodized line's height
        return wrong.model(theta) * scale

    scale = np.max(line.model(line.truth)) / np.max(wrong.model(line.truth))
    noises = line.chain.process(
        line.noise_scale() * _complex_noise(rng, (realizations, line.chain.n))
    )[:, line.index]
    results = {"ls": [], "gls": []}
    for noise in noises:
        y = line.model(line.truth) + noise
        results["ls"].append(
            least_squares(lambda th, y=y: model(th) - y, line.truth, method="lm").x
        )
        results["gls"].append(
            least_squares(lambda th, y=y: whiten @ (model(th) - y), line.truth, method="lm").x
        )
    sds = {name: np.array(values).std(axis=0) for name, values in results.items()}
    biases = {name: np.array(values).mean(axis=0) - line.truth for name, values in results.items()}
    return {
        "condition_number_of_r": float(np.linalg.cond(r)),
        **{
            f"{name}_position_bias_over_own_sd": float(biases[name][1] / sds[name][1])
            for name in results
        },
        "gls_over_ls_position_sd": float(sds["gls"][1] / sds["ls"][1]),
    }


# ---------------------------------------------------------------- CEST comparison


def cest(results: Path, spectrum: Path) -> dict:
    """Header consistency, predicted against measured correlation, sandwich factors."""
    import nmrglue as ng  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).parent))
    from validate_sigma_uncertainties import _noise_autocorrelation  # noqa: PLC0415

    from peakfit.engine.algorithms.noise import estimate_noise  # noqa: PLC0415
    from peakfit.engine.scientific.sampling import sampling_problem  # noqa: PLC0415
    from peakfit.io.readers.results import ResultsLoader  # noqa: PLC0415

    dic, data = ng.pipe.read(str(spectrum))
    header = {}
    for axis, fdf in (("15N", "FDF3"), ("1H", "FDF2")):
        stored = data.shape[1] if axis == "15N" else data.shape[2]
        header[axis] = {
            "tdsize": dic[f"{fdf}TDSIZE"],
            "ftsize": dic[f"{fdf}FTSIZE"],
            "apod": dic[f"{fdf}APOD"],
            "apod_x_ftsize_over_stored": dic[f"{fdf}APOD"] * dic[f"{fdf}FTSIZE"] / stored,
            "first_point": dic[f"{fdf}C1"] + 1.0,
            "p1": dic[f"{fdf}P1"],
        }
    # Reading APOD as the acquired length would describe a different chain.
    header["1H"]["rho1_if_apod_were_acquired_points"] = float(
        Chain(int(dic["FDF2APOD"]), int(dic["FDF2FTSIZE"]), 0.3, first=0.5).rho()[1]
    )
    sigma = float(estimate_noise(data))
    measured = [_noise_autocorrelation(data, sigma, axis)[:5] for axis in (0, 1)]
    predicted = [CEST_15N.rho()[:5], CEST_1H.rho()[:5]]
    completed = ResultsLoader(results).load_completed()
    rho = (CEST_15N.rho(), CEST_1H.rho())
    factors = []
    for outcome in completed.outcome.clusters:
        if not outcome.usable:
            continue
        problem = sampling_problem(completed, outcome.cluster_id)
        cluster = problem.cluster
        shapes = cluster.evaluate(problem.parameters)  # (peaks, points)
        r = lag_matrix(rho[0], cluster.grid_indices[0]) * lag_matrix(
            rho[1], cluster.grid_indices[1]
        )
        inverse = np.linalg.pinv(shapes @ shapes.T)
        sandwich = inverse @ shapes @ r @ shapes.T @ inverse
        factors.extend(np.sqrt(np.diag(sandwich) / np.diag(inverse)).tolist())
    return {
        "header": header,
        "autocorrelation_15N": {
            "predicted": [float(v) for v in predicted[0]],
            "measured": [float(v) for v in measured[0]],
        },
        "autocorrelation_1H": {
            "predicted": [float(v) for v in predicted[1]],
            "measured": [float(v) for v in measured[1]],
        },
        "conditional_sandwich_factor_median_iqr": np.percentile(factors, [50, 25, 75]).tolist(),
        "peaks": len(factors),
    }


def main() -> None:
    """Run C1–C5 and the GLS check, and the CEST comparison when given a fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--realizations", type=int, default=1000)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--spectrum", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # One independent stream per study, so a change to one leaves the others' numbers.
    streams = iter(np.random.default_rng(s) for s in np.random.SeedSequence(20261104).spawn(8))
    narrow = Line(SYNTHETIC, 1.3, 64.0)
    broad = Line(SYNTHETIC, 5.2, 64.0)
    zero_fill = Line(Chain(92, 184, None), 1.3, 46.0)
    report = {
        "C1": c1(next(streams)),
        "C3_prediction": c3_prediction(next(streams)),
        "C2_C4": {
            "narrow line (R2 aq 1.3)": c2_c4(narrow, next(streams), args.realizations),
            "broad line (R2 aq 5.2)": c2_c4(broad, next(streams), args.realizations),
            "zero fill x2, no apodization": c2_c4(zero_fill, next(streams), args.realizations),
        },
        "C5_grid": {"narrow line (R2 aq 1.3)": c5(narrow, next(streams), 600)},
        "GLS_under_misfit": {
            "narrow line, apodization not modelled": gls_under_misfit(narrow, next(streams), 600)
        },
    }
    if args.results and args.spectrum:
        report["CEST"] = cest(args.results, args.spectrum)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
