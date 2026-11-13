"""Detect linear prediction on the bundled example's ¹⁵N axis from its noise (§2.2).

The FFT of a processed axis's noise is its time-domain noise. For acquired points it
has power w(t)², and it is zero in the zero-filled range. Points that linear
prediction (LP) added carry less noise than acquired ones, so a drop in measured
power before the header's TDSIZE reveals LP that the header does not record.

The script compares the measured power profile, normalized over the first 40 points,
with the header's prediction, and fits the number of acquired points N0 (noise beyond
N0 set to zero). It also compares lag correlations on both axes.
"""

import nmrglue as ng
import numpy as np
from cest_example import NOISE_ONLY, SPECTRUM, noise_only_columns
from processed_lines import Chain

HEADER_15N = Chain(92, 256, 0.4, first=1.0, p1=180.0)  # TDSIZE 92, FTSIZE 256, SP 0.4
HEADER_1H = Chain(768, 2048, 0.3, first=0.5)
DRAWS = 4000
NORMALIZE = 40  # points at the start of the FID, well before any candidate N0


def power_profile(spectra: np.ndarray) -> np.ndarray:
    """Mean time-domain power, normalized over the first points."""
    p = (np.abs(np.fft.ifft(spectra, axis=1)) ** 2).mean(axis=0)
    return p / p[:NORMALIZE].sum()


def predicted(acquired: int, rng) -> np.ndarray:
    """Power profile if only the first ``acquired`` points carry noise."""
    z = rng.standard_normal((DRAWS, HEADER_15N.n)) + 1j * rng.standard_normal((DRAWS, HEADER_15N.n))
    z[:, acquired:] = 0.0
    return power_profile(HEADER_15N.process(z))


def lag_correlation(v: np.ndarray, lags: int) -> np.ndarray:
    """Correlation at lags 1…lags, pooled over rows."""
    v = v - v.mean(axis=1, keepdims=True)
    c0 = (v * v).mean()
    return np.array([(v[:, :-k] * v[:, k:]).mean() / c0 for k in range(1, lags + 1)])


def main() -> None:
    """Print the power profile, the fitted N0 and the lag correlations."""
    rng = np.random.default_rng(0)
    columns = noise_only_columns()
    measured = power_profile(columns)
    header = predicted(HEADER_15N.n, rng)
    print(f"noise-only ¹⁵N columns: {len(columns)}")
    print("t    measured/predicted (header, N = 92)")
    for t in (*range(0, 44, 4), *range(44, 53), 56, 64, 72, 80, 88):
        print(f"{t:3d}  {measured[t] / header[t]:.2f}")
    print("mismatch Σ|measured − predicted| / Σ measured over t < 100, by acquired points N0:")
    for acquired in (92, 64, 50, 48, 47, 46, 45, 44):
        p = predicted(acquired, rng)
        mismatch = np.abs(measured[:100] - p[:100]).sum() / measured[:100].sum()
        print(f"  N0 = {acquired:2d}: {mismatch:.3f}")

    _, data = ng.pipe.read(str(SPECTRUM))
    x = np.asarray(data, dtype=float)
    sigma = 1.4826 * np.median(np.abs(x - np.median(x)))
    rows = x.reshape(-1, x.shape[2])
    rows = rows[np.abs(rows).max(axis=1) < NOISE_ONLY * sigma]
    np.set_printoptions(precision=3, suppress=True)
    print(f"noise-only ¹H rows: {len(rows)}")
    print("¹H  lag 1–6 measured :", lag_correlation(rows, 6))
    print("¹H  lag 1–6 predicted:", HEADER_1H.rho()[1:7])
    print("¹⁵N lag 1–6 measured :", lag_correlation(columns, 6))
    print("¹⁵N lag 1–6 predicted:", HEADER_15N.rho()[1:7])


if __name__ == "__main__":
    main()
