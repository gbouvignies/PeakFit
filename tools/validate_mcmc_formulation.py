"""Evidence for the MCMC formulation in docs/mcmc-formulation.md.

One Lorentzian with an unknown linewidth, S planes with independent amplitudes and
known noise. The linewidth posterior is evaluated on a grid (a uniform prior on the
grid's range) under three formulations:

- profile: L_p(lw) = exp(-chi2_min / 2), the integrated likelihood under the
  conditional Jeffreys amplitude prior;
- flat-height / flat-area: the flat-amplitude-prior marginal,
  L_p * det(Phi Phi^T)^(-S/2), with height- or area-normalized shapes.

It reports the bias of the posterior mean in units of the posterior standard
deviation and the frequentist coverage of the central 68% and 95% intervals, and
checks the derivation numerically:

- the Jeffreys-weighted amplitude integral equals L_p up to a constant;
- the profile posterior does not depend on the shape normalization.

    uv run python tools/validate_mcmc_formulation.py --output /tmp/mcmc-formulation.json
"""

# ruff: noqa: T201

import argparse
import json
from pathlib import Path

import numpy as np

X = np.arange(-40, 41, dtype=float)  # sampled points
TRUE_LW = 6.0
SIGMA = 1.0
GRID = np.linspace(3.0, 10.0, 1401)  # prior support of the linewidth
SCENARIOS = [(planes, snr) for planes in (1, 10, 50) for snr in (5.0, 20.0)]
METHODS = ("profile", "flat-height", "flat-area")


def shapes(widths: np.ndarray, normalization: str) -> np.ndarray:
    """Lorentzian shapes, one row per linewidth, height- or area-normalized."""
    height = 1.0 / (1.0 + (2.0 * X[None, :] / widths[:, None]) ** 2)
    if normalization == "height":
        return height
    return height / (np.pi * widths[:, None] / 2.0)


def log_posterior(y: np.ndarray, method: str) -> np.ndarray:
    """Unnormalized log posterior of the linewidth on GRID; y is (points, planes)."""
    normalization = "area" if method == "flat-area" else "height"
    phi = shapes(GRID, normalization)
    gram = np.sum(phi**2, axis=1)  # Phi Phi^T for a single peak
    projected = phi @ y  # (grid, planes)
    chi2 = (np.sum(y**2) - np.sum(projected**2, axis=1) / gram) / SIGMA**2
    log_p = -0.5 * chi2
    if method != "profile":
        log_p -= 0.5 * y.shape[1] * np.log(gram)
    return log_p


def summarize(log_p: np.ndarray) -> tuple[float, float, tuple[float, ...]]:
    """Posterior mean, standard deviation and the 2.5/16/84/97.5 % quantiles."""
    p = np.exp(log_p - log_p.max())
    p /= p.sum()
    mean = float(p @ GRID)
    std = float(np.sqrt(p @ (GRID - mean) ** 2))
    quantiles = tuple(float(q) for q in np.interp([0.025, 0.16, 0.84, 0.975], np.cumsum(p), GRID))
    return mean, std, quantiles


def coverage_study(realizations: int, rng: np.random.Generator) -> list[dict]:
    """Bias and interval coverage of each formulation over noise realizations."""
    truth = shapes(np.array([TRUE_LW]), "height")[0]
    rows = []
    for planes, snr in SCENARIOS:
        amplitudes = np.full(planes, snr * SIGMA)
        stats = {method: [] for method in METHODS}
        for _ in range(realizations):
            y = np.outer(truth, amplitudes) + SIGMA * rng.standard_normal((X.size, planes))
            for method in METHODS:
                mean, std, (q025, q16, q84, q975) = summarize(log_posterior(y, method))
                stats[method].append((mean, std, q16 <= TRUE_LW <= q84, q025 <= TRUE_LW <= q975))
        for method, values in stats.items():
            array = np.array(values, dtype=float)
            rows.append(
                {
                    "planes": planes,
                    "peak_snr": snr,
                    "method": method,
                    "bias_over_std": float((array[:, 0].mean() - TRUE_LW) / array[:, 1].mean()),
                    "coverage_68": float(array[:, 2].mean()),
                    "coverage_95": float(array[:, 3].mean()),
                }
            )
    return rows


def derivation_check(rng: np.random.Generator) -> dict:
    """Check the derivation numerically on one synthetic data set.

    The Jeffreys-weighted amplitude integral is L_p times a constant, and L_p does
    not depend on the shape normalization.
    """
    truth = shapes(np.array([TRUE_LW]), "height")[0]
    y = (truth * 5.0 + SIGMA * rng.standard_normal(X.size))[:, None]
    widths = GRID[::50]
    phi = shapes(widths, "height")
    gram = np.sum(phi**2, axis=1)
    best = (phi @ y)[:, 0] / gram
    profile = np.exp(log_posterior(y, "profile")[::50])
    ratios = []
    for row, g, a_hat, lp in zip(phi, gram, best, profile, strict=True):
        spread = SIGMA / np.sqrt(g)
        a = np.linspace(a_hat - 12 * spread, a_hat + 12 * spread, 20001)
        chi2 = np.sum((y[:, 0][None, :] - a[:, None] * row[None, :]) ** 2, axis=1) / SIGMA**2
        integral = np.trapezoid(np.exp(-0.5 * chi2) * np.sqrt(g), a)
        ratios.append(integral / lp)
    ratios = np.array(ratios)
    many = SIGMA * rng.standard_normal((X.size, 10)) + np.outer(truth, np.full(10, 5.0))
    height = log_posterior(many, "profile")
    phi_area = shapes(GRID, "area")
    projected = phi_area @ many
    chi2_area = np.sum(many**2) - np.sum(projected**2, axis=1) / np.sum(phi_area**2, axis=1)
    area = -0.5 * chi2_area / SIGMA**2
    return {
        "jeffreys_integral_over_profile_relative_spread": float(np.ptp(ratios) / ratios.mean()),
        "jeffreys_constant_expected": float(np.sqrt(2 * np.pi) * SIGMA),
        "jeffreys_constant_observed": float(ratios.mean()),
        "profile_normalization_max_abs_difference": float(np.max(np.abs(height - area))),
    }


def main() -> None:
    """Run the study and print one line per scenario."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--realizations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    check = derivation_check(rng)
    rows = coverage_study(args.realizations, rng)
    for key, value in check.items():
        print(f"{key}: {value:.3e}")
    for row in rows:
        print(
            f"S={row['planes']:3d} SNR={row['peak_snr']:4.0f} {row['method']:<11} "
            f"bias/std={row['bias_over_std']:+.2f} "
            f"cover68={row['coverage_68']:.3f} cover95={row['coverage_95']:.3f}"
        )
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "realizations": args.realizations,
                    "derivation": check,
                    "scenarios": rows,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
