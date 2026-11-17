# J coupling of doublet lineshapes

Validation date: 2026-11-17 (W2a). Tool: `tools/validate_doublet_j.py`, output
[doublet-j.json](doublet-j.json). Regression tests: `tests/test_doublet_j.py`,
`tests/test_lineshapes_equivalence.py`.

## Scientific impact

Before this change, **no fitted J value or J uncertainty from a doublet lineshape can
be assumed reliable**, in any release. That covers `gaussian_doublet`,
`lorentzian_doublet`, `pvoigt_doublet`, `no_apod_doublet`, `sp1_doublet` and
`sp2_doublet`, in the CLI and the GUI.

- **J was not fitted by its gradient.** It either stayed exactly at its start or moved
  by numerical accident.
- **Other values were affected too.** Every nonlinear standard error (cs, lw, J, phase)
  and every profile σ of a doublet cluster was withheld, and every doublet cluster
  carried the "Unreliable local uncertainties" warning. Positions, linewidths and
  amplitudes were fitted with J held wherever it happened to sit, so they can be
  biased too.
- **What to do with old results:** refit them.
- **Singlet results are unaffected** and bit-identical (below).

## Cause

A doublet is k(Δω₊) + k(Δω₋), with its components at cs ± J/2. No doublet
lineshape returned a derivative with respect to J. `Cluster.evaluate` skipped any
parameter without one, so J's column of the VarPro Jacobian was exactly zero.

**Why J sometimes still moved.** This was traced on the bundled pseudo-3D with a
`lorentzian_doublet`, instrumenting SciPy 1.17's `trf`. There, the ¹⁵N J moved
from 5.0 to 6.37 Hz while the ¹H J stayed at exactly 5.0 Hz.

- With a zero gradient, the Coleman–Li scaling of a J component is 1.
- The SVD of the augmented Jacobian therefore has two (near-)null directions, one
  per zero column:
    - for the ¹H J, the singular value came out exactly 0, giving no step;
    - for the ¹⁵N J, it came out at about 10⁻¹⁵.
- With the trust-region regularisation α ≈ 10⁻¹⁶, the step component
  s·uf/(s² + α) became about ±157 Hz. It was clipped back from the 1–10 Hz bounds.
- Such trial points were accepted whenever the true cost happened to fall: the
  residual does depend on J, even though its gradient said it did not.

The ¹⁵N J was therefore a round-off-driven random search, not an estimate. Which J
moves depends on floating-point details of each SVD. With the default 1–10 Hz
bounds, it landed at about 1.045, 9.955 or points between.

## Change

1. **Derivative.** ∂/∂J = ½(s₋ ∂k/∂Δω₋ − s₊ ∂k/∂Δω₊), from each kernel's existing
   offset derivative (`lineshapes/doublet.py`). It covers the three real doublets
   and, through the phase rotation, the three phased ones.
2. **Fail fast.** A varied parameter whose lineshape returns no derivative now raises
   a `ValueError` naming it, instead of leaving a zero Jacobian column
   (`Cluster.evaluate`). A fixed parameter needs none.
3. **Unresolved couplings.**
    - The doublet is even in J, so near J = 0 it depends on J², and a first-order
      σ(J) holds only while σ(J) ≪ J.
    - σ(J) is withheld when σ(J)/Ĵ > 0.1. That is the limit already used for
      first-order profile ratios (`profiles.RELIABILITY_LIMIT`).
    - Whenever a free J has no σ, the cluster's lineshape covariance factors, and so
      its profile σ, are withheld too, because they propagate the same first-order J
      uncertainty. A free J can lack σ through this rule or through the existing rule
      for an error larger than the distance to a bound.
    - The existing cluster warning follows, since a varied parameter has no finite
      error.

## Derivative checks

- **Against finite differences.** The analytic ∂/∂J matches central finite
  differences (step 10⁻³ Hz), to 10⁻⁶ of each peak's largest derivative, for:
    - all six doublets;
    - J = 2, 7, 35 and 92 Hz;
    - two peaks of different linewidth evaluated together;
    - a direct axis, and an indirect axis with P180 sign alternation where a ¹J
      component aliases;
    - phased doublets unphased, at 0° and at 12°;
    - the sine-bell apodization of SP1 and SP2.
- **Through the VarPro objective.** The full VarPro Jacobian, J column included,
  matches finite differences of the projected residual for all six doublets.

## Synthetic recovery and coverage

**Setup:**
- one 1D doublet over three planes (profile 1 : 0.6 : 0.3);
- 512 points on a 3000 Hz, 600 MHz axis;
- the stated first-plane peak height over the noise;
- J starts 20 % below the truth;
- 200 noise realizations per case.

"Withheld" is the fraction of fits without σ(J). Coverage is the fraction of reported
intervals containing the truth: nominal 0.683 / 0.950, with a ±0.065 binomial
half-width at 68 %. Bias and scatter are in Hz.

| Lineshape, case | J / lw (Hz) | SNR | Bias | Scatter | Median σ | Withheld | Coverage ±1σ / ±1.96σ |
|---|---|---:|---:|---:|---:|---:|---|
| Lorentzian, resolved | 40 / 8 | 50 | +0.004 | 0.080 | 0.088 | 0 | 0.755 / 0.960 |
| Lorentzian, partly resolved | 12 / 10 | 50 | +0.033 | 0.199 | 0.196 | 0 | 0.660 / 0.955 |
| Lorentzian, one-bond | 92 / 15 | 50 | −0.011 | 0.178 | 0.181 | 0 | 0.665 / 0.970 |
| Lorentzian, partly resolved | 12 / 10 | 8 | +0.088 | 1.27 | 1.05 | 0.305 | 0.612 / 0.957 |
| Lorentzian, unresolved | 5 / 20 | 50 | −0.215 | 1.66 | 0.696 | 0.975 | 0.000 / 0.000 (5 fits) |
| Lorentzian, unresolved | 5 / 20 | 8 | +0.345 | 4.27 | 1.17 | 0.995 | 0.000 / 0.000 (1 fit) |
| SP1, resolved | 40 / 8 | 50 | +0.004 | 0.083 | 0.092 | 0 | 0.760 / 0.955 |
| SP1, partly resolved | 12 / 10 | 50 | +0.029 | 0.174 | 0.170 | 0 | 0.655 / 0.945 |
| SP1, one-bond | 92 / 15 | 50 | −0.010 | 0.174 | 0.175 | 0 | 0.675 / 0.965 |
| SP1, partly resolved | 12 / 10 | 8 | +0.114 | 1.13 | 0.949 | 0.115 | 0.621 / 0.955 |
| SP1, unresolved | 5 / 20 | 50 | −0.225 | 1.67 | 0.693 | 0.975 | 0.000 / 0.000 (5 fits) |
| SP1, unresolved | 5 / 20 | 8 | +0.180 | 4.28 | 1.14 | 0.985 | 0.000 / 0.000 (3 fits) |

- **Resolved, partly resolved and one-bond cases.** Estimates are unbiased, and the
  reported σ(J) matches the observed scatter. Coverage is within binomial tolerance,
  including at SNR 8.
- **Unresolved doublets (J = ¼ lw).**
    - Their J estimates are bimodal. At SNR 8, about half collapse to the 0.5 Hz
      bound; the rest land at 9–14 Hz with a local σ near 1.7 Hz, far below the true
      scatter.
    - Without the rule, the fits reporting σ(J) covered the truth 0.27 / 0.54
      (Lorentzian) and 0.25 / 0.53 (SP1) of the time.
    - The rule withholds 97.5–99.5 % of them.
- **Remaining limitation.** 0.5–2.5 % of unresolved fits still report a σ(J) that
  is too small. Those are realizations whose estimate came out large by chance;
  no fit-local rule separates them. A likelihood-ratio rule (keep σ(J) only if fixing
  J at its lower bound raises χ² by ≥ 3.84) was tested and rejected: it costs a refit
  and keeps more biased fits (coverage 0.47 / 0.76 at SNR 50, 0 at SNR 8). For an
  unresolved coupling, use MCMC.

On the commit before this change, the same tool gives:
- Lorentzian: J never moved in 1,200 fits.
- SP1: J moved in 62–81 % of fits, by the round-off mechanism above, still biased
  toward the start.
- σ(J): withheld in every fit.

**Why every case is at equal SNR.** Each case is scaled to the same first-plane
SNR because the SP1 kernel's peak height is 25–60 times smaller than the
Lorentzian's at equal amplitude, depending on the case. A first version of the study used equal
amplitudes, and that put SP1 at an SNR near 1.5.

## The bundled example fitted with doublets

`peakfit fit` with the golden reference's arguments (`--refine 1`, 121 clusters,
332 J values):

| | lorentzian_doublet before | after | sp1_doublet before | after |
|---|---:|---:|---:|---:|
| J exactly at its 5 Hz start | 139 | 0 | 135 | 0 |
| J at a 1 or 10 Hz bound | 0 | 39 | 0 | 15 |
| J with a reported σ | 0 | 285 | 0 | 310 |
| J withheld by σ/J > 0.1 | — | 0 | — | 4 |
| Other nonlinear σ reported | 0 % | 98.0 % | 0 % | 99.7 % |
| Profile σ reported (clusters) | 0 % | 97.5 % | 0 % | 94.2 % |
| Warned clusters | 121 | 27 | 121 | 17 |
| χ² | 5.51 × 10⁶ | 3.61 × 10⁶ | 2.20 × 10⁶ | 1.80 × 10⁶ |
| Wall time (s) | 7.2 | 8.3 | 7.6 | 7.8 |

The J values now at the 1 or 10 Hz bound show that the default 1–10 Hz range is too
narrow for this spectrum. Per-axis J ranges are W2b.

## Singlets bit-identical, performance

- **Bit identity.** The bundled example was fitted with `auto` and with each of
  gaussian, lorentzian, pvoigt, no_apod, sp1 and sp2, on this change and on its base
  commit (6a5dc18), in one environment. The outcome and sample digests of
  `tools/scientific_digest.py` are identical. The golden regression passes unchanged.
- **Timing.** One VarPro Jacobian of a two-peak 32 × 32 cluster (best of 7 × 200):
    - singlets: unchanged (Lorentzian 339 vs 341 µs, SP1 433 vs 431 µs);
    - doublets: 6–7 % slower (500 vs 472 µs, 706 vs 658 µs), for the one added
      derivative array.
- **Whole fits.** Wall time of a whole doublet fit is within run-to-run variation
  (table above).
