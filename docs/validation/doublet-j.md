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
   a `MissingDerivativeError` naming it, instead of leaving a zero Jacobian column
   (`Cluster.evaluate`). It is not a lineshape evaluation error, so the profile-σ
   path cannot absorb it. A fixed parameter needs none.
3. **Unresolved couplings.**
    - The doublet is even in J, so near J = 0 it depends on J², and a first-order
      σ(J) holds only while σ(J) ≪ J.
    - σ(J) is withheld when σ(J)/Ĵ > 0.1, the same `RELIABILITY_LIMIT` that applies
      to first-order profile ratios.
    - Whenever a varied J has no σ, the cluster's lineshape covariance factors, and so
      its profile σ, are withheld too. This applies whether J is unresolved, at a
      bound, or has an error larger than its distance to a bound. Those factors would
      propagate the same first-order J uncertainty, or condition on an arbitrary bound.
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
- the stated first-plane peak height over the noise (SNR);
- J starts 20 % below the truth.

**Protocol.** It was fixed and committed (20a6461) before this run; see the tool's
docstring.
- 1000 realizations per case, with seeds (20261117, case, index), in 5 blocks.
- Calibration gate on resolved, partly resolved and one-bond cases: coverage of the
  reported ±1σ and ±1.96σ within 3 SE of 0.6827 and 0.95.
- Misleading-confidence gate on every case: the fraction of all fits that report a
  ±1.96σ interval missing the truth is at most 0.05 + 3 SE.

In the table, "Withheld" is the fraction of fits without σ(J), and each gated rate is
followed by its deviation in SE. Bias, scatter and RMS σ are in Hz.

| Lineshape, case | J / lw (Hz) | SNR | Bias | Scatter | RMS σ | Withheld | ±1σ (SE) | ±1.96σ (SE) | Misleading (SE) | Gate |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| Lorentzian, resolved (J = 5 lw) | 40 / 8 | 50 | +0.002 | 0.090 | 0.089 | 0.000 | 0.670 (-0.9) | 0.943 (-1.0) | 0.057 (+1.0) | pass |
| Lorentzian, partly resolved (J = 1.2 lw) | 12 / 10 | 50 | -0.001 | 0.194 | 0.196 | 0.000 | 0.688 (+0.4) | 0.950 (+0.0) | 0.050 (+0.0) | pass |
| Lorentzian, unresolved (J = 0.25 lw) | 5 / 20 | 50 | -0.291 | 1.513 | 0.677 | 0.984 | — | — | 0.016 (-4.9) | pass |
| Lorentzian, one-bond coupling (J = 92 Hz) | 92 / 15 | 50 | -0.005 | 0.186 | 0.181 | 0.000 | 0.685 (+0.2) | 0.941 (-1.3) | 0.059 (+1.3) | pass |
| Lorentzian, partly resolved, low SNR | 12 / 10 | 8 | +0.073 | 1.219 | 1.018 | 0.297 | 0.623 (-3.4) | 0.927 (-2.7) | 0.051 (+0.1) | FAIL |
| Lorentzian, unresolved, low SNR | 5 / 20 | 8 | +0.046 | 4.105 | 1.109 | 0.995 | — | — | 0.005 (-6.5) | pass |
| SP1, resolved (J = 5 lw) | 40 / 8 | 50 | +0.003 | 0.091 | 0.092 | 0.000 | 0.678 (-0.3) | 0.953 (+0.4) | 0.047 (-0.4) | pass |
| SP1, partly resolved (J = 1.2 lw) | 12 / 10 | 50 | +0.005 | 0.172 | 0.170 | 0.000 | 0.682 (-0.0) | 0.953 (+0.4) | 0.047 (-0.4) | pass |
| SP1, unresolved (J = 0.25 lw) | 5 / 20 | 50 | -0.277 | 1.513 | 0.653 | 0.990 | — | — | 0.010 (-5.8) | pass |
| SP1, one-bond coupling (J = 92 Hz) | 92 / 15 | 50 | -0.001 | 0.169 | 0.175 | 0.000 | 0.713 (+2.1) | 0.951 (+0.1) | 0.049 (-0.1) | pass |
| SP1, partly resolved, low SNR | 12 / 10 | 8 | +0.009 | 1.062 | 0.949 | 0.108 | 0.622 (-3.9) | 0.936 (-1.9) | 0.057 (+1.0) | FAIL |
| SP1, unresolved, low SNR | 5 / 20 | 8 | -0.015 | 4.117 | 1.030 | 0.980 | — | — | 0.020 (-4.4) | pass |

**Calibrated: resolved, partly resolved at SNR 50, and one-bond doublets.**
- Estimates are unbiased, and the RMS reported σ(J) matches the scatter within 4 %.
- Every gate passes, in the full run and in the replication over blocks 2–5.

**Fails the calibration gate: partly resolved doublets at SNR 8.**
- ±1σ coverage is 0.623 (Lorentzian) and 0.622 (SP1), against 0.683: −3.4 and −3.9 SE.
- ±1.96σ coverage is 0.927 and 0.936, inside its gate.
- The reported σ(J) is 10–16 % below the scatter (1.02 against 1.22 Hz; 0.95 against
  1.06 Hz).
- The rule already withholds σ(J) in 30 % and 11 % of these fits. The undercoverage
  is in the fits it keeps, where the RMS σ(J) is about 0.08 of J, near the limit.
- The blocks agree (consistency p = 0.52–0.94), and blocks 2–5 replicate the failure.
- This is first-order nonlinearity near the reliability limit, the same size as the
  4–6 % first-order shortfall recorded in `profile-uncertainty.md`.
- The misleading-confidence gate passes: 5.1 % and 5.7 % of all fits report a missed
  ±1.96σ interval.
- It is not decided here whether to lower the limit; changing it requires a new
  protocol and run.

**Unresolved doublets (J = ¼ lw).**
- Their J estimates are bimodal. Some collapse onto the 0.5 Hz bound, others land
  well above the truth: the scatter is 1.5 Hz at SNR 50 and 4.1 Hz at SNR 8, for a
  true J of 5 Hz.
- The rule withholds σ(J) in 98–99.5 % of these fits, so only 0.5–2.0 % of all fits
  report a missed interval.
- No fit-local rule removes that remainder, so use MCMC for an unresolved coupling.

**Exploratory runs before this protocol.** These were 200 realizations with seeds
0–199, and their results are not part of the record.
- **Equal amplitudes.** A first design used equal amplitudes, which put SP1 near
  SNR 1.5, because the SP1 kernel's peak is 25–60 times lower than the Lorentzian's.
  Equal SNR replaced it.
- **The rule was chosen on these runs.** Without it, unresolved fits at SNR 8 that
  reported σ(J) covered the truth 0.25–0.27 (±1σ) and 0.53–0.54 (±1.96σ) of the
  time.
- **The alternative rule was rejected.** A likelihood-ratio rule (keep σ(J) only if
  fixing J at its lower bound raises χ² by ≥ 3.84) cost a refit and kept more biased
  fits: coverage 0.47 / 0.76 at SNR 50, and 0 at SNR 8.
- **Before the fix.** On the base commit, Lorentzian J never moved in 1,200 fits.
  SP1 J moved in 62–81 % of fits, by the round-off mechanism above, still biased
  toward the start. σ(J) was withheld in every fit.

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
| Profile σ reported (clusters) | 0 % | 77.7 % | 0 % | 86.8 % |
| Warned clusters | 121 | 27 | 121 | 17 |
| χ² | 5.51 × 10⁶ | 3.61 × 10⁶ | 2.20 × 10⁶ | 1.80 × 10⁶ |
| Wall time (s) | 7.2 | 8.0 | 7.6 | 7.6 |

The J values now at the 1 or 10 Hz bound show that the default 1–10 Hz range is too
narrow for this spectrum. Their clusters lose their profile σ, which would otherwise
be conditional on that bound. Per-axis J ranges are W2b.

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
