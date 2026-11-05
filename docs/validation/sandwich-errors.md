# Sandwich errors with PeakFit's fitting engine

Validation dates: 2026-11-04 (step 2) and 2026-11-05 (step 3, the published errors).
This study is the gate of [correlated-noise.md](../correlated-noise.md) §11, and the
evidence for [ADR 0002](../adr/0002-processing-derived-sandwich-errors.md). Tool:
`tools/validate_sandwich_errors.py`. Results: [`sandwich-errors.json`](sandwich-errors.json),
plus [`sandwich-errors-sp1-replication.json`](sandwich-errors-sp1-replication.json).

## Set-up

- **Clusters:** 2D pseudo-3D clusters over four planes, drawn with PeakFit's own
  lineshapes. Peak heights are 0.5–1 and σ = 0.02.
- **Processing:** one chain per lineshape. Signal and noise go through the same chain,
  and each chain's NMRPipe header records it:

  | Lineshape | ¹⁵N | ¹H |
  |---|---|---|
  | `sp2` | 32 → 128 points, sine bell² (off 0.4), P1 180° | 64 → 256 points, sine bell² (off 0.3), first point 0.5 |
  | `sp1` | as `sp2`, sine bell to the power 1 | as `sp2`, power 1 |
  | `no_apod` | zero fill ×4, no window | zero fill ×4, no window |

- **Noise:** from the exact, separable processing operators.
- **Fits:** each realization is fitted by `run_pipeline` with the noise model that
  `read_noise_model` reads from the header, starting at the truth. Its *published*
  errors are compared:
  - nonlinear SEs, and conditional amplitude errors with amplitudes solved at the
    true lineshape;
  - against independent-point (σ-only) errors computed for the same fit.
- **White-noise control:** an `sp2` signal with unprocessed white noise, whose header
  models R = I.

**Pass criteria, fixed in advance:**
- For the isolated and overlapping cases of each lineshape, sandwich coverage within
  3 binomial SE of 68% and 95%, and scatter / SD within 0.9–1.1 for every quantity.
- Identical covariance withholding in every case.
- Published errors equal to the sandwich.
- R = I reproduces σ-only to 10⁻¹⁰.

## Results

Scatter / median stated SD is given per quantity (range over the parameters, or
over the peaks and planes); coverage is pooled through z. Each case has 400
realizations.

| Case | Quantity | σ-only scatter/SD · 68% · 95% | Published (sandwich) scatter/SD · 68% · 95% |
|---|---|---|---|
| Isolated, `sp2` | positions | 2.86–3.56 · 0.25 · 0.45 | 1.01–1.08 · 0.65 · 0.95 |
| | linewidths | 2.80–3.01 · 0.27 · 0.48 | 1.02–1.04 · 0.65 · 0.95 |
| | conditional amplitudes | 3.40–3.53 · 0.22 · 0.43 | 0.97–1.01 · 0.69 · 0.96 |
| Overlapping, `sp2` | positions | 2.43–2.57 · 0.30 · 0.56 | 1.00–1.03 · 0.67 · 0.95 |
| | linewidths | 2.40–2.86 · 0.30 · 0.54 | 0.94–1.02 · 0.68 · 0.95 |
| | conditional amplitudes | 3.09–3.53 · 0.24 · 0.44 | 0.92–1.05 · 0.67 · 0.95 |
| Isolated, `sp1` | positions | 2.44–2.45 · 0.29 · 0.55 | **0.88**–1.00 · 0.69 · 0.97 |
| | linewidths | 2.41–2.71 · 0.28 · 0.56 | 0.96–1.04 · 0.68 · 0.95 |
| | conditional amplitudes | 2.84–2.96 · 0.26 · 0.49 | 0.95–0.99 · 0.69 · 0.96 |
| Overlapping, `sp1` | positions | 2.05–2.09 · 0.36 · 0.65 | 0.97–0.98 · 0.69 · 0.95 |
| | linewidths | 2.13–2.45 · 0.33 · 0.61 | 0.91–0.99 · 0.71 · 0.95 |
| | conditional amplitudes | 2.83–2.98 · 0.28 · 0.49 | 0.98–1.03 · 0.67 · 0.95 |
| Isolated, `no_apod` | positions | 1.93–2.06 · 0.40 · 0.69 | 0.99–1.00 · 0.70 · 0.95 |
| | linewidths | 1.97–2.00 · 0.38 · 0.68 | 0.98–0.99 · 0.69 · 0.96 |
| | conditional amplitudes | 2.06–2.18 · 0.36 · 0.64 | 1.00–1.05 · 0.66 · 0.94 |
| Overlapping, `no_apod` | positions | 1.77–2.02 · 0.40 · 0.68 | 0.92–1.04 · 0.67 · 0.94 |
| | linewidths | 1.95–2.03 · 0.39 · 0.68 | 0.98–1.02 · 0.69 · 0.95 |
| | conditional amplitudes | 2.02–2.18 · 0.36 · 0.64 | 0.98–1.06 · 0.67 · 0.95 |
| White control, R = I (150) | all | identical | identical, to 1.1 × 10⁻¹⁵ |

**Publication and withholding:**
- In every realization of every case, the published errors were the sandwich.
- The covariance gate withheld exactly the same realizations under both conventions.
  For duplicated peaks that was the nonlinear covariance in 134 of 150 realizations,
  and the amplitude covariance in 133.
- The bound-crossing rule acts on the published SE, so the larger sandwich SE crosses
  a bound more often: near coalescence, 112 parameter errors were withheld against
  69.

**Verdict:** every pre-registered criterion passes except one, in the recorded run.
"Isolated, `sp1`", ¹⁵N position: scatter / SD was 0.88, below 0.9, with coverage
0.69 · 0.97.
- An independent replication with fresh seeds (realizations 400–1,999) gave
  1.009 and 1.013 (± 0.018) for the two positions. Every criterion passed there.
- The recorded value is therefore a sampling fluctuation. With about 18 such checks
  at 400 realizations, the 0.9 bound lies only 2.9 SE from 1.

The recorded failure is kept, not rerun away. The replication reproduces with:

```bash
uv run python tools/validate_sandwich_errors.py --cases "isolated peak, sp1" --first-index 400 --realizations 1600 --output /tmp/sp1.json
```

## Near-coalescent peaks and warnings

The pair is 0.4 points apart (150 realizations, `sp2`). Conditional amplitude errors
stay calibrated (0.94–1.10 · 0.68 · 0.95). The nonlinear errors do not.

| Realizations | n | Positions: scatter/SD · 68% · 95% |
|---|---:|---|
| All | 150 | 1.30–1.54 · 0.53 · 0.74 |
| With the uncertainty warning | 63 | (warned: errors withheld or flagged) |
| Without it | 87 | 1.08–1.36 · 0.65 · 0.82 |
| … with amplitude \|ρ\| < 0.9 | 37 | 1.55–3.33 · 0.37 · 0.67 |
| … with amplitude \|ρ\| ≥ 0.9 | 50 | 0.56–0.73 · 0.85 · 0.94 |

- **The badly covered fits are biased, not miscalibrated.**
  - The fit separates the peaks by more than they are: the median fitted separation
    is 1.9× the true one.
  - The separated shapes overlap less, so amplitude |ρ| drops and the local errors
    shrink around the biased solution.
  - No error formula can repair that.
- **Amplitude correlation is no warning criterion here.** Flagging |ρ| ≥ 0.9 would
  catch the 50 unwarned fits that actually over-cover, and miss the 37 that
  under-cover.
- **What the documentation says:** strongly overlapping or near-coalescent peaks
  can have unreliable local errors even when no warning fires
  ([outputs.md](../outputs.md)). The calibrated overlapping cases above show this
  is not true of overlap as such, and no boundary between the regimes is claimed.
- **No resolution-based warning yet.** A threshold on the fitted separation relative
  to the linewidth could miss fits that separate the peaks artificially, and flag
  well-resolved 2D pairs. Any such flag must first be shown useful in a separate
  study: candidate measures across separations, linewidths, amplitudes, SNRs and
  2D geometries, judged on false positives and missed biased or under-covering
  fits. It would be an advisory review flag, separate from covariance withholding.

## Scope

- Synthetic 2D clusters with the stated processing. Other apodizations are not
  modelled.
- The fits start at the truth, so other optima are not searched.
- Real-data calibration needs replicates (C6), and MCMC is unchanged.

```bash
uv run python tools/validate_sandwich_errors.py --output /tmp/sandwich-errors.json
```
