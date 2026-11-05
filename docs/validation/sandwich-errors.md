# Sandwich errors with PeakFit's fitting engine

Validation dates: 2026-11-04 (step 2) and 2026-11-05 (step 3, the published errors).
Rerun on 2026-11-05 after the tool's processing emulation adopted NMRPipe's
first-order phase sign ([decisions](../decisions.md)); the numbers below are from
that run. The earlier run used the mirrored sign for both noise and model, and
reached the same verdict.
This study is the gate of [correlated-noise.md](../correlated-noise.md) §11, and the
evidence for [ADR 0002](../adr/0002-processing-derived-sandwich-errors.md). Tool:
`tools/validate_sandwich_errors.py`. Results: [`sandwich-errors.json`](sandwich-errors.json),
plus [`sandwich-errors-sp1-replication.json`](sandwich-errors-sp1-replication.json)
for the two `sp1` cases.

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
| Isolated, `sp2` | positions | 3.01–3.41 · 0.25 · 0.45 | 1.02–1.07 · 0.65 · 0.94 |
| | linewidths | 2.76–2.90 · 0.28 · 0.50 | 1.02–1.04 · 0.66 · 0.95 |
| | conditional amplitudes | 3.27–3.40 · 0.22 · 0.44 | 0.97–1.01 · 0.69 · 0.96 |
| Overlapping, `sp2` | positions | 2.52–2.71 · 0.29 · 0.54 | 1.01–1.06 · 0.66 · 0.94 |
| | linewidths | 2.40–2.89 · 0.31 · 0.55 | 0.94–1.05 · 0.69 · 0.95 |
| | conditional amplitudes | 2.99–3.44 · 0.25 · 0.44 | 0.92–1.06 · 0.67 · 0.95 |
| Isolated, `sp1` | positions | 2.35–2.55 · 0.29 · 0.54 | **0.88**–1.01 · 0.69 · 0.97 |
| | linewidths | 2.37–2.59 · 0.27 · 0.55 | 0.97–1.03 · 0.68 · 0.95 |
| | conditional amplitudes | 2.73–2.85 · 0.27 · 0.50 | 0.95–0.99 · 0.69 · 0.96 |
| Overlapping, `sp1` | positions | 2.09–2.18 · 0.37 · 0.65 | 0.98–1.00 · 0.69 · 0.95 |
| | linewidths | 2.11–2.42 · 0.34 · 0.61 | 0.91–0.99 · **0.72** · 0.96 |
| | conditional amplitudes | 2.76–2.90 · 0.28 · 0.50 | 0.99–1.04 · 0.67 · 0.95 |
| Isolated, `no_apod` | positions | 1.93–1.99 · 0.40 · 0.70 | 0.98–1.01 · 0.70 · 0.95 |
| | linewidths | 1.90–1.96 · 0.39 · 0.69 | 0.97–0.99 · 0.69 · 0.96 |
| | conditional amplitudes | 1.99–2.10 · 0.37 · 0.65 | 1.00–1.05 · 0.66 · 0.94 |
| Overlapping, `no_apod` | positions | 1.77–1.99 · 0.40 · 0.69 | 0.92–1.03 · 0.69 · 0.95 |
| | linewidths | 1.87–1.98 · 0.41 · 0.70 | 0.96–1.01 · 0.70 · 0.95 |
| | conditional amplitudes | 1.95–2.12 · 0.38 · 0.66 | 0.98–1.06 · 0.66 · 0.95 |
| White control, R = I (150) | all | identical | identical, to 1.1 × 10⁻¹⁵ |

**Publication and withholding:**
- In every realization of every case, the published errors were the sandwich.
- The covariance gate withheld exactly the same realizations under both conventions.
  For duplicated peaks that was the nonlinear covariance in 132 of 150 realizations,
  and the amplitude covariance in 131.
- The bound-crossing rule acts on the published SE, so the larger sandwich SE crosses
  a bound more often: near coalescence, 115 parameter errors were withheld against
  82.

**Verdict:** every pre-registered criterion passes except two, in the recorded run.
- "Isolated, `sp1`", ¹H position: scatter / SD was 0.88, below 0.9, with coverage
  0.69 · 0.97.
- "Overlapping, `sp1`", linewidths: 68% coverage was 0.716, just above the 3-SE
  limit of 0.715, with scatter / SD 0.91–0.99: conservative, not optimistic.
- An independent replication with fresh seeds (realizations 400–1,999) passed every
  criterion for both cases: positions 1.01 and 1.01 (isolated), linewidths
  1.01–1.02 with coverage 0.68 · 0.94 (overlapping).
- The recorded values are therefore sampling fluctuations. With about 18 such
  checks at 400 realizations, the 0.9 bound lies only 2.9 SE from 1.

The recorded failures are kept, not rerun away. The replication reproduces with:

```bash
uv run python tools/validate_sandwich_errors.py --cases "isolated peak, sp1" "overlapping pair, sp1" --first-index 400 --realizations 1600 --output /tmp/sp1.json
```

## Near-coalescent peaks and warnings

The pair is 0.4 points apart (150 realizations, `sp2`). Conditional amplitude errors
stay calibrated (0.95–1.10 · 0.68 · 0.95). The nonlinear errors do not.

| Realizations | n | Positions: scatter/SD · 68% · 95% |
|---|---:|---|
| All | 150 | 1.43–1.85 · 0.48 · 0.71 |
| With the uncertainty warning | 69 | (warned: errors withheld or flagged) |
| Without it | 81 | 1.02–1.43 · 0.65 · 0.82 |
| … with amplitude \|ρ\| < 0.9 | 30 | 1.51–2.69 · 0.38 · 0.66 |
| … with amplitude \|ρ\| ≥ 0.9 | 51 | 0.54–0.72 · 0.82 · 0.92 |

- **The badly covered fits are biased, not miscalibrated.**
  - The fit separates the peaks by more than they are: the median fitted separation
    along each axis is 1.5× the true one.
  - The separated shapes overlap less, so amplitude |ρ| drops and the local errors
    shrink around the biased solution.
  - No error formula can repair that.
- **Amplitude correlation is no warning criterion here.** Flagging |ρ| ≥ 0.9 would
  catch the 51 unwarned fits that actually over-cover, and miss the 30 that
  under-cover.
- **What the documentation says:** strongly overlapping or near-coalescent peaks
  can have unreliable local errors even when no warning fires
  ([outputs.md](../outputs.md)). The calibrated overlapping cases above show this
  is not true of overlap as such, and no boundary between the regimes is claimed.
- **No resolution-based warning.** The [resolution-flag study](resolution-flag.md)
  tested eight candidate measures across separations, linewidths, amplitudes,
  SNRs, lineshapes, starts and 2D geometries. None met its pre-registered criteria
  for an advisory review flag. Separation significance discriminates risk well,
  but the fits it leaves unflagged still under-cover, so it cannot certify
  reliability.

## Scope

- Synthetic 2D clusters with the stated processing. Other apodizations are not
  modelled.
- Uniformly sampled, conventionally processed data only. Linear prediction and NUS
  reconstruction are outside this calibration
  ([correlated-noise.md §2.5](../correlated-noise.md#25-what-the-final-header-cannot-show)).
- The fits start at the truth, so other optima are not searched.
- Real-data calibration needs replicates (C6), and MCMC is unchanged.

```bash
uv run python tools/validate_sandwich_errors.py --output /tmp/sandwich-errors.json
```
