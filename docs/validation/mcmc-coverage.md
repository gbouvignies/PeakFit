# MCMC interval coverage (V6)

Validation date: 2026-11-04. Acceptance criterion V6 of
[the MCMC formulation](../mcmc-formulation.md) §10: on one peak and on overlapping
pairs, reported intervals cover the truth at their nominal rate, and near-coalescent
pairs produce flags rather than confident per-peak intervals.

## Set-up

`tools/validate_mcmc_coverage.py`; results in
[`mcmc-coverage.json`](mcmc-coverage.json).

- **Data:** 1D Gaussian peaks over four planes, noise σ = 0.05, true amplitudes of
  order 1–3. The default 25 Hz linewidth spans about 5 points.
- **Runs:** 200 noise realizations per scenario, each fitted by the ordinary pipeline
  (starting at the truth) and sampled with 16 walkers × 2,000 steps and automatic
  burn-in.
- **Scoring:** a provisional run reports no intervals. Exchanged peaks are withheld.
  Peaks not separately resolved (amplitude |ρ| ≥ 0.9) are scored in their own
  "(flagged)" rows.
- **Standard errors:** the ± values are binomial standard errors. They understate the
  uncertainty of the amplitude rows, whose four planes per realization are not
  independent.

## Results

Coverage of the 68% and 95% intervals. Status is out of 200 runs.

| Scenario | Status | Quantity | n | 68% | 95% |
|---|---|---|---:|---|---|
| One peak | 200 passed | position | 200 | 0.665 ± 0.033 | 0.955 ± 0.015 |
| | | linewidth | 200 | 0.635 ± 0.033 | 0.950 ± 0.015 |
| | | amplitude | 800 | 0.660 ± 0.016 | 0.941 ± 0.008 |
| Resolved pair (10 points) | 187 passed, 13 provisional | position | 374 | 0.682 ± 0.024 | 0.952 ± 0.011 |
| | | linewidth | 374 | 0.687 ± 0.024 | 0.960 ± 0.011 |
| | | amplitude | 1,496 | 0.678 ± 0.012 | 0.952 ± 0.006 |
| Partially overlapping pair (4 points) | 186 passed, 14 provisional | position | 372 | 0.710 ± 0.024 | 0.962 ± 0.011 |
| | | linewidth | 372 | 0.675 ± 0.024 | 0.954 ± 0.011 |
| | | amplitude | 1,488 | 0.700 ± 0.012 | 0.956 ± 0.006 |
| Near-coalescent pair (1.5 points) | 1 passed, 199 provisional | flagged rows only | | see below | |

**Near-coalescent pair:**
- All 200 runs flag the amplitudes as not separately determined.
- 168 detect the peaks exchanging identity.
- 199 are provisional.
- The one that passed (realization 72) explored a single ordering. Every interval it
  reported for the pair carries the "not separately resolved" caution.
  - Its positions and linewidths covered the truth 0/2 times at 68% and 1/2 at 95%.
  - Its per-peak amplitudes covered it 2/8 at 68% and 5/8 at 95%.
  - Its summed amplitudes covered it 2/4 at both levels.

## Reading

- **Separated cases:** one peak, and pairs at 10 and 4 points, cover at their nominal
  rates within about two standard errors. The largest deviation is the partially
  overlapping pair's amplitudes at 68%: 0.700, or +1.7 SE, before allowing for the
  correlation between planes.
- **Provisional rate:** about 7% of the pair runs are provisional, mostly because
  2,000 steps leave the kept chain under 50 τ_max. They report no intervals, so
  their exclusion is by design, not selection on the truth.
- **Near coalescence:** these runs produce the flags in every case. The single run
  that passed its convergence checks reported cautioned intervals that undercover.
  This illustrates the limit stated in §7: one ensemble started at the fit can miss
  a second mode, and no chain-only check reveals it.
- **Policy:** strong amplitude correlation alone is not evidence that positions and
  linewidths are unidentifiable, so those intervals stay visible with the caution.
  Withholding remains reserved for exchanges and demonstrated failures.

Reproduce:

```bash
uv run python tools/validate_mcmc_coverage.py --output /tmp/mcmc-coverage.json
```
