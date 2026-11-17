# Plane exclusion as a fit-time choice

Validation date: 2026-11-17 (W2d, issue #213). Tool:
`tools/validate_plane_exclusion.py`, output [plane-exclusion.json](plane-exclusion.json).
Tests: `tests/test_plane_exclusion.py`. Behaviour and output format:
[outputs](../outputs.md#excluded-planes).

## Claim

Fitting a spectrum with planes K excluded gives the same result as fitting the same
spectrum with K physically removed: parameters and their σ, amplitudes, conditional,
profile and marginal σ, χ² (per cluster and global), noise estimate, classifications
and uncertainty warnings. Gate, fixed before the first run: relative difference at
most 10⁻¹² for every quantity, identical identities and classifications.

A run without exclusion must not change at all.

## Protocol

- **Bundled example.** `examples/02-advanced-fitting`, 131 planes, with the golden
  reference's arguments (`tools/write_golden_reference.py`, one refinement pass).
  K = {0, 40, 41, 130}, fixed in advance: the off-resonance reference plane, two
  adjacent offsets and the last plane.
    - *Excluded:* `peakfit fit … --exclude 0 --exclude 40 --exclude 41 --exclude 130`.
    - *Removed:* an NMRPipe file written by nmrglue from the original without K, with
      only `FDF3SIZE` (the plane count) changed, and the offset list without K. The
      tool checks that its planes are bit for bit the original's remaining planes,
      and its spectral axes identical.
- **Synthetic** (`tests/test_plane_exclusion.py`). Two clusters, three Gaussian peaks,
  six planes with repeated plane values; the two excluded planes are scaled by 40, so
  any leak into the fit would show. Both fits go through the CLI's `load_data`,
  `run_fit` and `write_fit_run_outputs`, with the noise estimated.

## Results

Bundled example ([record](plane-exclusion.json)): **passed, bit for bit.**

| Quantity | Largest relative difference | Bit-identical |
|---|---|---|
| Noise estimate | 0 | yes |
| Nonlinear parameters and their σ | 0 | yes |
| Amplitudes | 0 | yes |
| Conditional, profile and marginal σ | 0 | yes |
| χ² per cluster; global χ², χ²ᵣ, observations, dof | 0 | yes |
| Classifications, uncertainty warnings (122 clusters) | identical | |

The excluded run records `excluded_planes` = (0, 40, 41, 130) in the outcome, keeps
131 plane values in the archive, and marks those four planes' rows in
`intensities.csv`, without intensity.

Synthetic: every compared quantity agrees within 10⁻¹² (the test's gate).

## Noise estimate and clusters

The noise is estimated from the fitted planes only, as a dataset without the
excluded planes would be. On the example the estimate from every plane would be
11 922.54 instead of 11 921.83 (6 × 10⁻⁵ relative): small here, but it is not the
equivalent fit, and an excluded plane is usually excluded because it is bad. In the
synthetic test, the corrupted planes raise the all-plane estimate more than 1.5-fold.

Clusters are also segmented from the fitted planes. On the example, excluding K
gives 122 clusters where the full spectrum gives 121: plane 0, the off-resonance
reference, is the most intense plane and joins two segments. The removed-plane fit
finds the same 122.

## No exclusion, no change

`tools/scientific_digest.py` on main (`e4556a0`) and on this change, the example fit
without exclusion, one environment (macOS, Python 3.14):

- `outcome` and `samples` digests are identical.
- All 1 710 array members of main's archive are present under the same names and
  byte-identical. The only new member is `context.source_plane_values`.
- `metadata.json` differs only by `schema_version` (12.0.0 → 13.0.0), the new
  `outcome.excluded_planes` (empty) and `context.source_plane_values`, and the input
  paths of the two checkouts.
- `parameters.csv`, `clusters.csv` and `shifts.csv` are byte-identical;
  `intensities.csv` is identical up to its new `plane_excluded` column (`no` on
  every row).

The golden regression (`tests/integration/test_golden_regression.py`) passes
unchanged.

## Not covered

- The desktop fit still refuses exclusion; its controls and the project's record of
  them come with W4.
- Detailed evidence and continuation hold the fitted planes only; the measured
  values of an excluded plane are in the spectrum, not the archive.
