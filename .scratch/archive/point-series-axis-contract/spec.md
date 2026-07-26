# Point/series axis contract

Status: completed
Lifecycle: historical implementation record

Implementation completed on 2026-07-26 and is protected by current tests. This
record preserves the original rationale and acceptance criteria; current
architecture and operational behavior are authoritative in code and current
documentation.

## Problem Statement

PeakFit stores each peak cluster's sampled data as points by series, but two
statistics interfaces interpret the first axis as the number of series. When the
axis lengths differ, PeakFit reports the wrong number of amplitudes and fitted
parameters. This propagates to degrees of freedom, reduced chi-squared,
information criteria, amplitude uncertainty scaling, optimizer summaries, and
fit-review thresholds.

This is verified on both a deterministic 5-point, 3-series reproduction and the
persisted example state. The example's first peak cluster has 39 grid points and
131 series, is stored as `(39, 131)`, and currently reports 39 series.

## Solution

- Keep point-major cluster data as the canonical convention:
  `(n_points, n_series)`.
- Make `(n_points, n_series)` the only valid peak-cluster data representation.
  A one-series cluster is represented as `(n_points, 1)`.
- Make the peak cluster the authoritative interface for `n_points`, `n_series`,
  `n_observations`, and `n_amplitude_params`.
- Validate that point count agrees with the flattened spectral grid and that
  cluster data are exactly two-dimensional.
- Reject invalid shapes explicitly at construction boundaries. Do not transpose,
  reshape, infer, or normalize ambiguous input.
- Make optimizer and persisted statistics consume those named quantities rather
  than indexing an array axis independently.
- Preserve the numerical lineshape, variable-projection, MCMC, and reconstruction
  kernels. They already consume point-major cluster data correctly.
- Invalidate development-only state and output artifacts with version bumps
  rather than preserving representations or statistics from the incorrect
  contract.

The authoritative definitions are:

- `n_points`: number of sampled spectral grid positions in the peak cluster;
  axis 0 of point-major cluster data.
- `n_series`: number of planes in the series dimension; axis 1 of point-major
  cluster data.
- `n_observations`: number of scalar residual values, `n_points * n_series`,
  equivalently the cluster-data size when no observations are masked.
- `n_amplitudes`: one fitted amplitude for every peak and plane,
  `n_peaks * n_series`.
- `n_varied_parameters`: varying nonlinear lineshape and cluster parameters that
  apply to the peak cluster; computed amplitudes are excluded from this count.
- `n_fitted_parameters`: `n_varied_parameters + n_amplitudes`.
- `degrees_of_freedom`: `max(1, n_observations - n_fitted_parameters)`, preserving
  the existing divide-by-zero guard.
- `reduced_chi_squared`: the sum of squared noise-normalized residuals divided by
  `degrees_of_freedom`.

Whether all scalar observations should be treated as independent, and whether a
rank-deficient amplitude design should use effective rank rather than nominal
amplitude count, remain scientific/statistical questions outside the approved
axis correction.

## User Stories

1. As a PeakFit user, I want one amplitude per peak and plane, so that an
   intensity profile has the expected length.
2. As a PeakFit user, I want reduced chi-squared to use the actual point and
   series counts, so that cluster quality indicators are dimensionally correct.
3. As a PeakFit user, I want amplitude uncertainties to use the corrected
   reduced chi-squared, so that their scaling is internally consistent.
4. As a PeakFit user, I want fit-review warnings to use corrected statistics, so
   that axis lengths do not change which clusters are flagged incorrectly.
5. As a PeakFit user, I want simulated spectra to preserve the input
   series-major spectrum shape, so that corrected statistics do not alter the
   fitted model.
6. As a PeakFit user, I want stale development-only fitting-state files rejected
   explicitly, so that incompatible axis assumptions cannot enter a fit.
7. As a downstream consumer, I want corrected outputs to carry a new schema
   version, so that their statistics cannot be confused with development output
   produced by the axis bug.
8. As a downstream consumer, I want corrected `n_params`, degrees of freedom,
   reduced chi-squared, AIC, and BIC values to be documented as a behavioral
   correction, so that changed values are not mistaken for optimizer drift.
9. As a maintainer, I want point and series counts exposed through named
   properties, so that consumers do not infer domain meaning from raw axis
   numbers.
10. As a maintainer, I want invalid cluster shapes rejected near construction,
    so that axis errors fail before optimization or persistence.
11. As a maintainer, I want unequal point and series dimensions in tests, so that
    an accidental axis swap cannot pass silently.
12. As a maintainer, I want cluster creation tested against the same convention
    as the fitting kernels, so that producer and consumer contracts cannot drift.
13. As a maintainer, I want both optimizer summaries and persisted summaries to
    use the same amplitude count, so that the CLI and output files agree.
14. As a maintainer, I want MCMC checked as a correct existing consumer, so that
    the fix does not add an unnecessary transpose there.
15. As a maintainer, I want reconstruction checked at the explicit transpose
    back to series-major spectra, so that output grid order remains unchanged.
16. As a maintainer, I want historical compatibility values classified
    separately from scientific reference values, so that tests do not invent
    scientific provenance.
17. As a maintainer, I want a small rollback surface, so that the axis correction
    can be reverted without migrating stored peak-cluster arrays.

## Implementation Decisions

- Canonical peak-cluster data shape: `(n_points, n_series)`.
- Single-series peak-cluster data shape: `(n_points, 1)`.
- Peak-cluster interface: named point, series, observation, and
  amplitude counts derived from one validated data shape.
- Peak-cluster construction rejects arrays that are not two-dimensional,
  spectral grid arrays with inconsistent lengths, and data whose axis-0 length
  does not equal the spectral grid point count.
- Construction does not transpose, reshape, or silently normalize data.
- The design matrix remains `(n_peaks, n_points)`.
- Solved amplitudes remain `(n_peaks, n_series)`.
- Residual matrices remain `(n_points, n_series)` and flatten to
  `n_observations`.
- Full spectra remain series-major: `(n_series, *spectral_grid)`.
- Cluster creation and automatic-picker ROI extraction retain their existing
  transpose from series-major spectra to point-major cluster data.
- Simulation and reconstructed spectra retain the explicit transpose from the
  point-major model back to series-major spectra.
- Variable projection, basin hopping, and persisted result construction use the
  peak-cluster amplitude count rather than independently interpreting an axis.
- MCMC continues to read the second cluster-data axis as the series count.
- The fitting-state version is bumped and stale development states are rejected.
- The output schema version is bumped. Existing field names and types are
  retained, while corrected values may change for `n_params`, degrees of
  freedom, reduced chi-squared, AIC, and BIC.
- Amplitude values and their count do not change. Amplitude standard errors may
  change because the existing uncertainty policy scales them by
  `sqrt(reduced_chi_squared)` when reduced chi-squared is greater than one.
- Chi-squared, raw and normalized residuals, residual RMS/mean/standard
  deviation, nonlinear optimum, and reconstructed data are expected to remain
  unchanged.
- No golden reference value or numerical tolerance is changed merely to accept
  the correction.
- No ADR is created until the canonical convention is approved.

### Compatibility

- PeakFit has not distributed this redesigned version. Internal and output
  compatibility with its development artifacts is not required.
- Stale fitting-state payloads fail version validation rather than being
  heuristically migrated.
- Old JSON output remains readable only by tooling that accepts its old schema;
  PeakFit does not silently reinterpret it as corrected output.
- Historical result comparisons classify old statistics as development output
  from the axis bug, not scientific reference data.

### Migration

1. Introduce and test strict peak-cluster construction plus the named
   shape/count interface.
2. Remove obsolete one-dimensional cluster-data branches in directly dependent
   fitting, MCMC, and statistics paths.
3. Change the optimizer result count and result-construction statistics to
   consume that interface.
4. Bump state and output schema versions to invalidate development artifacts
   with the incorrect semantics.
5. Run the deterministic reproduction, existing unit suite, real-data workflow,
   static checks, and import checks.
6. Compare pre/post real-data outputs. Expect unchanged chi-squared and fitted
   values, with only the identified count-derived fields and uncertainty scaling
   changing.
7. Document the behavioral correction.

### Rollback

Revert the count-interface, validation, version, and statistics-consumer changes
as one coherent slice. No persisted state migration is required because
development-only stale artifacts are deliberately invalidated. Outputs produced
with corrected statistics remain historical artifacts and are not silently
rewritten.

## Testing Decisions

- Tests exercise public domain and result-building behavior rather than private
  helpers.
- The primary deterministic fixture uses one real Gaussian peak, five spectral
  points, and three series.
- The fixture constructs residuals orthogonal to the lineshape, giving a known
  chi-squared without fitting or stochastic behavior.
- The primary seam covers peak-cluster counts, flattened observations,
  `FitResult`, structured result construction, amplitude uncertainty scaling,
  and simulation.
- A producer test must verify that cluster creation converts series-major
  spectra to point-major cluster data.
- A merge test must verify that combining peak clusters concatenates points,
  not series.
- A single-series test must protect the canonical `(n_points, 1)`
  representation.
- Invalid-shape tests must prove that one-dimensional data, higher-dimensional
  data, inconsistent grid lengths, and mismatched data/grid point counts fail at
  construction.
- Optimizer integration tests must verify that variable projection and basin
  hopping pass the corrected amplitude count into their result types.
- An MCMC test must verify that amplitude names and blobs use the second
  point-major axis as the series count.
- Persistence tests must accept the new state version and reject an older
  development state version explicitly.
- Output tests must assert the new schema version.
- A real-data comparison must assert invariants rather than new unexplained
  constants: unchanged chi-squared and amplitude values, corrected parameter
  count from `sum(n_peaks * n_series)`, and algebraically derived degrees of
  freedom and reduced chi-squared.
- Existing strict lineshape tolerances remain unchanged.
- The existing golden chi-squared remains a broad compatibility baseline, not a
  scientific reference.

## Out of Scope

- Changing the scientific lineshape models.
- Changing optimization algorithms or tolerances.
- Deciding whether spectral samples are statistically independent after
  zero-filling.
- Replacing nominal fitted-parameter count with effective design-matrix rank.
- Redesigning result convergence metadata.
- Refactoring the broader fitting architecture.
- Replacing scalar and vector parameter representations.
- Redesigning automatic peak picking, MCMC, plotting, or the user interface.
- Migrating full spectra to point-major storage.
- Compatibility migration for development-only pickle or JSON artifacts.
- Creating an ADR for this localized correctness fix.
- Committing or pushing.

## Further Notes

### Verified evidence

- Cluster creation and automatic-picker ROI extraction transpose series-major
  spectra into point-major cluster data.
- Lineshape evaluation returns `(n_peaks, n_points)`.
- Linear least squares consumes `(n_points, n_series)` and returns
  `(n_peaks, n_series)`.
- Variable projection, common residual calculation, MCMC, cross-talk
  correction, and simulation already use point-major cluster data.
- Peak-cluster `n_series` and structured statistics currently use axis 0.
- The deterministic reproduction reports current values
  `(n_series, n_amplitudes) = (5, 5)`, `n_observations = 15`,
  `n_fitted_parameters = 7`, degrees of freedom `8`, and reduced chi-squared
  `7.5`; the canonical contract requires `(3, 3)`, `15`, `5`, `10`, and `6.0`.
- The persisted example's first cluster has shape `(39, 131)`, 39 grid points,
  and 131 computed amplitude parameters for its one peak, while its peak-cluster
  property reports 39 series.
- Applying only the count correction to the persisted example changes
  the global fitted-parameter count from 11393 to 22410 and the algebraic
  reduced chi-squared from approximately 2.295585 to 2.327831, with chi-squared
  held fixed.

### Compatibility evidence

- The historical baseline records `pre_simplification_parameter_rows = 22410`,
  exactly the fitted-parameter count implied by the canonical axis contract for
  the persisted example.
- Repository documentation classifies this row count as a historical
  compatibility value, not an independently validated scientific reference.

### Approved implementation constraints

- Point-major `(n_points, n_series)` is the only peak-cluster representation.
- No compatibility heuristic or silent normalization is permitted.
- Nominal `n_peaks * n_series` remains the amplitude count used by the existing
  degrees-of-freedom definition; effective-rank statistics remain out of scope.
- The existing amplitude-error scaling policy remains unchanged apart from
  receiving the corrected statistic.
- Development-only state and output compatibility may be broken explicitly.
