# Scientific outputs and analysis projects

GUI and CLI use one scientific model and `fit.export.write_completed_outputs`.
A `.peakfit` analysis saves working state and references its external spectrum; it
does not replace scientific exports. The current formats are scientific archive
**12.0.0**, fit JSON **9.0.0**, analysis project **15.0.0** and MCMC record **1.1.0**.

## Default scientific export

```text
results/
  scientific.npz
  summary/report.md
  tables/parameters.csv
  tables/intensities.csv
  tables/clusters.csv
  tables/shifts.csv                 # when chemical shifts are available
```

The default formats are `csv` and `txt` (the short Markdown report). The archive
is mandatory. It contains small structured metadata and named typed NPY arrays
in a deflate-level-1 ZIP, with `kind: peakfit_scientific_result` and
`evidence: results`. No pickle or dense-array JSON is written by default.

Retain exactly:

- Published nonlinear values, bounds, freedom, units and uncertainties; signed
  per-plane amplitudes, covariance and all reported amplitude errors.
- Per usable cluster, the **covariance factors** (`analytical_evaluation.covariance_factors`,
  [profile-uncertainty §3.1](profile-uncertainty.md)): the tensors M E_m
  (`shape_gain`) and σ²M R P⊥ E_m H⁻¹ (`cross_gain`), each L × L × n_θ, and σ²C_θ
  (`parameter_covariance`, n_θ × n_θ), or all three `null` where withheld. Their
  conditional block is the stored amplitude covariance, kept once. With the amplitudes
  they rebuild the amplitude covariance between any two planes
  (`covariance_between_planes`) without the spectrum or a refit.
- Cluster/global statistics, classifications, optimizer termination diagnostics,
  noise, terminal correction revision, pass counts and provenance.
- Per usable cluster, the **largest recorded residual**
  (`analytical_evaluation.largest_recorded_residual`): the maximum |recorded
  optimizer residual| / σ over the cluster's `statistics.n_observations` corrected
  fitting samples, computed once from the live residuals at finalization.
- Exact observation dtype/checksums, paired sample indices and **terminal corrections**, including unusable
  clusters. Corrections cannot be regenerated from the final model.
- Model/calibration/processing descriptors and a small sampled-shape reference
  against which a reconstructed evaluator is checked.
- Stable peak IDs, assignments, initial/fitted/adopted coordinate distinctions,
  exact source-plane identities, ordered experimental values and exclusions.

Large observations, sampled models and both residual representations are absent
from default exports; the largest recorded residual is the only residual summary
retained, so review measures need no source. Published values are never re-fitted or re-solved to read
them. The small shape reference is retained for validation, rather than for size
reduction; its cost is approximately 0.12 MB including archive framing here.

The covariance factors add **0.19 MB** to the bundled example's archive (6.81 →
7.00 MB, CLI; 121 clusters): 0.09 MB of compressed factors, the rest member framing.
Their size does not depend on the number of planes. The `intensity_err_marginal`
column adds 0.52 MB to `intensities.csv`. Older figures below predate both.

The bundled two-pass result directory is about **12.02 MB (CLI) / 12.06 MB (GUI)**, including CSV/report.
It exceeds the audit's 5.77 MB result-only estimate because exact corrections are
retained for continuation. The archive is about **6.86 MB (CLI) / 6.90 MB (GUI)**; the GUI also records adoption. Removing corrections
to meet the smaller number would discard irreducible parent-problem information.

## Detailed evidence and continuation

CLI `--detailed-evidence`, configuration `[output].detailed_evidence = true`, or
the GUI export picker's single **Include detailed evidence** checkbox adds exact
observations and historical sampled model values to the same archive schema
(`evidence: detailed`). It costs about **21.80 MB including CSV/report**, or
**16.64 MB for the GUI archive**. It is a portable terminal sample record, not a copy
of the full experimental spectrum.

Raw residuals are regenerated as `(observations - terminal_corrections) - model`;
normalized residuals are `raw.ravel() / noise`. Recorded model values, amplitudes,
uncertainties and statistics remain authoritative. Model reconstruction evaluates
the recorded descriptors and final values against the saved shape/model reference
with declared tolerances (shapes rtol 1e-10/atol 1e-12, model rtol 1e-10/atol 1e-9).
Reconstructed observations must match their recorded per-cluster dtype and byte checksum.
Reconstructed parent statistics are checked at rtol 1e-8/atol 1e-9. Unsupported or
drifting evaluators cannot continue a fit. Bit-exact reconstruction on one runtime
is evidence, not a cross-platform or future-library guarantee.

`ResultsLoader(directory)` reads only `scientific.npz`; `load_summary()` projects it
into the fit JSON schema. Plotting and MCMC consume the same scientific
representation. `load_fitting_state()` reconstructs the validated terminal parent:
compact archives need the SHA-256-verified source for observations; detailed
archives need neither the original spectrum nor peak/plane text files. The source
may be explicitly supplied to the Python `load_completed` interface or found at
the recorded path and bounded nearby locations. An unmatched source is rejected.
No fabricated one-point cluster is used for native continuation. Unusable parents
must be adjusted/refitted before continuation. Starting an MCMC analysis of this
parent is distinct from resuming an existing chain/RNG state; no new MCMC feature
is introduced.

```python
from pathlib import Path
from peakfit.io.scientific import read_scientific
from peakfit.io.readers.continuation import load_completed, continuation_state

outcome, samples, context = read_scientific(Path("results/scientific.npz"))
amplitudes = outcome.clusters[0].analytical_evaluation.amplitudes
indices = samples[0].grid_indices  # paired, never a Cartesian product
corrections = samples[0].neighbor_corrections
# Compact samples deliberately have no .data or sampled residual fields.
completed = load_completed(Path("results/scientific.npz"))
state = continuation_state(completed)
```

`FinalAnalyticalResult` owns published values and the evaluator reference.
`FinalAnalyticalEvaluation` adds sample predictions/residuals when evidence is
available. `FinalClusterInputs` carries mappings/corrections; `FinalClusterSamples`
also carries observations. Absence of evidence is explicit, never zeros or dummy
observations. All arrays read from archives are read-only.

## Other outputs and interface parity

Full-grid `simulated.npy` stays opt-in via `[output].save_simulated = true` in both
interfaces. It is float64, with shape/calibration/plane mappings in `scientific.npz`;
it adds about **146.49 MB** for this dataset and contains predictions, not data.
Explicit `--format json` remains available for older interoperability workflows;
it contains large numerical lists and can add about 77.71 MB. It is never the
scientific authority. Formats and simulation settings are shared; CSV uses
17-place formatting. There is one writer and one schema, not separate GUI/CLI
scientific implementations. The default has one human report instead of a second
README projection.

GUI exports stage a new directory and publish by same-filesystem rename;
existing destinations are refused. Failure cleans staging. This provides atomic
visibility, not a power-loss durability guarantee. Project saving has its stronger
flush/manifest transaction described below. Export-time Current/Outdated and
recorded GUI adoption remain context separate from the immutable numerical result;
CLI records absent GUI adoption explicitly. Unusable clusters retain diagnostics
without fabricated predictions. Exports from the retired GUI local-fit preview
contain only their fitted group and remain readable.

Readers inspect NPY headers/lengths before allocation, bound metadata to 16 MiB
and total expanded scientific payloads to 8 GiB, reject duplicate/unreferenced
members, and validate identities, dimensions, correction revisions and algebra.
Finite scalar numbers use round-trip precision; nonfinite scalars and array
values preserve their IEEE representation.

## MCMC records

Each `peakfit mcmc` run adds one immutable record to the fit's results directory:

```text
results/
  scientific.npz
  mcmc/20261104T120000123456Z-1a2b3c4d.npz
```

A record uses the archive's encoding (`metadata.json` plus NPY members, the same
reader limits) with `kind: peakfit_mcmc_record`. It is written to a hidden partial
file, flushed, published by a hard link and the directory flushed. The link appears
complete and fails if the name exists, in one atomic step, so an existing record is
never replaced, even by a concurrent writer; the filesystem must support hard links. It binds to its parent by the parent's run ID and `scientific.npz`
SHA-256, and stores each cluster's whole chain losslessly with complete summaries
and the parent cluster's reduced χ² ([mcmc-formulation.md](mcmc-formulation.md) §8).
The reader rejects another version,
a missing or extra field, a wrongly typed value, arrays inconsistent with the
recorded settings, and posteriors not seeded by the record's own seed.

## Coordinate provenance and adoption

`context.coordinate_table` in `scientific.npz` is version **1.0.0**, keyed by
stable `peak_id` under `peaks`, and includes the immutable run ID and ppm unit.
Each row records `engine_name`, assignment, cluster ID, ordered axis labels and
exact engine `parameter_names`, plus:

- `initial_unfolded_ppm`: exact submitted positions, never inferred from fitted values.
- `fitted_unfolded_ppm`: usable, valid final chemical shifts, or `None` for an
  unusable/unmappable tuple. No partially fitted tuple is substituted.
- `resolution_reason`: why fitted coordinates are unavailable, when applicable.
- `adopted_unfolded_ppm`, `adoption_status`, `adoption_reason`: the recorded
  completion transition, independently of later manual edits.

The GUI's automatic dataset completion records `adopted_fitted` for usable finite
positions, including usable non-converged fits. Uncertainty warnings do not veto
this automatic transition. An unusable or invalid tuple records `retained_initial`
and its exact starting positions with a reason. So does a usable peak whose
recorded fitted shape is near-collinear with another's (1 − |cos| ≤ 10⁻⁴ on the
fitting samples): its row keeps `fitted_unfolded_ppm` and records
`adoption_reason` “near-collinear fitted shapes with p_…”, recomputed by readers
from the recorded shapes. Runs recorded before that rule may show such peaks as
`adopted_fitted`, which remains valid. The GUI completion and both
exporters use the same coordinate-resolution function; no second scientific
position rule or GUI-specific writer exists. Without a recorded completion
transition, exports report `not_recorded` and no adopted tuple rather than guessing.

CLI uses the identical row schema and exact initial/fitted coordinates, with
`adoption_status: not_recorded`, `adopted_unfolded_ppm: None` and an explicit reason:
no working-coordinate transition was recorded. It does not invent editable GUI
state. All coordinates remain named typed arrays; the table is a validated
projection referencing the authoritative engine parameters, not another fitted
parameter authority. Readers reject mismatched coordinates/parameter mappings
and false adoption claims.

`context.export_state` is separate from the immutable outcome and the run-owned
coordinate evidence. It contains `scope` (`dataset`; `cluster` only in earlier local-fit exports), `currentness`
(`current`/`outdated`/`unknown`/`not_applicable`), and recorded `adoption_warnings`.
GUI actions supply the actual export-time Current/Outdated state. A direct GUI
API call without that state explicitly reports `unknown`; CLI reports
`not_applicable`. An Outdated export retains original scientific inputs and its
recorded completion transition, never later working edits, and never mutates the
outcome. Only this explicit state differs when re-exporting Current versus Outdated.

## Plane identity in standalone CSV

`intensities.csv` now contains both `plane_index` (zero-based retained fitting
column) and `source_plane_index` (zero-based original spectrum plane). These are
identities; `z_value` is the experimental value and may repeat. For example,
excluding source plane 1 from values `[2, 0, 2]` produces retained indices `[0, 1]`,
source indices `[0, 2]` and values `[2, 2]`. Both interfaces pass the actual source
mapping through the shared writer. Low-level projection calls omitting a mapping
assume the unexcluded identity mapping. This is an additive CSV column change.
`reduced_chi_squared`, beside `intensity_err`, is the cluster's normalized residual
diagnostic; it does not scale the error (see below). `noise_correlation`, in
`intensities.csv` and `parameters.csv`, is "modelled" when the errors account for
the processed noise correlation and "not modelled" when they assume independent
points (see below).

## Unavailable and unreliable uncertainties

Fit JSON permits `null` entries in amplitude error/covariance arrays and in
nonlinear `standard_error`, and always records
`analytical_evaluation.uncertainty_warning`. Missing errors stay unavailable when
loading results, rather than becoming zero.

CSV uses `unavailable` for missing nonlinear errors and empty amplitude-error
cells. The `uncertainty_warning` column accompanies all numerical rows and the
cluster table. `cs_F2_err` and `cs_F3_err` contain their respective standard
errors in ppm, independently of the fitted chemical shifts.

A finite model can have unidentifiable individual peaks. Such a model remains
available for residual inspection and export even when its covariance is
unavailable. Optimizer success/classification describes termination; it does
not certify peak identifiability. Inspect `uncertainty_warning`, residuals and
alternative fits before interpreting positions. The GUI displays this warning
and disables applying positions from a warned group; manual editing, Undo/Redo,
export and model visualization remain available.

Nonlinear errors are local, noise-based errors from the reduced VarPro
Jacobian, with parameter-column scaling before SVD. A rank-deficient or
numerically ill-conditioned joint covariance is withheld. Individual symmetric
errors are also withheld at an active bound (a finite bound within an absolute
10⁻⁷ of the value) or when one standard error crosses a parameter bound. The
"At bound" review warning in the CLI and GUI is a separate, interval-relative rule
(10⁻⁶ of the bound interval) and does not change uncertainties. These conservative checks are **not** confidence intervals or
proof of identifiability. Fixed parameters may retain zero error because they
were not estimated.

Amplitude errors condition on the fitted nonlinear parameters, one per peak for every
plane: they are *conditional* amplitude errors. They do not include uncertainty in
positions or linewidths, which the *total* amplitude uncertainty would; MCMC
estimates that ([mcmc.md](mcmc.md)), under its own independent-noise likelihood.

**Marginal σ.** `intensity_err_marginal`, in `intensities.csv`, is the first-order
total uncertainty of each absolute amplitude, per peak and plane: √Σ_ss, from the
stored covariance factors ([profile-uncertainty §3.3](profile-uncertainty.md)). It
includes the lineshape uncertainty and is never smaller than `intensity_err`. It is
the right error for one amplitude on its own. Its lineshape part is correlated
across planes, so do not use the column as independent per-plane errors of a
profile: that overstates downstream errors. It is empty where the factors are
withheld, and where round-off makes the reconstructed variance negative (seen for
near-coincident peaks, whose cluster carries the uncertainty warning). The CLI and the GUI write it with the same writer, and the GUI inspector
shows it for the selected peak on the current plane.

**Correlated noise.** Zero filling and apodization correlate the noise of
neighbouring points ([correlated-noise.md](correlated-noise.md)).
- **When it is modelled:** each fit derives that correlation, R, from the processing
  its NMRPipe header records.
  - A sine bell or no apodization, first-point scale, zero fill, the Bruker
    digital-filter correction when FT applied it, and first-order phase; the header
    must also pass a consistency check.
  - Nonlinear and amplitude errors are then the least-squares sandwich, for example
    σ√diag(G⁻¹ΦRΦᵀG⁻¹) for amplitudes.
  - It is exact under the processing assumptions, not necessarily for the actual
    spectrum.
- **Otherwise:** the run records "not modelled" with the reason, and the errors
  assume independent points: σ√diag((ΦΦᵀ)⁻¹) for amplitudes. That covers any other
  apodization, composite or digital-filter-adjusted windows (`SP -elb`, `-glb`,
  `-df`), an incomplete or inconsistent header, and input that is not NMRPipe.
- **Not visible in the header:** linear prediction, NUSZF extrapolation, NUS
  reconstruction, earlier windows and solvent filters leave a header that looks like
  ordinary processing, so the run is recorded as "modelled". A modelled noise
  correlation is then not a calibrated uncertainty: in synthetic tests with LP and
  with NMRPipe's IST, parameter scatter exceeded the published errors 1.5–6.6-fold
  ([correlated-noise.md §2.5](correlated-noise.md#25-what-the-final-header-cannot-show)).
  SMILE and other reconstruction methods were not tested.
- **What the noise model changes:** only the published errors. Fitted values, σ and
  reduced chi-squared are the same either way.
- **Where it is stated:** the scientific archive and fit JSON record the noise model.
  The CLI, the Markdown report, the GUI inspector and the profile note state it, and
  so does the CSV column above.
- **Validation:** with PeakFit's own fitting engine, the sandwich errors are
  calibrated for isolated and overlapping 2D peaks with `sp1`, `sp2` and `no_apod`
  lineshapes ([validation](validation/sandwich-errors.md)). Independent-point errors
  understated the scatter 1.8–3.4-fold there.

**Strongly overlapping or near-coalescent peaks: local errors can be unreliable
without a warning.** All these errors are local, and rest on linearizing the model
at the fit.
- Where peaks overlap strongly, the fit can separate them by more than they are,
  and then report smaller errors around that biased solution.
- In the validation, a pair 0.4 points apart, 69 of 150 fits carried the uncertainty
  warning. The other 81 did not, and their positions were covered only 0.65 · 0.82
  (68% · 95%), at worst 0.38 · 0.66. Neither the existing warnings nor amplitude
  correlation identify those fits.
- This is not a property of overlap as such: the overlapping pairs of the
  validation were calibrated. No separation or linewidth boundary between the two
  regimes has been established.
- For strongly overlapping peaks, do not take an absent warning as evidence that
  the errors are reliable; compare alternative starts or a profile likelihood.

**One noise convention.** Nonlinear errors, amplitude errors (`intensity_err`,
`amplitude_standard_errors`) and MCMC intervals all use the supplied or estimated
statistical noise σ (with its correlation, where modelled, for the analytical
errors). None is rescaled by the reduced chi-squared. Reduced
chi-squared is reported beside them, in `clusters.csv`, `intensities.csv`, the JSON
statistics and the GUI, as a normalized residual diagnostic.

**What reduced chi-squared is.** It is Σr²/(σ²·dof), computed from the residuals as
fitted, not whitened by the noise correlation. It measures misfit relative to the
noise level and serves as a heuristic review flag: the fit review, the CLI and the
terminal summary mark clusters above `HIGH_REDUCED_CHI_SQUARED` (5.0). It is not
a calibrated chi-squared goodness-of-fit test. Under correlated noise its
distribution is not χ²(dof)/dof, so no p-value or acceptance threshold follows from
it. Even with independent points it would only be approximate (see degrees of
freedom below).

A reduced chi-squared well above 1 means that σ, the lineshape model or the noise
assumptions do not describe the data. The causes affect the errors differently:
- σ may be underestimated, or the noise may vary locally (t1 noise near intense
  peaks, plane-dependent noise). The errors are then too small.
- The model may be inadequate, leaving systematic residuals. That mainly biases the
  fitted values; scaling the errors would not represent it.
- The noise may not be independent between points.

Inspect the residuals before deciding which applies. A reduced chi-squared near 1
does not validate the errors either.
- Correlated noise leaves it near 1, or lowers it: its expectation becomes
  tr((I − H)R)/(n − p). Its definition is unchanged.
- Where the correlation is not modelled, independent-point errors can be too small.
  In a synthetic test with lag-1 correlation 0.67, they were 1.66× too small
  ([validation](validation/sigma-uncertainties.md)).

Degrees of freedom are `max(1, n_observations - n_fitted_parameters)`, which treats
every spectral sample as independent and counts the nominal `n_peaks × n_series`
amplitudes rather than the effective design rank; reduced chi-squared inherits both
approximations.
For nonlinear uncertainty or multimodal fits, compare alternative starts and
inspect sampling/profile likelihoods; the local errors alone cannot
certify a scientific assignment. See [the overlap investigation](validation/overlapping-peaks.md).


## External-source analysis projects

```text
Analysis.peakfit/
  manifest.json
  runs/<run-id>/result.npz            # results, exact corrections, sample mappings
  runs/<run-id>/review.json           # submitted/working transition metadata
```

A `.peakfit` project always references an independent experimental spectrum. Save
and Save as never copy experimental spectra or simulated data into the project.
Several analyses can reference the same spectral file. There is no portable-project
mode, embedded cube or shared-asset registry. Full calculated spectra and residuals
are generated on demand.

Projects preserve exact nonlinear values/bounds/freedom/uncertainties, amplitudes,
conditional covariance and errors, covariance factors, optimizer diagnostics/statistics, scientific
identities, starting/optimized/adopted coordinates, settings and provenance. They
also retain exact terminal corrections, paired sample indices, ordered source-plane
identities, observation dtype/SHA-256, calibration and a small component-basis
reference for evaluator validation. Terminal corrections are independent terminal
inputs and cannot be substituted with final neighboring models.

Each run's `review.json` keeps its exact submitted starts as `initial_parameters`
and their provenance as `continuation`: `run_id` (the run that supplied continued
starts, or `null` for a first fit), the stable peak IDs `continued` from it and
`initialized` from configured defaults, and a `reason` when that run's model was
incompatible. Runs saved before this record existed read it as not recorded.

Projects do not retain extracted observations, sampled model predictions or dense
residual arrays; review measures (χ²ᵣ, sample count, largest recorded residual,
amplitude correlation) all come from retained scalars and reopen identically. The bundled two-pass one-run project is approximately **7.12 MB**.
Observed values are sampled from the verified source in the original order/dtype
and checked against their exact digests. Reconstruction uses saved fitted values
and amplitudes without optimization or an amplitude solve. Existing continuation
checks validate evaluator references (`rtol=1e-10`, `atol=1e-12`) and regenerated
terminal chi-squared (`rtol=1e-8`, `atol=1e-9`). Published results remain exact;
regenerated prediction bits are not guaranteed across platforms/versions.

Reopen validates immutable numerical payloads and the external spectrum's SHA-256,
calibration, shape/dtype and sample identities. Only recorded relative/absolute
locations are tried. Moving source and analysis together preserves the relative
link; separately moving the source requires explicit **Relink source**. Changed,
corrupt or unreadable source files never replace the scientific result. Numerical
parameters, uncertainties, amplitude profiles, diagnostics, coordinates and exact
terminal corrections remain available without the source. Experimental sample
inspection, spectral views and adjusted fitting are disabled until verification.
Current/Outdated status is independent of source availability.

Sample/model/residual reconstruction is temporary and requested by sample review,
detailed export or continuation. Whole-spectrum GUI projections use the existing
bounded on-demand evaluator. Unsupported evaluators retain numerical review;
failed scientific checks disable spectral work rather than silently refitting.
GUI and CLI still share the same scientific exporter: standard compact exports
and explicit detailed evidence are unchanged. Missing-source numerical review can
export compact results; detailed observations require relinking.

Atomic manifest publication, PID locks, cancellation/failure recovery, lazy history,
ten unprotected full result records and the 1 GiB historical budget remain. Active/
protected results are exempt from eviction, not reader limits. The project reader
limit is 8 GiB and project JSON is bounded at 512 MiB. Routine Save reuses immutable
run files. Recovery checkpoints use the same compact external-source format;
source-free recovery supports numerical review and relinking, not spectral work.
New projects use schema **15.0.0**. Earlier project formats are rejected;
no embedded-project readers, conversion adapters or alternative project modes remain.

## Reading and migrating existing files

- Only scientific archive **12.0.0**, fit JSON **9.0.0** and project **15.0.0** are
  read. Earlier versions are rejected with "Unsupported scientific result
  version", "Unsupported fit JSON schema version" or "Unsupported analysis
  version". Archive 11.0.0 and project 14.0.0 lack the covariance factors, which
  only a fit computes. Older ones record configuration keys (`[auto_peak]`, `fix_positions`,
  `fit_j_coupling`, `[parameters].from_file`, `[fitting].optimizer_seed`) or run metadata (`git_commit`,
  `completion_warnings`) that no longer exist. To
  continue an older analysis, open its spectrum and peak list and refit. When
  detailed evidence or a verified source is available, the stored largest recorded
  residual is checked against the regenerated residuals.
  No old project is silently rewritten or stripped of data.
- A results directory without `scientific.npz` is not read: JSON-only results and
  pickled fitting state (`metadata/fitting_state.pkl`) from earlier versions are
  no longer supported. Explicit `--format json` export is unchanged. Native
  summaries mark unavailable dense evidence explicitly; they do not invent it.

The numerical domain moved from `fit.{final_outcome,completed,simulation,coordinates}`
to `engine.scientific`; storage lives in `io.scientific`, reconstruction in
`io.readers.continuation`, metadata in `io.output_metadata`. Update internal imports.
GUI, CLI and MCMC therefore share these modules without depending on higher-level
fit orchestration. This is not a new plugin registry or framework.

See [external-project validation](validation/external-projects.md),
[earlier scientific policy validation](validation/scientific-storage-policy.md), the
and the [original per-array audit](validation/scientific-storage-audit.md).
