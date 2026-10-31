# Scientific outputs and analysis projects

GUI and CLI use one scientific model and `fit.export.write_completed_outputs`.
A `.peakfit` analysis saves working state and full experimental data; it does not
replace scientific exports. The current formats are scientific archive **2.0.0**
and analysis project **3.0.0**.

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
- Cluster/global statistics, classifications, optimizer termination diagnostics,
  noise, terminal correction revision, pass counts and provenance.
- Exact observation dtype/checksums, paired sample indices and **terminal corrections**, including unusable
  clusters. Corrections cannot be regenerated from the final model.
- Model/calibration/processing descriptors and a small sampled-shape reference
  against which a reconstructed evaluator is checked.
- Stable peak IDs, assignments, initial/fitted/adopted coordinate distinctions,
  exact source-plane identities, ordered experimental values and exclusions.

Large observations, sampled models and both residual representations are absent
from default exports. Published values are never re-fitted or re-solved to read
them. The small shape reference is retained for validation, rather than for size
reduction; its cost is approximately 0.12 MB including archive framing here.

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

`ResultsLoader(directory).load_summary()` reads native archives first and supports
older JSON summaries. Plotting and existing MCMC consume the same scientific
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
CLI records absent GUI adoption explicitly. Local exports contain their fitted
group, and unusable clusters retain diagnostics without fabricated predictions.

Readers inspect NPY headers/lengths before allocation, bound metadata to 16 MiB
and total expanded scientific payloads to 8 GiB, reject duplicate/unreferenced
members, and validate identities, dimensions, correction revisions and algebra.
Finite scalar numbers use round-trip precision; nonfinite scalars and array
values preserve their IEEE representation. Future intensity/MCMC products can use
an explicit result kind and named arrays without changing fit semantics.

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
and its exact starting positions with a reason. The GUI completion and both
exporters use the same coordinate-resolution function; no second scientific
position rule or GUI-specific writer exists. Manual local-preview application
has its existing stricter guards; without a recorded completion transition,
exports report `not_recorded` and no adopted tuple rather than guessing.

CLI uses the identical row schema and exact initial/fitted coordinates, with
`adoption_status: not_recorded`, `adopted_unfolded_ppm: None` and an explicit reason:
no working-coordinate transition was recorded. It does not invent editable GUI
state. All coordinates remain named typed arrays; the table is a validated
projection referencing the authoritative engine parameters, not another fitted
parameter authority. Readers reject mismatched coordinates/parameter mappings
and false adoption claims.

`context.export_state` is separate from the immutable outcome and the run-owned
coordinate evidence. It contains `scope` (`dataset`/`cluster`), `currentness`
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

## Unavailable and unreliable uncertainties

Schema `4.1.0` permits `null` entries in amplitude error/covariance arrays and
adds `analytical_evaluation.uncertainty_warning`. Nonlinear `standard_error`
already permits `null`. The reader continues to accept `4.0.0`; older readers
must be upgraded to read new `4.1.0` files. Missing errors stay unavailable when
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
errors are also withheld at active bounds or when one standard error crosses a
parameter bound. These conservative checks are **not** confidence intervals or
proof of identifiability. Fixed parameters may retain zero error because they
were not estimated.

Amplitude errors condition on the fitted nonlinear parameters. They do not
include uncertainty in positions or linewidths. As before, the exported scaled
amplitude errors inflate by `sqrt(reduced chi_squared)` when it exceeds one;
nonlinear errors use the supplied/estimated statistical noise without that
inflation. Degrees of freedom are `max(1, n_observations - n_fitted_parameters)`,
which treats every spectral sample as independent (zero-filling correlates them)
and counts the nominal `n_peaks × n_series` amplitudes rather than the effective
design rank; reduced chi-squared and that inflation inherit both approximations.
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
conditional covariance and errors, optimizer diagnostics/statistics, scientific
identities, starting/optimized/adopted coordinates, settings and provenance. They
also retain exact terminal corrections, paired sample indices, ordered source-plane
identities, observation dtype/SHA-256, calibration and a small component-basis
reference for evaluator validation. Terminal corrections are independent terminal
inputs and cannot be substituted with final neighboring models.

Projects do not retain extracted observations, sampled model predictions or dense
residual arrays. The bundled two-pass one-run project is approximately **7.12 MB**.
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
New projects use schema **5.0.0**. Earlier experimental project formats are rejected;
no embedded-project readers, conversion adapters or alternative project modes remain.

## Reading and migrating existing files

- Scientific NPZ **1.0.0** and **2.0.0** remain readable, including the first PR
  schema-1 context without later coordinate/adoption fields. Missing old facts
  remain unknown; new exports require the complete context. New writes use 2.0.0.
- Earlier `.peakfit` project formats are unsupported. Scientific exports remain
  independently readable. To continue an older analysis, retain its original
  spectrum and scientific outputs; open the spectrum and refit into a new project.
  No old project is silently rewritten or stripped of data.
- Existing JSON 4.0/4.1 readers and explicit exports remain supported. Native
  summaries mark unavailable dense evidence explicitly; they do not invent it.
- Legacy pickle state remains readable. `save_state` now explicitly stores terminal
  corrections. Old pickles that omitted them cannot recover lost values alone;
  native scientific evidence takes precedence when present. Reading such a pickle
  warns, and multi-pass MCMC refuses it when corrections are required. Refit or
  use its scientific evidence. New default exports no longer write a pickle.

The numerical domain moved from `fit.{final_outcome,completed,simulation,coordinates}`
to `engine.scientific`; storage lives in `io.scientific`, reconstruction in
`io.readers.continuation`, metadata in `io.output_metadata`. Update internal imports.
GUI, CLI and MCMC therefore share these modules without depending on higher-level
fit orchestration. This is not a new plugin registry or framework.

See [external-project validation](validation/external-projects.md),
[earlier scientific policy validation](validation/scientific-storage-policy.md), the
[original per-array audit](validation/scientific-storage-audit.md), and regression
fixtures generated by the actual older implementations in `tests/data/storage-compat/`.
