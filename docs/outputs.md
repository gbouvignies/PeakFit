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
inflation. For nonlinear uncertainty or multimodal fits, compare alternative
starts and inspect sampling/profile likelihoods; the local errors alone cannot
certify a scientific assignment. See [the overlap investigation](validation/overlapping-peaks.md).


## Linked and portable analysis projects

```text
Analysis.peakfit/
  manifest.json
  spectra/<identity>.npz             # portable export only: complete experimental cube
  runs/<run-id>/result.npz            # detailed scientific evidence
  runs/<run-id>/review.json           # submitted/working transition metadata
```

Standard **Save analysis** and **Save analysis as** write linked analyses. They
reference the original spectrum using its immutable content identity, calibration,
shape/dtype and recorded locations (including a relative location). Several
analyses can share one spectrum. Saving never modifies original input files.
Reopening reads and hashes the source, validates its scientific metadata and
checks exact source-plane/sample mappings and retained model references. Moving
analysis and source together preserves the relative link. If the source moves
separately, **Relink source** explicitly verifies the replacement; there is no
filesystem search or substitution of changed data.

**Export portable analysis** creates a separate analysis containing the complete
experimental cube losslessly in its original dtype, including observations outside
fitted masks. It uses the same serializer, reader, evidence and atomic publication
as linked Save. It does not switch the current analysis path or mark unsaved edits
saved. Full calculated grids are reconstructed. Portable reopen validates the
embedded checksums, NPY headers and scientific sample mappings and supports full
spectral inspection, adjustments and fitting without original files. Export needs
verified complete experimental data or an existing embedded cube, and a new target
directory. Standard Save of an opened portable analysis intentionally produces a
linked project; the embedded payload is removed only after successful publication.
Use Export portable analysis for another portable snapshot.

Both modes retain detailed sampled observations, exact terminal corrections,
historical predictions, fitted values/uncertainties, identities, provenance,
calibration, ordered planes and working settings/view state. This deliberately
chooses about **16.86 MB** for a bundled one-run linked project over the audit's
~7.11 MB source-dependent reconstruction alternative. It preserves source-free
observed/model/residual numerical review and historical prediction bits without
assuming future evaluator arithmetic is identical. It does not duplicate the full
cube. Portable export adds **68.39 MB**, totaling about **85.25 MB**. These estimates
refer to one run; bounded history adds per-run evidence.

Missing, changed or unreadable linked sources retain numerical review and export;
full spectral views, peak adjustment and refitting are disabled until a source
verifies. Current/Outdated scientific status is independent of source availability.
Unsupported model evaluators retain numerical review; corrupt immutable project
evidence is rejected. No optimization or amplitude solve occurs on reopen.

Atomic manifest publication, PID locks, failure recovery, lazy historical loading,
ten unprotected full runs and the 1 GiB historical budget remain. Active/protected
runs are exempt from eviction, not reader limits. The full project stored-byte
limit is 8 GiB; project JSON is bounded at 512 MiB. Routine Save reuses immutable
run files. Accepted runs use linked `Recovery.peakfit` checkpoints with detailed
numerical evidence; recovery without the source supports numerical review, while
spectral work requires relinking. Existing cleanup/ownership rules remain.
Project schema **4.0.0** distinguishes this new save policy; scientific export
schema/defaults are unchanged. Older PeakFit readers do not read schema 4.

## Reading and migrating existing files

- Scientific NPZ **1.0.0** and **2.0.0** remain readable, including the first PR
  schema-1 context without later coordinate/adoption fields. Missing old facts
  remain unknown; new exports require the complete context. New writes use 2.0.0.
- Existing project readers for **1.0.0/1.1.0/2.0.0/3.0.0** remain available.
  New Save writes linked **4.0.0**, reusing immutable evidence without requiring a
  missing source. Portable export requires the complete experiment; fitted masks
  cannot recreate unavailable observations outside them. No new compatibility
  adapter or reconstruction layer was introduced.
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

See [linked/portable validation](validation/project-portability.md),
[earlier scientific policy validation](validation/scientific-storage-policy.md), the
[original per-array audit](validation/scientific-storage-audit.md), and regression
fixtures generated by the actual older implementations in `tests/data/storage-compat/`.
