# Scientific outputs and analysis projects

GUI and CLI scientific exports use the same `CompletedFit` authority and
`fit.export.write_completed_outputs` implementation. A `.peakfit` analysis is
working-state persistence; it does not replace scientific exports.

## Scientific export contract

Every export contains `scientific.npz`, independently of selected presentation
formats. Schema **1.0.0**, kind **peakfit_scientific_result**, contains small
`metadata.json` and named numeric NPY members in a ZIP with deflate level 1.
NumPy arrays retain exact dtype, shape, sample order and floating-point bits,
including unavailable uncertainty NaNs. There is no pickle or numeric-list
projection. Standard NumPy/ZIP tools can inspect the file; no custom compression
or HDF5 dependency is required by the format.

```text
results/
  scientific.npz                 # complete lossless scientific record, always
  summary/fit.json                # optional json projection, schema 4.1.0
  summary/report.md               # optional txt report
  tables/parameters.csv           # optional csv tables
  tables/intensities.csv
  tables/clusters.csv
  tables/shifts.csv               # when shifts are available
  simulated.npy                  # optional full-grid float64 model
```

The archive retains:

- Final nonlinear values, bounds, freedom, units and uncertainties, owned once by
  each cluster. The global parameter table and cluster lookup are reconstructed.
- Analytical shapes, signed amplitudes, uncertainty/covariance arrays, exact model
  values and both residual representations. These are deliberately retained rather
  than recomputed: regeneration could change floating-point rounding.
- Classification, actual terminal optimizer diagnostics, correction revision,
  pass counts, statistical noise and cluster/global statistics.
- Exact paired native sample indices, original observations and terminal neighbor
  corrections, including unusable clusters.
- Source provenance/effective configuration, calibration for every dimension,
  ordered plane values and explicit original source-plane indices (including after
  CLI exclusions; duplicate/nonmonotonic values remain distinct columns),
  resolved per-peak geometry and scientific peak identity/assignment mappings.

Both interfaces use the same identity entry fields: `engine_name`, stable
`peak_id`, and original `assignment`. GUI peak IDs come from retained working
identities; CLI IDs come from unique submitted peak names. Assignments may repeat.

Archive metadata uses explicit array references (`array`, `dtype`, `shape`).
Arbitrary provenance dictionaries are wrapped as `mapping` to distinguish them
from array and nonfinite-scalar descriptors. Finite scalar JSON numbers retain
Python's round-trip precision; nonfinite scalars retain tagged hexadecimal IEEE-754 binary64 bits.
Readers reject unsupported versions, duplicate/extra members, malformed dtype,
shape or payload lengths, excessive expansion, inconsistent identities, correction
revisions or terminal algebra. Arrays restore read-only. Limits are 16 MiB for
scientific metadata and 8 GiB expanded per archive (including NPY headers).

```python
from pathlib import Path
from peakfit.fit.storage import read_scientific

outcome, samples, context = read_scientific(Path("results/scientific.npz"))
amplitudes = outcome.clusters[0].analytical_evaluation.amplitudes
plane_values = context["plane_values"]
indices = samples[0].grid_indices  # paired, never a Cartesian product
```

The common export contract is a completed scientific fit plus coordinate context
and operational metadata. Future intensity analyses or MCMC results can introduce
an explicit result kind and associated named arrays; they must not overload fit
fields or editor state. This change adds neither feature.

## Formats and interface parity

The default presentation formats remain `json` and `csv`. Add `txt` to
`[output].formats` or CLI `--format txt` for a bounded Markdown report.
Both interfaces honor the recorded output formats and `save_simulated` setting,
and both use 17-place CSV formatting. `scientific.npz` is lossless regardless of
presentation precision. Explicit JSON still contains numerical lists and costs
more to write; it is a derived interoperability export, never the project authority.
`ResultsLoader(directory).load_summary()` and existing plot/analysis commands
continue to consume `summary/fit.json` when JSON is selected.

GUI **Export fit results…** stages a new `dataset-fit-<id>` or `local-fit-<id>`
child directory, writes all selected scientific artifacts, then publishes by
same-filesystem rename under a sibling export lock. Existing destinations are
refused. A failure cleans staging and does not mark the result exported. The
rename provides atomic visibility, not a power-loss durability guarantee; the
lock coordinates PeakFit writers. CLI run directories additionally contain a
README and trusted optimizer continuation state for existing post-fit commands.
Those operational files are not alternative scientific outputs and are not
required to read the scientific record.

The GUI's former `local_fit.json` / `dataset_fit.json` companions are removed.
Their scientific geometry, identity, coordinate and provenance information lives
in the shared archive. Manual revision, live editable peak lists and undo history
remain project state. Scientific initial/fitted/adopted distinctions and explicit
export-time Current/Outdated status are retained as described below.
The separate GUI sample-only NPZ and summary-only journal JSON exports are also
removed. Use the shared retained-run export for complete samples and corrections.
Evicted Summary-only entries remain reviewable inside the project but cannot
produce a full scientific export; protect or export important runs before eviction.

Local exports contain only their fitted group; an incomplete/all-unusable result
retains its diagnostics and never fabricates a full model.

Optional simulation is now **`simulated.npy` in both interfaces**, replacing the
CLI-only float32 `simulated.ftN`. It evaluates retained geometry and amplitudes
without an optimizer or amplitude solve. Calibration and plane mapping are in
`scientific.npz`. Use `numpy.load(..., allow_pickle=False)`; consumers that need
NMRPipe must explicitly convert with the recorded calibration. This closes the
previous GUI/CLI simulation and report gaps without depending on a linked source
being available during numerical-only review.

Cross-interface regressions exercise the real CLI and GUI writer entry points
with the same completed result and options. They compare every archive metadata
and array member, CSV values, report semantics and simulated samples. Report
generation time and the explicit export-time GUI/CLI state are checked separately;
scientific coordinate/adoption differences are asserted rather than discarded.

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


## Linked-source analysis projects

The production editor writes project schema **2.0.0**:

```text
Analysis.peakfit/
  manifest.json
  runs/<immutable-run-id>/
    result.npz           # same scientific codec: outcome and retained samples
    review.json          # editor inputs, geometry, stable IDs and transition
```

No scientific JSON/gzip document or separate sample NPZ is written. The compact
review metadata owns original submitted values/bounds/freedom, initial coordinates,
working identities, effective settings, warnings and automatic position transition;
it does not own a second fitted-value representation. Full runs own fitted values
through `result.npz`. Only summary-only historical entries keep a compact numerical
projection after full evidence is evicted.

Earlier `.peakfit` versions are deliberately rejected; there is no migration or
compatibility reader. Export any needed science using the prior application before
upgrading, or rerun the fit. Existing explicit JSON exports remain readable by the
existing JSON tooling. Programmatic CLI writers now require a retained
`CompletedFit`; outcome-only objects cannot fabricate sample evidence.

The manifest records linked spectrum identity/calibration/shape/dtype, ordered
planes, working settings/view state, journal, protected runs, byte counts, and
SHA-256/lengths for every immutable payload. Projects never embed the whole
spectrum or calculated display cubes. Reopening validates payloads, materializes
only the active run, verifies the linked source by SHA-256, checks exact original
samples, and evaluates retained geometry without fitting. Missing or unsupported
sources/models preserve numerical review; corrupt scientific evidence is rejected.

Routine Save reuses immutable run files and writes working metadata only. New
payloads are flushed before atomic manifest publication; old referenced files
survive a failure. Save As stages beside a new destination. Full history remains
bounded by ten unprotected runs and a 1 GiB historical byte budget; active and
protected runs are exempt from eviction, not reader limits. Before eviction the
full record supplies its exact compact summary. Historical arrays load only when
requested and are released after inspection.

Accepted results are durably checkpointed in an owned `Recovery.peakfit` before
GUI adoption. Normal Save/close cleans owned scratch storage after workers finish;
interrupted POSIX sessions remain available through **Recover interrupted
session…** and require a permanent Save. No new recovery format is introduced.
Atomic saving, locks, source relocation/currentness and numerical-only workflows
retain their existing semantics. Project JSON is bounded at 512 MiB, total stored
evidence at 8 GiB, and scientific archives have their stricter limits above.

See [measured validation and limitations](validation/scientific-storage.md).
