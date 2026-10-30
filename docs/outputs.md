# PeakFit Output System

This document describes the output files produced by `peakfit fit` in the current codebase.

## Overview

PeakFit writes a structured results directory (default: `Fits` or a timestamped subdirectory when enabled). The content depends on selected output formats and available result data.

### Default Formats

The default output formats are:

- `json` → machine‑readable summaries
- `csv` → tabular data

Markdown reports are optional. Add `txt` to `[output] formats` or pass
`--format txt` to write `summary/report.md`. There is no TSV output; CSV is the
canonical tabular format.

## Output Directory Layout

```
<output_dir>/
├── README.md
├── summary/
│   ├── fit.json
│   └── report.md                 # optional: only if txt format is enabled
├── tables/
│   ├── clusters.csv              # status, statistics, and terminal provenance
│   ├── parameters.csv            # usable final nonlinear parameters
│   ├── intensities.csv           # usable final per-plane amplitudes
│   └── shifts.csv                # only when shift parameters are present
├── metadata/
│   └── fitting_state.pkl
```

### Notes

- `README.md` is generated for each run and summarizes the result status, files,
  and common next commands for that run directory.
- `metadata/fitting_state.pkl` is mutable continuation state for post-fit
  workflows. It is not used to reconstruct completed scientific values.
- `summary/fit.json` is schema `4.1.0`: each cluster is keyed by stable
  `cluster_id` and carries final classification, terminal correction revision,
  optimizer provenance, and either frozen analytical values or an unusable reason.
- `summary/report.md` is created only when `txt` is in `output.formats`; it is a
  bounded review report with fit-quality checks and key parameters, not a
  complete parameter export.
- `tables/parameters.csv` is intentionally lean: parameter identity, fitted
  value, uncertainty, fixed status, and optional units/bounds. Rich metadata
  such as parameter category and global/shared status belongs in
  `summary/fit.json`.
- CSV columns are ordered for reading: `peak_name` first, the main independent
  variable or parameter column next, and final classification/provenance columns
  (including `cluster_id`) last. `clusters.csv` retains one row even for an
  unusable outcome; numerical tables omit unusable rows.
- `tables/intensities.csv` may contain signed amplitudes. CEST plots preserve
  signed normalized intensities; CPMG plots use only points with positive
  `I/I0` ratios because `R2eff` is log-transformed.

## Optional Artifacts

These files are written only when explicitly enabled:

- `simulated.ftN` (at output dir root) when `output.save_simulated = true` and
  `nmrglue` is installed. It is projected from final nonlinear parameters and
  stored analytical amplitudes without solving amplitudes again.

## Configuration

### CLI

```
peakfit fit spectrum.ft2 peaks.list \
  --format json --format csv
```

### TOML

```toml
[output]
formats = ["json", "csv"]
save_simulated = false
include_timestamp = true
```

## Programmatic Access

The primary JSON entry point is:

- `summary/fit.json`

This file is the canonical machine-readable summary for downstream analysis.
Old schema `3.0.0` result documents are intentionally rejected; rerun the fit to
produce the unambiguous `4.1.0` outcome format.
Post-fit commands and `ResultsLoader` expect the run output directory
(`<output_dir>`), not the `summary/` subdirectory.

## Implementation Notes

- `build_output_plan()` resolves concrete files from requested formats and
  available data.
- Direct writer functions serialize JSON, CSV, optional Markdown, run-level
  README/state companions, and optional simulated spectra.
- The fixed layout is the output contract.
- Per-plane amplitudes are included in canonical JSON analytical evaluations and
  in `tables/intensities.csv`.

The in-memory completed-review contract also retains exact sampled coordinates,
original observations and terminal neighbor-correction arrays. These inputs are
not yet added to canonical output or a new persisted companion by this change.
Current JSON alone cannot reconstruct full model geometry or the original
numerical fitting problem; a correction revision alone does not supply correction
values. See [GUI design](gui-design.md#persistence-direction-and-compatibility) for
the evaluated metadata/array separation and the next writer/reader milestone.


## GUI dataset-fit export

**File → Export fit results…** on a whole-dataset result writes a new
`dataset-fit-<run_id>/` directory. It uses the same authoritative final-outcome
writers, canonical `summary/fit.json` 4.1.0 and CSV tables as the CLI and local
GUI exports. The staged/validated publication and refusal to overwrite an existing
destination are shared with local export below.

`summary/dataset_fit.json` is a provenance companion (`kind:
peakfit_gui_dataset_fit`, companion version 1.0.0). It records the reviewed
configuration, threshold, estimated noise, resolved model names, optimizer/correction
pass counts, source-file identities and checksums, original working peaks,
engine-ID/assignment mapping, ordered original plane indices/values and exact sampled
indices per cluster. Canonical rows use engine-safe IDs; this mapping preserves
duplicate assignments. Export always uses the retained run's original inputs,
including after working edits make it outdated. Export state is tracked by run ID.

This companion is not an analysis archive: it does not persist all model geometry,
sampled observations or terminal correction arrays needed for lossless reopening.
These export companions are not analysis projects; use `.peakfit` for reopening. No pickle file is written.

## GUI local-fit export

**File → Export fit results…** creates a new child directory in the selected parent:

```text
local-fit-<run_id>/
├── summary/
│   ├── fit.json                 # canonical PeakFit JSON 4.1.0
│   └── local_fit.json           # GUI local-fit companion 1.0.0
└── tables/
    ├── clusters.csv
    ├── parameters.csv
    ├── intensities.csv
    └── shifts.csv              # when usable shift parameters exist
```

The existing `write_fit_outputs` serializers consume the exact retained
`FinalFitOutcome`. There is no second fit, amplitude solve, assignment renaming,
whole-dataset state, cross-cluster correction, simulated spectrum cube, or pickle
continuation file. Scientific JSON keeps full float precision; CSV uses the same
columns as the CLI with 17-digit formatting. Unusable outcomes keep their actual
classification, reason and optimizer diagnostics, with absent numerical results
and empty numerical tables. Canonical null bounds/errors retain their established
meaning for unbounded/unavailable values.

The companion `summary/local_fit.json` has `kind: peakfit_gui_local_fit` and
`schema_version: 1.0.0`. It records the completed `run_id`, completion timestamp,
software version, isolated-cluster scope, absence of cross-cluster corrections,
original full array shape, spectral metadata, lineshape names, noise, absolute
contour threshold, effective configuration and optimizer settings, warnings and
execution diagnostics. `grid_indices` are paired arrays in the exact point order
of canonical shapes, model values and residuals. `planes` lists the original
zero-based index and experimental value for every plane, preserving duplicates
and nonmonotonic order. Canonical amplitude columns and CSV `plane_index` use
that same order.

Canonical `peak_names` and parameter prefixes remain internal engine identifiers
(e.g. `p_0`), not original assignments. Companion `peak_identities` explicitly maps
`engine_name` to stable `working_id` and original `assignment`; assignments may
repeat. `original_working_peaks` preserves the preparation's full segmentation
peak list, including peaks outside the fitted group and original unfolded ppm
coordinates. None of these fields is taken from subsequently edited GUI state.

`source_files` contains absolute original import paths and SHA-256 checksums only
when verified unchanged across file loading. These are imported sources, not a
claim that later working edits equal the imported peak list. Programmatically
supplied data may have no source records. Process-local spectrum identity is
omitted. Canonical metadata uses the recorded completion time/version and original
configuration; the companion supplies the GUI-specific scope and identity contract.

Export writes and validates all required files in a temporary sibling directory,
then takes an exclusive sibling lock (`.<directory-name>.export-lock`), checks
that the destination is still absent, and renames the complete tree into place.
The destination is never pre-created, so Windows need not replace an existing
directory. Existing paths are refused, including on repeated export of the same
result to the same parent. Choose a different parent for another copy.

The same-filesystem rename provides atomic visibility where the filesystem
supports it; this is not a power-loss durability guarantee. The sibling lock
coordinates PeakFit exporters only. Python has no portable atomic no-replace
directory rename: on POSIX an unrelated program could create an empty directory
between the final check and rename, and that empty directory could be replaced.
A populated results directory cannot be replaced by this directory rename.
Do not concurrently create the destination with another application.

Errors clean up staging and the export's own lock and do not mark the result
exported. A process crash can leave hidden staging or a stale sibling lock,
which may be removed after checking that no export is active. Export runs in
a dedicated short-lived thread while Qt continues processing events. Progress
cannot be dismissed once writing begins.

`ResultsLoader(directory).load_summary()` validates and reads the canonical
scientific result. Its existing minimal `load_fitting_state()` reconstruction is
available for usable values but is not the original sampled fitting state or a
continuation contract. The GUI cannot reopen this export, and it does not restore
a session, undo history, display settings, or apply fitted coordinates to working
peaks. CLI post-fit commands requiring `metadata/fitting_state.pkl` cannot use this
export as continuation state. These limitations do not affect downstream reading
of canonical parameters, amplitudes, statistics or the companion provenance.

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

The production `peakfit-editor` saves a version 1.1.0 analysis directory:

```text
Analysis.peakfit/
  manifest.json
  runs/<immutable-run-id>/
    summary/fit.json.gz  # lossless gzip of the validated scientific JSON document
    review.json          # exact inputs/bounds/freedom, geometry, identities, transition
    samples.npz          # paired native indices, observations, terminal corrections
```

The manifest declares dataset axes/processing/dtype/shape, ordered source planes,
working state/settings, view state, full/summary journal levels, protected run IDs,
active/historical/protected byte counts and SHA-256/byte lengths of every payload.
Each NPZ member has an explicit name, dtype and shape. Arrays load with
`allow_pickle=False`; full source and calculated-plane arrays are forbidden.
For this focused implementation, analytical shapes, model/residuals, parameters and amplitudes remain in
`summary/fit.json.gz`, with no second fitted-value authority for Full runs. Its JSON float precision
round-trips the float64 values; unavailable errors retain NaN semantics through
canonical nullable fields. No lower-precision derived CSV is used for reopening.

Spectrum identity includes original absolute path, usable relative path, separately
relinked location, SHA-256 algorithm/version, size, verification timestamp and cheap
file metadata. Read/hash stability is checked. Peak-list/plane-value paths are
provenance; reopening uses saved working peaks and ordered values, not those files.
Relative location is tried first, then recorded absolute candidates. Every successful
reopen/relink requires content equality, not size/mtime alone. Source resolution
never scans disks for replacement spectra.

Full-run evidence is immutable and reused on routine Save. New run directories are
flushed before atomically replacing the fsynced manifest. Old referenced payloads
survive failures before publication; unreferenced orphans are removed after success.
Save As stages beside a new destination then renames it. On POSIX, a stale lock from a dead
PID can be reclaimed. On Windows the error identifies the lock directory for
manual removal after confirming that no save is running. Do not edit project files or run two independent writers that
ignore the lock. Durability ultimately depends on the filesystem's atomic rename
and fsync behavior; network shares/Windows crash behavior have not been validated.

Reader safety limits are 512 MiB per JSON document and 8 GiB total compressed payloads and 8 GiB expanded per NPZ payload. These limits are distinct from the 1 GiB historical
retention budget: active/protected evidence is never evicted for that budget.
Larger workloads need a chunked format and an independently reviewed limit change.
The initial format supports the editor's one-file NMRPipe inputs, not multi-file
spectral directory trees. Unsupported models keep numerical review; corrupt
payloads are rejected. Legacy local/dataset export companions remain scientific
exports with their original scope and cannot be opened as analyses.


Project-internal scientific JSON uses deterministic gzip level 1 (no filename or
wall-clock gzip metadata). Decompression reproduces the validated document exactly;
SHA-256/size checks cover compressed bytes, and the 512 MiB JSON limit also applies
to expanded bytes before parsing. Existing plaintext project members remain readable.
CLI/export JSON and CSV writers do not need to change for this storage improvement.
Their formats are not constraints on the future internal authority: GUI v2 may
replace APIs, configuration, storage and export formats when scientific reliability
and product clarity justify it.

Only the active Full run is materialized on reopen. `runs` entries provide checked
locations for on-demand inspection. Before either count or byte-budget eviction,
the full record supplies the promised exact compact summary, including fitted
parameter values, uncertainties/bounds, original coordinates, initial freedom,
settings and diagnostics. Already demoted summaries remain unchanged through
subsequent save/reopen cycles. Full runs use their scientific record as authority;
Summary-only entries retain the smaller contract and cannot fabricate sample arrays.

Temporary session projects have an owner and an explicit lifetime. Accepted results
are published there before resident replacement. Save copies retained immutable
evidence and only then releases temporary storage; normal discard/close removes it
after workers finish. Interrupted POSIX sessions can be selected through **File →
Recover interrupted session…**, verified through the same reader and saved as a new
permanent project. Recovery locations appear in run diagnostics. They are not saved
analyses until the explicit permanent Save. On non-POSIX systems automatic claiming
is disabled rather than taking ownership of a possibly live session; an ordinary
Open analysis can still read the retained `Recovery.peakfit` explicitly.
