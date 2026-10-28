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
- `summary/fit.json` is schema `4.0.0`: each cluster is keyed by stable
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
produce the unambiguous `4.0.0` outcome format.
Post-fit commands and `ResultsLoader` expect the run output directory
(`<output_dir>`), not the `summary/` subdirectory.

## Implementation Notes

- `build_output_plan()` resolves concrete files from requested formats and
  available data.
- Direct writer functions serialize JSON, CSV, optional Markdown, run-level
  README/state companions, and optional simulated spectra.
- The fixed layout is the output contract.
- Per-plane amplitudes are exported only in `tables/intensities.csv`.


## GUI local-fit export

**File → Export fit results…** creates a new child directory in the selected parent:

```text
local-fit-<run_id>/
├── summary/
│   ├── fit.json                 # canonical PeakFit JSON 4.0.0
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
