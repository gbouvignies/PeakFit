# Output Revamp Plan (Deep Review + Full Redesign)

## Goal

Make PeakFit output deterministic, compact, and easy to navigate by:

- writing only relevant information by default,
- assigning one clear purpose per file,
- using the right format per data shape (summary vs tables vs dense arrays),
- preserving a compatibility path for downstream tooling.

## What is wrong today

### Measured symptoms (real run sample)

Sample run: `examples/02-advanced-fitting/Fits/20260206_153332`

- `summary/analysis_report.md`: 23,771 lines
- `parameters/parameters.csv`: 22,531 rows
- `parameters/amplitudes.csv`: 21,746 rows
- 96.5% of `parameters.csv` rows are amplitude-like `I*` rows (`21746 / 22531`)
- `statistics/residuals.csv` has headers only (no residual data rows)
- `diagnostics/`, `figures/`, and `legacy/` are created even when empty

### Functional inconsistencies (code-level)

1. Output format selection is ignored.
   - CLI accepts `--format` and stores `config.output.formats`
   - Writer path does not consume it
   - References:
     - `src/peakfit/cli/commands/fit.py:128`
     - `src/peakfit/cli/commands/fit.py:310`
     - `src/peakfit/fit/fitting.py:417`
     - `src/peakfit/io/writers/orchestrator.py:39`

2. Directory clutter by default.
   - All subdirectories are created unconditionally
   - References:
     - `src/peakfit/io/writers/orchestrator.py:24`
     - `src/peakfit/io/writers/orchestrator.py:33`

3. Statistics file has wrong cluster identifiers.
   - Uses enumerate index, not actual `cluster.cluster_id`
   - References:
     - `src/peakfit/io/writers/json.py:139`
     - `src/peakfit/io/writers/json.py:141`

4. `residuals.csv` is effectively a stub.
   - Header only, placeholder logic
   - References:
     - `src/peakfit/io/writers/csv.py:413`
     - `src/peakfit/io/writers/csv.py:432`

5. Parameters are duplicated across two channels and conflict semantically.
   - `FitResultsBuilder` includes amplitude `I*` entries in `lineshape_params`
   - Amplitudes are also exported in dedicated `amplitudes` table
   - References:
     - `src/peakfit/fit/builder.py:176`
     - `src/peakfit/fit/builder.py:385`
     - `src/peakfit/fit/builder.py:391`
     - `src/peakfit/io/writers/orchestrator.py:68`
   - In sample output, all amplitude-like rows in `parameters.csv` have `std_error=0`, while `amplitudes.csv` has non-zero errors for all rows.

6. Human report is overwhelmed by irrelevant volume.
   - Markdown includes every `lineshape_param`, including `I*` blocks
   - References:
     - `src/peakfit/io/writers/markdown.py:224`
     - `src/peakfit/io/writers/markdown.py:251`

7. Numeric formatting logic is inconsistent with the config contract.
   - `scientific_notation_threshold` semantics do not match implementation
   - Condition effectively triggers scientific-rounding path for most values
   - References:
     - `src/peakfit/io/writers/config.py:39`
     - `src/peakfit/io/writers/json.py:404`
     - `src/peakfit/io/writers/json.py:410`

8. Metadata and summary are duplicated with overlap.
   - `fit_summary.json` includes metadata + config payload
   - `run_metadata.json` repeats metadata and adds aggregate counts
   - References:
     - `src/peakfit/io/writers/json.py:62`
     - `src/peakfit/io/writers/json.py:107`

9. Config/documentation drift.
   - Docs claim format gating and optional outputs that are not wired in the writer path
   - References:
     - `docs/output_system.md:17`
     - `docs/output_system.md:48`
     - `docs/output_system.md:57`
     - `src/peakfit/fit/fitting.py:417`

## Revamp design principles

1. One source of truth per concept.
2. One file = one audience + one purpose.
3. Do not write placeholders.
4. Do not create empty directories.
5. Default output should be concise and analysis-ready.
6. Large/tabular dense data should be exported in table format, not embedded in summary objects.
7. Keep a migration bridge for existing consumers of `summary/fit_summary.json`.

## Target output contract (v2)

### Top-level layout

```text
<run_dir>/
├── README.md
├── manifest.json
├── summary/
│   ├── fit.json
│   └── report.md
├── tables/
│   ├── parameters.csv
│   ├── intensities.csv
│   └── shifts.csv
├── diagnostics/
│   ├── statistics.json
│   ├── residuals.npz          # only when requested
│   └── mcmc.json              # only for MCMC runs
├── metadata/
│   ├── run.json
│   ├── config.toml
│   └── fitting_state.pkl
└── compatibility/             # only when compatibility mode is enabled
    └── summary/fit_summary.json
```

### Why these formats

- `summary/fit.json`: canonical machine-readable summary object.
- `summary/report.md`: concise human report.
- `tables/*.csv`: user-facing and spreadsheet/pandas-friendly tables.
- `diagnostics/residuals.npz`: dense numeric arrays are compact and lossless in NPZ.
- `metadata/config.toml`: human-readable configuration snapshot.
- `manifest.json`: stable run index with schema version, file inventory, and quick metrics.

## Data inclusion policy

### `tables/parameters.csv`

Include only true model parameters:

- chemical shifts (`cs_*`)
- linewidths (`lw_*`)
- phase/J/global/shared terms
- uncertainty and bounds

Exclude amplitude series (`I*`) from this table.

### `tables/intensities.csv`

Include all per-plane intensities and their uncertainty:

- `cluster_id`
- `peak_name`
- `plane_index`
- `z_value`
- `intensity`
- `intensity_err`
- optional intervals (`ci_68_*`, `ci_95_*`)

### `summary/report.md`

Never dump full amplitude series.

Include:

- run headline (cluster count, peak count, reduced chi2)
- outlier/problem clusters
- top-N parameter warnings (boundary hits, high relative error)
- compact per-cluster stats table

### `diagnostics/statistics.json`

Use actual `cluster_id` values, not list position.

## Verbosity redesign (strict matrix)

- `minimal`
  - `summary/fit.json`
  - `tables/parameters.csv`
  - `tables/intensities.csv`
  - `metadata/run.json`
  - `manifest.json`

- `standard`
  - everything in `minimal`
  - `summary/report.md`
  - `tables/shifts.csv`
  - `diagnostics/statistics.json`

- `full`
  - everything in `standard`
  - `diagnostics/residuals.npz`
  - `diagnostics/mcmc.json` (if MCMC)
  - correlation exports (if available)

## Migration and compatibility

### Phase-in strategy

1. Add v2 writers and emit `manifest.json` with `schema_version: "2.0.0"`.
2. Add compatibility writer for old paths when `output.compatibility = "v1"` (default for one release).
3. Switch default to v2-only after one stable release.
4. Remove v1 compatibility after deprecation window.

### Compatibility adapter outputs

- `compatibility/summary/fit_summary.json` generated from v2 canonical data.
- Optional root-level symlink/copy strategy can be provided by config.

## Implementation plan

### Step 1: Introduce output plan object

- Add `OutputPlan` model in `fit` slice.
- Resolve files to write from:
  - `output.formats`
  - `output.verbosity`
  - runtime data availability (MCMC, residuals, figures).
- Create directories lazily from planned files.

### Step 2: Normalize result extraction

- In `FitResultsBuilder`, classify parameters by `param_id.label`.
- Move `I*` entries out of `lineshape_params`.
- Ensure category is set correctly (`AMPLITUDE` vs `LINESHAPE`).

### Step 3: Replace writer orchestration

- Remove hard-coded `_write_all`/`_write_minimal`.
- Implement table/summary/metadata/diagnostic writers behind `OutputPlan`.
- Delete placeholder writers (or gate them behind actual data availability).

### Step 4: Redesign report generator

- New concise report sections.
- Add configurable limits:
  - max clusters in body
  - max warnings shown
  - no amplitude series dumps.

### Step 5: Fix statistical and numeric consistency

- Use true `cluster_id` everywhere.
- Align formatting semantics:
  - numeric JSON values stay numeric
  - precision and scientific notation thresholds applied consistently.

### Step 6: Config cleanup

- Keep only wired fields or wire currently declared fields fully.
- Add:
  - `output.schema_version`
  - `output.compatibility`
  - `output.include_residual_arrays`
- Remove/retire dead options after deprecation.

### Step 7: Tests and golden baselines

- Unit tests:
  - output planning matrix (format × verbosity × data availability)
  - parameter classification and de-duplication
  - no-empty-directory guarantee
- Integration tests:
  - `--format` gating correctness
  - `fit_statistics` cluster IDs match `summary` cluster IDs
  - residuals file is either valid data or absent
- Golden:
  - new v2 goldens for `summary/fit.json`, `tables/*.csv`, `manifest.json`
  - compatibility goldens for v1 adapter during migration window.

## Definition of done

- `--format` and `verbosity` strictly determine written files.
- Output directory has no empty structural directories.
- `parameters.csv` excludes `I*` rows.
- `intensities.csv` is the single amplitude source.
- `report.md` remains compact on large datasets.
- `statistics.json` uses real cluster IDs.
- Docs and code match exactly.
