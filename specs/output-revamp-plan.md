# Output Revamp Plan (Deep Review + Full Redesign)

## Goal

Make PeakFit output deterministic, compact, and easy to navigate by:

- writing only relevant information by default,
- assigning one clear purpose per file,
- using the right format per data shape (summary vs tables vs dense arrays),
- documenting intentional breaking changes clearly.

## What is wrong today

### Measured symptoms (real run sample)

Sample run: `examples/02-advanced-fitting/Fits/20260206_153332`

- `summary/analysis_report.md`: 23,771 lines
- `tables/parameters.csv`: 22,531 rows
- legacy amplitude table: 21,746 rows
- 96.5% of `parameters.csv` rows are amplitude-like `I*` rows (`21746 / 22531`)
- legacy residuals CSV has headers only (no residual data rows)
- empty placeholder output directories may be created without useful content

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

4. Legacy residual table export is effectively a stub.
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
   - In sample output, all amplitude-like rows in `parameters.csv` have `std_error=0`, while the separate amplitude table has non-zero errors for all rows.

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
   - `summary/fit.json` includes metadata + config payload
   - separate metadata JSON files repeat metadata and aggregate counts
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
7. Document public output breaks instead of preserving obsolete formats indefinitely.

## Target output contract (v2)

### Top-level layout

```text
<run_dir>/
├── README.md
├── summary/
│   ├── fit.json
│   └── report.md
├── tables/
│   ├── parameters.csv
│   ├── intensities.csv
│   └── shifts.csv
├── metadata/
│   └── fitting_state.pkl
```

### Why these formats

- `summary/fit.json`: canonical machine-readable summary object.
- `summary/report.md`: concise human report.
- `tables/*.csv`: user-facing and spreadsheet/pandas-friendly tables.
- Run metadata, fit statistics, MCMC diagnostics, and input checksums live in
  `summary/fit.json`; separate diagnostic JSON files would duplicate the
  canonical summary.

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
- bounded outlier/problem cluster table
- bounded parameter-to-check table (boundary hits, high relative error)
- bounded warnings section

Do not include:

- full amplitude series
- every parameter for every cluster
- decorative status glyphs that make logs harder to scan

## Output selection

- `summary/fit.json` when `json` is enabled.
- `tables/parameters.csv` and `tables/intensities.csv` when `csv` is enabled.
- `tables/shifts.csv` when `csv` is enabled and shift parameters are present.
- `summary/report.md` when `txt` is enabled.

## Migration

Breaking output changes are acceptable when they simplify the contract. Each break should be
documented with the old path or option, the replacement, and the reason the simpler output
model is better.

## Implementation plan

### Step 1: Introduce output plan object

- Resolve planned output paths with a plain mapping.
- Resolve files to write from:
  - `output.formats`
  - runtime data availability (shifts, optional reports).
- Create directories lazily from planned files.

### Step 2: Normalize result extraction

- In `FitResultsBuilder`, classify parameters by `param_id.label`.
- Move `I*` entries out of `lineshape_params`.
- Ensure category is set correctly (`AMPLITUDE` vs `LINESHAPE`).

### Step 3: Replace writer orchestration

- Remove hard-coded `_write_all`/`_write_minimal`.
- Implement table and summary writers behind the planned path mapping.
- Delete placeholder writers (or gate them behind actual data availability).

### Step 4: Redesign report generator

- New concise report sections.
- Add fixed, documented limits:
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
  - `output.include_residual_arrays`
- Remove/retire dead options after deprecation.

### Step 7: Tests and golden baselines

- Unit tests:
  - output planning matrix (format × data availability)
  - parameter classification and de-duplication
  - no-empty-directory guarantee
- Integration tests:
  - `--format` gating correctness
  - output directories contain only useful files
  - no duplicate JSON diagnostic files are written
- Golden:
  - new v2 goldens for `summary/fit.json` and `tables/*.csv`

## Definition of done

- `--format` and result data strictly determine written files.
- Output directory has no empty structural directories.
- `parameters.csv` excludes `I*` rows.
- `intensities.csv` is the single amplitude source.
- `report.md` remains compact on large datasets.
- Docs and code match exactly.
