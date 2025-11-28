# PeakFit Output System Instructions

## Overview

PeakFit produces structured outputs for downstream analysis (ChemEx, plotting, archival). The output system prioritizes usability, reproducibility, and compatibility with common tools.

## Output Directory Structure
```
output_dir/
├── fit_results.json      # Complete machine-readable results
├── parameters.csv        # All parameters (long format)
├── shifts.csv            # Chemical shifts (wide format)
├── intensities.csv       # Fitted intensities per peak/plane
├── report.md             # Human-readable summary
├── peakfit.log           # Run log
├── simulated.ft3         # Simulated spectrum (optional)
└── cache/
    └── state.pkl         # Internal state for MCMC continuation
```

## File Formats

### JSON (`fit_results.json`)

Complete structured output containing everything:
```json
{
  "schema_version": "1.0.0",
  "metadata": {
    "timestamp": "...",
    "software_version": "...",
    "git_commit": "...",
    "input_files": {...},
    "configuration": {...}
  },
  "dimensions": [
    {"label": "F1", "nucleus": "15N", "sf_mhz": 60.8, "is_pseudo": false},
    {"label": "F2", "nucleus": "1H", "sf_mhz": 600.0, "is_pseudo": false}
  ],
  "clusters": [...],
  "statistics": {...},
  "pseudo_axis": {...}
}
```

Design principles:
- Self-documenting with schema version
- Include all metadata for reproducibility
- Hierarchical: metadata -> dimensions -> clusters -> parameters/amplitudes
- Use arrays efficiently (not verbose repeated objects)

#### Schema Versioning

The `schema_version` field in `fit_results.json` follows semantic versioning:

- **Patch** (1.0.x): Bug fixes, no structure changes
- **Minor** (1.x.0): New optional fields added, backward compatible
- **Major** (x.0.0): Breaking changes to structure

When modifying JSON output:
- Adding optional fields: bump minor version
- Changing field names/types or removing fields: bump major version
- Document changes in CHANGELOG

A JSON schema file (`fit_results.schema.json`) is provided in `src/peakfit/io/schemas/` for validation.

### CSV Files

**`parameters.csv`** — Long format, all parameters:
```csv
cluster_id,peak_name,parameter,category,value,std_error,unit,is_fixed,is_global
423,2N-H,cs_F2,lineshape,6.867319,0.000012,ppm,False,False
423,2N-H,cs_F1,lineshape,115.631358,0.000152,ppm,False,False
423,2N-H,lw_F2,lineshape,25.756623,0.113798,Hz,False,False
423,2N-H,lw_F1,lineshape,31.222411,0.136570,Hz,False,False
```

**`shifts.csv`** — Wide format, easy for downstream tools:
```csv
peak_name,cs_F2_ppm,cs_F2_err,cs_F1_ppm,cs_F1_err
2N-H,6.867319,0.000012,115.631358,0.000152
4N-H,7.821281,0.000025,123.143265,0.000287
```

**`intensities.csv`** — For CEST/relaxation analysis:
```csv
peak_name,offset,intensity,intensity_err
2N-H,-12000.0,1225656.0,5344.0
2N-H,-1800.0,715756.8,5344.0
```

CSV conventions:
- Header row with column names
- Comment lines start with `#` (metadata at top)
- Use comma separator
- Quote strings containing commas
- No trailing commas

### Markdown (`report.md`)

Human-readable summary:
- Executive summary (clusters, peaks, overall χ²)
- Cluster table with fit quality flags
- Warnings for problematic fits
- Auto-generated, suitable for lab notebooks

### Log File (`peakfit.log`)

Plain text log:
- Timestamps
- Progress information
- Warnings and errors
- Suitable for debugging

Do NOT produce both `peakfit.log` and `logs.html`. Choose one format (plain text preferred).

## Parameter Naming

Use NMRPipe dimension convention:

| Parameter | 2D | 3D | 4D |
|-----------|----|----|-----|
| Chemical shift (direct) | `cs_F2` | `cs_F3` | `cs_F4` |
| Chemical shift (indirect 1) | `cs_F1` | `cs_F2` | `cs_F3` |
| Chemical shift (indirect 2) | — | `cs_F1` | `cs_F2` |
| Chemical shift (indirect 3) | — | — | `cs_F1` |
| Linewidth (direct) | `lw_F2` | `lw_F3` | `lw_F4` |
| Linewidth (indirect 1) | `lw_F1` | `lw_F2` | `lw_F3` |

**Never** embed peak names in parameter names. Use separate `peak_name` column/field.

## Writer Architecture
```python
class OutputWriter(Protocol):
    def write(self, results: FitResults, output_dir: Path) -> None: ...

class JSONWriter(OutputWriter): ...
class CSVWriter(OutputWriter): ...
class MarkdownReportWriter(OutputWriter): ...

class ResultsWriter:
    """Coordinates all output writers."""

    def __init__(self, config: OutputConfig):
        self.writers = [
            JSONWriter(),
            CSVWriter(),
            MarkdownReportWriter(),
        ]

    def write_all(self, results: FitResults, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for writer in self.writers:
            writer.write(results, output_dir)
```

## Design Rules

1. **No hidden files**: Use `cache/` directory, not `.peakfit_state.pkl`

2. **Lazy directory creation**: Only create directories when writing files to them

3. **Self-contained outputs**: No absolute paths in output files; all paths relative

4. **Reproducibility**: Include git commit, input checksums, full configuration

5. **Minimal redundancy**: Don't duplicate data across files; each file has a purpose

6. **Easy parsing**: Standard formats that load directly into pandas, R, Excel

## Legacy Format Support

Legacy `.out` files are opt-in only:
```bash
peakfit fit spectrum.ft2 peaks.list --legacy
```

When enabled, legacy files go in `legacy/` subdirectory:
```
output_dir/
├── ... (new outputs)
└── legacy/
    ├── 2N-H.out
    ├── 4N-H.out
    └── shifts.list
```

## MCMC-Specific Outputs

For MCMC runs, additional outputs:
```
output_dir/
├── ... (standard outputs)
├── mcmc/
│   ├── diagnostics.json    # R-hat, ESS, convergence status
│   └── chains.h5           # Posterior samples (HDF5)
└── figures/
    ├── trace_plots.png
    └── corner.png
```

## What NOT to Do

- Don't put internal state files (`.pkl`) in the main output directory
- Don't create empty directories
- Don't use `logs.html` alongside `peakfit.log`
- Don't use internal parameter names (`_2N_H_y0`) in outputs
- Don't produce 150,000+ line JSON files (consolidate amplitudes)
- Don't hardcode 2D assumptions in column names
