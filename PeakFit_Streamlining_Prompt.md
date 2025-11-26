# PeakFit Codebase Streamlining - Implementation Prompt

## Objective

Streamline and rationalize the PeakFit codebase by removing orphan code, consolidating small packages, and cleaning up the `plotting/` package structure. This is a cleanup refactoring with no functional changes.

## Context

PeakFit is a Python NMR spectroscopy lineshape fitting tool. The codebase has undergone significant architectural refactoring and now has some leftover orphan code and organizational inconsistencies that need to be cleaned up.

**Important**: All tests must pass after each phase. Run `pytest tests/` to verify.

---

## Phase 1: Clean Up `plotting/` Package

### 1.1 Delete Orphan Wrapper Files

The files in `plotting/plots/` (except `spectra.py`) are thin wrappers that are never used. The CLI (`cli/commands/plot.py`) calls `plotting/profiles.py` directly.

**Delete these files:**
- `src/peakfit/plotting/plots/intensity.py`
- `src/peakfit/plotting/plots/cest.py`
- `src/peakfit/plotting/plots/cpmg.py`
- `src/peakfit/plotting/plots/__init__.py`

**Verify no imports exist before deleting:**
```bash
grep -rn "from peakfit.plotting.plots.intensity\|from peakfit.plotting.plots.cest\|from peakfit.plotting.plots.cpmg" src/ tests/
```

### 1.2 Move `spectra.py` Up One Level

Move the Qt viewer from the nested location to the main plotting package:

```bash
mv src/peakfit/plotting/plots/spectra.py src/peakfit/plotting/spectra.py
rmdir src/peakfit/plotting/plots/
```

**Update the import in `cli/commands/plot.py`:**

Find:
```python
from peakfit.plotting.plots.spectra import main as spectra_main
```

Replace with:
```python
from peakfit.plotting.spectra import main as spectra_main
```

### 1.3 Delete `plotting/common.py`

This file contains CLI-specific helpers (`plot_wrapper`, `argparse` utilities) that are no longer used. The CLI has its own implementation.

**Verify no imports exist:**
```bash
grep -rn "from peakfit.plotting.common\|peakfit.plotting.common" src/ tests/
```

If there are imports from `plotting/plots/*.py` (which we're deleting), ignore them. If there are other imports, they need to be addressed first.

**Delete:**
- `src/peakfit/plotting/common.py`

### 1.4 Update `plotting/__init__.py`

Replace the current `__init__.py` with a complete public API:

```python
"""Plotting module for PeakFit.

This module provides visualization functions for NMR data analysis:
- MCMC diagnostic plots (trace, corner, autocorrelation)
- Profile plots (intensity, CEST, CPMG)
- Interactive spectrum viewer
"""

from peakfit.plotting.diagnostics import (
    plot_autocorrelation,
    plot_corner,
    plot_marginal_distributions,
    plot_correlation_pairs,
    plot_posterior_summary,
    plot_trace,
    save_diagnostic_plots,
)
from peakfit.plotting.profiles import (
    make_cest_figure,
    make_cpmg_figure,
    make_intensity_ensemble,
    make_intensity_figure,
    intensity_to_r2eff,
    ncyc_to_nu_cpmg,
)

__all__ = [
    # Diagnostics
    "plot_autocorrelation",
    "plot_corner",
    "plot_correlation_pairs",
    "plot_marginal_distributions",
    "plot_posterior_summary",
    "plot_trace",
    "save_diagnostic_plots",
    # Profiles
    "make_cest_figure",
    "make_cpmg_figure",
    "make_intensity_ensemble",
    "make_intensity_figure",
    "intensity_to_r2eff",
    "ncyc_to_nu_cpmg",
]
```

### 1.5 Run Tests

```bash
pytest tests/ -v
```

Fix any import errors that arise.

---

## Phase 2: Merge `infra/` into `io/`

The `infra/` package contains only one file (`state.py`, 51 LOC) and should be consolidated with `io/`.

### 2.1 Move State Repository

```bash
mv src/peakfit/infra/state.py src/peakfit/io/state.py
```

### 2.2 Update All Imports

Find all imports of `peakfit.infra.state` or `peakfit.infra`:

```bash
grep -rn "from peakfit.infra\|import peakfit.infra" src/ tests/
```

Update each occurrence:

**From:**
```python
from peakfit.infra.state import StateRepository
from peakfit.infra import StateRepository
```

**To:**
```python
from peakfit.io.state import StateRepository
from peakfit.io import StateRepository
```

### 2.3 Update `io/__init__.py`

Add the StateRepository export:

```python
"""I/O module for PeakFit.

Handles file operations including:
- Configuration file loading/saving (TOML)
- Result file output
- Fitting state persistence
"""

from peakfit.io.config import generate_default_config, load_config, save_config
from peakfit.io.output import write_output_files, write_shifts_list
from peakfit.io.state import StateRepository

__all__ = [
    "generate_default_config",
    "load_config",
    "save_config",
    "write_output_files",
    "write_shifts_list",
    "StateRepository",
]
```

### 2.4 Delete `infra/` Package

```bash
rm src/peakfit/infra/__init__.py
rmdir src/peakfit/infra/
```

### 2.5 Update Main Package `__init__.py`

If `src/peakfit/__init__.py` references `infra`, remove that reference.

### 2.6 Run Tests

```bash
pytest tests/ -v
```

---

## Phase 3: Relocate `analysis/` Package

The `analysis/` package contains development/benchmarking tools that should not be part of the production package.

### 3.1 Create Tools Directory

```bash
mkdir -p tools/analysis
```

### 3.2 Move Files

```bash
mv src/peakfit/analysis/benchmarks.py tools/analysis/benchmarks.py
mv src/peakfit/analysis/profiling.py tools/analysis/profiling.py
```

### 3.3 Update Imports in Moved Files

The moved files may import from `peakfit`. Since they're now outside the package, these imports should still work if PeakFit is installed. Add a note at the top of each file:

```python
"""Performance benchmarking utilities for PeakFit development.

Note: This module is located outside the main package in tools/analysis/.
It requires PeakFit to be installed to run.
"""
```

### 3.4 Update CLI Benchmark Command

Check if `cli/commands/benchmark.py` imports from `analysis/`:

```bash
grep -n "from peakfit.analysis\|import peakfit.analysis" src/peakfit/cli/commands/benchmark.py
```

If it does, update the import path or inline the necessary code.

### 3.5 Delete Empty Package

```bash
rm src/peakfit/analysis/__init__.py
rmdir src/peakfit/analysis/
```

### 3.6 Update Main Package `__init__.py`

Remove any reference to `analysis` from `src/peakfit/__init__.py`.

### 3.7 Run Tests

```bash
pytest tests/ -v
```

---

## Phase 4: Final Validation

### 4.1 Run Full Test Suite

```bash
pytest tests/ -v --tb=short
```

### 4.2 Verify Import Validation

If there's a `tools/validate_imports.py` script, run it:

```bash
python tools/validate_imports.py
```

### 4.3 Check for Broken Imports

```bash
# Try importing the main package
python -c "import peakfit; print('OK')"

# Try importing key submodules
python -c "from peakfit.plotting import plot_trace, make_intensity_figure; print('OK')"
python -c "from peakfit.io import StateRepository, load_config; print('OK')"
python -c "from peakfit.services.fit import FitPipeline; print('OK')"
```

### 4.4 Update Documentation

Update `docs/dependency-map.md` to reflect the new structure:

```markdown
## Package Structure

```
peakfit/
├── cli/           - Command-line interface
├── core/          - Domain logic and algorithms
├── services/      - Application services
├── ui/            - Terminal output formatting
├── plotting/      - Visualization
│   ├── diagnostics.py  - MCMC diagnostic plots
│   ├── profiles.py     - Profile visualization
│   └── spectra.py      - Interactive Qt viewer
└── io/            - File I/O
    ├── config.py       - TOML configuration
    ├── output.py       - Result file output
    └── state.py        - State persistence
```

Removed packages:
- `infra/` - merged into `io/`
- `analysis/` - moved to `tools/analysis/` (development utilities)
- `plotting/plots/` - orphan code deleted
```

---

## Summary of Changes

### Files to Delete
| File | Reason |
|------|--------|
| `src/peakfit/plotting/plots/__init__.py` | Empty, package deleted |
| `src/peakfit/plotting/plots/intensity.py` | Unused wrapper |
| `src/peakfit/plotting/plots/cest.py` | Unused wrapper |
| `src/peakfit/plotting/plots/cpmg.py` | Unused wrapper |
| `src/peakfit/plotting/common.py` | Unused CLI helpers |
| `src/peakfit/infra/__init__.py` | Package merged into io/ |
| `src/peakfit/analysis/__init__.py` | Package relocated |

### Files to Move
| From | To |
|------|-----|
| `src/peakfit/plotting/plots/spectra.py` | `src/peakfit/plotting/spectra.py` |
| `src/peakfit/infra/state.py` | `src/peakfit/io/state.py` |
| `src/peakfit/analysis/benchmarks.py` | `tools/analysis/benchmarks.py` |
| `src/peakfit/analysis/profiling.py` | `tools/analysis/profiling.py` |

### Files to Update
| File | Changes |
|------|---------|
| `src/peakfit/plotting/__init__.py` | Complete public API exports |
| `src/peakfit/io/__init__.py` | Add StateRepository export |
| `src/peakfit/__init__.py` | Remove infra/analysis references |
| `src/peakfit/cli/commands/plot.py` | Update spectra import path |
| `docs/dependency-map.md` | Update architecture documentation |
| All files importing from `peakfit.infra` | Update to `peakfit.io` |

### Directories to Remove
- `src/peakfit/plotting/plots/`
- `src/peakfit/infra/`
- `src/peakfit/analysis/`

---

## Verification Checklist

After completing all phases:

- [ ] `pytest tests/` passes (all tests green)
- [ ] `python -c "import peakfit"` works
- [ ] `python tools/validate_imports.py` passes (if available)
- [ ] No references to deleted packages remain in codebase
- [ ] Documentation updated to reflect new structure
- [ ] Git shows only expected file changes (no unintended modifications)

---

## Notes

- **No functional changes**: This refactoring only reorganizes code, it does not change any behavior
- **Backward compatibility**: If any external code imports from the deleted/moved modules, it will break. This is acceptable as these were internal implementation details.
- **One phase at a time**: Complete each phase and run tests before moving to the next
- **Commit after each phase**: Make a separate commit for each phase for easy rollback if needed
