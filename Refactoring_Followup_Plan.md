# PeakFit Architecture Refactoring - Follow-up Plan

## Executive Summary

The first pass of refactoring successfully completed the major architectural changes. This follow-up plan addresses the remaining cleanup work, focusing on removing legacy files, decomposing remaining large modules, and ensuring CLI commands fully delegate to services.

**Estimated effort**: 2-3 weeks
**Priority**: High (prevents architectural drift)

---

## Current State Assessment

### Completed ✅

| Item | Before | After | Status |
|------|--------|-------|--------|
| `cli/app.py` | 993 LOC | 63 LOC | ✅ Complete |
| Reporter abstraction | N/A | 199 LOC | ✅ Complete |
| `io/output.py` ui dependency | Yes | No | ✅ Fixed |
| FitService facade | N/A | 198 LOC | ✅ Complete |
| PlotService facade | N/A | 187 LOC | ✅ Complete |
| Strategy pattern (optimizers) | N/A | 234 LOC | ✅ Complete |
| Import validation tool | N/A | 118 LOC | ✅ Complete |
| Tests | 309 | 332 | ✅ +23 tests |

### Remaining Work 🔄

| File | Current LOC | Target | Priority |
|------|-------------|--------|----------|
| `ui/style.py` | 987 | <400 | Medium |
| `core/diagnostics/plots.py` | 865 | DELETE | High |
| `cli/analyze_command.py` | 809 | <200 | High |
| `services/fit/pipeline.py` | 655 | <400 | Medium |
| `core/fitting/advanced.py` | 584 | <400 | Low |
| `cli/plot_command.py` | 553 | DELETE | High |
| `data/` package | N/A | DELETE | High |

---

## Phase 1: Legacy File Cleanup (Week 1)

**Objective**: Remove deprecated files that have been superseded by new implementations.

### Task 1.1: Remove `data/` Package

**Why**: Package only re-exports from `core/domain` and is not used anywhere.

**Verification**:
```bash
# Confirmed: No imports of peakfit.data found in codebase
grep -rn "from peakfit.data\|import peakfit.data" . --include="*.py"
```

**Steps**:
1. Delete `src/peakfit/data/` directory
2. Run tests to confirm nothing breaks
3. Update any documentation referencing `data` package

**Risk**: None - package is completely unused.

---

### Task 1.2: Remove `core/diagnostics/plots.py`

**Why**: Superseded by `plotting/diagnostics.py` which properly separates visualization from core.

**Current state**:
- `core/diagnostics/plots.py`: 865 LOC (OLD - visualization in core)
- `plotting/diagnostics.py`: 571 LOC (NEW - visualization in plotting layer)
- `core/diagnostics/metrics.py`: 357 LOC (NEW - pure computation)

**Steps**:

1. **Audit current usage**:
```bash
grep -rn "from peakfit.core.diagnostics.plots\|from peakfit.core.diagnostics import.*plot" . --include="*.py"
```

2. **Update imports** in any file still using old module:
```python
# OLD
from peakfit.core.diagnostics.plots import plot_trace, plot_corner

# NEW
from peakfit.plotting.diagnostics import plot_trace, plot_corner
```

3. **Verify function parity** - ensure all functions from old module exist in new location:

| Old Function | New Location | Status |
|-------------|--------------|--------|
| `plot_trace()` | `plotting/diagnostics.py` | ✅ |
| `plot_corner()` | `plotting/diagnostics.py` | ✅ |
| `plot_autocorrelation()` | `plotting/diagnostics.py` | ✅ |
| `save_diagnostic_plots()` | `plotting/diagnostics.py` | ✅ |

4. **Delete file**: `rm src/peakfit/core/diagnostics/plots.py`

5. **Update `core/diagnostics/__init__.py`** to not export plotting functions

6. **Run tests**: `pytest tests/`

---

### Task 1.3: Remove `cli/plot_command.py`

**Why**: Superseded by `cli/commands/plot.py`.

**Current state**:
- `cli/plot_command.py`: 553 LOC (OLD - monolithic)
- `cli/commands/plot.py`: 328 LOC (NEW - modular)

**Steps**:

1. **Audit current usage**:
```bash
grep -rn "from peakfit.cli.plot_command\|import peakfit.cli.plot_command" . --include="*.py"
```

2. **Check if `commands/plot.py` delegates to old module**:
```python
# If commands/plot.py still imports from plot_command.py, 
# we need to move the implementation first
```

3. **Move any remaining implementation** from `plot_command.py` to appropriate location:
   - Pure plotting logic → `plotting/` modules
   - CLI-specific helpers → `cli/commands/plot.py`

4. **Update services** to not depend on CLI modules:
```python
# BAD (in services/plot/service.py)
from peakfit.cli.plot_command import plot_intensity_profiles

# GOOD
from peakfit.plotting.profiles import plot_intensity_profiles
```

5. **Delete file**: `rm src/peakfit/cli/plot_command.py`

6. **Run tests**

---

### Task 1.4: Clean Up Legacy CLI Files

**Files to review**:
- `cli/fit_command.py` (751 bytes) - Check if still needed
- `cli/validate_command.py` (7164 bytes) - Check if delegated properly

**Decision tree**:
```
For each legacy file:
├── Is it imported by commands/* module?
│   ├── Yes → Keep as internal implementation
│   └── No → Delete
├── Does it contain logic that should be in services?
│   ├── Yes → Move to services, update imports
│   └── No → Keep in cli/
```

---

## Phase 2: CLI Delegation Completion (Week 1-2)

**Objective**: Ensure all CLI commands delegate to services, not directly to core.

### Task 2.1: Refactor `cli/analyze_command.py` (809 LOC)

This is the largest remaining CLI file. It contains:
- `load_fitting_state()` - Should use `FittingStateService`
- `run_mcmc()` - Should delegate to `MCMCAnalysisService`
- `run_profile_likelihood()` - Should delegate to `ProfileLikelihoodService`
- `run_correlation()` - Should delegate to `ParameterCorrelationService`
- `run_uncertainty()` - Should delegate to `ParameterUncertaintyService`

**Current structure** (809 LOC):
```python
# cli/analyze_command.py
def load_fitting_state(...)  # ~30 LOC - uses service ✅
def run_mcmc(...)            # ~200 LOC - UI-heavy, delegates to service ✅
def run_profile_likelihood(...) # ~200 LOC - UI-heavy
def run_correlation(...)     # ~100 LOC - UI-heavy
def run_uncertainty(...)     # ~100 LOC - UI-heavy
# + Rich table formatting helpers
```

**Target structure**:
```
cli/
├── commands/
│   └── analyze.py          # CLI interface only (~180 LOC)
└── _analyze_helpers.py     # Rich table formatters (~200 LOC)

services/analyze/
├── mcmc_service.py         # Already exists ✅
├── profile_service.py      # Already exists ✅
├── correlation_service.py  # Already exists ✅
└── formatters.py           # NEW: Service-level result formatting
```

**Refactoring steps**:

1. **Create `services/analyze/formatters.py`**:
```python
"""Result formatting for analyze services.

These formatters convert service results into display-ready structures.
They don't do Rich/console output - that stays in CLI layer.
"""

from dataclasses import dataclass
from typing import Any

@dataclass
class MCMCResultSummary:
    """Summary of MCMC results for display."""
    parameter_name: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    rhat: float
    ess: float
    converged: bool

def format_mcmc_results(analysis_result) -> list[MCMCResultSummary]:
    """Convert MCMCAnalysisResult to display summaries."""
    ...
```

2. **Simplify `cli/analyze_command.py`**:
```python
def run_mcmc(...):
    # 1. Load state (already uses service)
    state = load_fitting_state(results_dir)
    
    # 2. Run analysis (already uses service)
    result = MCMCAnalysisService.run(state, ...)
    
    # 3. Format for display (NEW service method)
    summaries = format_mcmc_results(result)
    
    # 4. Display using Rich (stays in CLI)
    _print_mcmc_table(summaries)
```

3. **Extract Rich table helpers to `cli/_analyze_formatters.py`**:
```python
"""Rich table formatters for analyze command output."""

from rich.table import Table
from peakfit.ui import console

def print_mcmc_table(summaries: list[MCMCResultSummary]) -> None:
    """Print MCMC results as Rich table."""
    table = Table(title="MCMC Results")
    table.add_column("Parameter")
    table.add_column("Mean ± Std")
    # ...
    console.print(table)
```

**Target**: `cli/analyze_command.py` → <200 LOC

---

### Task 2.2: Update `cli/commands/analyze.py`

Currently delegates to legacy `cli/analyze_command.py`:
```python
from peakfit.cli.analyze_command import (
    run_correlation,
    run_mcmc,
    run_profile_likelihood,
    run_uncertainty,
)
```

After refactoring, should use simplified functions or inline the logic.

---

## Phase 3: UI Module Decomposition (Week 2)

**Objective**: Split `ui/style.py` (987 LOC) into focused modules.

### Current `ui/style.py` Contents

Analyzing the 50+ methods, they group into:

| Category | Methods | ~LOC |
|----------|---------|------|
| **Logging** | `setup_logging`, `log`, `log_section`, `log_dict`, `close_logging` | 100 |
| **Branding** | `show_banner`, `show_version`, `show_run_info`, `show_footer` | 150 |
| **Headers** | `show_header`, `show_subheader`, `subsection_header` | 50 |
| **Messages** | `success`, `warning`, `error`, `info`, `action`, `bullet` | 100 |
| **Layout** | `spacer`, `separator`, `create_panel`, `print_panel` | 80 |
| **Progress** | `create_progress` | 30 |
| **Tables** | `create_table`, `print_summary`, `print_validation_table` | 100 |
| **Errors** | `show_error_with_details`, `show_file_not_found` | 80 |
| **Fit Display** | `print_fit_report`, `print_peaks_panel`, `print_data_summary`, `print_fit_summary` | 200 |
| **Cluster Display** | `create_cluster_status`, `print_cluster_info` | 100 |

### Target Structure

```
ui/
├── __init__.py          # Exports PeakFitUI, console
├── console.py           # Console instance, theme (50 LOC)
├── logging.py           # Logging setup (~100 LOC)
├── branding.py          # Banner, version, run info (~150 LOC)
├── messages.py          # success/warning/error/info (~100 LOC)
├── tables.py            # Table creation and printing (~150 LOC)
├── panels.py            # Panel creation (~100 LOC)
├── progress.py          # Progress bars (~50 LOC)
├── fit_display.py       # Fit-specific display (~200 LOC)
├── reporter.py          # ConsoleReporter (existing, 50 LOC)
└── style.py             # Backward compat re-exports (<50 LOC)
```

### Migration Strategy

1. **Create new modules** with extracted code
2. **Update `ui/__init__.py`** to import from new modules
3. **Keep `style.py` as facade** for backward compatibility:
```python
# ui/style.py (after refactor)
"""Backward compatibility - import from submodules."""
from peakfit.ui.console import console, PEAKFIT_THEME
from peakfit.ui.logging import setup_logging, log, log_section
from peakfit.ui.branding import show_banner, show_version
from peakfit.ui.messages import PeakFitUI
# ... etc
```

4. **Gradually update imports** across codebase to use new modules directly

---

## Phase 4: Service Pipeline Simplification (Week 2-3)

**Objective**: Reduce `services/fit/pipeline.py` from 655 LOC to <400 LOC.

### Current Structure

```python
# services/fit/pipeline.py (655 LOC)
class FitPipeline:
    @staticmethod
    def run(...):  # Entry point
        ...
    
    @staticmethod
    def _run_fit(...):  # Main implementation (~400 LOC)
        # 1. Setup logging
        # 2. Load spectrum
        # 3. Load peaks
        # 4. Create clusters
        # 5. Run fitting
        # 6. Refine
        # 7. Write output
        # 8. Save state
```

### Target Structure

Split into focused modules:

```
services/fit/
├── __init__.py
├── service.py           # FitService facade (existing, 198 LOC)
├── pipeline.py          # Orchestration only (~200 LOC)
├── loader.py            # Data loading (~100 LOC)
├── fitting.py           # Fitting logic (~150 LOC)
└── writer.py            # Output writing (~100 LOC)
```

### Refactoring Steps

1. **Extract `loader.py`**:
```python
"""Data loading for fit pipeline."""

from pathlib import Path
from peakfit.core.domain.spectrum import Spectra, read_spectra

def load_spectrum(
    spectrum_path: Path,
    z_values_path: Path | None,
    exclude_planes: list[int],
) -> Spectra:
    """Load and validate spectrum data."""
    ...

def load_peaks(peaklist_path: Path, spectra: Spectra) -> list[Peak]:
    """Load and validate peak list."""
    ...
```

2. **Extract `fitting.py`**:
```python
"""Fitting execution logic."""

from peakfit.core.fitting.strategies import get_strategy

def fit_all_clusters(
    clusters: list[Cluster],
    params: Parameters,
    noise: float,
    optimizer: str,
    refine_iterations: int,
) -> Parameters:
    """Fit all clusters and return updated parameters."""
    strategy = get_strategy(optimizer)
    for cluster in clusters:
        result = strategy.optimize(params, cluster, noise)
        # Handle refinement
    return params
```

3. **Extract `writer.py`**:
```python
"""Output writing for fit results."""

from peakfit.io.output import write_profiles, write_shifts

def write_all_outputs(
    output_dir: Path,
    clusters: list[Cluster],
    params: Parameters,
    z_values: np.ndarray,
    reporter: Reporter,
) -> None:
    """Write all output files."""
    ...
```

4. **Simplify `pipeline.py`**:
```python
"""Fit pipeline orchestration."""

class FitPipeline:
    @staticmethod
    def run(...):
        # Load
        spectra = loader.load_spectrum(...)
        peaks = loader.load_peaks(...)
        clusters = create_clusters(...)
        
        # Fit
        params = fitting.fit_all_clusters(...)
        
        # Write
        writer.write_all_outputs(...)
        
        # Save state
        StateRepository.save(...)
```

---

## Phase 5: Final Validation (Week 3)

### Task 5.1: Update Import Validator

Add stricter rules:
```python
FORBIDDEN_PATTERNS = [
    ("io", "ui"),
    ("core", "ui"),
    ("core", "cli"),
    ("core", "services"),
    ("infra", "ui"),
    ("infra", "cli"),
    # NEW: CLI should not import from core (except config)
    ("cli.commands", "core.fitting"),
    ("cli.commands", "core.algorithms"),
    ("cli.commands", "core.diagnostics"),
]
```

### Task 5.2: Add Pre-commit Hook

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: validate-imports
        name: Validate architectural imports
        entry: python tools/validate_imports.py
        language: python
        pass_filenames: false
```

### Task 5.3: Update Documentation

1. Update `docs/architecture/` with new structure
2. Add ADR for each major decision
3. Update `CONTRIBUTING.md` with import guidelines

### Task 5.4: Final Metrics Check

| Metric | Target | Validation |
|--------|--------|------------|
| Max file size | <500 LOC | `find . -name "*.py" -exec wc -l {} \; \| awk '$1 > 500'` |
| CLI fan-out | <5 packages | AST analysis |
| Layering violations | 0 | `python tools/validate_imports.py` |
| Test count | 350+ | `pytest --collect-only \| grep "test_"` |
| Test coverage | >80% | `pytest --cov` |

---

## Checklist

### Phase 1: Legacy Cleanup
- [ ] Delete `data/` package
- [ ] Delete `core/diagnostics/plots.py`
- [ ] Update imports to use `plotting/diagnostics.py`
- [ ] Delete `cli/plot_command.py`
- [ ] Move plotting logic to `plotting/` modules
- [ ] Update `PlotService` to use `plotting/` directly
- [ ] Clean up unused CLI files
- [ ] Run full test suite

### Phase 2: CLI Delegation
- [ ] Create `services/analyze/formatters.py`
- [ ] Refactor `cli/analyze_command.py` to <200 LOC
- [ ] Extract Rich formatters to `cli/_analyze_formatters.py`
- [ ] Update `cli/commands/analyze.py`
- [ ] Verify all CLI commands delegate to services
- [ ] Run full test suite

### Phase 3: UI Decomposition
- [ ] Create `ui/console.py`
- [ ] Create `ui/logging.py`
- [ ] Create `ui/branding.py`
- [ ] Create `ui/messages.py`
- [ ] Create `ui/tables.py`
- [ ] Create `ui/panels.py`
- [ ] Create `ui/progress.py`
- [ ] Create `ui/fit_display.py`
- [ ] Update `ui/__init__.py` exports
- [ ] Convert `ui/style.py` to re-export facade
- [ ] Run full test suite

### Phase 4: Pipeline Simplification
- [ ] Create `services/fit/loader.py`
- [ ] Create `services/fit/fitting.py`
- [ ] Create `services/fit/writer.py`
- [ ] Refactor `services/fit/pipeline.py` to <200 LOC
- [ ] Run full test suite

### Phase 5: Validation
- [ ] Update import validator with stricter rules
- [ ] Add pre-commit hook
- [ ] Update architecture documentation
- [ ] Verify all metrics meet targets
- [ ] Final test run with coverage

---

## Success Criteria

After completing this follow-up plan:

| Metric | Current | Target |
|--------|---------|--------|
| Files >500 LOC | 7 | 0 |
| Legacy files removed | 0 | 3+ |
| CLI import violations | 0 | 0 |
| Test count | 332 | 360+ |
| Documentation updated | Partial | Complete |

---

## Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| Week 1 | Phase 1 + Phase 2 start | Legacy files removed, analyze_command refactored |
| Week 2 | Phase 2 + Phase 3 | CLI delegation complete, UI decomposed |
| Week 3 | Phase 4 + Phase 5 | Pipeline simplified, validation complete |

---

## Risk Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking CLI behavior | High | Medium | Run integration tests after each change |
| Missing edge cases in new code | Medium | Medium | Review test coverage before deleting old files |
| Import cycles after reorganization | High | Low | Run import validator after each module move |
| Performance regression | Medium | Low | Benchmark before/after for fit command |

---

*Follow-up plan for PeakFit v2.0 architecture refactoring*
*Prepared: November 2025*
