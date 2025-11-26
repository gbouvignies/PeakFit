# PeakFit Architecture Cleanup - Final Phase

## Context

The architectural refactoring of PeakFit has been largely completed through two passes:
1. **Initial refactoring**: CLI decomposition, Reporter abstraction, service layer creation
2. **Follow-up refactoring**: Legacy file removal, UI decomposition, pipeline simplification

However, analysis reveals remaining issues: **duplicate code**, **orphaned modules**, **broken imports**, and **legacy API wrappers** that should be cleaned up. Since backward compatibility is not required, we can fully clean up the codebase.

## Critical Issues

### 1. Broken Import in PlotService

`services/plot/service.py` imports from deleted `cli/plot_command.py`:

```python
# services/plot/service.py - BROKEN
from peakfit.cli.plot_command import plot_intensity_profiles  # File deleted!
from peakfit.cli.plot_command import plot_cest_profiles       # File deleted!
from peakfit.cli.plot_command import plot_cpmg_profiles       # File deleted!
```

**Fix**: Update to use `plotting/profiles.py` or `cli/commands/plot.py`.

### 2. Legacy CLI Wrapper Files

These files are thin wrappers that should be inlined:

| File | LOC | Purpose | Used By |
|------|-----|---------|---------|
| `cli/fit_command.py` | 30 | Wraps `FitPipeline.run()` | `cli/commands/fit.py` |
| `cli/validate_command.py` | 230 | Validation logic | `cli/commands/validate.py` |
| `cli/analyze_command.py` | 710 | Analysis runners | `cli/commands/analyze.py` |

**Action**: Inline or move content to appropriate locations:
- `fit_command.py` → inline into `commands/fit.py` or delete (just calls `FitPipeline.run`)
- `validate_command.py` → move to `services/validate/` or inline into `commands/validate.py`
- `analyze_command.py` → further decompose into services

### 3. Duplicate R-hat/ESS Implementations

Two separate implementations exist:

**`core/diagnostics/metrics.py`**:
```python
def _compute_ess(chains: FloatArray) -> float: ...  # Simple variance ratio
def _compute_rhat(chains: FloatArray) -> float: ...  # Basic Gelman-Rubin
```

**`core/diagnostics/convergence.py`**:
```python
def compute_ess(chains: FloatArray, method: str = "bulk") -> float: ...  # Full implementation with bulk/tail
def compute_rhat(chains: FloatArray) -> float: ...  # Same algorithm
```

**Action**: 
- Keep `convergence.py` as the canonical implementation
- Update `metrics.py` to import from `convergence.py`:
```python
from peakfit.core.diagnostics.convergence import compute_ess, compute_rhat

def compute_trace_metrics(...) -> TraceMetrics:
    ess = compute_ess(chains[:, :, param_index])
    rhat = compute_rhat(chains[:, :, param_index])
    ...
```

### 4. Duplicate UI Functions in `ui/style.py`

`ui/style.py` (948 LOC) still contains implementations that duplicate submodules:

| Function in `style.py` | Already in | Status |
|------------------------|------------|--------|
| `PeakFitUI.log()` | `ui/logging.py:log()` | Duplicate |
| `PeakFitUI.success()` | `ui/messages.py:success()` | Duplicate |
| `PeakFitUI.warning()` | `ui/messages.py:warning()` | Duplicate |
| `PeakFitUI.error()` | `ui/messages.py:error()` | Duplicate |
| `PeakFitUI.info()` | `ui/messages.py:info()` | Duplicate |
| `PeakFitUI.action()` | `ui/messages.py:action()` | Duplicate |
| `PeakFitUI.create_progress()` | `ui/progress.py:create_progress()` | Duplicate |

**Action**: Convert `PeakFitUI` class to delegate to submodule functions:
```python
class PeakFitUI:
    """Facade for backward compatibility - delegates to submodules."""
    
    @staticmethod
    def success(message: str, indent: int = 0, log: bool = True) -> None:
        from peakfit.ui.messages import success
        success(message, indent, log)
```

Or better: **Delete `PeakFitUI` class entirely** and update all imports to use submodule functions directly.

### 5. Orphaned `analysis/` Package

The `analysis/` package contains benchmarking/profiling tools:
- `analysis/benchmarks.py` (350 LOC)
- `analysis/profiling.py` (219 LOC)

These are development tools, not production code.

**Action**: Move to `tools/` directory (outside of main package) or `peakfit/_dev/`.

---

## Cleanup Plan

### Phase 1: Fix Broken Imports (Critical)

1. **Fix `services/plot/service.py`**:
   - Update imports to use `plotting/profiles.py` for figure creation
   - Or use `cli/commands/plot.py` functions

### Phase 2: Remove Legacy CLI Wrappers

1. **Delete `cli/fit_command.py`**:
   - It's just a 30-line wrapper around `FitPipeline.run()`
   - Inline into `cli/commands/fit.py`

2. **Move `cli/validate_command.py`**:
   - Create `services/validate/service.py` with validation logic
   - Update `cli/commands/validate.py` to use service
   - Delete `cli/validate_command.py`

3. **Decompose `cli/analyze_command.py`**:
   - Extract remaining logic to services
   - Keep only UI/Rich output in CLI layer
   - Target: <200 LOC or delete entirely

### Phase 3: Consolidate Diagnostics

1. **Merge R-hat/ESS implementations**:
   - Keep `core/diagnostics/convergence.py` as canonical
   - Update `core/diagnostics/metrics.py` to import from convergence
   - Remove duplicate `_compute_ess` and `_compute_rhat` from metrics.py

2. **Review `core/diagnostics/` structure**:
   - `metrics.py` - Data structures (TraceMetrics, etc.)
   - `convergence.py` - Convergence diagnostics (R-hat, ESS)
   - `burnin.py` - Burn-in determination
   - Clear separation of concerns

### Phase 4: Clean Up UI Layer

1. **Option A: Delete `PeakFitUI` class**:
   - Update all imports from `from peakfit.ui import PeakFitUI as ui` 
   - To `from peakfit.ui import success, error, info, ...`
   - Delete `ui/style.py` entirely

2. **Option B: Convert to thin facade**:
   - `PeakFitUI` methods delegate to submodule functions
   - Reduce `style.py` to <100 LOC

**Recommended: Option A** (cleaner, no legacy API)

### Phase 5: Relocate Development Tools

1. **Move `analysis/` package**:
   - Move to `tools/benchmarks/` (outside package)
   - Or rename to `peakfit/_dev/` (internal development)
   - Update any imports

---

## File Changes Summary

### Files to Delete

| File | LOC | Reason |
|------|-----|--------|
| `cli/fit_command.py` | 30 | Trivial wrapper, inline |
| `cli/validate_command.py` | 230 | Move to services |
| `cli/analyze_command.py` | 710 | Decompose to services |
| `ui/style.py` | 948 | Duplicates submodules |
| `analysis/__init__.py` | - | Move to tools/ |
| `analysis/benchmarks.py` | 350 | Move to tools/ |
| `analysis/profiling.py` | 219 | Move to tools/ |

**Total: ~2,500 LOC to remove/relocate**

### Files to Create/Update

| File | Action |
|------|--------|
| `services/validate/service.py` | Create from validate_command.py |
| `services/plot/service.py` | Fix broken imports |
| `core/diagnostics/metrics.py` | Remove duplicate functions |
| `cli/commands/fit.py` | Inline from fit_command.py |
| `cli/commands/validate.py` | Use new service |
| `cli/commands/analyze.py` | Use services directly |
| `ui/__init__.py` | Remove PeakFitUI re-export |
| `tools/benchmarks.py` | Move from analysis/ |

---

## Import Update Guide

After cleanup, imports should follow this pattern:

```python
# OLD (legacy)
from peakfit.ui import PeakFitUI as ui
ui.success("Done!")
ui.error("Failed!")

# NEW (direct)
from peakfit.ui import success, error, info, warning
success("Done!")
error("Failed!")
```

```python
# OLD (wrapper)
from peakfit.cli.fit_command import run_fit
run_fit(spectrum, peaklist, ...)

# NEW (direct)
from peakfit.services.fit.pipeline import FitPipeline
FitPipeline.run(spectrum, peaklist, ...)
```

---

## Validation Checklist

After cleanup:

- [ ] `python tools/validate_imports.py` passes
- [ ] All tests pass (`pytest tests/`)
- [ ] No imports from deleted files
- [ ] No duplicate function implementations
- [ ] `ui/style.py` deleted or <100 LOC
- [ ] `cli/` contains only `app.py`, `commands/`, `callbacks.py`, `models.py`
- [ ] `analysis/` package removed from main package

---

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Files in `cli/` (excluding commands/) | 7 | 4 |
| `ui/style.py` | 948 LOC | DELETED |
| Duplicate R-hat/ESS | 2 implementations | 1 implementation |
| `analysis/` in package | Yes | No (moved to tools/) |
| Total LOC removed | - | ~2,500 |

---

## Notes

- **No backward compatibility required**: All legacy wrappers can be removed
- **Tests may need updates**: Some tests might import from deleted modules
- **Documentation**: Update any docs referencing deleted files
- Run `grep -rn "from peakfit.cli.fit_command\|from peakfit.ui import PeakFitUI" . --include="*.py"` to find all usages before deleting
