# Legacy Code and Incomplete Migration Issues

**Found:** November 18, 2025
**Status:** Documented

## Issue 1: Outdated `peakfit-legacy` Documentation ❌

### Problem
The README.md documents a `peakfit-legacy` command that no longer exists:

```markdown
## Legacy CLI

The original CLI is still available as `peakfit-legacy`:

```bash
peakfit-legacy -s spectrum.ft2 -l peaks.list -o Fits -r 2 --pvoigt
```
```

### Evidence
- **Source code search**: No `peakfit-legacy`, `cli_legacy.py`, or related files found in `src/peakfit/`
- **Entry points**: No `peakfit-legacy` entry in `pyproject.toml`
- **CHANGELOG.md**: Confirms removal: "Legacy argparse CLI (`peakfit-legacy`)" listed under "Removed"

### Impact
- **User confusion**: Users may try to use `peakfit-legacy` which doesn't exist
- **Documentation debt**: Misleading documentation

### Resolution
✅ **Fixed**: Removed outdated legacy CLI documentation from README.md

---

## Issue 2: Incomplete Plotting Migration - Two Commands Exist 🔴

### Problem
PeakFit has **TWO separate plotting commands** with **incomplete feature parity**:

#### 1. `peakfit-plot` (Old CLI - Full Featured)
- **Location**: `src/peakfit/plotting/main.py`
- **Style**: argparse-based
- **Entry point**: `pyproject.toml` line 46: `peakfit-plot = "peakfit.plotting.main:main"`
- **Functionality**: ✅ **Complete**
  - `peakfit-plot intensity` - Intensity profiles
  - `peakfit-plot cest` - CEST plots
  - `peakfit-plot cpmg` - CPMG plots
  - `peakfit-plot spectra` - Interactive spectra viewer (PyQt5)

#### 2. `peakfit plot` (New CLI - Partial)
- **Location**: `src/peakfit/cli/plot_command.py`
- **Style**: Typer-based
- **Entry point**: Part of main `peakfit` CLI
- **Functionality**: ⚠️ **Incomplete**
  - ✅ `intensity` - Works
  - ❌ `cest` - Placeholder: "use peakfit-plot cest for full functionality"
  - ❌ `cpmg` - Placeholder: "use peakfit-plot cpmg for full functionality"
  - ⚠️ `spectra` - Wrapper that calls old code

### Code Evidence

**`src/peakfit/cli/plot_command.py` lines 66-75:**
```python
def _plot_cest(results: Path, output: Path | None, show: bool) -> None:
    """Generate CEST plots."""
    console.print("[yellow]CEST plotting - use peakfit-plot cest for full functionality[/yellow]")
    # Placeholder - integrate with existing cest.py module


def _plot_cpmg(results: Path, output: Path | None, show: bool) -> None:
    """Generate CPMG plots."""
    console.print("[yellow]CPMG plotting - use peakfit-plot cpmg for full functionality[/yellow]")
    # Placeholder - integrate with existing cpmg.py module
```

### Impact
- **User confusion**: Two commands with overlapping but not identical functionality
- **Inconsistent UX**: New CLI modernization incomplete
- **Documentation burden**: Must explain both commands
- **Maintenance**: Two codebases for plotting

### Comparison

| Feature | `peakfit-plot` (old) | `peakfit plot` (new) | Status |
|---------|---------------------|---------------------|--------|
| Intensity plots | ✅ Full | ✅ Full | ✅ Migrated |
| CEST plots | ✅ Full | ❌ Placeholder | 🔴 Not Migrated |
| CPMG plots | ✅ Full | ❌ Placeholder | 🔴 Not Migrated |
| Spectra viewer | ✅ Full | ⚠️ Wrapper | ⚠️ Partial |
| CLI Style | argparse | Typer | - |

### Current State

Users must use **BOTH commands** depending on what they need:
```bash
# Basic intensity plotting - use new CLI
peakfit plot results/ --output plots.pdf

# Advanced plotting - use old CLI
peakfit-plot cest Fits/ --ref -1
peakfit-plot cpmg Fits/ --time_t2 0.02
peakfit-plot spectra --exp spectrum.ft2 --sim Fits/simulated.ft2
```

### Resolution Options

#### Option 1: Complete the Migration (Recommended for v2.0) ⭐
**Pros:**
- Consistent user experience
- Single command for all plotting
- Completes modernization vision
- Can deprecate `peakfit-plot` in future

**Cons:**
- Significant implementation work
- Requires thorough testing
- May introduce regressions

**Tasks:**
1. Implement full CEST plotting in `peakfit plot --type cest`
2. Implement full CPMG plotting in `peakfit plot --type cpmg`
3. Integrate PyQt5 spectra viewer properly
4. Add comprehensive tests
5. Update documentation
6. Deprecate `peakfit-plot` with warning
7. Remove `peakfit-plot` in next major version

#### Option 2: Document Current State (Implemented) ✅
**Pros:**
- No code changes needed
- Honest about current state
- Users know what to expect

**Cons:**
- Keeps two commands
- Incomplete modernization
- Technical debt

**Implementation:**
- ✅ Updated README.md to document both commands
- ✅ Clarified which command to use for which feature
- ✅ Added note that full integration is planned

#### Option 3: Remove Incomplete `peakfit plot`
**Pros:**
- Eliminates confusion
- Single source of truth
- Clean separation

**Cons:**
- Loses Typer-based plotting interface
- Regression in modernization
- Inconsistent with rest of CLI

### Current Resolution
✅ **Option 2 implemented**: Documentation updated to clearly explain:
- `peakfit plot` for basic intensity plotting
- `peakfit-plot` for advanced features (CEST, CPMG, spectra)
- Note that full integration is planned

---

## Recommendations

### Short-term (This PR)
✅ **Done:**
1. Remove outdated `peakfit-legacy` documentation
2. Document current plotting situation clearly
3. Set user expectations correctly

### Medium-term (Next Release)
🔧 **Recommended:**
1. Complete CEST/CPMG integration into `peakfit plot`
2. Add comprehensive plotting tests
3. Deprecate `peakfit-plot` with migration guide

### Long-term (v2.0+)
📋 **Planned:**
1. Remove `peakfit-plot` command entirely
2. Fully unified Typer-based CLI
3. Complete modernization

---

## Testing

### Verified Commands Exist

```bash
# Old plotting command (argparse-based)
$ peakfit-plot --help
✅ Works - shows subcommands

# New plotting command (Typer-based)
$ peakfit plot --help
✅ Works - shows options

# Both are functional but have different features
```

### Entry Points in pyproject.toml

```toml
[project.scripts]
peakfit = "peakfit.cli.app:app"           # ✅ Modern Typer CLI
peakfit-plot = "peakfit.plotting.main:main"  # ⚠️ Old argparse CLI
```

---

## Summary

| Issue | Severity | Status | Resolution |
|-------|----------|--------|------------|
| Outdated `peakfit-legacy` docs | Minor | ✅ Fixed | Removed from README |
| Incomplete plotting migration | Medium | ⚠️ Documented | Clear docs, recommend full migration |

**Overall Status**: Documented, recommended for future completion
**User Impact**: Low (documented clearly)
**Technical Debt**: Medium (two plotting commands)
