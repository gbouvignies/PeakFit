# PeakFit Development Instructions

## Project Overview

PeakFit is an NMR spectroscopy lineshape fitting tool for analyzing protein dynamics through pseudo-ND experiments (CEST, relaxation dispersion, etc.). It fits lineshapes to extract chemical shifts, linewidths, and intensities.

## Architecture
```
src/peakfit/
├── cli/           # Command-line interface
├── core/          # Core data models and algorithms
├── io/            # File I/O (readers, writers, schemas)
├── fitting/       # Fitting algorithms
├── lineshapes/    # Lineshape models
└── utils/         # Shared utilities
```

## Conventions

### Python Version
- Python 3.11+ required, 3.13+ features allowed
- Use `str | None` syntax (not `Optional[str]`)

### Naming
- Dimension labels follow NMRPipe convention: `F1`, `F2`, `F3`, `F4`
- Parameter names: `cs_F1`, `cs_F2`, `lw_F1`, `lw_F2` (not `x`, `y`)
- Use descriptive names, avoid abbreviations except standard NMR terms

### Code Style
- Type hints required for all public functions
- Dataclasses preferred for data containers (use `frozen=True` where appropriate)
- No global state or singletons
- Functions should do one thing and be < 50 lines
- Use `ruff` for linting

### Error Handling
- Use `ValueError`, `TypeError`, `FileNotFoundError` with clear messages
- Custom exceptions only when genuinely needed (e.g., `PeakFitError` base class)
- Validate inputs early, fail fast
- Error messages should be actionable and include context

### Testing
- Tests in `tests/` mirroring `src/` structure
- Use pytest with parametrized tests for N-dimensional cases
- Test edge cases: empty input, single peak, 1D through 4D spectra

### Output Files
- Machine-readable: JSON (with schema validation), CSV
- Human-readable: Markdown reports
- No hidden files (no `.` prefix)
- Internal cache goes in `cache/` subdirectory
- See `output-system.instructions.md` for details

## Key Design Decisions

1. **N-dimensional support**: Code must handle 1D through 4D spectra generically. No hardcoded 2D assumptions. Currently, only one pseudo-dimension per spectrum is supported.

2. **Separation of concerns**: IO, core logic, and CLI are separate. Core should have no knowledge of file formats.

3. **Immutable data models**: Fit results and spectra should be immutable after creation (`frozen=True` dataclasses).

4. **Performance-critical paths**: Lineshape evaluation and Jacobian computation are hot paths. Keep them simple, use NumPy vectorization, and prepare for future performance optimizations.

## Dependencies

Core dependencies (minimal):
- numpy: Array operations
- scipy: Optimization, signal processing

Avoid adding heavy dependencies for minor features.

## What NOT to Do

- Don't embed peak names in parameter names (use separate fields)
- Don't create hidden files or directories
- Don't hardcode dimension labels as `x`, `y`, `z`, `a`
- Don't mix IO with business logic
- Don't use pickle for user-facing outputs (only internal cache)
- Don't add dependencies without justification

## Current Focus

1. Consolidate codebase before optimization
2. Complete N-dimensional support
3. Clean output system with JSON, CSV, Markdown
4. Prepare for future performance optimizations of hot paths
