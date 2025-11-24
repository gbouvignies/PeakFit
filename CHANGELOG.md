# Changelog

All notable changes to PeakFit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2025.12.0] - 2025-11-22

### Breaking Changes
- **Removed Numba dependency**: PeakFit now uses pure NumPy implementations for all lineshape functions
  - Performance impact: 2-5x slower lineshape evaluation (acceptable for most use cases)
  - Simpler dependency stack, better compatibility, easier maintenance
  - No action required: NumPy implementations are drop-in replacements

### Major Refactoring
- **Complete module reorganization** for improved clarity and maintainability:
  - `peakfit.lineshapes/` - All lineshape functions and models
  - `peakfit.fitting/` - Fitting algorithms, parameters, and results
  - `peakfit.data/` - Spectrum and peak data structures
  - `peakfit.models/` - Configuration models
  - `peakfit.analysis/` - Benchmarking and profiling
  - `peakfit.io/` - Input/output operations
- Removed `core/` module - functionality redistributed to logical packages
- Backend selection system removed - always uses NumPy

### Migration Guide
Old imports → New imports:
- `from peakfit.shapes import Gaussian` → `from peakfit.lineshapes import Gaussian`
- `from peakfit.core.fitting import Parameters` → `from peakfit.fitting import Parameters`
- `from peakfit.clustering import Cluster` → `from peakfit.data import Cluster`
- `from peakfit.spectra import Spectra` → `from peakfit.data import Spectra`
- And more... (see documentation)

### Removed
- Numba JIT compilation support
- Backend selection (`--backend` option)
- `performance` optional dependency group

## [Unreleased]

### Added

- **Parallel Cluster Fitting**: Multi-core support for faster fitting (automatic)
  - Multi-core support is chosen automatically; the CLI no longer exposes a `--parallel` flag
  - Automatic CPU core detection
  - Linear scaling with number of clusters
  - Maintains refinement iterations for cross-talk correction

- **Optimized Fitting Engine**: Direct scipy.optimize integration
  - New `peakfit.fitting` module with scipy.optimize.least_squares
  - Custom Parameters class with bounds validation
  - Reduced overhead compared to lmfit wrapper
  - FitResult class with chi-squared statistics

- **Modern CLI with Typer**: Intuitive command-line interface with subcommands (`fit`, `validate`, `init`, `plot`)
  - `peakfit fit spectrum.ft2 peaks.list` - Clear, positional arguments
  - `peakfit init config.toml` - Generate configuration templates
  - `peakfit validate` - Input file validation
  - Rich progress bars and colored terminal output

- **TOML Configuration System**: Reproducible analyses with configuration files
  - Full configuration validation with Pydantic models
  - Auto-generated config templates with documentation
  - Support for fitting, clustering, and output settings

- **Type-Safe Data Models**: Pydantic v2 models for all configuration and data structures
  - `FitConfig` - Fitting parameters with validation
  - `ClusterConfig` - Clustering settings
  - `OutputConfig` - Output format configuration
  - `PeakFitConfig` - Main configuration container
  - `PeakData`, `FitResult` - Structured result data

- **Comprehensive Test Suite**: pytest-based testing infrastructure
  - Unit tests for lineshapes, models, configuration, clustering
  - Integration tests with synthetic spectrum generation
  - CLI command tests
  - Coverage reporting configuration
  - Test fixtures for reproducible testing

- **Enhanced Project Configuration**
  - pytest configuration in pyproject.toml
  - Coverage reporting settings
  - Optional dependencies for development and performance
  - Proper package structure with submodules

- **Improved Documentation**
  - Comprehensive README with usage examples
  - CLI reference documentation
  - Migration guide from old CLI
  - Development setup instructions
  - Code quality guidelines

### Changed

- **CLI Interface**: From cryptic flags to intuitive commands
  - Old: `peakfit -s spectrum.ft2 -l peaks.list -o Fits -r 2 --pvoigt`
  - New: `peakfit fit spectrum.ft2 peaks.list --output Fits --refine 2 --lineshape pvoigt`

- **Package Structure**: Modular organization
  - `peakfit/lineshapes/` - Lineshape functions and models
  - `peakfit/fitting/` - Fitting algorithms and parameters
  - `peakfit/data/` - Spectrum and peak data structures
  - `peakfit/models/` - Configuration models
  - `peakfit/analysis/` - Benchmarking and profiling
  - `peakfit/io/` - Input/output operations
  - `peakfit/cli/` - Modern Typer CLI

- **Dependencies**:
  - Added: pydantic>=2.5.0, typer>=0.9.0, tomli-w>=1.0.0, openpyxl>=3.0.0
  - Changed: pandas dependency simplified (removed [excel] extra)
  - Python version requirement relaxed to >=3.11

- **Entry Points**:
  - `peakfit` now launches modern Typer CLI with integrated plotting
  - Plotting now accessible via `peakfit plot` subcommands (cest, cpmg, intensity, spectra)

### Fixed

- Version discovery with fallback for development installations
- CLI module renamed to avoid namespace conflicts

### Removed

- Legacy argparse CLI (`peakfit-legacy`)
- JAX backend support
- Standalone `peakfit-plot` command (replaced with `peakfit plot` subcommands)

## [Previous] - Legacy Versions

### Features from Original Implementation

- Multiple lineshape models (Gaussian, Lorentzian, Pseudo-Voigt, SP1, SP2, No-Apod)
- Automatic lineshape detection from NMRPipe apodization parameters
- Peak clustering using connected components analysis
- Iterative refinement for cross-talk correction
- NMRPipe format support via nmrglue
- Rich console output with HTML reports
- Interactive PyQt5-based spectra viewer
- Multiple peak list format support (Sparky, CSV, JSON, Excel)
