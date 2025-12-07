# PeakFit

Modern lineshape fitting for pseudo-3D NMR spectra.

## Features

- **Multiple lineshape models**: Gaussian, Lorentzian, Pseudo-Voigt, and apodization-specific models (SP1, SP2, No-Apod)
- **Automatic lineshape detection**: Detects optimal lineshape from NMRPipe processing parameters
- **Peak clustering**: Automatic grouping of overlapping peaks for simultaneous fitting
- **Modern CLI**: Intuitive command-line interface with rich terminal output
- **Configuration files**: TOML-based configuration for reproducible analyses
- **Type-safe**: Full type hints and Pydantic models for validation
- **Comprehensive testing**: Extensive test suite with synthetic data validation

## Installation

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager. Install it first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install PeakFit:

```bash
# Install PeakFit
uv pip install peakfit

# Or create a new project with PeakFit
uv init my-project
cd my-project
uv add peakfit
```

### Using pip

```bash
pip install peakfit
```

### Development Installation

```bash
git clone https://github.com/gbouvignies/PeakFit.git
cd PeakFit
uv sync --all-extras  # Install all dependencies including dev tools
```

## Requirements

- Python >= 3.13
- NMRPipe format spectrum files (.ft2, .ft3)

## Quick Start

### Basic Fitting

```bash
# Fit peaks in a pseudo-3D spectrum
peakfit fit spectrum.ft2 peaks.list

# Specify output directory and refinement iterations
peakfit fit spectrum.ft2 peaks.list --output Results --refine 2

# Use specific lineshape model
peakfit fit spectrum.ft2 peaks.list --lineshape pvoigt

# Fix peak positions during fitting
peakfit fit spectrum.ft2 peaks.list --fixed
```

### Using Configuration Files

```bash
# Generate a default configuration file
peakfit init config.toml

# Edit the configuration file, then run:
peakfit fit spectrum.ft2 peaks.list --config config.toml
```

Example configuration (`config.toml`):

```toml
[fitting]
lineshape = "auto"
refine_iterations = 2
fix_positions = false
max_iterations = 1000
tolerance = 1e-8

[clustering]
contour_factor = 5.0

[output]
directory = "Fits"
formats = ["txt"]
save_simulated = true
save_html_report = true

exclude_planes = []
```

### Validation

```bash
# Validate input files before fitting
peakfit validate spectrum.ft2 peaks.list
```

### Plotting

```bash
# Generate intensity plots
peakfit plot intensity Results/ --show

# Launch interactive spectra viewer
peakfit plot spectra Results/ --spectrum spectrum.ft2
```

## CLI Reference

### `peakfit fit`

Fit lineshapes to peaks in pseudo-3D NMR spectrum.

```bash
peakfit fit SPECTRUM PEAKLIST [OPTIONS]

Arguments:
  SPECTRUM                Path to NMRPipe spectrum file (.ft2, .ft3)
  PEAKLIST                Path to peak list file (.list, .csv, .json, .xlsx)

Options:
  -z, --z-values PATH     Path to Z-dimension values file
  -o, --output PATH       Output directory [default: Fits]
  -c, --config PATH       Path to TOML configuration file
  -l, --lineshape TEXT    Lineshape: auto, gaussian, lorentzian, pvoigt, sp1, sp2
  -r, --refine INTEGER    Number of refinement iterations [default: 1]
  -t, --contour FLOAT     Contour level for segmentation
  -n, --noise FLOAT       Manual noise level
  --fixed/--no-fixed      Fix peak positions
  --jx/--no-jx            Fit J-coupling constant
  --phx/--no-phx          Fit phase correction in X
  --phy/--no-phy          Fit phase correction in Y
  -e, --exclude INTEGER   Plane indices to exclude
  --help                  Show this message and exit

Note: The CLI option `--workers` has been removed. PeakFit now runs sequentially by default.
```

### `peakfit validate`

Validate input files before fitting.

```bash
peakfit validate SPECTRUM PEAKLIST
```

### `peakfit init`

Generate a default configuration file.

```bash
peakfit init [PATH] [OPTIONS]

Arguments:
  PATH    Path for new configuration file [default: peakfit.toml]

Options:
  -f, --force    Overwrite existing file
```

### `peakfit plot`
 
 Generate plots from fitting results using subcommands.
 
 ```bash
 peakfit plot [SUBCOMMAND] [RESULTS] [OPTIONS]
 ```
 
 #### `peakfit plot intensity`
 
 Plot intensity profiles vs. plane index.
 
 ```bash
 peakfit plot intensity RESULTS [--output PATH] [--show]
 ```
 
 #### `peakfit plot cest`
 
 Plot CEST profiles (normalized intensity vs. B1 offset).
 
 ```bash
 peakfit plot cest RESULTS [--output PATH] [--show] [--ref INDICES...]
 ```
 
 #### `peakfit plot cpmg`
 
 Plot CPMG relaxation dispersion (R2eff vs. νCPMG).
 
 ```bash
 peakfit plot cpmg RESULTS --time-t2 FLOAT [--output PATH] [--show]
 ```
 
 #### `peakfit plot spectra`
 
 Launch interactive spectra viewer.
 
 ```bash
 peakfit plot spectra RESULTS --spectrum PATH
 ```
 
 #### `peakfit plot diagnostics`
 
 Generate MCMC diagnostic plots.
 
 ```bash
 peakfit plot diagnostics RESULTS [--output PATH] [--peaks NAMES...]
 ```
 
 ### `peakfit analyze`
 
 Perform uncertainty analysis on fitting results.
 
 ```bash
 peakfit analyze [SUBCOMMAND] [RESULTS] [OPTIONS]
 ```
 
 #### `peakfit analyze mcmc`
 
 Run MCMC sampling for uncertainty estimation.
 
 ```bash
 peakfit analyze mcmc RESULTS [--chains INT] [--samples INT] [--burn-in INT]
 ```
 
 #### `peakfit analyze profile`
 
 Compute profile likelihood confidence intervals.
 
 ```bash
 peakfit analyze profile RESULTS [--param NAME] [--points INT] [--confidence FLOAT]
 ```
 
 #### `peakfit analyze uncertainty`
 
 Display parameter uncertainties from fitting results.
 
 ```bash
 peakfit analyze uncertainty RESULTS [--output PATH]
 ```

## Peak List Formats

### Sparky Format (`.list`)

```
# Sparky peak list
Assignment  w1   w2
Peak1  8.50  120.5
Peak2  7.80  115.3
Peak3  8.52  120.8
```

### CSV Format (`.csv`)

```csv
Assign F1,Assign F2,Pos F1,Pos F2
Peak1,Peak1,8.50,120.5
Peak2,Peak2,7.80,115.3
```

### JSON Format (`.json`)

```json
[
  {"name": "Peak1", "x": 8.50, "y": 120.5},
  {"name": "Peak2", "x": 7.80, "y": 115.3}
]
```

## Output Files

After fitting, PeakFit generates the following files in the output directory:

- **`{peak_name}.out`** - Per-peak fitting results with intensity profiles
- **`shifts.list`** - Fitted chemical shift positions
- **`simulated.ft2/ft3`** - Reconstructed spectrum from fitted parameters
- **`logs.html`** - HTML report with detailed fitting information

## Lineshape Models

### Frequency-Domain Shapes

- **Gaussian**: `exp(-(dx²) * 4*ln(2) / FWHM²)`
- **Lorentzian**: `(0.5*FWHM)² / (dx² + (0.5*FWHM)²)`
- **Pseudo-Voigt**: `(1-η)*Gaussian + η*Lorentzian`

### Time-Domain Apodized Shapes

- **NO_APOD**: No apodization window applied
- **SP1**: Sine-bell apodization (power 1)
- **SP2**: Sine-bell apodization (power 2)

## Advanced Usage

### Fitting behavior

PeakFit performs sequential cluster fitting using scipy.optimize least squares for predictable execution and minimal memory usage.

**Notes:**
- Multi-process/parallel cluster fitting was removed to simplify the execution model.
- For datasets with many clusters, performance can be improved by optimizing lineshape calculations or using the benchmark tools to tune your environment.

### Excluding Planes

```bash
# Exclude specific planes from fitting
peakfit fit spectrum.ft2 peaks.list --exclude 0 --exclude 5 --exclude 10
```

### Custom Noise Level

```bash
# Set manual noise level instead of auto-detection
peakfit fit spectrum.ft2 peaks.list --noise 100.0
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/gbouvignies/PeakFit.git
cd PeakFit

# Install with all dependencies (recommended)
uv sync --all-extras

# Or install development dependencies only
uv sync --extra dev
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=peakfit --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_lineshapes.py
```

### Code Quality

```bash
# Linting with Ruff
uv run ruff check peakfit/

# Type checking
uv run mypy peakfit/

# Format code
uv run ruff format peakfit/

# Run pre-commit hooks
uv run pre-commit run --all-files
```

### Building the Package

```bash
# Build wheel and source distribution
uv build

# Build artifacts are in dist/
```

### Project Structure

```
peakfit/
├── cli/                # Modern CLI with Typer
│   ├── app.py          # Main Typer application
│   └── commands/       # Command implementations
├── core/               # Core domain logic
│   ├── domain/         # Data models (Spectrum, Peak, Config)
│   ├── fitting/        # Fitting algorithms and protocols
│   └── lineshapes/     # Lineshape functions
├── io/                 # Input/output operations
│   ├── readers/        # File readers (Sparky, NMRPipe)
│   └── writers/        # File writers
├── plotting/           # Visualization
│   └── plots/          # Plot generators
├── services/           # Application services
│   ├── fit/            # Fitting pipeline service
│   └── validate/       # Validation service
└── ui/                 # User interface
    └── console.py      # Rich console integration
```

## Plotting

PeakFit provides comprehensive plotting capabilities through the `peakfit plot` command with dedicated subcommands for each plot type:

### Intensity Profiles

```bash
# Generate intensity plots
peakfit plot intensity Fits/ --output plots.pdf

# Interactive display (limited to first 10 plots for large datasets)
peakfit plot intensity Fits/ --show
```

### CEST Plots

```bash
# Auto-detect reference points (|offset| >= 10 kHz)
peakfit plot cest Fits/ --output cest.pdf

# Manually specify reference point indices
peakfit plot cest Fits/ --ref 0 1 2

# Interactive display (limited to first 10 plots)
peakfit plot cest Fits/ --show
```

### CPMG Relaxation Dispersion

```bash
# Generate CPMG plots (--time-t2 is required)
peakfit plot cpmg Fits/ --time-t2 0.04 --output cpmg.pdf

# With interactive display
peakfit plot cpmg Fits/ --time-t2 0.04 --show
```

### Interactive Spectra Viewer

```bash
# Launch PyQt5 viewer with spectrum overlay
peakfit plot spectra Fits/ --spectrum data.ft2
```

## Migration from Previous Version

The new CLI provides a more intuitive interface while maintaining all functionality:

| Old Command | New Command |
|-------------|-------------|
| `peakfit -s spec.ft2 -l peaks.list` | `peakfit fit spec.ft2 peaks.list` |
| `peakfit -s spec.ft2 -l peaks.list -o Out -r 3` | `peakfit fit spec.ft2 peaks.list --output Out --refine 3` |
| `peakfit -s spec.ft2 -l peaks.list --pvoigt` | `peakfit fit spec.ft2 peaks.list --lineshape pvoigt` |

## Citation

If you use PeakFit in your research, please cite:

```
[Citation information to be added]
```

## License

GPL-3.0-or-later

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- NMRPipe file format support via [nmrglue](https://www.nmrglue.com/)
- Rich terminal output via [Rich](https://github.com/Textualize/rich)
- CLI framework via [Typer](https://typer.tiangolo.com/)

## API Reference

### Parameters System

PeakFit uses a custom parameter system optimized for NMR fitting with domain-specific bounds and metadata:

```python
from peakfit.fitting import Parameters, Parameter, ParameterType

# Create parameters with NMR-specific types
params = Parameters()

# Position parameter (ppm)
params.add(
    "peak1_x0",
    value=8.50,
    min=8.40,
    max=8.60,
    param_type=ParameterType.POSITION,
    unit="ppm"
)

# Linewidth parameter with automatic bounds
# FWHM type defaults to bounds (0.1, 200.0) Hz
params.add(
    "peak1_fwhm",
    value=25.0,
    param_type=ParameterType.FWHM,
    unit="Hz"
)

# Phase correction with automatic bounds
# PHASE type defaults to bounds (-180.0, 180.0) degrees
params.add(
    "peak1_phase",
    value=0.0,
    param_type=ParameterType.PHASE,
    unit="deg"
)

# J-coupling constant with automatic bounds
# JCOUPLING type defaults to bounds (0.0, 20.0) Hz
params.add(
    "peak1_j",
    value=7.0,
    param_type=ParameterType.JCOUPLING,
    unit="Hz"
)

# Fraction (mixing) parameter
# FRACTION type defaults to bounds (0.0, 1.0)
params.add(
    "peak1_eta",
    value=0.5,
    param_type=ParameterType.FRACTION
)

# Parameter operations
params.freeze(["peak1_x0"])  # Fix parameters
params.unfreeze(["peak1_x0"])  # Release parameters
boundary_params = params.get_boundary_params()  # Check for boundary issues
print(params.summary())  # Formatted parameter table
```

### NMR Parameter Types

- **POSITION**: Peak center position (units: ppm or points)
- **FWHM**: Full width at half maximum (units: Hz, bounds: 0.1-200.0)
- **FRACTION**: Mixing parameters like eta (bounds: 0.0-1.0)
- **PHASE**: Phase correction (units: deg, bounds: -180.0 to 180.0)
- **JCOUPLING**: J-coupling constants (units: Hz, bounds: 0.0-20.0)
- **AMPLITUDE**: Peak amplitudes (bounds: 0.0 to inf)
- **GENERIC**: Other parameters (no default bounds)

### Fitting Engine

The fitting engine uses `scipy.optimize.least_squares` directly for optimal performance:

```python
from typing import Any

from peakfit.fitting import fit_cluster, FitResult

# Example placeholders for demonstration purposes
# In real code, construct these via peak lists and spectra
params: Any = ...  # Parameters
cluster: Any = ...  # Cluster
noise: float = 1.0

# Fit a single cluster
result: FitResult = fit_cluster(
    params,
    cluster,
    noise,
    max_nfev=1000,
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8
)

# Access fit statistics
print(f"Chi-squared: {result.chisqr}")
print(f"Reduced chi-squared: {result.redchi}")
print(f"Function evaluations: {result.nfev}")
print(f"Success: {result.success}")
```

### Advanced Optimization

PeakFit includes global optimization methods for difficult fitting problems.
These are available as development utilities in `tools/analysis/`:

```python
# From tools/analysis (development utilities, not part of installed package)
from typing import Any

from tools.analysis import (
    benchmark_lineshape_backends,
    profile_fit_cluster,
    Profiler,
    estimate_optimal_workers,
)

# Example placeholders for demonstration purposes
params: Any = ...
cluster: Any = ...
noise: float = 1.0

# Profile fitting stages
profile = profile_fit_cluster(params, cluster, noise)
print(f"Shape calculation: {profile['shape_calculation']*1000:.3f} ms")
print(f"Residual calculation: {profile['residual_calculation']*1000:.3f} ms")
print(f"Full fit: {profile['full_fit']*1000:.3f} ms")

# Benchmark different lineshape backends
results = benchmark_lineshape_backends(n_points=1000, n_iterations=100)
print(results)
```
