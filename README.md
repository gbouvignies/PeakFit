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

```bash
# Using pip
pip install peakfit

# Using uv (recommended)
uv pip install peakfit

# For development
git clone https://github.com/gbouvignies/PeakFit.git
cd PeakFit
uv pip install -e ".[dev]"
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
peakfit plot Results/ --type intensity --show

# Launch interactive spectra viewer
peakfit plot Results/ --spectrum spectrum.ft2 --type spectra
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
  --parallel/--no-parallel Enable parallel fitting
  -b, --backend TEXT      Computation backend: auto, numpy, numba, jax
  --help                  Show this message and exit
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

Generate plots from fitting results.

```bash
peakfit plot RESULTS [OPTIONS]

Arguments:
  RESULTS                 Path to results directory or file

Options:
  -s, --spectrum PATH     Path to original spectrum for overlay
  -o, --output PATH       Output file for plots (PDF)
  --show/--no-show        Display plots interactively
  -t, --type TEXT         Plot type: intensity, cest, cpmg, spectra
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

### Parallel Fitting

Enable multi-core parallel processing for faster fitting of multiple clusters:

```bash
# Fit clusters in parallel (recommended for datasets with many peaks)
peakfit fit spectrum.ft2 peaks.list --parallel

# Parallel fitting scales with number of CPU cores
# Particularly beneficial when you have many independent clusters
```

**Benefits:**
- Linear scaling with number of clusters
- Automatic CPU core detection
- Maintains cross-talk correction through refinement iterations
- Significant speedup for large peak lists (10+ clusters)

### Backend Selection

PeakFit supports multiple computation backends for optimal performance:

```bash
# Auto-select best available backend (default)
peakfit fit spectrum.ft2 peaks.list --backend auto

# Use specific backend
peakfit fit spectrum.ft2 peaks.list --backend numpy   # Pure NumPy (always available)
peakfit fit spectrum.ft2 peaks.list --backend numba   # Numba JIT (2-5x faster)
peakfit fit spectrum.ft2 peaks.list --backend jax     # JAX (GPU/TPU support)
```

**Available Backends:**
- **NumPy**: Always available, pure Python/NumPy implementation
- **Numba**: JIT-compiled functions (2-5x faster), requires `pip install peakfit[performance]`
- **JAX**: GPU/TPU acceleration + autodiff, requires `pip install peakfit[jax]`

Check available backends:
```bash
peakfit info
```

### Performance Optimization

Install optional performance dependencies for faster lineshape calculations:

```bash
# Install with Numba JIT compilation support
pip install peakfit[performance]

# Or install numba directly
pip install numba

# Install JAX for GPU acceleration
pip install peakfit[jax]

# Install all optional dependencies
pip install peakfit[all]
```

Numba provides:
- JIT-compiled lineshape functions (2-5x faster)
- Automatic fallback to NumPy if not available
- No code changes required

JAX provides:
- GPU/TPU acceleration for large datasets
- Automatic differentiation for advanced optimization
- Exact gradient/Hessian computation

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

### Running Tests

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=peakfit --cov-report=html

# Run specific test file
pytest tests/unit/test_lineshapes.py
```

### Code Quality

```bash
# Linting with Ruff
ruff check peakfit/

# Type checking
mypy peakfit/

# Format code
ruff format peakfit/
```

### Project Structure

```
peakfit/
├── core/               # Core data models and fitting logic
│   ├── models.py       # Pydantic models
│   └── ...
├── cli/                # Modern CLI with Typer
│   ├── app.py          # Main Typer application
│   ├── fit_command.py  # Fit command implementation
│   └── ...
├── io/                 # File I/O operations
│   ├── config.py       # TOML configuration loader
│   └── ...
├── plotting/           # Visualization
│   └── plots/          # Individual plot generators
├── shapes.py           # Lineshape models
├── clustering.py       # Peak clustering
├── computing.py        # Fitting computations
└── ...
```

## Legacy CLI

The original CLI is still available as `peakfit-legacy`:

```bash
peakfit-legacy -s spectrum.ft2 -l peaks.list -o Fits -r 2 --pvoigt
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
from peakfit.core.fitting import Parameters, Parameter, ParameterType

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
from peakfit.core.fitting import fit_cluster, FitResult

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

PeakFit includes global optimization methods for difficult fitting problems:

```python
from peakfit.core.advanced_optimization import (
    fit_basin_hopping,
    fit_differential_evolution,
    estimate_uncertainties_mcmc,
    compute_profile_likelihood,
)

# Basin-hopping for escaping local minima
result = fit_basin_hopping(
    params,
    cluster,
    noise,
    n_iterations=100,
    temperature=1.0,
    step_size=0.5,
)

# Differential evolution for global search
result = fit_differential_evolution(
    params,
    cluster,
    noise,
    max_iterations=1000,
    population_size=15,
    polish=True,
)

# MCMC for uncertainty estimation
uncertainties = estimate_uncertainties_mcmc(
    params,
    cluster,
    noise,
    n_walkers=32,
    n_steps=1000,
    burn_in=200,
)
print(f"68% CI: {uncertainties.confidence_intervals_68}")
print(f"95% CI: {uncertainties.confidence_intervals_95}")

# Profile likelihood for accurate confidence intervals
values, chi2, ci = compute_profile_likelihood(
    params,
    cluster,
    noise,
    param_name="peak1_fwhm",
    delta_chi2=3.84,  # 95% CI
)
```

### Backend Registry

Control which computation backend is used for lineshape calculations:

```python
from peakfit.core.backend import (
    get_available_backends,
    get_best_backend,
    set_backend,
    get_backend,
    auto_select_backend,
)

# Check available backends
print(get_available_backends())  # ['numpy', 'numba', 'jax']
print(get_best_backend())        # 'jax' (prefers JAX > Numba > NumPy)

# Set specific backend
set_backend("numba")  # Use Numba JIT compilation
print(get_backend())  # 'numba'

# Auto-select best available
selected = auto_select_backend()  # Automatically picks best
print(f"Using: {selected}")

# Get backend-specific functions
from peakfit.core.backend import (
    get_gaussian_func,
    get_lorentzian_func,
    get_pvoigt_func,
)

gaussian = get_gaussian_func()  # Returns current backend's Gaussian
result = gaussian(dx, fwhm)     # Compute using selected backend
```

### JAX Backend (Optional)

For GPU acceleration and automatic differentiation:

```bash
# Install JAX support
pip install peakfit[jax]
```

```python
from peakfit.core.jax_backend import (
    is_jax_available,
    gaussian_jax,
    lorentzian_jax,
    pseudo_voigt_jax,
    compute_gradient,
    compute_hessian,
)

# Check JAX availability
if is_jax_available():
    # Direct JAX lineshape computation
    result = gaussian_jax(dx, fwhm)
    result = lorentzian_jax(dx, fwhm)
    result = pseudo_voigt_jax(dx, fwhm, eta)

    # Autodiff for exact gradients (advanced usage)
    grad = compute_gradient(params, cluster, noise)
    hess = compute_hessian(params, cluster, noise)
```

### Performance Benchmarking

```python
from peakfit.core.benchmarks import (
    benchmark_lineshape_backends,
    compare_backends_report,
    profile_fit_cluster,
)

# Compare backend performance
results = benchmark_lineshape_backends(n_points=1000, n_iterations=100)
print(compare_backends_report(results))

# Profile fitting stages
profile = profile_fit_cluster(params, cluster, noise)
print(f"Shape calculation: {profile['shape_calculation']*1000:.3f} ms")
print(f"Residual calculation: {profile['residual_calculation']*1000:.3f} ms")
print(f"Full fit: {profile['full_fit']*1000:.3f} ms")
```
