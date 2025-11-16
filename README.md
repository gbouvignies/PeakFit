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

- Python >= 3.11
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

### Parallel Fitting (Experimental)

```bash
peakfit fit spectrum.ft2 peaks.list --parallel
```

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
