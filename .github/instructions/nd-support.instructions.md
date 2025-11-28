# PeakFit N-Dimensional Support Instructions

## Overview

PeakFit handles pseudo-ND experiments: series of 1D, 2D, 3D, or 4D spectra where one dimension is the "pseudo" axis (e.g., CEST offsets, relaxation delays, CPMG frequencies).

The code must be fully generic for N dimensions. No hardcoded 2D assumptions.

## Terminology

| Term | Meaning |
|------|---------|
| Spectral dimensions | Dimensions with chemical shift axis (fit lineshapes) |
| Pseudo dimension | Series axis (offsets, delays, etc.) — extract intensities |
| Direct dimension | Acquired dimension (typically 1H) |
| Indirect dimension | Evolved dimensions (15N, 13C, etc.) |

Example: A 2D 1H-15N HSQC-CEST with 131 offset planes:
- 2 spectral dimensions (1H, 15N)
- 1 pseudo dimension (131 CEST offsets)
- File is 3D (pseudo-3D)

## NMRPipe Dimension Convention

NMRPipe numbers dimensions in reverse acquisition order:

| Dimension | Label | Description |
|-----------|-------|-------------|
| First indirect | F1 | Slowest |
| Second indirect | F2 | |
| Third indirect | F3 | |
| Direct/acquired | F4 (or highest) | Fastest |

For a 2D 1H-15N HSQC:
- F2 = 1H (direct, 512 points)
- F1 = 15N (indirect, 64 points)

For a 3D HNCO:
- F3 = 1H (direct)
- F2 = 15N (indirect)
- F1 = 13C (indirect)

## Data Model

### DimensionInfo
```python
@dataclass(frozen=True)
class DimensionInfo:
    """Metadata for one dimension of a spectrum."""
    index: int              # 0-based index in array
    label: str              # "F1", "F2", "F3", "F4"
    nucleus: str | None     # "1H", "15N", "13C" (from header or None)
    size: int               # Number of points
    sw_hz: float            # Spectral width in Hz
    sf_mhz: float           # Spectrometer frequency in MHz
    ppm_limits: tuple[float, float]  # (min_ppm, max_ppm)
    is_direct: bool         # True for acquisition dimension
    is_pseudo: bool         # True for series dimension
```

### Spectrum
```python
@dataclass
class Spectrum:
    data: np.ndarray                    # N-dimensional array
    dimensions: tuple[DimensionInfo, ...]  # One per dimension

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def n_spectral_dims(self) -> int:
        return sum(1 for d in self.dimensions if not d.is_pseudo)

    @property
    def pseudo_dim(self) -> DimensionInfo | None:
        for d in self.dimensions:
            if d.is_pseudo:
                return d
        return None

    def get_dimension(self, label: str) -> DimensionInfo:
        """Get dimension by label (F1, F2, etc.)."""
        for d in self.dimensions:
            if d.label == label:
                return d
        raise ValueError(f"No dimension with label {label}")
```

## Parameter Naming

Parameters must use dimension labels, not `x`/`y`:
```python
def parameter_names_for_peak(n_spectral_dims: int) -> list[str]:
    """Generate parameter names for a peak."""
    params = []
    for i in range(n_spectral_dims):
        # F2, F1 for 2D; F3, F2, F1 for 3D; etc.
        label = f"F{n_spectral_dims - i}"
        params.extend([f"cs_{label}", f"lw_{label}"])
    return params

# 2D: ["cs_F2", "lw_F2", "cs_F1", "lw_F1"]
# 3D: ["cs_F3", "lw_F3", "cs_F2", "lw_F2", "cs_F1", "lw_F1"]
```

## Peak List Support

### Supported Formats

**Sparky format:**
```
      Assignment         w1         w2
           2N-H    115.631      6.867
           4N-H    123.143      7.821
```

**Sparky 3D format:**
```
      Assignment         w1         w2         w3
      G45N-H-CA    176.234    119.456      8.234
```

**NMRPipe format:**
```
VARS   INDEX X_AXIS Y_AXIS ASS
FORMAT %5d %9.3f %9.3f %s
     1   512.3   128.7 2N-H
```

### Dimension Mapping

Peak list columns must map to spectrum dimensions:
```python
@dataclass
class PeakList:
    peaks: list[Peak]
    dimension_mapping: dict[str, str]  # column_name -> dimension_label

    # Example: {"w1": "F1", "w2": "F2"} for Sparky 2D
    # Example: {"w1": "F1", "w2": "F2", "w3": "F3"} for Sparky 3D
```

### Validation
```python
def validate_peaklist_spectrum_match(
    peaklist: PeakList,
    spectrum: Spectrum
) -> None:
    """Ensure peak list dimensions match spectrum."""
    n_peaklist_dims = len(peaklist.dimension_mapping)
    n_spectral_dims = spectrum.n_spectral_dims

    if n_peaklist_dims != n_spectral_dims:
        raise ValueError(
            f"Peak list has {n_peaklist_dims} position columns, "
            f"but spectrum has {n_spectral_dims} spectral dimensions. "
            f"Peak list columns: {list(peaklist.dimension_mapping.keys())}"
        )
```

## Lineshape Models

### N-Dimensional Lineshape
```python
class NDLineshape:
    """N-dimensional separable lineshape."""

    def evaluate(
        self,
        positions: np.ndarray,      # Shape: (n_dims,) in ppm
        linewidths: np.ndarray,     # Shape: (n_dims,) in Hz
        amplitude: float,
        grid: list[np.ndarray],     # One 1D grid per dimension, same order as spectrum.dimensions
    ) -> np.ndarray:
        """Evaluate lineshape on N-dimensional grid.

        Returns array with shape (len(grid[0]), len(grid[1]), ...).
        """
        output_shape = tuple(len(g) for g in grid)
        result = np.full(output_shape, amplitude, dtype=np.float64)

        for i in range(self.n_dims):
            ls_1d = self._eval_1d(
                self.lineshape_types[i],
                positions[i],
                linewidths[i],
                grid[i]
            )
            # Reshape for broadcasting: (1, 1, ..., len(grid[i]), ..., 1)
            shape = [1] * self.n_dims
            shape[i] = len(grid[i])
            result = result * ls_1d.reshape(shape)

        return result
```

## Limitations

**Single pseudo-dimension**: Currently, PeakFit supports exactly one pseudo-dimension per spectrum (the series axis for CEST offsets, relaxation delays, etc.). Multiple pseudo-dimensions are not supported.

### Useful Helper Methods
```python
@property
def spectral_dims(self) -> list[DimensionInfo]:
    """Return only spectral (non-pseudo) dimensions."""
    return [d for d in self.dimensions if not d.is_pseudo]
```

## Code Patterns to Avoid

### [BAD] Hardcoded 2D
```python
# BAD
def fit_peak(peak, spectrum):
    x_pos = peak.position_x
    y_pos = peak.position_y
    x_lw = params["lw_x"]
    y_lw = params["lw_y"]
```

### [GOOD] Generic N-D
```python
# GOOD
def fit_peak(peak, spectrum):
    positions = peak.positions  # Array of length n_spectral_dims
    linewidths = [params[f"lw_{dim.label}"] for dim in spectrum.spectral_dims]
```

### [BAD] Dimension-Specific Loops
```python
# BAD
for x in range(spectrum.shape[0]):
    for y in range(spectrum.shape[1]):
        ...
```

### [GOOD] Generic Iteration
```python
# GOOD
for idx in np.ndindex(spectrum.shape[:n_spectral_dims]):
    ...
```

## Testing Requirements

Test cases for:

1. **1D pseudo-2D**: Array of 1D spectra
2. **2D pseudo-3D**: Standard HSQC-CEST (current main use case)
3. **3D pseudo-4D**: HNCO-based experiment

Each test should verify:
- Correct dimension detection from file
- Correct parameter names in output
- Correct peak list parsing
- Lineshape evaluation produces correct shape

## CLI Behavior
```bash
# Auto-detect dimensions (default)
peakfit fit spectrum.ft3 peaks.list

# Verbose output shows dimensions
peakfit fit spectrum.ft3 peaks.list -v

# Output:
# Spectrum: hsqc_cest.ft3
#   Total dimensions: 3 (2 spectral + 1 pseudo)
#   F2: 1H (direct), 512 pts, 14.0 ppm, SF=600.13 MHz
#   F1: 15N (indirect), 64 pts, 35.0 ppm, SF=60.81 MHz
#   Pseudo: 131 planes
# Peak list: peaks.list
#   Format: Sparky
#   Peaks: 166
#   Dimensions: w1 -> F1, w2 -> F2
```

## What NOT to Do

- Don't use `x`, `y`, `z`, `a` as dimension identifiers
- Don't assume 2 spectral dimensions anywhere
- Don't hardcode loop bounds for specific dimensionality
- Don't create separate code paths for 2D vs 3D vs 4D
- Don't embed dimension info in variable names (`x_position`, `y_linewidth`)
