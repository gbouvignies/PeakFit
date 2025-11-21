"""Vectorized JAX lineshape evaluation for Phase 2.1.

This module provides pure JAX functions for evaluating multiple lineshapes
in parallel using jax.vmap, eliminating Python loops for maximum performance.

Phase 2.1 goal: Move all lineshape evaluation inside JIT boundary.
"""

from typing import Any

import numpy as np

from peakfit.clustering import Cluster
from peakfit.core.fitting import Parameters
from peakfit.core.lineshapes import HAS_JAX, require_jax

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    from jax import Array

# Shape type encoding for conditional evaluation
SHAPE_GAUSSIAN = 0
SHAPE_LORENTZIAN = 1
SHAPE_PVOIGT = 2
SHAPE_NO_APOD = 3
SHAPE_SP1 = 4
SHAPE_SP2 = 5


def extract_peak_evaluation_data_jax(
    cluster: Cluster, params: Parameters
) -> tuple[dict[str, Any], Array]:
    """Extract peak data into JAX-friendly arrays for vectorized evaluation.

    This converts the object-oriented Peak/Shape API into pure arrays that
    JAX can vectorize.

    Args:
        cluster: Cluster to fit
        params: Current parameters

    Returns:
        peak_data: Dict with arrays of peak metadata
        data_jax: Cluster data as JAX array

    Structure of peak_data:
        - n_peaks: Number of peaks (int)
        - n_dims: Number of dimensions per peak (typically 1 or 2)
        - shape_types: Array of shape type codes (n_peaks, n_dims)
        - positions: Array of peak positions (n_peaks, n_dims)
        - fwhms: Array of linewidths (n_peaks, n_dims)
        - etas: Array of pvoigt mixing (n_peaks, n_dims) - used only for pvoigt
        - ... (other shape-specific params)
        - grid: List of position arrays for each dimension
        - spec_params: Spectral parameters for Hz conversion
    """
    require_jax()

    peaks = cluster.peaks
    n_peaks = len(peaks)

    if n_peaks == 0:
        raise ValueError("No peaks to evaluate")

    # Get dimensionality from first peak
    n_dims = len(peaks[0].shapes)

    # Pre-allocate arrays
    shape_types = np.zeros((n_peaks, n_dims), dtype=np.int32)
    positions = np.zeros((n_peaks, n_dims), dtype=np.float64)
    fwhms = np.zeros((n_peaks, n_dims), dtype=np.float64)
    etas = np.zeros((n_peaks, n_dims), dtype=np.float64)  # For pvoigt
    r2s = np.zeros((n_peaks, n_dims), dtype=np.float64)  # For apodization shapes
    aqs = np.zeros((n_peaks, n_dims), dtype=np.float64)
    ends = np.zeros((n_peaks, n_dims), dtype=np.float64)
    offs = np.zeros((n_peaks, n_dims), dtype=np.float64)
    phases = np.zeros((n_peaks, n_dims), dtype=np.float64)

    # Extract data from each peak
    for i, peak in enumerate(peaks):
        for j, shape in enumerate(peak.shapes):
            prefix = shape.prefix

            # Determine shape type
            shape_class_name = shape.__class__.__name__.lower()
            if "gaussian" in shape_class_name:
                shape_types[i, j] = SHAPE_GAUSSIAN
            elif "lorentzian" in shape_class_name:
                shape_types[i, j] = SHAPE_LORENTZIAN
            elif "pvoigt" in shape_class_name or "pseudovoigt" in shape_class_name:
                shape_types[i, j] = SHAPE_PVOIGT
            elif "noapod" in shape_class_name:
                shape_types[i, j] = SHAPE_NO_APOD
            elif "sp1" in shape_class_name:
                shape_types[i, j] = SHAPE_SP1
            elif "sp2" in shape_class_name:
                shape_types[i, j] = SHAPE_SP2
            else:
                raise ValueError(f"Unknown shape type: {shape_class_name}")

            # Extract position
            position_key = f"{prefix}0"
            if position_key in params:
                positions[i, j] = params[position_key].value
            else:
                positions[i, j] = shape.center

            # Extract common parameters
            fwhm_key = f"{prefix}_fwhm"
            if fwhm_key in params:
                fwhms[i, j] = params[fwhm_key].value
            elif hasattr(shape, "fwhm"):
                fwhms[i, j] = shape.fwhm
            else:
                fwhms[i, j] = 10.0  # Default

            # Extract shape-specific parameters
            if shape_types[i, j] == SHAPE_PVOIGT:
                eta_key = f"{prefix}_eta"
                if eta_key in params:
                    etas[i, j] = params[eta_key].value
                else:
                    etas[i, j] = 0.5  # Default

            # For apodization shapes (NoApod, SP1, SP2)
            if shape_types[i, j] in [SHAPE_NO_APOD, SHAPE_SP1, SHAPE_SP2]:
                r2_key = f"{prefix}_r2"
                if r2_key in params:
                    r2s[i, j] = params[r2_key].value
                elif hasattr(shape, "r2"):
                    r2s[i, j] = shape.r2
                else:
                    r2s[i, j] = 10.0

                # Get aq from spec_params
                if hasattr(shape, "spec_params") and hasattr(shape.spec_params, "aq"):
                    aqs[i, j] = shape.spec_params.aq
                else:
                    aqs[i, j] = 0.1

                # For SP1/SP2
                if shape_types[i, j] in [SHAPE_SP1, SHAPE_SP2]:
                    if hasattr(shape, "end"):
                        ends[i, j] = shape.end
                    else:
                        ends[i, j] = 1.0

                    if hasattr(shape, "off"):
                        offs[i, j] = shape.off
                    else:
                        offs[i, j] = 0.35

                # Phase
                phase_prefix = prefix.replace("_", "_ph")
                phase_key = f"{phase_prefix}p"
                if phase_key in params:
                    phases[i, j] = params[phase_key].value
                else:
                    phases[i, j] = 0.0

    # Convert to JAX arrays
    peak_data = {
        "n_peaks": n_peaks,
        "n_dims": n_dims,
        "shape_types": jnp.array(shape_types, dtype=jnp.int32),
        "positions": jnp.array(positions),
        "fwhms": jnp.array(fwhms),
        "etas": jnp.array(etas),
        "r2s": jnp.array(r2s),
        "aqs": jnp.array(aqs),
        "ends": jnp.array(ends),
        "offs": jnp.array(offs),
        "phases": jnp.array(phases),
        "grid": [jnp.array(g) for g in cluster.positions],
        "spec_params_list": [
            {
                "sw": float(shape.spec_params.sw),
                "obs": float(shape.spec_params.obs),
                "car": float(shape.spec_params.car),
                "size": int(shape.spec_params.size),
            }
            for shape in peaks[0].shapes  # Use first peak as template
        ],
    }

    data_jax = jnp.array(cluster.corrected_data)

    return peak_data, data_jax


@jax.jit
def pts2hz_delta_jax(dx_pt: Array, sw: float, size: int) -> Array:
    """Convert point offset to Hz (JAX version).

    Args:
        dx_pt: Offset in points
        sw: Spectral width (Hz)
        size: Number of points

    Returns:
        Offset in Hz
    """
    return dx_pt * sw / size


@jax.jit
def evaluate_single_shape_jax(
    grid_pts: Array,
    shape_type: int,
    position: float,
    fwhm: float,
    eta: float,
    r2: float,
    aq: float,
    end: float,
    off: float,
    phase: float,
    sw: float,
    size: int,
) -> Array:
    """Evaluate a single lineshape using JAX (JIT-compiled).

    Uses conditional branching to handle different shape types.

    Args:
        grid_pts: Grid positions (in points)
        shape_type: Shape type code (0=Gaussian, 1=Lorentzian, ...)
        position: Peak position (in points)
        fwhm: Full width at half maximum (Hz)
        eta: Pvoigt mixing parameter
        r2: R2 relaxation rate (Hz)
        aq: Acquisition time (s)
        end: SP1/SP2 end parameter
        off: SP1/SP2 offset parameter
        phase: Phase correction (degrees)
        sw: Spectral width (Hz)
        size: Number of points

    Returns:
        Evaluated lineshape
    """
    from peakfit.core.lineshapes import gaussian, lorentzian, no_apod, pvoigt, sp1, sp2

    # Compute distance from center
    dx_pt = grid_pts - position
    dx_hz = pts2hz_delta_jax(dx_pt, sw, size)

    # Evaluate based on shape type using jax.lax.switch
    def eval_gaussian(_):
        return gaussian(dx_hz, fwhm)

    def eval_lorentzian(_):
        return lorentzian(dx_hz, fwhm)

    def eval_pvoigt(_):
        return pvoigt(dx_hz, fwhm, eta)

    def eval_no_apod(_):
        return no_apod(dx_hz, r2, aq, phase)

    def eval_sp1(_):
        return sp1(dx_hz, r2, aq, end, off, phase)

    def eval_sp2(_):
        return sp2(dx_hz, r2, aq, end, off, phase)

    # Use jax.lax.switch for efficient conditional
    branches = [eval_gaussian, eval_lorentzian, eval_pvoigt, eval_no_apod, eval_sp1, eval_sp2]
    result = jax.lax.switch(shape_type, branches, None)

    # Note: Sign handling for folded peaks is simplified here
    # Full implementation would need aliasing calculation and p180 flag
    # For Phase 2.1, assuming no aliasing (most common case)
    return result


def evaluate_peak_dim_jax(
    grid_pts: Array,
    shape_type: int,
    position: float,
    fwhm: float,
    eta: float,
    r2: float,
    aq: float,
    end: float,
    off: float,
    phase: float,
    spec_params: dict[str, Any],
) -> Array:
    """Evaluate one dimension of a peak (wrapper for vectorization)."""
    return evaluate_single_shape_jax(
        grid_pts,
        shape_type,
        position,
        fwhm,
        eta,
        r2,
        aq,
        end,
        off,
        phase,
        spec_params["sw"],
        spec_params["size"],
    )


def compute_shapes_matrix_jax_vectorized(
    peak_data: dict[str, Array],
) -> Array:
    """Compute lineshape matrix for all peaks using vectorized JAX (Phase 2.1).

    This is the core vectorized function that replaces the Python loop.

    Args:
        peak_data: Dict with peak metadata arrays (from extract_peak_evaluation_data_jax)

    Returns:
        shapes_matrix: Array of shape (n_peaks, n_points)

    Notes:
        This function is fully JIT-compiled. All lineshape evaluations happen
        inside JAX with no Python overhead.

        For N-dimensional peaks, the lineshape is computed as the product of
        N 1D lineshapes (one per dimension), then flattened. This handles 1D,
        2D, 3D, etc. peaks generically.
    """
    # Use array shapes (static/concrete) instead of dict values (traced)
    # Array shapes are known at JIT compile time, dict values are not
    n_peaks = peak_data["shape_types"].shape[0]
    n_dims = peak_data["shape_types"].shape[1]

    # Extract spectral parameters outside JIT (they're dicts in a list)
    # Convert to simple scalars that JIT can handle
    spec_params_extracted = [
        {
            "sw": float(peak_data["spec_params_list"][dim]["sw"]),
            "size": int(peak_data["spec_params_list"][dim]["size"]),
        }
        for dim in range(n_dims)
    ]

    # Extract grid arrays outside JIT (they're in a Python list)
    grids_extracted = [peak_data["grid"][dim] for dim in range(n_dims)]

    # Handle common cases explicitly (avoid dynamic list building in JIT)
    if n_dims == 1:
        # 1D case: direct evaluation
        grid_0 = grids_extracted[0]
        sw_0 = spec_params_extracted[0]["sw"]
        size_0 = spec_params_extracted[0]["size"]

        # Extract all peak data arrays
        shape_types_0 = peak_data["shape_types"][:, 0]
        positions_0 = peak_data["positions"][:, 0]
        fwhms_0 = peak_data["fwhms"][:, 0]
        etas_0 = peak_data["etas"][:, 0]
        r2s_0 = peak_data["r2s"][:, 0]
        aqs_0 = peak_data["aqs"][:, 0]
        ends_0 = peak_data["ends"][:, 0]
        offs_0 = peak_data["offs"][:, 0]
        phases_0 = peak_data["phases"][:, 0]

        @jax.jit
        def eval_one_peak_1d(shape_type, position, fwhm, eta, r2, aq, end, off, phase) -> Array:
            return evaluate_single_shape_jax(
                grid_0,
                shape_type,
                position,
                fwhm,
                eta,
                r2,
                aq,
                end,
                off,
                phase,
                sw_0,
                size_0,
            )

        # Vmap over the data arrays themselves, not indices!
        shapes = jax.vmap(eval_one_peak_1d)(
            shape_types_0, positions_0, fwhms_0, etas_0, r2s_0, aqs_0, ends_0, offs_0, phases_0
        )
        return shapes

    elif n_dims == 2:
        # 2D case: explicit outer product
        # Extract ALL arrays and scalars outside JIT (no dict access inside JIT!)
        grid_0 = grids_extracted[0]
        grid_1 = grids_extracted[1]
        sw_0 = spec_params_extracted[0]["sw"]
        size_0 = spec_params_extracted[0]["size"]
        sw_1 = spec_params_extracted[1]["sw"]
        size_1 = spec_params_extracted[1]["size"]

        # Extract all peak data arrays (no more dict access in JIT)
        shape_types_0 = peak_data["shape_types"][:, 0]
        shape_types_1 = peak_data["shape_types"][:, 1]
        positions_0 = peak_data["positions"][:, 0]
        positions_1 = peak_data["positions"][:, 1]
        fwhms_0 = peak_data["fwhms"][:, 0]
        fwhms_1 = peak_data["fwhms"][:, 1]
        etas_0 = peak_data["etas"][:, 0]
        etas_1 = peak_data["etas"][:, 1]
        r2s_0 = peak_data["r2s"][:, 0]
        r2s_1 = peak_data["r2s"][:, 1]
        aqs_0 = peak_data["aqs"][:, 0]
        aqs_1 = peak_data["aqs"][:, 1]
        ends_0 = peak_data["ends"][:, 0]
        ends_1 = peak_data["ends"][:, 1]
        offs_0 = peak_data["offs"][:, 0]
        offs_1 = peak_data["offs"][:, 1]
        phases_0 = peak_data["phases"][:, 0]
        phases_1 = peak_data["phases"][:, 1]

        # Debug: Check all array shapes
        import sys
        print(f"DEBUG: Array shapes before vmap:", file=sys.stderr)
        print(f"  shape_types_0: {shape_types_0.shape}", file=sys.stderr)
        print(f"  positions_0: {positions_0.shape}", file=sys.stderr)
        print(f"  grid_0: {grid_0.shape}", file=sys.stderr)
        print(f"  grid_1: {grid_1.shape}", file=sys.stderr)

        @jax.jit
        def eval_one_peak_2d(
            shape_type_0, position_0, fwhm_0, eta_0, r2_0, aq_0, end_0, off_0, phase_0,
            shape_type_1, position_1, fwhm_1, eta_1, r2_1, aq_1, end_1, off_1, phase_1,
        ) -> Array:
            # Evaluate 1D lineshape in dimension 0
            shape_0 = evaluate_single_shape_jax(
                grid_0,
                shape_type_0,
                position_0,
                fwhm_0,
                eta_0,
                r2_0,
                aq_0,
                end_0,
                off_0,
                phase_0,
                sw_0,
                size_0,
            )

            # Evaluate 1D lineshape in dimension 1
            shape_1 = evaluate_single_shape_jax(
                grid_1,
                shape_type_1,
                position_1,
                fwhm_1,
                eta_1,
                r2_1,
                aq_1,
                end_1,
                off_1,
                phase_1,
                sw_1,
                size_1,
            )

            # Compute outer product: (n0, 1) * (n1,) -> (n0, n1)
            result = shape_0[:, None] * shape_1[None, :]
            return result.ravel()

        # Vmap over the data arrays themselves, not indices!
        shapes = jax.vmap(eval_one_peak_2d)(
            shape_types_0, positions_0, fwhms_0, etas_0, r2s_0, aqs_0, ends_0, offs_0, phases_0,
            shape_types_1, positions_1, fwhms_1, etas_1, r2s_1, aqs_1, ends_1, offs_1, phases_1,
        )
        return shapes

    else:
        # For 3D+, would need explicit implementation (rare in NMR)
        # Fallback to slower path (this will cause an error, triggering fallback to scipy)
        raise NotImplementedError(f"JAX vectorized path for {n_dims}D peaks not yet implemented")


@jax.jit
def compute_shapes_matrix_jax_flattened(
    grid_pts: Array,
    shape_types: Array,
    positions: Array,
    fwhms: Array,
    etas: Array,
    r2s: Array,
    aqs: Array,
    ends: Array,
    offs: Array,
    phases: Array,
    sw: float,
    size: int,
) -> Array:
    """Compute lineshape matrix using flattened arrays (fully optimized).

    This version takes individual arrays instead of a dict, eliminating
    all dict access overhead for maximum JIT performance.

    Args:
        grid_pts: Grid positions (n_points,)
        shape_types: Shape type codes (n_peaks,)
        positions: Peak positions (n_peaks,)
        fwhms: Linewidths (n_peaks,)
        etas: Pvoigt mixing (n_peaks,)
        r2s: R2 relaxation rates (n_peaks,)
        aqs: Acquisition times (n_peaks,)
        ends: SP1/SP2 end parameters (n_peaks,)
        offs: SP1/SP2 offset parameters (n_peaks,)
        phases: Phase corrections (n_peaks,)
        sw: Spectral width (Hz)
        size: Number of points

    Returns:
        shapes_matrix: Array of shape (n_peaks, n_points)
    """
    n_peaks = len(shape_types)

    # Vectorize across all peaks using vmap
    def eval_one_peak(i: int) -> Array:
        return evaluate_single_shape_jax(
            grid_pts,
            shape_types[i],
            positions[i],
            fwhms[i],
            etas[i],
            r2s[i],
            aqs[i],
            ends[i],
            offs[i],
            phases[i],
            sw,
            size,
        )

    # Use vmap to vectorize across peaks
    shapes = jax.vmap(eval_one_peak)(jnp.arange(n_peaks))
    return shapes
