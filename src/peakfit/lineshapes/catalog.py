"""Batch computation system for lineshapes - all shapes treated equally."""

import numpy as np

from peakfit.lineshapes import functions
from peakfit.typing import FloatArray, IntArray

try:
    from peakfit.fitting.parameters import Parameters
except ImportError:
    Parameters = object  # type: ignore[misc, assignment]


def batch_evaluate_with_catalog(shapes: list, x_pt: IntArray, params: "Parameters") -> FloatArray:
    """Batch evaluation - groups shapes by type and calls appropriate compute_all_* function.

    All shapes are treated symmetrically - no special cases.

    Args:
        shapes: List of shape instances (can be mixed types)
        x_pt: Point indices
        params: Parameters object

    Returns:
        (n_shapes, n_points) array
    """
    if not shapes:
        return np.empty((0, len(x_pt)))

    # Extract once
    parvals = params.valuesdict()
    spec_params = shapes[0].spec_params
    positions_hz = spec_params.pts2hz(x_pt)

    # Group by shape type
    shape_groups: dict[str, list[tuple[int, object]]] = {}
    for idx, shape in enumerate(shapes):
        shape_name = shape.shape_name
        shape_groups.setdefault(shape_name, []).append((idx, shape))

    # Allocate result
    result = np.empty((len(shapes), len(x_pt)), dtype=np.float64)

    # Registry: shape_name -> (compute_func, param_builder)
    def _params_gaussian(group_shapes, parvals, spec):
        fwhms = np.array([parvals[f"{s._prefix}_fwhm"] for s in group_shapes])
        return (fwhms,)

    def _params_lorentzian(group_shapes, parvals, spec):
        fwhms = np.array([parvals[f"{s._prefix}_fwhm"] for s in group_shapes])
        return (fwhms,)

    def _params_pvoigt(group_shapes, parvals, spec):
        fwhms = np.array([parvals[f"{s._prefix}_fwhm"] for s in group_shapes])
        etas = np.array([parvals[f"{s._prefix}_eta"] for s in group_shapes])
        return (fwhms, etas)

    def _params_no_apod(group_shapes, parvals, spec):
        r2s = np.array([parvals[f"{s._prefix}_r2"] for s in group_shapes])
        phases = np.array([parvals.get(f"{s._prefix}_phase", 0.0) for s in group_shapes])
        aqs = np.full_like(r2s, float(spec.aq_time))
        return (r2s, aqs, phases)

    def _params_sp(group_shapes, parvals, spec):
        r2s = np.array([parvals[f"{s._prefix}_r2"] for s in group_shapes])
        phases = np.array([parvals.get(f"{s._prefix}_phase", 0.0) for s in group_shapes])
        aqs = np.full_like(r2s, float(spec.aq_time))
        ends = np.full_like(r2s, float(spec.apodq2))
        offs = np.full_like(r2s, float(spec.apodq1))
        return (r2s, aqs, ends, offs, phases)

    REGISTRY: dict[str, tuple] = {
        "gaussian": (functions.compute_all_gaussian_shapes, _params_gaussian),
        "lorentzian": (functions.compute_all_lorentzian_shapes, _params_lorentzian),
        "pvoigt": (functions.compute_all_pvoigt_shapes, _params_pvoigt),
        "no_apod": (functions.compute_all_no_apod_shapes, _params_no_apod),
        "sp1": (functions.compute_all_sp1_shapes, _params_sp),
        "sp2": (functions.compute_all_sp2_shapes, _params_sp),
    }

    # Compute each group via registry
    for shape_name, group in shape_groups.items():
        indices = [idx for idx, _ in group]
        group_shapes = [shape for _, shape in group]

        centers = np.array([parvals[f"{s._prefix}0"] * spec_params.obs for s in group_shapes])
        j_couplings = np.array([parvals.get(f"{s._prefix}_j", 0.0) for s in group_shapes])

        if shape_name not in REGISTRY:
            raise ValueError(f"Unknown shape: {shape_name}")

        compute_func, param_builder = REGISTRY[shape_name]
        param_arrays = param_builder(group_shapes, parvals, spec_params)
        # Call using the wrapper signatures (j_couplings last)
        group_result = compute_func(positions_hz, centers, *param_arrays, j_couplings)

        # Place results at original indices
        for i, idx in enumerate(indices):
            result[idx] = group_result[i]

    return result


# Legacy compatibility
class ShapeComputationCatalog:
    """Legacy wrapper for batch_evaluate_with_catalog."""

    def __init__(self, shapes, x_pt, params, chunk_size=100):
        self.shapes = shapes
        self.x_pt = x_pt
        self.params = params
        self.chunk_size = chunk_size
        self.n_shapes = len(shapes)
        self.n_points = len(x_pt)
        self.results = np.empty((self.n_shapes, self.n_points))

    def compute_all(self):
        self.results[:] = batch_evaluate_with_catalog(self.shapes, self.x_pt, self.params)
        return self.results

    def compute_all_chunked(self, chunk_size=None):
        return self.compute_all()

    def get_statistics(self):
        return {
            "n_shapes": self.n_shapes,
            "n_points": self.n_points,
            "chunk_size": self.chunk_size,
        }
