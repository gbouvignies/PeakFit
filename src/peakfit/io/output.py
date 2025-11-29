"""Output file writers for peak fitting results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from peakfit.core.fitting.computation import calculate_amplitudes_with_uncertainty, calculate_shapes
from peakfit.core.shared.reporter import NullReporter

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.core.domain.cluster import Cluster
    from peakfit.core.domain.peaks import Peak
    from peakfit.core.fitting.parameters import Parameters
    from peakfit.core.shared.reporter import Reporter
    from peakfit.core.shared.typing import FittingOptions, FloatArray


def write_profiles(
    path: Path,
    z_values: np.ndarray,
    clusters: list[Cluster],
    params: Parameters,
    args: FittingOptions,
    reporter: Reporter | None = None,
) -> None:
    """Write profile information to output files.

    Args:
        path: Output directory path
        z_values: Z-dimension values array
        clusters: List of clusters to write
        params: Fitting parameters
        args: Fitting options containing noise level
        reporter: Optional reporter for status messages (default: silent)
    """
    if reporter is None:
        reporter = NullReporter()

    reporter.action("Writing profiles...")
    for cluster in clusters:
        if args.noise is None:
            raise ValueError("Noise must be provided to compute amplitudes with uncertainty")
        # Compute amplitudes with proper uncertainty propagation from linear least-squares
        shapes = calculate_shapes(params, cluster)
        amplitudes, amplitudes_err, _covariance = calculate_amplitudes_with_uncertainty(
            shapes, cluster.corrected_data, args.noise
        )
        for i, peak in enumerate(cluster.peaks):
            # amplitudes_err[i] is a scalar (same error for all planes)
            # We need to broadcast it to match the number of planes
            peak_amplitudes = amplitudes[i]
            n_planes = len(peak_amplitudes) if hasattr(peak_amplitudes, "__len__") else 1
            peak_errors = np.full(n_planes, amplitudes_err[i])
            write_profile(
                path,
                peak,
                params,
                z_values,
                peak_amplitudes,
                peak_errors,
            )


def print_heights(z_values: np.ndarray, heights: FloatArray, height_err: FloatArray) -> str:
    """Print the heights and errors.

    Raises
    ------
        ValueError: If array lengths don't match
    """
    if not (len(z_values) == len(heights) == len(height_err)):
        msg = (
            f"Array length mismatch: z_values={len(z_values)}, "
            f"heights={len(heights)}, height_err={len(height_err)}"
        )
        raise ValueError(msg)

    result = f"# {'Z':>10s}  {'I':>14s}  {'I_err':>14s}\n"
    result += "\n".join(
        f"  {z!s:>10s}  {ampl:14.6e}  {ampl_e:14.6e}"
        for z, ampl, ampl_e in zip(z_values, heights, height_err, strict=True)
    )
    return result


def write_profile(
    path: Path,
    peak: Peak,
    params: Parameters,
    z_values: np.ndarray,
    heights: np.ndarray,
    heights_err: np.ndarray,
) -> None:
    """Write individual profile data to a file."""
    filename = path / f"{peak.name}.out"
    with filename.open("w") as f:
        f.write(peak.print(params))
        f.write("\n#---------------------------------------------\n")
        f.write(print_heights(z_values, heights, heights_err))


def write_shifts(
    peaks: list[Peak],
    params: Parameters,
    file_shifts: Path,
    reporter: Reporter | None = None,
) -> None:
    """Write the shifts to the output file.

    Args:
        peaks: List of peaks
        params: Fitting parameters
        file_shifts: Output file path
        reporter: Optional reporter for status messages (default: silent)
    """
    if reporter is None:
        reporter = NullReporter()

    reporter.action("Writing shifts...")
    with file_shifts.open("w") as f:
        for peak in peaks:
            peak.update_positions(params)
            name = peak.name
            positions_str = " ".join(f"{position:10.5f}" for position in peak.positions)
            f.write(f"{name:>15s} {positions_str}\n")
