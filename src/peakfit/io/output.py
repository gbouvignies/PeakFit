"""Output file writers for peak fitting results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from peakfit.core.domain.cluster import Cluster
from peakfit.core.domain.peaks import Peak
from peakfit.core.fitting.computation import calculate_shape_heights
from peakfit.core.fitting.parameters import Parameters
from peakfit.core.shared.reporter import NullReporter, Reporter
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
        _shapes, amplitudes = calculate_shape_heights(params, cluster)
        amplitudes_err = np.full_like(amplitudes, args.noise)
        for i, peak in enumerate(cluster.peaks):
            write_profile(
                path,
                peak,
                params,
                z_values,
                amplitudes[i],
                amplitudes_err[i],
            )


def print_heights(z_values: np.ndarray, heights: FloatArray, height_err: FloatArray) -> str:
    """Print the heights and errors.

    Raises:
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
