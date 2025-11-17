from pathlib import Path

import numpy as np

from peakfit.cli_legacy import Arguments
from peakfit.clustering import Cluster
from peakfit.computing import calculate_shape_heights
from peakfit.core.fitting import Parameters
from peakfit.messages import print_writing_profiles, print_writing_shifts
from peakfit.peak import Peak
from peakfit.typing import FloatArray


def write_profiles(
    path: Path,
    z_values: np.ndarray,
    clusters: list[Cluster],
    params: Parameters,
    args: Arguments,
) -> None:
    """Write profile information to output files."""
    print_writing_profiles()
    for cluster in clusters:
        shapes, amplitudes = calculate_shape_heights(params, cluster)
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


def print_heights(
    z_values: np.ndarray, heights: FloatArray, height_err: FloatArray
) -> str:
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


def write_shifts(peaks: list[Peak], params: Parameters, file_shifts: Path) -> None:
    """Write the shifts to the output file."""
    print_writing_shifts()
    with file_shifts.open("w") as f:
        for peak in peaks:
            peak.update_positions(params)
            name = peak.name
            positions_str = " ".join(f"{position:10.5f}" for position in peak.positions)
            f.write(f"{name:>15s} {positions_str}\n")
