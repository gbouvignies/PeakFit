from pathlib import Path

import lmfit as lf
import numpy as np

from peakfit.cli import Arguments
from peakfit.clustering import Cluster
from peakfit.computing import calculate_shape_heights
from peakfit.messages import print_writing_profiles, print_writing_shifts
from peakfit.peak import Peak
from peakfit.typing import FloatArray


def write_profiles(
    path: Path,
    z_values: np.ndarray,
    clusters: list[Cluster],
    params: lf.Parameters,
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
    """Print the heights and errors."""
    result = f"# {'Z':>10s}  {'I':>14s}  {'I_err':>14s}\n"
    result += "\n".join(
        f"  {z!s:>10s}  {ampl:14.6e}  {ampl_e:14.6e}"
        for z, ampl, ampl_e in zip(z_values, heights, height_err, strict=False)
    )
    return result


def write_profile(
    path: Path,
    peak: Peak,
    params: lf.Parameters,
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


def write_shifts(peaks: list[Peak], params: lf.Parameters, file_shifts: Path) -> None:
    """Write the shifts to the output file."""
    print_writing_shifts()
    shifts = {peak.name: peak.positions for peak in peaks}
    with file_shifts.open("w") as f:
        for peak in peaks:
            peak.update_positions(params)
            name = peak.name
            shifts = " ".join(f"{position:10.5f}" for position in peak.positions)
            f.write(f"{name:>15s} {shifts}\n")
