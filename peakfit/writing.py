from pathlib import Path

import lmfit as lf
import numpy as np

from peakfit.clustering import Cluster
from peakfit.peak import Peak


def write_profiles(
    path: Path,
    z_values: np.ndarray,
    cluster: Cluster,
    params: lf.Parameters,
    heights: np.ndarray,
    height_err: np.ndarray,
) -> None:
    """Write profile information to output files."""
    for i, peak in enumerate(cluster.peaks):
        write_profile(
            path,
            peak,
            params,
            z_values,
            heights[i],
            height_err[i],
        )


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
        f.write(f"# Name: {peak.name}\n")
        f.write(peak.print(params))
        f.write("\n#---------------------------------------------\n")
        f.write(f"# {'Z':>10s}  {'I':>14s}  {'I_err':>14s}\n")
        f.write(
            "\n".join(
                f"  {z!s:>10s}  {ampl:14.6e}  {ampl_e:14.6e}"
                for z, ampl, ampl_e in zip(z_values, heights, heights_err, strict=False)
            )
        )


def write_shifts(peaks: list[Peak], shifts: dict, file_shifts: Path) -> None:
    """Write the shifts to the output file."""
    with file_shifts.open("w") as f:
        for peak in peaks:
            name = peak.name
            f.write(f"{name:>15s} {shifts[name][1]:10.5f} {shifts[name][0]:10.5f}\n")
