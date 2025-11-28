"""Legacy output format writer for backward compatibility.

This module provides writers that produce output in the original PeakFit
format for users who depend on the existing .out file structure.
All legacy output is written to the legacy/ subdirectory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.io.writers.base import WriterConfig, format_float

if TYPE_CHECKING:
    from peakfit.core.results.estimates import ClusterEstimates
    from peakfit.core.results.fit_results import FitResults

# Pattern to match position parameters: F10, F20, x0, y0, etc.
POSITION_PARAM_PATTERN = re.compile(r"(F\d+0|[xyza]0)$")


class LegacyWriter:
    """Writer for legacy .out format files.

    Produces output files compatible with the original PeakFit format:
    - {peak_name}.out: Per-peak profile with parameters and amplitudes
    - shifts.out: Chemical shift positions
    - params.out: All fitted parameters in tabular format

    All files are written to a legacy/ subdirectory.
    """

    def __init__(self, config: WriterConfig | None = None) -> None:
        """Initialize legacy writer.

        Args:
            config: Writer configuration for formatting.
        """
        self.config = config or WriterConfig()

    def write_all(self, results: FitResults, output_dir: Path) -> None:
        """Write all legacy format files.

        Args:
            results: FitResults object
            output_dir: Base output directory (legacy/ will be created inside)
        """
        legacy_dir = output_dir / "legacy"
        legacy_dir.mkdir(parents=True, exist_ok=True)

        self.write_profiles(results, legacy_dir)
        self.write_shifts(results, legacy_dir / "shifts.out")
        self.write_params(results, legacy_dir / "params.out")

    def write_profiles(self, results: FitResults, output_dir: Path) -> None:
        """Write per-peak .out files.

        Each file contains:
        - Peak parameters (position, width, etc.)
        - Amplitude table with z-values

        Args:
            results: FitResults object
            output_dir: Output directory for .out files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for cluster in results.clusters:
            # Group amplitudes by peak
            peak_amplitudes: dict[str, list] = {}
            for amp in cluster.amplitudes:
                if amp.peak_name not in peak_amplitudes:
                    peak_amplitudes[amp.peak_name] = []
                peak_amplitudes[amp.peak_name].append(amp)

            for peak_name in cluster.peak_names:
                filepath = output_dir / f"{peak_name}.out"
                self._write_peak_profile(
                    filepath,
                    peak_name,
                    cluster,
                    peak_amplitudes.get(peak_name, []),
                    results.z_values,
                )

    def _write_peak_profile(
        self,
        filepath: Path,
        peak_name: str,
        cluster: ClusterEstimates,
        amplitudes: list,
        z_values: np.ndarray | None,
    ) -> None:
        """Write a single peak's .out file."""
        prec = self.config.precision

        with filepath.open("w") as f:
            # Header with peak info
            f.write(f"# Peak: {peak_name}\n")
            f.write(f"# Cluster: {cluster.cluster_id}\n")
            f.write("#\n")

            # Parameters section
            f.write("# Parameters:\n")
            for param in cluster.lineshape_params:
                # Filter parameters for this peak or shared parameters
                if peak_name in param.name or param.is_global:
                    status = "fixed" if param.is_fixed else ""
                    f.write(
                        f"# {param.name:>20s} = "
                        f"{format_float(param.value, prec, 4):>14s} "
                        f"+/- {format_float(param.std_error, prec, 4):>14s} "
                        f"{param.unit:>6s} {status}\n"
                    )

            f.write("#---------------------------------------------\n")

            # Amplitudes section
            if amplitudes and z_values is not None:
                f.write(f"# {'Z':>10s}  {'I':>14s}  {'I_err':>14s}\n")
                # Sort by plane index
                sorted_amps = sorted(amplitudes, key=lambda a: a.plane_index)
                for amp in sorted_amps:
                    z_val = z_values[amp.plane_index] if amp.plane_index < len(z_values) else 0
                    f.write(f"  {z_val!s:>10s}  {amp.value:14.6e}  {amp.std_error:14.6e}\n")

    def write_shifts(self, results: FitResults, filepath: Path) -> None:
        """Write shifts.out file.

        Format:
        peak_name  pos_F1  pos_F2  [pos_F3...]

        Supports both new Fn convention and legacy x/y/z/a naming.

        Args:
            results: FitResults object
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        def sort_key(p: Any) -> tuple[int, int | str]:
            """Sort position parameters by dimension label."""
            match = POSITION_PARAM_PATTERN.search(p.name)
            if match:
                dim = match.group(1)
                if dim.startswith("F"):
                    # F10 -> (0, 1), F20 -> (0, 2)
                    return (0, int(dim[1:-1]))
                # x0 -> (1, 0), y0 -> (1, 1), z0 -> (1, 2), a0 -> (1, 3)
                return (1, "xyza".index(dim[0]))
            return (2, 0)

        with filepath.open("w") as f:
            for cluster in results.clusters:
                for peak_name in cluster.peak_names:
                    # Collect position parameters for this peak
                    position_params = [
                        param
                        for param in cluster.lineshape_params
                        if peak_name in param.name and POSITION_PARAM_PATTERN.search(param.name)
                    ]

                    if position_params:
                        # Sort by dimension
                        position_params.sort(key=sort_key)
                        positions_str = " ".join(f"{p.value:10.5f}" for p in position_params)
                        f.write(f"{peak_name:>15s} {positions_str}\n")

    def write_params(self, results: FitResults, filepath: Path) -> None:
        """Write params.out file with all parameters.

        Format:
        cluster  peak  name  value  error  unit  fixed

        Args:
            results: FitResults object
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        prec = self.config.precision

        with filepath.open("w") as f:
            # Header
            f.write(
                f"# {'Cluster':>8s}  {'Peak':>15s}  {'Parameter':>20s}  "
                f"{'Value':>14s}  {'Error':>14s}  {'Unit':>8s}  {'Fixed':>5s}\n"
            )

            for cluster in results.clusters:
                peak_label = ", ".join(cluster.peak_names[:2])
                if len(cluster.peak_names) > 2:
                    peak_label += "..."

                for param in cluster.lineshape_params:
                    fixed_str = "yes" if param.is_fixed else "no"
                    f.write(
                        f"  {cluster.cluster_id:>8d}  {peak_label:>15s}  {param.name:>20s}  "
                        f"{format_float(param.value, prec, 4):>14s}  "
                        f"{format_float(param.std_error, prec, 4):>14s}  "
                        f"{param.unit:>8s}  {fixed_str:>5s}\n"
                    )


def write_legacy_output(results: FitResults, output_dir: Path) -> None:
    """Convenience function to write all legacy output.

    Args:
        results: FitResults object
        output_dir: Base output directory
    """
    writer = LegacyWriter()
    writer.write_all(results, output_dir)
