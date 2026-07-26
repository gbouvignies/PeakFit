"""Pre-fit setup summary."""

import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING

from peakfit.ui.branding import show_command_summary

if TYPE_CHECKING:
    from peakfit.engine.domain.config import PeakFitConfig
    from peakfit.fit.fitting import LoadedData


def _format_workers(workers: int) -> str:
    """Format workers count for display."""
    if workers == -1:
        cpu_count = multiprocessing.cpu_count()
        return f"{cpu_count} parallel"
    if workers == 1:
        return "1 sequential"
    return f"{workers} parallel"


def _format_noise_source(noise_source: str) -> str:
    """Format noise source for display."""
    if noise_source.lower() in ("estimated", "auto"):
        return "Estimated"
    return "User-specified"


def _summarize_spectra(loaded_data: LoadedData) -> tuple[str, int]:
    """Return (shape_type, n_series) for display."""
    data = loaded_data.spectra.data
    n_series = data.shape[0]
    return "Pseudo-ND", n_series


def _describe_method(optimizer: str) -> str:
    opt = optimizer.lower()
    if opt == "varpro":
        return "VARPRO (Variable Projection)"
    if opt == "basin_hopping":
        return "Basin Hopping (Global)"
    return optimizer.upper()


def _format_contour(config: PeakFitConfig, noise_val: float) -> str:
    if config.clustering.contour_level is not None:
        return f"{config.clustering.contour_level:.2e}"

    factor = config.clustering.contour_factor
    contour_val = factor * noise_val
    return f"{contour_val:.2e} ({factor:g} x noise)"


def _relative_path(path: Path) -> str:
    cwd = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(cwd))
    except Exception:
        return str(resolved)


def show_prefit_check(
    loaded_data: LoadedData,
    output_dir: Path,
    optimizer: str,
    config: PeakFitConfig,
    spectrum_path: Path,
    peaklist_path: Path | None,
    workers: int | str = 1,
) -> None:
    """Display the pre-fit setup panel.

    Shows a boxed summary of:
    - Program name and version
    - Input files with metadata
    - Configuration parameters
    - Output directory

    This is an informational display only; fitting starts immediately after.
    """
    shape_type, n_series = _summarize_spectra(loaded_data)
    n_peaks = len(loaded_data.peaks)
    n_clusters = len(loaded_data.clusters) if loaded_data.clusters else 0
    noise_val = loaded_data.noise
    noise_source = loaded_data.noise_source

    contour_str = _format_contour(config, noise_val)
    method_str = _describe_method(optimizer)

    refine = config.fitting.refine_iterations
    refine_str = f"{refine} iteration{'s' if refine != 1 else ''}"

    worker_int = int(workers) if isinstance(workers, int) else -1
    worker_str = _format_workers(worker_int)

    rel_output = _relative_path(output_dir)

    peaklist_label = (
        f"Auto-detected ({n_peaks} peaks)"
        if peaklist_path is None
        else f"{peaklist_path.name} ({n_peaks} peaks)"
    )

    show_command_summary(
        "Fitting",
        sections=[
            (
                "Run Setup",
                {
                    "Method": method_str,
                    "Contour": contour_str,
                    "Clusters": str(n_clusters),
                    "Refine": refine_str,
                    "Workers": worker_str,
                },
            ),
            (
                "Input Files",
                {
                    "Spectrum": f"{spectrum_path.name} ({n_series} spectra, {shape_type})",
                    "Peak list": peaklist_label,
                    "Noise": f"{noise_val:.2e} ({_format_noise_source(noise_source)})",
                },
            ),
            ("Output", {"Directory": rel_output}),
        ],
    )
