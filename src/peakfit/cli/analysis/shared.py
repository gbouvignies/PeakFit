"""Shared utilities for analysis CLI commands."""

from pathlib import Path

from peakfit.core.domain.peaks import Peak
from peakfit.core.domain.state import FittingState
from peakfit.core.fitting.parameters import Parameters
from peakfit.services.analyze import FittingStateService, StateFileMissingError, StateLoadError
from peakfit.ui import error, info


def load_fitting_state(results_dir: Path, verbose: bool = True) -> FittingState:
    """Load fitting state from results directory.

    Args:
        results_dir: Path to results directory containing .peakfit_state.pkl
        verbose: Whether to print the state summary (deprecated, now handled by caller)

    Returns
    -------
        FittingState with clusters, params, noise, and peaks
    """
    try:
        loaded_state = FittingStateService.load(results_dir)
    except StateFileMissingError as exc:
        error(f"No fitting state found in {results_dir}")
        info("Run 'peakfit fit' with --save-state (enabled by default)")
        raise SystemExit(1) from exc
    except StateLoadError as exc:  # pragma: no cover - safety guard
        error(str(exc))
        raise SystemExit(1) from exc

    return loaded_state.state


def _update_output_files(results_dir: Path, params: Parameters, peaks: list[Peak]) -> None:
    """Update .out files with new uncertainty estimates."""
    for peak in peaks:
        out_file = results_dir / f"{peak.name}.out"
        if out_file.exists():
            # Read existing file
            lines = out_file.read_text().splitlines()

            # Update parameter lines with new stderr
            new_lines = []
            for line in lines:
                if line.startswith("# ") and ":" in line and "±" in line:
                    # Parse parameter line
                    parts = line.split(":")
                    if len(parts) >= 2:
                        param_part = parts[0].strip("# ").strip()
                        # Find matching parameter
                        for shape in peak.shapes:
                            for param_name in shape.param_names:  # type: ignore[attr-defined]
                                if (
                                    param_name.endswith(param_part) or param_part in param_name
                                ) and param_name in params:
                                    value = params[param_name].value
                                    stderr = params[param_name].stderr
                                    shortname = param_part
                                    updated_line = (
                                        f"# {shortname:<10s}: {value:10.5f} ± {stderr:10.5f}"
                                    )
                                    line = updated_line
                                    break
                new_lines.append(line)

            out_file.write_text("\n".join(new_lines))
