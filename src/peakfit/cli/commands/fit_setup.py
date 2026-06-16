"""Configuration and output setup for the fit CLI command."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_args

import typer

from peakfit.engine.domain.config import (
    ClusterConfig,
    FitConfig,
    LineshapeName,
    OutputConfig,
    OutputFormat,
    PeakFitConfig,
)
from peakfit.io.config import load_config
from peakfit.shared.constants import BASIN_HOPPING_NITER

if TYPE_CHECKING:
    from peakfit.engine.domain.peaks import Peak

VALID_OUTPUT_FORMATS = get_args(OutputFormat)
VALID_OPTIMIZERS = ("varpro", "basin_hopping")


def build_fit_config(
    config: Path | None,
    output: Path | None,
    formats: list[str] | None,
    headless: bool | None,
    noise: float | None,
    contour_level: float | None,
    lineshape: LineshapeName,
    refine: int,
    fixed: bool,
    jx: bool,
    phx: bool,
    phy: bool,
    exclude: list[int] | None,
    optimizer: str,
) -> PeakFitConfig:
    """Build fit config from CLI options or a TOML config file."""
    validate_optimizer(optimizer)
    normalized_formats = normalize_output_formats(formats)

    if config is not None:
        cfg = load_config(config)
        if output:
            cfg.output.directory = output
        if normalized_formats is not None:
            cfg.output.formats = normalized_formats
        if headless is not None:
            cfg.output.headless = headless
        if noise is not None:
            cfg.noise_level = noise
        if contour_level is not None:
            cfg.clustering.contour_level = contour_level
        return cfg

    fit_phase = []
    if phx:
        fit_phase.append("F3")
    if phy:
        fit_phase.append("F2")

    default_formats: list[OutputFormat] = ["json", "csv"]
    output_config = OutputConfig(
        formats=normalized_formats if normalized_formats is not None else default_formats,
        headless=headless if headless is not None else False,
    )
    if output:
        output_config.directory = output

    cfg = PeakFitConfig(
        fitting=FitConfig(
            lineshape=lineshape,
            refine_iterations=refine,
            fix_positions=fixed,
            fit_j_coupling=jx,
            fit_phase=fit_phase,
        ),
        clustering=ClusterConfig(contour_level=contour_level),
        output=output_config,
        noise_level=noise,
        exclude_planes=exclude or [],
    )

    if optimizer == "basin_hopping":
        cfg.fitting.max_iterations = BASIN_HOPPING_NITER

    return cfg


def validate_optimizer(optimizer: str) -> None:
    """Reject unsupported optimizer names before data loading starts."""
    if optimizer in VALID_OPTIMIZERS:
        return

    choices = ", ".join(VALID_OPTIMIZERS)
    raise typer.BadParameter(f"Unknown optimizer: {optimizer}. Choose from: {choices}.")


def normalize_output_formats(formats: list[str] | None) -> list[OutputFormat] | None:
    """Validate and normalize repeated --format values."""
    if not formats:
        return None

    allowed = set(VALID_OUTPUT_FORMATS)
    normalized = [fmt.lower() for fmt in formats]
    invalid = sorted({fmt for fmt in normalized if fmt not in allowed})
    if invalid:
        allowed_text = ", ".join(VALID_OUTPUT_FORMATS)
        invalid_text = ", ".join(invalid)
        raise typer.BadParameter(
            f"Unknown output format(s): {invalid_text}. Choose from: {allowed_text}."
        )

    deduped = list(dict.fromkeys(normalized))
    return cast("list[OutputFormat]", deduped)


def resolve_output_dir(fit_config: PeakFitConfig) -> Path:
    """Resolve output directory and apply timestamp policy."""
    base = fit_config.output.directory or Path("./")
    output_dir = Path(base)
    if fit_config.output.include_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if str(output_dir) == ".":
            output_dir = Path(f"output_{timestamp}")
        else:
            output_dir = output_dir / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    fit_config.output.directory = output_dir
    fit_config.output.include_timestamp = False
    return output_dir


def write_autopicked_peaklist(output_dir: Path, peaks: list[Peak]) -> Path:
    """Write auto-detected peaks to Sparky-like list format for reproducibility."""
    peaklist_dir = output_dir / "metadata"
    peaklist_dir.mkdir(parents=True, exist_ok=True)
    peaklist_path = peaklist_dir / "autopicked.list"

    columns = [f"w{i + 1}" for i in range(len(peaks[0].positions))] if peaks else []
    header = "Assignment" if not columns else f"Assignment {' '.join(columns)}"

    lines = [header]
    for peak in peaks:
        positions = " ".join(f"{float(pos):.6f}" for pos in peak.positions)
        lines.append(f"{peak.name} {positions}")

    peaklist_path.write_text("\n".join(lines) + "\n")
    return peaklist_path
