"""Configuration and output setup for the fit CLI command."""

import datetime
from pathlib import Path
from typing import cast, get_args

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

VALID_OUTPUT_FORMATS = get_args(OutputFormat)


def build_fit_config(
    config: Path | None,
    output: Path | None,
    formats: list[str] | None,
    headless: bool | None,
    noise: float | None,
    contour_level: float | None,
    lineshape: LineshapeName,
    refine: int,
    phx: bool,
    phy: bool,
    exclude: list[int] | None,
) -> PeakFitConfig:
    """Build fit config from CLI options or a TOML config file."""
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

    default_formats: list[OutputFormat] = ["csv", "txt"]
    output_config = OutputConfig(
        formats=normalized_formats if normalized_formats is not None else default_formats,
        headless=headless if headless is not None else False,
    )
    if output:
        output_config.directory = output

    return PeakFitConfig(
        fitting=FitConfig(
            lineshape=lineshape,
            refine_iterations=refine,
            fit_phase=fit_phase,
        ),
        clustering=ClusterConfig(contour_level=contour_level),
        output=output_config,
        noise_level=noise,
        exclude_planes=exclude or [],
    )


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
        timestamp = datetime.datetime.now(datetime.UTC).astimezone().strftime("%Y%m%d_%H%M%S")
        if str(output_dir) == ".":
            output_dir = Path(f"output_{timestamp}")
        else:
            output_dir = output_dir / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    fit_config.output.directory = output_dir
    fit_config.output.include_timestamp = False
    return output_dir
