"""Configuration and output setup for the fit CLI command."""

import datetime
from pathlib import Path
from typing import Any, cast, get_args

import typer
from pydantic import ValidationError

from peakfit.engine.domain.config import (
    ClusterConfig,
    FitConfig,
    LineshapeFamily,
    OutputConfig,
    OutputFormat,
    PeakFitConfig,
)
from peakfit.io.config import load_config

VALID_OUTPUT_FORMATS = get_args(OutputFormat)
FAMILIES = get_args(LineshapeFamily)
_DOUBLET_NUMBERS = (1, 3)  # J start, or J start, minimum and maximum


def build_fit_config(
    config: Path | None,
    output: Path | None,
    formats: list[str] | None,
    headless: bool | None,
    noise: float | None,
    contour_level: float | None,
    lineshapes: FitConfig,
    refine: int,
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

    default_formats: list[OutputFormat] = ["csv", "txt"]
    output_config = OutputConfig(
        formats=normalized_formats if normalized_formats is not None else default_formats,
        headless=headless if headless is not None else False,
    )
    if output:
        output_config.directory = output

    return PeakFitConfig(
        fitting=lineshapes.model_copy(update={"refine_iterations": refine}),
        clustering=ClusterConfig(contour_level=contour_level),
        output=output_config,
        noise_level=noise,
        exclude_planes=exclude or [],
    )


def lineshape_options(
    families: list[str] | None,
    doublets: list[str] | None,
    fixed_j: list[str] | None,
    phases: list[str] | None,
) -> FitConfig:
    """The lineshape model (``family`` and ``axes``) from the per-axis options.

    ``--family NAME`` sets every axis and ``--family F2=NAME`` one axis;
    ``--doublet F2=J`` or ``F2=J,MIN,MAX`` adds a doublet with its J start (Hz);
    ``--fix-j F2`` holds that doublet's J; ``--phase F2`` fits F2's phase.
    """
    family = "auto"
    axes: dict[str, dict[str, Any]] = {}

    def axis_value(option: str, value: str) -> tuple[str, str]:
        axis, sep, rest = value.partition("=")
        if not sep or not axis.strip() or not rest.strip():
            msg = f"{option} takes AXIS=VALUE, for example {option} F2=..., not {value!r}."
            raise typer.BadParameter(msg)
        return axis.strip().upper(), rest.strip()

    for value in families or []:
        axis, name = axis_value("--family", value) if "=" in value else ("", value.strip())
        if name not in FAMILIES:
            msg = f"Unknown lineshape family {name!r}. Choose from: {', '.join(FAMILIES)}."
            raise typer.BadParameter(msg)
        if axis:
            axes.setdefault(axis, {})["family"] = name
        else:
            family = name
    for value in doublets or []:
        axis, numbers = axis_value("--doublet", value)
        try:
            parts = [float(part) for part in numbers.split(",")]
        except ValueError:
            parts = []
        if len(parts) not in _DOUBLET_NUMBERS:
            msg = f"--doublet takes AXIS=J or AXIS=J,MIN,MAX in Hz, not {value!r}."
            raise typer.BadParameter(msg)
        bounds = dict(zip(("min", "max"), parts[1:], strict=True)) if len(parts) > 1 else {}
        axes.setdefault(axis, {})["doublet"] = {"j": parts[0], **bounds}
    for value in fixed_j or []:
        axis = value.strip().upper()
        doublet = axes.get(axis, {}).get("doublet")
        if doublet is None:
            msg = f"--fix-j {axis} needs a doublet on {axis}: add --doublet {axis}=J."
            raise typer.BadParameter(msg)
        doublet["vary"] = False
    for value in phases or []:
        axes.setdefault(value.strip().upper(), {})["phase"] = True
    try:
        return FitConfig.model_validate({"family": family, "axes": axes})
    except ValidationError as exc:
        messages = "; ".join(error["msg"].removeprefix("Value error, ") for error in exc.errors())
        raise typer.BadParameter(messages) from exc


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
