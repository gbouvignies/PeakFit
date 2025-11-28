"""Validate command implementation."""

from __future__ import annotations

from pathlib import Path  # Required at runtime by Typer  # noqa: TC003
from typing import Annotated

import typer  # Required at runtime by Typer

from peakfit.ui import (
    console,
    error,
    info,
    print_summary,
    print_validation_table,
    show_banner,
    show_header,
    spacer,
    success,
    warning,
)


def validate_command(
    spectrum: Annotated[
        Path,
        typer.Argument(
            help="Path to NMRPipe spectrum file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    peaklist: Annotated[
        Path,
        typer.Argument(
            help="Path to peak list file",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show banner and verbose output",
        ),
    ] = False,
) -> None:
    """Validate input files before fitting.

    Check that spectrum and peak list files are readable and compatible.
    """
    from peakfit.services.validate import ValidationService

    # Show banner based on verbosity
    show_banner(verbose)

    show_header("Validating Input Files")

    # Run validation
    result = ValidationService.validate(spectrum, peaklist)

    # Display spectrum validation
    spacer()
    info(f"Checking spectrum: [path]{spectrum.name}[/path]")
    if result.spectrum:
        success(f"Spectrum readable - Shape: {result.spectrum.shape}")
    else:
        for err in result.errors:
            if "spectrum" in err.lower():
                error(err)

    # Display peak list validation
    spacer()
    info(f"Checking peak list: [path]{peaklist.name}[/path]")
    if result.peaks:
        success(f"Peak list readable - {len(result.peaks)} peaks found")
    else:
        for err in result.errors:
            if "peak" in err.lower():
                error(err)

    # Summary table
    spacer()
    print_summary(result.info, title="File Information")

    # Validation checks table
    if result.checks:
        spacer()
        checks_dict = {c.name: (c.passed, c.message) for c in result.checks}
        print_validation_table(checks_dict, title="Validation Checks")

    # Warnings
    if result.warnings:
        spacer()
        for warn in result.warnings:
            warning(warn)

    # Errors
    if result.errors:
        spacer()
        for err in result.errors:
            error(err)
        spacer()
        error("Validation failed!")
        raise SystemExit(1)

    spacer()
    success("All validation checks passed!")

    # Next steps
    spacer()
    info("Ready for fitting. Run:")
    console.print(f"    [cyan]peakfit fit {spectrum.name} {peaklist.name}[/cyan]")
    spacer()
