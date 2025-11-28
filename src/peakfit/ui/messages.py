"""UI messages and status indicators.

This module provides functions for displaying status messages, headers,
footers, and other text-based UI elements with consistent styling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging
    from datetime import datetime

from .console import REPO_URL, console
from .logging import log, log_section

__all__ = [
    "action",
    "bullet",
    "error",
    "info",
    "print_next_steps",
    "separator",
    "show_error_with_details",
    "show_file_not_found",
    "show_footer",
    "show_header",
    "show_subheader",
    "spacer",
    "subsection_header",
    "success",
    "warning",
]

# Module-level state for logger
_state: dict = {"logger": None}


def set_logger(logger: logging.Logger | None) -> None:
    """Set the module-level logger reference.

    Args:
        logger: Logger instance to use for file logging
    """
    _state["logger"] = logger


def show_header(text: str, do_log: bool = True) -> None:
    """Display a prominent section header with consistent spacing.

    Spacing: Calling code must add ONE blank line before. ZERO blank lines after.

    Args:
        text: Header text to display
        do_log: Whether to log this header to file
    """
    console.print("[bold cyan]" + "━" * 60 + "[/bold cyan]")
    console.print(f"[bold cyan]  {text}[/bold cyan]")
    console.print("[bold cyan]" + "━" * 60 + "[/bold cyan]")
    if do_log:
        log_section(text)


def show_subheader(text: str) -> None:
    """Display a standard subheader.

    Args:
        text: Subheader text to display
    """
    console.print(f"\n[bold white]{text}[/bold white]")
    console.print("[dim]" + "─" * 40 + "[/dim]")


def subsection_header(title: str) -> None:
    """Print subsection header with correct spacing.

    Spacing: ONE blank line before, ONE blank line after.

    Args:
        title: Subsection title to display
    """
    console.print()  # ONE blank line before
    console.print(f"[bold]{title}[/bold]")
    console.print()  # ONE blank line after


def success(message: str, indent: int = 0, do_log: bool = True) -> None:
    """Display a success message.

    Args:
        message: Success message to display
        indent: Indentation level (spaces = indent * 2)
        do_log: Whether to log this message to file
    """
    spaces = "  " * indent
    console.print(f"{spaces}[success]✓[/success] {message}")
    if do_log:
        log(message)


def warning(message: str, indent: int = 0, do_log: bool = True) -> None:
    """Display a warning message.

    Args:
        message: Warning message to display
        indent: Indentation level (spaces = indent * 2)
        do_log: Whether to log this message to file
    """
    spaces = "  " * indent
    console.print(f"{spaces}[warning]⚠[/warning]  {message}")
    if do_log:
        log(message, level="warning")


def error(message: str, indent: int = 0, do_log: bool = True) -> None:
    """Display an error message.

    Args:
        message: Error message to display
        indent: Indentation level (spaces = indent * 2)
        do_log: Whether to log this message to file
    """
    spaces = "  " * indent
    console.print(f"{spaces}[error]✗[/error] {message}")
    if do_log:
        log(message, level="error")


def info(message: str, indent: int = 0, do_log: bool = True) -> None:
    """Display an info message.

    Args:
        message: Info message to display
        indent: Indentation level (spaces = indent * 2)
        do_log: Whether to log this message to file
    """
    spaces = "  " * indent
    console.print(f"{spaces}[dim]▸[/dim] {message}")
    if do_log:
        log(message)


def action(message: str) -> None:
    """Display an action/process message with visual separation.

    Use this for ongoing operations like 'Fitting peaks...', 'Loading data...'

    Args:
        message: Action message to display
    """
    console.print(f"\n[bold yellow]—[/bold yellow] {message}")


def bullet(message: str, indent: int = 1, style: str = "default") -> None:
    """Display a bullet point item.

    Args:
        message: Message to display
        indent: Indentation level (spaces = indent * 2)
        style: Style name (success/warning/error/default)
    """
    spaces = "  " * indent
    if style == "success":
        icon = "[success]‣[/success]"
    elif style == "warning":
        icon = "[warning]‣[/warning]"
    elif style == "error":
        icon = "[error]‣[/error]"
    else:
        icon = "[cyan]‣[/cyan]"
    console.print(f"{spaces}{icon} {message}")


def spacer() -> None:
    """Print an empty line for visual spacing."""
    console.print()


def separator(char: str = "─", width: int = 60, style: str = "dim") -> None:
    """Print a visual separator line.

    Args:
        char: Character to use for separator
        width: Width of separator
        style: Rich style to apply
    """
    console.print(f"[{style}]{char * width}[/{style}]")


def show_footer(start_time: datetime, end_time: datetime) -> None:
    """Show completion footer with timing information.

    Args:
        start_time: When the program started
        end_time: When the program completed
    """
    runtime = (end_time - start_time).total_seconds()

    # Format runtime
    if runtime < 60:
        runtime_str = f"{runtime:.1f}s"
    else:
        minutes = int(runtime // 60)
        seconds = int(runtime % 60)
        runtime_str = f"{minutes}m {seconds}s"

    console.print("\n" + "━" * 70)
    console.print(
        f"[green]✓[/green] [dim]Completed:[/dim] {end_time.strftime("%Y-%m-%d %H:%M:%S")} | "
        f"[dim]Total runtime:[/dim] [cyan]{runtime_str}[/cyan]"
    )

    # Log completion
    if _state["logger"]:
        log("=" * 60)
        log(f"Completed: {end_time.strftime("%Y-%m-%d %H:%M:%S")}")
        log(f"Total runtime: {runtime_str}")
        log("=" * 60)


def show_error_with_details(
    context: str,
    err: Exception,
    suggestion: str | None = None,
) -> None:
    """Display an error with details in a panel.

    Args:
        context: Context of where the error occurred
        err: The exception that was raised
        suggestion: Optional suggestion for fixing the error
    """
    from .panels import create_panel

    error(f"{context} failed")

    # Show error details in panel
    error_panel = create_panel(
        f"[error]{type(err).__name__}[/error]: {err!s}",
        title="Error Details",
        style="error",
    )
    console.print(error_panel)

    # Show suggestion if available
    if suggestion:
        info(f"Suggestion: {suggestion}")

    # Link to docs
    console.print(f"\n[dim]See documentation: {REPO_URL}/docs[/dim]")


def show_file_not_found(
    filepath: Path,
    similar_files: list[Path] | None = None,
) -> None:
    """Show file not found error with suggestions.

    Args:
        filepath: Path that was not found
        similar_files: Optional list of similar files to suggest
    """
    error(f"File not found: [path]{filepath}[/path]")

    if similar_files:
        info("Did you mean one of these?")
        for file in similar_files[:5]:
            console.print(f"  • [path]{file}[/path]")

    # Show files in current directory
    parent = filepath.parent if filepath.parent.exists() else Path()
    if parent.is_dir():
        pattern = f"*{filepath.suffix}" if filepath.suffix else "*"
        matching_files = list(parent.glob(pattern))
        if matching_files and not similar_files:
            console.print(f"\n[dim]Available {pattern} files in {parent}:[/dim]")
            for file in matching_files[:10]:
                console.print(f"  • [green]{file.name}[/]")
            if len(matching_files) > 10:
                console.print(f"  [dim]... and {len(matching_files) - 10} more[/]")


def print_next_steps(steps: list[str]) -> None:
    """Print suggested next steps for the user.

    Args:
        steps: List of suggested commands or actions
    """
    console.print("\n[bold cyan]📋 Next steps:[/]")
    for i, step in enumerate(steps, 1):
        console.print(f"  {i}. {step}")
    console.print()
