"""UI messages and status indicators."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

from .console import REPO_URL, console, display_path, hr, icon
from .logging import log, log_section
from .panels import create_panel

_SECONDS_PER_MINUTE = 60
_DEFAULT_SEPARATOR_WIDTH = 60
_MAX_MATCHING_FILES_SHOWN = 10

__all__ = [
    "action",
    "bullet",
    "error",
    "info",
    "log_section",
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
_state: dict[str, Any] = {"logger": None}


def show_header(text: str, do_log: bool = True) -> None:
    """Display a prominent section header."""
    console.print(hr(style="panel.border"))
    console.print(f"[header]  {text}[/header]")
    console.print(hr(style="panel.border"))
    console.print()
    if do_log:
        log_section(text)


def show_subheader(text: str) -> None:
    """Display a standard subheader."""
    console.print(f"\n[subheader]{text}[/subheader]")
    console.print(hr(style="box.border"))


def subsection_header(title: str) -> None:
    """Print subsection header with correct spacing."""
    console.print()
    console.print(f"[emphasis]{title}[/emphasis]")
    console.print()


def success(message: str, indent: int = 0, do_log: bool = True) -> None:
    """Display a success message."""
    spaces = "  " * indent
    console.print(f"{spaces}[success]{icon('check')}[/success] {message}")
    if do_log:
        log(message)


def warning(message: str, indent: int = 0, do_log: bool = True) -> None:
    """Display a warning message."""
    spaces = "  " * indent
    console.print(f"{spaces}[warning]{icon('warn')}[/warning]  {message}")
    if do_log:
        log(message, level="warning")


def error(message: str, indent: int = 0, do_log: bool = True) -> None:
    """Display an error message."""
    spaces = "  " * indent
    console.print(f"{spaces}[error]{icon('error')}[/error] {message}")
    if do_log:
        log(message, level="error")


def info(message: str, indent: int = 0, do_log: bool = True) -> None:
    """Display an info message."""
    spaces = "  " * indent
    console.print(f"{spaces}[dim]{icon('info')}[/dim] {message}")
    if do_log:
        log(message)


def action(message: str) -> None:
    """Display an action/process message with visual separation."""
    console.print(f"\n[info]{icon('info')} {message}[/info]")


def bullet(message: str, indent: int = 1, style: str = "default") -> None:
    """Display a bullet point item."""
    spaces = "  " * indent
    if style == "success":
        _icon = f"[success]{icon('bullet')}[/success]"
    elif style == "warning":
        _icon = f"[warning]{icon('bullet')}[/warning]"
    elif style == "error":
        _icon = f"[error]{icon('bullet')}[/error]"
    else:
        _icon = f"[info]{icon('bullet')}[/info]"
    console.print(f"{spaces}{_icon} {message}")


def spacer() -> None:
    """Print an empty line for visual spacing."""
    console.print()


def separator(char: str = "─", width: int = _DEFAULT_SEPARATOR_WIDTH, style: str = "dim") -> None:
    """Print a visual separator line.

    Note: width/char are kept for backward-compat; default uses console width.
    """
    if width and char:
        console.print(f"[{style}]{char * width}[/{style}]")
    else:
        console.print(hr(style=style))


def show_footer(start_time: datetime, end_time: datetime) -> None:
    """Show completion footer with timing information."""
    runtime = (end_time - start_time).total_seconds()

    # Format runtime
    if runtime < _SECONDS_PER_MINUTE:
        runtime_str = f"{runtime:.1f}s"
    else:
        minutes = int(runtime // _SECONDS_PER_MINUTE)
        seconds = int(runtime % _SECONDS_PER_MINUTE)
        runtime_str = f"{minutes}m {seconds}s"

    console.print("\n" + hr())
    completed = end_time.strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[success]{icon('check')}[/success] [dim]Completed:[/dim] {completed}")
    console.print(f"[dim]Total runtime:[/dim] [metric]{runtime_str}[/metric]")

    # Log completion
    if _state["logger"]:
        log("=" * 60)
        log(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Total runtime: {runtime_str}")
        log("=" * 60)


def show_error_with_details(
    context: str,
    err: Exception,
    style_overrides: dict[str, Any] | None = None,
    suggestion: str | None = None,
) -> None:
    """Display an error with details in a panel."""
    error(f"{context} failed")

    error_panel = create_panel(
        f"[error]{type(err).__name__}[/error]: {err!s}",
        title="[panel.title]Error Details[/panel.title]",
        style="error",
    )
    console.print(error_panel)

    if suggestion:
        info(f"Suggestion: {suggestion}")

    console.print(f"\n[dim]See documentation: {REPO_URL}/docs[/dim]")


def show_file_not_found(
    filepath: Path,
    similar_files: list[Path] | None = None,
) -> None:
    """Show file not found error with suggestions."""
    error(f"File not found: [path]{display_path(filepath)}[/path]")

    if similar_files:
        info("Did you mean one of these?")
        for file in similar_files[:5]:
            console.print(f"  • [path]{display_path(file)}[/path]")

    parent = filepath.parent if filepath.parent.exists() else Path()
    if parent.is_dir():
        pattern = f"*{filepath.suffix}" if filepath.suffix else "*"
        matching_files = list(parent.glob(pattern))
        if matching_files and not similar_files:
            console.print(f"\n[dim]Available {pattern} files in {display_path(parent)}:[/dim]")
            for file in matching_files[:_MAX_MATCHING_FILES_SHOWN]:
                console.print(f"  • [value]{file.name}[/value]")
            if len(matching_files) > _MAX_MATCHING_FILES_SHOWN:
                remaining = len(matching_files) - _MAX_MATCHING_FILES_SHOWN
                console.print(f"  [dim]... and {remaining} more[/dim]")


def print_next_steps(steps: list[str]) -> None:
    """Print suggested next steps for the user."""
    console.print(f"\n[header]{icon('bullet')} Next steps:[/header]")
    for i, step in enumerate(steps, 1):
        console.print(f"  {i}. {step}")
    console.print()
