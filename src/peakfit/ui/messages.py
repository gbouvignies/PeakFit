"""UI messages and status indicators."""

from rich import box
from rich.panel import Panel

from .console import REPO_URL, console, icon

__all__ = [
    "action",
    "bullet",
    "error",
    "info",
    "show_error_with_details",
    "success",
    "warning",
]


def success(message: str, indent: int = 0) -> None:
    """Display a success message."""
    spaces = "  " * indent
    console.print(f"{spaces}[success]{icon('check')}[/success] {message}")


def warning(message: str, indent: int = 0) -> None:
    """Display a warning message."""
    spaces = "  " * indent
    console.print(f"{spaces}[warning]{icon('warn')}[/warning]  {message}")


def error(message: str, indent: int = 0) -> None:
    """Display an error message."""
    spaces = "  " * indent
    console.print(f"{spaces}[error]{icon('error')}[/error] {message}")


def info(message: str, indent: int = 0) -> None:
    """Display an info message."""
    spaces = "  " * indent
    console.print(f"{spaces}[dim]{icon('info')}[/dim] {message}")


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


def show_error_with_details(
    context: str,
    err: Exception,
) -> None:
    """Display an error with details in a panel."""
    error(f"{context} failed")

    error_panel = Panel(
        f"[error]{type(err).__name__}[/error]: {err!s}",
        title="[panel.title]Error Details[/panel.title]",
        title_align="left",
        border_style="error",
        box=box.ROUNDED,
    )
    console.print(error_panel)

    console.print(f"\n[dim]See documentation: {REPO_URL}/docs[/dim]")
