"""UI table constructors used by terminal views."""

from typing import Any

from rich import box
from rich.table import Table

__all__ = [
    "create_live_metrics_table",
    "create_table",
]


def create_live_metrics_table(metrics: dict[str, Any]) -> Table:
    """Create a compact table for live metrics.

    Args:
        metrics: Dictionary of metric name -> (value, style) or just value.

    Returns:
    -------
        A Table for displaying live metrics.
    """
    table = Table(box=box.SIMPLE, show_header=False, show_edge=False, pad_edge=False)

    row_values = []
    for label, val in metrics.items():
        table.add_column(label, justify="center")

        style = "value"
        display_val = str(val)

        # Handle (value, style) tuple
        if isinstance(val, tuple):
            display_val = str(val[0])
            style = val[1]

        row_values.append(f"[key]{label}:[/key] [{style}]{display_val}[/{style}]")

    table.add_row(*row_values)
    return table


def create_table(
    title: str | None = None,
    caption: str | None = None,
    show_header: bool = True,
) -> Table:
    """Create a standard table with consistent styling.

    Args:
        title: Optional table title
        caption: Optional table caption
        show_header: Whether to show table header

    Returns:
    -------
        Configured Table instance
    """
    return Table(
        title=title,
        caption=caption,
        title_style="panel.title" if title else None,
        caption_style="neutral",
        box=box.ROUNDED,
        show_header=show_header,
        header_style="subheader",
        border_style="box.border",
        expand=False,
    )
