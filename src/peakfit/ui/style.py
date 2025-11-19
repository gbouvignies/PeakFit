"""
Centralized UI style definitions for consistent terminal output.
All terminal output MUST use these styles for consistency.
"""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from peakfit import __version__

# Define custom theme for consistent colors
PEAKFIT_THEME = Theme(
    {
        # Status colors
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "info": "cyan",
        # UI elements
        "header": "bold cyan",
        "subheader": "bold white",
        "emphasis": "bold",
        "dim": "dim",
        "code": "bold magenta",
        # Values
        "value": "green",
        "metric": "cyan",
        "path": "blue underline",
    }
)

# Single console instance for entire application
console = Console(theme=PEAKFIT_THEME, record=True)

# Version and branding
VERSION = __version__
REPO_URL = "https://github.com/gbouvignies/PeakFit"
LOGO_EMOJI = "🎯"

# ASCII Logo
LOGO_ASCII = r"""
   ___           _     _____ _ _
  / _ \ ___  __ _| | __|  ___(_) |_
 / /_)/ _ \/ _` | |/ /| |_  | | __|
/ ___/  __/ (_| |   < |  _| | | |_
\/    \___|\__,_|_|\_\|_|   |_|\__|
"""


class PeakFitUI:
    """Centralized UI manager for consistent terminal output."""

    # ==================== BRANDING ====================

    @staticmethod
    def show_banner(verbose: bool = False) -> None:
        """Show PeakFit banner based on verbosity level.

        Args:
            verbose: If True, show full banner with logo
        """
        if not verbose:
            return

        logo_text = Text(LOGO_ASCII, style="bold cyan")
        description_text = Text(
            "Modern NMR Peak Fitting for Pseudo-3D Spectra\n"
            f"{REPO_URL}\n\n",
            style="dim",
        )
        version_text = Text("Version: ", style="dim")
        version_number_text = Text(f"{VERSION}", style="bold green")
        all_text = Text.assemble(logo_text, description_text, version_text, version_number_text)
        panel = Panel.fit(all_text, border_style="cyan", title=f"{LOGO_EMOJI} PeakFit")
        console.print(panel)

    @staticmethod
    def show_version() -> None:
        """Show version information (for --version flag)."""
        console.print(f"\n{LOGO_EMOJI} [header]PeakFit[/header] [dim]v{VERSION}[/dim]")
        console.print(f"[dim]{REPO_URL}[/dim]\n")

    # ==================== HEADERS ====================

    @staticmethod
    def show_header(text: str) -> None:
        """Display a standard header.

        Args:
            text: Header text to display
        """
        console.print(f"\n[header]{text}[/header]")

    @staticmethod
    def show_subheader(text: str) -> None:
        """Display a standard subheader.

        Args:
            text: Subheader text to display
        """
        console.print(f"[subheader]{text}[/subheader]")

    # ==================== STATUS MESSAGES ====================

    @staticmethod
    def success(message: str, indent: int = 0) -> None:
        """Display a success message.

        Args:
            message: Success message to display
            indent: Indentation level (spaces = indent * 2)
        """
        spaces = "  " * indent
        console.print(f"{spaces}[success]✓[/success] {message}")

    @staticmethod
    def warning(message: str, indent: int = 0) -> None:
        """Display a warning message.

        Args:
            message: Warning message to display
            indent: Indentation level (spaces = indent * 2)
        """
        spaces = "  " * indent
        console.print(f"{spaces}[warning]⚠[/warning]  {message}")

    @staticmethod
    def error(message: str, indent: int = 0) -> None:
        """Display an error message.

        Args:
            message: Error message to display
            indent: Indentation level (spaces = indent * 2)
        """
        spaces = "  " * indent
        console.print(f"{spaces}[error]✗[/error] {message}")

    @staticmethod
    def info(message: str, indent: int = 0) -> None:
        """Display an info message.

        Args:
            message: Info message to display
            indent: Indentation level (spaces = indent * 2)
        """
        spaces = "  " * indent
        console.print(f"{spaces}[info]ℹ[/info]  {message}")

    @staticmethod
    def action(message: str) -> None:
        """Display an action/process message with visual separation.

        Use this for ongoing operations like 'Fitting peaks...', 'Loading data...'

        Args:
            message: Action message to display
        """
        console.print(f"\n[bold yellow]—[/bold yellow] {message}")

    @staticmethod
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

    @staticmethod
    def spacer() -> None:
        """Print an empty line for visual spacing."""
        console.print()

    @staticmethod
    def separator(char: str = "─", width: int = 60, style: str = "dim") -> None:
        """Print a visual separator line.

        Args:
            char: Character to use for separator
            width: Width of separator
            style: Rich style to apply
        """
        console.print(f"[{style}]{char * width}[/{style}]")

    # ==================== PROGRESS INDICATORS ====================

    @staticmethod
    def create_progress() -> Progress:
        """Create a standard progress bar with consistent styling.

        Returns:
            Configured Progress instance
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="cyan", finished_style="green"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        )

    # ==================== TABLES ====================

    @staticmethod
    def create_table(title: str | None = None, show_header: bool = True) -> Table:
        """Create a standard table with consistent styling.

        Args:
            title: Optional table title
            show_header: Whether to show table header

        Returns:
            Configured Table instance
        """
        from rich import box

        return Table(
            title=title,
            title_style="header" if title else None,
            box=box.ROUNDED,
            show_header=show_header,
            header_style="bold cyan",
            border_style="dim",
        )

    @staticmethod
    def print_summary(items: dict[str, Any], title: str = "Summary") -> None:
        """Print a standard two-column summary table.

        Args:
            items: Dictionary of key-value pairs to display
            title: Table title
        """
        table = PeakFitUI.create_table(title, show_header=False)
        table.add_column("Item", style="metric")
        table.add_column("Value", style="value")

        for key, value in items.items():
            table.add_row(key, str(value))

        console.print(table)

    # ==================== PANELS ====================

    @staticmethod
    def create_panel(
        content: str,
        title: str | None = None,
        style: str = "info",
    ) -> Panel:
        """Create a standard panel with consistent styling.

        Args:
            content: Panel content
            title: Optional panel title
            style: Border style (info, success, warning, error)

        Returns:
            Configured Panel instance
        """
        from rich import box

        return Panel(
            content,
            title=title,
            title_align="left",
            border_style=style,
            box=box.ROUNDED,
        )

    @staticmethod
    def print_panel(
        content: str,
        title: str | None = None,
        style: str = "info",
    ) -> None:
        """Print a standard panel.

        Args:
            content: Panel content
            title: Optional panel title
            style: Border style (info, success, warning, error)
        """
        panel = PeakFitUI.create_panel(content, title, style)
        console.print(panel)

    # ==================== ERROR HANDLING ====================

    @staticmethod
    def show_error_with_details(
        context: str,
        error: Exception,
        suggestion: str | None = None,
    ) -> None:
        """Display an error with details in a panel.

        Args:
            context: Context of where the error occurred
            error: The exception that was raised
            suggestion: Optional suggestion for fixing the error
        """
        PeakFitUI.error(f"{context} failed")

        # Show error details in panel
        error_panel = PeakFitUI.create_panel(
            f"[error]{type(error).__name__}[/error]: {str(error)}",
            title="Error Details",
            style="error",
        )
        console.print(error_panel)

        # Show suggestion if available
        if suggestion:
            PeakFitUI.info(f"Suggestion: {suggestion}")

        # Link to docs
        console.print(f"\n[dim]See documentation: {REPO_URL}/docs[/dim]")

    @staticmethod
    def show_file_not_found(
        filepath: Path,
        similar_files: list[Path] | None = None,
    ) -> None:
        """Show file not found error with suggestions.

        Args:
            filepath: Path that was not found
            similar_files: Optional list of similar files to suggest
        """
        PeakFitUI.error(f"File not found: [path]{filepath}[/path]")

        if similar_files:
            PeakFitUI.info("Did you mean one of these?")
            for file in similar_files[:5]:
                console.print(f"  • [path]{file}[/path]")

        # Show files in current directory
        parent = filepath.parent if filepath.parent.exists() else Path(".")
        if parent.is_dir():
            pattern = f"*{filepath.suffix}" if filepath.suffix else "*"
            matching_files = list(parent.glob(pattern))
            if matching_files and not similar_files:
                console.print(f"\n[dim]Available {pattern} files in {parent}:[/dim]")
                for file in matching_files[:10]:
                    console.print(f"  • [green]{file.name}[/]")
                if len(matching_files) > 10:
                    console.print(f"  [dim]... and {len(matching_files) - 10} more[/]")

    # ==================== NEXT STEPS ====================

    @staticmethod
    def print_next_steps(steps: list[str]) -> None:
        """Print suggested next steps for the user.

        Args:
            steps: List of suggested commands or actions
        """
        console.print("\n[bold cyan]📋 Next steps:[/]")
        for i, step in enumerate(steps, 1):
            console.print(f"  {i}. {step}")
        console.print()

    # ==================== VALIDATION DISPLAY ====================

    @staticmethod
    def print_validation_table(
        checks: dict[str, tuple[bool, str]],
        title: str = "Input Validation",
    ) -> None:
        """Print a validation results table.

        Args:
            checks: Dictionary mapping check name to (passed, message) tuple
            title: Table title
        """
        table = PeakFitUI.create_table(title)
        table.add_column("Check", style="metric")
        table.add_column("Status", style="value", justify="center")

        for check_name, (passed, message) in checks.items():
            if passed:
                status = f"[success]✓[/success] {message}"
            else:
                status = f"[warning]⚠[/warning] {message}"
            table.add_row(check_name, status)

        console.print(table)

    # ==================== PERFORMANCE SUMMARY ====================

    @staticmethod
    def print_performance_summary(
        total_time: float,
        n_items: int,
        item_name: str = "items",
        successful: int | None = None,
    ) -> None:
        """Print a performance summary table.

        Args:
            total_time: Total elapsed time in seconds
            n_items: Number of items processed
            item_name: Name of items being processed
            successful: Optional number of successful items
        """
        from datetime import timedelta

        table = PeakFitUI.create_table("⏱️  Performance Summary", show_header=False)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="green")

        table.add_row(f"Total {item_name}", str(n_items))
        if successful is not None:
            success_rate = (successful / n_items * 100) if n_items > 0 else 0
            table.add_row("Successful", f"{successful} ({success_rate:.1f}%)")
            if successful < n_items:
                failed = n_items - successful
                table.add_row("Failed", f"[red]{failed}[/]")

        # Format time nicely
        td = timedelta(seconds=total_time)
        if total_time < 60:
            time_str = f"{total_time:.2f}s"
        elif total_time < 3600:
            time_str = f"{td.seconds // 60}m {td.seconds % 60}s"
        else:
            time_str = str(td)

        table.add_row("Total time", time_str)

        if n_items > 0:
            avg_time = total_time / n_items
            if avg_time < 1:
                avg_str = f"{avg_time * 1000:.0f}ms"
            else:
                avg_str = f"{avg_time:.3f}s"
            table.add_row(f"Average per {item_name.rstrip('s')}", avg_str)

        console.print()
        console.print(table)
        console.print()

    # ==================== SPECIALIZED DISPLAYS ====================

    @staticmethod
    def print_cluster_info(cluster_index: int, total_clusters: int, peak_names: list[str]) -> None:
        """Display information about a cluster being processed.

        Args:
            cluster_index: Current cluster number (1-based)
            total_clusters: Total number of clusters
            peak_names: List of peak names in this cluster
        """
        peaks_str = ", ".join(peak_names)
        console.print(
            f"\n[bold cyan]Cluster {cluster_index}/{total_clusters}[/] │ Peaks: [green]{peaks_str}[/]"
        )

    @staticmethod
    def print_peaks_panel(peaks: list[Any]) -> None:
        """Display list of peaks in a panel.

        Args:
            peaks: List of Peak objects with .name attribute
        """
        peak_list = ", ".join(peak.name for peak in peaks)
        panel = Panel.fit(peak_list, title="Peaks", style="green")
        console.print(panel)

    @staticmethod
    def print_data_summary(
        spectrum_shape: tuple,
        n_planes: int,
        n_peaks: int,
        n_clusters: int,
        noise_level: float,
        contour_level: float,
    ) -> None:
        """Print a formatted summary of loaded data.

        Args:
            spectrum_shape: Shape of spectrum data
            n_planes: Number of planes (z-values)
            n_peaks: Number of peaks
            n_clusters: Number of clusters
            noise_level: Estimated noise level
            contour_level: Contour level for clustering
        """
        table = PeakFitUI.create_table("Data Summary", show_header=False)
        table.add_column("Property", style="cyan", width=20)
        table.add_column("Value", style="green")

        table.add_row("Spectrum shape", str(spectrum_shape))
        table.add_row("Number of planes", str(n_planes))
        table.add_row("Number of peaks", str(n_peaks))
        table.add_row("Number of clusters", str(n_clusters))
        table.add_row("Noise level", f"{noise_level:.4f}")
        table.add_row("Contour level", f"{contour_level:.4f}")

        console.print()
        console.print(table)
        console.print()

    @staticmethod
    def print_fit_summary(
        total_clusters: int,
        total_peaks: int,
        total_time: float,
        success_count: int,
    ) -> None:
        """Print a summary of fitting results in a panel.

        Args:
            total_clusters: Total number of clusters fitted
            total_peaks: Total number of peaks
            total_time: Total time elapsed
            success_count: Number of successful fits
        """
        console.print()
        panel_content = Text()
        panel_content.append("Fitting Complete\n\n", style="bold green")
        panel_content.append(f"  Clusters fitted:  {total_clusters}\n")
        panel_content.append(f"  Total peaks:      {total_peaks}\n")
        panel_content.append(f"  Successful:       {success_count}/{total_clusters}\n")
        panel_content.append(f"  Total time:       {total_time:.2f}s\n")

        if total_clusters > 0:
            avg_time = total_time / total_clusters
            panel_content.append(f"  Avg per cluster:  {avg_time:.3f}s")

        console.print(Panel(panel_content, border_style="green"))

    @staticmethod
    def print_optimization_settings(ftol: float, xtol: float, max_nfev: int) -> None:
        """Print optimization settings in dim style.

        Args:
            ftol: Function tolerance
            xtol: Parameter tolerance
            max_nfev: Maximum function evaluations
        """
        console.print(f"[dim]Optimization: ftol={ftol:.0e}, xtol={xtol:.0e}, max_nfev={max_nfev}[/]")

    @staticmethod
    def print_file_item(filepath: Path, indent: int = 2) -> None:
        """Print a file path as a bullet item.

        Args:
            filepath: Path to display
            indent: Indentation level
        """
        spaces = "  " * indent
        console.print(f"{spaces}[cyan]‣[/cyan] [path]{filepath}[/path]")

    # ==================== HTML EXPORT ====================

    @staticmethod
    def export_html(filepath: Path) -> None:
        """Export console output to an HTML file.

        Args:
            filepath: Path to save HTML file
        """
        filepath.write_text(console.export_html())
