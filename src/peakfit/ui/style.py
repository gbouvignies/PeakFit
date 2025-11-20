"""
Centralized UI style definitions for consistent terminal output.
All terminal output MUST use these styles for consistency.
"""

import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.logging import RichHandler
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

# Module-level logger (configured by setup_logging)
_logger: logging.Logger | None = None

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

    # ==================== LOGGING SETUP ====================

    @staticmethod
    def setup_logging(
        log_file: Path | None = None,
        verbose: bool = False,
        level: int = logging.INFO,
    ) -> None:
        """Configure logging for PeakFit.

        Args:
            log_file: Path to log file. If None, logging is disabled.
            verbose: If True, show all log messages in console
            level: Logging level (default: INFO)
        """
        global _logger  # noqa: PLW0603 - necessary for module-level logger management

        if log_file is None:
            _logger = None
            return

        # Create log directory if needed
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure logger
        _logger = logging.getLogger("peakfit")
        _logger.setLevel(level)
        _logger.handlers.clear()

        # File handler with structured format
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        _logger.addHandler(file_handler)

        # Console handler (only if verbose)
        if verbose:
            console_handler = RichHandler(
                console=console,
                show_time=False,
                show_path=False,
                markup=True,
            )
            console_handler.setLevel(level)
            _logger.addHandler(console_handler)

        # Log session start
        _logger.info("━" * 60)
        _logger.info(f"PeakFit v{VERSION} - Session Started")
        _logger.info("━" * 60)
        _logger.info(f"Command: {' '.join(sys.argv)}")
        _logger.info(f"Working directory: {Path.cwd()}")
        _logger.info(f"Python: {sys.version.split()[0]} | Platform: {sys.platform}")
        _logger.info("")

    @staticmethod
    def log(message: str, level: str = "info") -> None:
        """Log a message to file (if logging is enabled).

        Args:
            message: Message to log
            level: Log level (info, warning, error, debug)
        """
        if _logger is None:
            return

        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }

        log_level = level_map.get(level.lower(), logging.INFO)
        _logger.log(log_level, message)

    @staticmethod
    def log_section(title: str) -> None:
        """Log a section header.

        Args:
            title: Section title
        """
        if _logger is None:
            return

        _logger.info("")
        _logger.info(f"=== {title.upper()} ===")

    @staticmethod
    def log_dict(data: dict[str, Any], indent: str = "  ") -> None:
        """Log a dictionary as key-value pairs.

        Args:
            data: Dictionary to log
            indent: Indentation string
        """
        if _logger is None:
            return

        for key, value in data.items():
            _logger.info(f"{indent}- {key}: {value}")

    @staticmethod
    def close_logging() -> None:
        """Close logging and finalize log file."""
        if _logger is None:
            return

        _logger.info("")
        _logger.info("━" * 60)
        _logger.info("PeakFit Session Completed Successfully")
        _logger.info("━" * 60)

        # Close all handlers
        for handler in _logger.handlers[:]:
            handler.close()
            _logger.removeHandler(handler)

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
            f"Modern NMR Peak Fitting for Pseudo-3D Spectra\n{REPO_URL}\n\n",
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

    @staticmethod
    def show_run_info(start_time: datetime) -> None:
        """Show run information header with context.

        Args:
            start_time: When the program started
        """
        # Logo and version
        console.print(f"\n{LOGO_EMOJI} [bold cyan]PeakFit[/bold cyan] [dim]v{VERSION}[/dim]")
        console.print("━" * 70 + "\n")

        # Get command line arguments and clean them
        import os

        # Remove absolute path from peakfit executable, keep just 'peakfit'
        if sys.argv and ('peakfit' in sys.argv[0] or sys.argv[0].endswith('.py')):
            clean_argv = ['peakfit', *sys.argv[1:]]
        else:
            clean_argv = sys.argv

        command_args = " ".join(clean_argv)

        # Truncate long commands
        max_cmd_length = 80
        if len(command_args) > max_cmd_length:
            command_display = command_args[:max_cmd_length-3] + "..."
        else:
            command_display = command_args

        # Simplify platform string (remove redundant parts)
        platform_str = platform.platform()
        # "macOS-26.1-arm64-arm-64bit-Mach-O" → "macOS-26.1-arm64"
        # "Linux-4.4.0-x86_64-with-glibc2.39" → "Linux-4.4.0-x86_64"
        platform_parts = platform_str.split('-')
        if len(platform_parts) > 3:
            platform_display = '-'.join(platform_parts[:3])
        else:
            platform_display = platform_str

        # Create run information panel
        info_text = (
            f"[cyan]Started:[/cyan] {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"[cyan]Command:[/cyan] {command_display}\n"
            f"[cyan]Working directory:[/cyan] {Path.cwd()}\n"
            f"[cyan]Python:[/cyan] {sys.version.split()[0]} | "
            f"[cyan]Platform:[/cyan] {platform_display}"
        )

        run_info_panel = Panel(
            info_text,
            title="Run Information",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 2),
            expand=False,  # Don't expand to full terminal width
        )
        console.print(run_info_panel)
        console.print()

        # Log this information (use full original command for log file)
        if _logger:
            original_command = " ".join(sys.argv)
            PeakFitUI.log("=" * 60)
            PeakFitUI.log(f"PeakFit v{VERSION} started")
            PeakFitUI.log("=" * 60)
            PeakFitUI.log(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            PeakFitUI.log(f"Command: {original_command}")
            PeakFitUI.log(f"Working directory: {Path.cwd()}")
            PeakFitUI.log(f"Python: {sys.version.split()[0]}")
            PeakFitUI.log(f"Platform: {platform.platform()}")
            PeakFitUI.log(f"User: {os.getenv('USER', 'unknown')}")
            try:
                import socket
                PeakFitUI.log(f"Hostname: {socket.gethostname()}")
            except (OSError, ImportError):
                # Socket operations may fail in restricted environments
                pass
            PeakFitUI.log("=" * 60)

    # ==================== HEADERS ====================

    @staticmethod
    def show_header(text: str, log: bool = True) -> None:
        """Display a prominent section header with consistent spacing.

        Spacing: Calling code must add ONE blank line before. ZERO blank lines after.

        Args:
            text: Header text to display
            log: Whether to log this header to file
        """
        # Calling code is responsible for adding blank line before
        console.print("[bold cyan]" + "━" * 60 + "[/bold cyan]")
        console.print(f"[bold cyan]  {text}[/bold cyan]")
        console.print("[bold cyan]" + "━" * 60 + "[/bold cyan]")
        # NO blank line after - content starts immediately
        if log:
            PeakFitUI.log_section(text)

    @staticmethod
    def show_subheader(text: str) -> None:
        """Display a standard subheader.

        Args:
            text: Subheader text to display
        """
        console.print(f"\n[bold white]{text}[/bold white]")
        console.print("[dim]" + "─" * 40 + "[/dim]")

    @staticmethod
    def subsection_header(title: str) -> None:
        """Print subsection header with correct spacing.

        Spacing: ONE blank line before, ONE blank line after.

        Args:
            title: Subsection title to display
        """
        console.print()  # ONE blank line before
        console.print(f"[bold]{title}[/bold]")
        console.print()  # ONE blank line after

    # ==================== STATUS MESSAGES ====================

    @staticmethod
    def success(message: str, indent: int = 0, log: bool = True) -> None:
        """Display a success message.

        Args:
            message: Success message to display
            indent: Indentation level (spaces = indent * 2)
            log: Whether to log this message to file
        """
        spaces = "  " * indent
        console.print(f"{spaces}[success]✓[/success] {message}")
        if log:
            PeakFitUI.log(f"{message}")

    @staticmethod
    def warning(message: str, indent: int = 0, log: bool = True) -> None:
        """Display a warning message.

        Args:
            message: Warning message to display
            indent: Indentation level (spaces = indent * 2)
            log: Whether to log this message to file
        """
        spaces = "  " * indent
        console.print(f"{spaces}[warning]⚠[/warning]  {message}")
        if log:
            PeakFitUI.log(f"{message}", level="warning")

    @staticmethod
    def error(message: str, indent: int = 0, log: bool = True) -> None:
        """Display an error message.

        Args:
            message: Error message to display
            indent: Indentation level (spaces = indent * 2)
            log: Whether to log this message to file
        """
        spaces = "  " * indent
        console.print(f"{spaces}[error]✗[/error] {message}")
        if log:
            PeakFitUI.log(f"{message}", level="error")

    @staticmethod
    def info(message: str, indent: int = 0, log: bool = True) -> None:
        """Display an info message.

        Args:
            message: Info message to display
            indent: Indentation level (spaces = indent * 2)
            log: Whether to log this message to file
        """
        spaces = "  " * indent
        console.print(f"{spaces}[dim]▸[/dim] {message}")
        if log:
            PeakFitUI.log(f"{message}")

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

    @staticmethod
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
            f"[green]✓[/green] [dim]Completed:[/dim] {end_time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"[dim]Total runtime:[/dim] [cyan]{runtime_str}[/cyan]"
        )

        # Log completion
        if _logger:
            PeakFitUI.log("=" * 60)
            PeakFitUI.log(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            PeakFitUI.log(f"Total runtime: {runtime_str}")
            PeakFitUI.log("=" * 60)

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
            f"[error]{type(error).__name__}[/error]: {error!s}",
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
    def create_cluster_status(
        cluster_index: int,
        total_clusters: int,
        peak_names: list[str],
        status: str = "fitting",
        result: Any = None,
    ) -> Panel:
        """Create a renderable cluster status panel for live display.

        Args:
            cluster_index: Current cluster number (1-based)
            total_clusters: Total number of clusters
            peak_names: List of peak names in this cluster
            status: Status message ("fitting", "optimizing", "done")
            result: Optional optimization result to display

        Returns:
            Panel object that can be rendered
        """
        from rich.text import Text

        peaks_str = ", ".join(peak_names)

        # Build content
        content = Text()
        content.append(f"Cluster {cluster_index}/{total_clusters}\n", style="bold cyan")
        content.append("Peaks: ", style="dim")
        content.append(f"{peaks_str}\n\n", style="green")

        if status == "fitting":
            content.append("Status: ", style="dim")
            content.append("● Fitting...", style="bold yellow")
        elif status == "optimizing":
            content.append("Status: ", style="dim")
            content.append("● Optimizing...", style="bold yellow")
        elif status == "done" and result:
            if hasattr(result, "success") and result.success:
                content.append("Status: ", style="dim")
                content.append("✓ Complete", style="bold green")
            else:
                content.append("Status: ", style="dim")
                content.append("⚠ Complete (with issues)", style="bold yellow")

            # Add key statistics
            if hasattr(result, "nfev"):
                content.append(f"\nEvaluations: {result.nfev}", style="dim")
            if hasattr(result, "cost"):
                content.append(f" │ Cost: {result.cost:.2e}", style="dim")

        return Panel(
            content,
            border_style="cyan" if status != "done" else "green",
            padding=(0, 2),
        )

    @staticmethod
    def print_cluster_info(cluster_index: int, total_clusters: int, peak_names: list[str]) -> None:
        """Display information about a cluster being processed.

        Args:
            cluster_index: Current cluster number (1-based)
            total_clusters: Total number of clusters
            peak_names: List of peak names in this cluster
        """
        # Add visual separation before cluster
        console.print()
        console.print("[dim]" + "─" * 60 + "[/dim]")

        peaks_str = ", ".join(peak_names)
        console.print(
            f"[bold cyan]Cluster {cluster_index}/{total_clusters}[/] │ Peaks: [green]{peaks_str}[/]"
        )

    @staticmethod
    def print_fit_report(result: Any) -> None:
        """Print fitting results in a styled panel.

        Args:
            result: scipy.optimize.OptimizeResult or similar object
        """
        # Create a table for fit statistics
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Property", style="cyan", width=18)
        table.add_column("Value", style="green")

        if hasattr(result, "success"):
            status_style = "green" if result.success else "red"
            status_text = "✓ Success" if result.success else "✗ Failed"
            table.add_row("Status", f"[{status_style}]{status_text}[/]")

        if hasattr(result, "message"):
            table.add_row("Message", str(result.message))

        if hasattr(result, "nfev"):
            table.add_row("Function evals", str(result.nfev))

        if hasattr(result, "njev") and result.njev:
            table.add_row("Jacobian evals", str(result.njev))

        if hasattr(result, "cost"):
            table.add_row("Final cost", f"{result.cost:.6e}")

        if hasattr(result, "optimality"):
            table.add_row("Optimality", f"{result.optimality:.6e}")

        if hasattr(result, "x"):
            table.add_row("Parameters", str(len(result.x)))

        # Display in a panel
        border_style = "green" if (hasattr(result, "success") and result.success) else "yellow"
        panel = Panel(
            table,
            title="[bold]Fit Statistics[/bold]",
            border_style=border_style,
            padding=(0, 1),
        )
        console.print(panel)

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
        noise_source: str,
        contour_level: float,
    ) -> None:
        """Print a formatted summary of loaded data.

        Args:
            spectrum_shape: Shape of spectrum data
            n_planes: Number of planes (z-values)
            n_peaks: Number of peaks
            n_clusters: Number of clusters
            noise_level: Noise level value
            noise_source: Source of noise level ('user-provided' or 'estimated')
            contour_level: Contour level for clustering
        """
        table = PeakFitUI.create_table("Data Summary")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Spectrum shape", str(spectrum_shape))
        table.add_row("Number of planes", str(n_planes))
        table.add_row("Number of peaks", str(n_peaks))
        table.add_row("Number of clusters", str(n_clusters))
        table.add_row("Noise level", f"{noise_level:.2f} ({noise_source})")
        table.add_row("Contour level", f"{contour_level:.2f}")

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
        console.print(
            f"[dim]Optimization: ftol={ftol:.0e}, xtol={xtol:.0e}, max_nfev={max_nfev}[/]"
        )

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
