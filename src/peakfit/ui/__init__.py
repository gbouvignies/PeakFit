"""UI and terminal output styling for PeakFit.

This package provides a consistent UI layer for terminal output.

Submodules:
- console: Theme and console instance
- logging: File logging utilities
- branding: Banner, version display
- messages: Status messages (success, error, warning, etc.)
- tables: Table display utilities
- panels: Panel display utilities
- progress: Progress bar utilities
- fit_display: Fit-specific display components
"""

from peakfit.ui.branding import show_banner, show_run_info, show_version

# Re-export submodule contents for direct access
from peakfit.ui.console import LOGO_ASCII, PEAKFIT_THEME, REPO_URL, VERSION, console
from peakfit.ui.fit_display import (
    create_cluster_status,
    export_html,
    print_cluster_info,
    print_data_summary,
    print_file_item,
    print_fit_report,
    print_fit_summary,
    print_optimization_settings,
    print_peaks_panel,
)
from peakfit.ui.logging import close_logging, log, log_dict, log_section, setup_logging
from peakfit.ui.messages import (
    action,
    bullet,
    error,
    info,
    print_next_steps,
    separator,
    show_error_with_details,
    show_file_not_found,
    show_footer,
    show_header,
    show_subheader,
    spacer,
    subsection_header,
    success,
    warning,
)
from peakfit.ui.panels import create_panel, print_panel
from peakfit.ui.progress import create_progress
from peakfit.ui.reporter import ConsoleReporter, default_reporter
from peakfit.ui.style import PeakFitUI
from peakfit.ui.tables import (
    create_table,
    print_performance_summary,
    print_summary,
    print_validation_table,
)

__all__ = [
    # Main classes
    "ConsoleReporter",
    "PeakFitUI",
    "console",
    "default_reporter",
    # Console
    "LOGO_ASCII",
    "PEAKFIT_THEME",
    "REPO_URL",
    "VERSION",
    # Logging
    "close_logging",
    "log",
    "log_dict",
    "log_section",
    "setup_logging",
    # Branding
    "show_banner",
    "show_run_info",
    "show_version",
    # Messages
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
    # Tables
    "create_table",
    "print_performance_summary",
    "print_summary",
    "print_validation_table",
    # Panels
    "create_panel",
    "print_panel",
    # Progress
    "create_progress",
    # Fit display
    "create_cluster_status",
    "export_html",
    "print_cluster_info",
    "print_data_summary",
    "print_file_item",
    "print_fit_report",
    "print_fit_summary",
    "print_optimization_settings",
    "print_peaks_panel",
]
