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

Note: The PeakFitUI class has been deprecated. Use the direct function
imports instead (success, error, info, warning, etc.).
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
from peakfit.ui.tables import (
    create_table,
    print_performance_summary,
    print_summary,
    print_validation_table,
)

__all__ = [
    "LOGO_ASCII",
    "PEAKFIT_THEME",
    "REPO_URL",
    "VERSION",
    "ConsoleReporter",
    "action",
    "bullet",
    "close_logging",
    "console",
    "create_cluster_status",
    "create_panel",
    "create_progress",
    "create_table",
    "default_reporter",
    "error",
    "export_html",
    "info",
    "log",
    "log_dict",
    "log_section",
    "print_cluster_info",
    "print_data_summary",
    "print_file_item",
    "print_fit_report",
    "print_fit_summary",
    "print_next_steps",
    "print_optimization_settings",
    "print_panel",
    "print_peaks_panel",
    "print_performance_summary",
    "print_summary",
    "print_validation_table",
    "separator",
    "setup_logging",
    "show_banner",
    "show_error_with_details",
    "show_file_not_found",
    "show_footer",
    "show_header",
    "show_run_info",
    "show_subheader",
    "show_version",
    "spacer",
    "subsection_header",
    "success",
    "warning",
]
