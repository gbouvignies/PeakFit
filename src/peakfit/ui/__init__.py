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
- panels: Panel display utilities
- progress: Progress bar utilities
"""

from peakfit.ui.branding import (
    show_command_manifest,
    show_standard_header,
    show_version,
)
from peakfit.ui.console import (
    LOGO_ASCII,
    PEAKFIT_THEME,
    REPO_URL,
    VERSION,
    Verbosity,
    console,
    display_path,
    export_html,
    get_verbosity,
    hr,
    icon,
    set_verbosity,
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
from peakfit.ui.progress import create_mcmc_progress, create_progress
from peakfit.ui.reporter import ConsoleReporter
from peakfit.ui.tables import (
    create_live_metrics_table,
    create_metadata_grid,
    create_table,
    print_summary,
    print_validation_table,
)

__all__ = [
    "LOGO_ASCII",
    "PEAKFIT_THEME",
    "REPO_URL",
    "VERSION",
    "ConsoleReporter",
    "Verbosity",
    "action",
    "bullet",
    "close_logging",
    "console",
    "create_live_metrics_table",
    "create_mcmc_progress",
    "create_metadata_grid",
    "create_panel",
    "create_progress",
    "create_table",
    "display_path",
    "error",
    "export_html",
    "get_verbosity",
    "hr",
    "icon",
    "info",
    "log",
    "log_dict",
    "log_section",
    "print_next_steps",
    "print_panel",
    "print_summary",
    "print_validation_table",
    "separator",
    "set_verbosity",
    "setup_logging",
    "show_command_manifest",
    "show_error_with_details",
    "show_file_not_found",
    "show_footer",
    "show_header",
    "show_standard_header",
    "show_subheader",
    "show_version",
    "spacer",
    "subsection_header",
    "success",
    "warning",
]
