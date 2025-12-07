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
- fit_display: HTML export utility
- cluster_status: Live cluster fitting status display
"""

from peakfit.ui.branding import (
    show_banner,
    show_run_info,
    show_standard_header,
    show_version,
)

# Re-export submodule contents for direct access
from peakfit.ui.cluster_status import (
    ClusterState,
    ClusterStatus,
    LiveClusterDisplay,
)
from peakfit.ui.console import (
    LOGO_ASCII,
    PEAKFIT_THEME,
    REPO_URL,
    VERSION,
    Verbosity,
    console,
    icon,
    hr,
    get_verbosity,
    set_verbosity,
)
from peakfit.ui.fit_display import export_html
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
from peakfit.ui.reporter import ConsoleReporter
from peakfit.ui.tables import (
    create_table,
    print_summary,
    print_validation_table,
)

__all__ = [
    "LOGO_ASCII",
    "PEAKFIT_THEME",
    "REPO_URL",
    "VERSION",
    "ClusterState",
    "ClusterStatus",
    "ConsoleReporter",
    "LiveClusterDisplay",
    "Verbosity",
    "action",
    "bullet",
    "close_logging",
    "console",
    "icon",
    "hr",
    "create_panel",
    "create_progress",
    "create_table",
    "error",
    "export_html",
    "get_verbosity",
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
    "show_banner",
    "show_error_with_details",
    "show_file_not_found",
    "show_footer",
    "show_header",
    "show_run_info",
    "show_standard_header",
    "show_subheader",
    "show_version",
    "spacer",
    "subsection_header",
    "success",
    "warning",
]
