import argparse
import pathlib
import re
import sys
from collections.abc import Callable

from matplotlib.backends.backend_pdf import PdfPages

from peakfit.messages import (
    print_all_files_missing_error,
    print_check_file_patterns_help,
    print_filename,
    print_files_not_found_warning,
    print_missing_file,
    print_no_files_specified,
    print_no_valid_files_error,
    print_plotting,
    print_processing_files_count,
    print_reading_files,
)


def expand_file_patterns(file_patterns: list[str]) -> list[pathlib.Path]:
    """Expand glob patterns into actual file paths.

    This function helps handle shell differences by manually expanding
    glob patterns when they might not be expanded by the shell.
    """
    expanded_files = []

    for pattern in file_patterns:
        if not pattern or pattern.strip() == "":
            continue  # Skip empty patterns

        # Try to expand the pattern using pathlib
        pattern_path = pathlib.Path(pattern)

        # Check if it's a direct file path first
        if pattern_path.exists():
            expanded_files.append(pattern_path)
        elif "*" in pattern or "?" in pattern:
            # Try to expand as a glob pattern
            # For patterns like "Fits/*.out", we need to handle the parent directory
            parent = (
                pattern_path.parent
                if pattern_path.parent != pathlib.Path()
                else pathlib.Path.cwd()
            )
            try:
                matches = list(parent.glob(pattern_path.name))
                if not matches:
                    # Try from current directory for relative patterns
                    matches = list(pathlib.Path.cwd().glob(pattern))

                if matches:
                    expanded_files.extend(matches)
                else:
                    # Pattern didn't match anything - add as-is for validation
                    expanded_files.append(pattern_path)
            except (OSError, ValueError):
                # Invalid pattern - add as-is for validation
                expanded_files.append(pattern_path)
        else:
            # Not a glob pattern but file doesn't exist - add for validation
            expanded_files.append(pattern_path)

    return expanded_files


def validate_and_filter_files(files: list[pathlib.Path]) -> list[pathlib.Path]:
    """Validate and filter files, ensuring they exist and are readable.

    Returns a list of valid files and prints warnings for missing files.
    """
    if not files:
        print_no_files_specified()
        sys.exit(1)

    # Filter out empty paths and non-existent files
    valid_files = []
    missing_files = []

    for file_path in files:
        if not file_path or str(file_path).strip() == "":
            continue  # Skip empty strings/paths

        if file_path.exists() and file_path.is_file():
            valid_files.append(file_path)
        else:
            missing_files.append(file_path)

    # Report missing files as warnings
    if missing_files:
        print_files_not_found_warning()
        for missing_file in missing_files:
            print_missing_file(missing_file)

    # Check if we have any valid files
    if not valid_files:
        print_no_valid_files_error()
        if missing_files:
            print_all_files_missing_error()
        print_check_file_patterns_help()
        sys.exit(1)

    print_processing_files_count(len(valid_files))
    return valid_files


def get_sorted_files(files: list[pathlib.Path]) -> list[pathlib.Path]:
    """Sort the list of files based on the numerical values in their names.

    Handles cases where no numeric values are found gracefully.
    """
    # First validate and filter the files
    valid_files = validate_and_filter_files(files)

    def extract_numeric_key(filepath: pathlib.Path) -> tuple[float, str]:
        """Extract numeric value from filepath for sorting."""
        # Remove non-digit characters and try to extract a number
        numeric_str = re.sub(r"\D", "", str(filepath.name))

        # If no digits found, use the filename stem as fallback
        if not numeric_str:
            return (float("inf"), str(filepath.stem))  # Sort non-numeric files last

        try:
            return (int(numeric_str), str(filepath.stem))
        except ValueError:
            # If conversion fails, use string representation
            return (float("inf"), str(filepath.stem))

    # Sort files, handling both numeric and string keys
    try:
        return sorted(valid_files, key=extract_numeric_key)
    except (TypeError, AttributeError):
        # Fallback to simple alphabetical sorting
        return sorted(valid_files, key=lambda x: str(x.name))


def save_figures(figs: dict, output: str) -> None:
    """Saves all figures into a single PDF file."""
    print_plotting(output)
    with PdfPages(output) as pdf:
        for fig in figs.values():
            pdf.savefig(fig)


def plot_wrapper(plot_func: Callable) -> Callable:
    """Decorator to wrap the plotting function with common preprocessing steps."""

    def wrapper(args: argparse.Namespace) -> None:
        figs = {}
        print_reading_files()

        # Expand file patterns and convert to Path objects
        if isinstance(args.files[0], str):
            # Files are still strings (patterns), expand them
            expanded_files = expand_file_patterns(args.files)
            files_ordered = get_sorted_files(expanded_files)
        else:
            # Files are already Path objects
            files_ordered = get_sorted_files(args.files)

        for a_file in files_ordered:
            print_filename(a_file)
            figs[a_file.name] = plot_func(a_file, args)

        save_figures(figs, args.out)

    return wrapper


def get_base_parser() -> argparse.ArgumentParser:
    """Creates the base argument parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        type=str,
        required=True,
        help="Input files (supports glob patterns)",
    )
    return parser
