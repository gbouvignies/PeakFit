import argparse
import pathlib
import re
from collections.abc import Callable

from matplotlib.backends.backend_pdf import PdfPages

from peakfit.messages import print_filename, print_plotting, print_reading_files


def get_sorted_files(files: list[pathlib.Path]) -> list[pathlib.Path]:
    """Sorts the list of files based on the numerical values in their names."""
    return sorted(files, key=lambda x: int(re.sub(r"\D", "", str(x))))


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
        files_ordered = get_sorted_files(args.files)

        for a_file in files_ordered:
            print_filename(a_file)
            figs[a_file.name] = plot_func(a_file, args)

        save_figures(figs, args.out)

    return wrapper


def get_base_parser() -> argparse.ArgumentParser:
    """Creates the base argument parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f", "--files", nargs="+", type=pathlib.Path, required=True)
    return parser
