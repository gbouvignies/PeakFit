"""The parsing module contains the code for the parsing of command-line arguments."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path


def build_parser() -> ArgumentParser:
    """Parse the command-line arguments."""
    description = "Perform peak integration in pseudo-3D spectra."

    parser = ArgumentParser(description=description)

    parser.add_argument(
        "--spectra",
        "-s",
        dest="path_spectra",
        required=True,
        type=Path,
        nargs="+",
    )
    parser.add_argument("--list", "-l", dest="path_list", required=True, type=Path)
    parser.add_argument(
        "--zvalues",
        "-z",
        dest="path_z_values",
        required=True,
        nargs="+",
    )
    parser.add_argument("--ct", "-t", dest="contour_level", type=float)
    parser.add_argument("--out", "-o", dest="path_output", default="Fits", type=Path)
    parser.add_argument("--noise", "-n", dest="noise", type=float)
    parser.add_argument(
        "--mc",
        nargs=5,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "N"),
    )
    parser.add_argument("--fixed", dest="fixed", action="store_true")
    parser.add_argument("--pvoigt", dest="pvoigt", action="store_true")
    parser.add_argument("--lorentzian", dest="lorentzian", action="store_true")
    parser.add_argument("--gaussian", dest="gaussian", action="store_true")
    parser.add_argument("--exclude", dest="exclude", type=int, nargs="+")

    return parser
