"""The parsing module contains the code for the parsing of command-line arguments."""

from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Arguments:
    """The dataclass for the command-line arguments."""

    path_spectra: Path = field(default_factory=Path)
    path_list: Path = field(default_factory=Path)
    path_z_values: Path | None = None
    contour_level: float | None = None
    noise: float | None = None
    path_output: Path = Path("Fits")
    refine_nb: int = 1
    fixed: bool = False
    pvoigt: bool = False
    lorentzian: bool = False
    gaussian: bool = False
    jx: bool = False
    phx: bool = False
    phy: bool = False
    exclude: list[int] = field(default_factory=list)


def build_parser() -> ArgumentParser:
    """Parse the command-line arguments."""
    description = "Perform peak integration in pseudo-3D spectra."

    parser = ArgumentParser(description=description)

    parser.add_argument("-s", dest="path_spectra", type=Path, required=True)
    parser.add_argument("-l", dest="path_list", type=Path, required=True)
    parser.add_argument("-z", dest="path_z_values", type=Path)
    parser.add_argument("-t", dest="contour_level", type=float)
    parser.add_argument("-n", dest="noise", type=float)
    parser.add_argument("-o", dest="path_output", type=Path, default="Fits")
    parser.add_argument("--refine", "-r", dest="refine_nb", type=int, default=1)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--pvoigt", action="store_true")
    parser.add_argument("--lorentzian", action="store_true")
    parser.add_argument("--gaussian", action="store_true")
    parser.add_argument("--jx", action="store_true")
    parser.add_argument("--phx", action="store_true")
    parser.add_argument("--phy", action="store_true")
    parser.add_argument("--exclude", type=int, nargs="+", default=[])

    return parser


def parse_args() -> Arguments:
    """Parse the command-line arguments."""
    args = Arguments()
    parser = build_parser()
    parser.parse_args(namespace=args)

    return args
