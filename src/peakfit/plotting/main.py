import argparse

from .common import get_base_parser
from .plots.cest import plot_cest
from .plots.cpmg import plot_cpmg
from .plots.intensity import plot_intensities
from .plots.spectra import plot_spectra


def main() -> None:
    """Main entry point for the PeakFit plotting tool."""
    parser = argparse.ArgumentParser(description="PeakFit plotting tool")
    subparsers = parser.add_subparsers(
        dest="type", required=True, help="Type of plot to generate"
    )

    # Intensity plot parser
    intensity_parser = subparsers.add_parser(
        "intensity", parents=[get_base_parser()], help="Generate intensity plot"
    )
    intensity_parser.add_argument(
        "-o",
        "--out",
        type=str,
        help="PDF file to save the plot to",
        default="profiles.pdf",
    )

    # CEST plot parser
    cest_parser = subparsers.add_parser(
        "cest", parents=[get_base_parser()], help="Generate CEST plot"
    )
    cest_parser.add_argument(
        "--ref",
        nargs="+",
        type=int,
        default=[-1],
        help="Reference points for CEST plot",
    )
    cest_parser.add_argument(
        "-o",
        "--out",
        type=str,
        help="PDF file to save the plot to",
        default="profiles.pdf",
    )

    # CPMG plot parser
    cpmg_parser = subparsers.add_parser(
        "cpmg", parents=[get_base_parser()], help="Generate CPMG plot"
    )
    cpmg_parser.add_argument(
        "--time_t2", type=float, required=True, help="T2 time for CPMG plot"
    )
    cpmg_parser.add_argument(
        "-o",
        "--out",
        type=str,
        help="PDF file to save the plot to",
        default="profiles.pdf",
    )

    # Spectra plot parser
    spectra_parser = subparsers.add_parser("spectra", help="Generate spectra plot")
    spectra_parser.add_argument(
        "--exp", dest="data_exp", required=True, help="Path to the first spectrum file"
    )
    spectra_parser.add_argument(
        "--sim", dest="data_sim", required=True, help="Path to the second spectrum file"
    )
    spectra_parser.add_argument(
        "--plist", dest="peak_list", help="Path to the peak list file"
    )

    args = parser.parse_args()

    if args.type == "intensity":
        plot_intensities(args)
    elif args.type == "cest":
        plot_cest(args)
    elif args.type == "cpmg":
        plot_cpmg(args)
    elif args.type == "spectra":
        plot_spectra(args)


if __name__ == "__main__":
    main()
