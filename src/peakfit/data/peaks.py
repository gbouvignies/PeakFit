"""Peak representation and peak list I/O."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from peakfit.core.fitting import Parameters
from peakfit.lineshapes import SHAPES, Shape
from peakfit.data.spectrum import Spectra
from peakfit.typing import FittingOptions, FloatArray, IntArray


@dataclass
class Peak:
    """Represents a single NMR peak with its shapes in each dimension.

    Attributes:
        name: Peak identifier
        positions: Peak positions in ppm for each dimension
        shapes: List of Shape objects for each dimension
        positions_start: Initial positions (set automatically)
    """

    name: str
    positions: FloatArray
    shapes: list[Shape]
    positions_start: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived fields after dataclass construction."""
        self.positions_start = self.positions.copy()

    def set_cluster_id(self, cluster_id: int) -> None:
        """Set cluster ID for all shapes.

        Args:
            cluster_id: Cluster identifier
        """
        for shape in self.shapes:
            shape.cluster_id = cluster_id

    def create_params(self) -> Parameters:
        """Create parameters for all shapes in this peak.

        Returns:
            Parameters object with all shape parameters
        """
        params = Parameters()
        for shape in self.shapes:
            params.update(shape.create_params())
        return params

    def fix_params(self, params: Parameters) -> None:
        """Fix all parameters for this peak.

        Args:
            params: Parameters to modify
        """
        for shape in self.shapes:
            shape.fix_params(params)

    def release_params(self, params: Parameters) -> None:
        """Release (unfreeze) all parameters for this peak.

        Args:
            params: Parameters to modify
        """
        for shape in self.shapes:
            shape.release_params(params)

    def evaluate(self, grid: Sequence[IntArray], params: Parameters) -> FloatArray:
        """Evaluate peak shape at grid points.

        The total peak shape is the product of shapes in each dimension.

        Args:
            grid: List of point arrays for each dimension
            params: Current parameter values

        Returns:
            Evaluated peak shape (product over dimensions)
        """
        evaluations = [
            shape.evaluate(pts, params) for pts, shape in zip(grid, self.shapes, strict=False)
        ]
        return np.prod(evaluations, axis=0)

    def print(self, params: Parameters) -> str:
        """Format peak parameters as string.

        Args:
            params: Current parameter values

        Returns:
            Formatted string with peak information
        """
        result = f"# Name: {self.name}\n"
        result += "\n".join(shape.print(params) for shape in self.shapes)
        return result

    @property
    def positions_i(self) -> IntArray:
        """Peak positions as integer points."""
        return np.array([shape.center_i for shape in self.shapes], dtype=np.int_)

    @property
    def positions_hz(self) -> FloatArray:
        """Peak positions in Hz."""
        return np.array(
            [shape.spec_params.pts2hz(shape.center_i) for shape in self.shapes],
            dtype=np.float64,
        )

    def update_positions(self, params: Parameters) -> None:
        """Update peak positions from fitted parameters.

        Args:
            params: Fitted parameter values
        """
        self.positions = np.array([params[f"{shape.prefix}0"].value for shape in self.shapes])
        for shape, position in zip(self.shapes, self.positions, strict=False):
            shape.center = position


def create_peak(
    name: str,
    positions: Sequence[float],
    shape_names: list[str],
    spectra: Spectra,
    args: FittingOptions,
) -> Peak:
    """Create a Peak object from positions and shape names.

    Args:
        name: Peak identifier
        positions: Peak positions in ppm for each dimension
        shape_names: Name of lineshape to use for each dimension
        spectra: Spectra object with spectral parameters
        args: Command-line arguments

    Returns:
        Peak object ready for fitting
    """
    shapes = [
        SHAPES[shape_name](name, center, spectra, dim, args)
        for dim, (center, shape_name) in enumerate(
            zip(positions, shape_names, strict=False), start=1
        )
    ]
    return Peak(name, np.array(positions), shapes)


def create_params(peaks: list[Peak], *, fixed: bool = False) -> Parameters:
    """Create combined parameters for all peaks.

    Args:
        peaks: List of peaks to create parameters for
        fixed: If True, fix position parameters (don't vary during fitting)

    Returns:
        Parameters object with all peak parameters
    """
    params = Parameters()
    for peak in peaks:
        params.update(peak.create_params())

    if fixed:
        for name in params:
            if name.endswith("0"):
                params[name].vary = False

    return params


Reader = Callable[[Path, Spectra, list[str], FittingOptions], list[Peak]]

READERS: dict[str, Reader] = {}

NUM_ITEMS = 4


def register_reader(file_types: str | Iterable[str]) -> Callable[[Reader], Reader]:
    """Decorator to register a reader function for specific file types."""
    if isinstance(file_types, str):
        file_types = [file_types]

    def decorator(fn: Reader) -> Reader:
        for ft in file_types:
            READERS[ft] = fn
        return fn

    return decorator


def _create_peak_list(
    peaks: pd.DataFrame, spectra: Spectra, shape_names: list[str], args_cli: FittingOptions
) -> list[Peak]:
    """Create a list of Peak objects from a DataFrame."""
    return [
        create_peak(name, positions, shape_names, spectra, args_cli)
        for name, *positions in peaks.itertuples(index=False, name=None)
    ]


@register_reader("list")
def read_sparky_list(
    path: Path, spectra: Spectra, shape_names: list[str], args_cli: FittingOptions
) -> list[Peak]:
    """Read a Sparky list file and return a list of peaks."""
    with path.open() as f:
        text = "\n".join(line for line in f if "Ass" not in line)
    ndim = spectra.data.ndim
    names_axis = sorted([f"{name}0_ppm" for name in "xyza"[: ndim - 1]], reverse=True)
    names_col = ["name", *names_axis]
    peaks = pd.read_table(
        StringIO(text),
        sep=r"\s+",
        comment="#",
        header=None,
        encoding="utf-8",
        names=names_col,
        usecols=range(ndim),
    )
    return _create_peak_list(peaks, spectra, shape_names, args_cli)


@np.vectorize
def _make_names(f1name: str | float, f2name: str | float, peak_id: int) -> str:
    """Create a peak name from the indirect and direct dimension names."""
    if not (isinstance(f1name, str) and isinstance(f2name, str)):
        return str(peak_id)
    items1, items2 = f1name.split("."), f2name.split(".")
    if len(items1) != NUM_ITEMS or len(items2) != NUM_ITEMS:
        return str(peak_id)
    if items1[1] == items2[1] and items1[2] == items2[2]:
        items2[1], items2[2] = "", ""
    return f"{items1[2]}{items1[1]}{items1[3]}-{items2[2]}{items2[1]}{items2[3]}"


def _read_ccpn_list(
    path: Path,
    spectra: Spectra,
    read_func: Callable[[Path], pd.DataFrame],
    shape_names: list[str],
    args_cli: FittingOptions,
) -> list[Peak]:
    """Read a generic list file and return a list of peaks."""
    peaks_csv = read_func(path)
    names = _make_names(peaks_csv["Assign F2"], peaks_csv["Assign F1"], peaks_csv["#"])
    peaks = pd.DataFrame(
        {"name": names, "y0_ppm": peaks_csv["Pos F2"], "x0_ppm": peaks_csv["Pos F1"]}
    )
    return _create_peak_list(peaks, spectra, shape_names, args_cli)


@register_reader("csv")
def read_csv_list(
    path: Path, spectra: Spectra, shape_names: list[str], args_cli: FittingOptions
) -> list[Peak]:
    return _read_ccpn_list(path, spectra, pd.read_csv, shape_names, args_cli)


@register_reader("json")
def read_json_list(
    path: Path, spectra: Spectra, shape_names: list[str], args_cli: FittingOptions
) -> list[Peak]:
    return _read_ccpn_list(path, spectra, pd.read_json, shape_names, args_cli)


@register_reader(["xlsx", "xls"])
def read_excel_list(
    path: Path, spectra: Spectra, shape_names: list[str], args_cli: FittingOptions
) -> list[Peak]:
    return _read_ccpn_list(path, spectra, pd.read_excel, shape_names, args_cli)


def read_list(spectra: Spectra, shape_names: list[str], args_cli: FittingOptions) -> list[Peak]:
    """Read a list of peaks from a file based on its extension."""
    path = args_cli.path_list
    extension = path.suffix.lstrip(".")
    reader = READERS.get(extension)
    if reader is None:
        msg = f"No reader registered for extension: {extension}"
        raise ValueError(msg)
    return reader(path, spectra, shape_names, args_cli)
