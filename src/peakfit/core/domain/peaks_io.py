"""Peak list readers and adapters around domain models."""

from collections.abc import Callable, Iterable
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from peakfit.core.domain.peaks import Peak, create_params, create_peak
from peakfit.core.domain.spectrum import Spectra
from peakfit.core.shared.typing import FittingOptions

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


def _get_position_column_names(n_spectral_dims: int) -> list[str]:
    """Generate position column names for N spectral dimensions.

    Uses NMRPipe F1/F2/F3/F4 convention internally but creates column
    names compatible with the existing DataFrame structure.

    Args:
        n_spectral_dims: Number of spectral dimensions (excluding pseudo)

    Returns:
        List of column names like ['F1_ppm', 'F2_ppm'] for 2D
        Ordered from F1 (first indirect) to Fn (direct)
    """
    return [f"F{i + 1}_ppm" for i in range(n_spectral_dims)]


def _create_peak_list(
    peaks: pd.DataFrame, spectra: Spectra, shape_names: list[str], args_cli: FittingOptions
) -> list[Peak]:
    """Create a list of Peak objects from a DataFrame.

    The DataFrame must have a 'name' column followed by position columns.
    Position columns should be ordered from F1 (first indirect) to Fn (direct).
    """
    return [
        create_peak(name, positions, shape_names, spectra, args_cli)
        for name, *positions in peaks.itertuples(index=False, name=None)
    ]


@register_reader("list")
def read_sparky_list(
    path: Path, spectra: Spectra, shape_names: list[str], args_cli: FittingOptions
) -> list[Peak]:
    """Read a Sparky list file and return a list of peaks.

    Sparky format has columns: Assignment, w1, w2, [w3, w4, ...]
    where w1 is the first position column (maps to our F1 dimension).

    Supports 1D through 4D peak lists.
    """
    with path.open() as f:
        text = "\n".join(line for line in f if "Ass" not in line)

    n_spectral_dims = spectra.n_spectral_dims
    position_cols = _get_position_column_names(n_spectral_dims)
    names_col = ["name", *position_cols]

    peaks = pd.read_table(
        StringIO(text),
        sep=r"\s+",
        comment="#",
        header=None,
        encoding="utf-8",
        names=names_col,
        usecols=range(n_spectral_dims + 1),  # name + position columns
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


def _detect_position_columns(df: pd.DataFrame) -> list[str]:
    """Detect position columns in a DataFrame.

    Looks for columns matching patterns:
    - 'Pos F1', 'Pos F2', 'Pos F3', 'Pos F4' (CCPN style)
    - 'w1', 'w2', 'w3', 'w4' (Sparky style)
    - 'Position F1', 'Position F2', etc.

    Returns:
        List of column names in order (F1/w1 first, then F2/w2, etc.)
    """
    columns = df.columns.tolist()
    position_cols = []

    # Try CCPN 'Pos Fn' pattern first
    for i in range(1, 5):  # F1 through F4
        col_name = f"Pos F{i}"
        if col_name in columns:
            position_cols.append(col_name)

    if position_cols:
        return position_cols

    # Try Sparky 'wn' pattern
    for i in range(1, 5):
        col_name = f"w{i}"
        if col_name in columns:
            position_cols.append(col_name)

    return position_cols


def _read_ccpn_list(
    path: Path,
    spectra: Spectra,
    read_func: Callable[[Path], pd.DataFrame],
    shape_names: list[str],
    args_cli: FittingOptions,
) -> list[Peak]:
    """Read a CCPN-style list file (CSV, JSON, Excel) and return a list of peaks.

    Supports N-dimensional peak lists by detecting position columns dynamically.
    Looks for 'Pos F1', 'Pos F2', etc. or falls back to first numeric columns.
    """
    peaks_csv = read_func(path)

    # Detect position columns
    position_cols = _detect_position_columns(peaks_csv)

    if not position_cols:
        # Fallback: use columns 'Pos F2', 'Pos F1' if available (legacy 2D format)
        if "Pos F2" in peaks_csv.columns and "Pos F1" in peaks_csv.columns:
            position_cols = ["Pos F1", "Pos F2"]
        else:
            msg = (
                f"Could not detect position columns in {path}. "
                "Expected 'Pos F1', 'Pos F2', ... or 'w1', 'w2', ..."
            )
            raise ValueError(msg)

    # Build peak names from assignment columns if available
    if "Assign F2" in peaks_csv.columns and "Assign F1" in peaks_csv.columns:
        names = _make_names(peaks_csv["Assign F2"], peaks_csv["Assign F1"], peaks_csv["#"])
    elif "#" in peaks_csv.columns:
        names = peaks_csv["#"].astype(str)
    elif "name" in peaks_csv.columns:
        names = peaks_csv["name"].astype(str)
    else:
        names = pd.Series([str(i) for i in range(len(peaks_csv))])

    # Build DataFrame with name and positions in F1, F2, ... order
    data = {"name": names}
    for i, col in enumerate(position_cols):
        data[f"F{i + 1}_ppm"] = peaks_csv[col]

    peaks = pd.DataFrame(data)
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


__all__ = [
    "Peak",
    "create_params",
    "create_peak",
    "read_csv_list",
    "read_excel_list",
    "read_json_list",
    "read_list",
    "read_sparky_list",
]
