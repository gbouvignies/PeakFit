"""Peak list readers that build Peak domain objects."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from peakfit.engine.domain.peaks import Peak
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.shared.paths import format_path

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from peakfit.engine.domain.config import FitConfig
    from peakfit.engine.domain.spectrum import Spectra

_CCPN_ASSIGNMENT_PARTS = 4


def _get_position_column_names(n_spectral_dims: int) -> list[str]:
    """Generate position column names for N spectral dimensions.

    Uses NMRPipe F1/F2/F3/F4 convention internally but creates column
    names compatible with the existing DataFrame structure.

    Args:
        n_spectral_dims: Number of spectral dimensions (excluding pseudo)

    Returns:
    -------
        List of column names like ['F1_ppm', 'F2_ppm'] for 2D
        Ordered from F1 (first indirect) to Fn (direct)
    """
    return [f"F{i + 1}_ppm" for i in range(n_spectral_dims)]


def _create_peak_list(
    peaks: pd.DataFrame, spectra: Spectra, shape_names: list[str], config: FitConfig
) -> list[Peak]:
    """Create a list of Peak objects from a DataFrame.

    The DataFrame must have a 'name' column followed by position columns.
    Position columns should be ordered from F1 (first indirect) to Fn (direct).
    Peak domain construction validates dimensionality.
    """
    peak_list: list[Peak] = []

    for name, *positions in peaks.itertuples(index=False, name=None):
        pos_values = tuple(float(p) for p in positions)
        pos_array = np.array(pos_values, dtype=np.float64)

        shapes = create_shapes(spectra, config, str(name), pos_values, shape_names)
        peak_list.append(Peak(name=str(name), positions=pos_array, shapes=shapes))

    return peak_list


def _create_peak_list_from_rows(
    rows: Iterable[tuple[str, ...]],
    spectra: Spectra,
    shape_names: list[str],
    config: FitConfig,
) -> list[Peak]:
    """Create peaks from whitespace-split rows (name + positions)."""
    peak_list: list[Peak] = []

    for row in rows:
        name, *positions = row
        pos_values = tuple(float(p) for p in positions)
        pos_array = np.array(pos_values, dtype=np.float64)

        shapes = create_shapes(spectra, config, str(name), pos_values, shape_names)
        peak_list.append(Peak(name=str(name), positions=pos_array, shapes=shapes))

    return peak_list


def read_sparky_list(
    path: Path, spectra: Spectra, shape_names: list[str], config: FitConfig
) -> list[Peak]:
    """Read a Sparky list file and return a list of peaks.

    Sparky format has columns: Assignment, w1, w2, [w3, w4, ...]
    where w1 is the first position column (maps to our F1 dimension).

    Supports 1D through 4D peak lists.
    """
    n_spectral_dims = spectra.n_spectral_dims
    rows: list[tuple[str, ...]] = []

    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "Ass" in line:
                continue

            parts = line.split()
            if len(parts) < n_spectral_dims + 1:
                msg = (
                    f"Invalid Sparky list row in {format_path(path)}: expected at least "
                    f"{n_spectral_dims + 1} columns, got {len(parts)}"
                )
                raise ValueError(msg)

            name = parts[0]
            positions = parts[1 : n_spectral_dims + 1]
            rows.append((name, *positions))

    return _create_peak_list_from_rows(rows, spectra, shape_names, config)


def _make_names(row: pd.Series) -> str:
    """Create a peak name from the indirect and direct dimension names."""
    f1name = str(row.get("Assign F2", ""))
    f2name = str(row.get("Assign F1", ""))
    peak_id = str(row.get("#", ""))

    if not (f1name and f2name):
        return peak_id

    items1, items2 = f1name.split("."), f2name.split(".")
    if len(items1) != _CCPN_ASSIGNMENT_PARTS or len(items2) != _CCPN_ASSIGNMENT_PARTS:
        return peak_id

    if items1[1] == items2[1] and items1[2] == items2[2]:
        items2[1], items2[2] = "", ""
    return f"{items1[2]}{items1[1]}{items1[3]}-{items2[2]}{items2[1]}{items2[3]}"


def _read_ccpn_list(
    path: Path,
    spectra: Spectra,
    read_func: Callable[[Path], pd.DataFrame],
    shape_names: list[str],
    config: FitConfig,
) -> list[Peak]:
    """Read a CCPN-style list file (CSV, JSON, Excel) and return a list of peaks.

    Supports N-dimensional peak lists.
    """
    peaks_csv = read_func(path)

    n_dims = spectra.n_spectral_dims

    data_dict = {}

    # 1. Handle Names
    if "Assign F2" in peaks_csv.columns and "Assign F1" in peaks_csv.columns:
        names = peaks_csv.apply(_make_names, axis=1)
    elif "#" in peaks_csv.columns:
        names = peaks_csv["#"].astype(str)
    elif "name" in peaks_csv.columns:
        names = peaks_csv["name"].astype(str)
    else:
        names = pd.Series([str(i) for i in range(len(peaks_csv))])

    data_dict["name"] = names

    # 2. Handle positions from canonical, Sparky, or CCPN column names.
    found_any = False
    for i in range(n_dims):
        col_std = f"F{i + 1}_ppm"
        col_sparky = f"w{i + 1}"
        col_ccpn = f"Pos F{i + 1}"

        target_key = f"F{i + 1}_ppm"

        if col_std in peaks_csv.columns:
            data_dict[target_key] = peaks_csv[col_std]
            found_any = True
        elif col_sparky in peaks_csv.columns:
            data_dict[target_key] = peaks_csv[col_sparky]
            found_any = True
        elif col_ccpn in peaks_csv.columns:
            data_dict[target_key] = peaks_csv[col_ccpn]
            found_any = True
        else:
            continue

    if not found_any:
        msg = (
            "Could not find position columns in "
            f"{format_path(path)}. Expected 'F1_ppm'/'w1'/'Pos F1', etc."
        )
        raise ValueError(msg)

    peaks = pd.DataFrame(data_dict)

    # Check if we have all necessary columns
    for i in range(n_dims):
        if f"F{i + 1}_ppm" not in peaks.columns:
            msg = f"Missing position column for dimension {i + 1} in {format_path(path)}"
            raise ValueError(msg)

    return _create_peak_list(peaks, spectra, shape_names, config)


def read_csv_list(
    path: Path, spectra: Spectra, shape_names: list[str], config: FitConfig
) -> list[Peak]:
    """Read peaks from a CSV file and return a list of Peak objects."""
    return _read_ccpn_list(path, spectra, pd.read_csv, shape_names, config)


def read_json_list(
    path: Path, spectra: Spectra, shape_names: list[str], config: FitConfig
) -> list[Peak]:
    """Read peaks from a JSON file and return a list of Peak objects."""
    return _read_ccpn_list(path, spectra, pd.read_json, shape_names, config)


def read_excel_list(
    path: Path, spectra: Spectra, shape_names: list[str], config: FitConfig
) -> list[Peak]:
    """Read peaks from an Excel file (xlsx/xls) and return a list of Peak objects."""
    return _read_ccpn_list(path, spectra, pd.read_excel, shape_names, config)


def read_list(
    path: Path, spectra: Spectra, shape_names: list[str], config: FitConfig
) -> list[Peak]:
    """Read a list of peaks from a file based on its extension."""
    extension = path.suffix.lower().lstrip(".")
    if extension == "list":
        return read_sparky_list(path, spectra, shape_names, config)
    if extension == "csv":
        return read_csv_list(path, spectra, shape_names, config)
    if extension == "json":
        return read_json_list(path, spectra, shape_names, config)
    if extension in {"xlsx", "xls"}:
        return read_excel_list(path, spectra, shape_names, config)

    msg = f"Unsupported peak list extension: {extension}"
    raise ValueError(msg)


__all__ = [
    "read_csv_list",
    "read_excel_list",
    "read_json_list",
    "read_list",
    "read_sparky_list",
]
