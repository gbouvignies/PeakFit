from collections.abc import Callable, Iterable
from io import StringIO
from pathlib import Path

import pandas as pd

from peakfit.peak import Peak, create_peak
from peakfit.spectra import Spectra

Reader = Callable[[Path, Spectra, list[str]], list[Peak]]

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
    peaks: pd.DataFrame, spectra: Spectra, shape_names: list[str]
) -> list[Peak]:
    """Create a list of Peak objects from a DataFrame."""
    return [
        create_peak(name, positions, shape_names, spectra)
        for name, *positions in peaks.itertuples(index=False, name=None)
    ]


@register_reader("list")
def read_sparky_list(
    path: Path, spectra: Spectra, shape_names: list[str]
) -> list[Peak]:
    """Read a Sparky list file and return a list of peaks."""
    with path.open() as f:
        text = "\n".join(line for line in f if "Ass" not in line)
    peaks = pd.read_table(
        StringIO(text),
        sep=r"\s+",
        comment="#",
        header=None,
        encoding="utf-8",
        names=("name", "y0_ppm", "x0_ppm"),
    )
    return _create_peak_list(peaks, spectra, shape_names)


def _make_names(f1name: str, f2name: str, peak_id: int) -> str:
    """Create a peak name from the indirect and direct dimension names."""
    if not isinstance(f1name, str) and not isinstance(f2name, str):
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
) -> list[Peak]:
    """Read a generic list file and return a list of peaks."""
    peaks_csv = read_func(path)
    names = _make_names(peaks_csv["Assign F2"], peaks_csv["Assign F1"], peaks_csv["#"])
    peaks = pd.DataFrame(
        {"name": names, "y0_ppm": peaks_csv["Pos F2"], "x0_ppm": peaks_csv["Pos F1"]}
    )
    return _create_peak_list(peaks, spectra, shape_names)


@register_reader("csv")
def read_csv_list(path: Path, spectra: Spectra, shape_names: list[str]) -> list[Peak]:
    return _read_ccpn_list(path, spectra, pd.read_csv, shape_names)


@register_reader("json")
def read_json_list(path: Path, spectra: Spectra, shape_names: list[str]) -> list[Peak]:
    return _read_ccpn_list(path, spectra, pd.read_json, shape_names)


@register_reader(["xlsx", "xls"])
def read_excel_list(path: Path, spectra: Spectra, shape_names: list[str]) -> list[Peak]:
    return _read_ccpn_list(path, spectra, pd.read_excel, shape_names)


def read_list(path: Path, spectra: Spectra, shape_names: list[str]) -> list[Peak]:
    """Read a list of peaks from a file based on its extension."""
    extension = path.suffix.lstrip(".")
    reader = READERS.get(extension)
    if reader is None:
        msg = f"No reader registered for extension: {extension}"
        raise ValueError(msg)
    return reader(path, spectra, shape_names)
