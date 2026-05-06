from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd

    from peakfit.plot.qt_core import QWidget
    from peakfit.shared.typing import FloatArray


class SpectraViewerBackend(Protocol):
    """Structural interface for spectra viewer backends.

    This is intentionally a Protocol (not an ABC) to avoid metaclass conflicts
    with Qt widgets (e.g., QWidget subclasses).
    """

    def get_widget(self) -> QWidget:
        """Return the Qt widget for embedding in the UI."""
        ...

    def plot(
        self,
        data1: FloatArray,
        data2: FloatArray | None,
        data_diff: FloatArray | None,
        plist: pd.DataFrame | None,
        show_spectra: dict[str, bool],
        contour_level: float,
        noise_level: float,
        current_plane: int,
        xlim: list[float],
        ylim: list[float],
        *,
        reset_view: bool = False,
    ) -> None:
        """Update the plot with new data or settings."""
        ...

    def set_plane(self, plane: int) -> None:
        """Update the current plane (e.g., pseudo-3D Z index)."""
        ...

    def clear(self) -> None:
        """Clear the plot."""
        ...
