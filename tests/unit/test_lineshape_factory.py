"""Tests for the lineshape factory utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from peakfit.core.domain.spectrum import Spectra, SpectralParameters
from peakfit.core.lineshapes.factory import LineshapeFactory
from peakfit.core.lineshapes.models import SP1, Gaussian, Lorentzian, NoApod


@dataclass
class _Options:
    jx: bool = False
    phx: bool = False
    phy: bool = False
    noise: float | None = None
    pvoigt: bool = False
    lorentzian: bool = False
    gaussian: bool = False
    path_list: Path = Path()


class _SpectraStub(Spectra):
    """Lightweight spectra stub for factory tests."""

    def __init__(self, params: list[SpectralParameters]) -> None:
        self.dic = {}
        self.data = np.zeros((1, 8, 8), dtype=np.float32)
        self.z_values = np.arange(self.data.shape[0])
        self.pseudo_dim_added = False
        self._params = params

    @property
    def params(self) -> list[SpectralParameters]:  # type: ignore[override]
        return self._params


def _make_options() -> _Options:
    return _Options()


def _make_params(
    *,
    size: int = 8,
    sw: float = 2000.0,
    obs: float = 500.0,
    car: float = 0.0,
    aq_time: float = 0.1,
    apocode: float = 0.0,
    apodq1: float = 0.0,
    apodq2: float = 0.0,
    apodq3: float = 0.0,
    p180: bool = False,
    direct: bool = True,
    ft: bool = True,
) -> SpectralParameters:
    return SpectralParameters(
        size=size,
        sw=sw,
        obs=obs,
        car=car,
        aq_time=aq_time,
        apocode=apocode,
        apodq1=apodq1,
        apodq2=apodq2,
        apodq3=apodq3,
        p180=p180,
        direct=direct,
        ft=ft,
    )


def _make_spectra(apocodes: list[tuple[float, float]] | None = None):
    params = [_make_params()]  # Dimension 0 (z) placeholder
    if apocodes is None:
        apocodes = [(0.0, 0.0), (0.0, 0.0)]
    for apocode, apodq3 in apocodes:
        params.append(_make_params(apocode=apocode, apodq3=apodq3))
    return _SpectraStub(params)


class TestLineshapeFactory:
    """Unit tests for LineshapeFactory."""

    def test_create_shapes_instantiates_registered_models(self):
        spectra = _make_spectra()
        factory = LineshapeFactory(spectra, _make_options())

        shapes = factory.create_shapes(
            "Peak-1",
            positions=[1.0, 2.0],
            shape_names=["gaussian", "lorentzian"],
        )

        assert len(shapes) == 2
        assert isinstance(shapes[0], Gaussian)
        assert isinstance(shapes[1], Lorentzian)
        assert shapes[0].name == "Peak-1"

    def test_unknown_shape_raises_value_error(self):
        spectra = _make_spectra()
        factory = LineshapeFactory(spectra, _make_options())

        with pytest.raises(ValueError, match="Unknown lineshape"):
            factory.create_shapes(
                "Peak-1",
                positions=[1.0, 2.0],
                shape_names=["does-not-exist", "gaussian"],
            )

    def test_mismatched_lengths_raise_value_error(self):
        spectra = _make_spectra()
        factory = LineshapeFactory(spectra, _make_options())

        with pytest.raises(ValueError, match="Number of positions"):
            factory.create_shapes(
                "Peak-1",
                positions=[1.0],
                shape_names=["gaussian", "lorentzian"],
            )

    def test_auto_shape_names_detects_apodization(self):
        spectra = _make_spectra(apocodes=[(1.0, 1.0), (0.0, 0.0)])
        factory = LineshapeFactory(spectra, _make_options())

        assert factory.auto_shape_names() == ["sp1", "no_apod"]

    def test_create_auto_shapes_uses_detected_names(self):
        spectra = _make_spectra(apocodes=[(1.0, 1.0), (0.0, 0.0)])
        factory = LineshapeFactory(spectra, _make_options())

        shapes = factory.create_auto_shapes("Peak-2", [1.0, 2.0])

        assert isinstance(shapes[0], SP1)
        assert isinstance(shapes[1], NoApod)
        assert shapes[0].name == "Peak-2"

    def test_create_auto_shapes_with_insufficient_dimensions(self):
        spectra = _make_spectra(apocodes=[(1.0, 1.0)])
        factory = LineshapeFactory(spectra, _make_options())

        with pytest.raises(ValueError, match="Not enough automatically detected"):
            factory.create_auto_shapes("Peak-2", [1.0, 2.0])
