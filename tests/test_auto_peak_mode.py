from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from typer.testing import CliRunner

from peakfit.cli.app import app
from peakfit.cli.commands.fit_setup import write_autopicked_peaklist
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.peaks import Peak
from peakfit.fit.auto_pick_types import AutoPickDiagnostics, AutoPickResult
from peakfit.fit.fitting import load_data
from peakfit.fit.validation import validate_inputs
from peakfit.shared.exceptions import DataIOError

if TYPE_CHECKING:
    from peakfit.engine.types import Shape


def _peak(name: str, positions: tuple[float, float]) -> Peak:
    shapes = cast("list[Shape]", [SimpleNamespace(), SimpleNamespace()])
    return Peak(
        name=name,
        positions=np.array(positions, dtype=np.float64),
        shapes=shapes,
    )


def test_load_data_uses_auto_pick_when_peaklist_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    spectra = SimpleNamespace(data=np.zeros((2, 4, 4), dtype=np.float64))
    config = PeakFitConfig()
    captured: dict[str, object] = {}
    auto_peaks = [_peak("p1", (8.1, 120.2)), _peak("p2", (7.9, 118.4))]

    monkeypatch.setattr("peakfit.fit.fitting.read_spectra", lambda *_args, **_kwargs: spectra)
    monkeypatch.setattr("peakfit.fit.fitting.prepare_noise_level", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(
        "peakfit.fit.fitting.get_shape_names", lambda *_args, **_kwargs: ["gaussian", "gaussian"]
    )

    def _auto_pick(
        _spectra: object,
        shape_names: list[str],
        noise: float,
        contour_level: float,
        _config: PeakFitConfig,
        cycle_callback: object | None = None,
    ) -> AutoPickResult:
        captured["shape_names"] = shape_names
        captured["noise"] = noise
        captured["contour_level"] = contour_level
        captured["cycle_callback"] = cycle_callback
        return AutoPickResult(
            peaks=auto_peaks,
            diagnostics=AutoPickDiagnostics(
                iterations=1,
                accepted_rois=1,
                rejected_rois=0,
                accepted_peaks=2,
                stopped_by_user=False,
            ),
        )

    monkeypatch.setattr("peakfit.fit.fitting.auto_pick_peaks", _auto_pick)

    def _create_clusters(_spectra: object, peaks: list[Peak], contour: float) -> list[str]:
        captured["cluster_peaks"] = peaks
        captured["cluster_contour"] = contour
        return ["cluster_1"]

    monkeypatch.setattr("peakfit.fit.fitting.create_clusters", _create_clusters)

    loaded = load_data(Path("spectrum.ft2"), None, None, config)

    assert loaded.peaks == auto_peaks
    assert loaded.clusters == ["cluster_1"]
    assert loaded.contour_level == pytest.approx(10.0)
    assert captured["shape_names"] == ["gaussian", "gaussian"]
    assert captured["noise"] == pytest.approx(2.0)
    assert captured["contour_level"] == pytest.approx(10.0)
    assert captured["cycle_callback"] is None
    assert captured["cluster_peaks"] == auto_peaks
    assert captured["cluster_contour"] == pytest.approx(10.0)


def test_load_data_raises_when_auto_pick_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    spectra = SimpleNamespace(data=np.zeros((2, 4, 4), dtype=np.float64))
    config = PeakFitConfig()
    config.auto_peak.enabled = False

    monkeypatch.setattr("peakfit.fit.fitting.read_spectra", lambda *_args, **_kwargs: spectra)
    monkeypatch.setattr("peakfit.fit.fitting.prepare_noise_level", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(
        "peakfit.fit.fitting.get_shape_names", lambda *_args, **_kwargs: ["gaussian", "gaussian"]
    )

    with pytest.raises(DataIOError, match="automatic peak picking is disabled"):
        load_data(Path("spectrum.ft2"), None, None, config)


def test_validation_skips_peaklist_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "peakfit.fit.validation._validate_spectrum",
        lambda _spectrum_path, _result: None,
    )

    result = validate_inputs(Path("spectrum.ft2"), None)

    assert result.errors == []
    assert result.is_valid


def test_write_autopicked_peaklist(tmp_path: Path) -> None:
    peaks = [_peak("p1", (8.123456, 120.654321)), _peak("p2", (7.987654, 118.123456))]

    peaklist_path = write_autopicked_peaklist(tmp_path, peaks)

    lines = peaklist_path.read_text().splitlines()
    assert lines[0] == "Assignment w1 w2"
    assert lines[1] == "p1 8.123456 120.654321"
    assert lines[2] == "p2 7.987654 118.123456"


def test_fit_cli_without_peaklist_records_autopicked_peaklist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`peakfit fit spectrum` should use auto-picked peaks as the run peak list."""
    spectrum_path = tmp_path / "spectrum.ft2"
    spectrum_path.write_text("placeholder", encoding="utf-8")
    output_base = tmp_path / "results"
    auto_peak = SimpleNamespace(name="auto1", positions=np.array([8.1, 120.2], dtype=np.float64))
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "peakfit.cli.commands.fit.validate_inputs",
        lambda _spectrum, _peaklist: SimpleNamespace(errors=[]),
    )

    def _load_fit_data(*, peaklist: Path | None, **_kwargs: object) -> object:
        captured["peaklist_arg"] = peaklist
        return SimpleNamespace(peaks=[auto_peak])

    def _run_fit(_data: object, _config: object, output_dir: Path, **_kwargs: object) -> object:
        output_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(output_dir=output_dir, spectra=SimpleNamespace())

    def _write_fit_run_outputs(
        _result: object,
        _spectra: object,
        _config: object,
        input_paths: dict[str, Path],
        _reporter: object,
    ) -> None:
        captured["input_paths"] = input_paths

    monkeypatch.setattr("peakfit.cli.commands.fit._load_fit_data", _load_fit_data)
    monkeypatch.setattr("peakfit.cli.commands.fit.run_fit", _run_fit)
    monkeypatch.setattr("peakfit.cli.commands.fit.write_fit_run_outputs", _write_fit_run_outputs)

    result = CliRunner().invoke(
        app,
        [
            "fit",
            str(spectrum_path),
            "--output",
            str(output_base),
            "--headless",
            "--format",
            "json",
            "--format",
            "csv",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["peaklist_arg"] is None

    input_paths = cast("dict[str, Path]", captured["input_paths"])
    peaklist_path = input_paths["peaklist"]

    assert peaklist_path.name == "autopicked.list"
    assert peaklist_path.exists()
    assert "auto1 8.100000 120.200000" in peaklist_path.read_text(encoding="utf-8")
