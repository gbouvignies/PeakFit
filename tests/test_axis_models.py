"""Per-axis lineshape models: family, splitting and phase on each spectral axis (ADR 0004).

Every entry point (TOML, CLI, GUI) builds one ``FitConfig``; ``axis_lineshapes``
resolves it against the spectrum, and each axis's shape takes its own family,
doublet J (start, bounds, freedom) and phase.
"""

import itertools
import json
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_args
from zipfile import ZipFile

import numpy as np
import pytest
import typer
from builders import completed_result, two_cluster_dataset
from click.utils import strip_ansi
from typer.testing import CliRunner

from peakfit.cli.app import app
from peakfit.cli.commands.fit_setup import build_fit_config, lineshape_options
from peakfit.engine.algorithms.noise_correlation import not_modelled
from peakfit.engine.domain.config import (
    PHASED_FAMILIES,
    AxisModel,
    Doublet,
    FitConfig,
    LineshapeFamily,
    PeakFitConfig,
)
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters, axis_lineshapes
from peakfit.engine.lineshapes.create import SHAPES, create_shapes
from peakfit.engine.lineshapes.templates import PhasedSingletBase
from peakfit.fit.fitting import load_data
from peakfit.fit.pipeline import run_pipeline
from peakfit.gui.dataset_fit import DatasetFitAttempt, finish_dataset_fit
from peakfit.gui.editor.setup import PER_AXIS, FitSetup
from peakfit.gui.fit_export import export_fit_result
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.spectrum import read_dataset
from peakfit.io.config import generate_default_config, load_config
from peakfit.io.readers.continuation import load_completed
from peakfit.io.readers.results import ResultsLoader
from peakfit.io.schemas import FitSummarySchema
from peakfit.io.scientific import BEFORE_AXIS_MODELS

if TYPE_CHECKING:
    from peakfit.engine.lineshapes.base import ShapeBase

DATA = Path(__file__).resolve().parents[1] / "examples" / "02-advanced-fitting" / "data"
FAMILIES = [f for f in get_args(LineshapeFamily) if f != "auto"]
SPLITTINGS = [None, Doublet(j=7.0), Doublet(j=92.0, min=80.0, max=100.0, vary=False)]
# NMRPipe apodization codes and what ``auto`` reads from them.
APODIZATIONS = {"sp1": (1.0, 1.0), "sp2": (1.0, 2.0), "no_apod": (0.0, 0.0), "pvoigt": (3.0, 0.0)}


def _axis(label, *, size=64, direct=False, apodization="sp1"):
    apocode, apodq3 = APODIZATIONS[apodization]
    return SpectralParameters(
        size=size,
        sw=1000.0 if direct else 400.0,
        obs=600.0 if direct else 60.0,
        car=0.0,
        aq_time=size / (1000.0 if direct else 400.0),
        apocode=apocode,
        apodq1=0.35,
        apodq2=0.98,
        apodq3=apodq3,
        p180=False,
        direct=direct,
        ft=True,
        label=label,
    )


def _spectra(f2="sp1", f3="sp2"):
    planes = SpectralParameters(
        size=3, sw=1.0, obs=1.0, car=0.0, aq_time=1.0, apocode=0.0, apodq1=0.0, apodq2=0.0,
        apodq3=0.0, p180=False, direct=False, ft=True, label="F1",
    )  # fmt: skip
    return Spectra(
        dic={},
        data=np.zeros((3, 64, 64)),
        z_values=np.arange(3.0),
        params=[planes, _axis("F2", apodization=f2), _axis("F3", direct=True, apodization=f3)],
    )


def _parameters(config, spectra=None):
    """The model parameters of one peak and its cluster, by name."""
    spectra = spectra or _spectra()
    positions = [float(axis.pts2ppm(32.0)) for axis in spectra.spectral_params]
    peak = Peak(
        "A",
        np.array(positions),
        create_shapes(spectra, axis_lineshapes(config, spectra), "A", positions),
    )
    peak.set_cluster_id(1)
    return peak, Parameters.from_peaks([peak])


# --------------------------------------------------------------------------- model


def test_a_doublet_needs_its_j_start_and_takes_bounds_around_it():
    assert (Doublet(j=92.0).min, Doublet(j=92.0).max, Doublet(j=92.0).vary) == (46.0, 184.0, True)
    assert Doublet(j=7.0, min=3.0).max == 14.0
    with pytest.raises(ValueError, match="Field required"):
        Doublet()  # ty: ignore[missing-argument]
    with pytest.raises(ValueError, match=r"must contain its start 92 Hz"):
        Doublet(j=92.0, min=95.0, max=100.0)
    with pytest.raises(ValueError, match="greater than 0"):
        Doublet(j=-5.0)


# ``auto`` resolves from _spectra()'s apodization: sp1 on F2, sp2 on F3.
DETECTED = {"F2": "sp1", "F3": "sp2"}


@pytest.mark.parametrize(
    ("axis", "family", "splitting", "phase"),
    list(itertools.product(["F2", "F3"], ["auto", *FAMILIES], SPLITTINGS, [False, True])),
)
def test_every_family_splitting_and_phase_builds_its_own_shapes_and_parameters(
    axis, family, splitting, phase
):
    # The model under test is on one axis; the other keeps a plain Gaussian singlet.
    other = "F3" if axis == "F2" else "F2"
    config = FitConfig(
        family="gaussian", axes={axis: AxisModel(family=family, doublet=splitting, phase=phase)}
    )
    resolved = DETECTED[axis] if family == "auto" else family
    if phase and resolved not in PHASED_FAMILIES:
        with pytest.raises(
            ValueError, match=rf"{axis}: phase fitting needs a phased lineshape.*{resolved}"
        ):
            axis_lineshapes(config, _spectra())
        return
    peak, params = _parameters(config)
    tested, plain = (s for s in sorted(peak.shapes, key=lambda s: s.axis != axis))
    assert (tested.axis, tested.shape_name, tested.doublet) == (
        axis,
        resolved,
        splitting is not None,
    )
    assert (plain.axis, plain.shape_name, plain.doublet) == (other, "gaussian", False)
    expected = {f"A.{a}.{p}" for a in ("F2", "F3") for p in ("cs", "lw")}
    expected |= {f"A.{axis}.eta"} if resolved == "pvoigt" else set()
    expected |= {f"A.{axis}.j"} if splitting is not None else set()
    expected |= {f"cluster_1.{axis}.phase"} if phase else set()
    assert set(params) == expected
    if splitting is not None:
        j = params[f"A.{axis}.j"]
        assert (j.value, j.min, j.max, j.vary) == (
            splitting.j,
            splitting.min,
            splitting.max,
            splitting.vary,
        )


def test_the_phased_families_are_the_complex_lineshapes():
    classes = {
        f: cast("tuple[type[ShapeBase], type[ShapeBase]]", pair) for f, pair in SHAPES.items()
    }
    phased = {f for f, (singlet, _) in classes.items() if issubclass(singlet, PhasedSingletBase)}
    assert phased == PHASED_FAMILIES
    assert all(d.doublet and not s.doublet for s, d in classes.values())


def test_a_doublet_on_one_axis_and_a_singlet_on_the_other():
    config = FitConfig(family="lorentzian", axes={"F2": AxisModel(doublet=Doublet(j=92.0))})
    peak, params = _parameters(config)
    assert [s.description for s in peak.shapes] == ["lorentzian doublet", "lorentzian"]
    assert [n for n in params if n.endswith(".j")] == ["A.F2.j"]
    # The doublet's components sit at cs ± J/2 on F2 only: F3 is unchanged by J.
    cluster_values = {}
    for j in (60.0, 120.0):
        params["A.F2.j"].value = j
        cluster_values[j] = [
            s.evaluate_cluster(
                np.arange(64.0), s.get_cluster_parameters([peak], params)
            ).values.ravel()
            for s in peak.shapes
        ]
    assert not np.allclose(cluster_values[60.0][0], cluster_values[120.0][0])
    np.testing.assert_array_equal(cluster_values[60.0][1], cluster_values[120.0][1])


def test_phase_is_fitted_on_one_axis_only():
    config = FitConfig(family="sp1", axes={"F3": AxisModel(phase=True)})
    _peak, params = _parameters(config)
    assert [n for n in params if n.endswith(".phase")] == ["cluster_1.F3.phase"]


def test_auto_takes_each_axis_family_from_its_apodization():
    lineshapes = axis_lineshapes(FitConfig(), _spectra(f2="no_apod", f3="sp2"))
    assert [(m.axis, m.family, m.detected) for m in lineshapes] == [
        ("F2", "no_apod", True),
        ("F3", "sp2", True),
    ]
    mixed = FitConfig(axes={"F3": AxisModel(family="lorentzian")})
    assert [(m.family, m.detected) for m in axis_lineshapes(mixed, _spectra())] == [
        ("sp1", True),
        ("lorentzian", False),
    ]


def test_phase_on_a_family_detected_without_phase_is_refused():
    config = FitConfig(axes={"F2": AxisModel(phase=True)})
    with pytest.raises(ValueError, match=r"F2: .*detected from the apodization family is pvoigt"):
        axis_lineshapes(config, _spectra(f2="pvoigt"))


def test_an_axis_the_spectrum_lacks_is_refused():
    with pytest.raises(ValueError, match=r"names F4, which this spectrum lacks.*F2, F3"):
        axis_lineshapes(FitConfig(axes={"F4": AxisModel(phase=True)}), _spectra())
    with pytest.raises(ValueError, match="Axis labels must look like F2"):
        FitConfig(axes={"x": AxisModel()})


# ------------------------------------------------------------------ entry points

TOML = """
[fitting]
family = "sp1"
refine_iterations = 2

[fitting.axes.F2]
doublet = { j = 92.0, min = 80.0, max = 100.0 }

[fitting.axes.F3]
family = "sp2"
phase = true
doublet = { j = 7.0, vary = false }
"""
CLI = {
    "families": ["sp1", "F3=sp2"],
    "doublets": ["F2=92,80,100", "F3=7"],
    "fixed_j": ["F3"],
    "phases": ["F3"],
}


def test_toml_and_cli_build_the_same_config(tmp_path):
    path = tmp_path / "peakfit.toml"
    path.write_text(TOML)
    from_toml = load_config(path)
    from_cli = build_fit_config(
        None, None, None, None, None, None, lineshape_options(**CLI), refine=2, exclude=None
    )
    assert from_cli.fitting == from_toml.fitting
    # A stored configuration (projects, results) reads back as the same model.
    stored = PeakFitConfig.model_validate_json(from_toml.model_dump_json())
    assert stored.fitting == from_toml.fitting
    lineshapes = axis_lineshapes(from_toml.fitting, _spectra())
    assert [(m.family, m.doublet, m.phase) for m in lineshapes] == [
        ("sp1", Doublet(j=92.0, min=80.0, max=100.0), False),
        ("sp2", Doublet(j=7.0, min=3.5, max=14.0, vary=False), True),
    ]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"families": ["voigt"]}, "Unknown lineshape family 'voigt'"),
        ({"doublets": ["F2"]}, "--doublet takes AXIS=VALUE"),
        ({"doublets": ["F2=7,3"]}, "--doublet takes AXIS=J or AXIS=J,MIN,MAX"),
        ({"doublets": ["F2=7,8,9"]}, "must contain its start 7 Hz"),
        ({"fixed_j": ["F2"]}, "--fix-j F2 needs a doublet on F2"),
    ],
)
def test_invalid_cli_lineshape_options_are_refused(options, message):
    arguments = {"families": None, "doublets": None, "fixed_j": None, "phases": None} | options
    with pytest.raises(typer.BadParameter, match=message):
        lineshape_options(**arguments)


def test_the_gui_family_applies_to_every_axis_and_keeps_doublets_and_phase(qapp, tmp_path):
    del qapp
    path = tmp_path / "peakfit.toml"
    path.write_text(TOML)
    imported = load_config(path)
    setup = FitSetup()
    emitted = []
    setup.changed.connect(emitted.append)
    setup.show_state(imported, [], None, "")
    assert setup.lineshape.currentText() == PER_AXIS  # F3 sets its own family
    setup.lineshape.setCurrentIndex(setup.lineshape.findData("lorentzian"))
    (chosen,) = emitted
    expected = imported.fitting.model_copy(deep=True)
    expected.family = "lorentzian"
    expected.axes["F3"] = expected.axes["F3"].model_copy(update={"family": None})
    assert chosen.fitting == expected
    setup.show_state(chosen, [], None, "")
    assert setup.lineshape.currentData() == "lorentzian"


def test_each_axis_doublet_sets_its_own_j_start_bounds_and_freedom():
    spectra, peaks, config, _prepared = two_cluster_dataset()
    config.fitting.axes = {
        "F2": AxisModel(doublet=Doublet(j=30.0, min=10.0, max=60.0)),
        "F3": AxisModel(doublet=Doublet(j=12.0, vary=False)),
    }
    prepared = prepare_dataset_fit(spectra, peaks, config)
    couplings = {
        name.partition(".")[2]: (p.value, p.min, p.max, p.vary)
        for name, p in prepared.parameters.items()
        if name.endswith(".j")
    }
    assert couplings == {"F2.j": (30.0, 10.0, 60.0, True), "F3.j": (12.0, 6.0, 24.0, False)}
    assert [m.doublet for m in prepared.lineshapes] == [
        config.fitting.axes["F2"].doublet,
        config.fitting.axes["F3"].doublet,
    ]


def test_cli_and_gui_build_the_same_model_from_one_config():
    # The bundled example: auto resolves sp2 on both axes, so phase may be fitted.
    config = PeakFitConfig(
        fitting=FitConfig(
            axes={
                "F2": AxisModel(doublet=Doublet(j=92.0, min=80.0, max=100.0)),
                "F3": AxisModel(phase=True),
            }
        )
    )
    cli = load_data(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt", config)
    dataset = read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt")
    gui = prepare_dataset_fit(dataset.spectra, dataset.peaks, config)
    assert cli.lineshapes == list(gui.lineshapes)

    def model(peaks, params):
        """Every parameter's settings, by axis and label, independent of peak naming."""
        rows = sorted(
            (p.param_id.axis, p.param_id.label, p.value, p.min, p.max, p.vary)
            for p in params.values()
            if p.param_id is not None
        )
        shapes = sorted((s.axis, s.shape_name, s.doublet) for peak in peaks for s in peak.shapes)
        return rows, shapes

    cli_peaks = [peak for cluster in cli.clusters for peak in cluster.peaks]
    gui_peaks = [peak for cluster in gui.clusters for peak in cluster.peaks]
    assert model(cli_peaks, Parameters.from_peaks(cli_peaks)) == model(gui_peaks, gui.parameters)
    ids = [p.param_id for p in gui.parameters.values() if p.param_id is not None]
    labels = {(i.axis, i.label) for i in ids}
    assert {("F2", "j"), ("F3", "phase")} <= labels
    assert not {("F3", "j"), ("F2", "phase")} & labels


# ----------------------------------------------------------------------- storage


def _mixed_model_result(tmp_path):
    """A fit with a doublet on F2 only, recorded as the GUI and CLI export it."""
    spectra, peaks, config, _prepared = two_cluster_dataset()
    config.fitting.axes = {"F2": AxisModel(doublet=Doublet(j=8.0, min=2.0, max=20.0))}
    config.output.detailed_evidence = True
    prepared = prepare_dataset_fit(spectra, peaks, config)
    pipeline = run_pipeline(
        config,
        prepared.clusters,
        prepared.noise,
        prepared.parameters,
        noise_model=not_modelled("synthetic spectrum"),
    )
    result = finish_dataset_fit(prepared, DatasetFitAttempt(pipeline.completed_fit, (), 1.0))
    return export_fit_result(result, tmp_path / "results")


def test_a_per_axis_model_round_trips_through_the_scientific_archive(tmp_path):
    directory = _mixed_model_result(tmp_path)
    restored = load_completed(directory / "scientific.npz")
    shapes = [s for c in restored.model_snapshot.clusters for p in c.peaks for s in p.shapes]
    assert {(s.axis, s.description) for s in shapes} == {
        ("F2", "gaussian doublet"),
        ("F3", "gaussian"),
    }
    assert all(s.lineshape.doublet == Doublet(j=8.0, min=2.0, max=20.0) for s in shapes[::2])


def _rewrite_archive(archive, mutate):
    with ZipFile(archive) as source:
        members = {name: source.read(name) for name in source.namelist()}
    document = json.loads(members["metadata.json"])
    mutate(document)
    members["metadata.json"] = json.dumps(document).encode()
    with ZipFile(archive, "w") as target:
        for name, payload in members.items():
            target.writestr(name, payload)


def test_an_archive_from_before_per_axis_models_names_the_peakfit_that_opens_it(tmp_path):
    _, _, _, result = completed_result([], False, True)
    directory = export_fit_result(result, tmp_path / "results")

    def as_before(document):
        document["schema_version"] = "12.0.0"
        fitting = document["context"]["mapping"]["metadata"]["mapping"]["configuration"]
        fitting["mapping"]["fitting"]["mapping"]["lineshape"] = "gaussian"

    _rewrite_archive(directory / "scientific.npz", as_before)
    with pytest.raises(
        ValueError,
        match=rf"'12\.0\.0'.*predates per-axis lineshape models; PeakFit at commit "
        rf"{BEFORE_AXIS_MODELS} opens it",
    ):
        ResultsLoader(directory).load_completed()


def test_fit_json_from_before_per_axis_models_names_the_peakfit_that_opens_it():
    with pytest.raises(ValueError, match=rf"'9\.0\.0'.*commit {BEFORE_AXIS_MODELS} opens it"):
        FitSummarySchema.model_validate({"schema_version": "9.0.0"})


# ------------------------------------------------------------------- old inputs


@pytest.mark.parametrize(
    ("fitting", "replacement"),
    [
        (
            'lineshape = "lorentzian"',
            r"'lineshape' was replaced by 'family'.*family = \"lorentzian\"\.",
        ),
        (
            'lineshape = "sp1_doublet"',
            r"family = \"sp1\", and on each split axis a doublet.*doublet = \{ j = 92\.0 \}"
            r".*started at 5 Hz within \[1, 10\] Hz",
        ),
        (
            'fit_phase = ["F2", "F3"]',
            r"phase = true under \[fitting\.axes\.F2\] and \[fitting\.axes\.F3\]\.",
        ),
    ],
)
def test_an_old_toml_field_is_refused_with_its_replacement(tmp_path, fitting, replacement):
    path = tmp_path / "old.toml"
    path.write_text(f"[fitting]\n{fitting}\n")
    with pytest.raises(ValueError, match=replacement):
        load_config(path)


def test_the_default_config_template_loads():
    config = PeakFitConfig.model_validate(tomllib.loads(generate_default_config()))
    assert config.fitting == FitConfig(refine_iterations=1)


@pytest.mark.parametrize(
    ("arguments", "replacement"),
    [
        (
            ["--lineshape", "lorentzian_doublet"],
            "use --family lorentzian plus --doublet AXIS=J on each split axis (the former "
            "doublets were --doublet AXIS=5,1,10)",
        ),
        (["--phx", "--phy"], "use --phase F2 --phase F3"),
    ],
)
def test_old_cli_options_are_refused_with_their_replacement(tmp_path, arguments, replacement):
    spectrum = tmp_path / "s.ft2"
    peaks = tmp_path / "p.list"
    spectrum.write_bytes(b"")
    peaks.write_text("")
    result = CliRunner().invoke(app, ["fit", str(spectrum), str(peaks), *arguments])
    assert result.exit_code == 2
    assert replacement in " ".join(strip_ansi(result.output).replace("│", " ").split())
