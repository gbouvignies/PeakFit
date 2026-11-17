"""The peak page qualifies a fitted J: below its linewidth, and its ρ (#214)."""

from copy import deepcopy
from pathlib import Path

import pytest
from builders import overlapping_gaussians
from gui_helpers import wait_until
from PySide6 import QtCore, QtWidgets

from peakfit.engine.diagnostics.couplings import CouplingQualification
from peakfit.engine.domain.config import AxisModel, Doublet, FitConfig
from peakfit.gui import formatting
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.editor.inspector import COUPLING_TIP, coupling_caption, coupling_shown
from peakfit.gui.editor.workspace import EditorWorkspace
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.peaks import WorkingEdits
from peakfit.gui.spectrum import Dataset

pytestmark = pytest.mark.gui


def test_the_caption_names_the_qualification_and_each_correlation() -> None:
    below = CouplingQualification("A.F3.j", "A.F3.lw", True, -0.79, "cluster_1.F3.phase", 0.27)
    assert coupling_caption(below, "¹H F3") == (
        f"J ¹H F3 below LW: determined together · ρ(J, LW) {formatting.correlation(-0.79)}"
        f" · ρ(J, phase) {formatting.correlation(0.27)}"
    )
    withheld = CouplingQualification("A.F3.j", "A.F3.lw", True)
    assert coupling_caption(withheld, "¹H F3") == "J ¹H F3 below LW: determined together"
    resolved = CouplingQualification("A.F3.j", "A.F3.lw", False, -0.03)
    assert coupling_caption(resolved, "¹H F3") == (
        f"J ¹H F3 · ρ(J, LW) {formatting.correlation(-0.03)}"
    )
    assert not coupling_shown(CouplingQualification("A.F3.j", "A.F3.lw", False))
    assert coupling_shown(resolved)


@pytest.fixture(scope="module")
def doublet_fit():
    """Singlet data fitted with a ¹H doublet: J settles below the 25 Hz linewidth."""
    spectra, peaks, config = overlapping_gaussians()
    config.fitting = FitConfig(
        family="gaussian", axes={"F3": AxisModel(doublet=Doublet(j=6.0, min=0.5, max=12.0))}
    )
    config.clustering.contour_level = 0.01
    prepared = prepare_dataset_fit(spectra, peaks, config)
    result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
    return spectra, peaks, config, result


def test_the_peak_page_shows_a_j_below_its_linewidth_as_a_qualification(qapp, doublet_fit):
    spectra, peaks, config, result = doublet_fit
    widget = EditorWorkspace(Dataset(Path("synthetic.ft2"), spectra, peaks, (), True, 0.01))
    widget.edits = WorkingEdits(widget.peaks, config.model_copy(deep=True))
    widget.fitting.model_seen = True
    widget.resize(1280, 750)
    widget.show()
    try:
        fitting = widget.fitting
        fitting._submitted_revision = widget.edits.revision
        fitting.process.state = "completed"
        fitting._finished(0, result)
        wait_until(lambda: not fitting.busy, timeout=10)
        couplings = result.completed_fit.couplings()
        identities = {i.engine_name: i.working_id for i in result.inputs.identities}
        engine, qualification = next(
            (name.split(".")[0], q) for name, q in couplings.items() if q.below_linewidth
        )
        widget.select_peak(identities[engine])
        fitted = widget.panel.fitted
        items = [fitted.itemAt(i) for i in range(fitted.count())]
        captions = {
            label.text(): label.toolTip()
            for item in items
            if item is not None and isinstance(label := item.widget(), QtWidgets.QLabel)
        }
        expected = coupling_caption(qualification, widget.panel._axes.get("F3", "F3"))
        assert expected.startswith("J ")
        assert "below LW" in expected
        assert captions[expected] == COUPLING_TIP
    finally:
        widget.stop()
        widget.close()
        widget.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        qapp.processEvents()
