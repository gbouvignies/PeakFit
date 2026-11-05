"""The default linewidth start reaches the true basin of overlapping pairs (#135).

A start broader than the true lines makes two overlapping shapes nearly identical
from the first step, and the fit can stop in the collapsed limit
(docs/validation/linewidth-start.md). Pairs of narrow lines on the bundled
spectrum's axes, with known truth: the default start must reach the minimum that
the fit reaches from the truth itself.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.gui.dataset_fit import compute_dataset_fit
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.peaks import WorkingPeak
from peakfit.gui.spectrum import read_dataset

DATA = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"
SETTINGS = PeakFitConfig(fitting=FitConfig(refine_iterations=1))
SIGMA = 1.0
CENTER = (118.0, 8.2)  # ppm, F2 (¹⁵N) and F3 (¹H)


@pytest.fixture(scope="module")
def spectra():
    return read_dataset(
        DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt"
    ).spectra


def _values(spectra, name, position, linewidth):
    peak = Peak(
        name,
        np.array(position),
        create_shapes(spectra, SETTINGS.fitting, name, position, ["sp2", "sp2"]),
    )
    params = Parameters.from_peaks([peak])
    for axis in ("F2", "F3"):
        params[f"{name}.{axis}.lw"].value = linewidth
    rows, columns = np.indices(spectra.data.shape[1:])
    return Cluster.evaluate_peaks([peak], params, [rows.ravel(), columns.ravel()])[0].reshape(
        rows.shape
    )


def _pair(spectra, linewidth, separation, ratio, seed):
    """Two lines ``separation`` linewidths apart on each axis, with CEST-like profiles."""
    rng = np.random.default_rng(seed)
    y, x = spectra.params[1], spectra.params[2]
    half = (separation * linewidth / y.obs / 2, separation * linewidth / x.obs / 2)
    a = (CENTER[0] - half[0], CENTER[1] - half[1])
    b = (CENTER[0] + half[0], CENTER[1] + half[1])
    planes = np.arange(spectra.data.shape[0])
    profiles = (
        1 - 0.8 * np.exp(-(((planes - 40) / 6) ** 2)),
        1 - 0.6 * np.exp(-(((planes - 80) / 10) ** 2)),
    )
    data = rng.normal(0, SIGMA, spectra.data.shape)
    va, vb = _values(spectra, "a", a, linewidth), _values(spectra, "b", b, linewidth)
    scale = 30 * SIGMA / va.max()
    data += scale * (profiles[0][:, None, None] * va + ratio * profiles[1][:, None, None] * vb)
    synthetic = spectra.model_copy(update={"data": data})
    # Listed positions a tenth of a linewidth off the truth, as a peak list would be.
    off = rng.normal(0, 0.1 * linewidth, 4)
    peaks = (
        WorkingPeak("a", "A", (a[0] + off[0] / y.obs, a[1] + off[1] / x.obs)),
        WorkingPeak("b", "B", (b[0] + off[2] / y.obs, b[1] + off[3] / x.obs)),
    )
    return synthetic, peaks, {"a": a, "b": b}


def _fit(prepared):
    (cluster,) = compute_dataset_fit(deepcopy(prepared)).completed_fit.outcome.clusters
    return cluster


@pytest.mark.parametrize("linewidth", [5.0, 10.0])
@pytest.mark.parametrize("separation", [0.5, 1.0])
@pytest.mark.parametrize("ratio", [1.0, 0.5])
def test_the_default_start_reaches_the_minimum_found_from_the_truth(
    spectra, linewidth, separation, ratio
):
    seed = [int(linewidth), int(10 * separation), int(10 * ratio)]
    synthetic, peaks, truth = _pair(spectra, linewidth, separation, ratio, seed)
    prepared = prepare_dataset_fit(synthetic, peaks, SETTINGS)
    assert len(prepared.clusters) == 1
    default = _fit(prepared)

    from_truth = deepcopy(prepared)
    engine = {i.engine_name: i.assignment.lower() for i in prepared.identities}
    for name, parameter in from_truth.parameters.params.items():
        peak, axis, kind = name.split(".")
        position = truth[engine[peak]][0 if axis == "F2" else 1]
        parameter.value = {"cs": position, "lw": linewidth}.get(kind, parameter.value)
    reference = _fit(from_truth)

    assert default.analytical_evaluation.uncertainty_warning is None
    assert default.analytical_evaluation.statistics.chi_squared == pytest.approx(
        reference.analytical_evaluation.statistics.chi_squared, rel=1e-6
    )
