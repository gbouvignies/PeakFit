"""Validate the default linewidth start of sp1/sp2/no_apod (#135).

Two studies, both on the bundled CEST spectrum (examples/02-advanced-fitting):

- ``real``: the whole dataset fitted as the GUI does (two passes), with every
  linewidth starting at a common value. Per start: clusters with the uncertainty
  warning, clusters whose chi-squared exceeds the best over all starts, and time.
- ``synthetic``: two SP2 lines with known truth on the spectrum's axes, CEST-like
  profiles, white noise, listed positions a tenth of a linewidth off. True widths
  2.5-60 Hz, separations 0.3-1 linewidth on each axis, intensity ratios 1 and 0.5,
  three noise draws. A fit fails when its chi-squared exceeds that of the fit started
  from the truth by more than 1.

    OPENBLAS_NUM_THREADS=1 uv run python tools/validate_linewidth_start.py [real|synthetic]
"""

# ruff: noqa: T201
from __future__ import annotations

import itertools
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.gui.dataset_fit import compute_dataset_fit, finish_dataset_fit
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.peaks import WorkingPeak
from peakfit.gui.spectrum import read_dataset

DATA = Path(__file__).resolve().parents[1] / "examples/02-advanced-fitting/data"
REAL_STARTS = (30.0, 20.0, 15.0, 12.0, 10.0, 6.0, 4.0, 2.0, 1.0)
SYNTHETIC_STARTS = (20.0, 10.0, 5.0, 2.0, 1.0)
WIDTHS = (2.5, 5.0, 10.0, 20.0, 35.0, 60.0)
SEPARATIONS = (0.3, 0.5, 1.0)
RATIOS = (1.0, 0.5)
DRAWS = 3
CENTER = (118.0, 8.2)  # ppm, F2 (15N) and F3 (1H)
MARGINS = (1, 10)  # chi-squared excess counted as worse, and as clearly worse


def _dataset():
    return read_dataset(DATA / "pseudo3d.ft2", DATA / "pseudo3d.list", DATA / "b1_offsets.txt")


def _start(prepared, linewidth):
    for name, parameter in prepared.parameters.params.items():
        if name.endswith(".lw"):
            parameter.value = linewidth
    return prepared


def real() -> None:
    """Whole-dataset fits from common linewidth starts."""
    dataset = _dataset()
    settings = PeakFitConfig(fitting=FitConfig(refine_iterations=2))
    base = prepare_dataset_fit(dataset.spectra, tuple(dataset.peaks), settings)
    names = {i.engine_name: i.assignment for i in base.identities}
    runs = {}
    for start in REAL_STARTS:
        began = time.perf_counter()
        prepared = _start(deepcopy(base), start)
        result = finish_dataset_fit(prepared, compute_dataset_fit(deepcopy(prepared)))
        seconds = time.perf_counter() - began
        runs[start] = (
            seconds,
            {
                "/".join(sorted(names[p] for p in c.peak_names)): (
                    evaluation.statistics.chi_squared,
                    evaluation.uncertainty_warning is not None,
                )
                for c in result.outcome.clusters
                if (evaluation := c.analytical_evaluation) is not None
            },
        )
    best = {k: min(run[1][k][0] for run in runs.values()) for k in runs[REAL_STARTS[0]][1]}
    print("start Hz | warned | worse than best by >1 / >10 | total chi2 | seconds")
    for start, (seconds, clusters) in runs.items():
        excess = np.array([clusters[k][0] - best[k] for k in best])
        warned = sorted(k for k, (_, w) in clusters.items() if w)
        total = sum(chi2 for chi2, _ in clusters.values())
        worse = " / ".join(str((excess > margin).sum()) for margin in MARGINS)
        print(f"{start:5.1f} | {len(warned)} {warned} | {worse} | {total:.0f} | {seconds:.1f}")


def _values(spectra, settings, name, position, linewidth):
    shapes = create_shapes(spectra, settings.fitting, name, position, ["sp2", "sp2"])
    peak = Peak(name, np.array(position), shapes)
    params = Parameters.from_peaks([peak])
    for axis in ("F2", "F3"):
        params[f"{name}.{axis}.lw"].value = linewidth
    rows, columns = np.indices(spectra.data.shape[1:])
    values = Cluster.evaluate_peaks([peak], params, [rows.ravel(), columns.ravel()])[0]
    return values.reshape(rows.shape)


def _case(spectra, settings, width, separation, ratio, draw):
    rng = np.random.default_rng([int(10 * width), int(10 * separation), int(10 * ratio), draw])
    y, x = spectra.params[1], spectra.params[2]
    half = (separation * width / y.obs / 2, separation * width / x.obs / 2)
    truth = {
        "a": (CENTER[0] - half[0], CENTER[1] - half[1]),
        "b": (CENTER[0] + half[0], CENTER[1] + half[1]),
    }
    planes = np.arange(spectra.data.shape[0])
    profiles = (
        1 - 0.8 * np.exp(-(((planes - 40) / 6) ** 2)),
        1 - 0.6 * np.exp(-(((planes - 80) / 10) ** 2)),
    )
    a = _values(spectra, settings, "a", truth["a"], width)
    b = _values(spectra, settings, "b", truth["b"], width)
    data = rng.standard_normal(spectra.data.shape)
    data += 30 / a.max() * (profiles[0][:, None, None] * a + ratio * profiles[1][:, None, None] * b)
    off = rng.normal(0, 0.1 * width, 4)
    peaks = (
        WorkingPeak("a", "A", (truth["a"][0] + off[0] / y.obs, truth["a"][1] + off[1] / x.obs)),
        WorkingPeak("b", "B", (truth["b"][0] + off[2] / y.obs, truth["b"][1] + off[3] / x.obs)),
    )
    prepared = prepare_dataset_fit(spectra.model_copy(update={"data": data}), peaks, settings)
    if len(prepared.clusters) != 1:
        raise RuntimeError("The pair must form one cluster.")
    return prepared, truth


def _fit(prepared):
    (cluster,) = compute_dataset_fit(deepcopy(prepared)).completed_fit.outcome.clusters
    return cluster


def synthetic() -> None:
    """Overlapping pairs with known truth, from each start and from the truth."""
    spectra = _dataset().spectra
    settings = PeakFitConfig(fitting=FitConfig(refine_iterations=1))
    print("true lw Hz, separation | failures of 6 per start " + str(SYNTHETIC_STARTS))
    evaluations = dict.fromkeys(SYNTHETIC_STARTS, 0)
    for width, separation in itertools.product(WIDTHS, SEPARATIONS):
        failures = dict.fromkeys(SYNTHETIC_STARTS, 0)
        for ratio, draw in itertools.product(RATIOS, range(DRAWS)):
            prepared, truth = _case(spectra, settings, width, separation, ratio, draw)
            from_truth = deepcopy(prepared)
            engine = {i.engine_name: i.assignment.lower() for i in prepared.identities}
            for name, parameter in from_truth.parameters.params.items():
                peak, axis, kind = name.split(".")
                position = truth[engine[peak]][0 if axis == "F2" else 1]
                parameter.value = {"cs": position, "lw": width}.get(kind, parameter.value)
            reference = _fit(from_truth).analytical_evaluation.statistics.chi_squared
            for start in SYNTHETIC_STARTS:
                cluster = _fit(_start(deepcopy(prepared), start))
                chi2 = cluster.analytical_evaluation.statistics.chi_squared
                failures[start] += int(chi2 > reference + MARGINS[0])
                evaluations[start] += cluster.optimizer_provenance.function_evaluations or 0
        print(f"{width:5.1f}, {separation:.1f} | {list(failures.values())}", flush=True)
    print("function evaluations per start:", evaluations)


if __name__ == "__main__":
    for study in sys.argv[1:] or ["real", "synthetic"]:
        {"real": real, "synthetic": synthetic}[study]()
