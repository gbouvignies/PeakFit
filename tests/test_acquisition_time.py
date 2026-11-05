"""Acquisition time and window of the sp1, sp2 and no_apod lineshapes (docs/lineshapes.md).

The fixtures are noiseless spectra NMRPipe processed from exact signals of known
position, linewidth and amplitude (tools/make_nmrpipe_lineshape_fixtures.py).
"""

import json
import math
from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.fit.fitting import load_data
from peakfit.fit.pipeline import run_pipeline
from peakfit.io.readers.spectrum import read_spectral_parameters

FIXTURES = Path(__file__).parent / "data/nmrpipe/lineshapes"
TRUTH = json.loads((FIXTURES / "truth.json").read_text())
CEST = Path(__file__).parents[1] / "examples/04-uncertainty-analysis/data/pseudo3d.ft2"
GROUP_DELAY = 67.98423767089844  # FDDMXVAL, float32

# Linewidth tolerance (relative). The 32-point series hold lines far narrower than
# their digital resolution (R2·aq 0.08 for the narrowest), whose width the data barely
# determine; Bruker no_apod keeps the established N·dw less the filter delay.
TOLERANCE = {
    "sp2_t0": 0.06,
    "sp1_half_dwell": 0.10,
    "sp2_digital_filter": 0.02,
    "sp2_digital_filter_df": 0.02,
    "sp2_digital_filter_converted": 0.01,
    "no_apod_digital_filter": 0.05,
    "sp2_indirect_t0": 0.01,
    "sp2_indirect_half_dwell": 0.01,
}


def _fit(name: str, tmp_path: Path):
    truth = TRUTH[name]
    peaks = tmp_path / "peaks.list"
    peaks.write_text(
        "".join(
            f"P{k} " + " ".join(f"{ppm:.6f}" for ppm in peak["ppm"]) + "\n"
            for k, peak in enumerate(truth["peaks"])
        )
    )
    config = PeakFitConfig()
    data = load_data(FIXTURES / f"{name}.ft", peaks, None, config)
    outcome = run_pipeline(
        config,
        data.clusters,
        data.noise,
        Parameters.from_peaks(data.peaks, fixed=False),
        noise_model=data.noise_model,
    ).completed_fit.outcome
    values = {p.name: p.value for p in outcome.final_nonlinear_parameters}
    amplitudes = {
        name: float(np.asarray(cluster.analytical_evaluation.amplitudes[k])[0])
        for cluster in outcome.clusters
        if cluster.analytical_evaluation is not None
        for k, name in enumerate(cluster.peak_names)
    }
    return truth, values, amplitudes


@pytest.mark.parametrize("name", list(TRUTH))
def test_nmrpipe_processed_lines_give_their_true_linewidths(name, tmp_path):
    truth, values, amplitudes = _fit(name, tmp_path)
    dims = [f"F{i + 2}" for i in range(len(truth["axes"]))]
    for k, peak in enumerate(truth["peaks"]):
        for dim, width in zip(dims, peak["lw_hz"], strict=True):
            assert values[f"P{k}.{dim}.lw"] == pytest.approx(width, rel=TOLERANCE[name])
    # Amplitudes relative to the first peak: the shapes are normalized alike.
    first = amplitudes["P0"] / truth["peaks"][0]["amplitude"]
    for k, peak in enumerate(truth["peaks"]):
        assert amplitudes[f"P{k}"] / peak["amplitude"] == pytest.approx(first, rel=0.02)


@pytest.mark.parametrize(
    "name", ["sp2_t0", "sp1_half_dwell", "sp2_indirect_t0", "sp2_indirect_half_dwell"]
)
def test_nmrpipe_processed_lines_keep_their_positions(name, tmp_path):
    truth, values, _ = _fit(name, tmp_path)
    for k, peak in enumerate(truth["peaks"]):
        for i, ppm in enumerate(peak["ppm"]):
            assert values[f"P{k}.F{i + 2}.cs"] == pytest.approx(ppm, abs=0.05 / 800.0)  # 0.05 Hz


def _axis(name: str):
    dic, data = ng.pipe.read(str(FIXTURES / f"{name}.ft"))
    return read_spectral_parameters(dic, data)[-1]


@pytest.mark.parametrize(
    ("name", "aq", "q1", "q2"),
    [
        # First sample at t = 0, first point ×½: the sine bell spans N − 1 dwells.
        ("sp2_t0", 31 / 2000, 0.4, 0.98),
        # Half-dwell (P1 180): N − ½ dwells, the window starting half a point early.
        ("sp1_half_dwell", 31.5 / 2000, 0.4 - 0.58 * 0.5 / 31, 0.98),
        # Digital filter corrected during FT: the signal starts g points into the window.
        ("sp2_digital_filter", (511 - GROUP_DELAY) / 8000, 0.4 + 0.58 * GROUP_DELAY / 511, 0.98),
        # SP -df shifts the window by g instead, so its end is cut at the last point.
        ("sp2_digital_filter_df", (511 - GROUP_DELAY) / 8000, 0.4, 0.98 - 0.58 * GROUP_DELAY / 511),
        # Filter corrected at conversion (bruk2pipe -DMX): the header records no delay and
        # TDSIZE (443 of 512) already excludes it, so nothing is subtracted twice.
        ("sp2_digital_filter_converted", 442 / 8000, 0.4, 0.98),
        # Without a window: the established N dwells, less the filter delay.
        ("no_apod_digital_filter", (512 - GROUP_DELAY) / 8000, 0.0, 0.0),
    ],
)  # fmt: skip
def test_the_window_spans_the_acquired_signal(name, aq, q1, q2):
    axis = _axis(name)
    assert axis.aq_time == pytest.approx(aq, rel=1e-6)
    assert (axis.apodq1, axis.apodq2) == (pytest.approx(q1, abs=1e-6), pytest.approx(q2, abs=1e-6))


def test_an_extracted_axis_uses_the_acquired_size_and_full_spectral_width():
    # The CEST ¹H axis: 768 acquired points, FT size 2048, 546 points kept, APOD 204
    # (the rounded 204.75); its dwell is that of the full spectral width.
    dic, data = ng.pipe.read(str(CEST))
    proton = read_spectral_parameters(dic, data)[-1]
    full_width = dic["FDF2SW"] * 2048 / 546
    assert proton.aq_time == pytest.approx(767 / full_width, rel=1e-6)
    nitrogen = read_spectral_parameters(dic, data)[1]  # half-dwell, P1 180
    assert nitrogen.aq_time == pytest.approx(91.5 / dic["FDF3SW"], rel=1e-6)


def test_a_header_with_no_signal_under_the_window_is_refused():
    dic, data = ng.pipe.read(str(FIXTURES / "sp2_t0.ft"))
    with pytest.raises(ValueError, match="no acquired signal under the window"):
        read_spectral_parameters({**dic, "FDF2P1": -360.0 * 40}, data)
    assert math.isfinite(_axis("sp2_t0").aq_time)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"FDDMXFLAG": -1.0}, "FDDMXFLAG -1 not recognised"),
        ({"FDDMXVAL": float("nan")}, "unreadable digital-filter correction"),
    ],
)
def test_ambiguous_digital_filter_metadata_is_refused(changes, message):
    dic, data = ng.pipe.read(str(FIXTURES / "sp2_digital_filter.ft"))
    with pytest.raises(ValueError, match=message):
        read_spectral_parameters({**dic, **changes}, data)


def test_a_header_without_the_digital_filter_state_is_refused():
    dic, data = ng.pipe.read(str(FIXTURES / "sp2_t0.ft"))
    header = {k: v for k, v in dic.items() if k != "FDDMXFLAG"}
    with pytest.raises(ValueError, match="not recorded"):
        read_spectral_parameters(header, data)
