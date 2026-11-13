"""CEST baseline planes as replicates: observed scatter against the published profile σ.

Validation T4 of ``docs/profile-uncertainty.md`` (§5.1).
"""

from pathlib import Path

import numpy as np
import pytest
from click.utils import strip_ansi
from typer.testing import CliRunner

from peakfit.cli.app import app
from peakfit.engine.scientific.replicates import (
    BaselineScatter,
    baseline_scatter,
    run_baseline_scatter,
)
from peakfit.io.scientific import read_scientific

DATA = Path(__file__).resolve().parents[1] / "examples" / "04-uncertainty-analysis" / "data"

REFERENCE = -12000.0


def _cest(rng, sigma, *, planes=2001, dip_at=300.0):
    """A sloping baseline with independent noise, a deep dip at the peak's own offset
    and a reference plane far off resonance, in shuffled acquisition order."""
    offsets = np.concatenate([[REFERENCE], np.linspace(-2000.0, 2000.0, planes)])
    baseline = 100.0 + 0.004 * offsets[1:] + 2e-6 * offsets[1:] ** 2
    dip = 95.0 * np.exp(-0.5 * ((offsets[1:] - dip_at) / 60.0) ** 2)
    amplitudes = np.concatenate([[1.0e4], baseline - dip]) + rng.normal(0, sigma, planes + 1)
    order = rng.permutation(planes + 1)
    return offsets[order], amplitudes[order]


def test_scatter_of_independent_noise_matches_its_sigma():
    rng = np.random.default_rng(166)
    offsets, amplitudes = _cest(rng, 2.0)
    result = baseline_scatter(
        offsets,
        amplitudes,
        np.full(offsets.size, 2.0),
        peak_offset=300.0,
        reference=offsets <= -1e4,
    )
    assert isinstance(result, BaselineScatter)
    assert abs(result.ratio - 1.0) < 0.06
    assert abs(result.level - 100.0) < 5.0
    # The dip (±400 Hz around the peak) and the reference plane are left out.
    assert result.planes == np.sum(np.abs(offsets - 300.0) > 400.0) - 1


def test_smooth_trends_do_not_count_as_scatter():
    offsets = np.linspace(-2000.0, 2000.0, 201)
    amplitudes = 100.0 + 0.004 * offsets + 2e-6 * offsets**2
    result = baseline_scatter(
        offsets, amplitudes, np.ones(201), peak_offset=0.0, reference=np.zeros(201, bool)
    )
    assert isinstance(result, BaselineScatter)
    assert result.observed < 1e-9


def test_stated_sigma_is_the_median_profile_sigma_of_the_baseline_planes():
    offsets = np.linspace(-2000.0, 2000.0, 101)
    errors = np.where(np.abs(offsets) > 400.0, 3.0, 50.0)
    errors[-3:] = 4.0
    result = baseline_scatter(
        offsets, np.full(101, 100.0), errors, peak_offset=0.0, reference=np.zeros(101, bool)
    )
    assert isinstance(result, BaselineScatter)
    assert result.stated == 3.0


def test_too_few_baseline_planes_say_why():
    offsets = np.linspace(-600.0, 600.0, 25)
    result = baseline_scatter(
        offsets, np.full(25, 100.0), np.ones(25), peak_offset=0.0, reference=np.zeros(25, bool)
    )
    assert result == "8 baseline planes, at least 20 needed"


def test_unavailable_profile_sigma_says_why():
    offsets = np.linspace(-2000.0, 2000.0, 101)
    result = baseline_scatter(
        offsets,
        np.full(101, 100.0),
        np.full(101, np.nan),
        peak_offset=0.0,
        reference=np.zeros(101, bool),
    )
    assert result == "profile σ unavailable"


@pytest.fixture(scope="module")
def example_run(tmp_path_factory):
    """The bundled CEST example, fitted once by the CLI."""
    output = tmp_path_factory.mktemp("example") / "Fits"
    result = CliRunner().invoke(
        app,
        [
            "fit",
            str(DATA / "pseudo3d.ft2"),
            str(DATA / "pseudo3d.list"),
            "--z-values",
            str(DATA / "b1_offsets.txt"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    (run,) = output.iterdir()
    return run


@pytest.mark.slow
def test_the_bundled_examples_profile_sigma_is_calibrated(example_run):
    outcome, _samples, context = read_scientific(example_run / "scientific.npz")
    results = run_baseline_scatter(outcome, context)
    ratios = [r.ratio for r in results.values() if isinstance(r, BaselineScatter)]
    assert len(ratios) >= 150
    assert 0.9 <= np.median(ratios) <= 1.15
    # Its one reference plane (−12 kHz) is the first in offset order.
    assert run_baseline_scatter(outcome, context, reference=[0]) == results


@pytest.mark.slow
def test_the_command_reports_the_calibration_of_a_published_run(example_run):
    result = CliRunner().invoke(app, ["replicates", str(example_run)])
    assert result.exit_code == 0, result.output
    text = " ".join(strip_ansi(result.output).split())
    assert "Observed / stated profile σ: median 1.07 (10–90%: 0.89–1.28), 164 peaks" in text
    assert "Skipped 2 peaks: baseline intensity not positive" in text


def test_the_command_says_when_a_run_has_no_archive(tmp_path):
    result = CliRunner().invoke(app, ["replicates", str(tmp_path)])
    assert result.exit_code == 1
    assert "No scientific.npz" in strip_ansi(result.output)
