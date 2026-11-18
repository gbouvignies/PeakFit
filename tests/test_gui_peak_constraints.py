"""How a peak's own constraints combine with the Model settings, and what they reach.

Each case states the effective start, bounds and vary the engine fits with: the
desktop prepares them through ``peakfit fit``'s own resolver, from the configuration
of the equivalent TOML (``engine_configuration``).
"""

from dataclasses import replace

import pytest
from builders import overlapping_gaussians, two_cluster_dataset

from peakfit.engine.domain.config import AxisModel, Doublet, FitConfig, PeakFitConfig
from peakfit.gui.fitting import prepare_dataset_fit
from peakfit.gui.peaks import ParameterLimit


def _regions(prepared) -> list[set[tuple[int, ...]]]:
    return [set(zip(*(i.tolist() for i in c.grid_indices), strict=True)) for c in prepared.clusters]


def test_a_peaks_own_j_bound_sets_its_clustering_footprint_as_in_peakfit_fit():
    # ``peakfit fit`` marks a doublet's footprint as cs ± J_max/2, J_max after every
    # constraint, a peak's own included; the desktop clusters with the same bound.
    spectra, peaks, config = overlapping_gaussians()
    config.fitting = FitConfig(
        family="gaussian", axes={"F3": AxisModel(doublet=Doublet(j=6.0, min=0.5, max=12.0))}
    )
    config.clustering.contour_level = 0.01
    wide = (("F3.j", ParameterLimit(upper=200.0)),)
    own = tuple(replace(p, limits=wide) for p in peaks)
    rule = config.model_copy(deep=True)
    rule.parameters = rule.parameters.model_validate({"defaults": {"*.F3.j": {"max": 200.0}}})
    narrow = _regions(prepare_dataset_fit(spectra, peaks, config))
    by_rule = _regions(prepare_dataset_fit(spectra, peaks, rule))
    by_own = _regions(prepare_dataset_fit(spectra, own, config))
    assert by_rule != narrow  # the bound reaches the footprint
    assert by_own == by_rule


@pytest.fixture(scope="module")
def dataset():
    spectra, peaks, config, _prepared = two_cluster_dataset()
    return spectra, peaks, config


def _effective(dataset, settings: dict, limits, key: str) -> tuple[float, float, float, bool]:
    """Peak 0's ``key`` as the engine starts it: (start, lower, upper, vary)."""
    spectra, peaks, config = dataset
    config = PeakFitConfig.model_validate(config.model_dump(exclude_defaults=True) | settings)
    own = (replace(peaks[0], limits=limits), *peaks[1:])
    prepared = prepare_dataset_fit(spectra, own, config)
    assert not prepared.problems
    p = prepared.parameters[f"p_0.{key}"]
    return p.value, p.min, p.max, p.vary


LW = "F2.lw"
# The model's own linewidth start and range: 25 Hz in 0.1–200 Hz, varied. The cases
# under RULE are the table in docs/constraints-and-fit-steps.md ("Combined example").
RULE = {"parameters": {"position_window": 0.04, "defaults": {"*.*.lw": {"min": 5, "max": 100}}}}


def _rule(**constraint) -> dict:
    return {"parameters": {"position_window": 0.04, "defaults": {"*.*.lw": constraint}}}


@pytest.mark.parametrize(
    ("settings", "limit", "expected"),
    [
        pytest.param({}, None, (25.0, 0.1, 200.0, True), id="model-alone"),
        pytest.param(RULE, None, (25.0, 5.0, 100.0, True), id="rule-bounds"),
        # A peak's own bound replaces the rule's on its side only.
        pytest.param(RULE, ParameterLimit(lower=20.0), (25.0, 20.0, 100.0, True), id="own-lower"),
        pytest.param(
            RULE,
            ParameterLimit(lower=10.0, upper=50.0),
            (25.0, 10.0, 50.0, True),
            id="own-range",
        ),
        pytest.param(RULE, ParameterLimit(start=40.0), (40.0, 5.0, 100.0, True), id="own-start"),
        # A start nobody typed follows the range: the default (or a rule's start) is
        # moved to the nearest bound, here by a rule's or the peak's own bound.
        pytest.param(_rule(min=30, max=100), None, (30.0, 30.0, 100.0, True), id="rule-moves"),
        pytest.param(RULE, ParameterLimit(lower=30.0), (30.0, 30.0, 100.0, True), id="own-moves"),
        pytest.param(
            _rule(value=50, min=5, max=100),
            ParameterLimit(upper=30.0),
            (30.0, 5.0, 30.0, True),
            id="own-bound-moves-rule-start",
        ),
        # Hold: the peak's own vary wins over the rule's, both ways.
        pytest.param({}, ParameterLimit(vary=False), (25.0, 0.1, 200.0, False), id="hold"),
        pytest.param(RULE, ParameterLimit(vary=False), (25.0, 5.0, 100.0, False), id="hold-rule"),
        pytest.param(
            {}, ParameterLimit(start=28.0, vary=False), (28.0, 0.1, 200.0, False), id="hold-at"
        ),
        pytest.param(_rule(vary=False), None, (25.0, 0.1, 200.0, False), id="rule-holds"),
        pytest.param(
            _rule(vary=False), ParameterLimit(vary=True), (25.0, 0.1, 200.0, True), id="own-frees"
        ),
    ],
)
def test_a_parameters_effective_start_bounds_and_vary(dataset, settings, limit, expected):
    limits = ((LW, limit),) if limit is not None else ()
    assert _effective(dataset, settings, limits, LW) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("settings", "limit", "window", "vary"),
    [
        pytest.param({}, None, 0.04, True, id="model-wide"),
        pytest.param(
            {"parameters": {"position_window": 0.04, "position_windows": {"F2": 0.02}}},
            None,
            0.02,
            True,
            id="axis-wide",
        ),
        pytest.param(
            {"parameters": {"position_window": 0.04, "position_windows": {"F2": 0.02}}},
            ParameterLimit(window=0.01),
            0.01,
            True,
            id="own-window",
        ),
        pytest.param({}, ParameterLimit(window=0.01, vary=False), 0.01, False, id="held-window"),
    ],
)
def test_a_positions_window_is_the_narrowest_scope_set(dataset, settings, limit, window, vary):
    # The peak's own window, then the axis's, then the Model-wide one; always centred
    # on the working position, which stays the start.
    _spectra, peaks, _config = dataset
    position = peaks[0].positions[0]
    limits = (("F2.cs", limit),) if limit is not None else ()
    start, low, high, varied = _effective(dataset, settings, limits, "F2.cs")
    assert (start, low, high, varied) == pytest.approx(
        (position, position - window, position + window, vary)
    )


def test_a_fit_steps_rule_wins_over_a_peaks_hold_as_in_peakfit_fit():
    # An imported schedule's fix/vary applies after every constraint (the engine's
    # ``prepare_cluster_parameters``): its first step fixes every linewidth, so the
    # peak's own vary=True does not free it, and a later "vary" frees a held one.
    spectra, peaks, config = overlapping_gaussians()
    freed = (replace(peaks[0], limits=((LW, ParameterLimit(vary=True)),)), *peaks[1:])
    prepared = prepare_dataset_fit(spectra, freed, config)
    assert not prepared.parameters[f"p_0.{LW}"].vary
    held = (replace(peaks[0], limits=((LW, ParameterLimit(vary=False)),)), *peaks[1:])
    config.fitting.steps[0].fix, config.fitting.steps[0].vary = [], ["*.*.lw"]
    assert prepare_dataset_fit(spectra, held, config).parameters[f"p_0.{LW}"].vary


def test_an_own_bound_beyond_the_others_side_leaves_an_empty_range_that_blocks_the_fit(dataset):
    spectra, peaks, config = dataset
    config = PeakFitConfig.model_validate(config.model_dump(exclude_defaults=True) | RULE)
    own = (replace(peaks[0], limits=((LW, ParameterLimit(lower=150.0)),)), *peaks[1:])
    [problem] = prepare_dataset_fit(spectra, own, config).problems
    assert problem.reason == "The range 150–100 Hz is empty. Widen it."
