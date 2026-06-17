from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.param_id import ParameterId
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.types import ParamSpec
from peakfit.fit.auto_pick import (
    _PeakNameCounter,
    _update_peak_positions,
)
from peakfit.fit.auto_pick_candidates import (
    extract_roi_indices as _extract_roi_indices,
)
from peakfit.fit.auto_pick_candidates import (
    find_global_seed as _find_global_seed,
)
from peakfit.fit.auto_pick_candidates import (
    initial_local_maxima_candidates as _initial_local_maxima_candidates,
)
from peakfit.fit.auto_pick_candidates import (
    select_manual_candidate as _select_manual_candidate,
)
from peakfit.fit.auto_pick_candidates import (
    select_next_candidate as _select_next_candidate,
)
from peakfit.fit.auto_pick_candidates import (
    select_seed_candidate as _select_seed_candidate,
)
from peakfit.fit.auto_pick_decision import (
    accept_trial as _accept_trial,
)
from peakfit.fit.auto_pick_decision import (
    addition_threshold as _addition_threshold,
)
from peakfit.fit.auto_pick_decision import (
    calculate_dof_scale_from_header as _calculate_dof_scale_from_header,
)
from peakfit.fit.auto_pick_parameters import (
    any_cs_close_to_constraint as _any_cs_close_to_constraint,
)
from peakfit.fit.auto_pick_parameters import (
    build_shared_param_aliases as _build_shared_param_aliases,
)
from peakfit.fit.auto_pick_parameters import (
    initialize_existing_params_from_previous as _initialize_existing_params_from_previous,
)
from peakfit.fit.auto_pick_parameters import (
    initialize_new_peak_from_median as _initialize_new_peak_from_median,
)
from peakfit.fit.auto_pick_state import TrialState as _TrialState


def test_select_next_candidate_respects_eligible_mask() -> None:
    residual = np.array([[10.0], [9.0], [8.0]], dtype=np.float64)
    roi_points = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64)
    used_points: list[tuple[int, ...]] = []
    eligible_mask = np.array([False, True, True], dtype=bool)

    result = _select_next_candidate(
        residual=residual,
        roi_points=roi_points,
        used_points=used_points,
        min_separation_pts=0,
        threshold=0.1,
        eligible_mask=eligible_mask,
    )

    assert result is not None
    idx, score = result
    assert idx == 1
    assert score == 9.0


def test_select_seed_candidate_uses_seed_point_as_first_trial() -> None:
    residual = np.array([[3.0], [10.0], [8.0]], dtype=np.float64)
    roi_points = np.array([[10, 20], [11, 21], [12, 22]], dtype=np.int64)

    result = _select_seed_candidate(
        residual=residual,
        roi_points=roi_points,
        seed_point=(10, 20),
        threshold=1.0,
    )

    assert result is not None
    idx, score = result
    assert idx == 0
    assert score == 3.0


def test_peak_name_counter_allocates_and_rolls_back() -> None:
    counter = _PeakNameCounter(value=4)

    assert counter.peek() == "ap4"
    assert counter.consume() == "ap4"
    assert counter.value == 5

    counter.rollback()
    assert counter.value == 4


def test_extract_roi_indices_does_not_wrap_edges() -> None:
    data = np.zeros((1, 3, 5), dtype=np.float64)
    data[0, 1, 0] = 10.0
    data[0, 1, 4] = 10.0

    roi = _extract_roi_indices(
        data=data,
        contour_level=1.0,
        seed_point=(1, 0),
    )
    y_idx, x_idx = roi
    points = set(zip(y_idx.tolist(), x_idx.tolist(), strict=True))

    assert (1, 0) in points
    assert (1, 4) not in points


def test_initial_local_maxima_candidates_finds_all_roi_local_maxima() -> None:
    working_data = np.zeros((1, 5, 5), dtype=np.float64)
    working_data[0, 1, 1] = 10.0
    working_data[0, 3, 3] = 8.0
    working_data[0, 2, 2] = 1.0

    y_idx = np.array([1, 3, 2], dtype=np.int64)
    x_idx = np.array([1, 3, 2], dtype=np.int64)
    roi_indices = [y_idx, x_idx]
    roi_points = np.column_stack(roi_indices)

    candidates = _initial_local_maxima_candidates(
        working_data=working_data,
        roi_indices=roi_indices,
        roi_points=roi_points,
        threshold=0.5,
    )

    assert len(candidates) == 2
    assert candidates[0][1] == 10.0
    assert candidates[1][1] == 8.0


def test_build_shared_param_aliases_only_for_lw_and_j() -> None:
    params = Parameters()
    p1_lw = ParameterId(peak_name="p1", axis="F2", label="lw")
    p2_lw = ParameterId(peak_name="p2", axis="F2", label="lw")
    p1_j = ParameterId(peak_name="p1", axis="F2", label="j")
    p2_j = ParameterId(peak_name="p2", axis="F2", label="j")
    p1_cs = ParameterId(peak_name="p1", axis="F2", label="cs")
    p2_cs = ParameterId(peak_name="p2", axis="F2", label="cs")

    params.add(p1_lw, value=20.0)
    params.add(p2_lw, value=25.0)
    params.add(p1_j, value=5.0)
    params.add(p2_j, value=6.0)
    params.add(p1_cs, value=8.2)
    params.add(p2_cs, value=8.0)

    aliases = _build_shared_param_aliases(params)

    assert aliases[p2_lw.name] == p1_lw.name
    assert aliases[p2_j.name] == p1_j.name
    assert p2_cs.name not in aliases


def test_any_cs_close_to_constraint_uses_nucleus_specific_margins() -> None:
    config = PeakFitConfig()
    params = Parameters()
    h_cs = ParameterId(peak_name="p1", axis="F3", label="cs")
    n_cs = ParameterId(peak_name="p1", axis="F2", label="cs")

    params.add(h_cs, value=8.0005, min=8.0, max=8.3)
    params.add(n_cs, value=120.015, min=120.0, max=121.0)

    spectral_params = [
        SimpleNamespace(label="F2", nucleus="15N"),
        SimpleNamespace(label="F3", nucleus="1H"),
    ]
    spectra = cast("Any", SimpleNamespace(spectral_params=spectral_params))

    assert _any_cs_close_to_constraint(params, spectra, config) is True


def test_initialize_new_peak_from_median_uses_previous_fits() -> None:
    previous_params = Parameters()
    previous_params.add(ParameterId(peak_name="p1", axis="F2", label="lw"), value=10.0)
    previous_params.add(ParameterId(peak_name="p2", axis="F2", label="lw"), value=14.0)
    previous_params.add(ParameterId(peak_name="p1", axis="F2", label="cs"), value=8.12)
    previous_params.add(ParameterId(peak_name="p2", axis="F2", label="cs"), value=8.34)

    params = Parameters()
    params.add(ParameterId(peak_name="p3", axis="F2", label="lw"), value=19.0, min=5.0, max=20.0)
    params.add(
        ParameterId(peak_name="p3", axis="F2", label="cs"),
        value=7.85,
        min=7.0,
        max=9.0,
    )

    _initialize_new_peak_from_median(params, previous_params, "p3")

    assert params["p3.F2.lw"].value == 12.0
    assert params["p3.F2.cs"].value == 7.85


def test_initialize_existing_params_from_previous_preserves_new_peak() -> None:
    previous_params = Parameters()
    previous_params.add(ParameterId(peak_name="p1", axis="F2", label="cs"), value=8.12)
    previous_params.add(ParameterId(peak_name="p1", axis="F2", label="lw"), value=18.0)

    params = Parameters()
    params.add(
        ParameterId(peak_name="p1", axis="F2", label="cs"),
        value=7.90,
        min=7.0,
        max=9.0,
    )
    params.add(
        ParameterId(peak_name="p1", axis="F2", label="lw"),
        value=10.0,
        min=5.0,
        max=25.0,
    )
    params.add(
        ParameterId(peak_name="p2", axis="F2", label="lw"),
        value=11.0,
        min=5.0,
        max=25.0,
    )

    _initialize_existing_params_from_previous(params, previous_params, new_peak_name="p2")

    assert params["p1.F2.cs"].value == 8.12
    assert params["p1.F2.lw"].value == 18.0
    assert params["p2.F2.lw"].value == 11.0


def test_find_global_seed_skips_processed_roi_points() -> None:
    data = np.zeros((2, 3, 3), dtype=np.float64)
    data[:, 1, 1] = 10.0
    data[:, 2, 2] = 7.0

    blocked = np.zeros((3, 3), dtype=bool)
    blocked[1, 1] = True
    point, height = _find_global_seed(data, blocked_mask=blocked)
    assert point == (2, 2)
    assert height == 7.0

    blocked[:, :] = True
    point2, height2 = _find_global_seed(data, blocked_mask=blocked)
    assert point2 is None
    assert height2 == 0.0


def test_calculate_dof_scale_from_header_uses_td_size() -> None:
    spectral_params = [
        SimpleNamespace(ft=True, size=4096, td_size=1024),
        SimpleNamespace(ft=True, size=256, td_size=64),
    ]
    spectra = cast("Any", SimpleNamespace(spectral_params=spectral_params))

    scale = _calculate_dof_scale_from_header(spectra)
    assert scale == 0.0625


def test_accept_trial_scales_dof_with_zero_filling() -> None:
    config = PeakFitConfig()
    params = Parameters()
    footprint = np.ones(10, dtype=bool)

    previous = _TrialState(
        peaks=[],
        data=np.zeros((10, 2), dtype=np.float64),
        model=np.zeros((10, 2), dtype=np.float64),
        residual=np.ones((10, 2), dtype=np.float64),
        footprint=footprint,
        n_params=2,
        dof_scale=0.5,
        params=params,
    )
    new = _TrialState(
        peaks=[],
        data=np.zeros((10, 2), dtype=np.float64),
        model=np.zeros((10, 2), dtype=np.float64),
        residual=np.full((10, 2), 0.5, dtype=np.float64),
        footprint=footprint,
        n_params=3,
        dof_scale=0.5,
        params=params,
    )

    decision = _accept_trial(previous, new, noise=1.0, config=config)
    assert decision.df1 == 1
    assert decision.df2 == 7


def test_addition_threshold_uses_auto_peak_sigma_multiplier() -> None:
    config = PeakFitConfig()
    config.auto_peak.add_threshold_sigma = 3.5

    threshold = _addition_threshold(config, noise=2.0)
    assert threshold == 10.0


def test_auto_peak_default_has_no_per_roi_peak_cap() -> None:
    config = PeakFitConfig()
    assert config.auto_peak.max_peaks_per_roi is None


def test_auto_peak_accepts_optional_per_roi_peak_cap() -> None:
    config = PeakFitConfig.model_validate({"auto_peak": {"max_peaks_per_roi": 8}})
    assert config.auto_peak.max_peaks_per_roi == 8


def test_select_manual_candidate_uses_clicked_position() -> None:
    residual = np.array([[10.0], [9.0], [8.0]], dtype=np.float64)
    roi_points = np.array([[10, 20], [11, 21], [12, 22]], dtype=np.int64)
    spectra = cast(
        "Any",
        SimpleNamespace(
            spectral_params=[
                SimpleNamespace(pts2ppm=lambda pts: np.asarray(pts, dtype=np.float64)),
                SimpleNamespace(pts2ppm=lambda pts: np.asarray(pts, dtype=np.float64)),
            ]
        ),
    )

    result = _select_manual_candidate(
        residual=residual,
        roi_points=roi_points,
        spectra=spectra,
        target_ppm=(11.0, 21.0),
        used_points=[],
        min_separation_pts=0,
        threshold=0.1,
    )

    assert result is not None
    idx, _score = result
    assert idx == 1


def test_update_peak_positions_uses_fitted_cs_values() -> None:
    shape = SimpleNamespace(
        axis="F2",
        center=8.0,
        get_parameter_spec=lambda: [ParamSpec("cs", 8.0, 7.0, 9.0, "ppm")],
    )
    peak = Peak(name="p1", positions=np.array([8.0], dtype=np.float64), shapes=[cast("Any", shape)])
    params = Parameters()
    params.add(ParameterId(peak_name="p1", axis="F2", label="cs"), value=8.25)

    _update_peak_positions([peak], params)

    assert peak.positions[0] == 8.25
    assert peak.shapes[0].center == 8.25
