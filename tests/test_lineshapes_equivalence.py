"""Equivalence tests for refactored sinebell lineshapes."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.testing as npt

from peakfit.engine.domain.config import FitConfig
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.lineshapes.gaussian import model as gaussian_model
from peakfit.engine.lineshapes.gaussian.kernel import kernel as gaussian_kernel
from peakfit.engine.lineshapes.gaussian.kernel import (
    kernel_with_derivs as gaussian_kernel_with_derivs,
)
from peakfit.engine.lineshapes.grid import SpectralGrid
from peakfit.engine.lineshapes.lorentzian import model as lorentzian_model
from peakfit.engine.lineshapes.lorentzian.kernel import kernel as lorentzian_kernel
from peakfit.engine.lineshapes.lorentzian.kernel import (
    kernel_with_derivs as lorentzian_kernel_with_derivs,
)
from peakfit.engine.lineshapes.no_apod import model as no_apod_model
from peakfit.engine.lineshapes.no_apod.kernel import kernel as no_apod_kernel
from peakfit.engine.lineshapes.no_apod.kernel import (
    kernel_with_derivs as no_apod_kernel_with_derivs,
)
from peakfit.engine.lineshapes.no_apod.model import make_state as no_apod_make_state
from peakfit.engine.lineshapes.pvoigt import model as pvoigt_model
from peakfit.engine.lineshapes.pvoigt.kernel import kernel as pvoigt_kernel
from peakfit.engine.lineshapes.pvoigt.kernel import (
    kernel_with_derivs as pvoigt_kernel_with_derivs,
)
from peakfit.engine.lineshapes.sp1 import model as sp1_model
from peakfit.engine.lineshapes.sp1.kernel import kernel as sp1_kernel
from peakfit.engine.lineshapes.sp1.kernel import (
    kernel_with_derivs as sp1_kernel_with_derivs,
)
from peakfit.engine.lineshapes.sp1.kernel import make_state as sp1_make_state
from peakfit.engine.lineshapes.sp2 import model as sp2_model
from peakfit.engine.lineshapes.sp2.kernel import kernel as sp2_kernel
from peakfit.engine.lineshapes.sp2.kernel import (
    kernel_with_derivs as sp2_kernel_with_derivs,
)
from peakfit.engine.lineshapes.sp2.kernel import make_state as sp2_make_state
from peakfit.engine.lineshapes.utils import (
    LineshapeContext,
    apply_phase,
    apply_phase_with_derivs,
    doublet_offsets,
    get_apodization_state,
    require_grid,
)
from peakfit.engine.types import ClusterParameters

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_spectra(size: int = 32) -> Spectra:
    data = np.zeros((1, size), dtype=np.float64)
    params = [
        SpectralParameters(
            size=1,
            sw=1.0,
            obs=1.0,
            car=0.0,
            aq_time=0.1,
            apocode=1.0,
            apodq1=0.0,
            apodq2=0.0,
            apodq3=1.0,
            p180=False,
            direct=False,
            ft=True,
            label="F1",
        ),
        SpectralParameters(
            size=size,
            sw=1200.0,
            obs=600.0,
            car=4.7,
            aq_time=0.25,
            apocode=1.0,
            apodq1=0.2,
            apodq2=0.35,
            apodq3=1.0,
            p180=False,
            direct=True,
            ft=True,
            label="F2",
        ),
    ]
    return Spectra(
        dic={},
        data=data,
        z_values=np.array([], dtype=np.float64),
        params=params,
    )


def _make_grid(size: int = 32) -> SpectralGrid:
    spectra = _make_spectra(size)
    return SpectralGrid(spectra, 1)


def _make_params():
    cs = np.array([7.1, 7.35], dtype=np.float64)
    lw = np.array([12.0, 24.5], dtype=np.float64)
    phase = np.array([0.25, -0.5], dtype=np.float64)
    j = np.array([4.2, 7.6], dtype=np.float64)
    eta = np.array([0.3, 0.7], dtype=np.float64)
    return cs, lw, j, phase, eta


def _reference_singlet_function(
    x: np.ndarray,
    cs: np.ndarray,
    lw: np.ndarray,
    phase: np.ndarray,
    *,
    context: LineshapeContext,
    kernel: Callable,
    make_state: Callable,
    state_key: str,
    shape_label: str,
) -> np.ndarray:
    state = get_apodization_state(
        context,
        state_key=state_key,
        shape=shape_label,
        make_state=make_state,
    )
    grid = require_grid(context, shape=f"{shape_label} singlet")
    dw_hz, sign = grid.compute_offsets(x, cs)
    z_values = kernel(dw_hz, lw, state)
    values = apply_phase(z_values, phase)
    return sign * values


def _reference_doublet_function(
    x: np.ndarray,
    cs: np.ndarray,
    lw: np.ndarray,
    j: np.ndarray,
    phase: np.ndarray,
    *,
    context: LineshapeContext,
    kernel: Callable,
    make_state: Callable,
    state_key: str,
    shape_label: str,
) -> np.ndarray:
    state = get_apodization_state(
        context,
        state_key=state_key,
        shape=shape_label,
        make_state=make_state,
    )
    grid = require_grid(context, shape=f"{shape_label} doublet")
    dw_p, sign_p, dw_m, sign_m = doublet_offsets(x, cs, j, grid)
    z_values = sign_p * kernel(dw_p, lw, state) + sign_m * kernel(dw_m, lw, state)
    return apply_phase(z_values, phase)


def _reference_singlet_cluster(
    x: np.ndarray,
    cs: np.ndarray,
    lw: np.ndarray,
    phase: np.ndarray,
    *,
    grid: SpectralGrid,
    kernel_with_derivs: Callable,
    kernel_extra_args: tuple[object, ...],
    aq: float,
) -> dict[str, np.ndarray]:
    dw_hz, sign = grid.compute_offsets(x, cs)
    z_values, z_derivs = kernel_with_derivs(dw_hz, lw, *kernel_extra_args)

    values, derivs = apply_phase_with_derivs(z_values, z_derivs, phase, aq)
    values = sign * values

    derivs["cs"] = sign * derivs["dw"] * (-grid.spec_params.ppm2hz(1.0))
    derivs["lw"] = sign * derivs["lw"]
    derivs["phase"] = sign * derivs["phase"]

    return {"values": values, **derivs}


def _reference_doublet_cluster(
    x: np.ndarray,
    cs: np.ndarray,
    lw: np.ndarray,
    j: np.ndarray,
    phase: np.ndarray,
    *,
    grid: SpectralGrid,
    kernel_with_derivs: Callable,
    kernel_extra_args: tuple[object, ...],
    aq: float,
) -> dict[str, np.ndarray]:
    dw_p, sign_p, dw_m, sign_m = doublet_offsets(x, cs, j, grid)
    v_p, d_p = kernel_with_derivs(dw_p, lw, *kernel_extra_args)
    v_m, d_m = kernel_with_derivs(dw_m, lw, *kernel_extra_args)
    z_values = sign_p * v_p + sign_m * v_m
    z_derivs = {
        "lw": sign_p * d_p["lw"] + sign_m * d_m["lw"],
        "dw": sign_p * d_p["dw"] + sign_m * d_m["dw"],
    }

    values, derivs = apply_phase_with_derivs(z_values, z_derivs, phase, aq)
    derivs["cs"] = derivs["dw"] * (-grid.spec_params.ppm2hz(1.0))
    derivs["lw"] = derivs["lw"]

    return {"values": values, **derivs}


def _reference_real_doublet_values(
    x: np.ndarray,
    cs: np.ndarray,
    j: np.ndarray,
    *,
    grid: SpectralGrid,
    kernel: Callable,
    kernel_args: tuple[object, ...],
) -> np.ndarray:
    dw_p, sign_p, dw_m, sign_m = doublet_offsets(x, cs, j, grid)
    return sign_p * kernel(dw_p, *kernel_args) + sign_m * kernel(dw_m, *kernel_args)


def _reference_real_doublet_with_derivs(
    x: np.ndarray,
    cs: np.ndarray,
    j: np.ndarray,
    *,
    grid: SpectralGrid,
    kernel_with_derivs: Callable,
    kernel_args: tuple[object, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dw_p, sign_p, dw_m, sign_m = doublet_offsets(x, cs, j, grid)
    v_p, d_p = kernel_with_derivs(dw_p, *kernel_args)
    v_m, d_m = kernel_with_derivs(dw_m, *kernel_args)
    values = sign_p * v_p + sign_m * v_m
    derivs: dict[str, np.ndarray] = {}
    for key in d_p.keys() & d_m.keys():
        derivs[key] = sign_p * d_p[key] + sign_m * d_m[key]
    return values, derivs


def _assert_values_close(actual: np.ndarray, expected: np.ndarray) -> None:
    npt.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def _assert_derivs_close(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> None:
    assert set(actual) == set(expected)
    for key, value in expected.items():
        npt.assert_allclose(actual[key], value, rtol=1e-12, atol=1e-12)


def test_sp1_sp2_singlet_function_equivalence() -> None:
    grid = _make_grid()
    context = LineshapeContext(grid=grid)
    x = np.arange(grid.spec_params.size, dtype=np.float64)
    cs, lw, _, phase, _ = _make_params()

    expected_sp1 = _reference_singlet_function(
        x,
        cs,
        lw,
        phase,
        context=context,
        kernel=sp1_kernel,
        make_state=sp1_make_state,
        state_key="sp1_state",
        shape_label="SP1",
    )
    actual_sp1 = sp1_model.function(x, cs, lw, phase, context=context)
    _assert_values_close(actual_sp1, expected_sp1)

    expected_sp2 = _reference_singlet_function(
        x,
        cs,
        lw,
        phase,
        context=context,
        kernel=sp2_kernel,
        make_state=sp2_make_state,
        state_key="sp2_state",
        shape_label="SP2",
    )
    actual_sp2 = sp2_model.function(x, cs, lw, phase, context=context)
    _assert_values_close(actual_sp2, expected_sp2)


def test_sp1_sp2_doublet_function_equivalence() -> None:
    grid = _make_grid()
    context = LineshapeContext(grid=grid)
    x = np.arange(grid.spec_params.size, dtype=np.float64)
    cs, lw, j, phase, _ = _make_params()

    expected_sp1 = _reference_doublet_function(
        x,
        cs,
        lw,
        j,
        phase,
        context=context,
        kernel=sp1_kernel,
        make_state=sp1_make_state,
        state_key="sp1_state",
        shape_label="SP1",
    )
    actual_sp1 = sp1_model.function_doublet(x, cs, lw, j, phase, context=context)
    _assert_values_close(actual_sp1, expected_sp1)

    expected_sp2 = _reference_doublet_function(
        x,
        cs,
        lw,
        j,
        phase,
        context=context,
        kernel=sp2_kernel,
        make_state=sp2_make_state,
        state_key="sp2_state",
        shape_label="SP2",
    )
    actual_sp2 = sp2_model.function_doublet(x, cs, lw, j, phase, context=context)
    _assert_values_close(actual_sp2, expected_sp2)


def test_sp1_sp2_singlet_derivatives_equivalence() -> None:
    spectra = _make_spectra()
    grid = SpectralGrid(spectra, 1)
    x = np.arange(grid.spec_params.size, dtype=np.float64)
    cs, lw, _, phase, _ = _make_params()

    config = FitConfig(fit_phase=[grid.axis_label])
    sp1_shape = sp1_model.SP1("P1", cs[0], spectra, 1, config)
    sp2_shape = sp2_model.SP2("P2", cs[0], spectra, 1, config)

    cluster_params = ClusterParameters(extras={"cs": cs, "lw": lw, "phase": phase})

    sp1_result = sp1_shape.evaluate_cluster(x, cluster_params, compute_derivs=True)
    sp2_result = sp2_shape.evaluate_cluster(x, cluster_params, compute_derivs=True)

    state_sp1 = sp1_make_state(
        grid.spec_params.aq_time, grid.spec_params.apodq1, grid.spec_params.apodq2
    )
    state_sp2 = sp2_make_state(
        grid.spec_params.aq_time, grid.spec_params.apodq1, grid.spec_params.apodq2
    )

    expected_sp1 = _reference_singlet_cluster(
        x,
        cs,
        lw,
        phase,
        grid=grid,
        kernel_with_derivs=sp1_kernel_with_derivs,
        kernel_extra_args=(state_sp1,),
        aq=grid.spec_params.aq_time,
    )
    expected_sp2 = _reference_singlet_cluster(
        x,
        cs,
        lw,
        phase,
        grid=grid,
        kernel_with_derivs=sp2_kernel_with_derivs,
        kernel_extra_args=(state_sp2,),
        aq=grid.spec_params.aq_time,
    )

    _assert_values_close(sp1_result.values, expected_sp1["values"])
    _assert_values_close(sp2_result.values, expected_sp2["values"])
    sp1_expected_derivs = {k: v for k, v in expected_sp1.items() if k != "values"}
    sp2_expected_derivs = {k: v for k, v in expected_sp2.items() if k != "values"}
    _assert_derivs_close(sp1_result.derivatives, sp1_expected_derivs)
    _assert_derivs_close(sp2_result.derivatives, sp2_expected_derivs)


def test_sp1_sp2_doublet_derivatives_equivalence() -> None:
    spectra = _make_spectra()
    grid = SpectralGrid(spectra, 1)
    x = np.arange(grid.spec_params.size, dtype=np.float64)
    cs, lw, j, phase, _ = _make_params()

    config = FitConfig(fit_phase=[grid.axis_label])
    sp1_shape = sp1_model.SP1Doublet("P1", cs[0], spectra, 1, config)
    sp2_shape = sp2_model.SP2Doublet("P2", cs[0], spectra, 1, config)

    cluster_params = ClusterParameters(extras={"cs": cs, "lw": lw, "j": j, "phase": phase})

    sp1_result = sp1_shape.evaluate_cluster(x, cluster_params, compute_derivs=True)
    sp2_result = sp2_shape.evaluate_cluster(x, cluster_params, compute_derivs=True)

    state_sp1 = sp1_make_state(
        grid.spec_params.aq_time, grid.spec_params.apodq1, grid.spec_params.apodq2
    )
    state_sp2 = sp2_make_state(
        grid.spec_params.aq_time, grid.spec_params.apodq1, grid.spec_params.apodq2
    )

    expected_sp1 = _reference_doublet_cluster(
        x,
        cs,
        lw,
        j,
        phase,
        grid=grid,
        kernel_with_derivs=sp1_kernel_with_derivs,
        kernel_extra_args=(state_sp1,),
        aq=grid.spec_params.aq_time,
    )
    expected_sp2 = _reference_doublet_cluster(
        x,
        cs,
        lw,
        j,
        phase,
        grid=grid,
        kernel_with_derivs=sp2_kernel_with_derivs,
        kernel_extra_args=(state_sp2,),
        aq=grid.spec_params.aq_time,
    )

    _assert_values_close(sp1_result.values, expected_sp1["values"])
    _assert_values_close(sp2_result.values, expected_sp2["values"])
    sp1_expected_derivs = {k: v for k, v in expected_sp1.items() if k != "values"}
    sp2_expected_derivs = {k: v for k, v in expected_sp2.items() if k != "values"}
    _assert_derivs_close(sp1_result.derivatives, sp1_expected_derivs)
    _assert_derivs_close(sp2_result.derivatives, sp2_expected_derivs)


def test_real_doublet_function_equivalence() -> None:
    grid = _make_grid()
    context = LineshapeContext(grid=grid)
    x = np.arange(grid.spec_params.size, dtype=np.float64)
    cs, lw, j, _, eta = _make_params()

    expected_gauss = _reference_real_doublet_values(
        x,
        cs,
        j,
        grid=grid,
        kernel=gaussian_kernel,
        kernel_args=(lw,),
    )
    actual_gauss = gaussian_model.function_doublet(x, cs, lw, j, context=context)
    _assert_values_close(actual_gauss, expected_gauss)

    expected_lorentz = _reference_real_doublet_values(
        x,
        cs,
        j,
        grid=grid,
        kernel=lorentzian_kernel,
        kernel_args=(lw[None, :],),
    )
    actual_lorentz = lorentzian_model.function_doublet(x, cs, lw, j, context=context)
    _assert_values_close(actual_lorentz, expected_lorentz)

    expected_pvoigt = _reference_real_doublet_values(
        x,
        cs,
        j,
        grid=grid,
        kernel=pvoigt_kernel,
        kernel_args=(lw[None, :], eta),
    )
    actual_pvoigt = pvoigt_model.function_doublet(x, cs, lw, eta, j, context=context)
    _assert_values_close(actual_pvoigt, expected_pvoigt)


def test_no_apod_doublet_function_equivalence() -> None:
    grid = _make_grid()
    context = LineshapeContext(grid=grid)
    x = np.arange(grid.spec_params.size, dtype=np.float64)
    cs, lw, j, phase, _ = _make_params()
    aq = grid.spec_params.aq_time
    state = no_apod_make_state(aq, 0.0, 0.0)

    z_values = _reference_real_doublet_values(
        x,
        cs,
        j,
        grid=grid,
        kernel=no_apod_kernel,
        kernel_args=(lw, state),
    )
    expected = apply_phase(z_values, phase)
    actual = no_apod_model.function_doublet(x, cs, lw, j, phase, context=context)
    _assert_values_close(actual, expected)


def test_real_doublet_derivatives_equivalence() -> None:
    spectra = _make_spectra()
    grid = SpectralGrid(spectra, 1)
    x = np.arange(grid.spec_params.size, dtype=np.float64)
    cs, lw, j, _, eta = _make_params()

    config = FitConfig()
    gaussian_shape = gaussian_model.GaussianDoublet("P1", cs[0], spectra, 1, config)
    lorentz_shape = lorentzian_model.LorentzianDoublet(
        "P2",
        cs[0],
        spectra,
        1,
        config,
    )
    pvoigt_shape = pvoigt_model.PseudoVoigtDoublet("P3", cs[0], spectra, 1, config)

    gaussian_params = ClusterParameters(extras={"cs": cs, "lw": lw, "j": j})
    lorentz_params = ClusterParameters(extras={"cs": cs, "lw": lw, "j": j})
    pvoigt_params = ClusterParameters(extras={"cs": cs, "lw": lw, "eta": eta, "j": j})

    gaussian_result = gaussian_shape.evaluate_cluster(
        x,
        gaussian_params,
        compute_derivs=True,
    )
    lorentz_result = lorentz_shape.evaluate_cluster(
        x,
        lorentz_params,
        compute_derivs=True,
    )
    pvoigt_result = pvoigt_shape.evaluate_cluster(x, pvoigt_params, compute_derivs=True)

    gauss_values, gauss_derivs = _reference_real_doublet_with_derivs(
        x,
        cs,
        j,
        grid=grid,
        kernel_with_derivs=gaussian_kernel_with_derivs,
        kernel_args=(lw,),
    )
    gauss_expected = {
        "cs": gauss_derivs["dw"] * (-grid.spec_params.ppm2hz(1.0)),
        "lw": gauss_derivs["lw"],
    }

    lorentz_values, lorentz_derivs = _reference_real_doublet_with_derivs(
        x,
        cs,
        j,
        grid=grid,
        kernel_with_derivs=lorentzian_kernel_with_derivs,
        kernel_args=(lw[None, :],),
    )
    lorentz_expected = {
        "cs": lorentz_derivs["dw"] * (-grid.spec_params.ppm2hz(1.0)),
        "lw": lorentz_derivs["lw"],
    }

    pvoigt_values, pvoigt_derivs = _reference_real_doublet_with_derivs(
        x,
        cs,
        j,
        grid=grid,
        kernel_with_derivs=pvoigt_kernel_with_derivs,
        kernel_args=(lw[None, :], eta),
    )
    pvoigt_expected = {
        "cs": pvoigt_derivs["dw"] * (-grid.spec_params.ppm2hz(1.0)),
        "lw": pvoigt_derivs["lw"],
        "eta": pvoigt_derivs["eta"],
    }

    _assert_values_close(gaussian_result.values, gauss_values)
    _assert_derivs_close(gaussian_result.derivatives, gauss_expected)
    _assert_values_close(lorentz_result.values, lorentz_values.real)
    lorentz_expected_real = {k: v.real for k, v in lorentz_expected.items()}
    _assert_derivs_close(lorentz_result.derivatives, lorentz_expected_real)
    _assert_values_close(pvoigt_result.values, pvoigt_values.real)
    pvoigt_expected_real = {k: v.real for k, v in pvoigt_expected.items()}
    _assert_derivs_close(pvoigt_result.derivatives, pvoigt_expected_real)


def test_no_apod_doublet_derivatives_equivalence() -> None:
    spectra = _make_spectra()
    grid = SpectralGrid(spectra, 1)
    x = np.arange(grid.spec_params.size, dtype=np.float64)
    cs, lw, j, phase, _ = _make_params()
    aq = grid.spec_params.aq_time
    state = no_apod_make_state(aq, 0.0, 0.0)

    config = FitConfig(fit_phase=[grid.axis_label])
    no_apod_shape = no_apod_model.NoApodDoublet("P1", cs[0], spectra, 1, config)
    cluster_params = ClusterParameters(extras={"cs": cs, "lw": lw, "j": j, "phase": phase})

    result = no_apod_shape.evaluate_cluster(x, cluster_params, compute_derivs=True)
    expected = _reference_doublet_cluster(
        x,
        cs,
        lw,
        j,
        phase,
        grid=grid,
        kernel_with_derivs=no_apod_kernel_with_derivs,
        kernel_extra_args=(state,),
        aq=aq,
    )

    _assert_values_close(result.values, expected["values"])
    expected_derivs = {k: v for k, v in expected.items() if k != "values"}
    _assert_derivs_close(result.derivatives, expected_derivs)
