"""Shared helpers for doublet lineshapes built from singlet kernels."""

from typing import TYPE_CHECKING, Any

from peakfit.engine.lineshapes.utils import doublet_offsets

if TYPE_CHECKING:
    from collections.abc import Callable


def doublet_kernel(
    x,
    cs,
    j,
    grid,
    *,
    kernel: Callable[..., Any],
    kernel_args: tuple[Any, ...] = (),
):
    """Combine +/- J/2 components for a singlet kernel."""
    dw_p, sign_p, dw_m, sign_m = doublet_offsets(x, cs, j, grid)
    return sign_p * kernel(dw_p, *kernel_args) + sign_m * kernel(dw_m, *kernel_args)


def doublet_kernel_with_derivs(
    x,
    cs,
    j,
    grid,
    *,
    kernel_with_derivs: Callable[..., tuple[Any, dict[str, Any]]],
    kernel_args: tuple[Any, ...] = (),
) -> tuple[Any, dict[str, Any]]:
    """Combine +/- J/2 components with derivative aggregation."""
    dw_p, sign_p, dw_m, sign_m = doublet_offsets(x, cs, j, grid)
    v_p, d_p = kernel_with_derivs(dw_p, *kernel_args)
    v_m, d_m = kernel_with_derivs(dw_m, *kernel_args)
    values = sign_p * v_p + sign_m * v_m
    derivs: dict[str, Any] = {}
    for key in d_p.keys() & d_m.keys():
        derivs[key] = sign_p * d_p[key] + sign_m * d_m[key]
    return values, derivs


__all__ = ["doublet_kernel", "doublet_kernel_with_derivs"]
