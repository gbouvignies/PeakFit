"""The built-in lineshape models.

Each model pairs a kernel (in its own package) with a template family and its
parameter specs. Real shapes (Gaussian, Lorentzian, Pseudo-Voigt) start ``lw`` at
25 Hz; phased shapes (SP1, SP2, NoApod) start it at 20 Hz and add ``phase``.
Doublets add the coupling ``j``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from peakfit.engine.lineshapes.gaussian import kernel as gaussian
from peakfit.engine.lineshapes.lorentzian import kernel as lorentzian
from peakfit.engine.lineshapes.no_apod import kernel as no_apod
from peakfit.engine.lineshapes.pvoigt import kernel as pvoigt
from peakfit.engine.lineshapes.sp1 import kernel as sp1
from peakfit.engine.lineshapes.sp2 import kernel as sp2
from peakfit.engine.lineshapes.templates import (
    PhasedDoubletBase,
    PhasedSingletBase,
    SimpleDoubletBase,
    SimpleSingletBase,
    parameter_specs,
)

if TYPE_CHECKING:
    from peakfit.engine.types import ClusterParameters
    from peakfit.shared.typing import FloatArray


# =============================================================================
# Real-valued lineshapes
# =============================================================================


class Gaussian(SimpleSingletBase):
    """Gaussian singlet lineshape: G(Δω) = exp(-c × Δω²)."""

    shape_name: ClassVar[str] = "gaussian"
    param_specs = staticmethod(parameter_specs(25.0))
    kernel: ClassVar = staticmethod(gaussian.kernel)
    kernel_with_derivs: ClassVar = staticmethod(gaussian.kernel_with_derivs)


class GaussianDoublet(SimpleDoubletBase):
    """Gaussian doublet lineshape with J-coupling."""

    shape_name: ClassVar[str] = "gaussian_doublet"
    param_specs = staticmethod(parameter_specs(25.0, "j"))
    kernel: ClassVar = staticmethod(gaussian.kernel)
    kernel_with_derivs: ClassVar = staticmethod(gaussian.kernel_with_derivs)


class Lorentzian(SimpleSingletBase):
    """Lorentzian singlet lineshape: L(Δω) = 1 / (1 + (2Δω/R)²)."""

    shape_name: ClassVar[str] = "lorentzian"
    param_specs = staticmethod(parameter_specs(25.0))
    kernel: ClassVar = staticmethod(lorentzian.kernel)
    kernel_with_derivs: ClassVar = staticmethod(lorentzian.kernel_with_derivs)


class LorentzianDoublet(SimpleDoubletBase):
    """Lorentzian doublet lineshape with J-coupling."""

    shape_name: ClassVar[str] = "lorentzian_doublet"
    param_specs = staticmethod(parameter_specs(25.0, "j"))
    kernel: ClassVar = staticmethod(lorentzian.kernel)
    kernel_with_derivs: ClassVar = staticmethod(lorentzian.kernel_with_derivs)


class PseudoVoigt(SimpleSingletBase):
    """Pseudo-Voigt singlet: eta*Lorentzian + (1-eta)*Gaussian."""

    shape_name: ClassVar[str] = "pvoigt"
    param_specs = staticmethod(parameter_specs(25.0, "eta"))
    kernel: ClassVar = staticmethod(pvoigt.kernel)
    kernel_with_derivs: ClassVar = staticmethod(pvoigt.kernel_with_derivs)

    def _get_extra_params(self, cluster_params: ClusterParameters) -> tuple[Any, ...]:
        """Extract eta parameter."""
        return (cluster_params.extras["eta"],)

    def _process_extra_derivs(
        self,
        derivs: dict[str, FloatArray],
        raw_derivs: dict[str, FloatArray],
        sign: FloatArray,
    ) -> dict[str, FloatArray]:
        """Add eta derivative."""
        if "eta" in raw_derivs:
            derivs["eta"] = sign * raw_derivs["eta"]
        return derivs


class PseudoVoigtDoublet(SimpleDoubletBase):
    """Pseudo-Voigt doublet lineshape with J-coupling."""

    shape_name: ClassVar[str] = "pvoigt_doublet"
    param_specs = staticmethod(parameter_specs(25.0, "eta", "j"))
    kernel: ClassVar = staticmethod(pvoigt.kernel)
    kernel_with_derivs: ClassVar = staticmethod(pvoigt.kernel_with_derivs)

    def _get_extra_params(self, cluster_params: ClusterParameters) -> tuple[Any, ...]:
        """Extract eta parameter."""
        return (cluster_params.extras["eta"],)

    def _process_extra_derivs(
        self,
        derivs: dict[str, FloatArray],
        raw_derivs: dict[str, FloatArray],
    ) -> dict[str, FloatArray]:
        """Add eta derivative."""
        if "eta" in raw_derivs:
            derivs["eta"] = raw_derivs["eta"]
        return derivs


# =============================================================================
# Phased lineshapes
# =============================================================================


class SP1(PhasedSingletBase):
    """SP1 singlet lineshape with sine-bell apodization."""

    shape_name: ClassVar[str] = "sp1"
    param_specs = staticmethod(parameter_specs(2.0, "phase"))
    kernel: ClassVar = staticmethod(sp1.kernel)
    kernel_with_derivs: ClassVar = staticmethod(sp1.kernel_with_derivs)
    make_state: ClassVar = staticmethod(sp1.make_state)


class SP1Doublet(PhasedDoubletBase):
    """SP1 doublet lineshape with sine-bell apodization."""

    shape_name: ClassVar[str] = "sp1_doublet"
    param_specs = staticmethod(parameter_specs(2.0, "j", "phase"))
    kernel: ClassVar = staticmethod(sp1.kernel)
    kernel_with_derivs: ClassVar = staticmethod(sp1.kernel_with_derivs)
    make_state: ClassVar = staticmethod(sp1.make_state)


class SP2(PhasedSingletBase):
    """SP2 singlet lineshape with squared sine-bell apodization."""

    shape_name: ClassVar[str] = "sp2"
    param_specs = staticmethod(parameter_specs(2.0, "phase"))
    kernel: ClassVar = staticmethod(sp2.kernel)
    kernel_with_derivs: ClassVar = staticmethod(sp2.kernel_with_derivs)
    make_state: ClassVar = staticmethod(sp2.make_state)


class SP2Doublet(PhasedDoubletBase):
    """SP2 doublet lineshape with squared sine-bell apodization."""

    shape_name: ClassVar[str] = "sp2_doublet"
    param_specs = staticmethod(parameter_specs(2.0, "j", "phase"))
    kernel: ClassVar = staticmethod(sp2.kernel)
    kernel_with_derivs: ClassVar = staticmethod(sp2.kernel_with_derivs)
    make_state: ClassVar = staticmethod(sp2.make_state)


class NoApod(PhasedSingletBase):
    """NoApod singlet lineshape (no apodization, exponential decay)."""

    shape_name: ClassVar[str] = "no_apod"
    param_specs = staticmethod(parameter_specs(2.0, "phase"))
    kernel: ClassVar = staticmethod(no_apod.kernel)
    kernel_with_derivs: ClassVar = staticmethod(no_apod.kernel_with_derivs)
    make_state: ClassVar = staticmethod(no_apod.make_state)


class NoApodDoublet(PhasedDoubletBase):
    """NoApod doublet lineshape (no apodization, exponential decay)."""

    shape_name: ClassVar[str] = "no_apod_doublet"
    param_specs = staticmethod(parameter_specs(2.0, "j", "phase"))
    kernel: ClassVar = staticmethod(no_apod.kernel)
    kernel_with_derivs: ClassVar = staticmethod(no_apod.kernel_with_derivs)
    make_state: ClassVar = staticmethod(no_apod.make_state)


__all__ = [
    "SP1",
    "SP2",
    "Gaussian",
    "GaussianDoublet",
    "Lorentzian",
    "LorentzianDoublet",
    "NoApod",
    "NoApodDoublet",
    "PseudoVoigt",
    "PseudoVoigtDoublet",
    "SP1Doublet",
    "SP2Doublet",
]
