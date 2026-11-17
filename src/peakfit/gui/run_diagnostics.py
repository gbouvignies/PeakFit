"""Per-member diagnostics of a recorded run that every review view shares.

Pure functions over the recorded outcome, from the engine's own diagnostics: the
amplitude decomposition (``amplitude_identifiability``) and the correlated-profile
flag (``profile_correlations``). Nothing is refitted. ``DatasetFitResult`` keeps them
per run, computed off the GUI thread.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from peakfit.engine.algorithms.covariance_factors import profile_correlations
from peakfit.engine.diagnostics.identifiability import Cause, amplitude_identifiability

if TYPE_CHECKING:
    from peakfit.engine.algorithms.covariance_factors import PeakProfileCorrelation
    from peakfit.engine.scientific.final_outcome import FinalClusterOutcome


# The warning kind of each amplitude-decomposition cause that leaves a member's split
# undetermined. An invalid shape or a missing σ is not one: the profile says so itself.
DECOMPOSITION_WARNINGS = {
    Cause.NULL_SPACE: "Amplitudes not identifiable",
    Cause.ILL_CONDITIONED: "Ill-conditioned shapes",
    Cause.NEAR_COLLINEAR: "Near-collinear shapes",
}


def profile_flags(cluster: FinalClusterOutcome) -> tuple[PeakProfileCorrelation, ...]:
    """Each member's correlated-profile flag, in member order; empty without a profile.

    A usable cluster with an analytical evaluation over two planes or more has a
    profile; the profile, the tiles, the cluster page and the list read this.
    """
    evaluation = cluster.analytical_evaluation
    if not cluster.usable or evaluation is None or evaluation.amplitudes.shape[1] < 2:  # noqa: PLR2004
        return ()
    return tuple(profile_correlations(evaluation.covariance_factors, evaluation.amplitudes))


@dataclass(frozen=True, slots=True)
class MemberDiagnostics:
    """One recorded cluster's per-member diagnostics, by engine peak name."""

    undetermined: tuple[tuple[str, str], ...]  # (decomposition warning kind, peak)
    correlated: tuple[str, ...]  # peaks with profile errors correlated across planes


def member_diagnostics(cluster: FinalClusterOutcome) -> MemberDiagnostics:
    """The amplitude decomposition and correlated-profile flags of one recorded cluster.

    An SVD and covariance modes per cluster: ``DatasetFitResult.diagnostics`` keeps
    them per run, computed by the workers that hand a run to Qt.
    """
    evaluation = cluster.analytical_evaluation
    undetermined = []
    if cluster.usable and evaluation is not None:
        members = amplitude_identifiability(evaluation.shapes, evaluation.amplitudes)
        undetermined = [
            (DECOMPOSITION_WARNINGS[member.cause], name)
            for name, member in zip(cluster.peak_names, members, strict=True)
            if member.cause in DECOMPOSITION_WARNINGS
        ]
    flagged = zip(cluster.peak_names, profile_flags(cluster), strict=False)
    return MemberDiagnostics(
        tuple(undetermined), tuple(name for name, flag in flagged if flag.flagged)
    )
