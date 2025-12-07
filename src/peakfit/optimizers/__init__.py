"""Deprecated optimizers namespace.

This package forwards to :mod:`peakfit.contrib.optimizers`. Prefer importing
from :mod:`peakfit.contrib.optimizers` directly. The alias remains for
backward compatibility.
"""

from __future__ import annotations

import os
import warnings

_warn_deprecated = os.environ.get("PEAKFIT_WARN_DEPRECATED", "").lower() not in {
    "",
    "0",
    "false",
}
if _warn_deprecated:
    warnings.warn(
        "'peakfit.optimizers' is deprecated; use 'peakfit.contrib.optimizers' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

try:
    from peakfit.contrib.optimizers import *  # noqa: F403
except ModuleNotFoundError:
    # Keep import errors deferred to the call site.
    pass
