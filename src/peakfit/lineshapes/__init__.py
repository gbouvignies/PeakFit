"""Deprecated lineshapes namespace.

For core shapes use :mod:`peakfit.core.lineshapes`; optional/experimental
shapes move to :mod:`peakfit.contrib.lineshapes`. This alias is retained for
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
        "'peakfit.lineshapes' is deprecated; use 'peakfit.core.lineshapes' or 'peakfit.contrib.lineshapes'.",
        DeprecationWarning,
        stacklevel=2,
    )

try:
    from peakfit.core.lineshapes import *  # noqa: F403
except ModuleNotFoundError:
    # Keep import errors deferred to the call site.
    pass
