"""Compatibility layer for fitting API.

This module re-exports the public fitting API from ``peakfit.core.fitting``
so that existing examples and user code importing ``peakfit.fitting`` continue
to work. Prefer importing from ``peakfit.core.fitting`` for internal code.
"""

import warnings

warnings.warn(
    "The 'peakfit.fitting' package is deprecated. Use 'peakfit.core.fitting' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from peakfit.core.fitting import *  # noqa: F401,F403,E402
