"""The debugging tools one public namespace re-exports (today: ``tl.debug``).

These are the numerical, parameter and gradient diagnostics and the report base class.
Their code stays in the private modules that define them; this module is the single
place a public facade imports them from, so re-homing them under another namespace is
one facade edit (move the import block that reads from this module).
"""

from __future__ import annotations

from ._debug_report import Report
from .checks._audit import ParamHealthReport, param_health
from .debug._determinism import DeterminismReport, check_determinism
from .debug._dtype_range import DTypeRangeReport, dtype_range
from .debug._gradients import grad_health
from .debug._nan import NonfiniteReport, nonfinite_origin

__all__ = [
    "DTypeRangeReport",
    "DeterminismReport",
    "NonfiniteReport",
    "ParamHealthReport",
    "Report",
    "check_determinism",
    "dtype_range",
    "grad_health",
    "nonfinite_origin",
    "param_health",
]
