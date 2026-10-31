"""The ``tl.debug.param_health`` door (checks memo 4.4 / D15).

The implementation lives in :mod:`torchlens.checks` (one scan kernel, two
spellings: this one-shot audit and the registry's scheduled scan); this
module is the debug-namespace door the memo names. Knob parity with
``dtype_range_audit`` is pinned by test against
``torchlens.checks._constants``.
"""

from __future__ import annotations

from ..checks._audit import ParamHealthReport, param_health

__all__ = ["ParamHealthReport", "param_health"]
