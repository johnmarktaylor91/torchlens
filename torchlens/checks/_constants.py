"""The ONE constants module for checks-kit knobs (checks memo 4.4).

``audit_params`` keeps knob parity with ``tl.debug.dtype_range_audit``;
``tests/test_checks_kit_scan.py`` pins the parity so the two doors can
never silently drift apart without a deliberate, reviewed change here.
"""

from __future__ import annotations

__tl_layer__ = "L5"

#: Fraction of the dtype ceiling above which headroom warns.
MAX_FRACTION_DEFAULT = 0.9

#: Minimum fraction of elements in the subnormal interval that warns.
SUBNORMAL_FRACTION_THRESHOLD_DEFAULT = 0.1

#: M-of-N warn window default (memo 4.5; pending the calibration matrix --
#: the window's calibration target is a MODEL property: legitimate
#: zero-gradient intermittency from MoE routing, conditional branches, and
#: embedding batch coverage).
WARN_WINDOW_DEFAULT = (5, 8)

#: Watchdog threshold: warn when backwards since the last accepted step
#: exceed ``accumulation_steps * WATCHDOG_FACTOR`` (memo D11).
WATCHDOG_FACTOR = 4

__all__ = [
    "MAX_FRACTION_DEFAULT",
    "SUBNORMAL_FRACTION_THRESHOLD_DEFAULT",
    "WARN_WINDOW_DEFAULT",
    "WATCHDOG_FACTOR",
]
