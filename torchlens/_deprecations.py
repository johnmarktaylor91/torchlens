"""The MISSING kwarg-omission sentinel.

Historical note: this module also carried the additive-deprecation helpers
(``warn_deprecated_alias``, ``TorchLensDeprecationWarning``, ``REMOVED_IN``).
The 2026-08-19 shim-removal lane deleted every deprecation shim outright per
the interim-phase ruling, so only the load-bearing sentinel machinery remains.
Interim-phase policy is remove-and-rename, not shim: do not add new
deprecation routes here (``tests/test_deprecation_inventory.py`` pins the
package deprecation-free).
"""

from __future__ import annotations

from typing import Final


class MissingType:
    """Sentinel type used to detect explicitly supplied public kwargs.

    Notes
    -----
    Public APIs must distinguish ``caller omitted this kwarg`` from ``caller
    explicitly passed the public default``. A dedicated sentinel keeps those
    cases separate without relying on value comparisons.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a stable debugging representation."""

        return "MISSING"


MISSING: Final[MissingType] = MissingType()
