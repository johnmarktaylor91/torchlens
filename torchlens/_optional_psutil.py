"""The optional ``psutil`` import, with its absence remembered for the process.

Python does not cache a failed import: every ``import psutil`` on a host without
it walks each ``sys.path`` entry again. The capture's forward bracket reads the
process RSS twice per trace, so without this a host lacking psutil paid two full
import-path searches on every capture, growing with ``sys.path``.
"""

from __future__ import annotations

import sys
from typing import Any

#: Holds ``True`` once ``import psutil`` has failed with nothing in ``sys.modules``.
_ABSENT: list[bool] = []


def optional_psutil() -> Any | None:
    """Return the ``psutil`` module, or ``None`` when it cannot be imported.

    A module present in ``sys.modules`` is always returned, so one imported
    later in the process is still seen. Absence is remembered only when the
    import failed with no ``sys.modules`` entry; a module hidden by
    ``sys.modules["psutil"] = None`` is reported missing without being
    remembered, so the real module is found again once the entry is restored.

    Returns
    -------
    Any | None
        The ``psutil`` module, or ``None``.
    """

    module = sys.modules.get("psutil")
    if module is not None:
        return module
    if _ABSENT:
        return None
    try:
        import psutil
    except ImportError:
        if "psutil" not in sys.modules:
            _ABSENT.append(True)
        return None
    return psutil
