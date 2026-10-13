"""Read values a probe model's forward stashed on a module, from the trace.

A default ``tl.trace`` is a pure observer: every plain attribute the forward
wrote is put back afterwards, so a probe that records observations on the
module (``self.seen.append(...)``) is read from the trace's attribute-write
record (``Module.attribute_writes``) instead of from the model.
"""

from __future__ import annotations

from typing import Any


def stashed(trace: Any, name: str, module: str = "self") -> Any:
    """Return what the forward left in ``module``'s attribute ``name``."""

    write = trace.modules[module].attribute_writes[name]
    assert write.kind in {"added", "replaced"}, write
    return write.value
