"""Bare module addresses for a bound-method trace.

``tl.trace(model.generate, x)`` captures through a TorchLens-authored wrapper
root that registers the model as its ``owner`` submodule, so the recorded
module addresses read ``owner.body.0``. A user who traced a method of their
model names its modules as they would on a plain trace (``body.0``); this
module maps such a bare address to the recorded ``owner.``-prefixed one. It
applies only to traces whose root entry point is a bound method, so a plain
model with a real ``owner`` submodule never gains the mapping.
"""

from __future__ import annotations

from typing import Any

__all__ = ("owner_address_for",)

#: Submodule name under which the wrapper root registers the owner
#: (``backends.torch.bound_root.OWNER_ATTRIBUTE``).
_OWNER_PREFIX = "owner."
_BOUND_METHOD_ENTRY = "bound_method:"


def owner_address_for(accessor: Any, key: Any) -> Any:
    """Return ``key`` rewritten to its ``owner.`` address on a bound-method trace.

    Parameters
    ----------
    accessor:
        The ``ModuleAccessor`` resolving ``key``.
    key:
        Lookup key as the caller passed it.

    Returns
    -------
    Any
        ``"owner." + key`` when ``key`` is unknown to the accessor, the
        prefixed spelling is known, and the trace was captured from a bound
        method; otherwise ``key`` unchanged.
    """

    if not isinstance(key, str) or not key or key == "self":
        return key
    known = (accessor._dict, accessor._alias_dict, accessor._pass_dict)
    if any(key in index for index in known):
        return key
    candidate = _OWNER_PREFIX + key
    if not any(candidate in index for index in known):
        return key
    root = accessor._dict.get("self")
    ref = getattr(root, "__dict__", {}).get("_source_trace_ref") if root is not None else None
    trace = ref() if ref is not None else None
    entry_point = getattr(trace, "_provenance_root_entry_point", None)
    if isinstance(entry_point, str) and entry_point.startswith(_BOUND_METHOD_ENTRY):
        return candidate
    return key
