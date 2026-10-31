"""Persistence-safe repr of an activation-transform callable.

Every save level stores ``Trace._activation_transform_repr``; the helper here
redacts what a callable's ``repr`` would otherwise leak into a saved artifact
(bound ``functools.partial`` arguments and live heap addresses).
"""

from __future__ import annotations

import functools
import re
from typing import Any

_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]+")
"""Matches CPython object/function ``repr`` heap addresses (``at 0x7f...``)."""


def _scrubbed_transform_repr(fn: Any) -> str | None:
    """Return a persistence-safe repr of an activation-transform callable.

    The repr is stored (``_activation_transform_repr``) at every save level. A plain
    function repr is inert, but ``functools.partial`` reprs embed the BOUND ARGUMENT
    VALUES (B8-20: a probe recovered a planted token from a saved artifact). Redact a
    partial's positional and keyword arguments to ``<scrubbed>`` while keeping the
    wrapped function's own (recursively scrubbed) repr, so the string stays useful
    without leaking captured values.

    Parameters
    ----------
    fn:
        Activation-transform callable, or ``None``.

    Returns
    -------
    str | None
        Scrubbed repr, or ``None`` when ``fn`` is ``None``.
    """
    if fn is None:
        return None
    if isinstance(fn, functools.partial):
        inner = _scrubbed_transform_repr(fn.func)
        parts = [inner] if inner is not None else []
        parts.extend("<scrubbed>" for _ in fn.args)
        parts.extend(f"{key}=<scrubbed>" for key in (fn.keywords or {}))
        return f"functools.partial({', '.join(parts)})"
    # A plain callable's ``repr`` embeds a live heap address for anything using
    # the default object/function repr (``<function f at 0x7f...>``,
    # ``<Foo object at 0x...>``) -- a non-deterministic value (breaks
    # byte-identical artifacts) and an ASLR heap-layout leak spliced into a
    # persisted KEEP field (b3-opus, completing the B8-20 scrub). Redact any
    # ``0x<hex>`` address before it is stored.
    return _ADDRESS_RE.sub("0x<scrubbed>", repr(fn))
