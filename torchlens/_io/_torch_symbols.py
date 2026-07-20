"""Single sanctioned spelling for resolving a top-level ``torch`` attribute on the
load / decode / exec path (r42 secC_1 / r45 secC_1).

The top-level ``torch`` module implements PEP 562 ``__getattr__``: a bare
``getattr(torch, name)`` on an attacker-derived manifest / literal string can trigger an
UNREQUESTED lazy submodule import (``torch.onnx`` / ``torch._dynamo`` / ``torch._inductor`` /
``torch._export``), fire a deprecated-attribute ``replacement()`` shim (``torch.has_cuda`` ->
``torch.backends.cuda.is_built()``), or leak a raw ``ImportError`` outside TorchLens' typed
error vocabulary -- all BEFORE any value gate runs. Reading ``torch.__dict__`` directly never
invokes ``torch.__getattr__``, and every real dtype / layout / memory_format / qscheme / ``Size``
symbol is a genuine ``torch.__dict__`` entry, so it still resolves.

Every attacker-derived top-level ``torch``-name resolution on the load/decode/exec path routes
through :func:`torch_attr`; an AST immunizer
(``tests/test_r45_torch_symbol_decode.py``) fails on any bare ``getattr(torch, <non-literal>)``
on that path so a future decode site cannot reintroduce the lazy-import / deprecated-replacement
side effect. Class roots (``torch.Tensor`` / ``torch._C`` / ``torch.backends``) and literal-name
``getattr(torch, "...")`` module-layout constants carry no lazy hazard and stay out of scope.
"""

from __future__ import annotations

from typing import Any

import torch


def torch_attr(name: str) -> Any | None:
    """Resolve one top-level ``torch`` attribute without firing ``torch.__getattr__``.

    Parameters
    ----------
    name:
        Bare top-level ``torch`` attribute name (no ``torch.`` prefix, no dotted path).

    Returns
    -------
    Any | None
        The value from ``torch.__dict__`` when present, otherwise ``None``. A dotted or
        non-identifier ``name`` (never a real top-level torch symbol) returns ``None``
        without touching ``torch.__dict__`` -- so no lazy import, deprecated
        ``replacement()``, or raw ``ImportError`` can escape this call.
    """

    if "." in name or not name.isidentifier():
        return None
    return torch.__dict__.get(name)
