"""Single sanctioned spelling for resolving a top-level ``torch`` attribute on the
load / decode / exec path (r42 secC_1 / r45 secC_1 / r47 secD_1).

The top-level ``torch`` module implements PEP 562 ``__getattr__``: a bare
``getattr(torch, name)`` on an attacker-derived manifest / literal string can trigger an
UNREQUESTED lazy submodule import (``torch.onnx`` / ``torch._dynamo`` / ``torch._inductor`` /
``torch._export``), fire a deprecated-attribute ``replacement()`` shim (``torch.has_cuda`` ->
``torch.backends.cuda.is_built()``), or leak a raw ``ImportError`` outside TorchLens' typed
error vocabulary -- all BEFORE any value gate runs. Reading ``torch.__dict__`` directly never
invokes ``torch.__getattr__``, and every real dtype / layout / memory_format / qscheme / ``Size``
symbol (plus every real top-level class such as ``UntypedStorage`` / ``TypedStorage`` and every
public free-function alias such as ``equal`` / ``allclose`` / ``is_nonzero`` / ``add``) is a
genuine ``torch.__dict__`` entry, so it still resolves.

r47 secD_1 / secF_1: this is the CANONICAL helper home. It lives under ``torchlens.utils`` -- a
low-level package that imports only ``torch`` -- so low-level importers (``utils/_torch_compat``,
``backends/torch/*``, ``_runnable_state``, ``capture/*``, ``fastlog/*``, ``intervention/*``,
``postprocess/*``) resolve it WITHOUT pulling ``_io/__init__.py`` (avoiding an import cycle).
``torchlens/_io/_torch_symbols.py`` re-exports :func:`torch_attr` unchanged for the load/decode
importers already spelling ``from ._torch_symbols import torch_attr``. The public contract is one
helper, not a physical module path.

Every attacker-derived top-level ``torch``-name resolution anywhere in the package routes through
:func:`torch_attr`; a whole-package AST immunizer (``tests/test_r45_torch_symbol_decode.py``) fails
on any bare ``getattr(torch, <non-literal>)`` OR ``hasattr(torch, <non-literal>)`` in
``torchlens/**/*.py`` so a future site cannot reintroduce the lazy-import / deprecated-replacement
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
