"""Content identity for intervention rule terms (F4).

Rule ids, ``action_repr`` and ``spec_digest`` render tensor arguments through
these helpers instead of ``repr``: torch's ``repr`` truncates above 1000
elements and tags the device, so two model-width steering vectors differing in
one element used to share one identity. A tensor renders as its dtype, shape
and a hash of every element, read at the moment the identity is read.
"""

from __future__ import annotations

import hashlib
import weakref
from typing import Any

import torch

from .._state import pause_logging
from .types import HelperSpec, InterventionDecision

#: Bounded memo of tensor content tokens keyed by ``id(tensor)``. Each entry
#: holds a WEAK reference (it never keeps a user tensor alive, and a recycled
#: id can never hit because the dead reference no longer resolves to the
#: queried object) and the version counter the token was computed at.
_TENSOR_TOKEN_MEMO: dict[int, tuple[weakref.ref[torch.Tensor], int, str]] = {}
_TENSOR_TOKEN_MEMO_LIMIT = 256


def tensor_version(tensor: torch.Tensor) -> int | None:
    """Return a tensor's autograd version counter, or ``None`` when it has none.

    Parameters
    ----------
    tensor:
        Tensor whose in-place edit counter is read.

    Returns
    -------
    int | None
        The version counter; ``None`` for tensors without one (inference-mode
        tensors), which therefore never reuse a cached identity.
    """

    try:
        return int(tensor._version)
    except (RuntimeError, AttributeError):
        return None


def tensor_content_sha(tensor: torch.Tensor) -> str:
    """Hash a tensor's full logical content (device- and layout-independent).

    Parameters
    ----------
    tensor:
        Tensor to hash. Its values are copied to host memory as one
        contiguous buffer, so a CUDA tensor and a strided view hash like the
        equal-valued contiguous CPU tensor.

    Returns
    -------
    str
        16 hex characters of SHA-256 over the raw element bytes; ``"meta"``
        for meta tensors (no values). Layouts whose bytes cannot be viewed
        (quantized or exotic subclasses) fall back to hashing ``repr``, which
        torch truncates above 1000 elements.
    """

    # Identity can be read mid-capture (a rule fires after its tensor was
    # edited); the copy ops below are TorchLens-internal and must not log.
    with pause_logging():
        data = tensor.detach()
        if data.is_meta:
            return "meta"
        try:
            if data.layout is not torch.strided:
                data = data.to_dense()
            flat = data.resolve_conj().resolve_neg().cpu().contiguous().reshape(-1)
            raw = flat.view(torch.uint8).numpy()
        except (RuntimeError, TypeError, NotImplementedError):
            return "repr-" + hashlib.sha256(repr(data).encode()).hexdigest()[:16]
    return hashlib.sha256(raw.data).hexdigest()[:16]


def tensor_token(tensor: torch.Tensor, *, exact: bool) -> str:
    """Render one tensor as its content identity token.

    Parameters
    ----------
    tensor:
        Tensor argument of a rule term.
    exact:
        When ``True`` re-hash the bytes even if the version counter is
        unchanged (writes through ``.data`` or a NumPy view do not bump it);
        the fresh token refreshes the memo.

    Returns
    -------
    str
        ``tensor(dtype=..., shape=..., sha256=...)``.
    """

    key = id(tensor)
    version = tensor_version(tensor)
    if not exact and version is not None:
        entry = _TENSOR_TOKEN_MEMO.get(key)
        if entry is not None and entry[0]() is tensor and entry[1] == version:
            return entry[2]
    token = (
        f"tensor(dtype={tensor.dtype}, shape={tuple(tensor.shape)}, "
        f"sha256={tensor_content_sha(tensor)})"
    )
    if version is not None:
        if key not in _TENSOR_TOKEN_MEMO and len(_TENSOR_TOKEN_MEMO) >= _TENSOR_TOKEN_MEMO_LIMIT:
            _TENSOR_TOKEN_MEMO.pop(next(iter(_TENSOR_TOKEN_MEMO)))
        _TENSOR_TOKEN_MEMO[key] = (weakref.ref(tensor), version, token)
    return token


def collect_tensors(value: Any, found: list[torch.Tensor]) -> None:
    """Append every tensor :func:`render_value` renders by content, in order.

    Parameters
    ----------
    value:
        Rule term (helper spec, decision, plain container, or leaf).
    found:
        Output list, appended in place.
    """

    if isinstance(value, torch.Tensor):
        found.append(value)
    elif isinstance(value, HelperSpec):
        collect_tensors(value.args, found)
        collect_tensors(value.kwargs, found)
    elif isinstance(value, InterventionDecision):
        collect_tensors(value.hook, found)
    elif type(value) in (tuple, list):
        for item in value:
            collect_tensors(item, found)
    elif type(value) is dict:
        for item in value.values():
            collect_tensors(item, found)


def render_value(value: Any, *, exact: bool = False) -> str:
    """Render an argument like ``repr`` with tensors replaced by content tokens.

    Plain tuples, lists and dicts are rendered element-wise with ``repr``'s own
    punctuation, so a tensor-free value renders byte-identically to ``repr``
    (released rule ids of tensor-free rules are unchanged). Tensors render as
    :func:`tensor_token` (dtype, shape, hash of every element), never through
    torch's truncating, device-tagged ``repr``.

    Parameters
    ----------
    value:
        Argument to render.
    exact:
        Forwarded to :func:`tensor_token`.

    Returns
    -------
    str
        Canonical rendering.
    """

    if isinstance(value, torch.Tensor):
        return tensor_token(value, exact=exact)
    if isinstance(value, HelperSpec):
        nested: list[torch.Tensor] = []
        collect_tensors(value, nested)
        if nested:
            return render_helper(value, exact=exact)
        return repr(value)
    if type(value) is tuple:
        items = [render_value(item, exact=exact) for item in value]
        return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
    if type(value) is list:
        return "[" + ", ".join(render_value(item, exact=exact) for item in value) + "]"
    if type(value) is dict:
        pairs = (f"{key!r}: {render_value(item, exact=exact)}" for key, item in value.items())
        return "{" + ", ".join(pairs) + "}"
    return repr(value)


def render_helper(helper: HelperSpec, *, exact: bool) -> str:
    """Render a helper spec's portable ``(name, args, kwargs)`` identity."""

    args = render_value(helper.args, exact=exact)
    kwargs = render_value(helper.kwargs, exact=exact)
    return f"helper:{helper.helper_name}:{args}:{kwargs}"
