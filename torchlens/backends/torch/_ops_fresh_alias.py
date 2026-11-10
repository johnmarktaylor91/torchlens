"""The payload of a fresh plain output on the direct path: the deferred alias or the eager copy.

:func:`fresh_payload` makes the same decision as ``_clone_tensor_payload`` in its
``"copy"`` mode with no move, and registers the same pending alias, for a tensor
the direct path's eligibility check already accepted (``_ops_fresh_payload``):
its class is exactly ``torch.Tensor``, its layout is strided, it is not a meta
tensor and it carries no TorchLens metadata. The checks of
``tensor_utils._try_defer_payload_alias`` that those facts settle are not
repeated here:

* the ``nn.Parameter`` test: the class is exactly ``torch.Tensor``;
* the layout test: strided;
* the meta-device test: not meta;
* the tracing-tensor test (``FakeTensor``, ``FunctionalTensor``): both are
  subclasses, which an exact ``torch.Tensor`` is not;
* the element-count test: the caller passes the byte size, which is zero
  exactly when the element count is.

Every other check runs, in one routed-read scope, with the same outcome: the
quantized, conjugate, negative and inference flags, the storage key, the
excluded state storages and the whole-storage coverage. The storage reads run
under the caller's ``pause_logging`` without the internal-read marker: every
consumer of that marker also requires logging to be on, which ``pause_logging``
turns off, and TorchLens's other storage-identity reads (the save budget's,
``_storage_exports._storage_of``) run under ``pause_logging`` alone. The alias
shares its source's storage, so its storage identity for the save budget is
the storage key already read.

``tests/test_fresh_payload_alias.py`` compares the decision and the registry
entry with ``_try_defer_payload_alias`` over the cases each check exists for.
"""

from __future__ import annotations

import weakref
from functools import partial
from typing import Any

import torch

from ..._utils import tensor_utils as _tensor_utils
from ..._utils._torch_symbols import tensor_read, tensor_reads
from ..._utils.tensor_utils import (
    _DEFER_GATE,
    _DEFER_PENDING,
    _copy_tensor_payload,
    _drop_dead_deferred_alias,
    _mint_graph_connected_alias,
    _PendingPayloadAlias,
)

__all__ = ("fresh_payload",)

#: ``(device string, storage data pointer, storage bytes)``, the save budget's storage identity.
StorageIdentity = tuple[str, int, int]


def _whole_storage_key(t: torch.Tensor) -> tuple[int, int, str] | None:
    """Return ``t``'s pending-registry key when an alias of it may be deferred, else ``None``.

    The flag, storage and coverage checks of ``_try_defer_payload_alias`` for an
    exact strided ``torch.Tensor``, read through one routed-read scope.
    """

    state_ptrs = _tensor_utils._DEFER_STATE_PTRS
    if state_ptrs is None or t.is_quantized:
        return None
    conj, neg, inference, storage_of, offset, element_size = tensor_reads(
        t, "is_conj", "is_neg", "is_inference", "untyped_storage", "storage_offset", "element_size"
    )
    try:
        if conj(t) or neg(t) or inference(t):
            return None
        storage = storage_of(t)
        ptr = storage.data_ptr()
        nbytes = storage.nbytes()
    except Exception:
        return None
    if ptr == 0 or nbytes == 0 or ptr in state_ptrs:
        # Model state storages stay eager (C++ side effects move their bytes).
        return None
    if not _covers_whole_storage(t, nbytes, offset, element_size):
        # Partial coverage stays eager: an alias would pin the whole storage.
        return None
    return (ptr, nbytes, str(t.device))


def _covers_whole_storage(t: torch.Tensor, nbytes: int, offset: Any, element_size: Any) -> bool:
    """Return whether ``t`` spans its storage end to end from offset 0.

    ``tensor_utils._alias_covers_whole_storage`` with the routed reads passed in.
    """

    try:
        if offset(t) != 0:
            return False
        span = 1
        for size, stride in zip(t.shape, t.stride(), strict=True):
            if size == 0:
                return False
            span += (size - 1) * (stride if stride >= 0 else -stride)
        return bool(span * element_size(t) == nbytes)
    except Exception:
        return False


def _defer_fresh_alias(
    t: torch.Tensor, graph_connected: bool
) -> tuple[torch.Tensor, StorageIdentity] | None:
    """Register a clone-on-write alias of ``t`` as ``_try_defer_payload_alias`` does.

    Returns the alias and its storage identity, or ``None`` when ``t`` keeps the
    eager copy.
    """

    key = _whole_storage_key(t)
    if key is None:
        return None
    try:
        alias = _mint_graph_connected_alias(t) if graph_connected else t.detach()
        version = int(alias._version)
    except Exception:
        return None
    ref = weakref.ref(alias, partial(_drop_dead_deferred_alias, key))
    entry = _PendingPayloadAlias(ref, version)
    if key in _DEFER_PENDING:
        entries = _DEFER_PENDING[key]
        entries.append(entry)
        _DEFER_PENDING[key] = entries  # stored after the append: a death callback may pop ``key``
    else:
        _DEFER_PENDING[key] = [entry]
    _DEFER_GATE.armed = True
    return alias, (key[2], key[0], key[1])


def _storage_identity(payload: torch.Tensor) -> StorageIdentity | None:
    """Return the save budget's storage identity of an eager copy, or ``None`` when unreadable."""

    try:
        storage = tensor_read(payload, "untyped_storage")(payload)
        return (str(payload.device), int(storage.data_ptr()), int(storage.nbytes()))
    except Exception:
        return None


def fresh_payload(
    t: torch.Tensor, detach: bool, num_bytes: int, want_identity: bool
) -> tuple[torch.Tensor, StorageIdentity | None, bool]:
    """Return the payload of ``t``, its storage identity and ``t.requires_grad``.

    Parameters
    ----------
    t:
        A fresh plain output the direct path accepted.
    detach:
        The trace's ``detach_saved_activations`` option.
    num_bytes:
        ``t``'s byte size (zero exactly when it has no elements).
    want_identity:
        Whether the caller needs the payload's storage identity (a budget admission
        is open); otherwise the identity is ``None``.

    Notes
    -----
    Callers hold ``pause_logging``.
    """

    found: Any = None
    requires_grad = t.requires_grad
    if _tensor_utils._DEFER_WINDOW_DEPTH and num_bytes > 0:
        if detach or not requires_grad or not torch.is_grad_enabled():
            found = _defer_fresh_alias(t, False)
        elif _tensor_utils._DEFER_GRAD_ENABLED:
            found = _defer_fresh_alias(t, True)
    if found is not None:
        return found[0], found[1], requires_grad
    payload = _copy_tensor_payload(t, detach_tensor=detach, save_mode="copy")
    return payload, (_storage_identity(payload) if want_identity else None), requires_grad
