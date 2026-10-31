"""Probe for a stable ``Tensor.untyped_storage()`` Python wrapper (a torch-only leaf).

The ``HAS_CACHED_UNTYPED_STORAGE_WRAPPER`` flag and its capability-snapshot row stay in
``torchlens._utils._torch_compat``, which imports this module at probe time (the sibling
of ``_type_new_slot`` and ``_script_call_slot``) so its eager import closure does not grow.
"""

from __future__ import annotations

import torch


def probe_cached_untyped_storage_wrapper() -> bool:
    """Return whether a tensor retains one stable untyped-storage Python wrapper.

    Torch 2.1 creates a fresh ``UntypedStorage`` wrapper on every
    ``Tensor.untyped_storage()`` call. Its wrapper destructor can leave a
    ``weakref.ref`` pointing at freed memory, so placing an ephemeral handle in a
    ``WeakKeyDictionary`` can later segfault CPython while clearing the weakref.
    Newer torch retains one wrapper on the tensor, which makes weak-key storage
    registries safe for the tensor's lifetime.

    The probe deliberately compares two live handles instead of constructing a
    weakref: exercising the broken weakref destructor would itself corrupt the
    interpreter on an unsupported runtime.

    Returns
    -------
    bool
        ``True`` when repeated calls return the same retained wrapper.
    """

    try:
        tensor = torch.empty(0)
        first = tensor.untyped_storage()
        second = tensor.untyped_storage()
    except (AttributeError, RuntimeError, TypeError):
        return False
    return first is second
