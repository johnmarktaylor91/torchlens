"""The capture-owned factory-device slot (W1-CTX, weightsfree memo D4).

TorchLens owns the meta factory device during an ADMITTED weights-free
capture; no torch ``DeviceContext`` mode is ever on the stack during the
captured forward. The ratified L7a 1.4-A scoped ``torch.device("meta")``
context is REPLACED by this slot (memo sec 11 ratification item): the torch
mode's catch-all ``__torch_function__`` re-entry respells every dunder op
(``__add__`` -> ``add``) and was measured as the sole cause of two
twin-refuting digest asymmetries (defect L3), while its ONE contribution —
factory-op device injection — is already replicated by TorchLens's own
wrapper (:func:`~torchlens.backends.torch.wrappers._maybe_inject_device_kwarg`).

Production form (memo build item 3): nest-safe (a stack, not a flag),
owner-thread-scoped (``threading.local`` — worker threads never see the
slot), restores on ``BaseException`` (context-manager ``finally``), wraps
ONLY the user forward (armed by the capture scope in
``torchlens/capture/trace.py``, never imports/preparation/postprocessing),
and is published through ``tl.compat.report()`` beside the W1 root set.

Every spelling is DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch

__all__ = ["active_factory_device", "factory_device_scope"]


class _FactoryDeviceSlot(threading.local):
    """Owner-thread-scoped nest-safe stack of capture factory devices."""

    def __init__(self) -> None:
        self.stack: list[torch.device] = []


# Module-level mutable state: thread-local, mutated only by the context
# manager below; inventoried by tests/test_global_state_inventory.py.
_SLOT = _FactoryDeviceSlot()


def active_factory_device() -> torch.device | None:
    """The innermost armed factory device on THIS thread, or ``None``.

    Consulted by the factory-injection helper in ``wrappers.py`` after the
    torch mode-stack query: a caller-pinned non-None ``device=`` kwarg always
    wins; the slot only fills the absent/None cell, exactly like native C
    dispatch under a device context.
    """

    stack = _SLOT.stack
    return stack[-1] if stack else None


@contextmanager
def factory_device_scope(device: torch.device) -> Iterator[None]:
    """Arm the factory-device slot for the duration of one captured forward.

    Nest-safe and ``BaseException``-safe: the slot entry is popped in a
    ``finally`` so a failing forward (or a ``KeyboardInterrupt``) never leaks
    the device into later captures.

    Parameters
    ----------
    device:
        The factory placement device (``torch.device("meta")`` for admitted
        weights-free captures).
    """

    _SLOT.stack.append(device)
    try:
        yield
    finally:
        _SLOT.stack.pop()
