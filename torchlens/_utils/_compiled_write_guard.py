"""Keep deferred payload aliases out of reach of compiled in-place writes.

Deferred payload clones (clone-on-write, :mod:`torchlens._utils.tensor_utils`)
stay zero-copy only because every write to a pending storage passes a TorchLens
torch wrapper first, which copies the pending aliases out. Code compiled by
``torch.compile`` breaks that premise: Inductor keeps an in-place write to an
input inside its generated kernel, so no wrapper runs at execution time and the
saved payload silently takes the written bytes. Two rules keep the premise true:

1. No deferral once Dynamo is loaded. A compiled function can exist only in a
   process that imported ``torch._dynamo``, and a guard-cache hit runs its
   kernel with no compile event to react to, so every capture made after that
   import keeps eager payload clones. The check reads module state through
   ``_torch_compat`` and never imports Dynamo.
2. Compilation materializes every pending alias. Dynamo and AOTAutograd trace
   a function by running its calls on tracing tensors (``FakeTensor``,
   ``FunctionalTensor``), and an in-place call reaches its TorchLens wrapper on
   one before any compiled kernel exists. Such an argument has no real storage
   to look up, so every pending alias is copied out instead, outside the
   tracer's dispatch modes and outside a fake kernel's Meta dispatch (Dynamo
   reaches the wrappers from inside one, where a real copy is left unfilled).

Dynamo's compile-start callback is not used: registering it imports Dynamo,
which a plain import or capture must not do, so it could not be in place for
the first compile after a capture; torch 2.1 does not have it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import tensor_utils
from ._torch_compat import (
    get_dynamo_compile_counters,
    get_tracing_tensor_types,
    suspend_dispatch_modes,
    torch_module_exists_without_importing,
)

_DYNAMO_MODULE = "torch._dynamo"


def compiled_writes_seen() -> bool:
    """Return whether Dynamo is loaded, cheaply enough for every wrapped call.

    Returns
    -------
    bool
        ``True`` once ``torch._dynamo`` is imported (a module is never
        unloaded). Reuses the lazy Dynamo-counters probe, which costs one
        ``sys.modules`` lookup until Dynamo loads and is cached after.
    """

    return get_dynamo_compile_counters() is not None


def compiled_writes_possible() -> bool:
    """Return whether compiled code that bypasses the torch wrappers may run.

    Returns
    -------
    bool
        ``True`` when ``torch._dynamo`` is imported; every pending alias left
        by earlier captures is then materialized before this returns.
    """

    if not compiled_writes_seen() and (
        torch_module_exists_without_importing(_DYNAMO_MODULE, unloaded=True)
        or not torch_module_exists_without_importing(_DYNAMO_MODULE)
    ):
        return False  # Dynamo not imported, or absent from this torch build
    materialize_all_deferred_payloads()
    return True


def materialize_all_deferred_payloads() -> None:
    """Copy every pending payload alias onto exclusive fresh storage.

    Runs with every active dispatch mode popped and Meta dropped from the
    thread's dispatch keys, so a flush requested from inside a tracer (fake or
    functional mode, or a fake kernel) copies real bytes. The version belt
    still refuses an alias whose storage was already written without
    interception.
    """

    live = [
        alias
        for entries in list(tensor_utils._DEFER_PENDING.values())
        for entry in list(entries)
        if (alias := entry.ref()) is not None
    ]
    if not live:
        return
    with suspend_dispatch_modes():
        tensor_utils.materialize_deferred_for_call(live)


def materialize_deferred_for_call(tensors: Iterable[Any]) -> None:
    """Materialize pending aliases before one mutating wrapped call.

    Parameters
    ----------
    tensors:
        The call's tensor arguments. A tracing tensor among them means a
        compiler is tracing the write for later execution on real tensors it
        does not name, so every pending alias is materialized (rule 2 above);
        otherwise :func:`tensor_utils.materialize_deferred_for_call` copies out
        the aliases of these storages only.
    """

    tensors = tensors if isinstance(tensors, list | tuple) else list(tensors)
    tracing_types = get_tracing_tensor_types()
    if tracing_types and any(isinstance(t, tracing_types) for t in tensors):
        materialize_all_deferred_payloads()
        return
    tensor_utils.materialize_deferred_for_call(tensors)


# The wrappers call the copy-out through the gate; bind this one so tracing tensors reach rule 2.
tensor_utils._DEFER_GATE.materialize = materialize_deferred_for_call
