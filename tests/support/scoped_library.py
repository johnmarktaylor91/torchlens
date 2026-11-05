"""A custom CPU operator registered for the length of a ``with`` block on every supported torch.

``torch.library._scoped_library``, ``torch.library.register_kernel`` and ``Library._destroy``
arrived after the torch floor. Where they exist the helper uses them unchanged. On older
torch it spells the same lifetime with parts every supported torch has: a FRAGMENT
``Library``, ``Library.impl`` for the CPU kernel, and on exit dropping the C++ registration
(which removes the block's ``define`` and ``impl`` rows), forgetting the Python-side impl
keys, and evicting a freshly defined operator from ``torch.ops`` so a later block can define
it again.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import torch


def _register(lib: Any, namespace: str, schema: str, kernel: Callable[..., Any]) -> bool:
    """Define ``schema`` unless the operator exists, then add ``kernel`` as its CPU kernel.

    Returns
    -------
    bool
        Whether this call defined the operator (so teardown must evict it).
    """

    name = schema.split("(", 1)[0]
    defined = not hasattr(getattr(torch.ops, namespace), name)
    if defined:
        lib.define(schema)
    register_kernel = getattr(torch.library, "register_kernel", None)
    if register_kernel is not None:
        register_kernel(f"{namespace}::{name}", "cpu", kernel, lib=lib)
    else:
        lib.impl(name, kernel, "CPU")
    return defined


def _release(lib: Any, namespace: str, name: str, defined: bool) -> None:
    """Undo a pre-``_scoped_library`` fragment: registrations, impl keys and the cached op."""

    torch.library._impls.difference_update(lib._op_impls)
    lib._op_impls.clear()
    lib.m = None
    ops_namespace = getattr(torch.ops, namespace)
    if defined and name in vars(ops_namespace):
        delattr(ops_namespace, name)
        if name in ops_namespace._dir:
            ops_namespace._dir.remove(name)


@contextlib.contextmanager
def scoped_cpu_op(namespace: str, schema: str, kernel: Callable[..., Any]) -> Iterator[Any]:
    """Yield ``torch.ops.<namespace>.<op>`` backed by ``kernel`` on CPU until the block ends.

    Parameters
    ----------
    namespace:
        The operator namespace the FRAGMENT library extends.
    schema:
        The operator schema; defined only when the operator does not exist yet (a real
        package may already have registered it, and then only the CPU kernel is added).
    kernel:
        The Python CPU kernel.

    Yields
    ------
    torch._ops.OpOverloadPacket
        The operator packet.
    """

    name = schema.split("(", 1)[0]
    scoped_library = getattr(torch.library, "_scoped_library", None)
    if scoped_library is not None:
        with scoped_library(namespace, "FRAGMENT") as lib:
            _register(lib, namespace, schema, kernel)
            yield getattr(getattr(torch.ops, namespace), name)
        return
    lib = torch.library.Library(namespace, "FRAGMENT")
    defined = False
    try:
        defined = _register(lib, namespace, schema, kernel)
        yield getattr(getattr(torch.ops, namespace), name)
    finally:
        _release(lib, namespace, name, defined)
