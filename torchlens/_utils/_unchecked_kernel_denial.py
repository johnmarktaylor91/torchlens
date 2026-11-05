"""Structural denial of torch kernels that skip their own memory bounds checks.

A saved bundle is UNTRUSTED input: it names the callable for every replayed op and
supplies that op's tensor and literal arguments. Most aten kernels validate their
indices, sizes, strides and offsets against the tensors they touch, so a hostile
argument raises. A small family deliberately does NOT, because it exists for callers
(torch's own decompositions and compilers) that have already proven the arguments in
range. Handed attacker-chosen arguments, each one reads or writes outside its tensor's
storage:

1. ``_unsafe_*`` and ``*_unsafe`` kernels (``_unsafe_index``, ``_unsafe_index_put``,
   ``_unsafe_masked_index``, ``_unsafe_view``, ``_sparse_coo_tensor_unsafe``,
   ``_sparse_csr_tensor_unsafe``): the aten naming convention for "no bounds check".
2. Kernels whose schema takes an ``unsafe`` argument that turns validation off
   (``segment_reduce(..., unsafe=True)`` skips the lengths check,
   ``_index_put_impl_(..., unsafe=True)`` skips the index bounds check).
3. Private view constructors that adopt caller-supplied size, stride or offset
   metadata without checking it against the storage: ``_nested_view_from_buffer``,
   ``_nested_view_from_jagged`` (and their ``_copy`` variants) and ``_reshape_alias``
   (built on ``alias_with_sizes_and_strides``, which sets sizes and strides with no
   storage-extent check, unlike the public ``as_strided``).

None of these is a forward op a user's model calls through the public API (the
public ``Tensor.index_put_``, ``reshape`` and ``nested.nested_tensor`` route to them
only with arguments torch validated first), so the resolution gate refuses the whole
family. The rule is structural: the name prefixes in :data:`UNCHECKED_NAME_PREFIXES`
plus every aten schema that declares an ``unsafe`` argument, read from the live
dispatcher, so a kernel torch adds later is denied the moment it ships.

Not covered: the PUBLIC sparse constructors (``torch.sparse_coo_tensor`` and siblings)
also skip index validation while torch's sparse invariant checks are off (the default),
but they are legitimate forward ops and cannot be refused by name.

This module imports torch and the standard library only, plus the torch compat
schema accessor.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from ._torch_compat import get_jit_all_schemas

__all__ = [
    "UNCHECKED_NAME_PREFIXES",
    "UNCHECKED_NAME_SUFFIXES",
    "is_unchecked_memory_kernel",
    "unchecked_switch_kernel_names",
]

#: Callable-name prefixes of kernels that skip bounds validation by design.
UNCHECKED_NAME_PREFIXES: tuple[str, ...] = (
    "_unsafe",
    "_nested_view_from_",
    "_reshape_alias",
)

#: Callable-name suffixes of the same family: the private sparse constructors that skip
#: index validation (``_sparse_coo_tensor_unsafe``, ``_sparse_csr_tensor_unsafe``,
#: ``_sparse_compressed_tensor_unsafe``, ...) and ``_view_func_unsafe``.
UNCHECKED_NAME_SUFFIXES: tuple[str, ...] = ("_unsafe",)

#: Kernels known to take an ``unsafe`` validation switch, used only when the live
#: dispatcher schema table is unavailable (the rule then still refuses these).
_FALLBACK_SWITCH_KERNELS: frozenset[str] = frozenset(
    {"segment_reduce", "_segment_reduce", "_index_put_impl_", "_index_put_impl"}
)


@lru_cache(maxsize=1)
def unchecked_switch_kernel_names() -> frozenset[str]:
    """Return every aten op name whose schema declares an ``unsafe`` argument.

    Returns
    -------
    frozenset[str]
        Unqualified op names (``segment_reduce``, ``_index_put_impl_``, ...); the
        fallback set when the dispatcher schema table cannot be read.
    """

    schemas = get_jit_all_schemas()
    if schemas is None:
        return _FALLBACK_SWITCH_KERNELS
    names: set[str] = set(_FALLBACK_SWITCH_KERNELS)
    for schema in schemas:
        qualified = str(getattr(schema, "name", ""))
        if not qualified.startswith("aten::"):
            continue
        arguments = getattr(schema, "arguments", ())
        if any(getattr(argument, "name", None) == "unsafe" for argument in arguments):
            names.add(qualified.removeprefix("aten::"))
    return frozenset(names)


def is_unchecked_memory_kernel(func: Callable[..., Any] | Any) -> bool:
    """Return whether ``func`` is a torch kernel that skips its memory bounds checks.

    Parameters
    ----------
    func:
        A resolved callable (a torch builtin, Tensor method descriptor or wrapper).

    Returns
    -------
    bool
        ``True`` for an ``_unsafe*`` / ``*_unsafe`` / ``_nested_view_from_*`` /
        ``_reshape_alias*`` kernel (matched on the op packet, so ``torch.ops.aten``
        overloads are covered) or any aten op whose schema takes an ``unsafe`` switch.
    """

    raw = getattr(func, "__name__", None)
    if not isinstance(raw, str) or not raw:
        return False
    # ``torch.ops.aten.<op>.<overload>`` reports ``"<op>.<overload>"``: match the op packet.
    name = raw.split(".", 1)[0]
    if name.startswith(UNCHECKED_NAME_PREFIXES) or name.endswith(UNCHECKED_NAME_SUFFIXES):
        return True
    return name in unchecked_switch_kernel_names()
