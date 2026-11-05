"""Pre-perturbation check for the declared unread arguments of known transformers mask builders.

Transformers 4.57's eager and sdpa masks are built by ``torch.vmap`` over a mask function
``f(batch_idx, head_idx, q_idx, kv_idx)`` (``masking_utils._vmap_for_bhqkv``), which TorchLens
records as one ``vmap`` boundary op whose four positional parents are the batch, head, query
and key aranges. The mask functions never read ``head_idx`` (``and_mask`` only forwards it),
so the head edge is perturbation-insensitive by construction, and the strict check used to
fail it as an invented edge. Ruling d-aad71aaad45648ba93beaf7e05f2a1bb (JMT, 4 Nov 2026)
chose a declared table, modelled on the FlashAttention per-output dependence table
(``_foreign_kernel_dependence``): position 1 of a recognized builder is declared unread.

Fail-closed by construction. Strict (``False``) unless every one of these holds:

- the op's replay callable is a chain of TorchLens-instrumented ``torch.vmap`` layers whose
  innermost mapped callable is a table builder (the build-time ``__tl_transform_inner_fn__``
  attribute names each layer's callable);
- that builder is defined in ``transformers.masking_utils`` itself (its ``__globals__`` are
  that module's namespace) and IS the table's function, or its code object IS the nested code
  constant of the table's factory (callable identity, never a version string);
- a composer (``and_mask``) composes only table builders, checked recursively through its
  ``mask_functions`` closure cell, so a user mask function or a future transformers function
  outside the table removes the excuse for the whole call;
- every leaf builder's code never references the declared parameter (no instruction names it
  and no closure captures it), so an edited table function that starts reading it loses the
  excuse even though its identity still matches;
- the call is plain: as many positional arguments as the builder takes and no keywords, and
  the perturbed parent sits at a declared position and nowhere else.

Registered in ``exemptions.CUSTOM_EXEMPTION_CHECKS`` under ``vmap``.
"""

from __future__ import annotations

import dis
import inspect
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace

MASKING_UTILS_MODULE = "transformers.masking_utils"
"""Module whose namespace a recognized builder's ``__globals__`` must be."""

_INNER_FN_ATTR = "__tl_transform_inner_fn__"
"""Attribute ``backends/torch/_transform_wrappers._set_transform_metadata`` stamps on a built
transform: the callable handed to the builder."""

_MAX_VMAP_LAYERS = 8
"""Upper bound on nested vmap layers walked (``_vmap_for_bhqkv`` nests four)."""

_MAX_COMPOSITION_DEPTH = 8
"""Upper bound on nested ``and_masks`` compositions walked."""

_HEAD_IDX = frozenset({1})


@dataclass(frozen=True)
class KnownMaskBuilder:
    """One transformers mask builder whose unread positional arguments are declared.

    ``factory`` is the module-level name in ``transformers.masking_utils``. When ``closure`` is
    None the factory is itself the mask function; otherwise the mask function is the nested
    function the factory returns, identified by its code object (``co_name == closure``).
    ``composed`` names the closure variable of a composer holding the mask functions it calls.
    """

    factory: str
    closure: str | None
    composed: str | None
    unread: frozenset[int]


KNOWN_MASK_BUILDERS: tuple[KnownMaskBuilder, ...] = (
    # and_masks(*mask_functions).and_mask(b, h, q, kv): ANDs mask(b, h, q, kv) over its
    # mask_functions; it forwards head_idx and reads it nowhere else.
    KnownMaskBuilder("and_masks", "and_mask", "mask_functions", _HEAD_IDX),
    # causal_mask_function(b, h, q, kv) = kv_idx <= q_idx.
    KnownMaskBuilder("causal_mask_function", None, None, _HEAD_IDX),
    # packed_sequence_mask_function(m).inner_mask = m[b, q] == m[b, kv].
    KnownMaskBuilder("packed_sequence_mask_function", "inner_mask", None, _HEAD_IDX),
    # padding_mask_function(m).inner_mask = m[b, kv].
    KnownMaskBuilder("padding_mask_function", "inner_mask", None, _HEAD_IDX),
    # sliding_window_overlay(w).inner_mask = kv_idx > q_idx - w.
    KnownMaskBuilder("sliding_window_overlay", "inner_mask", None, _HEAD_IDX),
)
"""Transformers mask builders whose ``head_idx`` (position 1) is declared unread.

Anything else (``or_masks``, the chunked overlays, ``add_offsets_to_mask_function``, a user
mask function) has no row and keeps every edge strict.
"""


def vmapped_callable(func: Any) -> Any:
    """Return the callable a TorchLens-recorded chain of ``torch.vmap`` layers maps.

    Parameters
    ----------
    func:
        A ``vmap`` op's replay callable (``Op.func``).

    Returns
    -------
    Any
        The innermost callable when ``func`` and every nested layer are instrumented ``vmap``
        builds; None for anything else (another transform kind, a missing attribute, a chain
        deeper than ``_MAX_VMAP_LAYERS``, a loaded bundle with no callable).
    """

    current = func
    for _ in range(_MAX_VMAP_LAYERS):
        if getattr(current, "__tl_transform_kind__", None) != "vmap":
            return None
        current = getattr(current, _INNER_FN_ATTR, None)
        if current is None:
            return None
        if not getattr(current, "__tl_is_transform_boundary__", False):
            return current
    return None


def _table_row(fn: Any) -> KnownMaskBuilder | None:
    """Return the table row ``fn`` is, by identity within ``transformers.masking_utils``."""

    code = getattr(fn, "__code__", None)
    namespace = getattr(fn, "__globals__", None)
    if not isinstance(code, types.CodeType) or not isinstance(namespace, dict):
        return None
    if namespace.get("__name__") != MASKING_UTILS_MODULE:
        return None
    for row in KNOWN_MASK_BUILDERS:
        factory = namespace.get(row.factory)
        if row.closure is None:
            if fn is factory:
                return row
        elif _nested_code(factory, row.closure) is code:
            return row
    return None


def _nested_code(factory: Any, name: str) -> types.CodeType | None:
    """Return the single nested code constant called ``name`` in ``factory``'s code."""

    code = getattr(factory, "__code__", None)
    if not isinstance(code, types.CodeType):
        return None
    nested = [c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == name]
    return nested[0] if len(nested) == 1 else None


def _references(code: types.CodeType, name: str) -> bool:
    """Return whether ``code`` names ``name`` in any instruction or captures it in a closure."""

    if name in code.co_cellvars:
        return True
    for instruction in dis.get_instructions(code):
        value = instruction.argval
        if value == name or (isinstance(value, tuple) and name in value):
            return True
    return False


def _plain_positional(code: types.CodeType) -> bool:
    """Return whether ``code`` takes positional parameters only (no ``*``, ``**``, keywords)."""

    starred = code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS)
    return not (starred or code.co_kwonlyargcount)


def _declared_unread(fn: Any, row: KnownMaskBuilder) -> frozenset[int]:
    """Return the row's declared positions ``fn``'s signature and code bear out."""

    code = fn.__code__
    if not _plain_positional(code):
        return frozenset()
    positions = frozenset(p for p in row.unread if p < code.co_argcount)
    if row.composed is not None:
        return positions
    return frozenset(p for p in positions if not _references(code, code.co_varnames[p]))


def _composed_functions(fn: Any, cell_name: str) -> tuple[Any, ...] | None:
    """Return the non-empty tuple a composer's closure cell ``cell_name`` holds, else None."""

    code = fn.__code__
    closure = getattr(fn, "__closure__", None)
    if cell_name not in code.co_freevars or closure is None:
        return None
    try:
        value = closure[code.co_freevars.index(cell_name)].cell_contents
    except ValueError:  # an empty cell
        return None
    return value if isinstance(value, tuple) and value else None


def builder_unread_positions(fn: Any, depth: int = 0) -> frozenset[int]:
    """Return the positional arguments a recognized mask builder provably never reads.

    Parameters
    ----------
    fn:
        The callable a ``vmap`` boundary maps.
    depth:
        Composition depth already walked (callers pass 0).

    Returns
    -------
    frozenset[int]
        The declared-unread positions of a table builder whose code bears them out,
        intersected over every function a composer composes; empty when ``fn`` or any
        composed function is not in the table, or nesting exceeds ``_MAX_COMPOSITION_DEPTH``.
    """

    row = _table_row(fn)
    if row is None or depth > _MAX_COMPOSITION_DEPTH:
        return frozenset()
    unread = _declared_unread(fn, row)
    if row.composed is None or not unread:
        return unread
    composed = _composed_functions(fn, row.composed)
    if composed is None:
        return frozenset()
    for inner in composed:
        unread &= builder_unread_positions(inner, depth + 1)
    return unread


def _only_at(layer: Op, label: str, unread: frozenset[int]) -> bool:
    """Return whether ``label`` fills declared-unread positional slots only, and at least one."""

    positions = record_member(layer, "parent_arg_positions", None) or {}
    if label in (positions.get("kwargs", {}) or {}).values():
        return False
    keys = [key for key, parent in (positions.get("args", {}) or {}).items() if parent == label]
    return bool(keys) and all(isinstance(key, int) and key in unread for key in keys)


def check_mask_builder_unread_exempt(
    source_trace: Trace, layer: Op, layers_to_perturb: list[str]
) -> bool:
    """Custom check: excuse a vmapped mask builder's parent at a declared-unread position only.

    Parameters
    ----------
    source_trace:
        Trace being validated (unused; custom-check signature).
    layer:
        ``vmap`` boundary op being perturbation-checked.
    layers_to_perturb:
        Parent labels selected for perturbation.

    Returns
    -------
    bool
        True when the op maps a recognized builder composed only of table builders, the
        call is plain positional with the builder's arity, and every perturbed parent sits
        only at positions declared unread; False keeps the perturbation strict.
    """

    del source_trace
    if not layers_to_perturb:
        return False
    builder = vmapped_callable(getattr(layer, "func", None))
    code = getattr(builder, "__code__", None)
    if not isinstance(code, types.CodeType):
        return False
    if record_member(layer, "num_pos_args", None) != code.co_argcount:
        return False
    if record_member(layer, "num_kwargs", None) != 0:
        return False
    unread = builder_unread_positions(builder)
    if not unread:
        return False
    return all(_only_at(layer, label, unread) for label in layers_to_perturb)
