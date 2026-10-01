"""Edge storage/distribution semantics (F10; lovely item 8 / D28).

Two INDEPENDENT axes over one parent->child dataflow edge:

- ``view_or_copy`` -- the STORAGE relation, populated at the capture-time
  construction sites from the closed function-semantics table below. It
  can never authorize a distribution mark (``split`` is a VIEW of a
  DIFFERENT multiset while ``contiguous`` is a COPY of the SAME one).
- ``distribution_relation`` -- the VALUE claim behind the stats-table
  ``= parent`` mark: ``same_multiset(parent)`` holds only under a
  capture-path-stable function-semantics verdict corroborated by the
  persisted geometry. The verdict is a PURE FUNCTION of persisted per-op
  facts (child func name + shapes + parent arity), derived at read time --
  no new persisted field, no payload read, and ``data_ptr`` comparison is
  BANNED from the semantics (measured 2% recall with 12/13 hits being
  ``__add__`` false positives).

An absent mark is honest; a wrong mark is not. Every uncertain case
degrades to ``unknown`` / ``None`` (fail-closed).

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Functions whose output is DOCUMENTED to always be a torch view of the
#: tensor at args[0]. Conditional cases (``reshape``, ``flatten``,
#: ``contiguous``, ``__getitem__`` advanced indexing) are deliberately
#: EXCLUDED -- they may copy, so their edges stay ``unknown``.
ALWAYS_VIEW_FUNCS: frozenset[str] = frozenset(
    {
        "view",
        "view_as",
        "permute",
        "transpose",
        "t",
        "squeeze",
        "unsqueeze",
        "expand",
        "expand_as",
        "narrow",
        "select",
        "movedim",
        "moveaxis",
        "swapaxes",
        "swapdims",
        "detach",
        "unbind",
        "split",
        "split_with_sizes",
        "chunk",
        "diagonal",
        "adjoint",
        "as_strided",
        "real",
        "imag",
        "unfold",
    }
)

#: Functions whose output is DOCUMENTED to always be a fresh allocation
#: holding copied values of args[0].
ALWAYS_COPY_FUNCS: frozenset[str] = frozenset({"clone"})

#: Functions preserving the FULL value multiset from args[0] to the output
#: (the ``= parent`` candidates). Note the independence from the view set:
#: ``expand``/``split`` are views but CHANGE the multiset; ``clone`` and
#: (geometry-corroborated) ``reshape``/``flatten``/``ravel``/``contiguous``
#: copy or re-layout the SAME multiset.
SAME_MULTISET_FUNCS: frozenset[str] = frozenset(
    {
        "view",
        "view_as",
        "reshape",
        "reshape_as",
        "permute",
        "transpose",
        "t",
        "squeeze",
        "unsqueeze",
        "flatten",
        "ravel",
        "contiguous",
        "clone",
        "detach",
        "movedim",
        "moveaxis",
        "swapaxes",
        "swapdims",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
    }
)


@dataclass(frozen=True)
class EdgeDistributionRelation:
    """One derived distribution verdict with its proof provenance (D28).

    ``relation`` is currently the single value ``same_multiset``;
    ``basis`` names the function-semantics table row; ``corroboration``
    names the geometry check that agreed. Marks render ONLY from this
    record -- never from ``view_or_copy``, payload identity, or name
    heuristics.
    """

    relation: str
    parent_label: str
    basis: str
    corroboration: str


def classify_view_or_copy(func_name: str | None, arg_path: tuple[Any, ...]) -> str:
    """Storage relation for one edge from the closed function table.

    Only the tensor at positional slot 0 (the viewed/copied ``self``) can
    carry a verdict; every other edge of the call -- and every function
    outside the closed table -- stays ``unknown``.

    Parameters
    ----------
    func_name:
        The child call's function name (persisted fact).
    arg_path:
        The edge's argument path within the call.

    Returns
    -------
    str
        ``"view"`` / ``"copy"`` / ``"unknown"``.
    """

    if not func_name or tuple(arg_path) != (0,):
        return "unknown"
    if func_name in ALWAYS_VIEW_FUNCS:
        return "view"
    if func_name in ALWAYS_COPY_FUNCS:
        return "copy"
    return "unknown"


def _numel(shape: Any) -> int | None:
    """Element count from a persisted shape, or None when unavailable."""

    if shape is None:
        return None
    count = 1
    try:
        for dim in shape:
            count *= int(dim)
    except (TypeError, ValueError):
        return None
    return count


def distribution_relation(trace: Any, record: Any) -> EdgeDistributionRelation | None:
    """Derive one edge's distribution verdict from persisted facts (D28).

    Fail-closed: the verdict exists only when the child call's function is
    in the closed same-multiset table, the edge is the positional slot-0
    tensor, the child has exactly ONE tensor parent, and the persisted
    parent/child geometries agree on element count. Anything else returns
    ``None`` -- an absent mark, never a wrong one.

    Parameters
    ----------
    trace:
        The owning trace (resolves labels to per-op persisted facts).
    record:
        One ``EdgeUseRecord``.

    Returns
    -------
    EdgeDistributionRelation | None
        The corroborated verdict, or ``None``.
    """

    from ..utils.fail_open import fail_open

    if record.arg_kind != "positional" or tuple(record.arg_path) != (0,):
        return None
    child = fail_open(lambda: trace[record.child_label], lambda _error: None)
    func_name = getattr(child, "func_name", None)
    if child is None or func_name not in SAME_MULTISET_FUNCS:
        return None
    parents = tuple(getattr(child, "parents", ()) or ())
    if len(parents) != 1 or parents[0] != record.parent_label:
        return None
    parent = fail_open(lambda: trace[record.parent_label], lambda _error: None)
    child_numel = _numel(getattr(child, "shape", None))
    parent_numel = _numel(getattr(parent, "shape", None)) if parent is not None else None
    if child_numel is None or parent_numel is None or child_numel != parent_numel:
        return None
    return EdgeDistributionRelation(
        relation="same_multiset",
        parent_label=record.parent_label,
        basis=f"func_semantics:{func_name}",
        corroboration=f"numel_match:{child_numel}",
    )
