"""The ``do(selection, edit)`` plan builder for :mod:`torchlens._selection.selection`.

``torchlens._selection.selection.build_selection_do_plan`` opens the label-index scope
and calls :func:`_build_selection_do_plan` here, which resolves the target
(ACT selections only), derives the per-site masked edit and builds the audit
record. The normative mask-application contract is stated beside
``build_selection_do_plan``.

Fork lineage: a resolved target binds through
:func:`torchlens._selection._fork_binding.bind_to_target` (``Selection._resolve``),
so a selection resolved on a fork ancestor carries over by site identity.
"""

from __future__ import annotations

from typing import Any

import torch

from .._data_classes._trace_intervention_records import _edit_identity
from ..intervention._masked_edit import _derive_masked_edit
from ..intervention._subspace_projection import subspace_projection_of
from ..intervention._types import Edit
from .resolver import _label_index_scope
from .selection import (
    Selection,
    _apply_invalid,
    _find_act_ops,
    _lift,
    _unresolvable,
)


def _build_selection_do_plan(
    trace: Any, selection_like: Any, edit: Any
) -> tuple[Selection, list[dict[str, Any]], dict[str, Any]]:
    """Build :func:`build_selection_do_plan`'s result inside its label-index scope."""

    if edit is None:
        raise ValueError(
            "do(selection, edit) requires an edit: pass an Edit/Edit "
            "(e.g. tl.zero_ablate()), a hook callable, or a replacement tensor."
        )

    # A bare tl.subspace(...) target is projection-valued: the edit acts on
    # the component along the basis (resolve() still reports the support).
    projection = subspace_projection_of(selection_like)
    if projection is None and _edit_identity(edit, "") == "set_affine_coordinates":
        from ..intervention._affine_source import needs_subspace

        # Refuse at do(), before anything is staged: a [k, d] map needs a basis.
        raise needs_subspace(repr(selection_like), code="affine_source_needs_subspace")
    resolved = _resolve_do_target(trace, selection_like)

    if not isinstance(edit, Edit) and not callable(edit):
        # Raw replacement TENSORS route through the shipped replace_with helper and
        # obey the scatter contract; anything else refuses with the string-label
        # path's validation (a scalar lifted through the tensor-only helper used to
        # crash bare at push time).
        if not isinstance(edit, torch.Tensor):
            raise ValueError(
                f"do(selection, edit) got a {type(edit).__name__} replacement value; "
                "replacement values must be tensors matching the site output. For a "
                "constant fill over the selected elements pass a full-shape tensor "
                "(e.g. torch.full_like(site_out, c)), or use an edit helper such as "
                "tl.zero_ablate() or tl.scale()."
            )
        from ..intervention._predicates import replace_with

        edit = replace_with(edit)

    plan: list[dict[str, Any]] = []
    for entry in resolved:
        layer_label, pass_index = entry.key
        ops = [
            op
            for op in _find_act_ops(trace, layer_label)
            if getattr(op, "pass_number", 1) == pass_index
        ]
        if not ops:
            raise _unresolvable(
                "site_not_in_trace",
                f"resolved site {entry.key!r} is not present on this trace.",
                site=layer_label,
            )
        op = ops[0]
        site_edit = edit if projection is None else projection.derive(edit, entry, op.label)
        plan.append(
            {
                "op": op,
                "entry": entry,
                "is_leaf": getattr(op, "func", None) is None,
                "edit": _derive_masked_edit(site_edit, entry, resolved.digest, op.label),
            }
        )

    # The per-site "source" disclosure (the dense-subspace point-of-use
    # stamp) is admitted as an OPTIONAL string field in the closed
    # load-validator site schema as of tlspec v9 (the pre-v9
    # validator refused it, so a saved selection-intervened artifact refused
    # its own load).
    audit: dict[str, Any] = {
        "kind": resolved.kind,
        "selection_repr": repr(selection_like),
        "resolve_digest": resolved.digest,
        "sites": [
            {
                "site_key": repr(entry.key),
                "relation": entry.provenance.relation,
                "selected": entry.num_selected,
                "source": entry.provenance.source + ("" if projection is None else projection.note),
            }
            for entry in resolved
        ],
        "edit": _edit_identity(edit, repr(type(edit))),
    }
    source_identity = getattr(edit, "kwargs", None)
    if _edit_identity(edit, "") == "patch_from" and source_identity:
        audit["patch_source"] = dict(source_identity)
    return resolved, plan, audit


def _resolve_do_target(trace: Any, selection_like: Any) -> Selection:
    """Resolve the ``do()`` target on this trace, refusing PARAM/EDGE kinds."""

    lifted = _lift(selection_like)
    if lifted is None:
        raise RuntimeError("build_selection_do_plan requires a selection-shaped input")
    # A selection resolved on a fork ancestor carries over by site identity.
    resolved = lifted._resolve(trace)
    if resolved.kind == "PARAM":
        raise _apply_invalid(
            "not_maskable",
            "learned-parameter edits do not ride the node-edit plan: PARAM "
            "selections apply through parameter SUBSTITUTION on the replay "
            "engine (do(tl.param(...), edit) substitutes the value each "
            "consuming op sees; the live parameter is never written).",
        )
    if resolved.kind == "EDGE":
        raise _apply_invalid(
            "not_maskable",
            "edge selections address edge substitution (stage 3), not node edits.",
        )
    return resolved


def build_selection_do_plan(
    trace: Any, selection_like: Any, edit: Any
) -> tuple[Selection, list[dict[str, Any]], dict[str, Any]]:
    """Resolve a selection target and derive the per-site edit plan.

    Returns ``(resolved, [{"op", "entry", "is_leaf", "edit"}, ...], audit_record)``.
    ACT selections only: PARAM edits route through parameter substitution
    (``intervention/_param_substitution.py`` — the value each consumer sees is
    substituted on the replay engine, the live parameter is never written);
    EDGE selections are stage-3 edge-substitution territory.
    """

    # Building the plan adds, removes and relabels no op, so plain-label terms
    # resolve through one label index per trace for the whole plan.
    with _label_index_scope():
        return _build_selection_do_plan(trace, selection_like, edit)
