"""Edge substitution (L6 stage 3): replace the value CONSUMED on one edge.

Replaces the value consumed on edge ``(parent -> child, occurrence k)``:
only the child's consumption changes; the parent's stored output remains
producer truth. THE STORAGE DECISION (normative, three tiers on the
supported replay/push engine):

 (i)  SESSION: the substituted payload is spliced into the child's
      reconstructed call at exactly the occurrence address; the cone
      recompute starts FROM THE CHILD; rollback rides the transactional
      replay commit (extended to the tier-(ii) store).
 (ii) ARTIFACT, intervention-owned: the substituted consumed value lands in
      the occurrence-granular ``Op.edge_substitutions`` store on the CHILD
      (DROP-gated, pre-release-registered) with the save-time corroboration
      stamp in ``Op.edge_replacement_stamps`` — never in
      ``out_versions_by_child``, never in ``saved_args``/``saved_kwargs``,
      never in ``parent.out``.
 (iii) ARTIFACT, capture truth: ``parent.out``, ``saved_args``/
      ``saved_kwargs``, and any pre-existing ``out_versions_by_child`` entry
      are RETAINED UNMODIFIED — they ARE the pre-edit snapshot that makes
      divergence decidable (pinned by the capture-surface parity test).

ENGINE SCOPE: edge substitution ships on the replay/push engine ONLY.
``engine="rerun"`` (and ``"auto"`` resolving to rerun) or ``"set_only"``
refuse typed (``edge_intervention_engine_unsupported``): on rerun the
capture pipeline itself would snapshot the substituted value into
``out_versions_by_child``/``saved_args`` (the indistinguishable-artifact
failure), and the capture-side fork is a NAMED FUTURE escalated to
capture-owner routing.

CONVENIENCE-FIELD COHERENCE: the consumer view (what did THIS consumer
receive) reads from the tier-(ii) store at the occurrence address and
carries the per-edge intervened marker (FireRecord.edge_address); the
producer view (what did the producer compute) stays ``parent.out``,
untouched. Neither surface lies; the edge provenance record is the bridge.

Node-level ``intervention_replaced`` is deliberately NOT set: edge
substitution replaces no node's output (the child's output is DERIVED from
the substituted input by re-execution), so the node-level corroboration key
correctly never fires for edges.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import TYPE_CHECKING, Any

import torch

from ..selection import (
    ResolvedSelection,
    SelectionError,
    _apply_invalid,
    _require_edge_provenance,
    _trace_edge_records,
    edge_address_of,
)
from .masked_edit import _validate_edited
from .types import HelperSpec

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

__all__ = ["apply_edge_substitution_do"]


def tensor_content_digest(value: torch.Tensor) -> str:
    """Return the sha256 of a tensor's raw element bytes (dtype-agnostic).

    The bytes are read through a flat ``uint8`` view of the contiguous
    payload, so the digest is byte-identical to the historical
    ``numpy().tobytes()`` spelling for every numpy-representable dtype and
    additionally covers dtypes numpy cannot represent (``bfloat16``,
    ``float8``). Two tensors of the same shape and dtype digest equal iff
    they are bitwise equal (NaN payloads included); the digest carries no
    shape or dtype, so callers compare digests only within one site.
    """

    payload = value.detach().cpu().contiguous()
    if payload.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    raw = payload.view(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _value_digest(value: torch.Tensor) -> str:
    """Return a stable content digest for one substituted payload."""

    return tensor_content_digest(value)


def require_depth1_arg_path(arg_path: Any, *, where: str, site: Any) -> None:
    """TRIPWIRE: refuse a nested store key at a top-level splice site.

    The v1 splice sites substitute at ``arg_path[0]`` directly. A nested
    (depth-2+) key reaching one of them would silently replace the WHOLE
    top-level argument at the wrong address -- well-formed, audited, wrong.
    The entry gate refuses nested addresses at construction, so an entry
    here is a foreign, forged, or future-schema store row; the
    nested-container splice is a named future that must extend every splice
    site together, and until it does each site refuses rather than guesses.
    """

    if len(tuple(arg_path)) == 1:
        return
    raise _apply_invalid(
        "not_maskable",
        f"edge-substitution store entry with nested argument path "
        f"{tuple(arg_path)!r} reached the top-level splice ({where}); v1 "
        "splices address top-level arguments only and never guess a nested "
        "address. The entry cannot be applied faithfully -- it is a foreign "
        "or future-schema store row.",
        code="selection_apply_invalid",
        site=site,
        where=where,
        arg_path=tuple(arg_path),
    )


def _consumed_value(trace: Any, parent_op: Any, child_op: Any) -> torch.Tensor:
    """Return the value the child currently consumes on this edge.

    Pass-qualified first: ``out_versions_by_child`` is keyed by the child's
    own op label, and the bare ``layer_label`` of a multi-pass child names
    its LAST pass, so it is consulted only as the single-pass fallback.
    """

    versions = getattr(parent_op, "out_versions_by_child", None) or {}
    version = versions.get(getattr(child_op, "label", None))
    if not isinstance(version, torch.Tensor):
        version = versions.get(child_op.layer_label)
    if isinstance(version, torch.Tensor):
        return version
    out = parent_op.out
    if not isinstance(out, torch.Tensor):
        raise _apply_invalid(
            "not_maskable",
            f"edge parent {parent_op.label!r} has no tensor value to substitute.",
            site=parent_op.label,
        )
    return out


def _current_edge_value(
    trace: Any, parent_op: Any, child_op: Any, store_key: tuple[Any, ...]
) -> torch.Tensor:
    """Return the value the child consumes at this occurrence RIGHT NOW.

    Edits COMPOSE like replay hooks: a later edge edit at an occurrence that
    already carries a tier-(ii) substitution derives from the substituted
    value (``zero_ablate`` then ``scale(0.5)`` stays zero; ``scale(0.5)``
    twice yields a quarter), never from a stale pre-edit snapshot that the
    store no longer asserts. Without an entry the consumed value is the
    parent's current out (pushed upstream edits included).
    """

    entries = getattr(child_op, "edge_substitutions", None) or {}
    existing = entries.get(store_key)
    if isinstance(existing, dict):
        value = existing.get("value")
        if isinstance(value, torch.Tensor):
            return value
    return _consumed_value(trace, parent_op, child_op)


def _edit_hook(edit: Any, site_label: str) -> tuple[Any, HelperSpec | None, str]:
    """Normalize the edit into a hook callable + disclosure identity."""

    if isinstance(edit, HelperSpec):
        if edit.factory is None:
            raise _apply_invalid(
                "not_maskable",
                f"edit {edit.helper_name!r} has no runtime factory.",
                site=site_label,
            )
        return edit.factory(), edit, edit.helper_name
    if callable(edit):
        return edit, None, getattr(edit, "__name__", "hook")
    if not isinstance(edit, torch.Tensor):
        # Same validation as the string-label path: lifting a scalar through
        # the tensor-only replace_with helper used to crash bare at fire time.
        raise _apply_invalid(
            "not_maskable",
            f"do(selection, edit) got a {type(edit).__name__} replacement "
            "value; replacement values must be tensors matching the consumed "
            "value. For a constant fill pass a full-shape tensor (e.g. "
            "torch.full_like(consumed, c)) or an edit helper such as "
            "tl.zero_ablate() or tl.scale().",
            code="selection_apply_invalid",
            site=site_label,
        )
    from .predicates import replace_with

    spec = replace_with(edit)
    if spec.factory is None:
        raise RuntimeError("replace_with spec lost its factory")
    return spec.factory(), spec, spec.helper_name


def apply_edge_substitution_do(
    trace: Trace,
    resolved: ResolvedSelection,
    edit: Any,
    *,
    engine: str,
    strict: bool,
) -> dict[str, Any]:
    """Apply one edge-substitution edit to a resolved EDGE selection.

    Returns the audit payload extension (per-edge addresses + digests).

    Raises
    ------
    SelectionError
        ``edge_intervention_engine_unsupported`` off the replay engine;
        ``edge_provenance_unavailable`` without an intervention_ready
        capture; ``selection_apply_invalid`` for ineligible payloads.
    """

    if engine != "replay":
        raise SelectionError(
            f"edge substitution ships on the replay/push engine only; "
            f"engine={engine!r} is unsupported (on rerun the capture pipeline "
            "would snapshot the substituted value into captured consumed-value "
            "fields — the capture-side design is a named future). Use "
            "engine='replay'.",
            code="edge_intervention_engine_unsupported",
            engine=engine,
        )
    _require_edge_provenance(trace)

    import importlib

    replay_module = importlib.import_module("torchlens.intervention.replay")
    hooks_module = importlib.import_module("torchlens.intervention.hooks")

    family = {edge_address_of(record): record for record in _trace_edge_records(trace)}
    applied: list[dict[str, Any]] = []
    committed: list[tuple[Any, tuple[Any, ...]]] = []
    try:
        for entry in resolved:
            if entry.selected_count == 0:
                continue
            address = entry.site_key
            record = family.get(tuple(address))
            if record is None:
                raise SelectionError(
                    f"edge occurrence {address!r} is not part of this trace's "
                    "dataflow edge family.",
                    code="selection_unresolvable",
                    reason="site_not_in_trace",
                    site=repr(address),
                )
            child_op = trace.layer_dict_all_keys[record.child_label].ops[0]
            parent_op = trace.layer_dict_all_keys[record.parent_label].ops[0]
            child_func_call_id, arg_kind, arg_path = address
            if len(arg_path) != 1:
                raise _apply_invalid(
                    "not_maskable",
                    f"edge occurrence {address!r} has a nested argument path; "
                    "v1 edge substitution addresses top-level arguments only.",
                    site=record.child_label,
                )

            store_key = (arg_kind, tuple(arg_path))
            consumed = _current_edge_value(trace, parent_op, child_op, store_key)
            hook_callable, helper_spec, helper_name = _edit_hook(edit, record.child_label)
            context = hooks_module.make_hook_context(
                name=helper_name,
                timing="post",
                direction="forward",
                layer_log=child_op,
                run_ctx={},
                args=(consumed,),
                kwargs={},
            )
            substituted = hook_callable(consumed, hook=context)
            substituted = _validate_edited(substituted, consumed, record.child_label)

            new_out = _reexecute_child_with_substitution(
                trace, child_op, address, substituted, strict=strict
            )

            value_digest = _record_edge_substitution(
                child_op,
                store_key,
                substituted,
                meta={
                    "parent_label": record.parent_label,
                    "resolve_digest": resolved.resolve_digest,
                    "helper_name": helper_name,
                },
            )
            committed.append((child_op, store_key))

            from .audit import build_fire_record

            fire_record = build_fire_record(
                target_label=child_op.layer_label,
                call_label=child_op.label,
                func_call_id=child_op.func_call_id,
                container_path=tuple(child_op.container_path or ()),
                engine="replay",
                helper=(
                    dataclasses.replace(
                        helper_spec,
                        selection_recipe={
                            "resolve_digest": resolved.resolve_digest,
                            "edge_address": repr(tuple(address)),
                        },
                    )
                    if helper_spec is not None
                    else None
                ),
                site_label=child_op.layer_label,
                timing="post",
                direction="forward",
                helper_name=helper_name,
                # Edge substitution replaces no node's OUTPUT: the child's
                # output is derived by re-execution, so the node-level
                # corroboration key must not fire.
                replaced=False,
                edge_address=tuple(address),
            )
            # PASS-QUALIFIED commit key (AUD-CODE 1.2): the bare
            # ``layer_label`` of a multi-pass child resolves to its LAST
            # pass, so an edge into pass k < N committed the new out and the
            # FireRecord onto pass N while ``push_from`` then pushed pass k's
            # UNCHANGED captured out -- a silent no-op whose audit row still
            # claimed success. ``_replay_site_key`` is the engine's own key.
            child_key = replay_module._replay_site_key(child_op)
            replay_module._commit_replay_updates(
                trace,
                {child_key: new_out},
                {child_key: [fire_record]},
            )
            replay_module.push_from(trace, child_op)
            applied.append(
                {
                    "edge_address": repr(tuple(address)),
                    "parent": record.parent_label,
                    "child": record.child_label,
                    "value_digest": value_digest,
                }
            )
    except Exception:
        _rollback_uncommitted_edges(committed, applied)
        raise
    return {"edges": applied}


def _reexecute_child_with_substitution(
    trace: Trace, child_op: Any, address: Any, substituted: Any, *, strict: bool
) -> Any:
    """SESSION splice: rebuild the child's call from capture truth, substitute
    at exactly the occurrence address, re-execute, and slice its output."""

    import importlib

    replay_module = importlib.import_module("torchlens.intervention.replay")
    _child_func_call_id, arg_kind, arg_path = address
    template = replay_module._template_for_site(child_op)
    args, kwargs = replay_module._reconstruct_args_from_template(
        template, child_op, trace, {}, strict=strict
    )
    # Composition coherence: every tier-(ii) entry the child already carries
    # (param, region, AND earlier edge substitutions) is re-spliced first, so
    # a later edge edit never silently reverts a sibling occurrence's edit;
    # the edge splice below overwrites its OWN occurrence last if they
    # collide.
    args, kwargs = replay_module._splice_param_substitutions([child_op], args, kwargs)
    require_depth1_arg_path(
        arg_path,
        where="edge re-execution splice",
        site=getattr(child_op, "label", None),
    )
    if arg_kind == "positional":
        position = int(arg_path[0])
        args = args[:position] + (substituted,) + args[position + 1 :]
    else:
        kwargs = dict(kwargs)
        kwargs[arg_path[0]] = substituted
    output = replay_module._execute_replay_func_strict(child_op, args, kwargs)
    # Path resolution shares the replay engine's output contract: a boundary
    # output node's recorded container_path addresses the MODEL's return
    # container, never the re-executed call's output (edge records do not
    # currently reach synthesized boundary nodes, but the contract lives in
    # one place).
    return replay_module._slice_output_by_path(
        output, replay_module._replay_container_path(child_op, trace)
    )


def _record_edge_substitution(
    child_op: Any, store_key: tuple[Any, ...], substituted: Any, *, meta: dict[str, Any]
) -> Any:
    """Tier (ii): write the occurrence-granular intervention-owned store entry
    with the save-time corroboration stamp; return the stamped value digest."""

    store = dict(getattr(child_op, "edge_substitutions", None) or {})
    store[store_key] = {
        "value": substituted.detach().clone(),
        "parent_label": meta["parent_label"],
        "resolve_digest": meta["resolve_digest"],
        "helper_name": meta["helper_name"],
    }
    value_digest = _value_digest(substituted)
    stamps = dict(getattr(child_op, "edge_replacement_stamps", None) or {})
    stamps[store_key] = {
        "verdict": True,
        "value_digest": value_digest,
        "resolve_digest": meta["resolve_digest"],
    }
    child_op._internal_set("edge_substitutions", store)
    child_op._internal_set("edge_replacement_stamps", stamps)
    return value_digest


def _rollback_uncommitted_edges(
    committed: list[tuple[Any, tuple[Any, ...]]], applied: list[dict[str, Any]]
) -> None:
    """Roll back tier-(ii) entries for occurrences that did not complete."""

    for child_op, store_key in committed:
        store = dict(getattr(child_op, "edge_substitutions", None) or {})
        stamps = dict(getattr(child_op, "edge_replacement_stamps", None) or {})
        if not any(
            item["edge_address"] == repr((child_op.func_call_id,) + store_key) for item in applied
        ):
            store.pop(store_key, None)
            stamps.pop(store_key, None)
            child_op._internal_set("edge_substitutions", store or None)
            child_op._internal_set("edge_replacement_stamps", stamps or None)
