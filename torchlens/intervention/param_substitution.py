"""Parameter substitution: apply edits "as if" a parameter were changed.

JMT ruling 2026-08-17 (supersedes the D3 typed-refusal default on the replay
path): do not change the weights at all — substitute the VALUE THE OPERATION
CONSUMES at each parameter->op occurrence, leaving the parameter object
untouched. Nothing mutates persistent model state, so there is nothing to
snapshot, restore, or attest about model state, and the rejected
differentiable-provider design's save_budget-invisible autograd retention
never exists. The scope is honestly replay-only, never a training-time
capability.

MECHANISM: a parameter is an input operand of its consuming operations. The
shipped L6 stage-3 edge-substitution engine already replaces the value
consumed at one occurrence address ``(child_func_call_id, arg_kind,
arg_path)`` on the replay/push engine. Parameters are NOT in the dataflow
edge family (EdgeUseRecords cover tensor-op parents only; parameters
classify as ``LiteralTensor`` template components carrying a param barcode),
so this module DERIVES the occurrence addresses for a PARAM selection —
``Param.used_by_ops`` reverse map + template-component identity/barcode
match, FAIL-CLOSED (a consumption the engine cannot address refuses typed,
never a silent partial "as if") — and drives the same engine. Recurrently
reused parameters (tied weights, a module called at several passes of a
recurrence-grouped layer) are substitutable: consumers are recorded and
staged PASS-QUALIFIED (``label:pass``), the edit lands at EVERY consumption
(a parameter has one identity across passes), and the pass-qualified replay
engine recomputes each pass faithfully. A bare multi-pass consumer
spelling or a consumer inventory missing a pass still refuses (they would
silently cover a subset of consumptions):

* tier-(ii) store: substituted values land in ``Op.edge_substitutions`` /
  ``Op.edge_replacement_stamps`` at the derived occurrence address, marked
  ``substitution_kind="param"`` with the parameter address. The save-time
  erasure-prevention invariant and the validation boundary check
  (re-execute-from-splice, never skip) apply unchanged.
* propagation: ONE replay pass over all consumer origins; cone
  recomputation re-splices param-kind entries at argument reconstruction
  (:func:`torchlens.intervention.replay._splice_param_substitutions`), so
  multi-consumer and nested-consumer parameters see the substituted value
  everywhere in the cone. Edge-kind entries keep their shipped
  no-re-splice semantics (parity-pinned).
* honesty: the audit record and per-occurrence FireRecords disclose that
  the parameter was SUBSTITUTED AT CONSUMPTION, never changed; the live
  parameter is bit-identical before and after (pinned by test); the
  intervened product stays DERIVED-class (fork outcomes settle UNATTESTED,
  and validation reports the distinct ``edge_intervention_boundary``
  verdict, never a plain VERIFIED pass).

ENGINE SCOPE: replay/push only. ``engine="rerun"`` (including ``"auto"``
resolving to rerun) and ``"set_only"`` refuse typed
(``param_substitution_engine_unsupported``): on rerun the capture pipeline
would snapshot the substituted value into captured consumed-value fields
(the same indistinguishable-artifact failure the edge engine refuses), and
set_only has no consumption to substitute.

Every spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import torch

from ..selection import (
    ResolvedSelection,
    SelectionError,
    SiteEntry,
    _apply_invalid,
)
from .masked_edit import _derive_masked_edit, _validate_edited
from .types import FireRecord, HelperSpec, LiteralTensor

if TYPE_CHECKING:
    from ..data_classes.param import Param
    from ..data_classes.trace import Trace

__all__ = ["apply_param_substitution_do"]

#: Marker distinguishing param-derived tier-(ii) entries from edge-selection
#: entries in the shared ``Op.edge_substitutions`` store. The replay
#: re-splice and nothing else branches on it; validation treats both kinds
#: identically (re-execute-from-splice).
PARAM_SUBSTITUTION_KIND = "param"


def _engine_refusal(engine: str) -> SelectionError:
    """Build the typed off-replay-engine refusal (teaching refusal)."""

    return SelectionError(
        "parameter substitution ships on the replay/push engine only: the "
        "edit is applied to the value each operation CONSUMES, 'as if' the "
        "parameter were changed, and the live parameter object is never "
        f"written. engine={engine!r} is unsupported (on rerun the capture "
        "pipeline would snapshot the substituted value into captured "
        "consumed-value fields; set_only has no consumption to substitute). "
        "Use fork.do(tl.params(...), edit) with engine='replay' (the "
        "default when no model/x is passed).",
        code="param_substitution_engine_unsupported",
        engine=engine,
    )


def _underivable(message: str, **fields: Any) -> SelectionError:
    """Build the typed occurrence-derivation refusal (fail-closed)."""

    return SelectionError(
        message + " Parameter substitution addresses every consumption of the "
        "parameter or none (a silent partial 'as if' would misreport the "
        "intervention).",
        code="param_substitution_occurrence_underivable",
        **fields,
    )


def _param_for_entry(trace: Trace, entry: SiteEntry) -> Param:
    """Return the Param record addressed by one resolved PARAM entry."""

    address = entry.site_key[0]
    for param in getattr(trace, "params", ()):
        if getattr(param, "address", None) == address or param.name == address:
            return param
    raise _underivable(
        f"parameter {address!r} is not recorded on this trace.",
        param_address=address,
    )


def _live_param_identity(param: Param) -> tuple[torch.Tensor | None, str | None]:
    """Return the live parameter reference and/or barcode for matching."""

    live = getattr(param, "_param_ref", None)
    if getattr(param, "_param_ref_released", False):
        live = None
    if not isinstance(live, torch.Tensor):
        live = None
    barcode = getattr(param, "barcode", None)
    return live, str(barcode) if barcode is not None else None


def _component_matches(
    component: Any, live_param: torch.Tensor | None, barcode: str | None
) -> bool:
    """Return whether one top-level template component is this parameter."""

    if not isinstance(component, LiteralTensor):
        return False
    if live_param is not None and component.value is live_param:
        return True
    component_barcode = getattr(component, "param_barcode", None)
    return barcode is not None and component_barcode is not None and component_barcode == barcode


def _nested_component_matches(
    component: Any, live_param: torch.Tensor | None, barcode: str | None
) -> bool:
    """Return whether the parameter hides inside a nested container component."""

    if _component_matches(component, live_param, barcode):
        return True
    if isinstance(component, tuple):
        return any(
            _nested_component_matches(
                item[1]
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                else item,
                live_param,
                barcode,
            )
            for item in component
        )
    return False


def _param_occurrences_for_op(
    op: Any,
    template: Any,
    live_param: torch.Tensor | None,
    barcode: str | None,
    param_address: str,
) -> list[tuple[str, tuple[Any, ...], torch.Tensor]]:
    """Derive this op's occurrence addresses for one parameter, fail-closed.

    Returns ``(arg_kind, arg_path, matched_value)`` triples: the matched
    template component's ``LiteralTensor.value`` is the capture-time live
    parameter, which doubles as the consumed-value source (postprocess
    releases ``Param._param_ref``, so the template is the surviving handle).
    """

    occurrences: list[tuple[str, tuple[Any, ...], torch.Tensor]] = []
    for position, component in enumerate(template.args):
        if _component_matches(component, live_param, barcode):
            occurrences.append(("positional", (position,), component.value))
        elif _nested_component_matches(component, live_param, barcode):
            raise _underivable(
                f"parameter {param_address!r} is consumed inside a nested "
                f"container argument of {op.label!r}; v1 parameter "
                "substitution addresses top-level arguments only.",
                param_address=param_address,
                site=op.label,
            )
    for key, component in template.kwargs:
        if _component_matches(component, live_param, barcode):
            occurrences.append(("keyword", (key,), component.value))
        elif _nested_component_matches(component, live_param, barcode):
            raise _underivable(
                f"parameter {param_address!r} is consumed inside a nested "
                f"container argument of {op.label!r}; v1 parameter "
                "substitution addresses top-level arguments only.",
                param_address=param_address,
                site=op.label,
            )
    if not occurrences:
        raise _underivable(
            f"consumer {op.label!r} of parameter {param_address!r} has no "
            "matching top-level template component (identity and barcode "
            "matching both failed).",
            param_address=param_address,
            site=op.label,
        )
    return occurrences


def _substituted_value(
    entry: SiteEntry,
    edit: Any,
    resolve_digest: str,
    consumed: torch.Tensor,
    site_label: str,
) -> tuple[torch.Tensor, HelperSpec | None, str]:
    """Compute the substituted value under the edit-then-scatter contract.

    The hook receives a detached CLONE of the current parameter value, so
    even a hostile in-place edit callable cannot write the live parameter
    through this path.
    """

    import importlib

    edge_module = importlib.import_module("torchlens.intervention.edge_substitution")
    hooks_module = importlib.import_module("torchlens.intervention.hooks")

    masked_edit = _derive_masked_edit(edit, entry, resolve_digest, site_label)
    hook_callable, helper_spec, helper_name = edge_module._edit_hook(masked_edit, site_label)
    context = hooks_module.make_hook_context(
        name=helper_name,
        timing="post",
        direction="forward",
        layer_log=None,
        run_ctx={},
        args=(consumed,),
        kwargs={},
    )
    substituted = hook_callable(consumed, hook=context)
    substituted = _validate_edited(substituted, consumed, site_label)
    return substituted, helper_spec, helper_name


@dataclasses.dataclass
class _Staging:
    """Mutable staging context threaded through one apply transaction."""

    applied_params: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    committed: list[tuple[Any, tuple[Any, ...]]] = dataclasses.field(default_factory=list)
    pending_records: dict[str, list[FireRecord]] = dataclasses.field(default_factory=dict)
    origin_ops: dict[int, Any] = dataclasses.field(default_factory=dict)


def apply_param_substitution_do(
    trace: Trace,
    resolved: ResolvedSelection,
    edit: Any,
    *,
    engine: str,
    strict: bool,
) -> dict[str, Any]:
    """Apply one PARAM-selection edit "as if" the parameter were changed.

    Every consumption of each selected parameter is substituted at its
    derived occurrence address through the shipped tier-(ii) substitution
    store, then ONE replay pass recomputes the union cone of all consumers
    (re-splicing param entries during argument reconstruction). The live
    parameter object is never written.

    Returns the audit payload extension (per-parameter consumers +
    occurrence addresses + value digests + the substitution disclosure).

    Raises
    ------
    SelectionError
        ``param_substitution_engine_unsupported`` off the replay engine;
        ``param_substitution_occurrence_underivable`` when any consumption
        cannot be addressed (fail-closed); ``selection_apply_invalid`` for
        ineligible payloads.
    """

    if engine != "replay":
        raise _engine_refusal(engine)

    import importlib

    replay_module = importlib.import_module("torchlens.intervention.replay")
    replay_module._preflight_log(trace)

    if not isinstance(edit, HelperSpec) and not callable(edit):
        if not isinstance(edit, torch.Tensor):
            # Same validation as the string-label path: lifting a scalar
            # through the tensor-only replace_with helper used to crash
            # bare at fire time.
            raise _apply_invalid(
                "not_maskable",
                f"do(selection, edit) got a {type(edit).__name__} replacement "
                "value; replacement values must be tensors matching the "
                "parameter. For a constant fill pass a full-shape tensor "
                "(e.g. torch.full_like(param, c)) or an edit helper such as "
                "tl.zero_ablate() or tl.scale().",
                code="selection_apply_invalid",
            )
        from .predicates import replace_with

        edit = replace_with(edit)

    staging = _Staging()
    try:
        for entry in resolved:
            if entry.selected_count == 0:
                continue
            _stage_param_entry(trace, entry, edit, resolved.resolve_digest, staging)
        _propagate_staged(trace, staging, strict=strict)
    except Exception:
        _rollback_param_substitutions(staging.committed)
        raise
    return {
        "params": staging.applied_params,
        "disclosure": (
            "parameter values substituted at consumption for replay; live parameters unchanged"
        ),
    }


def _stage_param_entry(
    trace: Trace,
    entry: SiteEntry,
    edit: Any,
    resolve_digest: str,
    staging: _Staging,
) -> None:
    """Derive one entry's occurrences, compute its value, stage the stores."""

    param_address = entry.site_key[0]
    occurrences_by_op = _derive_entry_occurrences(trace, entry)
    matched_values = [
        value for _, _, occurrences in occurrences_by_op for _, _, value in occurrences
    ]
    first_value = matched_values[0]
    if any(value is not first_value for value in matched_values[1:]):
        raise _underivable(
            f"parameter {param_address!r} matched DISTINCT tensor objects "
            "across its consumers; the current value is ambiguous.",
            param_address=param_address,
        )
    with torch.no_grad():
        consumed = first_value.detach().clone()
    substituted, helper_spec, helper_name = _substituted_value(
        entry, edit, resolve_digest, consumed, param_address
    )
    param_occurrences: list[dict[str, Any]] = []
    for child_op, consumer_label, occurrences in occurrences_by_op:
        for arg_kind, arg_path, _matched in occurrences:
            record = _stage_occurrence(
                child_op,
                (arg_kind, tuple(arg_path)),
                substituted,
                meta={
                    "param_address": param_address,
                    "resolve_digest": resolve_digest,
                    "helper_name": helper_name,
                    "helper_spec": helper_spec,
                },
            )
            staging.committed.append((child_op, (arg_kind, tuple(arg_path))))
            # Pass-qualified key: the bare layer_label of a multi-pass layer
            # resolves to its LAST pass, which would land every pass's
            # FireRecord on one op.
            record_key = getattr(child_op, "label", None) or child_op.layer_label
            staging.pending_records.setdefault(record_key, []).append(record["fire_record"])
            param_occurrences.append(
                {
                    "edge_address": record["edge_address"],
                    "consumer": consumer_label,
                    "value_digest": record["value_digest"],
                }
            )
        staging.origin_ops[id(child_op)] = child_op
    staging.applied_params.append(
        {
            "param_address": param_address,
            "consumers": [label for _, label, _ in occurrences_by_op],
            "occurrences": param_occurrences,
        }
    )


def _derive_entry_occurrences(
    trace: Trace, entry: SiteEntry
) -> list[tuple[Any, str, list[tuple[str, tuple[Any, ...], torch.Tensor]]]]:
    """Resolve one PARAM entry to ``(op, consumer_label, occurrences)`` rows."""

    import importlib

    replay_module = importlib.import_module("torchlens.intervention.replay")
    param = _param_for_entry(trace, entry)
    param_address = entry.site_key[0]
    live_param, barcode = _live_param_identity(param)
    if live_param is None and barcode is None:
        raise _underivable(
            f"parameter {param_address!r} has neither a live reference nor a "
            "barcode on this trace (released or legacy capture); its "
            "consumptions cannot be identified.",
            param_address=param_address,
        )
    consumers = list(getattr(param, "used_by_ops", ()) or ())
    if not consumers:
        raise _underivable(
            f"parameter {param_address!r} records no consuming ops on this trace.",
            param_address=param_address,
        )
    rows: list[tuple[Any, str, list[tuple[str, tuple[Any, ...], torch.Tensor]]]] = []
    staged_op_ids: set[int] = set()
    for consumer_label in consumers:
        layer = trace.layer_dict_all_keys.get(consumer_label)
        if layer is None or not getattr(layer, "ops", None):
            raise _underivable(
                f"consumer {consumer_label!r} of parameter {param_address!r} "
                "is not present on this trace.",
                param_address=param_address,
                site=consumer_label,
            )
        child_op = layer.ops[0]
        if (
            int(getattr(child_op, "num_passes", 1) or 1) > 1
            and consumer_label == child_op.layer_label
        ):
            # A bare layer label of a multi-pass layer resolves to its LAST
            # pass only; staging from it would silently substitute a subset
            # of the parameter's consumptions.
            raise _underivable(
                f"consumer spelling {consumer_label!r} of parameter "
                f"{param_address!r} is the bare label of a multi-pass "
                "(recurrence-grouped) layer, which addresses only its last "
                "pass; every consuming pass must be recorded pass-qualified "
                "(``label:pass``) to be substitutable.",
                param_address=param_address,
                site=consumer_label,
            )
        if id(child_op) in staged_op_ids:
            continue
        staged_op_ids.add(id(child_op))
        template = replay_module._template_for_site(child_op)
        rows.append(
            (
                child_op,
                consumer_label,
                _param_occurrences_for_op(child_op, template, live_param, barcode, param_address),
            )
        )
    _check_multipass_coverage(rows, param_address)
    return rows


def _check_multipass_coverage(
    rows: list[tuple[Any, str, list[tuple[str, tuple[Any, ...], torch.Tensor]]]],
    param_address: str,
) -> None:
    """Refuse unless every pass of each multi-pass consumer layer is addressed.

    Recurrence grouping guarantees that all passes of a multi-pass layer are
    equivalent ops consuming the same parameters, so a consumer inventory
    that names only SOME passes of a layer is incoherent metadata — staging
    the named subset would be a silent partial "as if" (fail-closed).
    """

    covered: dict[str, set[int]] = {}
    total_passes: dict[str, int] = {}
    for child_op, _, _ in rows:
        num_passes = int(getattr(child_op, "num_passes", 1) or 1)
        if num_passes > 1:
            covered.setdefault(child_op.layer_label, set()).add(
                int(getattr(child_op, "pass_index", 0) or 0)
            )
            total_passes[child_op.layer_label] = num_passes
    for bare_label, seen in covered.items():
        missing = set(range(1, total_passes[bare_label] + 1)) - seen
        if missing:
            raise _underivable(
                f"parameter {param_address!r} is consumed by multi-pass layer "
                f"{bare_label!r}, but its recorded consumers omit "
                f"pass(es) {sorted(missing)}; substituting only the addressed "
                "passes would be a silent partial 'as if'.",
                param_address=param_address,
                site=bare_label,
            )


def _stage_occurrence(
    child_op: Any,
    store_key: tuple[Any, ...],
    substituted: torch.Tensor,
    *,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Stage one occurrence's tier-(ii) entry, stamp, and FireRecord."""

    import importlib

    edge_module = importlib.import_module("torchlens.intervention.edge_substitution")
    param_address = meta["param_address"]
    value_digest = edge_module._record_edge_substitution(
        child_op,
        store_key,
        substituted,
        meta={
            "parent_label": param_address,
            "resolve_digest": meta["resolve_digest"],
            "helper_name": meta["helper_name"],
        },
    )
    _stamp_param_kind(child_op, store_key, param_address)
    address = (child_op.func_call_id,) + tuple(store_key)
    fire_record = _param_fire_record(
        child_op,
        meta["helper_spec"],
        meta["helper_name"],
        address,
        meta={"param_address": param_address, "resolve_digest": meta["resolve_digest"]},
    )
    return {
        "edge_address": repr(address),
        "value_digest": value_digest,
        "fire_record": fire_record,
    }


def _propagate_staged(trace: Trace, staging: _Staging, *, strict: bool) -> None:
    """Run the ONE replay pass over all consumer origins, then attach records."""

    if not staging.origin_ops:
        return
    import importlib

    replay_module = importlib.import_module("torchlens.intervention.replay")
    replay_module._run_replay(
        trace,
        list(staging.origin_ops.values()),
        hook_entries=[],
        strict=strict,
        preserve_origins=False,
    )
    for label, records in staging.pending_records.items():
        trace.layer_dict_all_keys[label].ops[0].interventions.extend(records)


def _stamp_param_kind(child_op: Any, store_key: tuple[Any, ...], param_address: str) -> None:
    """Mark one tier-(ii) entry + stamp as param-derived (drives the re-splice)."""

    store = dict(getattr(child_op, "edge_substitutions", None) or {})
    payload = dict(store[store_key])
    payload["substitution_kind"] = PARAM_SUBSTITUTION_KIND
    payload["param_address"] = param_address
    store[store_key] = payload
    stamps = dict(getattr(child_op, "edge_replacement_stamps", None) or {})
    stamp = dict(stamps[store_key])
    stamp["substitution_kind"] = PARAM_SUBSTITUTION_KIND
    stamp["param_address"] = param_address
    stamps[store_key] = stamp
    child_op._internal_set("edge_substitutions", store)
    child_op._internal_set("edge_replacement_stamps", stamps)


def _param_fire_record(
    child_op: Any,
    helper_spec: HelperSpec | None,
    helper_name: str,
    address: tuple[Any, ...],
    meta: dict[str, Any],
) -> FireRecord:
    """Mint the per-occurrence FireRecord (discloses substitution, not change)."""

    return FireRecord(
        target_label=child_op.layer_label,
        call_label=child_op.label,
        func_call_id=child_op.func_call_id,
        container_path=tuple(child_op.container_path or ()),
        engine="replay",
        helper=(
            dataclasses.replace(
                helper_spec,
                selection_recipe={
                    "resolve_digest": meta["resolve_digest"],
                    "edge_address": repr(tuple(address)),
                    "param_address": meta["param_address"],
                    "note": (
                        "parameter substituted at consumption for replay; live parameter unchanged"
                    ),
                },
            )
            if helper_spec is not None
            else None
        ),
        site_label=child_op.layer_label,
        timing="post",
        direction="forward",
        helper_name=helper_name,
        # Parameter substitution replaces no node's OUTPUT: the child's
        # output is derived by re-execution, so the node-level
        # corroboration key must not fire (same rule as edge substitution).
        replaced=False,
        edge_address=tuple(address),
    )


def _rollback_param_substitutions(committed: list[tuple[Any, tuple[Any, ...]]]) -> None:
    """Strip staged tier-(ii) entries after a failed apply."""

    for child_op, store_key in committed:
        store = dict(getattr(child_op, "edge_substitutions", None) or {})
        stamps = dict(getattr(child_op, "edge_replacement_stamps", None) or {})
        store.pop(store_key, None)
        stamps.pop(store_key, None)
        child_op._internal_set("edge_substitutions", store or None)
        child_op._internal_set("edge_replacement_stamps", stamps or None)
