"""Backward event flow and intervention evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from .invariants import (
        MetadataInvariantError,
        _resolve_op_grad_event_label,
    )

__all__ = (
    "_check_backward_event_flow_invariants",
    "_intervention_spec_is_armed",
    "op_has_genuine_replacement_evidence",
    "_is_func_call_id_exempt",
    "_plain_func_call_group_signature",
)


def _check_backward_event_flow_invariants(trace: Trace, name: str) -> None:
    """Check runtime backward event stream consistency against projections.

    Parameters
    ----------
    trace:
        Postprocessed model log to validate.
    name:
        Invariant check name to use in raised errors.

    Raises
    ------
    MetadataInvariantError
        If runtime backward sidecar events are internally inconsistent or no
        longer match projected records.
    """

    from .._ir.events import (
        BackwardCoverageGap,
        BackwardPassEnd,
        BackwardPassStart,
        GradFnDiscovered,
        GradFnFired,
        OpGradObserved,
        ParamGradObserved,
    )

    capture_events = getattr(trace, "_capture_events", None)
    events = list(getattr(capture_events, "backward_events", ()) or ())
    if not events:
        return

    starts = [event for event in events if isinstance(event, BackwardPassStart)]
    ends = [event for event in events if isinstance(event, BackwardPassEnd)]
    op_grad_events = [event for event in events if isinstance(event, OpGradObserved)]
    fired_events = [event for event in events if isinstance(event, GradFnFired)]
    param_grad_events = [event for event in events if isinstance(event, ParamGradObserved)]
    start_indices = [event.pass_index for event in starts]
    end_indices = [event.pass_index for event in ends]
    if len(start_indices) != len(set(start_indices)):
        raise MetadataInvariantError(name, "backward events contain duplicate pass starts")
    if len(end_indices) != len(set(end_indices)):
        raise MetadataInvariantError(name, "backward events contain duplicate pass ends")
    # A detached stream (pickle restore / fork over an existing projection)
    # records ``pass_index_base``: the passes materialized before the stream
    # existed. Within the stream, brackets must be dense from base + 1 — a
    # live capture stream has base 0, so this is the historical dense-from-1
    # check there. The base is written only by the two detach sites; events
    # at or below it can never appear here (they would break density and the
    # missing-pass checks below).
    pass_index_base = int(getattr(capture_events, "pass_index_base", 0) or 0)
    if pass_index_base < 0:
        raise MetadataInvariantError(
            name,
            f"backward stream pass-index base {pass_index_base!r} is negative",
        )
    bracket_indices = sorted(set(start_indices) | set(end_indices))
    if bracket_indices != list(
        range(pass_index_base + 1, pass_index_base + 1 + len(bracket_indices))
    ):
        raise MetadataInvariantError(
            name,
            f"backward event pass indices {bracket_indices!r} are not dense "
            f"from {pass_index_base + 1}",
        )
    if set(start_indices) != set(end_indices):
        raise MetadataInvariantError(
            name,
            "backward events must contain exactly one start and end per pass",
        )

    valid_pass_indices = set(bracket_indices)
    for op_grad_event in op_grad_events:
        if op_grad_event.pass_index not in valid_pass_indices:
            raise MetadataInvariantError(
                name,
                f"backward event references missing pass {op_grad_event.pass_index!r}",
            )
    for fired_event in fired_events:
        if fired_event.pass_index not in valid_pass_indices:
            raise MetadataInvariantError(
                name,
                f"backward event references missing pass {fired_event.pass_index!r}",
            )
    for param_grad_event in param_grad_events:
        if param_grad_event.pass_index not in valid_pass_indices:
            raise MetadataInvariantError(
                name,
                f"backward event references missing pass {param_grad_event.pass_index!r}",
            )

    seq_values = [event.seq for event in events]
    if seq_values != sorted(seq_values) or len(seq_values) != len(set(seq_values)):
        raise MetadataInvariantError(name, "backward event seq values must be unique and monotonic")

    # Exact bracketing: the writer stamps one run-monotonic seq on every
    # backward event, so every pass-scoped fact must sit strictly between its
    # pass's start and terminal records. This is a recorded fact, not an
    # inference from timestamps or list positions.
    start_seq_by_pass = {event.pass_index: event.seq for event in starts}
    end_seq_by_pass = {event.pass_index: event.seq for event in ends}
    coverage_gap_events = [event for event in events if isinstance(event, BackwardCoverageGap)]
    for gap_event in coverage_gap_events:
        if gap_event.pass_index not in valid_pass_indices:
            raise MetadataInvariantError(
                name,
                f"coverage gap references missing pass {gap_event.pass_index!r}",
            )
    pass_scoped_events: list[
        OpGradObserved | GradFnFired | ParamGradObserved | BackwardCoverageGap
    ] = [
        *op_grad_events,
        *fired_events,
        *param_grad_events,
        *coverage_gap_events,
    ]
    for event in pass_scoped_events:
        start_seq = start_seq_by_pass.get(event.pass_index)
        if start_seq is not None and event.seq < start_seq:
            raise MetadataInvariantError(
                name,
                f"backward event seq {event.seq} precedes its pass "
                f"{event.pass_index} start (seq {start_seq})",
            )
        end_seq = end_seq_by_pass.get(event.pass_index)
        if end_seq is not None and event.seq > end_seq:
            raise MetadataInvariantError(
                name,
                f"backward event seq {event.seq} follows its pass "
                f"{event.pass_index} end (seq {end_seq})",
            )
    for pass_index, start_seq in start_seq_by_pass.items():
        end_seq = end_seq_by_pass.get(pass_index)
        if end_seq is not None and end_seq <= start_seq:
            raise MetadataInvariantError(
                name,
                f"backward pass {pass_index} end (seq {end_seq}) does not "
                f"follow its start (seq {start_seq})",
            )

    # Higher-order discoveries carry ``created_in_pass``: the writer emits them
    # during that pass (hook-time terminals and the pre-End rewalk), so their
    # seq must sit inside the same bracket as every other pass-scoped fact.
    for discovered_event in events:
        if (
            not isinstance(discovered_event, GradFnDiscovered)
            or discovered_event.created_in_pass is None
        ):
            continue
        created_in_pass = discovered_event.created_in_pass
        if created_in_pass not in valid_pass_indices:
            raise MetadataInvariantError(
                name,
                f"GradFnDiscovered references missing pass {created_in_pass!r}",
            )
        start_seq = start_seq_by_pass.get(created_in_pass)
        if start_seq is not None and discovered_event.seq < start_seq:
            raise MetadataInvariantError(
                name,
                f"higher-order GradFnDiscovered seq {discovered_event.seq} precedes "
                f"its pass {created_in_pass} start (seq {start_seq})",
            )
        end_seq = end_seq_by_pass.get(created_in_pass)
        if end_seq is not None and discovered_event.seq > end_seq:
            raise MetadataInvariantError(
                name,
                f"higher-order GradFnDiscovered seq {discovered_event.seq} follows "
                f"its pass {created_in_pass} end (seq {end_seq})",
            )

    # Exact bracketing also forbids partial interleaving: with one
    # run-monotonic seq, pass brackets must be disjoint or properly nested
    # (a reentrant pass opened inside a hook closes before its outer pass).
    bracket_intervals = sorted(
        (start_seq, end_seq_by_pass[pass_index], pass_index)
        for pass_index, start_seq in start_seq_by_pass.items()
        if pass_index in end_seq_by_pass
    )
    open_bracket_stack: list[tuple[int, int, int]] = []
    for interval in bracket_intervals:
        interval_start, interval_end, interval_pass = interval
        while open_bracket_stack and open_bracket_stack[-1][1] < interval_start:
            open_bracket_stack.pop()
        if open_bracket_stack and interval_end > open_bracket_stack[-1][1]:
            outer_start, outer_end, outer_pass = open_bracket_stack[-1]
            raise MetadataInvariantError(
                name,
                f"backward pass {interval_pass} bracket (seq {interval_start}.."
                f"{interval_end}) partially overlaps pass {outer_pass} bracket "
                f"(seq {outer_start}..{outer_end})",
            )
        open_bracket_stack.append(interval)

    # Reconcile by MULTIPLICITY, not membership: a duplicated or dropped
    # record/event PAIR keeps set equality but changes the count, so only a
    # multiset comparison catches it. Reconciliation is scoped to the
    # stream's window: records for passes at or below ``pass_index_base`` are
    # preserved projections whose source events were dropped with the
    # pre-detach stream by design, so the stream is authoritative (and this
    # comparison exact) only for passes strictly above the base. With base 0
    # every record is in scope — the historical full comparison.
    layer_labels = set(getattr(trace, "layer_dict_all_keys", {}))
    projected_grad_records: Counter[tuple[str, int]] = Counter()
    for layer in getattr(trace, "layer_list", []):
        for record in getattr(layer, "_grad_records", ()):
            if record.backward_pass_index <= pass_index_base:
                continue
            projected_grad_records[(layer.layer_label, record.backward_pass_index)] += 1

    event_grad_records: Counter[tuple[str, int]] = Counter()
    for op_grad_event in op_grad_events:
        event_label = _resolve_op_grad_event_label(trace, op_grad_event.op_label)
        if event_label not in layer_labels:
            raise MetadataInvariantError(
                name,
                f"OpGradObserved points to missing op label {op_grad_event.op_label!r}",
            )
        event_op = trace[event_label]
        event_grad_records[(event_op.layer_label, op_grad_event.pass_index)] += 1
    if projected_grad_records != event_grad_records:
        raise MetadataInvariantError(
            name,
            "projected op gradient records do not match OpGradObserved events by multiplicity",
        )

    param_addresses = set(getattr(trace, "param_logs", {}).keys())
    projected_param_records: Counter[tuple[str, int]] = Counter()
    for param_address, param_log in getattr(trace, "param_logs", {}).items():
        for record in getattr(param_log, "_grad_records", ()):
            if record.backward_pass_index <= pass_index_base:
                continue
            projected_param_records[(param_address, record.backward_pass_index)] += 1
    event_param_records: Counter[tuple[str, int]] = Counter()
    for param_grad_event in param_grad_events:
        if param_grad_event.param_address not in param_addresses:
            raise MetadataInvariantError(
                name,
                "ParamGradObserved points to missing param address "
                f"{param_grad_event.param_address!r}",
            )
        event_param_records[(param_grad_event.param_address, param_grad_event.pass_index)] += 1
    if projected_param_records != event_param_records:
        raise MetadataInvariantError(
            name,
            "projected param gradient records do not match ParamGradObserved events "
            "by multiplicity",
        )

    projected_calls: dict[tuple[int, int], int] = defaultdict(int)
    for grad_fn_handle in getattr(trace, "grad_fn_logs", {}).values():
        for call in grad_fn_handle.calls.values():
            if call.backward_pass_index <= pass_index_base:
                continue
            projected_calls[(grad_fn_handle.grad_fn_object_id, call.backward_pass_index)] += 1
    event_calls: dict[tuple[int, int], int] = defaultdict(int)
    grad_fn_ids = set(getattr(trace, "grad_fn_logs", {}))
    for fired_event in fired_events:
        if fired_event.object_id not in grad_fn_ids:
            raise MetadataInvariantError(
                name,
                f"GradFnFired points to missing grad_fn id {fired_event.object_id!r}",
            )
        event_calls[(fired_event.object_id, fired_event.pass_index)] += 1
    if projected_calls != event_calls:
        raise MetadataInvariantError(
            name,
            "projected grad_fn calls do not match GradFnFired events",
        )


def _intervention_spec_is_armed(spec: object | None) -> bool:
    """Return whether an intervention spec carries actual user intent.

    Plain captures can own an EMPTY ``InterventionSpec`` object (the lazy
    ``_ensure_intervention_spec`` default), so ``is not None`` alone is not
    evidence of an intervened trace. Armed means the user registered at least
    one target, hook, value spec, or helper.

    Parameters
    ----------
    spec:
        ``Trace._intervention_spec`` value.

    Returns
    -------
    bool
        True when the spec carries at least one registered intervention.
    """

    if spec is None:
        return False
    if any(
        getattr(spec, field_name, None)
        for field_name in ("targets", "target_value_specs", "hook_specs")
    ):
        return True
    return any(
        getattr(spec, field_name, None) is not None for field_name in ("hook", "helper", "value")
    )


def op_has_genuine_replacement_evidence(layer: Op, trace: Trace | None = None) -> bool:
    """Return whether trace-level evidence corroborates a replacement stamp.

    Every ``intervention_replacement`` exemption used to trust per-op
    attributes (``func_name``/``is_edited``/``is_internal_source``)
    that the placeholder synthesizer ITSELF writes, so a placeholder minted or
    forged during PLAIN capture passed validation -- defeating the 2026-06-02
    lesson that a placeholder op appearing during plain capture must STILL
    fail. This helper is the cross-check: the op must appear in the
    journal's intervention-edit records (``InterventionAppliedEvent``) with a
    live causal binding (run token matching the stream nonce plus the exact
    target op-event instance), appended ONLY by the capture sites that
    directly observed the replacement
    (``wrapped_hook`` seeing a raw forward hook return a new object; a
    live-fire hook reporting ``replaced=True`` while intervention machinery
    is armed), or the trace must carry no journal authority at all (loaded
    bundles, backend-neutral traces) in which case the legacy per-op behavior
    is preserved.

    Parameters
    ----------
    layer:
        Operation pass claiming to be an intervention replacement.
    trace:
        Trace being validated. Falls back to ``layer.source_trace``.

    Returns
    -------
    bool
        True when the claim is corroborated (or no ledger authority exists).
    """

    if trace is None:
        trace = getattr(layer, "source_trace", None)
    if trace is None:
        # No trace-level authority reachable (detached op). A persisted
        # save-time stamp is still per-op authority when present (a False
        # stamp is a preserved plain-capture-placeholder verdict and must
        # keep failing); with no stamp at all, preserve the legacy per-op
        # behavior rather than failing structures we cannot cross-check.
        stamp = _persisted_replacement_stamp(layer)
        if stamp is not None:
            return stamp
        return True
    if bool(getattr(trace, "_loaded_from_bundle", False)):
        # Journal edit records are live-capture runtime facts (never
        # serialized), so loaded artifacts carry the save-time verdict
        # instead: ``_stamp_replacement_evidence`` evaluates this exact
        # corroboration while the live authority is intact and persists it in
        # the op's ``annotations``. A missing or False stamp fails CLOSED --
        # returning True here let a plain-capture placeholder that failed
        # live PASS after a save/load round trip, contradicting the locked
        # 2026-06-02 rule.
        return _persisted_replacement_stamp(layer) is True
    from .._ir.events import InterventionAppliedEvent

    stream = getattr(trace, "_capture_events", None)
    # Causal binding: an edit counts only when it is bound to THIS stream's
    # run (its run_token matches the stream nonce) AND the journal really
    # contains the exact target op event it was stamped against at the
    # observation site -- (label_raw, seq) identifies one event instance, so
    # a bare record appended through the ordinary writer (forged) and a
    # genuine record replayed from a DIFFERENT run's journal both stay
    # refused, and a pass-1 edit can no longer bless a same-labelled pass-2
    # op after a multi-pass merge (concat re-binds sanctioned merges).
    run_nonce = getattr(stream, "run_nonce", None)
    target_event_ids = {
        (event.label_raw, event.seq) for event in getattr(stream, "op_events", ()) or ()
    }
    bound_edits = [
        event
        for event in getattr(stream, "intervention_events", ()) or ()
        if isinstance(event, InterventionAppliedEvent)
        and event.kind == "replaced"
        and event.run_token is not None
        and event.run_token == run_nonce
        and event.target_seq
        and (event.label_raw, event.target_seq) in target_event_ids
    ]
    if bound_edits:
        candidate_labels = {
            getattr(layer, "_label_raw", None),
            getattr(layer, "label", None),
            getattr(layer, "layer_label", None),
        }
        candidate_labels.discard(None)
        layer_func_call_id = getattr(layer, "func_call_id", None)
        for event in bound_edits:
            if event.label_raw not in candidate_labels:
                continue
            # Pin the op instance when both sides carry a func_call_id;
            # synthesized boundary ops may legitimately carry None on the
            # layer, which keeps the label+target binding as the authority.
            if (
                layer_func_call_id is not None
                and event.target_func_call_id is not None
                and event.target_func_call_id != layer_func_call_id
            ):
                continue
            return True
    # Push/rerun fallback: the journal is a run-scoped stream on the
    # capture-time trace object, and the intervention rerun engine rebuilds a
    # fresh trace off to the side then swaps its FIELD-ORDER state into the
    # original object -- the stream does not survive the swap, and push()
    # stamps sites without a capture at all. Both are explicit user
    # interventions, so accept the conjunction of two signals a plain-capture
    # placeholder can never carry together: (1) this op holds a hook-minted
    # FireRecord with ``replaced=True`` (only hook EXECUTION creates these;
    # the placeholder synthesizer writes ``interventions=[]``), AND (2) the
    # trace itself owns an ARMED intervention spec -- one with actual
    # targets/hooks/values, populated only by ``trace(intervene=...)`` /
    # ``attach_hooks`` / ``set``. Plain captures carry an EMPTY spec object,
    # so a stale ``_tl_live_fire_results`` leak into a plain capture carries
    # records but no armed spec and stays refused.
    if _intervention_spec_is_armed(getattr(trace, "_intervention_spec", None)):
        for record in getattr(layer, "interventions", ()) or ():
            if getattr(record, "replaced", False):
                return True
    # A live capture with NO corroborating replacement evidence: any op
    # claiming to be a replacement is a plain-capture gap or a forged stamp.
    return False


def _persisted_replacement_stamp(layer: Op) -> bool | None:
    """Return the save-time replacement-corroboration verdict, if persisted.

    ``torchlens._io.scrub._stamp_replacement_evidence`` evaluates
    :func:`op_has_genuine_replacement_evidence` at SAVE time (while the
    journal/spec authority is live) and stores the verdict in the op's
    portable ``annotations`` as ``replacement_evidence_v1``.

    Parameters
    ----------
    layer:
        Operation pass claiming to be an intervention replacement.

    Returns
    -------
    bool | None
        The stamped verdict, or ``None`` when no stamp is present (legacy
        artifacts saved before the stamp existed, or live never-saved ops).
    """

    annotations = getattr(layer, "annotations", None) or {}
    stamp = annotations.get("replacement_evidence_v1")
    if isinstance(stamp, dict):
        return stamp.get("corroborated") is True
    return None


def _is_func_call_id_exempt(layer: Op) -> bool:
    """Return whether a layer is exempt from Invariant S.

    Parameters
    ----------
    layer:
        Layer pass to classify.

    Returns
    -------
    bool
        Whether the layer is synthetic input/output/buffer metadata.
    """

    if layer.is_input or layer.is_output or layer.is_buffer:
        return True
    # A GENUINE raw-forward-hook output replacement is legitimately functionless
    # (the user substituted an opaque tensor for a module's output, so there is
    # no torch function -- hence no ``func_call_id`` -- to validate). Mirror the
    # deliberately NARROW predicate the ``op_log_fields`` invariant already uses
    # (func_name + is_edited + NOT internal_source), AND require the
    # trace-level replacement-event ledger to corroborate it
    # (``op_has_genuine_replacement_evidence``): the per-op attributes alone are
    # written by the placeholder synthesizer itself, so trusting them let a
    # plain-capture placeholder pass (round-26 W3-2). Widening this to a blanket
    # ``func_name`` check would disarm the plain-capture tripwire (see project
    # CLAUDE.md "Validation Integrity").
    if (
        getattr(layer, "func_name", "") == "intervention_replacement"
        and getattr(layer, "is_edited", False)
        and not getattr(layer, "is_internal_source", False)
        and op_has_genuine_replacement_evidence(layer)
    ):
        return True
    # The bookkeeping func_name sentinels are only legitimate when the
    # corresponding structural FLAG corroborates them -- and the flagged
    # cases (is_input/is_output/is_buffer) already returned True above, so a
    # bare "input"/"output"/"buffer" string reaching this point is per-op
    # metadata corruption, not an exemption. The one remaining legitimate
    # spelling is the functionless internal source ("none" + the
    # is_internal_source flag + no callable func, exactly the conjunction the
    # payloads invariant requires). A bare string set here let a traced relu
    # with a nulled func_call_id and a rewritten func_name pass the whole
    # suite (deephunt M2).
    func_name = str(getattr(layer, "func_name", "")).lower()
    return (
        func_name == "none"
        and bool(getattr(layer, "is_internal_source", False))
        and getattr(layer, "func", None) is None
    )


def _plain_func_call_group_signature(layer: Op) -> tuple[object, ...]:
    """Return plain-capture-stable same-call metadata.

    Parameters
    ----------
    layer:
        Layer pass to summarize.

    Returns
    -------
    tuple[object, ...]
        Function name and container spec representation for same-call grouping.
    """

    return (
        getattr(layer, "func_name", None),
        repr(getattr(layer, "container_spec", None)),
    )
