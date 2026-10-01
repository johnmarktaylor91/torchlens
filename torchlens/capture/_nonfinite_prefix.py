"""Failure-safe prefix finalization + the ONE raw-to-public label resolver.

Observe memo item 1 (release-blocking): three NaN surfaces used to give three
names for one op -- ``bisect_nan`` a final ordinal, ``find_nan`` a raw ordinal
with the suffix stripped, ``raise_on_nan`` the raw spelling itself. The offset
between raw and finalized ordinals is provably not a constant (pseudo-rows
consume raw ordinals but not finalized ones) and finalization genuinely
deletes committed records (input-disconnected dead chains), so NO arithmetic
remap is ever correct. The only correct mechanism is the identity join: run
postprocess over the committed prefix and read every public label out of the
produced step-8 raw-to-final map -- and when that map cannot be produced, the
public label is ``None`` with an explicit status, never a raw spelling.

Two doors live here:

- :func:`resolve_raw_label` -- the one resolver every public
  message/field-construction site uses to turn an internal raw label into a
  public label. Returns ``(label, status)`` with status in the closed set
  ``{"final", "pruned", "unavailable"}``.
- :func:`maybe_finalize_nonfinite_prefix` -- the failure-safe prefix
  finalization run on the aborted-nonfinite path (where postprocess never ran
  historically). It mirrors the shipped halt finalizer: the offending tensor
  is seeded as the output frontier (so the OFFENDER can never be elided by the
  orphan flood -- only dead-branch bystanders can), ordinary postprocess runs,
  and the original ``CaptureError``'s public fields are rewritten through the
  identity map. Any finalization failure attaches as SECONDARY evidence on the
  primary error and never replaces it.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

#: Closed status vocabulary for resolved public labels.
LABEL_STATUS_FINAL = "final"
LABEL_STATUS_PRUNED = "pruned"
LABEL_STATUS_UNAVAILABLE = "unavailable"


def canonical_public_label(op: Any) -> str:
    """Return the canonical public lookup spelling for one op record.

    The canonical spelling is the step-8 final LOOKUP label: the bare
    ``layer_label`` for single-pass layers and the pass-qualified ``label``
    (``linear_1_1:2``) for multi-pass layers -- an unqualified spelling on a
    recurrent block would report a pass-2 NaN under a name that equally names
    pass 1.

    Parameters
    ----------
    op:
        Finalized op record.

    Returns
    -------
    str
        Canonical public label.
    """

    num_passes = int(getattr(op, "num_passes", 1) or 1)
    if num_passes > 1:
        try:
            label = op.label
        except ValueError:
            # An AGGREGATE multi-pass Layer deliberately refuses per-pass
            # reads; the honest spelling for the aggregate is the bare
            # layer_label below (it names the layer, not one pass).
            label = None
        if isinstance(label, str) and label:
            return label
    layer_label = getattr(op, "layer_label", None)
    if isinstance(layer_label, str) and layer_label:
        return layer_label
    label = getattr(op, "label", None)
    return str(label) if isinstance(label, str) else ""


def resolve_raw_label(trace: Any, raw_label: Any) -> tuple[str | None, str]:
    """Resolve one internal raw label to its public label through the map.

    This is the ONE door: public message- and field-construction sites never
    interpolate a raw label directly and never compute a public spelling by
    ordinal arithmetic (the offset is not a constant). A raw label that the
    finalized map does not know is reported ``None`` with an explicit status.

    Parameters
    ----------
    trace:
        Trace whose step-8 identity map should serve the join. Un-finalized
        traces (no map) resolve everything ``unavailable``.
    raw_label:
        Internal raw label (``"log_1_4_raw"``) recorded at capture time.

    Returns
    -------
    tuple[str | None, str]
        ``(public_label, status)``. ``status`` is ``"final"`` when the map
        served the join, ``"pruned"`` when the record was removed as an
        input-disconnected bystander (public label is honestly ``None``), and
        ``"unavailable"`` when no finalized identity exists (never a raw
        spelling).
    """

    if not isinstance(raw_label, str) or not raw_label:
        return None, LABEL_STATUS_UNAVAILABLE
    mapping = getattr(trace, "_raw_to_final_layer_labels", None) if trace is not None else None
    if isinstance(mapping, dict):
        final = mapping.get(raw_label)
        if isinstance(final, str) and final:
            return final, LABEL_STATUS_FINAL
    if trace is not None:
        orphan_labels = trace.__dict__.get("_orphan_labels") if hasattr(trace, "__dict__") else None
        if isinstance(orphan_labels, list) and raw_label in orphan_labels:
            return None, LABEL_STATUS_PRUNED
        for record in getattr(trace, "orphan_records", ()) or ():
            if isinstance(record, dict) and record.get("raw_label") == raw_label:
                return None, LABEL_STATUS_PRUNED
    return None, LABEL_STATUS_UNAVAILABLE


def _resolve_many(trace: Any, raw_labels: Any) -> tuple[str | None, ...]:
    """Resolve a raw-label sequence, preserving positions with ``None`` gaps."""

    if not isinstance(raw_labels, (list, tuple)):
        return ()
    return tuple(resolve_raw_label(trace, raw)[0] for raw in raw_labels)


def _rewrite_nonfinite_error(trace: Any, exc: BaseException, *, finalized: bool) -> None:
    """Rewrite the nonfinite ``CaptureError``'s public payload through the map.

    The raw spellings stay available for identity under explicitly raw-named
    keys (``layer_raw`` / ``parents_raw`` -- the ``orphan_records`` house
    pattern); every PUBLIC key carries a final label or ``None`` + status.

    Parameters
    ----------
    trace:
        Trace serving the identity join.
    exc:
        The original nonfinite ``CaptureError`` (mutated in place; identity
        and traceback preserved).
    finalized:
        Whether the prefix finalization ran (statuses can still be
        ``unavailable`` for individual labels either way).
    """

    fields = getattr(exc, "fields", None)
    if not isinstance(fields, dict):
        return
    raw_label = fields.get("layer_raw", fields.get("layer"))
    if not isinstance(raw_label, str):
        return
    label, status = resolve_raw_label(trace, raw_label)
    if not finalized and status == LABEL_STATUS_FINAL:
        # A stale pre-existing map must not serve a failed finalization.
        label, status = None, LABEL_STATUS_UNAVAILABLE
    fields["layer_raw"] = raw_label
    fields["layer"] = label
    fields["layer_status"] = status
    raw_parents = fields.get("parents_raw", fields.get("parents"))
    if isinstance(raw_parents, (list, tuple)):
        fields["parents_raw"] = list(raw_parents)
        if finalized:
            fields["parents"] = [
                resolved for resolved in _resolve_many(trace, raw_parents) if resolved is not None
            ]
        else:
            fields["parents"] = []
    from ..errors import TorchLensError

    if isinstance(exc, TorchLensError) and isinstance(exc.affected_sites, list):
        exc.affected_sites = [label] if label is not None else []
    # Rebuild the message with the public spelling (or the explicit absence).
    if exc.args and isinstance(exc.args[0], str) and repr(raw_label) in exc.args[0]:
        if label is not None:
            replacement = repr(label)
        else:
            replacement = f"None (public label {status})"
        exc.args = (exc.args[0].replace(repr(raw_label), replacement),) + exc.args[1:]


def maybe_finalize_nonfinite_prefix(
    trace: Trace,
    backend: Any,
    exc: BaseException,
    model: Any,
    input_tensors: list[Any],
) -> bool:
    """Run the failure-safe prefix finalization for an aborted-nonfinite capture.

    No-op unless the terminal exception IS the latched nonfinite abort. On
    success the committed prefix is postprocessed with the offending tensor
    seeded as the output frontier, and the error's public payload is rewritten
    through the step-8 identity map. On ANY failure the primary error and its
    settlement classification are untouched: the failure attaches as secondary
    evidence and public labels stay ``None`` + ``unavailable``.

    Parameters
    ----------
    trace:
        Active trace being settled.
    backend:
        Active capture backend (model-session cleanup + output extraction).
    exc:
        The exception propagating out of the forward.
    model:
        Model object for session cleanup.
    input_tensors:
        Input tensors tagged for the current pass.

    Returns
    -------
    bool
        Whether the prefix finalization ran to completion.
    """

    from .outcome import StopRequest
    from .trace import _extract_and_mark_outputs

    stop_request = trace.__dict__.get("_stop_requested")
    if not (
        isinstance(stop_request, StopRequest)
        and stop_request.kind == "nonfinite"
        and stop_request.error_ref is not None
        and stop_request.error_ref() is exc
    ):
        trace.__dict__.pop("_nonfinite_frontier_out", None)
        return False
    frontier_output = trace.__dict__.pop("_nonfinite_frontier_out", None)
    saved_phase = trace.__dict__.get("_capture_phase")
    try:
        if frontier_output is None:
            raw_layer_dict = trace._raw_graph_ws.raw_layer_dict
            for event in reversed(getattr(trace.capture_events, "op_events", ())):
                entry = raw_layer_dict.get(event.label_raw)
                if entry is not None and getattr(entry, "out", None) is not None:
                    frontier_output = entry.out
                    break
        if frontier_output is None:
            raise RuntimeError("no tensor frontier is recoverable for the aborted-nonfinite prefix")
        # Mirror the halt finalizer ordering: recover the frontier while
        # capture-time state is intact (done above), THEN clean the model
        # session, then extract/mark outputs and postprocess. The later
        # failed-forward cleanup's own model-session action dedupes through
        # the session cleanup registry, so it does not run twice.
        backend.cleanup_model_session(trace, (model, input_tensors))
        output_tensors, output_tensor_addresses = _extract_and_mark_outputs(
            trace,
            frontier_output,
            backend,
        )
        trace.__dict__.pop("_output_attribution_input_tensors", None)
        trace.raw_output = None
        trace._postprocess(output_tensors, output_tensor_addresses)
        trace.__dict__["_nonfinite_prefix_finalized"] = True
    except Exception as finalize_exc:  # noqa: BLE001 - secondary evidence, never the primary error.
        trace.__dict__["_nonfinite_prefix_finalize_error"] = (
            f"{type(finalize_exc).__name__}: {finalize_exc}"
        )
        with contextlib.suppress(Exception):
            exc.add_note(
                "TorchLens prefix finalization for public NaN labels failed "
                f"({type(finalize_exc).__name__}: {finalize_exc}); the public "
                "label is unavailable (never a raw spelling)."
            )
        with contextlib.suppress(Exception):
            _rewrite_nonfinite_error(trace, exc, finalized=False)
        with contextlib.suppress(Exception):
            _update_stop_request(trace, stop_request, exc)
        return False
    finally:
        # Settlement attributes the PRIMARY error; the finalization must not
        # relabel its capture phase.
        if saved_phase is not None:
            trace.__dict__["_capture_phase"] = saved_phase
        else:
            trace.__dict__.pop("_capture_phase", None)
    _rewrite_nonfinite_error(trace, exc, finalized=True)
    _update_stop_request(trace, stop_request, exc)
    return True


def _update_stop_request(trace: Any, stop_request: Any, exc: BaseException) -> None:
    """Mirror the rewritten public payload onto the settlement latch.

    ``CaptureOutcome.boundary_label`` and ``.reason`` are public surfaces; the
    latch is replaced (it is frozen) with the resolved boundary label and the
    rewritten message, preserving the error-identity reference.
    """

    import dataclasses

    fields = getattr(exc, "fields", None)
    if not isinstance(fields, dict) or "layer_status" not in fields:
        return
    label = fields.get("layer")
    message = exc.args[0] if exc.args and isinstance(exc.args[0], str) else None
    replacement = dataclasses.replace(
        stop_request,
        boundary_label=label if isinstance(label, str) else None,
        reason=message if message is not None else stop_request.reason,
    )
    trace.__dict__["_stop_requested"] = replacement
