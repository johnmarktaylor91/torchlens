"""Edge-intervention boundary validation (L6 4.3), split out of ``core.py``.

One replay-phase check family: a child op carrying tier-(ii)
edge-substitution entries gets a DIFFERENT check, never NO check. The
functions import their ``core`` helpers lazily so this module stays
import-light and cycle-free (``core`` imports this module at its top).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .core import ValidationCheckResult


def _check_edge_intervention_boundary(
    trace: Trace,
    target_op: Op,
) -> ValidationCheckResult | None:
    """Validate one edge-intervened child (L6 4.3 — a DIFFERENT check, never NO check).

    Returns ``None`` when the op carries no tier-(ii) edge-substitution
    entries (the generic tripwires run unchanged). Otherwise:

    * POSITIVE INVARIANT: every tier-(ii) entry MUST be corroborated by (a) a
      FireRecord carrying the exact occurrence address AND (b) the save-time
      corroboration stamp. An uncorroborated entry FAILS validation.
    * ACCEPT SIDE (re-execute-from-tier-(ii), NOT skip): the child is
      RE-EXECUTED with the substituted values spliced in at exactly the
      corroborated occurrence addresses (all other args from capture truth),
      and the stored child output MUST match the re-execution. The verdict is
      the DISTINCT closed term ``edge_intervention_boundary`` — never
      "exempted". A wrong stored output under a corroborated entry FAILS
      (skip-shaped acceptance is impossible by construction).

    Node-level ``intervention_replaced`` corroboration never fires for edges:
    edge substitution replaces no node's output.
    """

    entries = getattr(target_op, "edge_substitutions", None) or {}
    if not entries:
        return None
    failure = _fail_uncorroborated_edge_entries(trace, target_op, entries)
    if failure is not None:
        return failure
    return _reexecute_edge_boundary(trace, target_op, entries)


def _fail_uncorroborated_edge_entries(
    trace: Trace, target_op: Op, entries: dict[Any, Any]
) -> ValidationCheckResult | None:
    """Fail any tier-(ii) entry missing its FireRecord address or stamp verdict."""

    from .core import ValidationCheckResult
    from .diagnostics import CHECK_REPLAY, ValidationFailure, record_validation_failure

    stamps = getattr(target_op, "edge_replacement_stamps", None) or {}
    fire_addresses = {
        tuple(getattr(record, "edge_address", ()) or ())
        for record in (getattr(target_op, "interventions", None) or ())
        if getattr(record, "edge_address", None) is not None
    }
    from ..intervention.edge_substitution import tensor_content_digest

    for store_key, payload in entries.items():
        arg_kind, arg_path = store_key
        address = (target_op.func_call_id, arg_kind, tuple(arg_path))
        stamp = stamps.get(store_key)
        if address not in fire_addresses or not stamp or not stamp.get("verdict"):
            record_validation_failure(
                trace,
                ValidationFailure(
                    check=CHECK_REPLAY,
                    op_label=target_op.label,
                    func_name=getattr(target_op, "func_name", None),
                    message=(
                        f"edge-substitution entry at {address!r} is UNCORROBORATED "
                        "(missing edge FireRecord and/or corroboration stamp)"
                    ),
                ),
            )
            return ValidationCheckResult.failed_result("edge_substitution_uncorroborated")
        # The stamp's value_digest is the CONTENT half of the corroboration:
        # a stored value that no longer digests to what was stamped is a
        # tampered or foreign store row, and re-executing the child from it
        # would corroborate the wrong value (AUD-CODE 4.10: the digest was
        # minted at every write and checked nowhere).
        value = payload.get("value") if isinstance(payload, dict) else None
        stamped_digest = stamp.get("value_digest")
        if (
            isinstance(value, torch.Tensor)
            and isinstance(stamped_digest, str)
            and tensor_content_digest(value) != stamped_digest
        ):
            record_validation_failure(
                trace,
                ValidationFailure(
                    check=CHECK_REPLAY,
                    op_label=target_op.label,
                    func_name=getattr(target_op, "func_name", None),
                    message=(
                        f"edge-substitution entry at {address!r} does not digest to its "
                        "corroboration stamp (stored value and stamped value_digest disagree)"
                    ),
                ),
            )
            return ValidationCheckResult.failed_result("edge_substitution_stamp_mismatch")
    return None


def _splice_edge_substitution_args(
    trace: Trace, target_op: Op, entries: dict[Any, Any], input_args: dict[str, Any]
) -> tuple[dict[str, Any] | None, ValidationCheckResult | None]:
    """Splice tier-(ii) substituted values into the captured call arguments."""

    from ._replay_grad_fidelity import replay_copy_for_slot
    from .core import ValidationCheckResult
    from .diagnostics import CHECK_REPLAY, ValidationFailure, record_validation_failure

    args = list(input_args["args"])
    kwargs = dict(input_args["kwargs"])
    for store_key, payload in entries.items():
        arg_kind, arg_path = store_key
        is_region = isinstance(payload, dict) and payload.get("substitution_kind") == "region"
        if is_region:
            # Region exit substitutions (F01) may sit at nested container
            # paths; the region splice rebuilds the container faithfully, so
            # the SAME re-execute-and-compare check runs at the nested
            # address -- a capability extension of this check, never a
            # weakening (the depth-1 tripwire below still refuses foreign
            # edge rows it cannot apply faithfully).
            value = payload.get("value")
            if isinstance(value, torch.Tensor):
                from ..intervention.regions import _splice_occurrence

                value = replay_copy_for_slot(
                    trace,
                    target_op,
                    "args" if arg_kind == "positional" else "kwargs",
                    arg_path[0] if len(tuple(arg_path)) == 1 else tuple(arg_path),
                    value,
                )

                spliced_args, kwargs = _splice_occurrence(
                    tuple(args), kwargs, (None, arg_kind, tuple(arg_path)), value
                )
                args = list(spliced_args)
                continue
        if len(tuple(arg_path)) != 1:
            # TRIPWIRE (shared with the replay/edge splice sites): a nested
            # store key at this top-level splice would silently re-execute
            # the child from the WRONG spliced argument and could validate
            # green. Refuse the entry as invalid instead of guessing.
            record_validation_failure(
                trace,
                ValidationFailure(
                    check=CHECK_REPLAY,
                    op_label=target_op.label,
                    func_name=getattr(target_op, "func_name", None),
                    message=(
                        f"edge-substitution entry at {store_key!r} has a nested "
                        "argument path; the top-level splice cannot apply it "
                        "faithfully (foreign or future-schema store row)"
                    ),
                ),
            )
            failed = ValidationCheckResult.failed_result("edge_substitution_nested_path")
            return None, failed
        value = payload.get("value") if isinstance(payload, dict) else None
        if not isinstance(value, torch.Tensor):
            record_validation_failure(
                trace,
                ValidationFailure(
                    check=CHECK_REPLAY,
                    op_label=target_op.label,
                    func_name=getattr(target_op, "func_name", None),
                    message=f"edge-substitution payload at {store_key!r} is not a tensor",
                ),
            )
            failed = ValidationCheckResult.failed_result("edge_substitution_payload_invalid")
            return None, failed
        if arg_kind == "positional":
            args[int(arg_path[0])] = replay_copy_for_slot(
                trace, target_op, "args", int(arg_path[0]), value
            )
        else:
            kwargs[arg_path[0]] = replay_copy_for_slot(
                trace, target_op, "kwargs", arg_path[0], value
            )
    spliced = dict(input_args)
    spliced["args"] = tuple(args)
    spliced["kwargs"] = kwargs
    return spliced, None


def _reexecute_edge_boundary(
    trace: Trace, target_op: Op, entries: dict[Any, Any]
) -> ValidationCheckResult:
    """Re-execute the child from the spliced tier-(ii) values and compare outputs."""

    from ..utils.tensor_utils import tensor_nanequal
    from .core import (
        ValidationCheckResult,
        _execute_func_with_restored_state,
        _prepare_input_args_for_validating_layer,
        _saved_out_payload,
    )
    from .diagnostics import CHECK_REPLAY, ValidationFailure, record_validation_failure

    input_args, unverified_reason = _prepare_input_args_for_validating_layer(trace, target_op, [])
    if input_args is None:
        return ValidationCheckResult.unverified(unverified_reason or "missing_saved_args")
    spliced, failure = _splice_edge_substitution_args(trace, target_op, entries, input_args)
    if spliced is None:
        return failure if failure is not None else ValidationCheckResult.unverified("unknown")
    recomputed = _execute_func_with_restored_state(target_op, spliced, [], target_op.label, False)
    saved_output = _saved_out_payload(target_op)
    if recomputed is None or saved_output is None:
        return ValidationCheckResult.unverified("edge_boundary_replay_unavailable")
    if isinstance(recomputed, (tuple, list)) and not isinstance(recomputed, torch.Tensor):
        container_path = tuple(getattr(target_op, "container_path", ()) or ())
        for component in container_path:
            recomputed = recomputed[component]
    if not tensor_nanequal(recomputed, saved_output, allow_tolerance=True):
        record_validation_failure(
            trace,
            ValidationFailure(
                check=CHECK_REPLAY,
                op_label=target_op.label,
                func_name=getattr(target_op, "func_name", None),
                message=(
                    "stored output does not match re-execution from the corroborated "
                    "tier-(ii) edge substitution (divergence without provenance)"
                ),
            ),
        )
        return ValidationCheckResult.failed_result("edge_boundary_reexecution_mismatch")
    return ValidationCheckResult("edge_intervention_boundary", "edge_boundary_reexecuted")


def _parent_arg_evidence(
    trace: Trace, target_layer: Op, parent_layer: Op
) -> tuple[torch.Tensor, str | None] | ValidationCheckResult:
    """Resolve the parent value a child's saved arg slot must be compared against.

    Returns ``(parent_outs, capture_digest)`` or an ``unverified`` result.
    ``out_versions_by_child`` stores per-child snapshots when an in-place op
    modified the tensor between uses (capture truth; compared directly, the
    pass-qualified child label first). Otherwise the parent's saved out is the
    evidence -- EXCEPT on a REPLAY-PROPAGATED trace (AUD-CODE 2.9): once the
    replay engine has overwritten this parent's out, the child's
    ``saved_args`` snapshot (capture truth, retained unmodified by design)
    can no longer be compared against ``parent.out`` (a pushed value that
    legitimately differs) -- every non-identity edited fork false-FAILED at
    the first downstream op. The engine records the capture-time content
    digest of every out it overwrites, so the SAME equality question ("does
    the snapshot at this slot equal the parent's capture-time value?") is
    answered against that digest: still exact, still bidirectional, never
    skipped. A recomputed parent with no recordable digest stays unverified,
    never validated.
    """

    from .core import ValidationCheckResult, _saved_out_payload

    target_op_label = getattr(target_layer, "label", target_layer.layer_label)
    versions = parent_layer.out_versions_by_child
    for key in (target_op_label, target_layer.layer_label):
        if key in versions:
            return versions[key], None
    capture_digest = _replay_capture_digest_for(trace, parent_layer)
    if capture_digest is None and _is_replay_recomputed(trace, parent_layer):
        return ValidationCheckResult.unverified("replay_recomputed_parent_unattested")
    parent_outs = _saved_out_payload(parent_layer)
    if parent_outs is None:
        return ValidationCheckResult.unverified("missing_saved_parent_payload")
    return parent_outs, capture_digest


def _tensor_content_digest(value: torch.Tensor) -> str:
    """Return the replay engine's content digest of one tensor (shared spelling)."""

    from ..intervention.edge_substitution import tensor_content_digest

    return tensor_content_digest(value)


def _replay_capture_digest_for(trace: Trace, op: Op) -> str | None:
    """Return the capture-time digest of an op's out if replay overwrote it."""

    from ..intervention._replay_context import replay_capture_digest

    return replay_capture_digest(trace, op)


def _is_replay_recomputed(trace: Trace, op: Op) -> bool:
    """Return whether the replay engine has overwritten this op's out on this trace.

    Derived from the replay run context's committed-site set (the digest
    ledger keys), so a site whose pre-edit out could not be digested still
    reads as recomputed and is never mistaken for capture truth.
    """

    from ..intervention._replay_context import REPLAY_CAPTURE_DIGESTS_KEY, _replay_site_key

    run_ctx = getattr(trace, "last_run", None)
    if not isinstance(run_ctx, dict):
        return False
    digests = run_ctx.get(REPLAY_CAPTURE_DIGESTS_KEY)
    return isinstance(digests, dict) and _replay_site_key(op) in digests


def _capture_payload_equal(trace: Trace, candidate: Op, target_layer: Op, value: Any) -> bool:
    """Return whether ``value`` equals the candidate's CAPTURE-TIME out as the target consumed it.

    A child-versioned snapshot (``out_versions_by_child``) is capture truth
    and compares directly; a replay-recomputed candidate compares through its
    recorded capture digest; an untouched candidate compares its saved out.
    Unknown evidence (no payload, no digest) reads as not equal.
    """

    from ..utils.tensor_utils import tensor_nanequal
    from .core import _saved_out_payload

    if not isinstance(value, torch.Tensor):
        return False
    versions = getattr(candidate, "out_versions_by_child", None) or {}
    for key in (getattr(target_layer, "label", None), target_layer.layer_label):
        if key is not None and key in versions:
            snapshot = versions[key]
            return isinstance(snapshot, torch.Tensor) and bool(
                tensor_nanequal(value, snapshot, allow_tolerance=False)
            )
    digest = _replay_capture_digest_for(trace, candidate)
    if digest is not None:
        return _tensor_content_digest(value) == digest
    if _is_replay_recomputed(trace, candidate):
        return False
    payload = _saved_out_payload(candidate)
    return payload is not None and bool(tensor_nanequal(value, payload, allow_tolerance=False))
