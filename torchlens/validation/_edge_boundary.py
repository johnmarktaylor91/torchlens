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
    for store_key in entries:
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
    return None


def _splice_edge_substitution_args(
    trace: Trace, target_op: Op, entries: dict[Any, Any], input_args: dict[str, Any]
) -> tuple[dict[str, Any] | None, ValidationCheckResult | None]:
    """Splice tier-(ii) substituted values into the captured call arguments."""

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
            args[int(arg_path[0])] = value
        else:
            kwargs[arg_path[0]] = value
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
