"""Live replay validation for the MLX backend.

The oracle mirrors the Paddle/tinygrad pattern: every captured operation is
re-invoked with arguments reconstructed from its DECLARED parents' saved
payloads, the replayed output must match the captured payload, and a
parent-perturbation tripwire proves the replay is not vacuous. Calls with no
perturbable tensor argument (constant producers) are recorded as explicit
evidence gaps that surface as UNVERIFIED at the trace level, never as silent
passes. The denominator is the captured (whitelisted) op set — MLX cannot
observe unwrapped internals, and that scope is documented rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._validation_shared import ops_by_label as _ops_by_label
from ._capture_records import (
    REPLAY_SLOT as REPLAY_SLOT,
    MLXOpCapture,
    _is_mlx_array,
    _iter_output_arrays,
    _payloads_close,
    _perturb_candidates,
    build_capture_template as build_capture_template,
)
from ._replay_args import _declared_parents_consistent, _reconstruct_call, _saved_payload

PERTURBATION_PROVED = "proved"
PERTURBATION_NO_PERTURBABLE_INPUT = "no_perturbable_input"
PERTURBATION_UNPROVED = "unproved"

#: Local cause -> reason code from ``VALIDATION_REASONS`` (SPEC 4.3), coded
#: once. Every per-op decision and every count key reads from this table.
MLX_VALIDATION_REASONS: dict[str, str] = {
    "coverage": "replay_coverage_incomplete",
    "parents_inconsistent": "declared_parents_inconsistent",
    "arity": "replay_mismatch",
    "value_mismatch": "replay_mismatch",
    "payload_missing": "parent_payload_missing",
    "replay_error": "no_replayable_function",
    "intervention_applier_missing": "user_intervention_replacement",
    "perturbation_unproved": "perturbation_insensitive",
    "perturbation_gap": "perturbation_insensitive",
    "constant_leaf": "constant_producer_leaf",
    "literal_producer": "constant_producer_leaf",
    "constant_parent": "constant_parent_payload",
    "metadata_invariant": "replay_mismatch",
    "nothing_replayed": "replay_mismatch",
}


@dataclass(frozen=True)
class MLXValidationOutcome:
    """What the replay oracle decided, op by op.

    Parameters
    ----------
    replayed_count:
        Ops replayed, compared equal and perturbation-proven.
    failed_count:
        Failed checks (coverage failures count one each).
    perturbation_gaps:
        Raw labels of replayed ops whose dependency could not be perturbation
        proven (unverified, never a pass).
    decisions:
        One record per op or trace-level check with keys ``op_label``,
        ``func_name``, ``phase``, ``decision``, ``reason`` and ``detail``.
    exempted_reason_counts:
        Provable exemptions by reason code (constant leaves only).
    unverified_reason_counts:
        Unverified decisions by reason code.
    """

    replayed_count: int
    failed_count: int
    perturbation_gaps: tuple[str, ...]
    decisions: tuple[dict[str, Any], ...]
    exempted_reason_counts: dict[str, int]
    unverified_reason_counts: dict[str, int]


@dataclass(frozen=True)
class _DecisionSite:
    """The op a validation decision is about.

    Parameters
    ----------
    label
        Raw op label (``"<trace>"`` for trace-wide decisions).
    op_name
        The op's ``func_name``; ``None`` for trace-wide decisions.
    """

    label: str
    op_name: str | None


def _decision(
    site: _DecisionSite,
    phase: str,
    decision: str,
    cause: str | None,
    detail: str,
) -> dict[str, Any]:
    """Build one decision record for ``site`` with the reason code for ``cause``."""

    return {
        "op_label": site.label,
        "func_name": site.op_name,
        "phase": phase,
        "decision": decision,
        "reason": None if cause is None else MLX_VALIDATION_REASONS[cause],
        "detail": detail,
    }


def _all_constant(labels: tuple[str | None, ...], constant_labels: frozenset[str]) -> bool:
    """Whether every leaf label in ``labels`` names a constant op."""

    return bool(labels) and all(label in constant_labels for label in labels)


def _perturbation_target(
    capture: MLXOpCapture, constant_labels: frozenset[str]
) -> tuple[int | None, str, Any, bool] | None:
    """Pick the argument to perturb: its position or keyword, its value, and constancy."""

    positions = [index for index, value in enumerate(capture.args) if _is_mlx_array(value)]
    positions.sort(
        key=lambda index: _all_constant(
            capture.arg_leaf_labels[index] if index < len(capture.arg_leaf_labels) else (),
            constant_labels,
        )
    )
    arg_position: int | None = positions[0] if positions else None
    kwarg_key = ""
    if arg_position is None:
        found_key = next(
            (key for key, value in capture.kwargs.items() if _is_mlx_array(value)),
            None,
        )
        if found_key is None:
            return None
        kwarg_key = found_key
    if arg_position is not None:
        original = capture.args[arg_position]
        perturbed_constant = _all_constant(
            capture.arg_leaf_labels[arg_position]
            if arg_position < len(capture.arg_leaf_labels)
            else (),
            constant_labels,
        )
    else:
        original = capture.kwargs[kwarg_key]
        perturbed_constant = _all_constant(
            capture.kwarg_leaf_labels.get(kwarg_key, ()), constant_labels
        )
    return arg_position, kwarg_key, original, perturbed_constant


def _substituted_call(
    capture: MLXOpCapture, arg_position: int | None, kwarg_key: str, candidate: Any
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The capture's call material with one argument replaced by ``candidate``."""

    if arg_position is not None:
        perturbed_args = (
            *capture.args[:arg_position],
            candidate,
            *capture.args[arg_position + 1 :],
        )
        return perturbed_args, capture.kwargs
    return capture.args, {**capture.kwargs, kwarg_key: candidate}


def _perturbation_evidence(
    capture: MLXOpCapture,
    baseline: tuple[Any, ...],
    constant_labels: frozenset[str] = frozenset(),
) -> tuple[str, bool]:
    """Classify the parent-perturbation evidence for one replayed call.

    Both positional AND keyword tensor arguments are scanned; the first array
    found is perturbed. A call with no array anywhere is a recorded evidence
    GAP, never a silent pass — the tripwire cannot prove dependency for a
    constant producer, and claiming it did would be fail-open.

    Parameters
    ----------
    capture:
        Captured call to perturb.
    baseline:
        Replayed baseline output arrays.

    constant_labels:
        Raw labels of constant source ops. An argument whose every leaf is a
        constant is perturbed only when no other array argument exists
        (SPEC section 2 rule 3: the consuming op is perturbed on its
        non-constant parents first).

    Returns
    -------
    tuple[str, bool]
        The evidence (``PERTURBATION_PROVED`` when some candidate changes the
        output, ``PERTURBATION_NO_PERTURBABLE_INPUT`` when the call has no
        tensor argument to perturb, ``PERTURBATION_UNPROVED`` when every
        candidate left the output unchanged) and whether the perturbed
        argument was a constant parent.
    """

    import mlx.core as mx

    target = _perturbation_target(capture, constant_labels)
    if target is None:
        return PERTURBATION_NO_PERTURBABLE_INPUT, False
    arg_position, kwarg_key, original, perturbed_constant = target
    for candidate in _perturb_candidates(original):
        try:
            perturbed_args, perturbed_kwargs = _substituted_call(
                capture, arg_position, kwarg_key, candidate
            )
            perturbed = _iter_output_arrays(capture.func(*perturbed_args, **perturbed_kwargs))
            mx.eval(*perturbed)
        except Exception:
            continue
        if len(perturbed) != len(baseline):
            return PERTURBATION_PROVED, perturbed_constant
        if any(
            not _payloads_close(p_out, b_out)
            for p_out, b_out in zip(perturbed, baseline, strict=True)
        ):
            return PERTURBATION_PROVED, perturbed_constant
    return PERTURBATION_UNPROVED, perturbed_constant


def _coverage_failure_count(trace: Any, captures: tuple[MLXOpCapture, ...]) -> int:
    """Count replay-evidence coverage failures against the immutable inventory.

    Parameters
    ----------
    trace:
        Trace carrying the emit-time ``_mlx_replay_inventory``.
    captures:
        Surviving replay capture records.

    Returns
    -------
    int
        Zero only when the capture records exactly cover the inventory —
        including each record's emit-time per-leaf parent-label fingerprint —
        AND every non-input materialized op label is inventoried. Any missing,
        extra, uninventoried, or label-tampered entry counts, so neither
        partial evidence deletion nor stripping a capture's recorded
        provenance (which would let replay silently reuse emit-time argument
        values) can yield a vacuous pass.
    """

    inventory = getattr(trace, "_mlx_replay_inventory", None)
    if inventory is None:
        return 1
    expected = sorted(
        (
            name,
            tuple(labels),
            tuple(tuple(slot) for slot in arg_labels),
            tuple((key, tuple(slot)) for key, slot in kwarg_labels),
            tuple(interventions),
        )
        for name, labels, arg_labels, kwarg_labels, interventions in tuple(inventory)
    )
    observed = sorted(
        (
            capture.op_name,
            tuple(capture.labels_raw),
            tuple(tuple(slot) for slot in capture.arg_leaf_labels),
            tuple(sorted((key, tuple(slot)) for key, slot in capture.kwarg_leaf_labels.items())),
            tuple(capture.interventions),
        )
        for capture in captures
    )
    failures = 0
    if expected != observed:
        expected_only = [call for call in expected if call not in observed]
        observed_only = [call for call in observed if call not in expected]
        failures += max(1, len(expected_only) + len(observed_only))
    inventoried_labels = {
        label for _name, labels, _args, _kwargs, _fires in expected for label in labels
    }
    for op in trace._layer_list:
        if getattr(op, "is_input", False):
            continue
        label = getattr(op, "_label_raw", None)
        if isinstance(label, str) and label not in inventoried_labels:
            failures += 1
    return failures


def validate_mlx_captures(trace: Any) -> tuple[int, int, tuple[str, ...]]:
    """Replay every captured MLX call; the count-only view of the oracle.

    Parameters
    ----------
    trace:
        Live MLX trace carrying ``_mlx_op_captures`` replay material.

    Returns
    -------
    tuple[int, int, tuple[str, ...]]
        ``(replayed_count, failed_count, perturbation_gap_labels)`` as
        :func:`validate_mlx_captures_detailed` decided them.
    """

    outcome = validate_mlx_captures_detailed(trace)
    return outcome.replayed_count, outcome.failed_count, outcome.perturbation_gaps


def _constant_leaf_decision(
    trace: Any, ops_by_label: dict[str, Any], capture: MLXOpCapture
) -> dict[str, Any]:
    """Decide a closure-constant leaf: its saved payload must exist.

    A constant has no function to replay and nothing upstream to perturb;
    the narrow carve-out (FORKS F-L) exempts exactly that, and only when the
    payload consumers replay from is saved. A constant without a payload is a
    failed check, never an exemption.
    """

    label = capture.labels_raw[0]
    try:
        _saved_payload(trace, ops_by_label, label)
    except ValueError as exc:
        return _decision(
            _DecisionSite(label, "constant"), "leaf", "failed", "payload_missing", str(exc)
        )
    return _decision(
        _DecisionSite(label, "constant"),
        "leaf",
        "exempted",
        "constant_leaf",
        "closure constant: saved payload is the leaf consumers replay from",
    )


def _apply_declared_interventions(
    mx: Any, capture: MLXOpCapture, final: list[Any], site: _DecisionSite
) -> dict[str, Any] | None:
    """Re-apply the declared hooks to ``final`` in place; a missing applier is a failure."""

    # Genuine-intervention carve-out, scoped by the emit-time inventory
    # fingerprint (coverage): replay recomputes the RAW producer output,
    # re-applies the declared hook, and the saved payload must equal
    # hook(raw). A record whose declared interventions lack a matching
    # applier fails closed; a plain capture never reaches this branch.
    declared = dict(capture.interventions)
    appliers = dict(capture.appliers)
    if set(declared) != set(appliers) or any(
        index < 0 or index >= len(final) for index in declared
    ):
        return _decision(
            site,
            "replay",
            "failed",
            "intervention_applier_missing",
            "declared interventions have no matching applier",
        )
    for index, apply in appliers.items():
        final[index] = apply(final[index])
    mx.eval(*final)
    return None


def _saved_payload_mismatch(
    mx: Any,
    trace: Any,
    ops_by_label: dict[str, Any],
    capture: MLXOpCapture,
    final: list[Any],
) -> dict[str, Any] | None:
    """The failed decision when ``final`` differs from the saved payloads, else ``None``."""

    label = capture.labels_raw[0] if capture.labels_raw else "<unlabeled>"
    name = capture.op_name
    expected = tuple(_saved_payload(trace, ops_by_label, lab) for lab in capture.labels_raw)
    mx.eval(*expected)
    # Fail CLOSED on an arity mismatch: pairing positionally without this
    # check would compare only the common prefix.
    if len(final) != len(expected):
        return _decision(
            _DecisionSite(label, name), "replay", "failed", "arity", "output count drifted"
        )
    mismatched = [
        lab
        for lab, f_out, e_out in zip(capture.labels_raw, final, expected, strict=True)
        if not _payloads_close(f_out, e_out)
    ]
    if mismatched:
        return _decision(
            _DecisionSite(mismatched[0], name),
            "replay",
            "failed",
            "value_mismatch",
            "replayed output differs from the saved payload",
        )
    return None


def _perturb_capture(
    capture: MLXOpCapture, replay_args: tuple[Any, ...], replay_kwargs: dict[str, Any]
) -> MLXOpCapture:
    """The capture re-bound to its reconstructed call material for perturbation."""

    return MLXOpCapture(
        labels_raw=capture.labels_raw,
        op_name=capture.op_name,
        func=capture.func,
        args=replay_args,
        kwargs=replay_kwargs,
        output=capture.output,
        arg_leaf_labels=capture.arg_leaf_labels,
        kwarg_leaf_labels=capture.kwarg_leaf_labels,
    )


def _replay_one(
    trace: Any,
    ops_by_label: dict[str, Any],
    capture: MLXOpCapture,
    constant_labels: frozenset[str],
) -> dict[str, Any]:
    """Replay, compare and perturbation-check one captured call.

    Returns
    -------
    dict[str, Any]
        The decision record (``decision`` is ``passed``, ``failed``,
        ``exempted`` for a literal producer, or ``unverified`` for a
        perturbation gap on a non-literal call).
    """

    import mlx.core as mx

    label = capture.labels_raw[0] if capture.labels_raw else "<unlabeled>"
    name = capture.op_name
    if not _declared_parents_consistent(trace, ops_by_label, capture):
        return _decision(
            _DecisionSite(label, name),
            "replay",
            "failed",
            "parents_inconsistent",
            "declared parents differ from the recorded argument leaves",
        )
    replay_args, replay_kwargs = _reconstruct_call(trace, ops_by_label, capture)
    replayed = _iter_output_arrays(capture.func(*replay_args, **replay_kwargs))
    mx.eval(*replayed)
    if len(replayed) != len(capture.labels_raw) or not replayed:
        return _decision(
            _DecisionSite(label, name),
            "replay",
            "failed",
            "arity",
            f"replay produced {len(replayed)} arrays for {len(capture.labels_raw)} labels",
        )
    final = list(replayed)
    if capture.interventions:
        failure = _apply_declared_interventions(mx, capture, final, _DecisionSite(label, name))
        if failure is not None:
            return failure
    mismatch = _saved_payload_mismatch(mx, trace, ops_by_label, capture, final)
    if mismatch is not None:
        return mismatch
    perturb_capture = _perturb_capture(capture, replay_args, replay_kwargs)
    evidence, perturbed_constant = _perturbation_evidence(
        perturb_capture, replayed, constant_labels
    )
    return _perturbation_decision(_DecisionSite(label, name), evidence, perturbed_constant)


def _perturbation_decision(
    site: _DecisionSite, evidence: str, perturbed_constant: bool
) -> dict[str, Any]:
    """The perturbation-step decision for a replayed-equal call (FORKS F-L for literals)."""

    name = site.op_name
    if evidence == PERTURBATION_UNPROVED:
        return _decision(
            site,
            "perturbation",
            "failed",
            "perturbation_unproved",
            "every perturbed parent left the output unchanged",
        )
    if evidence == PERTURBATION_NO_PERTURBABLE_INPUT:
        from .wrappers import MLX_LITERAL_PRODUCER_OPS

        if name in MLX_LITERAL_PRODUCER_OPS:
            # The one by-design case: a literal producer was re-executed from
            # its saved literal arguments and compared equal; it has no
            # tensor parent to perturb (FORKS F-L).
            return _decision(
                site,
                "perturbation",
                "exempted",
                "literal_producer",
                "literal producer replayed equal; no tensor parent to perturb",
            )
        return _decision(
            site,
            "perturbation",
            "unverified",
            "perturbation_gap",
            "no tensor argument to perturb",
        )
    return _decision(
        site,
        "perturbation",
        "passed",
        "constant_parent" if perturbed_constant else None,
        "replayed equal; dependency proven"
        + (" by perturbing a constant parent" if perturbed_constant else ""),
    )


def _constant_source_labels(trace: Any) -> frozenset[str]:
    """Raw labels of the constant source ops in the emit-time replay inventory."""

    return frozenset(
        label
        for name, labels, _args, _kwargs, _fires in tuple(
            getattr(trace, "_mlx_replay_inventory", None) or ()
        )
        if name == "constant"
        for label in labels
    )


def _coverage_decisions(failed_count: int) -> list[dict[str, Any]]:
    """The trace-level coverage decision when the inventory shows failures, else none."""

    if not failed_count:
        return []
    return [
        _decision(
            _DecisionSite("<trace>", None),
            "coverage",
            "failed",
            "coverage",
            f"{failed_count} replay-inventory coverage failure(s)",
        )
    ]


def _capture_record(
    trace: Any,
    ops_by_label: dict[str, Any],
    capture: MLXOpCapture,
    constant_labels: frozenset[str],
) -> dict[str, Any]:
    """The decision record for one capture; a raising replay becomes a failed record."""

    if capture.func is None and capture.op_name == "constant":
        return _constant_leaf_decision(trace, ops_by_label, capture)
    try:
        return _replay_one(trace, ops_by_label, capture, constant_labels)
    except ValueError as exc:
        cause = "payload_missing" if "no saved payload" in str(exc) else "replay_error"
        return _decision(
            _DecisionSite(
                capture.labels_raw[0] if capture.labels_raw else "<unlabeled>",
                capture.op_name,
            ),
            "replay",
            "failed",
            cause,
            str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - a raising replay is a failed check
        return _decision(
            _DecisionSite(
                capture.labels_raw[0] if capture.labels_raw else "<unlabeled>",
                capture.op_name,
            ),
            "replay",
            "failed",
            "replay_error",
            f"{type(exc).__name__}: {exc}",
        )


def validate_mlx_captures_detailed(trace: Any) -> MLXValidationOutcome:
    """Replay every captured MLX call against the trace's saved payloads.

    Coverage is fail-closed: the emit-time ``_mlx_replay_inventory`` is the
    denominator, so partial deletion of replay records fails instead of
    shrinking the evidence set. Replay arguments are reconstructed from the
    saved payloads of each op's DECLARED parents (closure constants included),
    and the declared parent set is cross-checked against the per-leaf labels
    recorded at emit time, so wrong-parent attribution fails either
    structurally or numerically. Constant leaves are exempted only with a
    saved payload; a literal producer is replayed and compared before its
    perturbation step is exempted; every other gap is unverified or failed.

    Parameters
    ----------
    trace:
        Live MLX trace carrying ``_mlx_op_captures`` replay material.

    Returns
    -------
    MLXValidationOutcome
        Counts, per-op decisions and reason-coded tallies.
    """

    ops_by_label = _ops_by_label(trace)
    captures = tuple(getattr(trace, "_mlx_op_captures", ()))
    constant_labels = _constant_source_labels(trace)
    replayed_count = 0
    failed_count = _coverage_failure_count(trace, captures)
    decisions: list[dict[str, Any]] = _coverage_decisions(failed_count)
    exempted: dict[str, int] = {}
    unverified: dict[str, int] = {}
    perturbation_gaps: list[str] = []
    for capture in captures:
        record = _capture_record(trace, ops_by_label, capture, constant_labels)
        decisions.append(record)
        verdict = record["decision"]
        reason = record["reason"]
        if verdict == "failed":
            failed_count += 1
        elif verdict == "exempted":
            exempted[reason] = exempted.get(reason, 0) + 1
            if record["phase"] == "perturbation":
                replayed_count += 1
        elif verdict == "unverified":
            unverified[reason] = unverified.get(reason, 0) + 1
            perturbation_gaps.extend(capture.labels_raw)
            replayed_count += 1
        else:
            replayed_count += 1
    return MLXValidationOutcome(
        replayed_count=replayed_count,
        failed_count=failed_count,
        perturbation_gaps=tuple(perturbation_gaps),
        decisions=tuple(decisions),
        exempted_reason_counts=exempted,
        unverified_reason_counts=unverified,
    )
