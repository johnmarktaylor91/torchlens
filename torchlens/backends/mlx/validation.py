"""Live replay validation for the technical-preview MLX backend.

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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._validation_shared import float_replay_tolerances, ops_by_label as _ops_by_label


class _ReplaySlot:
    """Sentinel standing in for a labeled array leaf in a capture template.

    Replay never reads the stored value of a labeled leaf — it substitutes
    the DECLARED parent's saved payload — so retaining the emit-time array
    only pinned every intermediate activation for the lifetime of the trace.
    The sentinel keeps the container structure and flatten order intact
    (paddle's ``_template_value`` retention model).
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the sentinel's stable display form."""

        return "<mlx-replay-slot>"


REPLAY_SLOT = _ReplaySlot()


@dataclass(frozen=True)
class MLXOpCapture:
    """One captured MLX call retained for live replay validation.

    Parameters
    ----------
    labels_raw:
        Raw TorchLens labels reserved for the call's output arrays, in
        emission order.
    op_name:
        Wrapped operation name.
    func:
        Original (unwrapped) callable invoked by the wrapper.
    args:
        Positional argument templates: labeled array leaves are replaced by
        ``REPLAY_SLOT`` (replay sources them from saved parent payloads);
        unlabeled leaves (parameters, constants) keep their emit-time values.
    kwargs:
        Keyword argument templates with the same slotting.
    output:
        Unused legacy slot retained for constructor compatibility; the
        expected replay values come from the trace's saved payloads.
    arg_leaf_labels:
        Per positional argument, the raw parent label of each array leaf in
        deterministic flatten order (``None`` for unlabeled leaves such as
        parameters).
    kwarg_leaf_labels:
        The same per-leaf parent labels for keyword arguments, keyed by
        keyword name.
    interventions:
        Declared genuine user interventions as ``(output_leaf_index,
        hook_identity)`` pairs. Mirrored into the emit-time inventory
        fingerprint, so a post-hoc claim cannot steer hook re-application.
    appliers:
        Runtime hook appliers as ``(output_leaf_index, callable)`` pairs,
        used by replay to reproduce the declared substitution. Never part of
        the fingerprint; a record whose declared interventions lack a
        matching applier fails closed.
    """

    labels_raw: tuple[str, ...]
    op_name: str
    func: Any
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    arg_leaf_labels: tuple[tuple[str | None, ...], ...] = ()
    kwarg_leaf_labels: dict[str, tuple[str | None, ...]] = field(default_factory=dict)
    interventions: tuple[tuple[int, str], ...] = ()
    appliers: tuple[tuple[int, Any], ...] = ()


def build_capture_template(
    value: Any,
    leaf_labels: tuple[str | None, ...],
) -> Any:
    """Return ``value`` with labeled array leaves replaced by ``REPLAY_SLOT``.

    Parameters
    ----------
    value:
        Emit-time argument value.
    leaf_labels:
        Recorded per-leaf parent labels for ``value`` in flatten order.

    Returns
    -------
    Any
        Template retaining structure and unlabeled leaves only.
    """

    cursor = iter(leaf_labels)

    def _slot(node: Any) -> Any:
        """Replace each labeled array leaf with ``REPLAY_SLOT``, preserving structure."""

        if _is_mlx_array(node):
            label = next(cursor, None)
            return node if label is None else REPLAY_SLOT
        # Exact container types only: an ``mlx.nn.Module`` is a dict subclass
        # and must stay the module object (its ``__call__`` reads attributes).
        if type(node) is tuple:
            return tuple(_slot(item) for item in node)
        if type(node) is list:
            return [_slot(item) for item in node]
        if type(node) is dict:
            return {key: _slot(item) for key, item in node.items()}
        return node

    return _slot(value)


def _is_mlx_array(value: Any) -> bool:
    """Return whether ``value`` is an ``mlx.core.array``.

    Parameters
    ----------
    value:
        Candidate object.

    Returns
    -------
    bool
        True for MLX arrays.
    """

    try:
        import mlx.core as mx
    except ImportError:
        return False
    return isinstance(value, mx.array)


def _iter_output_arrays(output: Any) -> tuple[Any, ...]:
    """Flatten an MLX call output into its array leaves.

    Parameters
    ----------
    output:
        Raw output object (array, tuple/list of arrays, or other).

    Returns
    -------
    tuple[Any, ...]
        Array leaves in deterministic order.
    """

    if _is_mlx_array(output):
        return (output,)
    if isinstance(output, (tuple, list)):
        leaves: list[Any] = []
        for item in output:
            leaves.extend(_iter_output_arrays(item))
        return tuple(leaves)
    return ()


def _payloads_close(a: Any, b: Any) -> bool:
    """Return whether two MLX arrays match within dtype-aware tolerance.

    Parameters
    ----------
    a, b:
        Arrays to compare.

    Returns
    -------
    bool
        True when shapes/dtypes agree and values are close.
    """

    a_np = np.asarray(a)
    b_np = np.asarray(b)
    if a_np.shape != b_np.shape or a_np.dtype != b_np.dtype:
        return False
    if a_np.dtype.kind in ("f", "c"):
        # Eps-derived per-dtype band, shared with every sibling oracle
        # (b5-opus R17-1: this ladder had hardcoded 1e-5 for EVERY >=4-byte
        # float while the replay oracle went eps-derived, leaving the fp64
        # validation band ~5e8 of fp64's own ULPs; ``np.finfo`` reports
        # component precision for complex).
        rtol, atol = float_replay_tolerances(np.finfo(a_np.dtype))
        return bool(np.allclose(a_np, b_np, rtol=rtol, atol=atol, equal_nan=True))
    return bool(np.array_equal(a_np, b_np))


def _perturb_candidates(value: Any) -> tuple[Any, ...]:
    """Return perturbed variants of one MLX array argument.

    Parameters
    ----------
    value:
        MLX array to perturb.

    Returns
    -------
    tuple[Any, ...]
        Candidate replacement arrays.
    """

    import mlx.core as mx

    value_np = np.asarray(value)
    if value_np.dtype.kind == "f":
        # The magnitude shifts mirror the paddle/tf/tinygrad tripwires: a
        # saturating call (ReLU over inputs all below -0.5) ignores the small
        # shift and the doubling, so a healthy capture would read UNPROVED.
        # A replay that ignores its argument stays UNPROVED under every one.
        magnitude = float(np.max(np.abs(value_np))) + 1.0 if value_np.size else 1.0
        shift = mx.array(magnitude, dtype=value.dtype)
        return (
            value + mx.array(0.5, dtype=value.dtype),
            value * 2,
            value + shift,
            value - shift,
        )
    if value_np.dtype.kind in ("i", "u"):
        return (value + 1, mx.zeros_like(value))
    if value_np.dtype.kind == "b":
        return (mx.logical_not(value),)
    return ()


PERTURBATION_PROVED = "proved"
PERTURBATION_NO_PERTURBABLE_INPUT = "no_perturbable_input"
PERTURBATION_UNPROVED = "unproved"

#: Local cause -> reason code from ``VALIDATION_REASONS`` (SPEC 4.3), coded
#: once. Every per-op decision and every count key reads from this table.
MLX_VALIDATION_REASONS: dict[str, str] = {
    "coverage": "replay_mismatch",
    "parents_inconsistent": "replay_mismatch",
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

    def _all_constant(labels: tuple[str | None, ...]) -> bool:
        """Whether every leaf label in ``labels`` names a constant op."""

        return bool(labels) and all(label in constant_labels for label in labels)

    positions = [index for index, value in enumerate(capture.args) if _is_mlx_array(value)]
    positions.sort(
        key=lambda index: _all_constant(
            capture.arg_leaf_labels[index] if index < len(capture.arg_leaf_labels) else ()
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
            return PERTURBATION_NO_PERTURBABLE_INPUT, False
        kwarg_key = found_key
    if arg_position is not None:
        original = capture.args[arg_position]
        perturbed_constant = _all_constant(
            capture.arg_leaf_labels[arg_position]
            if arg_position < len(capture.arg_leaf_labels)
            else ()
        )
    else:
        original = capture.kwargs[kwarg_key]
        perturbed_constant = _all_constant(capture.kwarg_leaf_labels.get(kwarg_key, ()))
    for candidate in _perturb_candidates(original):
        try:
            if arg_position is not None:
                perturbed_args = (
                    *capture.args[:arg_position],
                    candidate,
                    *capture.args[arg_position + 1 :],
                )
                perturbed_kwargs = capture.kwargs
            else:
                perturbed_args = capture.args
                perturbed_kwargs = {**capture.kwargs, kwarg_key: candidate}
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


def _saved_payload(trace: Any, ops_by_label: dict[str, Any], label_raw: str) -> Any:
    """Return the trace's saved payload for one raw label.

    Parameters
    ----------
    trace:
        Trace being validated.
    ops_by_label:
        Label-to-op index from :func:`_ops_by_label`.
    label_raw:
        Raw label whose payload is required.

    Returns
    -------
    Any
        Saved MLX array (public ``op.out`` or the selective-save hidden copy).
    """

    op = ops_by_label.get(label_raw)
    payload = None if op is None else getattr(op, "out", None)
    if payload is None:
        hidden = getattr(trace, "_selective_save_hidden_payloads", {}) or {}
        payload = hidden.get(label_raw)
    if payload is None:
        raise ValueError(f"MLX validation found no saved payload for {label_raw!r}.")
    return payload


def _capture_parent_labels(capture: MLXOpCapture) -> set[str]:
    """Return every recorded parent label across a capture's argument leaves.

    Parameters
    ----------
    capture:
        Captured call.

    Returns
    -------
    set[str]
        Non-``None`` per-leaf parent labels.
    """

    labels = {label for slot in capture.arg_leaf_labels for label in slot if label is not None}
    for slot in capture.kwarg_leaf_labels.values():
        labels.update(label for label in slot if label is not None)
    return labels


def _declared_parents_consistent(
    trace: Any,
    ops_by_label: dict[str, Any],
    capture: MLXOpCapture,
) -> bool:
    """Cross-check a capture's parent labels against the trace's declared graph.

    Parameters
    ----------
    trace:
        Trace being validated.
    ops_by_label:
        Label-to-op index from :func:`_ops_by_label`.
    capture:
        Captured call whose output ops carry the declared parent edges.

    Returns
    -------
    bool
        ``True`` when, for every materialized output op, the declared parent
        op set equals the parent op set implied by the recorded argument
        leaves (the op itself excluded on both sides, so aliasing outputs
        cannot self-shadow).
    """

    implied_ops = {
        id(ops_by_label[label])
        for label in _capture_parent_labels(capture)
        if label in ops_by_label
    }
    for label_raw in capture.labels_raw:
        op = ops_by_label.get(label_raw)
        if op is None:
            continue
        declared = getattr(op, "parents", None)
        if declared is None:
            return False
        declared_ops = {id(ops_by_label[parent]) for parent in declared if parent in ops_by_label}
        declared_ops.discard(id(op))
        implied_without_self = set(implied_ops)
        implied_without_self.discard(id(op))
        if declared_ops != implied_without_self:
            return False
    return True


def _reconstruct_value(
    trace: Any,
    ops_by_label: dict[str, Any],
    value: Any,
    leaf_labels: tuple[str | None, ...],
) -> Any:
    """Rebuild one argument from the trace's saved parent payloads.

    Parameters
    ----------
    trace:
        Trace being validated.
    ops_by_label:
        Label-to-op index from :func:`_ops_by_label`.
    value:
        Emit-time argument value.
    leaf_labels:
        Recorded per-leaf parent labels for ``value`` in flatten order.

    Returns
    -------
    Any
        ``value`` with every labeled array leaf replaced by the saved payload
        the declared graph stores for that label; unlabeled leaves (parameters,
        constants) keep their emit-time values. Emit-time reuse is legitimate
        ONLY because coverage pins each capture's leaf labels to the immutable
        emit-time inventory fingerprint first — a label stripped after capture
        fails coverage instead of laundering the emit-time array through
        replay.
    """

    cursor = iter(leaf_labels)

    def _rebuild(node: Any) -> Any:
        """Rebuild one template node, binding each slot to its recorded parent label.

        Raises
        ------
        ValueError
            If a ``_ReplaySlot`` has no recorded label, which means the template and
            the label fingerprint disagree. Fail closed rather than guess.
        """

        if _is_mlx_array(node) or isinstance(node, _ReplaySlot):
            label = next(cursor, None)
            if label is None:
                if isinstance(node, _ReplaySlot):
                    # A slot with no recorded label means the template and
                    # label fingerprint disagree; fail closed, never guess.
                    raise ValueError("MLX replay template slot has no recorded parent label.")
                return node
            return _saved_payload(trace, ops_by_label, label)
        # Exact container types only (see ``build_capture_template``): a
        # module instance replays as itself, never as a plain dict.
        if type(node) is tuple:
            return tuple(_rebuild(item) for item in node)
        if type(node) is list:
            return [_rebuild(item) for item in node]
        if type(node) is dict:
            return {key: _rebuild(item) for key, item in node.items()}
        return node

    return _rebuild(value)


def _reconstruct_call(
    trace: Any,
    ops_by_label: dict[str, Any],
    capture: MLXOpCapture,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Rebuild a capture's full call material from declared parent payloads.

    Parameters
    ----------
    trace:
        Trace being validated.
    ops_by_label:
        Label-to-op index from :func:`_ops_by_label`.
    capture:
        Captured call.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Positional and keyword arguments with labeled leaves sourced from the
        trace's saved payloads, so replay exercises the recorded graph wiring
        rather than trusting emit-time argument objects.
    """

    args = tuple(
        _reconstruct_value(
            trace,
            ops_by_label,
            value,
            capture.arg_leaf_labels[index] if index < len(capture.arg_leaf_labels) else (),
        )
        for index, value in enumerate(capture.args)
    )
    kwargs = {
        key: _reconstruct_value(
            trace,
            ops_by_label,
            value,
            capture.kwarg_leaf_labels.get(key, ()),
        )
        for key, value in capture.kwargs.items()
    }
    return args, kwargs


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
    for op in getattr(trace, "layer_list", ()):
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
                _DecisionSite(label, name),
                "replay",
                "failed",
                "intervention_applier_missing",
                "declared interventions have no matching applier",
            )
        for index, apply in appliers.items():
            final[index] = apply(final[index])
        mx.eval(*final)
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
    perturb_capture = MLXOpCapture(
        labels_raw=capture.labels_raw,
        op_name=name,
        func=capture.func,
        args=replay_args,
        kwargs=replay_kwargs,
        output=capture.output,
        arg_leaf_labels=capture.arg_leaf_labels,
        kwarg_leaf_labels=capture.kwarg_leaf_labels,
    )
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
    constant_labels = frozenset(
        label
        for name, labels, _args, _kwargs, _fires in tuple(
            getattr(trace, "_mlx_replay_inventory", None) or ()
        )
        if name == "constant"
        for label in labels
    )
    decisions: list[dict[str, Any]] = []
    replayed_count = 0
    failed_count = _coverage_failure_count(trace, captures)
    if failed_count:
        decisions.append(
            _decision(
                _DecisionSite("<trace>", None),
                "coverage",
                "failed",
                "coverage",
                f"{failed_count} replay-inventory coverage failure(s)",
            )
        )
    exempted: dict[str, int] = {}
    unverified: dict[str, int] = {}
    perturbation_gaps: list[str] = []
    for capture in captures:
        if capture.func is None and capture.op_name == "constant":
            record = _constant_leaf_decision(trace, ops_by_label, capture)
        else:
            try:
                record = _replay_one(trace, ops_by_label, capture, constant_labels)
            except ValueError as exc:
                cause = "payload_missing" if "no saved payload" in str(exc) else "replay_error"
                record = _decision(
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
                record = _decision(
                    _DecisionSite(
                        capture.labels_raw[0] if capture.labels_raw else "<unlabeled>",
                        capture.op_name,
                    ),
                    "replay",
                    "failed",
                    "replay_error",
                    f"{type(exc).__name__}: {exc}",
                )
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
