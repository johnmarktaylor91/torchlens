"""Backward three-state gating and the named counterfactual door (F09).

Costreport D14: the executed-backward answer is keyed on grad state AND
observation -- the historical number was byte-identical under five
configurations where the true executed answer is 0 in four. The three
states:

- ``grad_disabled``: no autograd graph was recorded for this capture --
  executed backward = 0, ``formula_exact``, reason "no backward could
  occur for this capture".
- ``grad_enabled_unobserved``: autograd recorded a graph but no backward
  ran through the capture -- executed backward is UNKNOWN; the
  counterfactual "what would one training backward cost" is requestable
  in every grad state through :func:`backward_estimate` (the named door),
  always labeled hypothetical.
- ``backward_observed``: ``log_backward`` recorded backward passes --
  timing is measured where fire timings exist, and the executed backward
  compute is COUNTED over the ATen calls that backward dispatched (each
  pass's ``BackwardPass.dispatched_ops``), exact for the matmul and
  convolution families at fma2 (FlopCounterMode's convention); every
  recorded op outside those families is named in ``uncounted_ops``, never
  zeroed or estimated. Each pass's count is attributed to the GradFn calls
  and forward ops that did the work, with a named unattributed bucket
  (``BackwardStatus.attributions``); a loaded trace reads the count saved
  with the pass. A pass with neither a dispatch record nor a saved count (an
  implicit pass, an artifact saved before the count was kept) or a pass that
  raised before finishing leaves the executed answer UNKNOWN.

D15/D23 plumbing (derived live-only; F09 holds no adjudicated persisted
fields): ``grad_enabled_at_capture``, per-op ``needs_param_grad`` /
``needs_input_grad``, and the recompute observation all DERIVE from facts
the capture already persists (grad_fn class names, per-op trainable/frozen
param counts, the L9 checkpoint witness). The MFU/HFU recompute
classification stays ``unknown`` unless checkpoint context was observed.

Every quantity declares its execution precondition; a counterfactual
never prints in an actual-cost slot (memo 3.8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..capture.flops import DISPATCH_COUNTING_CONVENTION
from ._backward_attribution import node_resolver_missing, pass_failed

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

#: Closed state vocabulary (D14).
BACKWARD_STATES: tuple[str, ...] = (
    "grad_disabled",
    "grad_enabled_unobserved",
    "backward_observed",
)

#: The multiplier table the named door cites (D14): backward of a
#: MAC-family op computes two GEMM-shaped products (dX = W^T dY and
#: dW = dY X^T), so the standard training estimate is 2.0x forward for
#: MAC-family work. Ops outside the table are EXCLUDED AND NAMED -- a 1.0
#: default is not an estimate of anything (D14).
BACKWARD_MULTIPLIER_TABLE: dict[str, float] = {"mac_family": 2.0}


def grad_enabled_at_capture(trace: Any) -> bool:
    """Whether autograd recorded a graph during this capture (derived).

    True exactly when at least one captured op carries a recorded
    ``grad_fn`` class -- under ``no_grad`` / ``inference_mode`` / an
    all-frozen model, no op does. A derived read over persisted facts,
    never a new field.
    """

    return any(_has_grad_fn(op) for op in getattr(trace, "layer_list", ()) or ())


def _has_grad_fn(op: Any) -> bool:
    """Whether one op carries a REAL recorded grad_fn class.

    Boundary rows record the literal string ``"none"`` rather than an
    absent value; both spellings mean "no grad_fn".
    """

    name = getattr(op, "grad_fn_class_name", None)
    return bool(name) and str(name).lower() != "none"


def op_needs_param_grad(op: Any) -> bool | None:
    """Whether this op's own parameters require grad (derived, D15).

    ``None`` when the op owns no parameters (the question does not apply).
    """

    num_params = getattr(op, "num_params", None)
    if not num_params:
        return None
    trainable = getattr(op, "num_params_trainable", None)
    if trainable is None:
        return None
    return int(trainable) > 0


def op_needs_input_grad(op: Any) -> bool | None:
    """Whether grad flowed into this op at capture (derived, D15).

    Reads the recorded autograd evidence: an op with a recorded
    ``grad_fn`` participated in the graph. ``None`` when the capture
    recorded no autograd facts at all (grad disabled).
    """

    name = getattr(op, "grad_fn_class_name", None)
    if name is None:
        return None
    return _has_grad_fn(op)


def recompute_observation(trace: Any) -> str:
    """The D23 checkpoint-recompute observation: never guessed.

    ``"observed"`` when the L9 checkpoint witness recorded checkpoint
    context during capture (a minted checkpoint invocation, or a reentrant
    checkpoint node in the walked graph); ``"none_observed"`` when the
    witness exists and recorded nothing; ``"unknown"`` when no witness is
    available or its evidence is incomplete -- and the MFU/HFU
    classification of duplicate passes stays ``unknown`` in that case (a
    recurrent second pass is model arithmetic; a checkpoint recompute is
    not; the capture can only tell them apart when it observed checkpoint
    context).
    """

    witness = getattr(trace, "checkpoint_invocation_witness", None)
    if not isinstance(witness, dict):
        return "unknown"
    verdict = witness.get("verdict")
    if verdict == "checkpoint_invocations_observed":
        return "observed"
    if verdict == "no_checkpoint_invocation_observed":
        return "none_observed"
    if _REENTRANT_CHECKPOINT_FLAG in (witness.get("degrade_flags") or ()):
        return "observed"
    return "unknown"


#: The witness degrade flag set when the walked graph holds a reentrant
#: checkpoint node: checkpointing certainly ran, though its tokens were not
#: minted.
_REENTRANT_CHECKPOINT_FLAG = "reentrant_node_discovered"


#: The executed-backward counting convention (matches FlopCounterMode's).
EXECUTED_BACKWARD_CONVENTION = DISPATCH_COUNTING_CONVENTION

#: Uncounted op names shown on the status line before it points at the field.
_STATUS_LINE_NAME_LIMIT = 5


@dataclass(frozen=True)
class BackwardStatus:
    """The always-visible backward STATUS facts (D14).

    ``executed_flops`` is the ACTUAL executed backward compute: ``0``
    (exact) in the ``grad_disabled`` state; ``None`` (unknown) in the
    ``grad_enabled_unobserved`` state; in the ``backward_observed`` state the
    count over the ATen calls the observed backward dispatched, exact for
    the matmul and convolution families (``counting_convention``), or
    ``None`` when a pass has no dispatch record on this object.
    ``executed_flops_per_pass`` pairs each observed pass index with its
    count (``None`` where unrecorded); ``uncounted_ops`` names every recorded
    op kind outside the counted families with its call count -- never
    zeroed, never estimated. ``measured_seconds`` is per-fire backward
    timing when the live trace recorded it. ``attributions`` holds one
    ``BackwardFlopsAttribution`` per counted pass (rows by GradFn call and
    forward op); ``unattributed_flops`` is the counted work that ran under no
    recorded node, so the rows plus it equal ``executed_flops`` exactly;
    ``recompute_flops`` is the part that replayed forward kernels under
    activation checkpointing (``None`` when the evidence cannot tell). The
    three are empty or ``None`` outside the ``backward_observed`` state.
    """

    state: str
    executed_flops: int | None
    executed_evidence: str
    reason: str
    n_backward_passes: int
    n_grad_fn_records: int
    measured_seconds: float | None
    recompute: str
    counting_convention: str = EXECUTED_BACKWARD_CONVENTION
    executed_flops_per_pass: tuple[tuple[int, int | None], ...] = ()
    uncounted_ops: tuple[tuple[str, int], ...] = ()
    attributions: tuple[Any, ...] = ()
    unattributed_flops: int | None = None
    recompute_flops: int | None = None

    @property
    def status_line(self) -> str:
        """The one-line STATUS render every cost report carries."""

        if self.state == "grad_disabled":
            return (
                "backward: executed backward = 0 (formula_exact; no backward could occur "
                "for this capture -- autograd recorded no graph). Counterfactual training "
                "cost available via tl.report.backward_estimate(trace) [hypothetical]."
            )
        if self.state == "grad_enabled_unobserved":
            return (
                "backward: grad was enabled but no backward pass was observed -- executed "
                "backward compute is UNKNOWN (not zero, not estimated). Counterfactual "
                "training cost: tl.report.backward_estimate(trace) [hypothetical]."
            )
        timing = (
            f"measured backward time {self.measured_seconds:.6f} s"
            if self.measured_seconds is not None
            else "per-fire timing unavailable on this object"
        )
        head = (
            f"backward: {self.n_backward_passes} backward pass(es) observed "
            f"({self.n_grad_fn_records} grad_fn records; {timing}); "
        )
        if self.executed_flops is None:
            return head + f"executed backward compute: UNKNOWN ({self.reason})."
        return (
            head
            + f"executed backward compute: {self.executed_flops} FLOPs counted from the "
            + "backward ops that ran (fma2; matmul and convolution families exact)"
            + self._attribution_clause()
            + self._uncounted_clause()
            + "."
        )

    def _attribution_clause(self) -> str:
        """Name the unattributed and recompute shares when they are nonzero."""

        parts = []
        if self.unattributed_flops:
            parts.append(f"{self.unattributed_flops} unattributed (ran under no recorded node)")
        if self.recompute_flops:
            parts.append(f"{self.recompute_flops} recomputed forward under checkpointing")
        elif self.recompute_flops is None and self.recompute == "observed":
            parts.append("recompute share unknown")
        return "; of which " + ", ".join(parts) if parts else ""

    def _uncounted_clause(self) -> str:
        """Name the recorded op kinds outside the counted families (bounded)."""

        if not self.uncounted_ops:
            return ""
        names = [name for name, _count in self.uncounted_ops]
        shown = ", ".join(names[:_STATUS_LINE_NAME_LIMIT])
        more = len(names) - _STATUS_LINE_NAME_LIMIT
        tail = f" (+{more} more; read .uncounted_ops)" if more > 0 else ""
        return f"; {len(names)} op kind(s) outside those families not counted: {shown}{tail}"


@dataclass(frozen=True)
class _ExecutedCount:
    """The observed-state count folded over every pass's attribution."""

    total: int | None
    reason: str
    per_pass: tuple[tuple[int, int | None], ...]
    uncounted: tuple[tuple[str, int], ...]
    attributions: tuple[Any, ...]
    unattributed: int | None
    recompute: int | None


def _executed_backward_count(trace: Any) -> _ExecutedCount:
    """Count the observed backward pass by pass from its attribution.

    ``total`` is ``None`` when a pass has no record or a counted-family call
    could not be read. Each pass's count is its attributed rows plus its
    unattributed bucket, so the total and the rows agree exactly.
    """

    per_pass: list[tuple[int, int | None]] = []
    uncounted: dict[str, int] = {}
    attributions: list[Any] = []
    missing: list[int] = []
    failed: list[int] = []
    unresolved: list[int] = []
    unreadable: list[int] = []
    other_trace = 0
    for pass_index, record in sorted((getattr(trace, "backward_pass_logs", {}) or {}).items()):
        attribution = getattr(record, "flops_attribution", None)
        if attribution is None:
            missing.append(pass_index)
            per_pass.append((pass_index, None))
            continue
        attributions.append(attribution)
        per_pass.append((pass_index, attribution.executed_flops))
        if pass_failed(record):
            failed.append(pass_index)
        elif attribution.source == "census" and node_resolver_missing():
            unresolved.append(pass_index)
        elif attribution.executed_flops is None:
            unreadable.append(pass_index)
        for name, count in attribution.uncounted_ops:
            uncounted[name] = uncounted.get(name, 0) + count
        other_trace += attribution.other_trace_ops or 0
    named = tuple(uncounted.items())
    unattributed = _sum_known(item.unattributed_flops for item in attributions)
    recompute = _sum_known(item.recompute_flops for item in attributions)
    unknown = _unknown_count_reason(missing, failed, unresolved, unreadable)
    if unknown is not None:
        return _ExecutedCount(
            None, unknown, tuple(per_pass), named, tuple(attributions), None, None
        )
    total = sum(count for _index, count in per_pass if count is not None)
    reason = (
        "counted from the ATen calls the observed backward dispatched, by GradFn call; "
        f"{len(named)} op kind(s) outside the counted families are named, not zeroed"
    )
    if other_trace:
        reason += (
            f"; {other_trace} call(s) that ran on another trace's nodes in a shared "
            "backward belong to that trace"
        )
    return _ExecutedCount(
        total, reason, tuple(per_pass), named, tuple(attributions), unattributed, recompute
    )


def _unknown_count_reason(
    missing: list[int], failed: list[int], unresolved: list[int], unreadable: list[int]
) -> str | None:
    """Why the executed backward has no count, first cause first, or ``None``."""

    def listed(indices: list[int]) -> str:
        """The pass indices as a comma-separated list."""
        return ", ".join(str(index) for index in indices)

    if missing:
        return (
            f"backward pass(es) {listed(missing)} carry no dispatch record on this object "
            "(an implicit pass opened by tensor hooks, or a trace saved before the "
            "count was kept)"
        )
    if failed:
        return (
            f"backward pass(es) {listed(failed)} raised before finishing: the executed "
            "backward was not run to completion, so it has no count (the calls dispatched "
            "before the raise stay on dispatched_ops)"
        )
    if unresolved:
        return (
            f"this torch build has no executing-node resolver (HAS_CURRENT_AUTOGRAD_NODE "
            f"is False), so the calls of backward pass(es) {listed(unresolved)} cannot be "
            "charged to the nodes that ran them and have no count"
        )
    if unreadable:
        return f"counted-family call(s) with unreadable shapes in backward pass(es) {listed(unreadable)}"
    return None


def _sum_known(values: Any) -> int | None:
    """Sum of ints, ``None`` when any value is unknown."""

    total = 0
    for value in values:
        if value is None:
            return None
        total += int(value)
    return total


def backward_status(trace: Trace) -> BackwardStatus:
    """Derive the three-state backward status for one finished trace."""

    grad_fn_logs = getattr(trace, "grad_fn_logs", {}) or {}
    backward_passes = getattr(trace, "backward_pass_logs", {}) or {}
    per_pass: tuple[tuple[int, int | None], ...] = ()
    uncounted: tuple[tuple[str, int], ...] = ()
    attributions: tuple[Any, ...] = ()
    unattributed: int | None = None
    recompute_flops: int | None = None
    if not grad_enabled_at_capture(trace):
        state = "grad_disabled"
        executed: int | None = 0
        evidence = "formula_exact"
        reason = "no backward could occur for this capture (no autograd graph recorded)"
    elif not backward_passes:
        state = "grad_enabled_unobserved"
        executed = None
        evidence = "unknown"
        reason = "grad enabled at capture; no backward pass observed"
    else:
        state = "backward_observed"
        counted = _executed_backward_count(trace)
        executed, reason = counted.total, counted.reason
        per_pass, uncounted = counted.per_pass, counted.uncounted
        attributions = counted.attributions
        unattributed, recompute_flops = counted.unattributed, counted.recompute
        evidence = "unknown" if executed is None else "formula_exact"
    measured: float | None = None
    if backward_passes:
        try:
            timings = trace.grad_fn_fire_timings
        except Exception:  # noqa: BLE001 -- loaded/cleaned traces refuse typed; status degrades
            timings = None
        if timings:
            # Each value is one fire's span (a Duration), or None for an untimed fire.
            values = [float(span) for span in timings.values() if span is not None]
            if values:
                measured = sum(values)
    return BackwardStatus(
        state=state,
        executed_flops=executed,
        executed_evidence=evidence,
        reason=reason,
        n_backward_passes=len(backward_passes),
        n_grad_fn_records=len(grad_fn_logs),
        measured_seconds=measured,
        recompute=recompute_observation(trace),
        executed_flops_per_pass=per_pass,
        uncounted_ops=uncounted,
        attributions=attributions,
        unattributed_flops=unattributed,
        recompute_flops=recompute_flops,
    )


@dataclass(frozen=True)
class BackwardEstimate:
    """The named-door training-backward counterfactual (D14): HYPOTHETICAL.

    ``hypothetical_flops`` multiplies MAC-family forward work by the cited
    table; ops outside the table are EXCLUDED AND NAMED in
    ``excluded_ops`` (never silently multiplied by 1.0), and the freeze
    pattern is disclosed: the multiplier is nearer 1.0x where weights are
    frozen (only dX is computed, not dW).
    """

    label: str
    hypothetical_flops: int
    multiplier_table: dict[str, float]
    n_ops_multiplied: int
    n_ops_frozen_weights: int
    excluded_ops: tuple[str, ...]
    freeze_pattern_from_captured_requires_grad: bool
    grad_disabled_at_capture: bool

    def __repr__(self) -> str:
        """Bounded identity card (F10/D31): the label leads, exclusions count.

        ``excluded_ops`` scales with the model, so the card counts it and
        points at ``.disclosure`` -- the HYPOTHETICAL label is never dropped.
        """

        return (
            f"BackwardEstimate({self.label}: {self.hypothetical_flops} FLOPs = "
            f"{self.multiplier_table['mac_family']}x over {self.n_ops_multiplied} MAC ops, "
            f"{self.n_ops_frozen_weights} frozen-weight, "
            f"{len(self.excluded_ops)} excluded; read .disclosure)"
        )

    @property
    def disclosure(self) -> str:
        """The mandatory disclosure block for any render of this figure."""

        lines = [
            "HYPOTHETICAL training-backward estimate (never an executed cost):",
            (
                f"{self.multiplier_table['mac_family']}x applied to "
                f"{self.n_ops_multiplied} MAC-family ops"
            ),
        ]
        if self.n_ops_frozen_weights:
            lines.append(
                f"{self.n_ops_frozen_weights} of them have fully frozen weights where the "
                "true multiplier is nearer 1.0x (dW is never computed)"
            )
        if self.excluded_ops:
            lines.append(
                f"{len(self.excluded_ops)} non-MAC op(s) excluded from the estimate "
                "(no validated multiplier; a 1.0 default is not an estimate)"
            )
        if self.grad_disabled_at_capture:
            lines.append(
                "grad was globally disabled at capture: the freeze pattern was read from "
                "requires_grad as captured and may not reflect training intent"
            )
        return "\n".join(lines)


def backward_estimate(trace: Trace) -> BackwardEstimate:
    """Request the training-backward counterfactual (the named door, D14).

    Requestable in every grad state; the result is ALWAYS labeled
    hypothetical and never enters an actual-cost slot.
    """

    from ._compute_truth import classify_row

    multiplier = BACKWARD_MULTIPLIER_TABLE["mac_family"]
    total = 0.0
    multiplied = 0
    frozen = 0
    excluded: list[str] = []
    for op in getattr(trace, "layer_list", ()) or ():
        if classify_row(op) != "known":
            continue
        record = getattr(op, "compute_record", None)
        flops = getattr(op, "flops_forward", None)
        if flops is None or int(flops) == 0:
            continue
        if record is not None and getattr(record, "mac_applicability", None) == "mac":
            multiplied += 1
            total += multiplier * int(flops)
            if op_needs_param_grad(op) is False:
                frozen += 1
        else:
            excluded.append(str(getattr(op, "layer_label", "?")))
    return BackwardEstimate(
        label="hypothetical",
        hypothetical_flops=int(total),
        multiplier_table=dict(BACKWARD_MULTIPLIER_TABLE),
        n_ops_multiplied=multiplied,
        n_ops_frozen_weights=frozen,
        excluded_ops=tuple(excluded),
        freeze_pattern_from_captured_requires_grad=True,
        grad_disabled_at_capture=not grad_enabled_at_capture(trace),
    )
