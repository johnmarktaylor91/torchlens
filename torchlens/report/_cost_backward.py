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
  timing is measured where fire timings exist; analytic executed-backward
  compute stays a ROADMAP item (D15) and reads UNKNOWN here, never an
  unlabeled estimate.

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

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

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
    context during capture; ``"none_observed"`` when the witness exists
    and recorded nothing; ``"unknown"`` when no witness is available --
    and the MFU/HFU classification of duplicate passes stays ``unknown``
    in that case (a recurrent second pass is model arithmetic; a
    checkpoint recompute is not; the capture can only tell them apart
    when it observed checkpoint context).
    """

    witness = getattr(trace, "checkpoint_invocation_witness", None)
    if witness is None:
        return "unknown"
    count = getattr(witness, "invocation_count", None)
    if count is None and isinstance(witness, dict):
        count = witness.get("invocation_count")
    if count is None:
        return "unknown"
    return "observed" if int(count) > 0 else "none_observed"


@dataclass(frozen=True)
class BackwardStatus:
    """The always-visible backward STATUS facts (D14).

    ``executed_flops`` is the ACTUAL executed backward compute: ``0``
    (exact) in the ``grad_disabled`` state, ``None`` (unknown) otherwise
    -- the analytic executed-backward pass is roadmap (D15) and its
    absence never becomes an unlabeled estimate. ``measured_seconds`` is
    per-fire backward timing when the live trace recorded it.
    """

    state: str
    executed_flops: int | None
    executed_evidence: str
    reason: str
    n_backward_passes: int
    n_grad_fn_records: int
    measured_seconds: float | None
    recompute: str

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
        return (
            f"backward: {self.n_backward_passes} backward pass(es) observed "
            f"({self.n_grad_fn_records} grad_fn records; {timing}); executed backward "
            "compute: UNKNOWN until the analytic executed-backward pass lands (roadmap) "
            "-- observing the graph does not make a formula a measurement."
        )


def backward_status(trace: Trace) -> BackwardStatus:
    """Derive the three-state backward status for one finished trace."""

    grad_fn_logs = getattr(trace, "grad_fn_logs", {}) or {}
    backward_passes = getattr(trace, "backward_pass_logs", {}) or {}
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
        executed = None
        evidence = "unknown"
        reason = "backward observed; analytic executed-backward compute is roadmap (D15)"
    measured: float | None = None
    if backward_passes:
        try:
            timings = trace.grad_fn_fire_timings
        except Exception:  # noqa: BLE001 -- loaded/cleaned traces refuse typed; status degrades
            timings = None
        if timings:
            values = [
                float(end) - float(start)
                for start, end in timings.values()
                if start is not None and end is not None
            ]
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
