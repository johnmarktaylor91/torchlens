"""Per-run validation decision log for the JAX backend.

Extracted from ``backend`` so that module stays under its size ceiling. Also
records the verdict of an equation no perturbation candidate moved.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

from ..._data_classes.trace import Trace
from .._constants import CONSTANT_PARENT_PAYLOAD_REASON, CONSTANT_PRODUCER_LEAF_REASON
from ._trace_graph_helpers import _data_parent_arg_positions
from ._zero_annihilator import ZERO_ANNIHILATOR_REASON, _zero_annihilator_justification

if TYPE_CHECKING:
    from .jaxpr import JaxEquationCapture


@dataclass
class _JaxValidationLog:
    """Per-run validation decisions (never stored on the trace: it is not portable state)."""

    decisions: list[dict[str, Any]] = field(default_factory=list)
    first_failure: tuple[str, str, str] | None = None
    exempted_reason_counts: dict[str, int] = field(default_factory=dict)


class _JaxVerdict(NamedTuple):
    """The phase, decision and reason of one per-op validation decision."""

    phase: str
    decision: str
    reason: str


def _record_jax_decision(
    log: _JaxValidationLog,
    op_label: str,
    func_name: str,
    verdict: _JaxVerdict,
    justification: str | None = None,
) -> None:
    """Append one per-op validation decision (torch record shape) to the log.

    The first ``failed`` decision is also kept so the status message can name
    the offending op.
    """

    record: dict[str, Any] = {
        "op_label": op_label,
        "func_name": func_name,
        "phase": verdict.phase,
        "decision": verdict.decision,
        "reason": verdict.reason,
    }
    if justification is not None:
        record["justification"] = justification
    log.decisions.append(record)
    if verdict.decision == "failed" and log.first_failure is None:
        log.first_failure = (op_label, func_name, verdict.reason)


def _record_jax_constant_leaves(trace: Trace, log: _JaxValidationLog) -> None:
    """Record the constants rule's named leaf decisions for ``constant`` source ops.

    A closure constant has no equation to replay (its saved payload is the
    record) and nothing upstream to perturb, so it is exempted under the two
    reasons the constants rule names (item 3).
    """

    for op in trace._layer_list:
        if str(getattr(op, "func_name", "")) != "constant":
            continue
        label = str(getattr(op, "label", ""))
        _record_jax_decision(
            log,
            label,
            "constant",
            _JaxVerdict("replay", "exempted", CONSTANT_PRODUCER_LEAF_REASON),
            "closure constant: its saved payload is the record",
        )
        _record_jax_decision(
            log,
            label,
            "constant",
            _JaxVerdict("perturbation", "exempted", CONSTANT_PARENT_PAYLOAD_REASON),
            "closure constant: no upstream op to perturb",
        )
        for reason in (CONSTANT_PRODUCER_LEAF_REASON, CONSTANT_PARENT_PAYLOAD_REASON):
            log.exempted_reason_counts[reason] = log.exempted_reason_counts.get(reason, 0) + 1


def _record_insensitive_outcome(
    log: _JaxValidationLog,
    labels: tuple[str, str],
    capture: JaxEquationCapture,
    op: Any,
    replay: tuple[Sequence[Any], Sequence[Any]],
) -> bool:
    """Record the verdict for an equation no perturbation candidate moved.

    The zero-annihilator proof (``_zero_annihilator``) is the only credit; anything
    it refuses fails ``perturbation_insensitive``.

    Parameters
    ----------
    log
        Per-run validation decision log.
    labels
        The op label and function name the decision is recorded under.
    capture
        Captured JAX equation metadata.
    op
        Materialized TorchLens operation for the equation.
    replay
        The replay inputs and the saved outputs.

    Returns
    -------
    bool
        True when the zero-annihilator proof holds (recorded as exempted);
        False otherwise (recorded as ``perturbation_insensitive``).
    """

    op_label, func_name = labels
    inputs, saved_outputs = replay
    positions_by_parent: dict[str, list[int]] = {}
    for position, parent_label in _data_parent_arg_positions(op).items():
        positions_by_parent.setdefault(str(parent_label), []).append(position)
    justification = _zero_annihilator_justification(
        str(capture.primitive), positions_by_parent, inputs, saved_outputs
    )
    if justification is None:
        _record_jax_decision(
            log,
            op_label,
            func_name,
            _JaxVerdict("perturbation", "failed", "perturbation_insensitive"),
        )
        return False
    _record_jax_decision(
        log,
        op_label,
        func_name,
        _JaxVerdict("perturbation", "exempted", ZERO_ANNIHILATOR_REASON),
        justification,
    )
    counts = log.exempted_reason_counts
    counts[ZERO_ANNIHILATOR_REASON] = counts.get(ZERO_ANNIHILATOR_REASON, 0) + 1
    return True
