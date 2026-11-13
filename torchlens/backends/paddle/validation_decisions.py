"""Per-op validation decisions for the Paddle backend.

The protocol contract (``BackendProtocol.validate``) asks every backend for a
status object whose failed form names the first offending op and a reason from
the shared closed list ``VALIDATION_REASONS``. This module holds the Paddle
side of that contract: the decision record shape, the reason spellings Paddle
uses (each one place), and the fold that turns decisions into a status.

The module imports no Paddle; it only shapes data the replay loop produced.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ..._validation_status import ValidationStatus

if TYPE_CHECKING:
    from ..._validation_status import ValidationReplaySource, ValidationStatus

#: Replay of the op's saved inputs did not reproduce its saved output.
REASON_REPLAY_MISMATCH = "replay_mismatch"
#: An input of the op has no recorded producer with a saved payload.
REASON_PARENT_PAYLOAD_MISSING = "parent_payload_missing"
#: The op could not be re-executed from its saved call.
REASON_NO_REPLAYABLE_FUNCTION = "no_replayable_function"
#: A recorded constant: a leaf with a saved payload and nothing upstream.
REASON_CONSTANT_PRODUCER_LEAF = "constant_producer_leaf"
#: The op's only value parents are recorded constants, so no perturbation applies.
REASON_CONSTANT_PARENT_PAYLOAD = "constant_parent_payload"
#: A live intervention replaced the op's output (corroborated or forged).
REASON_USER_INTERVENTION = "user_intervention_replacement"
#: The forward had an untraced effect (an in-place parameter write, an alias).
REASON_EFFECTFUL_REGION = "effectful_region"
#: No perturbation of a value parent changed the op's replayed output.
REASON_PERTURBATION_INSENSITIVE = "perturbation_insensitive"
#: The backend-neutral metadata invariants rejected the trace before replay
#: (a registered status-level reason, ``validation._status.STATUS_REASONS``).
REASON_METADATA_INVARIANT = "metadata_invariant_exception"

PADDLE_VALIDATION_REASONS: frozenset[str] = frozenset(
    {
        REASON_REPLAY_MISMATCH,
        REASON_PARENT_PAYLOAD_MISSING,
        REASON_NO_REPLAYABLE_FUNCTION,
        REASON_CONSTANT_PRODUCER_LEAF,
        REASON_CONSTANT_PARENT_PAYLOAD,
        REASON_USER_INTERVENTION,
        REASON_EFFECTFUL_REGION,
        REASON_PERTURBATION_INSENSITIVE,
        REASON_METADATA_INVARIANT,
    }
)

#: Decision kinds, in torch's ``ValidationDecision`` vocabulary.
OUTCOME_REPLAYED = "validated"
OUTCOME_FAILED = "failed"
OUTCOME_EXEMPTED = "exempted"


def decision(
    label: str | None,
    outcome: str,
    reason: str | None = None,
    detail: str = "",
    *,
    func_name: str | None = None,
) -> dict[str, Any]:
    """Build one per-op decision row in torch's ``ValidationDecision`` shape.

    Parameters
    ----------
    label:
        The op's public label (``layer_label`` when resolvable, else the raw
        capture label), or ``None`` when the failure concerns no single op.
    outcome:
        ``"validated"``, ``"failed"`` or ``"exempted"``.
    reason:
        A reason from :data:`PADDLE_VALIDATION_REASONS`, or ``None`` for a plain
        verified replay.
    detail:
        Free text for humans (stored as ``justification``); never pinned by tests.
    func_name:
        The op's ``func_name`` when known.

    Returns
    -------
    dict[str, Any]
        Keys ``op_label``, ``func_name``, ``decision``, ``reason`` and, when
        ``detail`` is set, ``justification``.
    """

    row: dict[str, Any] = {
        "op_label": label,
        "func_name": func_name,
        "decision": outcome,
        "reason": reason,
    }
    if detail:
        row["justification"] = detail
    return row


@dataclass(frozen=True)
class PaddleReplayOutcome:
    """Folded result of the Paddle replay loop.

    Parameters
    ----------
    decisions:
        Every per-op decision in capture order.
    """

    decisions: tuple[Mapping[str, Any], ...]

    @property
    def replayed(self) -> int:
        """Number of ops replay- and perturbation-verified."""

        return sum(1 for item in self.decisions if item["decision"] == OUTCOME_REPLAYED)

    @property
    def failed(self) -> int:
        """Number of ops that failed a check."""

        return sum(1 for item in self.decisions if item["decision"] == OUTCOME_FAILED)

    @property
    def exempted_reason_counts(self) -> dict[str, int]:
        """Reason counts of ops provably outside the replay contract."""

        counts = Counter(
            str(item["reason"]) for item in self.decisions if item["decision"] == OUTCOME_EXEMPTED
        )
        return dict(sorted(counts.items()))

    def first_failure(self) -> Mapping[str, Any] | None:
        """The first failed decision, or ``None``."""

        return next((item for item in self.decisions if item["decision"] == OUTCOME_FAILED), None)


def fold_status(
    outcome: PaddleReplayOutcome,
    *,
    backend: str,
    source: ValidationReplaySource,
    num_unverified_ops: int,
    payload_load_status: str | None,
) -> ValidationStatus:
    """Fold decisions into a status whose failed form names the op and reason.

    Parameters
    ----------
    outcome:
        Decisions from the replay loop.
    backend:
        Backend name.
    source:
        ``"live"`` or ``"loaded"``.
    num_unverified_ops:
        Importer-region nodes that replay does not cover (zero on failure).
    payload_load_status:
        Payload load status of a loaded trace.

    Returns
    -------
    ValidationStatus
        ``failed`` when any decision failed (also when nothing was replayed,
        which is a failure for Paddle), otherwise the status
        ``from_replay_counts`` derives; its failed message names the first
        failed op and reason.
    """

    decisions: list[Mapping[str, Any]] = list(outcome.decisions)
    if outcome.failed == 0 and outcome.replayed < 1:
        decisions.append(
            decision(
                None,
                OUTCOME_FAILED,
                REASON_NO_REPLAYABLE_FUNCTION,
                "no op of the trace was replay-verified",
            )
        )
    folded = PaddleReplayOutcome(tuple(decisions))

    return ValidationStatus.from_replay_counts(
        backend=backend,
        source=source,
        num_replayed_ops=folded.replayed,
        num_unverified_ops=num_unverified_ops if folded.failed == 0 else 0,
        num_failed_ops=folded.failed,
        payload_load_status=payload_load_status,
        exempted_reason_counts=folded.exempted_reason_counts,
        decisions=tuple(decisions),
    )


def with_failure_message(status: ValidationStatus) -> ValidationStatus:
    """Name the first failed op and its reason in a failed status's message.

    ``ValidationStatus.from_replay_counts`` already words its failed message
    this way; this covers statuses built through ``ValidationStatus.result``.

    Parameters
    ----------
    status:
        A folded status.

    Returns
    -------
    ValidationStatus
        ``status`` unchanged unless it failed with a failed decision on record.
    """

    if status.state != "failed":
        return status
    first = PaddleReplayOutcome(tuple(status.decisions)).first_failure()
    if first is None:
        return status
    reason = first.get("reason") or "unknown"
    label = first.get("op_label")
    message = (
        f"Replay validation failed at {label} ({reason})."
        if label
        else f"Replay validation failed ({reason})."
    )
    return replace(status, message=message)


def public_label(raw_label: str | None, ops_by_label: Mapping[str, Any]) -> str | None:
    """Resolve a raw capture label to the op's public label when possible.

    Parameters
    ----------
    raw_label:
        Raw capture label.
    ops_by_label:
        Trace ops keyed by raw and public labels.

    Returns
    -------
    str | None
        The op's ``layer_label`` (or ``label``), else ``raw_label`` itself.
    """

    if raw_label is None:
        return None
    op = ops_by_label.get(raw_label)
    for attribute in ("layer_label", "label"):
        value = getattr(op, attribute, None) if op is not None else None
        if isinstance(value, str) and value:
            return value
    return raw_label


def reasons_outside(decisions: Iterable[Mapping[str, Any]], allowed: frozenset[str]) -> set[str]:
    """Reasons used by ``decisions`` that ``allowed`` does not list.

    Parameters
    ----------
    decisions:
        Decision mappings.
    allowed:
        The closed reason list to check against.

    Returns
    -------
    set[str]
        Offending reason spellings (empty when all are listed).
    """

    return {
        str(item["reason"])
        for item in decisions
        if item.get("reason") is not None and item["reason"] not in allowed
    }
