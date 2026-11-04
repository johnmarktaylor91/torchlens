"""Derived reads over the arms of a recorded conditional.

``Conditional`` and ``ConditionalArm`` live in :mod:`torchlens._data_classes.trace`
(their module path is part of the saved-file format). The reads here are derived
from their stored fields on demand and persist nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

#: Which way a conditional went: ``Conditional.outcome``'s values.
ConditionalOutcome: TypeAlias = Literal[
    "then", "elif", "else", "fallthrough", "skipped", "multiple", "unknown"
]

_ARM_KINDS = ("then", "elif", "else", "fallthrough")
_TESTED_ARM_KINDS = ("then", "elif")


def _arm_kind(arm: Any) -> str | None:
    """Return an arm's kind, ``None`` when a legacy record lacks it."""

    return getattr(arm, "kind", None)


def derive_conditional_outcome(arms: Sequence[Any]) -> ConditionalOutcome:
    """Return which way a conditional went, from its arms' recorded evidence.

    Rules, first match wins: one fired arm gives its kind; several give
    ``"multiple"``; with none fired, a ``then``/``elif`` arm whose test read
    ``True`` gives its kind; when every ``then``/``elif`` test was evaluated and
    read ``False`` the result is ``"else"`` or ``"fallthrough"`` when the
    conditional has that arm and ``"skipped"`` when it has neither; anything
    else is ``"unknown"``. Missing fields on legacy records read as no evidence.

    Parameters
    ----------
    arms:
        The conditional's arms, in ``Conditional.arms`` order.

    Returns
    -------
    ConditionalOutcome
        One of ``then``, ``elif``, ``else``, ``fallthrough``, ``skipped``,
        ``multiple`` or ``unknown``.
    """

    fired = [arm for arm in arms if getattr(arm, "fired", False)]
    if len(fired) > 1:
        return "multiple"
    if fired:
        kind = _arm_kind(fired[0])
        return kind if kind in _ARM_KINDS else "unknown"  # type: ignore[return-value]
    tested = [arm for arm in arms if _arm_kind(arm) in _TESTED_ARM_KINDS]
    for arm in tested:
        if getattr(arm, "bool_value_at_run", None) is True:
            return _arm_kind(arm)  # type: ignore[return-value]
    every_test_false = bool(tested) and all(
        getattr(arm, "condition_evaluated", False)
        and getattr(arm, "bool_value_at_run", None) is False
        for arm in tested
    )
    if not every_test_false:
        return "unknown"
    present = {_arm_kind(arm) for arm in arms}
    for kind in ("else", "fallthrough"):
        if kind in present:
            return kind  # type: ignore[return-value]
    return "skipped"


def outcome_evidence_conflicts(arms: Sequence[Any]) -> list[str]:
    """Return disagreements between fired arms and witnessed test values.

    With exactly one fired arm, a ``then``/``elif`` arm that fired must not
    have read ``False``, and no other ``then``/``elif`` arm may have read
    ``True``. ``None`` (no single witnessed value) never conflicts.

    Parameters
    ----------
    arms:
        The conditional's arms, in ``Conditional.arms`` order.

    Returns
    -------
    list[str]
        One message per conflict; empty when the evidence agrees.
    """

    fired_indices = [index for index, arm in enumerate(arms) if getattr(arm, "fired", False)]
    if len(fired_indices) != 1:
        return []
    conflicts: list[str] = []
    for index, arm in enumerate(arms):
        if _arm_kind(arm) not in _TESTED_ARM_KINDS:
            continue
        value = getattr(arm, "bool_value_at_run", None)
        if index == fired_indices[0] and value is False:
            conflicts.append(f"arm {index} ({_arm_kind(arm)}) fired but its test read False")
        elif index != fired_indices[0] and value is True:
            conflicts.append(
                f"arm {fired_indices[0]} fired but arm {index} ({_arm_kind(arm)}) test read True"
            )
    return conflicts


def arm_role_ops(
    trace: Any,
    conditional_id: str | None,
    arm_index: int | None,
    role: Literal["evaluation", "body"],
) -> list[str]:
    """Return labels of the ops holding ``role`` in one conditional arm.

    Parameters
    ----------
    trace:
        Trace containing the role-bearing ops, or ``None`` for an unbound arm.
    conditional_id:
        Owning ``Conditional.id``.
    arm_index:
        Position of the arm within ``Conditional.arms``.
    role:
        Conditional role to collect.

    Returns
    -------
    list[str]
        Participating op labels in trace order; empty for an unbound arm.
    """

    if trace is None or conditional_id is None or arm_index is None:
        return []
    labels: list[str] = []
    for op in trace._layer_list:
        if any(
            ref.conditional_id == conditional_id and ref.arm_index == arm_index and ref.role == role
            for ref in op.in_conditionals or []
        ):
            labels.append(op.layer_label)
    return labels


__all__ = [
    "ConditionalOutcome",
    "arm_role_ops",
    "derive_conditional_outcome",
    "outcome_evidence_conflicts",
]
