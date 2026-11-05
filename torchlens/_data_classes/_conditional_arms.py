"""Derived reads over the arms of a recorded conditional.

``Conditional`` and ``ConditionalArm`` live in :mod:`torchlens._data_classes.trace`
(their module path is part of the saved-file format). The reads here are derived
from their stored fields on demand and persist nothing. :class:`ArmRef` is the
descriptor behind ``ConditionalArm._trace``: an arm holds its Trace weakly, like
every other record view, so the conditional records never keep a dropped Trace
alive for the cyclic collector.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

from ._call_tree import _live_call_trace

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


class _ArmTraceRef:
    """A picklable weak reference from one ``ConditionalArm`` to its Trace.

    Pickling stores the Trace itself, so a Trace pickled with its conditionals
    (the Trace's own pickle state, or ``Trace.__deepcopy__``, which pickles)
    reloads with every arm bound to the reloaded Trace. A reference whose Trace
    is already gone pickles as a dead reference.
    """

    __slots__ = ("_ref",)

    def __init__(self, trace: Any) -> None:
        """Hold ``trace`` weakly; ``None`` makes a dead reference."""

        self._ref = None if trace is None else weakref.ref(trace)

    def __call__(self) -> Any:
        """Return the Trace, or ``None`` once it has been collected."""

        return None if self._ref is None else self._ref()

    def __reduce__(self) -> tuple[Any, tuple[Any]]:
        """Pickle as the referenced Trace (``None`` when it is gone)."""

        return (_ArmTraceRef, (self(),))

    def __copy__(self) -> _ArmTraceRef:
        """Share the reference: it is immutable."""

        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _ArmTraceRef:
        """Point at the Trace's copy when this deep copy made one, else at the Trace.

        A deep copy never copies the Trace through an arm: an arm is a view, so a
        copied arm stays bound to the Trace it viewed unless the same copy
        already produced a copy of that Trace.
        """

        trace = self()
        if trace is None or id(trace) not in memo:
            return self
        return _ArmTraceRef(memo[id(trace)])


class ArmRef:
    """Data descriptor for ``ConditionalArm._trace``: the arm's weak Trace reference.

    Used as the dataclass field's default (``field(default=ArmRef(), ...)``), so the
    generated ``__init__`` and every ``arm._trace = trace`` assignment route through
    :meth:`__set__`, which stores an :class:`_ArmTraceRef` in the instance's
    ``_trace_ref`` slot. Reading returns the live Trace, ``None`` for an arm never
    bound (or detached for saving), and refuses ``trace_reference_collected`` for
    an arm whose Trace was dropped, the same refusal an ``Op`` gives.
    """

    def __get__(self, arm: Any, owner: type | None = None) -> Any:
        """Return the bound Trace (``self`` on class access, the dataclass default)."""

        if arm is None:
            return self
        return _live_call_trace(arm, slot="_trace_ref", record="Conditional")

    def __set__(self, arm: Any, value: Any) -> None:
        """Bind ``value`` weakly; ``None`` (or this default) leaves the arm unbound."""

        unbound = value is None or value is self
        arm.__dict__["_trace_ref"] = None if unbound else _ArmTraceRef(value)

    def __repr__(self) -> str:
        """Render as the field's effective default in signatures: unbound."""

        return "None"


__all__ = [
    "ArmRef",
    "ConditionalOutcome",
    "arm_role_ops",
    "derive_conditional_outcome",
    "outcome_evidence_conflicts",
]
