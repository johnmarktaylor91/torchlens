"""Replay session: owns values, substitutions, last-use counts and cleanup.

A :class:`ReplaySession` replays a :class:`ReplayPlan` (an ordered list of
:class:`ReplayStep` plus initial values, substitutions and kept labels)
through one replay adapter. It checks the whole plan before the first
framework call (every op replayable, every parent available, no output slot
produced twice) and refuses with :class:`ReplayRefusal`; nothing runs
on a refused plan.

During the run it

- passes one parent value per input occurrence, so a parent used twice is
  passed twice and counted twice;
- commits a multi-output op atomically: all of its slots or none, and a
  wrong output count leaves the session unchanged;
- applies substitutions (the edited sites): a substituted label reads the
  substitute everywhere, including the output slot of the op that produced
  it, and a step whose every slot is substituted does not run;
- releases a value once its last consumer has run, unless the label is kept.

:func:`pullback_step` maps one step's output cotangents onto its parents
with the adapter's ``vjp``, summing the contributions of repeated parents;
:func:`merge_grads` folds those contributions into running totals.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._contract import InputOccurrence, OpRecipe, ReplayAdapter, ReplayRefusal

__all__ = ("ReplayPlan", "ReplaySession", "ReplayStep", "merge_grads", "pullback_step")


@dataclass(frozen=True)
class ReplayStep:
    """One op to replay: the adapter's op object, its parent occurrences in
    call order (repeats kept) and its output slots in emission order."""

    op: Any
    inputs: tuple[InputOccurrence, ...]
    output_slots: tuple[str, ...]

    @property
    def label(self) -> str:
        """The op's label (its first output slot)."""

        return self.output_slots[0]

    @classmethod
    def from_recipe(cls, op: Any, recipe: OpRecipe) -> ReplayStep:
        """Build a step from the op and its recipe."""

        return cls(op=op, inputs=recipe.inputs, output_slots=recipe.output_slots)


@dataclass(frozen=True)
class ReplayPlan:
    """What a :class:`ReplaySession` replays.

    Attributes
    ----------
    steps:
        Ops in graph order.
    initial:
        Values available before the first step, keyed by output label: only
        the cone's sources (a label a step also produces is refused).
    substitutions:
        Edited values keyed by output label; they take precedence over
        initial and replayed values.
    keep:
        Labels whose values outlive their last consumer (the requested
        outputs). ``None`` keeps every value.
    """

    steps: tuple[ReplayStep, ...]
    initial: Mapping[str, Any]
    substitutions: Mapping[str, Any] = field(default_factory=dict)
    keep: Collection[str] | None = None


class ReplaySession:
    """Replay a :class:`ReplayPlan` in order through ``adapter``.

    Parameters
    ----------
    adapter:
        A replay adapter (module or object filling :class:`ReplayAdapter`).
    trace:
        The trace passed through to the adapter.
    plan:
        The steps, initial values, substitutions and kept labels.

    Raises
    ------
    ReplayRefusal
        When the plan is refused; raised before any adapter call other than
        ``can_replay``.
    """

    def __init__(self, adapter: ReplayAdapter, trace: Any, plan: ReplayPlan) -> None:
        self._adapter = adapter
        self._trace = trace
        self._steps = tuple(plan.steps)
        self._subs = dict(plan.substitutions)
        self._keep = None if plan.keep is None else frozenset(plan.keep)
        self._values: dict[str, Any] = dict(plan.initial)
        self._remaining: Counter[str] = Counter(
            occ.parent_label for step in self._steps for occ in step.inputs
        )
        self._next = 0
        self._closed = False
        self._check_plan()

    # --- plan check -------------------------------------------------------

    def _check_plan(self) -> None:
        available = set(self._values) | set(self._subs)
        produced: set[str] = set()
        for step in self._steps:
            self._check_step(step, available, produced)
            produced.update(step.output_slots)
            available.update(step.output_slots)
        stray = set(self._subs) - set(self._values) - produced
        if stray:
            raise ReplayRefusal(
                "substitution", f"substituted labels not in the plan: {sorted(stray)}"
            )

    def _check_step(self, step: ReplayStep, available: set[str], produced: set[str]) -> None:
        if not step.output_slots:
            raise ReplayRefusal("arity", "step has no output slots")
        label = step.label
        clash = (set(step.output_slots) & produced) or (set(step.output_slots) & set(self._values))
        if clash or len(set(step.output_slots)) != len(step.output_slots):
            raise ReplayRefusal(
                "duplicate_slot", f"output slots produced twice: {sorted(clash)}", op_label=label
            )
        missing = [o.parent_label for o in step.inputs if o.parent_label not in available]
        if missing:
            raise ReplayRefusal(
                "missing_parent", f"parents not available: {missing}", op_label=label
            )
        reason = self._adapter.can_replay(self._trace, step.op)
        if reason is not None:
            raise ReplayRefusal("cannot_replay", reason, op_label=label)

    # --- running ----------------------------------------------------------

    @property
    def done(self) -> bool:
        """Whether every step has run."""

        return self._next >= len(self._steps)

    def run(self) -> dict[str, Any]:
        """Run every remaining step and return the live values."""

        while not self.done:
            self.step()
        return self.values()

    def step(self) -> tuple[str, ...]:
        """Run the next step; return its output slots."""

        self._require_open()
        step = self._steps[self._next]
        if all(slot in self._subs for slot in step.output_slots):
            outputs = {slot: self._subs[slot] for slot in step.output_slots}
        else:
            outputs = self._replay(step)
        self._values.update(outputs)
        self._next += 1
        self._release(step)
        return step.output_slots

    def _replay(self, step: ReplayStep) -> dict[str, Any]:
        parents = [self.value(occ.parent_label) for occ in step.inputs]
        result = tuple(self._adapter.forward(self._trace, step.op, parents))
        if len(result) != len(step.output_slots):
            raise ReplayRefusal(
                "arity",
                f"replay produced {len(result)} values for {len(step.output_slots)} slots",
                op_label=step.label,
            )
        return {
            slot: self._subs.get(slot, value)
            for slot, value in zip(step.output_slots, result, strict=True)
        }

    def _release(self, step: ReplayStep) -> None:
        for occ in step.inputs:
            name = occ.parent_label
            self._remaining[name] -= 1
            if self._remaining[name] <= 0 and not self._kept(name):
                self._values.pop(name, None)
        for slot in step.output_slots:
            if self._remaining[slot] <= 0 and not self._kept(slot):
                self._values.pop(slot, None)

    def _kept(self, label: str) -> bool:
        return self._keep is None or label in self._keep

    # --- reading and cleanup ---------------------------------------------

    def value(self, label: str) -> Any:
        """Return the current value of ``label`` (substitutes first)."""

        self._require_open()
        if label in self._subs:
            return self._subs[label]
        try:
            return self._values[label]
        except KeyError:
            raise ReplayRefusal(
                "unknown_label", f"no live value for {label!r}", op_label=label
            ) from None

    def values(self) -> dict[str, Any]:
        """Return a copy of the live values, substitutes applied."""

        self._require_open()
        live = dict(self._values)
        live.update({k: v for k, v in self._subs.items() if k in live})
        return live

    def remaining_uses(self, label: str) -> int:
        """Return how many parent occurrences of ``label`` have not run yet."""

        return max(self._remaining[label], 0)

    def close(self) -> None:
        """Drop every value and substitute; later reads refuse."""

        self._values.clear()
        self._subs.clear()
        self._closed = True

    def __enter__(self) -> ReplaySession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise ReplayRefusal("closed", "the replay session is closed")


def pullback_step(
    adapter: ReplayAdapter,
    trace: Any,
    step: ReplayStep,
    parent_values: Sequence[Any],
    cotangents: Sequence[Any],
) -> dict[str, Any]:
    """Return one step's gradient contribution per parent label.

    ``cotangents`` holds one entry per output slot (``None`` for a slot with
    no incoming gradient). A repeated parent's occurrences are summed through
    ``adapter.add``; a parent whose every occurrence gets ``None`` is absent.

    Raises
    ------
    ReplayRefusal
        When ``vjp`` returns a count other than one per parent occurrence.
    """

    result = tuple(adapter.vjp(trace, step.op, parent_values, cotangents))
    if len(result) != len(step.inputs):
        raise ReplayRefusal(
            "arity",
            f"vjp produced {len(result)} cotangents for {len(step.inputs)} parents",
            op_label=step.label,
        )
    contributions = [(occ.parent_label, cot) for occ, cot in zip(step.inputs, result, strict=True)]
    return merge_grads(adapter, {}, contributions)


def merge_grads(
    adapter: ReplayAdapter,
    grads: Mapping[str, Any],
    contributions: Mapping[str, Any] | Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    """Return a new label-to-gradient mapping: ``grads`` plus ``contributions``
    (``None`` contributions skipped, repeats summed through ``adapter.add``)."""

    merged = dict(grads)
    items = contributions.items() if isinstance(contributions, Mapping) else contributions
    for name, cot in items:
        if cot is None:
            continue
        merged[name] = cot if name not in merged else adapter.add(merged[name], cot)
    return merged
