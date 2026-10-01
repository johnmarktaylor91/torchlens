"""RenderIR-derived answer keys for the naive battery (memo section 7).

Answer keys are GENERATED from the machine record of the render (the lens
resolution, the trace's own graph, and the channel values); hand-maintained
keys are prohibited -- a hand key encodes the author's conventions, which
is exactly the knowledge the battery is designed to exclude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...._errors import InvalidArgumentError
from .._families import scalar_or_none

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

__all__ = ["AnswerKey", "KeyQuestion", "generate_answer_key"]


@dataclass(frozen=True)
class KeyQuestion:
    """One battery question with its machine-derived answer.

    ``kind`` is the probe family (``extrema_max`` / ``extrema_min`` /
    ``status_nonfinite`` / ``ranking_pair`` / ``path_reachability`` /
    ``bridged_vs_direct`` / ``io_identification`` / ``direction``);
    ``honesty_class`` names the zero-tolerance class the question guards,
    when it guards one.
    """

    id: str
    kind: str
    question: str
    answer: Any
    distractors: tuple[str, ...] = ()
    honesty_class: str | None = None


@dataclass(frozen=True)
class AnswerKey:
    """A full generated key for one rendered artifact."""

    member: str
    lens: str
    questions: tuple[KeyQuestion, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable key."""

        return {
            "member": self.member,
            "lens": self.lens,
            "questions": [question.__dict__ for question in self.questions],
        }


def _channel_values(trace: Trace, member: str) -> dict[str, float]:
    """Collect the resolved member's values by op label."""

    values: dict[str, float] = {}
    for op in trace.ops:
        value = scalar_or_none(getattr(op, member, None))
        if value is not None:
            values[op.layer_label] = value
    return values


def _reachable(trace: Trace, source_label: str, target_label: str) -> bool:
    """Directed reachability over the trace's recorded children edges."""

    frontier = [source_label]
    seen: set[str] = set()
    while frontier:
        label = frontier.pop()
        if label == target_label:
            return True
        if label in seen:
            continue
        seen.add(label)
        try:
            op = trace[label]
        except (KeyError, InvalidArgumentError):
            # A frontier label the trace cannot serve (boundary pseudo-label)
            # is simply not on any path; the walk continues past it.
            continue
        frontier.extend(op.children)
    return False


def generate_answer_key(
    trace: Trace,
    *,
    member_name: str,
    resolution: Any = None,
) -> AnswerKey:
    """Generate the answer key for one rendered lens artifact.

    Every answer derives from the trace record and the resolution: extrema
    and ranking from the resolved channel values, status from the derived
    nonfinite channel, paths from the recorded DAG, bridged-vs-direct from
    the compiled filter, boundaries from the boundary flags.
    """

    questions: list[KeyQuestion] = []
    lens_name = resolution.lens.name if resolution is not None else "bare"

    inputs = [op.layer_label for op in trace.ops if getattr(op, "is_input", False)]
    outputs = [op.layer_label for op in trace.ops if getattr(op, "is_final_output", False)]
    if inputs and outputs:
        questions.append(
            KeyQuestion(
                id="io-1",
                kind="io_identification",
                question="Where does information enter and leave this graph?",
                answer={"inputs": inputs, "outputs": outputs},
            )
        )
        questions.append(
            KeyQuestion(
                id="path-1",
                kind="path_reachability",
                question=(
                    f"Is there a path from {inputs[0]!r} to {outputs[0]!r} following the arrows?"
                ),
                answer=_reachable(trace, inputs[0], outputs[0]),
            )
        )

    if resolution is not None and resolution.source is not None:
        values = _channel_values(trace, resolution.source.member)
        if values:
            max_label = max(values, key=values.__getitem__)
            min_label = min(values, key=values.__getitem__)
            others = sorted(label for label in values if label != max_label)[:3]
            questions.append(
                KeyQuestion(
                    id="extrema-max",
                    kind="extrema_max",
                    question="Which node is HIGHEST on the encoded colour scale?",
                    answer=max_label,
                    distractors=tuple(others),
                )
            )
            questions.append(
                KeyQuestion(
                    id="extrema-min",
                    kind="extrema_min",
                    question="Which node is LOWEST on the encoded colour scale?",
                    answer=min_label,
                )
            )
            if len(values) >= 2 and max_label != min_label:
                questions.append(
                    KeyQuestion(
                        id="ordinal-1",
                        kind="ranking_pair",
                        question=(
                            f"The scale is ordinal. Is {max_label!r} necessarily "
                            f"MANY TIMES larger than {min_label!r}?"
                        ),
                        answer=False,
                        honesty_class="ORDINAL-as-RATIO",
                    )
                )

    if resolution is not None and resolution.nonfinite is not None:
        channel = resolution.nonfinite
        nan_labels = sorted(
            label for label, state in channel.states.items() if state in ("nan", "mixed")
        )
        unchecked = sorted(
            label for label, state in channel.states.items() if state == "not_checked"
        )
        questions.append(
            KeyQuestion(
                id="status-1",
                kind="status_nonfinite",
                question="Which nodes carry NaN values?",
                answer=nan_labels,
            )
        )
        if unchecked and not channel.zero_coverage:
            questions.append(
                KeyQuestion(
                    id="status-2",
                    kind="status_nonfinite",
                    question=(f"Is {unchecked[0]!r} verified finite, or merely not checked?"),
                    answer="not checked",
                    honesty_class="NOT-CHECKED-as-FINITE",
                )
            )

    if resolution is not None and resolution.display_filter is not None:
        questions.append(
            KeyQuestion(
                id="bridged-1",
                kind="bridged_vs_direct",
                question=(
                    "Does a dashed edge mean the two operations are directly "
                    "adjacent in the computation?"
                ),
                answer=False,
                honesty_class="ADJACENCY",
            )
        )

    if (
        resolution is not None
        and resolution.source is not None
        and resolution.source.view == "rolled"
    ):
        questions.append(
            KeyQuestion(
                id="aggregate-1",
                kind="per_call_vs_aggregate",
                question=(
                    "Does a node's colour show the cost of ONE call, or the "
                    "total across every pass?"
                ),
                answer="total across passes",
                honesty_class="PER-CALL-vs-AGGREGATE",
            )
        )

    return AnswerKey(member=member_name, lens=lens_name, questions=tuple(questions))
