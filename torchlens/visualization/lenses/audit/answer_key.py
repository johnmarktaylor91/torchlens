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
    ``bridged_vs_direct`` / ``io_identification`` / ``direction`` /
    ``channel_semantics``); ``honesty_class`` names the zero-tolerance
    class the question guards, when it guards one. ``accepted`` lists
    ADDITIONAL correct answers -- a tied extremum names every tied label
    here (a key that demands one label on a tie is battery ambiguity,
    D03-R7), and naming the complete tie as a collection also scores.
    """

    id: str
    kind: str
    question: str
    answer: Any
    distractors: tuple[str, ...] = ()
    honesty_class: str | None = None
    accepted: tuple[Any, ...] = ()


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


def _canonical_display_labels(labels: Any) -> list[str]:
    """Collapse alternative spellings of ONE op to one displayed entry.

    The nonfinite channel keys BOTH the pass-qualified ``label:N`` and the
    bare layer label per op; a key packing both spellings into one
    set-equality list is unanswerable -- the image prints one (D03-R7).
    The bare spelling wins where both are present; a genuinely distinct
    pass spelling (bare stem absent) is kept as-is.
    """

    pool = {str(label) for label in labels}
    kept: list[str] = []
    for label in sorted(pool):
        stem, sep, tail = label.rpartition(":")
        if sep and tail.isdigit() and stem in pool:
            continue
        kept.append(label)
    return kept


def _extrema_questions(values: dict[str, float]) -> list[KeyQuestion]:
    """The extrema/ordinal probe family over the resolved channel values.

    A tied extremum accepts EVERY tied label (D03-R7: a key that demands
    one label on a tie is battery ambiguity, not an evaluator error);
    distractors never name a tied label.
    """

    max_label = max(values, key=values.__getitem__)
    min_label = min(values, key=values.__getitem__)
    max_ties = tuple(sorted(label for label, value in values.items() if value == values[max_label]))
    min_ties = tuple(sorted(label for label, value in values.items() if value == values[min_label]))
    others = sorted(label for label in values if label not in max_ties)[:3]
    questions = [
        KeyQuestion(
            id="extrema-max",
            kind="extrema_max",
            question="Which node is HIGHEST on the encoded colour scale?",
            answer=max_label,
            distractors=tuple(others),
            accepted=max_ties,
        ),
        KeyQuestion(
            id="extrema-min",
            kind="extrema_min",
            question="Which node is LOWEST on the encoded colour scale?",
            answer=min_label,
            accepted=min_ties,
        ),
    ]
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
    return questions


def _sentinel_probe() -> KeyQuestion:
    """The channel-semantics probe (D03-R6).

    The ONE question a known-bad control can fail: the fill of a sentinel
    render is dishonest by construction, so the machine answer is the
    no-meaning option; a judge who reads the bad channel as meaningful
    misses it and confirms the ABSENCE class.
    """

    return KeyQuestion(
        id="channel-semantics-1",
        kind="channel_semantics",
        question=(
            "What does the node FILL COLOUR encode in this picture? "
            "If you believe it carries no reliable meaning, say so."
        ),
        answer="no meaning",
        distractors=(
            "a per-node numeric quantity on an ordered colour scale",
            "the operation's category or type",
            "how recently each operation ran",
        ),
        honesty_class="ABSENCE",
    )


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
    sentinel: bool = False,
) -> AnswerKey:
    """Generate the answer key for one rendered lens artifact.

    Every answer derives from the trace record and the resolution: extrema
    and ranking from the resolved channel values, status from the derived
    nonfinite channel, paths from the recorded DAG, bridged-vs-direct from
    the compiled filter, boundaries from the boundary flags.

    ``sentinel=True`` declares a KNOWN-BAD control artifact (a render whose
    colour channel is dishonest BY CONSTRUCTION) and mints the
    channel-semantics probe that lets the control FAIL: without it the
    sentinel's only scoreable probes never touch the bad channel and the
    battery cannot separate its own anchors (D03-R6). The probe is still
    machine-generated -- the flag states how the artifact was constructed,
    never a hand-authored answer.
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
            questions.extend(_extrema_questions(values))

    if resolution is not None and resolution.nonfinite is not None:
        channel = resolution.nonfinite
        nan_labels = _canonical_display_labels(
            label for label, state in channel.states.items() if state in ("nan", "mixed")
        )
        unchecked = _canonical_display_labels(
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

    if sentinel:
        questions.append(_sentinel_probe())
    return AnswerKey(member=member_name, lens=lens_name, questions=tuple(questions))
