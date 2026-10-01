"""The naive-evaluator battery harness (themes memo section 7, Stage 1).

The battery measures what a viewer who does NOT know TorchLens's
conventions reads off a picture -- especially FALSE MEANINGS, confident
interpretations of things we do not encode. This module is the HARNESS:
packet building (masked filenames, seeded ordering, the memo's fixed
question-order discipline), response scoring including the nine
zero-tolerance honesty classes, and the anchor-midpoint threshold-freezing
arithmetic. Executing packets against fresh evaluator instances is D03's
delegated job with quantitative stop criteria.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from ...._errors import InvalidArgumentError
from .answer_key import AnswerKey

__all__ = [
    "HONESTY_CLASSES",
    "BatteryPacket",
    "BatteryScore",
    "build_packet",
    "freeze_threshold",
    "score_responses",
]

#: The nine honesty classes, zero tolerance forever (memo section 7): these
#: encode shipped rules, never scores, and are never subject to threshold
#: arithmetic. One confirmed sentinel hit sends the lens back.
HONESTY_CLASSES = (
    "UNIFORMITY",
    "ABSENCE",
    "ADJACENCY",
    "PER-CALL-vs-AGGREGATE",
    "NAMED-AXIS",
    "NOT-CHECKED-as-FINITE",
    "ORDINAL-as-RATIO",
    "ADVERTISED-SCALE",
    "CHANNEL-COLLISION",
)

#: The memo's fixed question-order discipline: later questions prime, so
#: they come later.
_QUESTION_PHASE_ORDER = (
    "free_response",
    "free_inventory",
    "forced_choice",
    "task_probe",
    "honesty_probe",
    "transcription",
    "legend_reveal",
)

_KIND_TO_PHASE = {
    "io_identification": "task_probe",
    "path_reachability": "task_probe",
    "extrema_max": "task_probe",
    "extrema_min": "task_probe",
    "ranking_pair": "honesty_probe",
    "status_nonfinite": "task_probe",
    "bridged_vs_direct": "honesty_probe",
    "per_call_vs_aggregate": "honesty_probe",
    "direction": "task_probe",
}


@dataclass(frozen=True)
class BatteryPacket:
    """One evaluator packet: masked images, ordered questions, and rules.

    ``judges`` is 3 for normal images and 5 for legendless/sentinel images
    (memo counts); ONE evaluator sees ONE image.
    """

    packet_id: str
    masked_image_names: tuple[str, ...]
    questions: tuple[dict[str, Any], ...]
    judges: int
    legend_withheld: bool
    presentation_rules: tuple[str, ...]


@dataclass(frozen=True)
class BatteryScore:
    """Scored battery responses for one packet.

    ``honesty_hits`` lists confirmed zero-tolerance class hits -- ANY entry
    fails the lens regardless of the accuracy numbers.
    """

    total: int
    correct: int
    by_kind: dict[str, tuple[int, int]]
    honesty_hits: tuple[str, ...] = field(default=())

    @property
    def accuracy(self) -> float:
        """Return overall accuracy (0.0 on an empty packet)."""

        return self.correct / self.total if self.total else 0.0


def build_packet(
    key: AnswerKey,
    image_paths: list[str],
    *,
    seed: int,
    legend_withheld: bool = False,
    sentinel: bool = False,
) -> BatteryPacket:
    """Build one evaluator packet from a generated key.

    Filenames are masked (an evaluator never sees the lens or model name),
    order is seeded-random within the memo's fixed phase discipline, and
    the presentation rules ride the packet so D03 does not re-derive them.
    """

    rng = random.Random(seed)
    masked = tuple(f"image_{index:02d}.png" for index in range(len(image_paths)))
    free_phase: list[dict[str, Any]] = [
        {
            "id": "free-1",
            "phase": "free_response",
            "question": (
                "What is this a picture of? Where does information enter and "
                "leave, and which way does it flow?"
            ),
        },
        {
            "id": "free-2",
            "phase": "free_inventory",
            "question": (
                "List every visual feature that looks MEANINGFUL (colours, "
                "sizes, borders, dashes, shapes), with your confidence that "
                "each carries meaning."
            ),
        },
    ]
    keyed: list[dict[str, Any]] = []
    for question in key.questions:
        phase = _KIND_TO_PHASE.get(question.kind, "task_probe")
        entry: dict[str, Any] = {
            "id": question.id,
            "phase": phase,
            "kind": question.kind,
            "question": question.question,
        }
        if question.distractors:
            options = [str(question.answer), *question.distractors, "no meaning"]
            rng.shuffle(options)
            entry["options"] = options
            entry["phase"] = "forced_choice"
        if question.honesty_class:
            entry["honesty_class"] = question.honesty_class
        keyed.append(entry)
    rng.shuffle(keyed)
    ordered = free_phase + sorted(
        keyed, key=lambda entry: _QUESTION_PHASE_ORDER.index(entry["phase"])
    )
    ordered.append(
        {
            "id": "transcribe-1",
            "phase": "transcription",
            "question": "Transcribe every label you can read, then the legend verbatim.",
        }
    )
    if legend_withheld:
        ordered.append(
            {
                "id": "legend-reveal-1",
                "phase": "legend_reveal",
                "question": (
                    "The legend is now revealed (second image). Repeat the "
                    "headline question: what does the colour mean?"
                ),
            }
        )
    return BatteryPacket(
        packet_id=f"{key.member}-{key.lens}-{seed}",
        masked_image_names=masked,
        questions=tuple(ordered),
        judges=5 if (legend_withheld or sentinel) else 3,
        legend_withheld=legend_withheld,
        presentation_rules=(
            "downscaled whole graph first, then original resolution",
            "deterministic 3x-4x crops of the densest label region, legend, extrema, marked path",
            "every difference question is asked on a crop",
            "filenames and metadata masked; one evaluator sees one image",
        ),
    )


def score_responses(key: AnswerKey, responses: dict[str, Any]) -> BatteryScore:
    """Score evaluator responses against the generated key.

    A wrong answer on a question guarding an honesty class is a CONFIRMED
    class hit: zero tolerance, no threshold arithmetic applies.
    """

    total = 0
    correct = 0
    by_kind: dict[str, tuple[int, int]] = {}
    honesty_hits: list[str] = []
    for question in key.questions:
        if question.id not in responses:
            continue
        total += 1
        response = responses[question.id]
        is_correct = _matches(question.answer, response)
        kind_total, kind_correct = by_kind.get(question.kind, (0, 0))
        by_kind[question.kind] = (kind_total + 1, kind_correct + (1 if is_correct else 0))
        if is_correct:
            correct += 1
        elif question.honesty_class:
            honesty_hits.append(question.honesty_class)
    return BatteryScore(
        total=total,
        correct=correct,
        by_kind=by_kind,
        honesty_hits=tuple(sorted(set(honesty_hits))),
    )


def _matches(expected: Any, actual: Any) -> bool:
    """Loose-but-deterministic answer matching (case/whitespace neutral)."""

    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return expected == actual
        text = str(actual).strip().lower()
        return text in ("yes", "true") if expected else text in ("no", "false")
    if isinstance(expected, (list, tuple, set, frozenset, dict)):
        return expected == actual
    return str(expected).strip().lower() == str(actual).strip().lower()


def _one_sided_bound(scores: list[float], *, upper: bool) -> float:
    """One-sided 95% normal-approximation bound on the mean of ``scores``."""

    if len(scores) < 2:
        raise InvalidArgumentError(
            "threshold freezing needs at least two anchor judgements per side",
            code="battery_anchor_insufficient",
            remedy="collect ten judgements per anchor (memo section 7) before freezing",
            argument="scores",
        )
    mean = sum(scores) / len(scores)
    variance = sum((value - mean) ** 2 for value in scores) / (len(scores) - 1)
    margin = 1.645 * math.sqrt(variance / len(scores))
    return mean + margin if upper else mean - margin


def freeze_threshold(known_good_scores: list[float], known_bad_scores: list[float]) -> float:
    """The anchor-midpoint freezing procedure (memo section 7, adopted 2-1).

    The frozen threshold is the midpoint between the bad anchor's one-sided
    95% UPPER bound and the good anchor's one-sided 95% LOWER bound, rounded
    UP to five points (percent scale). Overlapping bounds invalidate the
    METRIC (or the anchors), never waive a candidate.
    """

    bad_upper = _one_sided_bound(known_bad_scores, upper=True)
    good_lower = _one_sided_bound(known_good_scores, upper=False)
    if bad_upper >= good_lower:
        raise InvalidArgumentError(
            f"anchor bounds overlap (bad upper {bad_upper:.1f} >= good lower "
            f"{good_lower:.1f}): the metric or the anchors are invalid",
            code="battery_metric_invalid",
            remedy=(
                "fix the metric or re-anchor; a threshold is never waived and "
                "never drifts down to fit a candidate"
            ),
            argument="known_bad_scores",
        )
    midpoint = (bad_upper + good_lower) / 2.0
    return math.ceil(midpoint / 5.0) * 5.0
