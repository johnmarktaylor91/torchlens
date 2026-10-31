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
import re
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
    "channel_semantics": "honesty_probe",
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


def _require_sentinel_probe(key: AnswerKey) -> None:
    """Refuse a sentinel packet whose known-bad control cannot fail.

    A battery that cannot separate its own anchors is a broken battery,
    not a verdict (D03-R6).
    """

    if not any(question.kind == "channel_semantics" for question in key.questions):
        raise InvalidArgumentError(
            "a sentinel (known-bad) packet needs a channel-semantics probe; "
            "without one the control cannot fail and the battery cannot "
            "separate its anchors",
            code="battery_sentinel_probe_missing",
            remedy="generate the key with generate_answer_key(..., sentinel=True)",
            argument="sentinel",
        )


def _forced_choice_options(question: Any) -> list[str]:
    """Assemble a forced-choice option list, deduped case-insensitively.

    A probe whose ANSWER is the no-meaning option (the sentinel
    channel-semantics probe) must not present it twice.
    """

    options: list[str] = []
    for option in (str(question.answer), *question.distractors, "no meaning"):
        if option.strip().lower() not in {seen.strip().lower() for seen in options}:
            options.append(option)
    return options


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

    A ``sentinel`` (known-bad control) packet REQUIRES a channel-semantics
    probe in the key: without one the control cannot fail and the battery
    cannot separate its own anchors -- a broken battery, not a verdict
    (D03-R6). Generate the key with ``generate_answer_key(...,
    sentinel=True)``.
    """

    if sentinel:
        _require_sentinel_probe(key)
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
            options = _forced_choice_options(question)
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
    class hit: zero tolerance, no threshold arithmetic applies. A response
    matching the answer, ANY ``accepted`` alternative (a tied extremum), or
    the complete accepted set named as a collection is correct (D03-R7).
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
        is_correct = (
            _matches(question.answer, response)
            or any(_matches(candidate, response) for candidate in question.accepted)
            or (bool(question.accepted) and _matches(list(question.accepted), response))
        )
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


#: Standalone tokens that decide a boolean response's polarity.
_BOOLEAN_TOKENS = {"yes": True, "true": True, "no": False, "false": False}


def _normalized(value: Any) -> str:
    """Case/whitespace-neutral text form."""

    return str(value).strip().lower()


def _label_stem(text: str) -> str:
    """Strip one trailing ``:N`` pass qualifier, else return the text."""

    stem, sep, tail = text.rpartition(":")
    return stem if sep and tail.isdigit() else text


def _string_matches(expected: Any, actual: Any) -> bool:
    """Case/whitespace-neutral equality with label-spelling interchange.

    The bare and ``:N``-qualified spellings of ONE label are the same
    answer (D03-R7); two DIFFERENTLY-qualified spellings never match
    through the bare stem.
    """

    expected_text, actual_text = _normalized(expected), _normalized(actual)
    if expected_text == actual_text:
        return True
    return _label_stem(expected_text) == actual_text or expected_text == _label_stem(actual_text)


def _boolean_matches(expected: bool, actual: Any) -> bool:
    """The FIRST standalone yes/true/no/false token decides polarity.

    An elaborated "No (false). The scale is ordinal ..." is a correct
    refusal, not a format miss (D03-R7); a response with no boolean token
    never matches.
    """

    if isinstance(actual, bool):
        return expected == actual
    for token in re.findall(r"[a-z]+", _normalized(actual)):
        polarity = _BOOLEAN_TOKENS.get(token)
        if polarity is not None:
            return polarity is expected
    return False


def _response_items(actual: Any) -> list[Any] | None:
    """Normalize a collection response; ``None`` = not collection-shaped.

    A string response is read as a comma-separated label list.
    """

    if isinstance(actual, str):
        return [part for part in map(str.strip, actual.split(",")) if part]
    if isinstance(actual, (list, tuple, set, frozenset)):
        return list(actual)
    return None


def _collection_matches(expected: Any, actual: Any) -> bool:
    """Order-insensitive label-set equality with spelling interchange.

    An empty expected set also accepts a bare "none".
    """

    actual_items = _response_items(actual)
    if actual_items is None:
        return False
    expected_items = list(expected)
    if not expected_items:
        return not actual_items or [_normalized(item) for item in actual_items] == ["none"]
    return all(
        any(_string_matches(expected_item, item) for item in actual_items)
        for expected_item in expected_items
    ) and all(
        any(_string_matches(expected_item, item) for expected_item in expected_items)
        for item in actual_items
    )


def _matches(expected: Any, actual: Any) -> bool:
    """Loose-but-deterministic answer matching (case/whitespace neutral)."""

    if isinstance(expected, bool):
        return _boolean_matches(expected, actual)
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        actual_by_key = {_normalized(key): value for key, value in actual.items()}
        return set(map(_normalized, expected)) == set(actual_by_key) and all(
            _matches(value, actual_by_key[_normalized(key)]) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple, set, frozenset)):
        return _collection_matches(expected, actual)
    return _string_matches(expected, actual)


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
