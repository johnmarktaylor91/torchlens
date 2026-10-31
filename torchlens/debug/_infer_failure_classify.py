"""Evidence-led probe-diary failure classification (quickstart memo D10).

Extracted from ``_infer_input_shape._run_search`` (god-file ratchet R43): the
CLIP-class misdiagnosis fix. A geometric reason may NEVER be the fallback for
probes that all died non-geometrically -- multi-input signatures get their own
reason quoting the probe exception, and other all-non-geometric diaries stay
honest as non-shape blockers.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

#: Probe-exception signatures of a forward that needs MORE inputs than the
#: one tensor the search synthesizes (quickstart memo D10: these diaries must
#: classify multi_input_required, never fall through to a geometric reason).
_MULTI_INPUT_RE = re.compile(
    r"(you have to specify|missing \d+ required (?:positional|keyword-only) argument"
    r"|either input_ids or inputs_embeds|both pixel_values and |requires the "
    r"|got multiple values for argument|takes \d+ positional arguments but)",
    re.IGNORECASE,
)

#: A None dereference inside the forward: the signature of a missing second
#: input that defaulted to ``None`` (only trusted when the diary proves the
#: error is shape-independent).
_NONE_DEREF_RE = re.compile(r"'NoneType' object", re.IGNORECASE)


def classify_nongeometric_failure(
    attempts: Sequence[tuple[Any, str]],
    is_skippable_shape_error: Callable[[str], bool],
) -> tuple[str, str] | None:
    """Classify an all-non-geometric probe diary, or return ``None``.

    Parameters
    ----------
    attempts:
        The probe diary: ``(shape, outcome)`` pairs where ``outcome`` is
        ``"ok"`` or the probe's exception text.
    is_skippable_shape_error:
        The search's own geometric-error predicate (passed in so this module
        never imports upward into the search).

    Returns
    -------
    tuple[str, str] | None
        ``(failure_reason, message)`` when the diary proves a non-geometric
        blocker; ``None`` when geometric classification should proceed.
    """

    failed_outcomes = [outcome for _, outcome in attempts if outcome != "ok"]
    distinct_failed_shapes = {
        shape for shape, outcome in attempts if outcome != "ok" and shape is not None
    }
    # An IDENTICAL error across >= 3 distinct probed shapes is shape-
    # INDEPENDENT evidence, even when the message happens to contain a
    # geometric-looking word (CLIP's probes all die with "'NoneType' object
    # has no attribute 'shape'" -- the missing-second-input None
    # dereference, not a geometry miss).
    shape_independent = (
        bool(failed_outcomes)
        and len(set(failed_outcomes)) == 1
        and len(distinct_failed_shapes) >= 3
    )
    multi_input_hits = [
        outcome
        for outcome in failed_outcomes
        if _MULTI_INPUT_RE.search(outcome) or (shape_independent and _NONE_DEREF_RE.search(outcome))
    ]
    if multi_input_hits:
        return (
            "multi_input_required",
            "Every probe died before geometry mattered: the forward requires more "
            f"than one input (last probe error: {multi_input_hits[-1]!r}). Pass the "
            "real forward call, or the mapping form of input_size= binding each "
            "forward keyword to its shape, e.g. input_size={'input_ids': (1, 16), "
            "'pixel_values': (1, 3, 224, 224)}.",
        )
    if shape_independent:
        return (
            "non_shape_blocker",
            f"All {len(failed_outcomes)} probes across "
            f"{len(distinct_failed_shapes)} distinct shapes failed with the "
            f"IDENTICAL error, so the blocker is shape-independent "
            f"(probe error: {failed_outcomes[-1]!r}).",
        )
    if failed_outcomes and not any(
        is_skippable_shape_error(outcome) for outcome in failed_outcomes
    ):
        return (
            "non_shape_blocker",
            "No probe failed on input geometry, so this is not an input-shape "
            f"problem (last probe error: {failed_outcomes[-1]!r}). The full probe "
            "diary is attached.",
        )
    return None
