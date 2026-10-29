"""``Bundle.token_divergence``: where two runs of one call part, by exact token ids.

A user who re-runs a harness with one call steered compares each call with
its re-run: did the prompt change, did the completion change, and from which
token. Both members need a ``token_boundary_v1`` binding (see
:mod:`torchlens.bundle._fed_by`) naming their ``completion`` range, with the
token ids retained: the prompt is the bound stream before the completion
start, the completion is the bound range. Positions are compared by exact
token id; nothing is decoded, aligned by content or reduced.

``shared`` reports the stretches both runs share as four-int rows in the
``token_spans`` grammar (``[orig_start, orig_end, rerun_start, rerun_end]``,
half-open stream positions): the common prompt prefix, the common prompt
tail after a divergence (the template after a quoted text), and the common
completion prefix. ``Bundle.token_pairs`` keeps its ``fed_by`` meaning.
Spelling documented-unstable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..errors.episode import BundleRelationError
from ._fed_by import member_binding, retained_ids

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from . import Bundle

UNAVAILABLE_CODE = "bundle_token_divergence_unavailable"


@dataclass(frozen=True)
class TokenDivergence:
    """Result of ``Bundle.token_divergence(original, rerun)`` (documented-unstable).

    Attributes
    ----------
    original, rerun:
        Member names, in the call's order.
    prompt_divergence:
        First prompt position whose token ids differ; the shorter prompt's
        length when one prompt extends the other; ``None`` when equal.
    completion_divergence:
        The same for the completions, counted from the completion start
        (``0`` is the first generated token).
    lengths:
        ``((prompt, completion), (prompt, completion))`` token counts for
        ``original`` and ``rerun``.
    shared:
        Stretches both runs share, as ``(orig_start, orig_end, rerun_start,
        rerun_end)`` half-open stream positions.
    intervened:
        Whether each member's capture ran under an intervention.
    """

    original: str
    rerun: str
    prompt_divergence: int | None
    completion_divergence: int | None
    lengths: tuple[tuple[int, int], tuple[int, int]]
    shared: tuple[tuple[int, int, int, int], ...]
    intervened: tuple[bool, bool]

    @property
    def prompt_same(self) -> bool:
        """Whether both prompts carry the same token ids."""

        return self.prompt_divergence is None

    @property
    def completion_same(self) -> bool:
        """Whether both completions carry the same token ids."""

        return self.completion_divergence is None


def capture_edit_rows(trace: Trace) -> list[dict[str, Any]]:
    """Return the trace's intervention ``ACT`` audit rows (empty on a plain capture)."""

    audit = getattr(trace, "intervention_audit", None) or ()
    return [row for row in audit if isinstance(row, dict) and row.get("kind") == "ACT"]


def _refuse(name: str, reason: str) -> BundleRelationError:
    return BundleRelationError(
        f"Bundle.token_divergence cannot split member {name!r} into prompt and "
        f"completion: {reason}.",
        code=UNAVAILABLE_CODE,
        remedy=(
            "attach a token_boundary_v1 descriptor naming the member's completion range "
            "(torchlens.io.attach_sidecar(trace, 'torchlens.boundary_facts', {'stream': "
            "'input', 'batch_row': 0, 'completion': [start, end]})) and keep its token-id "
            "input saved"
        ),
        member=name,
        reason=reason,
    )


def _split(name: str, trace: Trace) -> tuple[list[int], list[int]]:
    """Return the member's ``(prompt ids, completion ids)``, refusing typed."""

    binding = member_binding(trace)
    if binding is None:
        raise _refuse(name, "it carries no token_boundary_v1 binding")
    completion = binding.get("completion")
    if completion is None:
        raise _refuse(name, "its binding names no completion range")
    ids = retained_ids(trace, binding)
    if ids is None:
        raise _refuse(name, "its token ids are not retained or disagree with its binding")
    start, end = int(completion[0]), int(completion[1])
    if end > len(ids):
        raise _refuse(name, f"its completion range ends at {end}, past the {len(ids)}-token stream")
    return ids[:start], ids[start:end]


def _first_divergence(left: Sequence[int], right: Sequence[int]) -> int | None:
    """First index where two id rows differ, ``None`` when equal."""

    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _shared_runs(
    original: tuple[list[int], list[int]], rerun: tuple[list[int], list[int]]
) -> tuple[tuple[int, int, int, int], ...]:
    """Return the shared prompt prefix, prompt tail and completion prefix as span rows."""

    (prompt_a, done_a), (prompt_b, done_b) = original, rerun
    len_a, len_b = len(prompt_a), len(prompt_b)
    rows: list[tuple[int, int, int, int]] = []
    head = _first_divergence(prompt_a, prompt_b)
    prefix = min(len_a, len_b) if head is None else head
    if prefix:
        rows.append((0, prefix, 0, prefix))
    if head is not None:
        tail = 0
        while tail < min(len_a, len_b) - prefix and prompt_a[-1 - tail] == prompt_b[-1 - tail]:
            tail += 1
        if tail:
            rows.append((len_a - tail, len_a, len_b - tail, len_b))
    split = _first_divergence(done_a, done_b)
    common = min(len(done_a), len(done_b)) if split is None else split
    if common:
        rows.append((len_a, len_a + common, len_b, len_b + common))
    return tuple(rows)


def divergence(names: tuple[str, str], traces: tuple[Trace, Trace]) -> TokenDivergence:
    """Return the token divergence of two traces (the shared body of both public doors)."""

    first, second = _split(names[0], traces[0]), _split(names[1], traces[1])
    return TokenDivergence(
        original=names[0],
        rerun=names[1],
        prompt_divergence=_first_divergence(first[0], second[0]),
        completion_divergence=_first_divergence(first[1], second[1]),
        lengths=((len(first[0]), len(first[1])), (len(second[0]), len(second[1]))),
        shared=_shared_runs(first, second),
        intervened=(bool(capture_edit_rows(traces[0])), bool(capture_edit_rows(traces[1]))),
    )


def _bundle_token_divergence(
    self: Bundle, original: str | Trace, rerun: str | Trace
) -> TokenDivergence:
    """Return where two members' prompts and completions part, by exact token ids.

    Exposed as ``Bundle.token_divergence`` (spelling documented-unstable).
    Typically ``original`` is a call of the first run and ``rerun`` its copy
    from a steered re-run (``b.token_divergence("critique", "cf_critique")``),
    but any two bound members compare.

    Parameters
    ----------
    self:
        Bundle holding both members.
    original, rerun:
        Member names or member Traces.

    Returns
    -------
    TokenDivergence
        First differing prompt and completion positions, lengths, the shared
        stretches and whether each capture ran under an intervention.

    Raises
    ------
    KeyError
        When a name is not a member.
    BundleRelationError
        ``bundle_token_divergence_unavailable`` when a member has no binding,
        its binding names no completion range, or its token ids are not
        retained (or disagree with the binding).
    """

    names = (self._coerce_member_name(original), self._coerce_member_name(rerun))
    for name in names:
        if name not in self._members:
            raise KeyError(f"{name!r} is not a member of this Bundle; members: {self.names}")
    return divergence(names, (self._members[names[0]], self._members[names[1]]))
