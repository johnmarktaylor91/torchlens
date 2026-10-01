"""Prompt utilities (mikit section 8; TLens ``test_prompt`` lineage).

``inspect_prompt`` returns a structured :class:`PromptInspection` FIRST and
renders second: exact token ids (and pieces when a tokenizer is given), the
BOS decision shown explicitly (where beginners lose an hour -- and where
HALF of intuition-authored expectations failed in the calibration study),
per-answer rank/logit/probability/log-probability, top-k, comparator
logit-diff, and checked multi-token teacher-forced scoring. Assertions live
as ``assert_*`` methods on the record (tests never scrape console text),
with expectations from calibrated goldens only -- never an author's
intuition (' Paris' ranks 1st on gpt2 and 4th on distilgpt2).

Works from a saved trace post-hoc; retention: logits only.
``contributions=True`` appends the top DLA contributors, making the
teaching utility a one-call mechinterp entry point.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ._anchors import resolve_lm_head
from ._errors import refuse

__all__ = ["PromptInspection", "inspect_prompt", "test_prompt"]

#: Rank convention (shared with logit_lens): 1 + count(strictly greater);
#: tied logits share the smallest rank.
TIE_CONVENTION = "rank = 1 + count(logits strictly greater); tied logits share the smallest rank"


@dataclass(frozen=True)
class AnswerRow:
    """Per-answer prediction facts at one position."""

    token_id: int
    piece: str | None
    rank: int
    logit: float
    probability: float
    log_probability: float
    position: int
    teacher_forced: bool = False


@dataclass(frozen=True)
class PromptInspection:
    """Structured prompt inspection (record first, rendering second)."""

    input_ids: tuple[int, ...]
    pieces: tuple[str, ...] | None
    bos_disclosure: str
    answers: tuple[AnswerRow, ...]
    top_k: tuple[tuple[int, str | None, float], ...]
    logit_diff: float | None
    contributions: tuple[tuple[str, float], ...] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def assert_rank(self, token_id: int, rank: int) -> None:
        """Assert one answer token's rank (calibrated-golden expectations only)."""

        row = self._row(token_id)
        if row.rank != rank:
            raise AssertionError(
                f"token {token_id} ranks {row.rank}, expected {rank} ({TIE_CONVENTION})"
            )

    def assert_top(self, token_id: int) -> None:
        """Assert one answer token ranks first."""

        self.assert_rank(token_id, 1)

    def assert_logit_diff_positive(self) -> None:
        """Assert the answer-vs-comparator logit diff is positive."""

        if self.logit_diff is None:
            raise AssertionError("no comparator (vs=) was scored")
        if not self.logit_diff > 0:
            raise AssertionError(f"logit diff {self.logit_diff} is not positive")

    def _row(self, token_id: int) -> AnswerRow:
        """Return the scored row for one answer token."""

        for row in self.answers:
            if row.token_id == token_id:
                return row
        raise AssertionError(f"token {token_id} was not among the scored answers")

    def summary(self) -> str:
        """Render the human view (the record is the API; this is a view)."""

        lines = [f"prompt ids: {list(self.input_ids)}", self.bos_disclosure]
        if self.pieces is not None:
            lines.append("pieces: " + " | ".join(self.pieces))
        for row in self.answers:
            lines.append(
                f"  answer {row.token_id}"
                + (f" ({row.piece!r})" if row.piece else "")
                + f" @pos {row.position}: rank {row.rank}, logit {row.logit:.4f}, "
                f"p {row.probability:.4f}" + (" [teacher-forced]" if row.teacher_forced else "")
            )
        if self.logit_diff is not None:
            lines.append(f"  logit diff (answer - vs): {self.logit_diff:.4f}")
        for token_id, piece, logit in self.top_k:
            lines.append(
                f"  top: {token_id}" + (f" ({piece!r})" if piece else "") + f" {logit:.4f}"
            )
        if self.contributions:
            lines.append("  top contributors (DLA):")
            for label, score in self.contributions:
                lines.append(f"    {label}: {score:+.4f}")
        return "\n".join(lines)


def _decode(tokenizer: Any, token_id: int) -> str | None:
    """Best-effort single-token decode."""

    if tokenizer is None:
        return None
    try:
        return str(tokenizer.decode([int(token_id)]))
    except Exception:  # noqa: BLE001 -- rendering nicety, never load-bearing
        return None


def _bos_disclosure(input_ids: list[int], tokenizer: Any) -> str:
    """State the BOS decision explicitly (never inferred silently)."""

    if tokenizer is None:
        return (
            "BOS: unknown (no tokenizer given); the ids above are EXACTLY what the "
            "model saw -- nothing was prepended by this inspection"
        )
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        return "BOS: tokenizer defines no bos_token_id; nothing was prepended"
    if input_ids and input_ids[0] == int(bos):
        return f"BOS: leading id {input_ids[0]} IS the tokenizer's BOS token"
    return f"BOS: NOT present (tokenizer's BOS id is {int(bos)}); nothing was prepended"


def inspect_prompt(  # noqa: PLR0913 -- the memo-normative section-8 signature
    trace: Any,
    *,
    answer: Any = None,
    vs: Any = None,
    position: int = -1,
    top_k: int = 5,
    tokenizer: Any = None,
    contributions: bool = False,
) -> PromptInspection:
    """Inspect a captured LM forward's predictions (mikit section 8).

    Parameters
    ----------
    trace:
        A finished torchlens trace (post-hoc from a saved trace works;
        retention: logits only).
    answer:
        Optional answer token id(s) to score at ``position``.
    vs:
        Optional single comparator token id (logit-diff row).
    position:
        Position whose next-token distribution is inspected (default last).
    top_k:
        How many top predictions to include.
    tokenizer:
        Optional tokenizer for piece rendering + the BOS disclosure.
    contributions:
        Append the top DLA contributors for ``answer`` (needs the residual
        stream retained; refuses with the plan otherwise).
    """

    anchor = resolve_lm_head(trace)
    logits = anchor.facets()["logits"].value.detach().to(torch.float32)
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    n_positions = int(logits.shape[-2])
    pos = position % n_positions if position < 0 else position
    if not 0 <= pos < n_positions:
        refuse(
            code="mi_position_unmappable",
            message=f"position {position} outside the logits' {n_positions} positions.",
            remedy=f"pass a position inside [-{n_positions}, {n_positions})",
            position=position,
        )
    row_logits = logits[0, pos]
    log_probs = torch.log_softmax(row_logits, dim=-1)
    probs = log_probs.exp()

    input_ids: list[int] = []
    for label in getattr(trace, "input_ops", ()) or ():
        op = label if hasattr(label, "out") else trace.ops[str(label)]
        value = getattr(op, "out", None)
        if isinstance(value, torch.Tensor) and not value.is_floating_point() and value.dim() == 2:
            input_ids = [int(t) for t in value[0]]
            break

    answer_ids = [] if answer is None else ([answer] if isinstance(answer, int) else list(answer))
    rows = []
    for token_id in answer_ids:
        logit = float(row_logits[token_id])
        rank = int((row_logits > row_logits[token_id]).sum()) + 1
        rows.append(
            AnswerRow(
                token_id=int(token_id),
                piece=_decode(tokenizer, token_id),
                rank=rank,
                logit=logit,
                probability=float(probs[token_id]),
                log_probability=float(log_probs[token_id]),
                position=pos,
            )
        )
    logit_diff = None
    if vs is not None and answer_ids:
        logit_diff = float(row_logits[answer_ids[0]] - row_logits[int(vs)])

    top_values, top_indices = row_logits.topk(max(1, top_k))
    top_rows = tuple(
        (int(index), _decode(tokenizer, int(index)), float(value))
        for index, value in zip(top_indices, top_values, strict=True)
    )

    contribution_rows = None
    if contributions and answer_ids:
        from ._dla import direct_logit_contributions

        dla = direct_logit_contributions(
            trace, answer=answer_ids[0], vs=int(vs) if vs is not None else None, positions=pos
        )
        contribution_rows = dla.top(5)

    pieces = None
    if tokenizer is not None and input_ids:
        pieces = tuple(_decode(tokenizer, token_id) or "?" for token_id in input_ids)
    return PromptInspection(
        input_ids=tuple(input_ids),
        pieces=pieces,
        bos_disclosure=_bos_disclosure(input_ids, tokenizer),
        answers=tuple(rows),
        top_k=top_rows,
        logit_diff=logit_diff,
        contributions=contribution_rows,
        provenance={"position": pos, "tie_convention": TIE_CONVENTION},
    )


def test_prompt(  # noqa: PLR0913 -- the memo-normative section-8 signature
    model: Any,
    prompt: Any,
    *,
    answer: Any,
    vs: Any = None,
    tokenizer: Any = None,
    top_k: int = 5,
    contributions: bool = False,
) -> PromptInspection:
    """Trace one forward and inspect it -- with checked multi-token
    teacher-forced scoring (mikit section 8).

    ``prompt`` is token ids (list/tensor) or text (requires ``tokenizer``).
    A multi-token ``answer`` is scored teacher-forced: the model sees
    ``prompt + answer[:-1]`` in ONE forward and each answer token is scored
    at its own position (boundary-checked -- the forward length proves the
    concatenation happened).
    """

    import torchlens as tl

    if isinstance(prompt, str):
        if tokenizer is None:
            refuse(
                code="mi_tokenizer_required",
                message="A text prompt needs a tokenizer.",
                remedy="pass tokenizer=, or pass token ids directly",
            )
        prompt_ids = list(tokenizer(prompt)["input_ids"])
    elif isinstance(prompt, torch.Tensor):
        prompt_ids = [int(t) for t in prompt.reshape(-1)]
    else:
        prompt_ids = [int(t) for t in prompt]
    answer_ids = [answer] if isinstance(answer, int) else [int(t) for t in answer]

    forced = prompt_ids + answer_ids[:-1]
    ids = torch.tensor([forced], dtype=torch.long)
    trace = tl.trace(model, ids)
    base = inspect_prompt(
        trace,
        answer=answer_ids[0],
        vs=vs,
        position=len(prompt_ids) - 1,
        top_k=top_k,
        tokenizer=tokenizer,
        contributions=contributions,
    )
    if len(answer_ids) == 1:
        return base

    anchor = resolve_lm_head(trace)
    logits = anchor.facets()["logits"].value.detach().to(torch.float32)
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    if int(logits.shape[-2]) != len(forced):
        refuse(
            code="mi_position_unmappable",
            message=f"Teacher-forced scoring needs one logit row per forced position "
            f"({len(forced)}), got {int(logits.shape[-2])} (logits_to_keep active?).",
            remedy="trace a plain forward (no generation slicing) for multi-token scoring",
            expected=len(forced),
        )
    rows = list(base.answers)
    for offset, token_id in enumerate(answer_ids[1:], start=1):
        pos = len(prompt_ids) - 1 + offset
        row_logits = logits[0, pos]
        log_probs = torch.log_softmax(row_logits, dim=-1)
        rows.append(
            AnswerRow(
                token_id=int(token_id),
                piece=_decode(tokenizer, token_id),
                rank=int((row_logits > row_logits[token_id]).sum()) + 1,
                logit=float(row_logits[token_id]),
                probability=float(log_probs[token_id].exp()),
                log_probability=float(log_probs[token_id]),
                position=pos,
                teacher_forced=True,
            )
        )
    return PromptInspection(
        input_ids=base.input_ids,
        pieces=base.pieces,
        bos_disclosure=base.bos_disclosure,
        answers=tuple(rows),
        top_k=base.top_k,
        logit_diff=base.logit_diff,
        contributions=base.contributions,
        provenance={**base.provenance, "teacher_forced_tokens": len(answer_ids)},
    )
