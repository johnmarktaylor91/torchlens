"""W051 regressions for the episode step-join grading (audit 3.1, 3.4, 3.5).

* 3.5 -- ``exogenous_positions`` was the length of the UNALIGNED SUFFIX, not
  a count of exogenous positions: one context token edited in an 8-token
  entry reported 8. It is now a positional diff (1).
* 3.4 -- teacher-forced episodes whose root returns the model's PREDICTIONS
  were graded as an exogenous break because settlement graded the join
  against the evidence column instead of ``forced_tokens``. Forced joins now
  grade ``forced`` (basis ``forced_tokens``) and never break the episode.
* 3.1 -- tail-aligned evidence derivation silently misattributed tokens when
  more than one position per step was appended (multi-token decoding), and
  the join re-grade then reported FALSE exogenous breaks. The derivation now
  refuses typed unless the source width is the emitted-only or the measured
  entry-plus-one-per-step shape.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._episode_join import _exogenous_position_count, default_break_teaching
from torchlens.capture._episode_ledger import episode_ledger_for
from torchlens.errors.episode import EpisodeDeclarationError
from torchlens.options import EpisodeSpec

pytestmark = pytest.mark.smoke

V = 16


class _Step(nn.Module):
    """next token = (last token + 1) % V; a deterministic permutation LM."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, V)
        self.out = nn.Linear(V, V, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.eye(V))
            self.out.weight.copy_(torch.roll(torch.eye(V), shifts=1, dims=0))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.emb(ids[:, -1]))


def _envelope(trace: tl.Trace) -> dict:
    return trace.annotations["episode"]["header"]["step_join"]


# ---------------------------------------------------------------------------
# 3.5 -- exogenous_positions is a positional diff.
# ---------------------------------------------------------------------------


class _InjectOne(nn.Module):
    """Greedy loop that edits ONE context position after the first step."""

    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.step = _Step()
        self.n = n

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for i in range(self.n):
            ids = torch.cat([ids, self.step(ids).argmax(-1, keepdim=True)], dim=1)
            if i == 0:
                ids = ids.clone()
                ids[0, 2] = 15
        return ids[:, -self.n :]


def test_one_edited_context_position_counts_one() -> None:
    model = _InjectOne()
    log = tl.trace(
        model,
        torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=3),
    )
    envelope = _envelope(log)
    assert envelope["grades"] == [None, "exogenous", "continuous"]
    assert envelope["exogenous_positions"] == {"1": 1}
    assert "1 position entered" in default_break_teaching(1, 1)
    assert "2 positions entered" in default_break_teaching(1, 2)


@pytest.mark.parametrize(
    ("cur", "expected", "wildcard", "count"),
    [
        # Append feed, exact continuation.
        ((1, 2, 3, 9), (1, 2, 3, 9), 0, 0),
        # Last-token KV feed.
        ((9,), (1, 2, 3, 9), 0, 0),
        # Sliding window.
        ((2, 3, 9), (1, 2, 3, 9), 0, 0),
        # Live rule: the emission position is a wildcard.
        ((1, 2, 3, 7), (1, 2, 3, 0), 1, 0),
        # One interior context edit: ONE position, not the whole entry.
        ((1, 15, 3, 9), (1, 2, 3, 9), 0, 1),
        # The emission replaced by an injected token: one position.
        ((1, 2, 3, 5), (1, 2, 3, 9), 0, 1),
        # Three tokens appended from outside after a true continuation.
        ((1, 2, 3, 9, 7, 7, 7), (1, 2, 3, 9), 0, 3),
        # Nothing aligns at all: every position is unexplained.
        ((5, 6, 7), (1, 2, 3, 9), 0, 3),
    ],
)
def test_exogenous_position_count_table(cur, expected, wildcard, count) -> None:
    assert _exogenous_position_count(cur, expected, wildcard) == count


# ---------------------------------------------------------------------------
# 3.4 -- forced joins grade against the declaration.
# ---------------------------------------------------------------------------


class _ForcedPredictions(nn.Module):
    """Teacher-forced loop whose ROOT returns the model's predictions."""

    def __init__(self, forced: tuple[int, ...]) -> None:
        super().__init__()
        self.step = _Step()
        self.forced = forced

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        predictions = []
        for token in self.forced:
            predictions.append(self.step(ids).argmax(-1, keepdim=True))
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
        return torch.cat(predictions, dim=1)


def test_forced_episode_returning_predictions_is_not_a_break() -> None:
    model = _ForcedPredictions((5, 9, 13))
    log = tl.trace(
        model,
        torch.tensor([[1]]),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=3, forced_tokens=(5, 9, 13)),
    )
    envelope = _envelope(log)
    assert envelope["grades"] == [None, "forced", "forced"]
    assert envelope["basis"] == {"1": "forced_tokens", "2": "forced_tokens"}
    assert envelope["break_step"] is None
    assert envelope["exogenous_positions"] == {}
    ledger = episode_ledger_for(log)
    assert ledger is not None
    # The evidence column is the root's output (the predictions), and the
    # series claim stays OPEN across forced joins.
    assert ledger.step_output_series() == ((2,), (6,), (10,))
    assert log.annotations["episode"]["header"]["token_feed"] == "forced"


class _ForcedButLying(nn.Module):
    """Declares forced tokens but feeds something else: a real break."""

    def __init__(self) -> None:
        super().__init__()
        self.step = _Step()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        predictions = []
        for token in (5, 6, 7):
            predictions.append(self.step(ids).argmax(-1, keepdim=True))
            ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
        return torch.cat(predictions, dim=1)


def test_forced_declaration_that_does_not_match_the_feed_grades_exogenous() -> None:
    model = _ForcedButLying()
    log = tl.trace(
        model,
        torch.tensor([[1]]),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=3, forced_tokens=(9, 9, 9)),
    )
    envelope = _envelope(log)
    assert envelope["grades"] == [None, "exogenous", "exogenous"]
    assert envelope["exogenous_positions"] == {"1": 1, "2": 1}


# ---------------------------------------------------------------------------
# 3.1 -- tail alignment must be licensed by the source width.
# ---------------------------------------------------------------------------


class _TwoTokensPerStep(nn.Module):
    """Multi-token decoding: two positions appended per stepped call."""

    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.step = _Step()
        self.n = n

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n):
            top2 = self.step(ids).topk(2, dim=-1).indices
            ids = torch.cat([ids, top2], dim=1)
        return ids[:, -2 * self.n :]


def test_multi_token_decoding_refuses_instead_of_misattributing() -> None:
    model = _TwoTokensPerStep()
    with pytest.raises(EpisodeDeclarationError) as info:
        tl.trace(
            model,
            torch.tensor([[1]]),
            episode=EpisodeSpec(stepped_module=model.step, n_steps=3),
        )
    assert info.value.fields["code"] == "episode_declaration_invalid"
    message = str(info.value)
    assert "6 positions" in message
    assert "exactly 3 (emitted-only root) or 4" in message
    assert "one position per step" in message.lower() or "ONE position per step" in message


class _PromptPlusCompletion(nn.Module):
    """A generate()-shaped root: prompt + one emission per step."""

    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.step = _Step()
        self.n = n

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n):
            ids = torch.cat([ids, self.step(ids).argmax(-1, keepdim=True)], dim=1)
        return ids


def test_prompt_plus_completion_shape_stays_licensed() -> None:
    model = _PromptPlusCompletion()
    log = tl.trace(
        model,
        torch.tensor([[1, 2, 3, 4]]),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=3),
    )
    ledger = episode_ledger_for(log)
    assert ledger is not None
    assert ledger.step_output_series() == ((5,), (6,), (7,))
    assert _envelope(log)["grades"] == [None, "continuous", "continuous"]
