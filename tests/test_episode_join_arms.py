"""Episode join arms (lane F40c): live + settlement ``step_join``.

The trace-verb verdict's mid-call oracle break contract: per-row join grades
(continuous / transformed / declared / exogenous / unchecked), the DEFAULT
return-and-disclose arm with episode-dependent claims refused across a
measured break, the FORK-4 ``on_feed_break="refuse"`` arm (typed refusal
carrying recoverable partial evidence), the ``feed="closed"`` STRICT arm
halting at the next step entry (one-step detection latency), declared
crossings, and both verbatim teaching texts.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._episode_join import ENTRY_SNAPSHOT_ELEMENT_CEILING
from torchlens.capture._episode_ledger import (
    derive_episode_status,
    episode_ledger_for,
    escalation_spec,
)
from torchlens.errors import EpisodeDeclarationError, EpisodeJoinError
from torchlens.options import EpisodeSpec


class TinyLM(nn.Module):
    """Minimal stepped model: embedding -> mean -> head logits."""

    def __init__(self, vocab: int = 16, width: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.emb(ids).mean(1)))


class GreedyRunner(nn.Module):
    """Append feed: each step sees the full growing context."""

    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for _ in range(self.n_steps):
            next_token = self.model(current).argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


class LastTokenRunner(nn.Module):
    """KV-cache-shaped feed: each step sees ONLY the previous emission."""

    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for _ in range(self.n_steps):
            next_token = self.model(current).argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = next_token
        return torch.cat(tokens, dim=1)


class WindowedRunner(nn.Module):
    """Sliding-context feed: each step sees a bounded SUFFIX of the history."""

    def __init__(self, model: nn.Module, n_steps: int, window: int = 3):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.window = window

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for _ in range(self.n_steps):
            next_token = self.model(current).argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)[:, -self.window :]
        return torch.cat(tokens, dim=1)


class ToolInjectRunner(nn.Module):
    """The tool-call shape: exogenous tokens enter the context mid-loop."""

    def __init__(self, model: nn.Module, n_steps: int, inject_at: int = 2, n_inject: int = 2):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.inject_at = inject_at
        self.n_inject = n_inject

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for step in range(self.n_steps):
            if step == self.inject_at:
                tool = torch.full((ids.shape[0], self.n_inject), 9, dtype=ids.dtype)
                current = torch.cat([current, tool], dim=1)
            next_token = self.model(current).argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


class Denoiser(nn.Module):
    """Float stepped model (the diffusion/fixed-point shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - 0.1 * self.lin(x)


class DirectLoop(nn.Module):
    """Direct float feed: step k+1's input IS step k's output."""

    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            x = self.model(x)
        return x


class TransformedLoop(nn.Module):
    """Captured transform between steps (the scheduler shape)."""

    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            x = self.model(x) * 0.5
        return x


class ReplacedLoop(nn.Module):
    """Context replacement: step inputs never continue the prior output."""

    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for step in range(self.n_steps):
            out = self.model(x + float(step))
        return out


def _prompt() -> torch.Tensor:
    return torch.tensor([[1, 2, 3]])


def _envelope(log) -> dict:
    return log.annotations["episode"]["header"]["step_join"]


# ---------------------------------------------------------------------------
# Continuous feeds: append, last-token, windowed suffix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runner_cls", [GreedyRunner, LastTokenRunner, WindowedRunner])
def test_continuous_feeds_grade_continuous(runner_cls) -> None:
    model = TinyLM()
    log = tl.trace(
        runner_cls(model, 4), _prompt(), episode=EpisodeSpec(stepped_module=model, n_steps=4)
    )
    envelope = _envelope(log)
    assert envelope["schema"] == "episode_step_join_v1"
    assert envelope["claim"] == "measured"
    assert envelope["grades"] == [None, "continuous", "continuous", "continuous"]
    assert envelope["break_step"] is None
    assert envelope["live_break_step"] is None
    # The series claim is grantable on a measured continuous episode.
    series = log.episode.step_output_series()
    assert len(series) == 4 and all(isinstance(step, tuple) for step in series)
    # The blessing fold blesses.
    result = derive_episode_status(
        [("COMPLETE", None)] * 4, n_declared=4, ledger=episode_ledger_for(log)
    )
    assert result.status == "episode_complete"


def test_direct_float_feed_grades_continuous_on_digest_basis() -> None:
    model = Denoiser()
    log = tl.trace(
        DirectLoop(model, 3),
        torch.randn(1, 6),
        episode=EpisodeSpec(stepped_module=model, n_steps=3, step_output_kind="none"),
    )
    envelope = _envelope(log)
    assert envelope["grades"] == [None, "continuous", "continuous"]
    assert set(envelope["basis"].values()) == {"digest_direct"}


def test_captured_transform_grades_transformed_via_graph_witness() -> None:
    model = Denoiser()
    log = tl.trace(
        TransformedLoop(model, 3),
        torch.randn(1, 6),
        episode=EpisodeSpec(stepped_module=model, n_steps=3, step_output_kind="none"),
    )
    envelope = _envelope(log)
    assert envelope["grades"] == [None, "transformed", "transformed"]
    assert set(envelope["basis"].values()) == {"graph"}
    assert envelope["break_step"] is None


@pytest.mark.smoke
def test_context_replacement_grades_exogenous() -> None:
    model = Denoiser()
    log = tl.trace(
        ReplacedLoop(model, 3),
        torch.randn(1, 6),
        episode=EpisodeSpec(stepped_module=model, n_steps=3, step_output_kind="none"),
    )
    envelope = _envelope(log)
    assert envelope["grades"][1] == "exogenous"
    assert envelope["break_step"] == 1


# ---------------------------------------------------------------------------
# DEFAULT arm: return-and-disclose; claims refuse across the break.
# ---------------------------------------------------------------------------


def _broken_capture(n_inject: int = 2):
    model = TinyLM()
    log = tl.trace(
        ToolInjectRunner(model, 4, inject_at=2, n_inject=n_inject),
        _prompt(),
        episode=EpisodeSpec(stepped_module=model, n_steps=4),
    )
    return log


def test_default_arm_returns_trace_and_marks_break() -> None:
    log = _broken_capture()
    envelope = _envelope(log)
    assert envelope["grades"] == [None, "continuous", "exogenous", "continuous"]
    assert envelope["break_step"] == 2
    assert envelope["exogenous_positions"] == {"2": 2}
    # One-step detection latency: the live check flagged the SAME join at
    # the step-2 entry, the first observable point.
    assert envelope["live_break_step"] == 2
    # The root call itself may truthfully be COMPLETE.
    assert log.outcome.status.name == "COMPLETE"


def test_default_arm_series_read_refuses_with_verbatim_teaching_text() -> None:
    log = _broken_capture()
    with pytest.raises(EpisodeJoinError) as excinfo:
        log.episode.step_output_series()
    assert excinfo.value.fields["code"] == "episode_feed_break_exogenous"
    message = str(excinfo.value)
    assert "Step 2's input does not continue step 1's output" in message
    assert "2 positions entered this loop from outside the capture" in message
    assert "chain-shaped at this join" in message
    assert "Token-series reads and whole-episode replay are refused" in message
    assert "ops, values, graph, and per-segment replay are unaffected" in message
    assert "declare `feed='closed'`" in message


def test_default_arm_whole_episode_replay_refuses() -> None:
    log = _broken_capture()
    with pytest.raises(EpisodeJoinError) as excinfo:
        log.run(fast=True)
    assert excinfo.value.fields["code"] == "episode_feed_break_exogenous"
    with pytest.raises(EpisodeJoinError):
        log.run()


def test_default_arm_blessing_fold_refuses_typed() -> None:
    log = _broken_capture()
    with pytest.raises(EpisodeJoinError) as excinfo:
        derive_episode_status(
            [("COMPLETE", None)] * 4, n_declared=4, ledger=episode_ledger_for(log)
        )
    assert excinfo.value.fields["code"] == "episode_feed_break_exogenous"


@pytest.mark.smoke
def test_default_arm_escalation_refuses() -> None:
    log = _broken_capture()
    model = TinyLM()
    with pytest.raises(EpisodeJoinError) as excinfo:
        escalation_spec(log, stepped_module=model, reason="requested")
    assert excinfo.value.fields["code"] == "episode_feed_break_exogenous"


def test_default_arm_per_segment_facts_stay_usable() -> None:
    """Ops, values, graph, and per-row reads are unaffected by the break."""

    log = _broken_capture()
    ledger = log.episode
    assert all(row.status == "complete" for row in ledger.rows)
    assert all(isinstance(row.step_output, tuple) for row in ledger.rows)
    output_op = log.output_ops[0]
    assert output_op.out is not None
    assert len(log.layer_list) > 0
    assert log["argmax_1_5"].parents  # graph edges readable


# ---------------------------------------------------------------------------
# FORK-4 arm (b): typed refusal carrying recoverable partial evidence.
# ---------------------------------------------------------------------------


def test_refuse_arm_raises_at_settlement_with_partial_evidence() -> None:
    model = TinyLM()
    with pytest.raises(EpisodeJoinError) as excinfo:
        tl.trace(
            ToolInjectRunner(model, 4),
            _prompt(),
            episode=EpisodeSpec(stepped_module=model, n_steps=4, on_feed_break="refuse"),
        )
    exc = excinfo.value
    assert exc.fields["code"] == "episode_feed_break_exogenous"
    assert "2 positions entered this loop from outside the capture" in str(exc)
    # Recoverable partial evidence: the settled product rides the refusal
    # WITH its measured envelope and full ledger.
    partial = exc.partial_log
    assert partial is not None
    payload = partial.trace.annotations["episode"]
    assert payload["header"]["step_join"]["break_step"] == 2
    assert [row["status"] for row in payload["rows"]] == ["complete"] * 4


# ---------------------------------------------------------------------------
# STRICT arm: feed="closed".
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_closed_feed_halts_at_next_step_entry_with_verbatim_text() -> None:
    """One-step detection latency, pinned by the row geometry."""

    model = TinyLM()
    with pytest.raises(EpisodeJoinError) as excinfo:
        tl.trace(
            ToolInjectRunner(model, 4),
            _prompt(),
            episode=EpisodeSpec(stepped_module=model, n_steps=4, feed="closed"),
        )
    exc = excinfo.value
    assert exc.fields["code"] == "episode_feed_closed_violation"
    assert exc.fields["break_step"] == 2
    message = str(exc)
    assert "`feed='closed'` was declared" in message
    assert "step 2 received input that does not continue step 1's output" in message
    assert "Capture stopped at the break: rows 0-1 complete, row 2 interrupted" in message
    assert "later rows absent" in message
    assert "Re-run without `feed='closed'`" in message
    # The partial product's ledger pins the halt geometry: the break step
    # ENTERED (its input arrived) and stopped; later steps never started.
    payload = exc.partial_log.trace.annotations["episode"]
    assert [row["status"] for row in payload["rows"]] == [
        "complete",
        "complete",
        "interrupted",
        "absent",
    ]
    envelope = payload["header"]["step_join"]
    assert envelope["grades"][2] == "exogenous"
    assert envelope["break_step"] == 2


def test_closed_feed_continuous_episode_captures_normally() -> None:
    model = TinyLM()
    log = tl.trace(
        GreedyRunner(model, 4),
        _prompt(),
        episode=EpisodeSpec(stepped_module=model, n_steps=4, feed="closed"),
    )
    assert _envelope(log)["break_step"] is None
    assert log.outcome.status.name == "COMPLETE"


@pytest.mark.smoke
def test_closed_feed_with_declared_crossing_stops_before_entry() -> None:
    model = TinyLM()
    with pytest.raises(EpisodeJoinError) as excinfo:
        tl.trace(
            ToolInjectRunner(model, 4),
            _prompt(),
            episode=EpisodeSpec(stepped_module=model, n_steps=4, feed="closed", crossings=(2,)),
        )
    exc = excinfo.value
    assert exc.fields["code"] == "episode_declared_crossing_stop"
    assert exc.fields["break_step"] == 2
    payload = exc.partial_log.trace.annotations["episode"]
    assert payload["header"]["step_join"]["grades"][2] == "declared"


# ---------------------------------------------------------------------------
# Declared-crossing arm (default feed): disclosed, chain-shaped.
# ---------------------------------------------------------------------------


def test_declared_crossing_grades_declared_and_claims_refuse() -> None:
    model = TinyLM()
    log = tl.trace(
        ToolInjectRunner(model, 4),
        _prompt(),
        episode=EpisodeSpec(stepped_module=model, n_steps=4, crossings=(2,)),
    )
    envelope = _envelope(log)
    assert envelope["grades"] == [None, "continuous", "declared", "continuous"]
    assert envelope["break_step"] == 2
    with pytest.raises(EpisodeJoinError) as excinfo:
        log.episode.step_output_series()
    assert excinfo.value.fields["code"] == "episode_join_declared_crossing"
    assert "DECLARED exogenous boundary" in str(excinfo.value)
    with pytest.raises(EpisodeJoinError):
        log.run(fast=True)


# ---------------------------------------------------------------------------
# Unchecked joins: measurement failure degrades with a reason, never a guess.
# ---------------------------------------------------------------------------


def test_oversize_entries_degrade_to_digest_basis_never_a_guess(monkeypatch) -> None:
    monkeypatch.setattr("torchlens.capture._episode_join.ENTRY_SNAPSHOT_ELEMENT_CEILING", 0)
    model = TinyLM()
    log = tl.trace(
        GreedyRunner(model, 3), _prompt(), episode=EpisodeSpec(stepped_module=model, n_steps=3)
    )
    envelope = _envelope(log)
    # Token ids were not snapshotted (ceiling 0); entry digests never match
    # the prior step's exit (logits vs ids), so the graph witness decides.
    assert envelope["grades"] == [None, "transformed", "transformed"]
    assert ENTRY_SNAPSHOT_ELEMENT_CEILING > 0  # the real ceiling is positive


@pytest.mark.smoke
def test_unchecked_join_refuses_series_claim() -> None:
    """A hand-built envelope with an unchecked join refuses the series read."""

    from torchlens.capture._episode_ledger import EpisodeLedger

    model = TinyLM()
    log = tl.trace(
        GreedyRunner(model, 3), _prompt(), episode=EpisodeSpec(stepped_module=model, n_steps=3)
    )
    payload = log.annotations["episode"]
    import json

    tampered = json.loads(json.dumps(payload))
    tampered["header"]["step_join"]["grades"][1] = "unchecked"
    tampered["header"]["step_join"]["basis"]["1"] = "none"
    tampered["header"]["step_join"]["reasons"]["1"] = "entry_snapshot_failed:probe"
    ledger = EpisodeLedger.from_payload(tampered)
    with pytest.raises(EpisodeJoinError) as excinfo:
        ledger.step_output_series()
    assert excinfo.value.fields["code"] == "episode_join_unmeasured"
    assert "entry_snapshot_failed:probe" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Declaration validation.
# ---------------------------------------------------------------------------


def test_feed_and_break_arm_tokens_are_closed_vocabularies() -> None:
    model = TinyLM()
    runner = GreedyRunner(model, 2)
    with pytest.raises(EpisodeDeclarationError, match="closed vocabulary"):
        tl.trace(
            runner,
            _prompt(),
            episode=EpisodeSpec(stepped_module=model, n_steps=2, feed="ajar"),  # type: ignore[arg-type]
        )
    with pytest.raises(EpisodeDeclarationError, match="closed vocabulary"):
        tl.trace(
            runner,
            _prompt(),
            episode=EpisodeSpec(stepped_module=model, n_steps=2, on_feed_break="warn"),  # type: ignore[arg-type]
        )


def test_crossing_declarations_are_validated() -> None:
    model = TinyLM()
    runner = GreedyRunner(model, 2)
    with pytest.raises(EpisodeDeclarationError, match="prefill"):
        tl.trace(
            runner,
            _prompt(),
            episode=EpisodeSpec(stepped_module=model, n_steps=2, crossings=(0,)),
        )
    with pytest.raises(EpisodeDeclarationError, match="only 2 steps"):
        tl.trace(
            runner,
            _prompt(),
            episode=EpisodeSpec(stepped_module=model, n_steps=2, crossings=(5,)),
        )


def test_forced_feed_joins_still_measure() -> None:
    """Teacher forcing is non-verifying, but its joins are measured facts."""

    model = TinyLM()

    class ForcedRunner(nn.Module):
        def __init__(self, inner: nn.Module, forced: tuple[int, ...]):
            super().__init__()
            self.inner = inner
            self.forced = forced

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            emitted = []
            current = ids
            for token in self.forced:
                self.inner(current)
                forced_token = torch.tensor([[token]])
                emitted.append(forced_token)
                current = torch.cat([current, forced_token], dim=1)
            return torch.cat(emitted, dim=1)

    log = tl.trace(
        ForcedRunner(model, (5, 6, 7)),
        _prompt(),
        episode=EpisodeSpec(stepped_module=model, n_steps=3, forced_tokens=(5, 6, 7)),
    )
    envelope = _envelope(log)
    assert envelope["claim"] == "measured"
    # The forced feed appends the DECLARED token; the join is graded against
    # the declaration (W051, audit 3.4), never against the evidence column --
    # this fixture's root happens to return the forced tokens, which made the
    # historical `continuous` grade degenerate.
    assert envelope["grades"] == [None, "forced", "forced"]
    assert envelope["basis"] == {"1": "forced_tokens", "2": "forced_tokens"}
    assert log.annotations["episode"]["header"]["token_feed"] == "forced"
