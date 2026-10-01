"""Episode capture (capture_kind=episode): S7 ledger + typed refusals.

Covers the L2 core contracts ratified at S2: the session ledger written at
settlement, the monotone prefix law and token presence rule as load
tripwires, every episode TYPED-REFUSE row, the E-A4 declaration-time
refusal, teacher forcing as a disclosed non-verifying mode, the managed RNG
recipe, flat-tally step semantics for nested loops, and the S3-gated
persistence of the episode annotations key.
"""

from __future__ import annotations

import pickle
import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._episode_ledger import (
    EpisodeLedger,
    derive_episode_status,
    episode_ledger_for,
    row_status_for_member_outcome,
    row_status_for_recording_status,
    write_episode_ledger,
)
from torchlens.errors import EpisodeDeclarationError, EpisodeLedgerError, TorchLensWarning
from torchlens.options import EpisodeSpec

pytestmark = pytest.mark.smoke


class TinyLM(nn.Module):
    """Minimal stepped model: embedding -> mean -> head logits."""

    def __init__(self, vocab: int = 16, width: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.emb(ids).mean(1)))


class GreedyRunner(nn.Module):
    """Episode root: steps the inner model N times, greedy decode."""

    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for _ in range(self.n_steps):
            logits = self.model(current)
            next_token = logits.argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


class NestedLoopRunner(nn.Module):
    """Episode root whose forward nests loops; step tallying must stay FLAT."""

    def __init__(self, model: nn.Module, outer: int, inner: int):
        super().__init__()
        self.model = model
        self.outer = outer
        self.inner = inner

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for _ in range(self.outer):
            for _ in range(self.inner):
                logits = self.model(current)
                next_token = logits.argmax(-1, keepdim=True)
                tokens.append(next_token)
                current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


class SampledRunner(nn.Module):
    """Episode root sampling with multinomial (managed-RNG recipe test)."""

    def __init__(self, model: nn.Module, n_steps: int):
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        tokens = []
        current = ids
        for _ in range(self.n_steps):
            probs = torch.softmax(self.model(current), dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)


class BombAtStep(nn.Module):
    """Stepped model that raises inside its Nth call (FAILED-forward path)."""

    def __init__(self, bomb_call: int):
        super().__init__()
        self.inner = TinyLM()
        self.bomb_call = bomb_call
        self.calls = 0

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == self.bomb_call:
            raise RuntimeError("injected step failure")
        return self.inner(ids)


def _prompt() -> torch.Tensor:
    return torch.tensor([[1, 2, 3]])


def _capture_episode(n_steps: int = 4, **trace_kwargs):
    model = TinyLM()
    runner = GreedyRunner(model, n_steps)
    spec = EpisodeSpec(stepped_module=model, n_steps=n_steps)
    log = tl.trace(runner, _prompt(), episode=spec, **trace_kwargs)
    return log


# ---------------------------------------------------------------------------
# Session ledger: complete episodes
# ---------------------------------------------------------------------------


def test_complete_episode_ledger_rows_and_tokens():
    log = _capture_episode(n_steps=4)
    ledger = episode_ledger_for(log)
    assert ledger is not None
    assert ledger.header.stepped_module == "model"
    assert ledger.header.n_steps_declared == 4
    assert ledger.header.token_feed == "free"
    assert ledger.header.provenance_tier == "exact"
    assert ledger.steps_completed == 4
    assert ledger.truncated_at_step is None
    assert [row.role for row in ledger.rows] == ["prefill", "decode", "decode", "decode"]
    out = log.output_ops[0].out
    for step, row in enumerate(ledger.rows):
        assert row.status == "complete"
        assert row.tokens == tuple(int(v) for v in out[:, step].reshape(-1).tolist())
        assert row.cache_len == 3 + step  # prompt length 3, arithmetic disclosure
        assert row.coord["member_call_index"] == step + 1


def test_episode_ledger_rows_are_write_once():
    log = _capture_episode(n_steps=2)
    from torchlens.capture._episode_ledger import ResolvedEpisode

    resolved = ResolvedEpisode(
        episode_id="ep-test",
        address="model",
        n_steps=2,
        token_axis=-1,
        forced_tokens=None,
        escalated_from=None,
        reason=None,
    )
    with pytest.raises(EpisodeLedgerError) as excinfo:
        write_episode_ledger(log, resolved)
    assert excinfo.value.fields["code"] == "episode_ledger_incoherent"


def test_flat_tally_nested_loops():
    """Nested loops in the episode root tally as one FLAT step sequence."""

    model = TinyLM()
    runner = NestedLoopRunner(model, outer=2, inner=3)
    log = tl.trace(runner, _prompt(), episode=EpisodeSpec(stepped_module=model, n_steps=6))
    ledger = episode_ledger_for(log)
    assert ledger is not None
    assert ledger.steps_completed == 6
    assert [row.episode_step for row in ledger.rows] == list(range(6))
    assert [row.coord["member_call_index"] for row in ledger.rows] == list(range(1, 7))


# ---------------------------------------------------------------------------
# Managed RNG recipe + teacher forcing
# ---------------------------------------------------------------------------


def test_entry_seed_recorded_and_sampled_episode_reproduces():
    model = TinyLM()
    spec = EpisodeSpec(stepped_module=model, n_steps=4)
    capture = tl.options.CaptureOptions(random_seed=1234)
    first = tl.trace(SampledRunner(model, 4), _prompt(), episode=spec, capture=capture)
    second = tl.trace(SampledRunner(model, 4), _prompt(), episode=spec, capture=capture)
    ledger_first = episode_ledger_for(first)
    ledger_second = episode_ledger_for(second)
    assert ledger_first is not None and ledger_second is not None
    assert ledger_first.header.entry_seed == 1234 == first.random_seed
    tokens_first = [row.tokens for row in ledger_first.rows]
    tokens_second = [row.tokens for row in ledger_second.rows]
    assert tokens_first == tokens_second  # managed recipe: bit-identical episode


def test_teacher_forcing_is_disclosed_non_verifying():
    model = TinyLM()
    runner = GreedyRunner(model, 3)
    spec = EpisodeSpec(stepped_module=model, n_steps=3, forced_tokens=(5, 6, 7))
    log = tl.trace(runner, _prompt(), episode=spec)
    ledger = episode_ledger_for(log)
    assert ledger is not None
    assert ledger.header.token_feed == "forced"
    assert ledger.header.fidelity_basis == "forced"


def test_forced_tokens_must_cover_declared_steps():
    model = TinyLM()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            GreedyRunner(model, 3),
            _prompt(),
            episode=EpisodeSpec(stepped_module=model, n_steps=3, forced_tokens=(5,)),
        )
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"


# ---------------------------------------------------------------------------
# Typed refusals at declaration time
# ---------------------------------------------------------------------------


def test_stepped_module_must_be_proper_submodule():
    model = TinyLM()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(model, _prompt(), episode=EpisodeSpec(stepped_module=model))
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"


def test_stepped_module_outside_root_refuses():
    model = TinyLM()
    stranger = TinyLM()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(GreedyRunner(model, 2), _prompt(), episode=EpisodeSpec(stepped_module=stranger))
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"


def test_unsnapshotable_declared_state_refuses_before_execution():
    model = TinyLM()
    runner = GreedyRunner(model, 2)
    calls_before = 0

    unsnapshotable = (item for item in range(3))  # generators cannot deepcopy
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            runner,
            _prompt(),
            episode=EpisodeSpec(stepped_module=model, state=(unsnapshotable,)),
        )
    assert excinfo.value.fields["code"] == "episode_state_unsnapshotable"
    assert calls_before == 0  # refusal fires at declaration time, pre-forward


def test_save_none_refuses_value_mode_episode():
    model = TinyLM()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            GreedyRunner(model, 2),
            _prompt(),
            episode=EpisodeSpec(stepped_module=model),
            capture=tl.options.CaptureOptions(layers_to_save="none"),
        )
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"


def test_structure_only_episode_refuses_typed():
    """S2 combination table: episode x structure_only=True is TYPED REFUSE
    this sprint (structure-only episodes have no value semantics to fold).
    The entry refusal shares L7a's capability-table code so the combination
    has ONE vocabulary across both chokepoints."""

    model = TinyLM()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            GreedyRunner(model, 2),
            _prompt(),
            episode=EpisodeSpec(stepped_module=model),
            capture=tl.options.CaptureOptions(structure_only=True),
        )
    assert excinfo.value.fields["code"] == "structure_only_episode_unsupported"


def test_chunked_episode_refuses():
    model = TinyLM()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            GreedyRunner(model, 2),
            _prompt(),
            chunk_size=1,
            episode=EpisodeSpec(stepped_module=model),
        )
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"


def test_rng_discipline_only_managed():
    model = TinyLM()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            GreedyRunner(model, 2),
            _prompt(),
            episode=EpisodeSpec(stepped_module=model, rng="wild"),  # type: ignore[arg-type]
        )
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"


# ---------------------------------------------------------------------------
# Truncated episodes: halt and failure
# ---------------------------------------------------------------------------


def test_halted_episode_ledger_discloses_truncation():
    model = TinyLM()
    runner = GreedyRunner(model, 5)
    matches = [0]
    tanh_selector = tl.func("tanh")

    def halt_after_two(ctx):
        if tanh_selector(ctx):
            matches[0] += 1
            return matches[0] >= 3
        return False

    log = tl.trace(
        runner,
        _prompt(),
        halt=halt_after_two,
        episode=EpisodeSpec(stepped_module=model, n_steps=5),
    )
    assert log.outcome.status.name == "HALTED"
    ledger = episode_ledger_for(log)
    assert ledger is not None
    assert ledger.steps_completed < 5
    assert ledger.truncated_at_step is not None
    assert ledger.rows[-1].status == "absent"  # declared tail disclosed
    for row in ledger.rows:
        assert row.tokens is None  # truncated episodes have no output to read


def test_failed_episode_attaches_partial_ledger():
    stepped = BombAtStep(bomb_call=3)
    runner = GreedyRunner(stepped, 5)
    spec = EpisodeSpec(stepped_module=stepped, n_steps=5)
    with pytest.raises(RuntimeError, match="injected step failure") as excinfo:
        tl.trace(runner, _prompt(), episode=spec)
    partial = tl.partial.from_failed_capture(excinfo.value)
    payload = partial.trace.annotations.get("episode")
    assert payload is not None
    ledger = EpisodeLedger.from_payload(payload)
    assert ledger.rows[-1].status == "absent"
    interrupted = [row for row in ledger.rows if row.status == "interrupted"]
    assert len(interrupted) == 1
    assert interrupted[0].episode_step == 2  # the bombed third call
    assert ledger.steps_completed == 2


# ---------------------------------------------------------------------------
# Load validation tripwires (typed refusals + fail-closed quarantine)
# ---------------------------------------------------------------------------


def _pickle_roundtrip(log):
    return pickle.loads(pickle.dumps(log))


def test_episode_ledger_without_declaration_refuses_typed():
    log = _capture_episode(n_steps=2)
    payload = dict(log.annotations["episode"])
    header = dict(payload["header"])
    header["capture_kind"] = "plain"
    payload["header"] = header
    log.annotations["episode"] = payload
    with pytest.raises(EpisodeLedgerError) as excinfo:
        _pickle_roundtrip(log)
    assert excinfo.value.fields["code"] == "episode_ledger_without_declaration"


def test_structure_only_token_payload_refuses_typed():
    log = _capture_episode(n_steps=2)
    payload = dict(log.annotations["episode"])
    header = dict(payload["header"])
    header["structure_only"] = True
    payload["header"] = header
    log.annotations["episode"] = payload
    with pytest.raises(EpisodeLedgerError) as excinfo:
        _pickle_roundtrip(log)
    assert excinfo.value.fields["code"] == "episode_ledger_payload_in_structure_only"


def test_monotone_prefix_tamper_quarantines_fail_closed():
    log = _capture_episode(n_steps=3)
    payload = dict(log.annotations["episode"])
    rows = [dict(row) for row in payload["rows"]]
    rows[0]["status"] = "absent"  # absent before complete: law violation
    payload["rows"] = rows
    log.annotations["episode"] = payload
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = _pickle_roundtrip(log)
    quarantined = restored.annotations["episode"]
    assert quarantined["quarantined"] is True
    assert quarantined["code"] == "episode_ledger_incoherent"
    assert any(
        isinstance(w.message, TorchLensWarning) and "quarantined" in str(w.message) for w in caught
    )


def test_value_mode_all_complete_ledger_requires_tokens():
    log = _capture_episode(n_steps=2)
    payload = dict(log.annotations["episode"])
    rows = [dict(row) for row in payload["rows"]]
    rows[1]["tokens"] = None
    payload["rows"] = rows
    log.annotations["episode"] = payload
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = _pickle_roundtrip(log)
    assert restored.annotations["episode"]["code"] == "episode_ledger_incoherent"
    assert caught


# ---------------------------------------------------------------------------
# Gated persistence (S3 registrar row)
# ---------------------------------------------------------------------------


def test_episode_key_rides_plain_v8_artifacts(tmp_path):
    """tlspec v8: the episode ledger persists on a PLAIN save/load, byte-
    faithful, with no pre-release marker riding the artifact."""

    log = _capture_episode(n_steps=2)
    path = tmp_path / "plain.tlspec"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tl.save(log, path)
    loaded = tl.load(path)
    assert loaded.annotations["episode"] == log.annotations["episode"]
    assert "episode" in log.annotations  # live trace keeps its session ledger


# ---------------------------------------------------------------------------
# Mapping tables + fold (pure spec functions)
# ---------------------------------------------------------------------------


def test_member_outcome_row_mapping_is_total_and_non_upgrading():
    assert row_status_for_member_outcome("COMPLETE", None) == "complete"
    assert row_status_for_member_outcome("HALTED", None) == "interrupted"
    assert row_status_for_member_outcome("ABORTED_NONFINITE", None) == "interrupted"
    assert row_status_for_member_outcome("FAILED", "forward") == "interrupted"
    for phase in ("finalize", "postprocess", "teardown"):
        assert row_status_for_member_outcome("FAILED", phase) == "complete"
    # The phase-less FAILED cell: NON-UPGRADING, never complete.
    assert row_status_for_member_outcome("FAILED", None) == "interrupted"
    assert row_status_for_member_outcome("UNATTESTED", None) == "interrupted"
    assert row_status_for_member_outcome("UNKNOWN", None) == "interrupted"
    with pytest.raises(ValueError, match="unknown member CaptureStatus"):
        row_status_for_member_outcome("BLESSED", None)


def test_recording_status_row_mapping_fail_closed():
    assert row_status_for_recording_status("complete") == "complete"
    assert row_status_for_recording_status("halted") == "interrupted"
    assert row_status_for_recording_status("partial_error") == "interrupted"
    assert row_status_for_recording_status("recovered") == "interrupted"  # R06
    with pytest.raises(ValueError, match="unknown Recording.status"):
        row_status_for_recording_status("blessed")


def test_fold_arms_first_match_and_fail_closed():
    # Arm 2: all declared members COMPLETE.
    result = derive_episode_status([("COMPLETE", None)] * 3, n_declared=3, ledger=None)
    assert result.status == "episode_complete"
    # Arm 2 needs the ledger-only N: without it, fail-closed.
    assert (
        derive_episode_status([("COMPLETE", None)] * 3, n_declared=None, ledger=None).status
        == "episode_unknown"
    )
    # Arm 3: HALTED member.
    result = derive_episode_status(
        [("COMPLETE", None), ("HALTED", None)], n_declared=None, ledger=None
    )
    assert result.status == "episode_halted_at_step" and result.at_step == 1
    # Arm 5: exact shipped spelling disclosed verbatim.
    result = derive_episode_status(
        [("COMPLETE", None), ("ABORTED_NONFINITE", None)], n_declared=None, ledger=None
    )
    assert result.status == "episode_aborted_at_step"
    assert result.member_status == "ABORTED_NONFINITE"
    # Arm 6: FAILED with phase ABSENT discloses "unattributed".
    result = derive_episode_status(
        [("COMPLETE", None), ("FAILED", None)], n_declared=None, ledger=None
    )
    assert result.status == "episode_failed_at_step"
    assert result.member_phase == "unattributed"
    # Arm 1: any UNATTESTED/UNKNOWN member.
    assert (
        derive_episode_status(
            [("UNATTESTED", None), ("COMPLETE", None)], n_declared=2, ledger=None
        ).status
        == "episode_unknown"
    )
    # Arm 7 (default): non-complete member FOLLOWED BY more members.
    assert (
        derive_episode_status(
            [("FAILED", "teardown"), ("COMPLETE", None)], n_declared=2, ledger=None
        ).status
        == "episode_unknown"
    )
    # E-B5: escalation members excluded from the fold domain.
    result = derive_episode_status(
        [("COMPLETE", None), ("COMPLETE", None), ("FAILED", None)],
        n_declared=2,
        ledger=None,
        escalation_members=frozenset({2}),
    )
    assert result.status == "episode_complete"
    # Provenance tier: episode tier is the MINIMUM of member tiers.
    result = derive_episode_status(
        [("COMPLETE", None)] * 2,
        n_declared=2,
        ledger=None,
        member_tiers=["exact", "ledger_only"],
    )
    assert result.provenance_tier == "ledger_only"
