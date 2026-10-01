"""Episode full-surface differential oracle + the gap-9 cross-tier token pin.

Three independent production paths over the SAME episode are compared on
their full public surface:

1. the SESSION product (one wrapped capture, the ruled representation),
2. the FLOOR product (N observer-discipline per-step captures in a Bundle
   with S6 episode_member rows, folded by the ratified 2.3 derivation),
3. the guarded-fast re-run of the session product (``trace.run(fast=True)``).

The gap-9 pin: the L2 spike measured wrapped-vs-native token identity only;
guarded-fast-vs-wrapped identity was UNMEASURED, and the E-A3 escalation
contract activates only once this pin holds. If these tests go red the
escalation contract must not be relied on until the divergence is
root-caused.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._episode_ledger import (
    episode_ledger_for,
    escalation_spec,
)
from torchlens.options import EpisodeSpec

pytestmark = pytest.mark.smoke


class TinyLM(nn.Module):
    """Minimal stepped model with deterministic parameters."""

    def __init__(self, vocab: int = 16, width: int = 8, seed: int = 7):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(self.emb.weight.shape, generator=generator))
            self.head.weight.copy_(torch.randn(self.head.weight.shape, generator=generator))
            self.head.bias.zero_()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.emb(ids).mean(1)))


class GreedyRunner(nn.Module):
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


class SampledRunner(nn.Module):
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


N_STEPS = 4
PROMPT = [[1, 2, 3]]


def _prompt() -> torch.Tensor:
    return torch.tensor(PROMPT)


def _native_greedy(model: nn.Module, ids: torch.Tensor, n_steps: int) -> list[int]:
    tokens = []
    current = ids
    with torch.no_grad():
        for _ in range(n_steps):
            next_token = model(current).argmax(-1, keepdim=True)
            tokens.append(int(next_token))
            current = torch.cat([current, next_token], dim=1)
    return tokens


def test_session_vs_floor_full_surface_differential():
    """SESSION and FLOOR products of one episode agree on every shared fact."""

    model = TinyLM()
    native_tokens = _native_greedy(model, _prompt(), N_STEPS)

    # SESSION product: one wrapped capture.
    session = tl.trace(
        GreedyRunner(model, N_STEPS),
        _prompt(),
        episode=EpisodeSpec(stepped_module=model, n_steps=N_STEPS),
    )
    assert session.outcome.status.name == "COMPLETE"
    ledger = episode_ledger_for(session)
    assert ledger is not None
    session_tokens = [row.step_output[0] for row in ledger.rows]

    # FLOOR product: observer discipline — the DRIVER owns the run; each
    # capture observes cloned step inputs; a capture product is never the
    # source of carried episode state (E-B2).
    members: dict[str, tl.Trace] = {}
    relations = []
    driver_tokens: list[int] = []
    current = _prompt()
    for step in range(N_STEPS):
        member = tl.trace(model, current.clone())
        members[f"step_{step}"] = member
        relations.append(
            {
                "kind": "episode_member",
                "member": f"step_{step}",
                "params": {
                    "episode_id": ledger.header.episode_id,
                    "at_step": step,
                    "role": "prefill" if step == 0 else "decode",
                },
            }
        )
        with torch.no_grad():
            next_token = model(current).argmax(-1, keepdim=True)
        driver_tokens.append(int(next_token))
        current = torch.cat([current, next_token], dim=1)
    bundle = tl.Bundle(members, member_relations=relations)

    # Token identity across all three sources of truth.
    assert session_tokens == native_tokens == driver_tokens

    # Fold vs settled outcome: the floor derivation agrees with the session
    # product's settled truth.
    # Arm 2 consumes the LEDGER-ONLY declared step count (fold arms 2/4 are
    # ledger-witnessed; without a re-supplied ledger the all-COMPLETE prefix
    # degrades fail-closed to episode_unknown — the disclosed pre-bump truth
    # loss). Supplying the ledger re-derives the full verdict.
    assert bundle.derive_episode_status(ledger.header.episode_id).status == "episode_unknown"
    fold = bundle.derive_episode_status(ledger.header.episode_id, ledger=ledger)
    assert fold.status == "episode_complete"
    assert fold.provenance_tier == "exact"

    # Per-step forward truth: each floor member's output equals the session
    # product's stepped-module call output bit-exactly.
    session_calls = session.modules["model"].calls
    assert len(session_calls) == N_STEPS == len(members)
    for step in range(N_STEPS):
        member_out = members[f"step_{step}"].output_ops[0].out
        call = session_calls[step]
        call_out = session[call.output_ops[0]].out
        assert torch.equal(member_out, call_out), f"step {step} diverged across homes"

    # Ledger geometry vs member statuses: every member settled COMPLETE and
    # every session row is complete.
    assert all(m.outcome.status.name == "COMPLETE" for m in members.values())
    assert all(row.status == "complete" for row in ledger.rows)


def test_gap9_guarded_fast_rerun_token_identity():
    """GAP-9 PIN: guarded-fast re-run reproduces the wrapped episode tokens
    bit-exactly. The E-A3 escalation contract activates only while this holds."""

    model = TinyLM()
    runner = GreedyRunner(model, N_STEPS)
    inputs = _prompt()
    # Guarded-fast recollects functional payloads only for predicate-requested
    # sites, so the pin captures the realistic episode shape: the stepped
    # module's interior plus the output-parent site.
    session = tl.trace(
        runner,
        inputs,
        save=tl.in_module("model") | tl.func("cat"),
        episode=EpisodeSpec(stepped_module=model, n_steps=N_STEPS),
    )
    ledger = episode_ledger_for(session)
    assert ledger is not None
    wrapped_tokens = [row.step_output[0] for row in ledger.rows]

    result = session.run(inputs=_prompt(), fast=True)
    fast_tokens = [int(v) for v in result.output.reshape(-1).tolist()]
    assert fast_tokens == wrapped_tokens, (
        "cross-tier token identity broken: guarded-fast and wrapped-episode "
        "tiers disagree — the escalation contract (E-A3) must not activate "
        "until this is root-caused"
    )


def test_escalation_reruns_whole_episode_and_verifies_tokens():
    """E-A1/E-A2/E-A3: a session escalation is a NEW whole-episode wrapped
    capture with the producer digest + reason disclosed and the token
    fidelity discharged at write time."""

    model = TinyLM()
    spec = EpisodeSpec(stepped_module=model, n_steps=N_STEPS)
    capture = tl.options.CaptureOptions(random_seed=777)
    producer = tl.trace(SampledRunner(model, N_STEPS), _prompt(), episode=spec, capture=capture)
    producer_ledger = episode_ledger_for(producer)
    assert producer_ledger is not None

    esc_spec = escalation_spec(producer, stepped_module=model, reason="requested")
    assert esc_spec.escalated_from is not None
    escalated = tl.trace(
        SampledRunner(model, N_STEPS),
        _prompt(),
        episode=esc_spec,
        capture=tl.options.CaptureOptions(random_seed=producer.random_seed),
    )
    ledger = episode_ledger_for(escalated)
    assert ledger is not None
    assert ledger.header.escalated_from == esc_spec.escalated_from
    assert ledger.header.reason == "requested"
    # Same declaration + same recorded entry seed -> bit-identical episode,
    # so the E-A3 comparison discharges as verified token fidelity.
    assert ledger.header.fidelity_basis == "tokens"
    assert [row.step_output for row in ledger.rows] == [
        row.step_output for row in producer_ledger.rows
    ]


def test_escalation_divergence_is_disclosed_never_settled():
    """A diverged escalation records fidelity_basis='diverged' on its ledger
    header while the product itself still settles through the ordinary
    authority (COMPLETE capture of what it actually ran)."""

    model = TinyLM()
    spec = EpisodeSpec(stepped_module=model, n_steps=N_STEPS)
    producer = tl.trace(
        SampledRunner(model, N_STEPS),
        _prompt(),
        episode=spec,
        capture=tl.options.CaptureOptions(random_seed=777),
    )
    esc_spec = escalation_spec(producer, stepped_module=model, reason="divergence")
    escalated = tl.trace(
        SampledRunner(model, N_STEPS),
        _prompt(),
        episode=esc_spec,
        capture=tl.options.CaptureOptions(random_seed=778),  # different stream
    )
    ledger = episode_ledger_for(escalated)
    assert ledger is not None
    producer_tokens = [
        row.step_output
        for row in episode_ledger_for(producer).rows  # type: ignore[union-attr]
    ]
    escalated_tokens = [row.step_output for row in ledger.rows]
    if escalated_tokens == producer_tokens:
        pytest.skip("seeds coincided; divergence scenario did not materialize")
    assert ledger.header.fidelity_basis == "diverged"
    # Settlement unchanged: the escalated capture is a valid COMPLETE capture
    # of what it ran — it just is not an escalation of the original episode.
    assert escalated.outcome.status.name == "COMPLETE"
