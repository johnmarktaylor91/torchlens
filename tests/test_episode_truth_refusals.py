"""Episode honesty refusals (lane F40a): pre-execution intervene refusal,
declared-step cost ceiling, partial_log on settlement refusals, and the
discoverability seam (capture_kind / episode step access)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.capture._episode_ledger import (
    EPISODE_DECLARED_STEP_CEILING,
    resolve_episode_declaration,
)
from torchlens.errors.episode import EpisodeDeclarationError, EpisodeLedgerError
from torchlens.options import EpisodeSpec
from torchlens.partial import PartialTrace, from_failed_capture

V = 16
N_STEPS = 3


class _Step(nn.Module):
    """Deterministic stepped model (next token = last token + 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, V)
        self.out = nn.Linear(V, V, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.eye(V))
            self.out.weight.copy_(torch.roll(torch.eye(V), shifts=1, dims=0))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.emb(ids[:, -1]))


class _Greedy(nn.Module):
    """Greedy generation loop returning the emitted-token tensor."""

    def __init__(self, n: int = N_STEPS) -> None:
        super().__init__()
        self.step = _Step()
        self.n = n

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n):
            ids = torch.cat([ids, self.step(ids).argmax(-1, keepdim=True)], dim=1)
        return ids[:, -self.n :]


class _FloatGreedy(_Greedy):
    """A float-emitting stepped root (the diffusion / hidden-state shape)."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return super().forward(ids).float()


class _PromptPlusCompletion(_Greedy):
    """A root returning what real ``generate()`` returns: prompt+completion."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n):
            ids = torch.cat([ids, self.step(ids).argmax(-1, keepdim=True)], dim=1)
        return ids


def test_episode_x_intervene_couples_with_the_evidence_bar_met():
    """The foldA D5 flip (lane F42): the cell is COUPLED, never silent.

    The D5 refusal was the floor UNDER the flip: it stood exactly until the
    verdict's evidence bar was met (product binding, step-qualified
    selectors, perturbed fidelity, per-step fire counts). This regression
    pins the flipped cell's evidence so a future change can never quietly
    fall back to the pre-verdict silent composition (outcome COMPLETE with
    zero coupling evidence -- the measured wrongness that forced D5).
    """

    model = _Greedy()
    log = tl.trace(
        model,
        torch.tensor([[1]]),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS),
        intervene=tl.when(tl.func("linear"), tl.zero_ablate()),
    )
    header = log.annotations["episode"]["header"]
    assert header["intervention_digest"] is not None
    assert header["capture_digest"] is not None
    assert header["fidelity_basis"] == "perturbed"
    counts = [row["fire_count"] for row in log.annotations["episode"]["rows"]]
    assert all(isinstance(count, int) for count in counts)
    attestation = log.episode_coupling
    assert attestation.coupled is True and attestation.bound is True


def test_step_ceiling_refuses_at_declaration_and_override_admits():
    """Declarations beyond the ceiling refuse typed; the override is explicit."""

    model = _Greedy()
    over = EPISODE_DECLARED_STEP_CEILING + 1

    with pytest.raises(EpisodeDeclarationError) as excinfo:
        resolve_episode_declaration(EpisodeSpec(stepped_module=model.step, n_steps=over), model)
    assert excinfo.value.fields["code"] == "episode_step_ceiling_exceeded"
    assert "acknowledge_step_cost" in str(excinfo.value)

    # The forced-tokens length is a declared step count too.
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        resolve_episode_declaration(
            EpisodeSpec(stepped_module=model.step, forced_tokens=tuple(range(over))),
            model,
        )
    assert excinfo.value.fields["code"] == "episode_step_ceiling_exceeded"

    # At the ceiling: admitted without the override.
    at_ceiling = resolve_episode_declaration(
        EpisodeSpec(stepped_module=model.step, n_steps=EPISODE_DECLARED_STEP_CEILING),
        model,
    )
    assert at_ceiling.n_steps == EPISODE_DECLARED_STEP_CEILING

    # Beyond the ceiling with the explicit acknowledgement: admitted.
    acknowledged = resolve_episode_declaration(
        EpisodeSpec(stepped_module=model.step, n_steps=over, acknowledge_step_cost=True),
        model,
    )
    assert acknowledged.n_steps == over


def test_entry_refusal_via_trace_keeps_root_unexecuted():
    """The ceiling fires through the tl.trace entry, before any forward."""

    forward_calls: list[int] = []
    original_forward = _Greedy.forward

    def counting_forward(self: _Greedy, ids: torch.Tensor) -> torch.Tensor:
        forward_calls.append(1)
        return original_forward(self, ids)

    model = _Greedy()
    monkey = pytest.MonkeyPatch()
    monkey.setattr(_Greedy, "forward", counting_forward)
    try:
        with pytest.raises(EpisodeDeclarationError) as excinfo:
            tl.trace(
                model,
                torch.tensor([[1]]),
                episode=EpisodeSpec(
                    stepped_module=model.step,
                    n_steps=EPISODE_DECLARED_STEP_CEILING + 1,
                ),
            )
    finally:
        monkey.undo()
    assert excinfo.value.fields["code"] == "episode_step_ceiling_exceeded"
    assert forward_calls == []


def _assert_settlement_refusal_keeps_product(excinfo: pytest.ExceptionInfo) -> None:
    """Shared assertions: settlement refusals carry the settled product."""

    partial = getattr(excinfo.value, "partial_log", None)
    assert isinstance(partial, PartialTrace)
    assert partial.trace is not None
    assert len(partial.trace.layer_list) > 0
    recovered = from_failed_capture(excinfo.value)
    assert recovered is partial


def test_settlement_count_mismatch_carries_partial_log():
    """episode_ledger_incoherent at settlement keeps the settled product."""

    model = _Greedy()
    with pytest.raises(EpisodeLedgerError) as excinfo:
        tl.trace(
            model,
            torch.tensor([[1]]),
            episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS + 1),
        )
    assert excinfo.value.fields["code"] == "episode_ledger_incoherent"
    _assert_settlement_refusal_keeps_product(excinfo)


def test_float_root_under_tokens_kind_teaches_digest_and_keeps_product():
    """The R13-C2 shape under the DEFAULT kind still refuses -- teaching.

    A float root is a declaration mismatch for ``step_output_kind='tokens'``
    (token ids are integers); the refusal names the two declarations that
    make float roots first-class (lane F40b) and keeps the settled product.
    The acceptance arms live in tests/test_episode_derivation_declared.py.
    """

    model = _FloatGreedy()
    with pytest.raises(EpisodeDeclarationError) as excinfo:
        tl.trace(
            model,
            torch.tensor([[1]]),
            episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS),
        )
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    assert "step_output_kind='digest'" in str(excinfo.value)
    assert "'none'" in str(excinfo.value)
    _assert_settlement_refusal_keeps_product(excinfo)


def test_prompt_plus_completion_matches_emitted_only_column():
    """The R13-C1 shape (real generate() return) STOPS refusing (foldA D8).

    The axis-length-equality refusal was root-output-shape guessing; the
    declared derivation reads the LAST n_steps positions (tail-aligned), so
    prompt+completion and emitted-only roots yield the SAME evidence column.
    """

    prompt = torch.tensor([[1]])
    generate_shaped = _PromptPlusCompletion()
    full_trace = tl.trace(
        generate_shaped,
        prompt,
        episode=EpisodeSpec(stepped_module=generate_shaped.step, n_steps=N_STEPS),
    )
    emitted_only = _Greedy()
    emitted_trace = tl.trace(
        emitted_only,
        prompt,
        episode=EpisodeSpec(stepped_module=emitted_only.step, n_steps=N_STEPS),
    )
    full_ledger = full_trace.episode
    emitted_ledger = emitted_trace.episode
    assert full_ledger is not None and emitted_ledger is not None
    full_column = [row.step_output for row in full_ledger.rows]
    emitted_column = [row.step_output for row in emitted_ledger.rows]
    assert full_column == emitted_column
    assert all(isinstance(entry, tuple) for entry in full_column)
    # The prompt prefix stays OUT of the evidence column.
    root_out = full_trace.output_ops[0].out
    assert root_out.shape[-1] == N_STEPS + prompt.shape[-1]
    assert full_ledger.header.step_output_from == "output"
    assert full_ledger.header.step_output_kind == "tokens"


def test_capture_kind_and_public_step_access():
    """Discoverability seam: Trace.capture_kind and Trace.episode."""

    model = _Greedy()
    episode_trace = tl.trace(
        model,
        torch.tensor([[1]]),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS),
    )
    assert episode_trace.capture_kind == "episode"
    ledger = episode_trace.episode
    assert ledger is not None
    assert ledger.steps_completed == N_STEPS
    assert ledger.truncated_at_step is None
    assert [row.status for row in ledger.rows] == ["complete"] * N_STEPS
    assert ledger.header.stepped_module == "step"

    plain_trace = tl.trace(model, torch.tensor([[1]]))
    assert plain_trace.capture_kind == "plain"
    assert plain_trace.episode is None

    product = episode_trace.run(inputs=torch.tensor([[9]]))
    assert product.trace.capture_kind == "plain"
    assert product.trace.episode is None
