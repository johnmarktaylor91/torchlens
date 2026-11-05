"""Episode honesty refusals (lane F40a): pre-execution intervene refusal,
declared-step cost ceiling, partial_trace on settlement refusals, and the
discoverability seam (capture_kind / episode step access)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._capture._episode_ledger import (
    EPISODE_DECLARED_STEP_CEILING,
)
from torchlens.errors._episode import EpisodeDeclarationError, EpisodeLedgerError
from torchlens.options import EpisodeOptions
from torchlens.types import PartialTrace

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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_coupling.py, _capture/_episode_join.py, _capture/_episode_ledger.py ....
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
        episode=EpisodeOptions(stepped_module=model.step, num_steps=N_STEPS),
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_ledger.py.
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
                episode=EpisodeOptions(
                    stepped_module=model.step,
                    num_steps=EPISODE_DECLARED_STEP_CEILING + 1,
                ),
            )
    finally:
        monkey.undo()
    assert excinfo.value.fields["code"] == "episode_step_ceiling_exceeded"
    assert forward_calls == []


def _assert_settlement_refusal_keeps_product(excinfo: pytest.ExceptionInfo) -> None:
    """Shared assertions: settlement refusals carry the settled product."""

    partial = getattr(excinfo.value, "partial_trace", None)
    assert isinstance(partial, PartialTrace)
    assert partial.trace is not None
    assert len(partial.trace._layer_list) > 0


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_ledger.py.
def test_settlement_count_mismatch_carries_partial_trace():
    """episode_ledger_incoherent at settlement keeps the settled product."""

    model = _Greedy()
    with pytest.raises(EpisodeLedgerError) as excinfo:
        tl.trace(
            model,
            torch.tensor([[1]]),
            episode=EpisodeOptions(stepped_module=model.step, num_steps=N_STEPS + 1),
        )
    assert excinfo.value.fields["code"] == "episode_ledger_incoherent"
    _assert_settlement_refusal_keeps_product(excinfo)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/_episode_derivation.py.
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
            episode=EpisodeOptions(stepped_module=model.step, num_steps=N_STEPS),
        )
    assert excinfo.value.fields["code"] == "episode_declaration_invalid"
    assert "step_output_kind='digest'" in str(excinfo.value)
    assert "'none'" in str(excinfo.value)
    _assert_settlement_refusal_keeps_product(excinfo)
