"""The four-clause episode declaration invariant + the priced settlement cost
(lane F40a, foldA D4/s3).

``episode=`` is a pure declaration in the sense the one-door rule needs:
declaring it changes NOTHING about the executed model call or the recorded
graph. The permanent regression is the FOUR-CLAUSE invariant -- identical
(1) ordered recorded op labels, (2) root output bytes, (3) post-run RNG
state, (4) stepped-module call count -- between episode-declared and
undeclared runs of the same seeded fixture. Wall time is NOT an oracle
(measured signs disagreed across labs; foldA D4).

The declaration is NOT literally free: settlement derives the token column
from the already-computed root output at exactly one ``aten.select.int``
plus one ``aten.view.default`` dispatcher read per declared step. That cost
is DISCLOSED and priced here, never denied. The CUDA per-step host-sync tail
is a named cluster measurement, not a claim this suite makes.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

import torchlens as tl
from torchlens._capture_honesty import episode_facts
from torchlens.capture import _episode_ledger as episode_ledger_module
from torchlens.capture._episode_ledger import (
    ResolvedEpisode,
    _derive_step_evidence,
    episode_step_join_claim,
)
from torchlens.options import CaptureOptions, EpisodeSpec

V = 16
N_STEPS = 3


class _SampledStep(nn.Module):
    """Stepped model that CONSUMES RNG (multinomial), so clause 3 has teeth."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, V)
        self.out = nn.Linear(V, V, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.emb(ids[:, -1]))


class _SampledRunner(nn.Module):
    """Sampling generation loop: one multinomial draw per step."""

    def __init__(self, n: int = N_STEPS) -> None:
        super().__init__()
        self.step = _SampledStep()
        self.n = n

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n):
            probs = torch.softmax(self.step(ids), dim=-1)
            ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=1)
        return ids[:, -self.n :]


def _seeded_capture(declare_episode: bool) -> tuple[tl.Trace, torch.Tensor]:
    """One seeded capture; returns the trace and the post-run RNG state."""

    torch.manual_seed(123)
    model = _SampledRunner()
    kwargs = {"capture": CaptureOptions(random_seed=777)}
    if declare_episode:
        kwargs["episode"] = EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS)
    trace = tl.trace(model, torch.tensor([[1]]), **kwargs)
    return trace, torch.get_rng_state()


def test_four_clause_declaration_invariant():
    """Declared and undeclared runs agree on all four clauses."""

    declared, declared_rng = _seeded_capture(declare_episode=True)
    undeclared, undeclared_rng = _seeded_capture(declare_episode=False)

    # Clause 1: identical ordered recorded op labels.
    declared_labels = [op.label for op in declared.layer_list]
    undeclared_labels = [op.label for op in undeclared.layer_list]
    assert declared_labels == undeclared_labels

    # Clause 2: bit-identical root output.
    declared_out = declared.output_ops[0].out
    undeclared_out = undeclared.output_ops[0].out
    assert declared_out.dtype == undeclared_out.dtype
    assert torch.equal(declared_out, undeclared_out)

    # Clause 3: identical post-run RNG state (the fixture consumes RNG).
    assert torch.equal(declared_rng, undeclared_rng)

    # Clause 4: identical stepped-module call count.
    declared_calls = len(declared.modules["step"].calls)
    undeclared_calls = len(undeclared.modules["step"].calls)
    assert declared_calls == undeclared_calls == N_STEPS

    # And the declared product actually carries its ledger (the declaration
    # did something -- on the product, never on the execution).
    assert declared.episode is not None
    assert undeclared.episode is None


class _AtenCounter(TorchDispatchMode):
    """Count aten dispatches by overload packet name."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


def test_settlement_cost_is_exactly_the_priced_dispatcher_reads():
    """The disclosed settlement price: one select + one view per declared step."""

    model = _SampledRunner()
    trace = tl.trace(
        model,
        torch.tensor([[1]]),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS),
    )
    resolved = ResolvedEpisode(
        episode_id="ep-price-probe",
        address="step",
        n_steps=N_STEPS,
        step_axis=-1,
        step_output_kind="tokens",
        step_output_from=None,
        forced_tokens=None,
        escalated_from=None,
        reason=None,
    )
    with _AtenCounter() as counter:
        tokens = _derive_step_evidence(trace, resolved, N_STEPS)
    assert tokens is not None and len(tokens) == N_STEPS
    assert counter.counts["aten.select.int"] == N_STEPS
    assert counter.counts["aten.view.default"] == N_STEPS


@pytest.mark.smoke
def test_step_join_claim_is_per_artifact_never_the_build_switch():
    """F40c flipped the claim-off switch WITH the measurement behind it; the
    honesty invariant survives per-artifact: an artifact without a measured
    envelope can never read as measured, whatever the build claims."""

    assert episode_ledger_module.EPISODE_STEP_JOIN_MEASURED is True
    assert episode_step_join_claim() == "measured"

    model = _SampledRunner()
    trace = tl.trace(
        model,
        torch.tensor([[1]]),
        episode=EpisodeSpec(stepped_module=model.step, n_steps=N_STEPS),
    )
    facts = episode_facts(trace)
    assert facts is not None
    assert facts["step_join"] == "continuous"  # measured on THIS artifact

    # Strip the artifact's envelope: the SAME build must read it unmeasured
    # -- the build switch never upgrades an artifact's own evidence, so the
    # F40a claim-off honesty holds for every pre-measurement artifact.
    trace.annotations["episode"]["header"]["step_join"] = None
    assert episode_facts(trace)["step_join"] == "unmeasured"
    report = tl.report.explain(trace)
    assert "Cross-step continuity is DECLARED, not measured" in report
