"""The causal audit measures EDGE sets whose edges enter several receivers.

A top-k, bottom-k or random-k set of an EDGE table is patched in one
``torchlens.mechinterp.patch_edges`` call: each later receiver reads the forward as
patched so far (the MIB circuit-track convention). Its measured effect is
compared with the same set patched by the independent plain-PyTorch hook
patcher (``tests/support/multi_patch_reference.py``) within 1e-12 absolute in
float64 (the two compute ``input - s + s_corrupt`` in a different order, so
the bound is a rounding bound, not ``==``). Each set's recorded edits, one
``(selection, value)`` pair per receiver read from the patched fork, replay on
a fork of the audit's trace to its measured effect exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from support.eap_fixture import CLEAN, CORRUPT, Fixture, fixture_metric
from support.multi_patch_reference import HookedPatch, PatchSpec, gpt2_spec
from torch import Tensor, nn

import torchlens.attribution as attribution
import torchlens.mechinterp

_BOUND = 1e-12


def _reduced(values: Tensor, reduction: str) -> float:
    total = math.fsum(values.detach().to(torch.float64).flatten().tolist())
    return total / values.shape[0] if reduction == "mean" else total


def _fixture_spec(model: Fixture, graph: Any) -> PatchSpec:
    """The fixture's modules under its graph names (matched by module address)."""

    by_address = {node.module_address: name for name, node in graph.nodes.items()}
    m_h1, m_h2 = by_address["h1"], by_address["h2"]
    return PatchSpec(
        {
            "input": (model.embed, lambda args, out: out),
            m_h1: (model.h1, lambda args, out: out),
            m_h2: (model.h2, lambda args, out: out),
        },
        {m_h1: model.norm1, m_h2: model.norm2, "logits": model.norm_f},
        record={"lm_head": model.lm_head},
    )


@dataclass(frozen=True)
class _Case:
    """A model, the reference patcher's map of it, its clean and corrupted inputs, the metric."""

    model: nn.Module
    spec: PatchSpec
    clean: Tensor
    corrupt: Tensor
    metric: Callable[[Tensor], Tensor]

    def score(self, audit: Any, logits: Tensor) -> float:
        """The metric on ``logits``, reduced over examples as the audited table was."""

        return _reduced(self.metric(logits), audit.extra["example_reduction"])

    def reference_effect(self, audit: Any, names: list[str]) -> float:
        """``M(patched) - M(clean)`` of ``names`` patched together by the plain-hook patcher."""

        hooked = HookedPatch(self.model, self.spec)
        corrupt_nodes, _ = hooked.run(self.corrupt)
        patched: dict[str, list[str]] = {}
        for name in names:
            sender, receiver = name.split("->")
            patched.setdefault(receiver, []).append(sender)
        _, edited = hooked.run(self.clean, patched, corrupt_nodes)
        _, plain = hooked.run(self.clean)
        return self.score(audit, edited["lm_head"]) - self.score(audit, plain["lm_head"])

    def replayed(self, audit: Any, pairs: Any) -> float:
        """Replay recorded pairs, in order, on a fresh fork of the audit's trace."""

        fork = audit.extra["trace"].fork()
        try:
            with torch.no_grad():
                for selection, value in pairs:
                    fork.do(selection, value)
            return self.score(audit, fork[audit.extra["logits_site"]].out)
        finally:
            fork.cleanup()


def _check_sets(audit: Any, case: _Case, table: Any) -> int:
    """Every set (and row) is measured, matches the reference within the bound and replays exactly.

    Returns the number of sets whose edges enter more than one receiver.
    """

    assert audit.extra["sets_not_measured"] == ()
    name_of = {(index,): str(row.address[0]) for index, row in enumerate(table.rows())}
    spanning = 0
    for item in audit.sets:
        names = [name_of[unit] for unit in item.indices]
        receivers = {name.split("->")[1] for name in names}
        spanning += len(receivers) > 1
        want = case.reference_effect(audit, names)
        assert abs(item.measured_effect - want) <= _BOUND, (item, want)
        assert len(item.build_interventions) == len(receivers)
        replayed = case.replayed(audit, item.build_interventions) - audit.original_score
        assert replayed == item.measured_effect, item
    for row in audit.units:
        want = case.reference_effect(audit, [row.component])
        assert abs(row.measured_effect - want) <= _BOUND, (row, want)
    return spanning


def _refuse_patching(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("recorded edits come from the fork that measured them")


def test_fixture_sets_across_receivers_match_the_reference_and_replay(monkeypatch) -> None:
    """On the float64 fixture, every set (one receiver or several) is measured and replays."""

    model = Fixture().eval()
    table = torchlens.mechinterp.edge_scores(
        model, CLEAN, corrupted_input=CORRUPT, metric=fixture_metric
    )
    audit = attribution.causal_audit(
        model,
        CLEAN,
        claim=table,
        corrupted_input=CORRUPT,
        metric=fixture_metric,
        k=2,
        num_random_draws=4,
    )
    assert len(audit.units) == 5 and len(audit.sets) == 6
    graph = torchlens.mechinterp.component_graph(audit.extra["trace"])
    case = _Case(model, _fixture_spec(model, graph), CLEAN, CORRUPT, fixture_metric)
    with monkeypatch.context() as patched:
        patched.setattr(torchlens.mechinterp, "patch_edges", _refuse_patching)
        recorded = [item.build_interventions for item in (*audit.units, *audit.sets)]
    assert _check_sets(audit, case, table) >= 1
    for item, pairs in zip((*audit.units, *audit.sets), recorded, strict=True):
        assert [value.tolist() for _, value in item.build_interventions] == [
            value.tolist() for _, value in pairs
        ]
    every = attribution.causal_audit(
        model,
        CLEAN,
        claim=table,
        corrupted_input=CORRUPT,
        metric=fixture_metric,
        k=5,
        controls=(),
        per_unit=False,
    )
    (whole,) = every.sets
    assert len(whole.build_interventions) == 3  # m0, m1 and logits, written in that order
    assert _check_sets(every, case, table) == 1


def _tiny_gpt2() -> tuple[nn.Module, Tensor, Tensor]:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=64, n_positions=32)
    model = transformers.GPT2LMHeadModel(config).to(torch.float64).eval()
    model.set_attn_implementation("eager")
    return model, torch.randint(0, 64, (3, 6)), torch.randint(0, 64, (3, 6))


def _logit_difference(logits: Tensor) -> Tensor:
    return logits[:, -1, 1] - logits[:, -1, 2]


# Receivers of every kind (q, k, v, MLP, logits), a bias row, and edges whose
# senders sit downstream of other patched receivers (input->a0.h0<q> then
# a0.h0->a1.h1<q>; input->m0 then m0->m1 and m0->a1.h1<q>).
EDGES = [
    "input->a0.h0<q>",
    "input->a0.h1<k>",
    "input->m0",
    "a0.h1->a1.h0<v>",
    "m0->a1.h1<q>",
    "a0.h0->a1.h1<q>",
    "m0->m1",
    "b0->m1",
    "a1.h1->logits",
]


@pytest.mark.heavy
def test_tiny_gpt2_sets_across_receivers_match_the_reference_and_replay() -> None:
    """q, k, v, MLP and logits receivers patched together match the plain-hook patcher.

    The rows are not re-measured here (per_unit=False): the edge audit's own tests
    measure them against the same patches.
    """

    model, clean, corrupt = _tiny_gpt2()
    table = torchlens.mechinterp.edge_scores(
        model, clean, corrupted_input=corrupt, metric=_logit_difference, method="eap"
    )
    common = {
        "claim": table,
        "corrupted_input": corrupt,
        "metric": _logit_difference,
        "edges": EDGES,
    }
    audit = attribution.causal_audit(
        model, clean, k=3, num_random_draws=2, per_unit=False, **common
    )
    case = _Case(model, gpt2_spec(model), clean, corrupt, _logit_difference)
    assert len(audit.sets) == 4
    assert _check_sets(audit, case, table) >= 1
    every = attribution.causal_audit(
        model, clean, k=len(EDGES), controls=(), per_unit=False, **common
    )
    assert _check_sets(every, case, table) == 1
