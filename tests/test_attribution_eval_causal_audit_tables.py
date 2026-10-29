"""The causal audit of direct-logit-attribution tables and one-backward read tables.

On a tiny linear-residual language model, each component's logit effect is known in closed
form, and every measured effect equals it exactly in float64; on a small linear network the
same holds for every site and element of an activation-times-gradient read. A correct claim
scores rank agreement 1.0 and a shuffled one less; every recorded ablation replays through
``trace.fork().do(...)`` to its measured effect; the caller's model and random state are
untouched; each refusal carries its code and a remedy that runs.

The language model's final LayerNorm has epsilon 2**60 and gain 2**30: on its small integer
residuals the variance is absorbed by the epsilon in float64 and float32 alike, so the norm is
exactly the centering map and the logits are linear in every writer. Its writers read a side
embedding that is not on the residual stream, so ablating one writer removes only its own
contribution: the total effect equals the direct one.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import re
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
import torchlens.attribution as attribution
import torchlens.mechinterp
from torchlens.mechinterp._records import ContributionScores

# Stated before the first run. The integer-valued float64 models are exact: measured effects,
# closed forms and replays are compared with ``==``. On the float32 language model with an
# ordinary LayerNorm a replayed effect matches the measured one within 1e-5 (absolute).
FLOAT32_REPLAY_ATOL = 1e-5

ANSWER, VS, POSITIONS = (1, 3), (2, 0), (-1, 0)


class Writer(nn.Module):
    """One residual writer: a bias-free linear map of the side embedding."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, side: Tensor) -> Tensor:
        """Write ``proj(side)`` onto the residual stream."""

        return self.proj(side)


class TinyLM(nn.Module):
    """``logits = lm_head(ln_f(embed(ids) + sum_i blocks[i](feat(ids))))``."""

    def __init__(self, *, exact: bool = True, vocab: int = 5, width: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.feat = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList([Writer(width) for _ in range(3)])
        self.ln_f = nn.LayerNorm(width, eps=2.0**60 if exact else 1e-5)
        self.lm_head = nn.Linear(width, vocab)
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for parameter in self.parameters():
                if exact:
                    parameter.copy_(torch.randint(-3, 4, parameter.shape, generator=generator))
                else:
                    parameter.copy_(torch.randn(parameter.shape, generator=generator))
            if exact:
                self.ln_f.weight.fill_(2.0**30)
                self.ln_f.bias.copy_(torch.tensor([1.0, -2.0, 0.0, 3.0]))
                self.lm_head.bias.copy_(torch.tensor([0.0, 1.0, -1.0, 2.0, 3.0]))

    def forward(self, ids: Tensor) -> Tensor:
        """Embedding plus three writers, final norm, unembedding."""

        hidden = self.embed(ids)
        side = self.feat(ids)
        for block in self.blocks:
            hidden = hidden + block(side)
        return self.lm_head(self.ln_f(hidden))


class LinearNet(nn.Module):
    """``out = fc2(fc1(x)) + side(x)``: every site enters the output linearly."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)
        self.side = nn.Linear(4, 2)
        generator = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randint(-3, 4, parameter.shape, generator=generator))

    def forward(self, x: Tensor) -> Tensor:
        """Linear path through two sites plus a side path."""

        return self.fc2(self.fc1(x)) + self.side(x)


IDS = torch.tensor([[1, 2, 3], [4, 0, 2]])
X = torch.tensor([[1.0, -2.0, 3.0, 0.0], [2.0, 1.0, -1.0, 1.0]], dtype=torch.float64)
SEED_INDEX = (0, 1)


def _lm(exact: bool = True) -> TinyLM:
    model = TinyLM(exact=exact).eval()
    return model.double() if exact else model


def _dla(model: nn.Module, ids: Tensor = IDS) -> ContributionScores:
    return torchlens.mechinterp.direct_logit_contributions(
        tl.trace(model, ids), answer=ANSWER, vs=VS, positions=POSITIONS
    )


def _component_outputs(model: TinyLM, ids: Tensor) -> list[Tensor]:
    with torch.no_grad():
        side = model.feat(ids)
        return [model.embed(ids)] + [block(side) for block in model.blocks]


def _logit_closed_form(model: TinyLM, ids: Tensor, change: Tensor) -> float:
    """``sum_{b, p, (a, v)} (U[a] - U[v]) . center(change)``: the logit effect of a change."""

    rows = [position % ids.shape[1] for position in POSITIONS]
    centered = change - change.mean(dim=-1, keepdim=True)
    weight = model.lm_head.weight.detach()
    total = 0.0
    for answer, vs in zip(ANSWER, VS, strict=True):
        total += float((centered[:, rows, :] @ (weight[answer] - weight[vs])).sum())
    return total


def _replacement(value: Tensor, ablation: str) -> Tensor:
    if ablation == "zero":
        return torch.zeros_like(value)
    if ablation == "mean":
        return value.mean(dim=0, keepdim=True).expand_as(value)
    return torch.ones_like(value)


def _audit_kwargs(ablation: str) -> dict[str, Any]:
    return {"ablation": 1.0 if ablation == "baseline" else ablation}


def _remedy(message: str) -> str:
    found = re.search(r"(tl\.attribution\.causal_audit\(.*\))\. See", message)
    assert found is not None, message
    return found.group(1)


def _logit_metric(audit: attribution.CausalAuditResult, run: Any, ids: Tensor = IDS) -> float:
    """The audited logit difference on a trace or fork, summed like the audit's score."""

    logits = run[audit.extra["logits_site"]].out.detach().to(torch.float64)
    rows = [position % ids.shape[1] for position in POSITIONS]
    terms = [
        logits[:, row, answer] - logits[:, row, vs]
        for row in rows
        for answer, vs in zip(ANSWER, VS, strict=True)
    ]
    return math.fsum(torch.stack(terms, dim=-1).flatten().tolist())


# ----------------------------------------------------------------------------------------------
# Direct-logit-attribution tables
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ablation", ["zero", "mean", "baseline"])
def test_dla_measured_effects_equal_the_closed_form(ablation: str) -> None:
    """Each component's measured logit effect equals the closed form exactly (float64)."""

    model = _lm()
    claim = _dla(model)
    audit = attribution.causal_audit(model, IDS, claim=claim, k=2, **_audit_kwargs(ablation))
    outputs = _component_outputs(model, IDS)
    expected = [
        _logit_closed_form(model, IDS, value - _replacement(value, ablation)) for value in outputs
    ]
    assert [row.measured_effect for row in audit.units] == expected
    assert [row.component for row in audit.units] == list(claim.labels)
    assert [row.site for row in audit.units] == [c.op_label for c in claim.coordinates]
    assert audit.layer is None and audit.site is None
    assert audit.extra["claim_kind"] == "direct_logit"
    if ablation == "zero":
        assert [row.claimed_effect for row in audit.units] == expected
        assert audit.rank_agreement == 1.0


def test_dla_constant_row_is_reported_apart_and_rows_add_to_the_native_logits() -> None:
    """The constant row has no site: it is reported, not ablated; rows plus it give native."""

    model = _lm()
    claim = _dla(model)
    audit = attribution.causal_audit(model, IDS, claim=claim, k=1, random_draws=1)
    bias = model.ln_f.bias.detach()
    head_bias = model.lm_head.bias.detach()
    weight = model.lm_head.weight.detach()
    per_cell = sum(
        float((weight[answer] - weight[vs]) @ bias + head_bias[answer] - head_bias[vs])
        for answer, vs in zip(ANSWER, VS, strict=True)
    )
    constant = per_cell * IDS.shape[0] * len(POSITIONS)
    assert constant != 0.0
    assert audit.extra["constant_row"] == constant
    assert len(audit.units) == len(claim.coordinates)
    assert math.fsum(row.claimed_effect for row in audit.units) + constant == audit.original_score
    assert audit.original_score == audit.extra["native_score"]


def test_dla_a_shuffled_claim_scores_below_the_correct_claim() -> None:
    """Permuting the rows' values breaks the ranking; the measured effects do not change."""

    model = _lm()
    claim = _dla(model)
    shuffled = ContributionScores(
        claim.coordinates,
        claim.values[torch.tensor([3, 2, 1, 0])],
        constant=claim.constant,
        native=claim.native,
        answer_tokens=claim.answer_tokens,
        vs_tokens=claim.vs_tokens,
        positions=claim.positions,
        identity_receipt=claim.identity_receipt,
    )
    correct = attribution.causal_audit(model, IDS, claim=claim, k=2)
    wrong = attribution.causal_audit(model, IDS, claim=shuffled, k=2)
    assert correct.rank_agreement == 1.0
    assert wrong.rank_agreement is not None and wrong.rank_agreement < correct.rank_agreement
    assert [row.measured_effect for row in wrong.units] == [
        row.measured_effect for row in correct.units
    ]


def test_dla_sets_add_up_and_random_draws_are_seeded() -> None:
    """Top, bottom and random sets measure the sum of their members (linear model)."""

    model = _lm()
    claim = _dla(model)
    audit = attribution.causal_audit(model, IDS, claim=claim, k=2, random_draws=3, seed=4)
    by_unit = {row.unit: row for row in audit.units}
    for item in audit.sets:
        assert item.measured_effect == math.fsum(by_unit[u].measured_effect for u in item.units)
        assert item.claimed_effect == math.fsum(by_unit[u].claimed_effect for u in item.units)
    order = sorted(by_unit, key=lambda u: -abs(by_unit[u].claimed_effect))
    assert audit.set_of("top")[0].units == tuple(order[:2])
    again = attribution.causal_audit(model, IDS, claim=claim, k=2, random_draws=3, seed=4)
    assert [s.units for s in again.set_of("random")] == [s.units for s in audit.set_of("random")]


@pytest.mark.parametrize("exact", [True, False], ids=["float64_exact", "float32_layernorm"])
def test_dla_recorded_ablations_replay_to_their_measured_effects(exact: bool) -> None:
    """Every row's and set's edit, replayed on a fresh trace, reproduces its measured effect."""

    model = _lm(exact)
    ids = IDS
    audit = attribution.causal_audit(model, ids, claim=_dla(model), k=2, random_draws=2)
    for item in (*audit.units, *audit.sets):
        trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(intervention_ready=True))
        fork = trace.fork()
        with torch.no_grad():
            for selection, edit in item.intervention:
                fork.do(selection, edit)
        replayed = audit.original_score - _logit_metric(audit, fork, ids)
        if exact:
            assert replayed == item.measured_effect
        else:
            assert abs(replayed - item.measured_effect) <= FLOAT32_REPLAY_ATOL
        fork.cleanup()


def test_dla_audit_leaves_the_caller_model_and_random_state_untouched() -> None:
    """Parameters and buffers stay bit-identical, the training flags and global RNG too."""

    model = TinyLM(exact=False).train()
    claim = _dla(model)
    model.train()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state()
    attribution.causal_audit(model, IDS, claim=claim, k=2, ablation="mean")
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.training and all(module.training for module in model.modules())
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("kind", ["other_inputs", "other_weights", "target_given"])
def test_dla_misaligned_claims_refuse_and_the_remedy_runs(kind: str) -> None:
    """A table whose native logits are not this run's, or a target= beside it, refuses."""

    model = _lm()
    if kind == "other_inputs":
        claim = _dla(model, torch.tensor([[0, 1, 4], [3, 3, 1]]))
    elif kind == "other_weights":
        other = _lm()
        with torch.no_grad():
            other.embed.weight.mul_(2.0)
        claim = _dla(other)
    else:
        claim = _dla(model)
    extra = {"target": 0} if kind == "target_given" else {}
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.causal_audit(model, IDS, claim=claim, k=1, **extra)
    assert excinfo.value.fields.get("code") == "attribution_audit_claim_misaligned"
    fixed = eval(_remedy(str(excinfo.value)), {"tl": tl, "model": model, "x": IDS})  # noqa: S307
    assert isinstance(fixed, attribution.CausalAuditResult) and fixed.rank_agreement == 1.0


def test_dla_a_row_without_a_site_refuses_and_the_remedy_runs() -> None:
    """A component row whose op label is not a captured site cannot be ablated: it refuses."""

    model = _lm()
    claim = _dla(model)
    coordinates = list(claim.coordinates)
    coordinates[2] = dataclasses.replace(coordinates[2], op_label=None, provenance="computed")
    broken = ContributionScores(
        tuple(coordinates),
        claim.values,
        constant=claim.constant,
        native=claim.native,
        answer_tokens=claim.answer_tokens,
        vs_tokens=claim.vs_tokens,
        positions=claim.positions,
        identity_receipt=claim.identity_receipt,
    )
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.causal_audit(model, IDS, claim=broken, k=1)
    assert excinfo.value.fields.get("code") == "attribution_audit_claim_unsupported"
    assert "blocks.1.proj" in str(excinfo.value)
    fixed = eval(_remedy(str(excinfo.value)), {"tl": tl, "model": model, "x": IDS})  # noqa: S307
    assert isinstance(fixed, attribution.CausalAuditResult)


def test_dla_k_larger_than_the_rows_refuses_and_the_remedy_runs() -> None:
    """``k`` above the number of component rows refuses, naming ``k=<rows>``."""

    model = _lm()
    claim = _dla(model)
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.causal_audit(model, IDS, claim=claim, k=9)
    assert excinfo.value.fields.get("code") == "attribution_eval_setting_invalid"
    scope = {"tl": tl, "model": model, "x": IDS, "claim": claim}
    fixed = eval(_remedy(str(excinfo.value)), scope)  # noqa: S307
    assert isinstance(fixed, attribution.CausalAuditResult)


# ----------------------------------------------------------------------------------------------
# One-backward read tables
# ----------------------------------------------------------------------------------------------


def _linear_sites(model: LinearNet, x: Tensor) -> dict[str, tuple[Tensor, Tensor]]:
    """Each read site's output and the output's derivative ``d out[0, 1] / d site``."""

    with torch.no_grad():
        hidden = model.fc1(x)
        through = model.fc2(hidden)
        side = model.side(x)
        out = through + side
    unit = torch.zeros_like(out)
    unit[SEED_INDEX] = 1.0
    hidden_grad = torch.zeros_like(hidden)
    hidden_grad[0] = model.fc2.weight.detach()[1]
    return {
        "linear_1_1:1": (hidden, hidden_grad),
        "linear_2_2:1": (through, unit),
        "linear_3_3:1": (side, unit),
        "add_1_4:1": (out, unit),
        "output_1:1": (out, unit),
    }


def _seed() -> Any:
    return attribution.seed("output_1", index=SEED_INDEX)


def _read(model: nn.Module, x: Tensor = X, reduce: str | None = "sum", **kwargs: Any) -> Any:
    return attribution.read(
        tl.trace(model, x),
        target=kwargs.pop("target", _seed()),
        frozen=None,
        reduce=reduce,
        **kwargs,
    )


@pytest.mark.parametrize("ablation", ["zero", "mean", "baseline"])
def test_read_site_rows_measure_the_closed_form(ablation: str) -> None:
    """Each site's measured effect is ``grad . (site - replacement)`` exactly (float64)."""

    model = LinearNet().eval().double()
    table = _read(model)
    audit = attribution.causal_audit(
        model, X, claim=table, target=_seed(), k=2, **_audit_kwargs(ablation)
    )
    sites = _linear_sites(model, X)
    expected = [
        float((grad * (value - _replacement(value, ablation))).sum())
        for value, grad in sites.values()
    ]
    assert [row.site for row in audit.units] == list(sites)
    assert [row.measured_effect for row in audit.units] == expected
    assert audit.extra["claim_kind"] == "read_table" and audit.extra["grain"] == "site"
    if ablation == "zero":
        assert [row.claimed_effect for row in audit.units] == expected
        assert audit.rank_agreement == 1.0


def test_read_element_rows_measure_the_closed_form() -> None:
    """Element-grain rows: each element's ablation effect equals its activation times gradient."""

    model = LinearNet().eval().double()
    audit = attribution.causal_audit(model, X, claim=_read(model, reduce=None), target=_seed(), k=3)
    sites = _linear_sites(model, X)
    expected = {}
    for row_index, (value, grad) in enumerate(sites.values()):
        for element in itertools.product(*(range(size) for size in value.shape[1:])):
            index = (slice(None), *element)
            expected[(row_index, *element)] = float((grad[index] * value[index]).sum())
    assert {row.unit: row.measured_effect for row in audit.units} == expected
    assert {row.unit: row.claimed_effect for row in audit.units} == expected
    assert audit.extra["grain"] == "element"


def test_read_a_shuffled_table_scores_below_the_correct_table() -> None:
    """Permuting the rows' scores breaks the ranking; the measured effects do not change."""

    model = LinearNet().eval().double()
    table = _read(model)
    rows = table.rows()
    scores = [row.score for row in rows][::-1]
    shuffled_rows = {
        row.key: dataclasses.replace(row, score=score)
        for row, score in zip(rows, scores, strict=True)
    }
    shuffled = type(table)(shuffled_rows, table.provenance, trace=table.trace)
    correct = attribution.causal_audit(model, X, claim=table, target=_seed(), k=2)
    wrong = attribution.causal_audit(model, X, claim=shuffled, target=_seed(), k=2)
    assert correct.rank_agreement == 1.0
    assert wrong.rank_agreement is not None and wrong.rank_agreement < 1.0


@pytest.mark.parametrize("reduce", ["sum", None], ids=["site", "element"])
def test_read_recorded_ablations_replay_to_their_measured_effects(reduce: str | None) -> None:
    """Every row's and set's edit, replayed on a fresh trace, reproduces its measured effect."""

    model = LinearNet().eval().double()
    audit = attribution.causal_audit(
        model, X, claim=_read(model, reduce=reduce), target=_seed(), k=2, random_draws=2
    )
    for item in (*audit.units, *audit.sets):
        trace = tl.trace(model, X, capture=tl.options.CaptureOptions(intervention_ready=True))
        fork = trace.fork()
        with torch.no_grad():
            for selection, edit in item.intervention:
                fork.do(selection, edit)
        replayed = audit.original_score - float(fork["output_1"].out[SEED_INDEX])
        assert replayed == item.measured_effect
        fork.cleanup()


def test_read_audit_leaves_the_caller_model_and_random_state_untouched() -> None:
    """Parameters, gradients, training flags and the global RNG are unchanged."""

    model = LinearNet().double().train()
    table = _read(model)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    grads = [parameter.grad for parameter in model.parameters()]
    rng = torch.get_rng_state()
    attribution.causal_audit(model, X, claim=table, target=_seed(), k=2, ablation="mean")
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert [parameter.grad for parameter in model.parameters()] == grads
    assert model.training and all(module.training for module in model.modules())
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("kind", ["other_seed", "no_target", "not_a_seed", "other_inputs"])
def test_read_misaligned_claims_refuse_and_the_remedy_runs(kind: str) -> None:
    """A target that is not the table's seed, or a table from other inputs, refuses."""

    model = LinearNet().eval().double()
    table = _read(model, X[:1] if kind == "other_inputs" else X)
    target: Any = {
        "other_seed": attribution.seed("output_1", index=(1, 0)),
        "no_target": None,
        "not_a_seed": 1,
        "other_inputs": _seed(),
    }[kind]
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.causal_audit(model, X, claim=table, target=target, k=1)
    assert excinfo.value.fields.get("code") == "attribution_audit_claim_misaligned"
    fixed = eval(_remedy(str(excinfo.value)), {"tl": tl, "model": model, "x": X})  # noqa: S307
    assert isinstance(fixed, attribution.CausalAuditResult) and fixed.rank_agreement == 1.0


@pytest.mark.parametrize("kind", ["edge_rows", "two_targets", "grad_method"])
def test_read_unsupported_tables_refuse_and_the_remedy_runs(kind: str) -> None:
    """Several targets and non-effect reads refuse; EDGE rows mixed into an ACT table refuse.

    A table of EDGE rows alone is audited by patching each edge
    (``tests/test_attribution_eval_causal_audit_edges.py``); one that mixes
    EDGE rows into site reads refuses, and the remedy audits the site reads.
    """

    model = LinearNet().eval().double()
    if kind == "edge_rows":
        table = _read(model)
        rows = {row.key: row for row in table.rows()}
        edge = dataclasses.replace(table.rows()[0], kind="EDGE", address=("fc1->fc2",))
        rows[edge.key] = edge
        claim = type(table)(rows, table.provenance, trace=table.trace)
    elif kind == "two_targets":
        claim = _read(model, target=[_seed(), attribution.seed("output_1", index=(1, 0))])
    else:
        claim = _read(model, method="grad")
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.causal_audit(model, X, claim=claim, target=_seed(), k=1)
    assert excinfo.value.fields.get("code") == "attribution_audit_claim_unsupported"
    if kind == "edge_rows":
        assert "mixes EDGE rows with ACT rows" in str(excinfo.value)
    scope = {"tl": tl, "model": model, "x": X, "claim": claim}
    fixed = eval(_remedy(str(excinfo.value)), scope)  # noqa: S307
    assert isinstance(fixed, attribution.CausalAuditResult) and fixed.rank_agreement == 1.0


def test_read_rows_without_a_score_are_reported_apart() -> None:
    """A row the read could not score (status not ok) is listed, not ablated."""

    model = LinearNet().eval().double()
    table = _read(model)
    rows = table.rows()
    unreachable = dataclasses.replace(
        rows[2], status="unreachable", status_reason="not_upstream_of_target", score=None
    )
    claim = type(table)(
        {row.key: (unreachable if row.key == unreachable.key else row) for row in rows},
        table.provenance,
        trace=table.trace,
    )
    audit = attribution.causal_audit(model, X, claim=claim, target=_seed(), k=1)
    assert audit.extra["rows_not_audited"] == (("linear_3_3:1", "unreachable"),)
    assert [row.unit for row in audit.units] == [(0,), (1,), (3,), (4,)]
