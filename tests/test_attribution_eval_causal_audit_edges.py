"""``tl.attribution.causal_audit`` on EDGE read tables (edge attribution patching scores).

Each ``ok`` row of a ``torchlens.mechinterp.edge_scores`` table is audited by
patching its edge alone (``torchlens.mechinterp.patch_edges``, the sender's value
from the corrupted run) and measuring ``M(patched) - M(clean)``, reduced over
examples as the table was. On the deterministic nonlinear fixture
(``tests/support/eap_fixture.py``) every measured effect equals the edge's
patch effect computed here by hand, exactly in float64; a shuffled claim
scores lower rank agreement; every recorded edit replays to its measured
effect on a fresh trace; the caller's model and the global random state are
untouched; each refusal names a spelling that runs. A tiny random GPT-2
covers the q, k and v receivers, whose patches write one head's projection
slice.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable
from typing import Any

import pytest
import torch
from support.eap_fixture import CLEAN, CORRUPT, Fixture, fixture_metric
from torch import Tensor

import torchlens as tl
import torchlens.attribution as attribution
import torchlens.mechinterp

PAIR_CLEAN, PAIR_CORRUPT = torch.tensor([[0], [0]]), torch.tensor([[1], [1]])
_CAPTURE = tl.options.CaptureOptions(layers_to_save="all", intervention_ready=True)


def _remedy(message: str) -> str:
    found = re.search(r"(tl\.attribution\.causal_audit\(.*\))\. See", message)
    assert found is not None, message
    return found.group(1)


def _table(model: Any = None, clean: Tensor = CLEAN, corrupt: Tensor = CORRUPT, **kwargs: Any):
    model = Fixture().eval() if model is None else model
    return torchlens.mechinterp.edge_scores(
        model, clean, corrupted_input=corrupt, metric=fixture_metric, **kwargs
    )


def _reduced(values: Tensor, reduction: str) -> float:
    total = math.fsum(values.detach().to(torch.float64).flatten().tolist())
    return total / values.shape[0] if reduction == "mean" else total


def _metric(model: Fixture, residual: Tensor) -> Tensor:
    return fixture_metric(model.lm_head(model.norm_f(residual)))


def _hand_effects(model: Fixture, clean: Tensor, corrupt: Tensor, reduction: str) -> dict:
    """Each fixture edge's patch effect by hand: ``M(patched) - M(clean)``, reduced."""

    with torch.no_grad():
        e_c, e_x = model.embed(clean), model.embed(corrupt)
        u1_c, u2_c = model.h1(model.norm1(e_c)), model.h2(model.norm2(e_c))
        u1_x, u2_x = model.h1(model.norm1(e_x)), model.h2(model.norm2(e_x))
        r = e_c + u1_c + u2_c
        patched = {
            "input->logits": r - e_c + e_x,
            "m0->logits": r - u1_c + u1_x,
            "m1->logits": r - u2_c + u2_x,
            "input->m0": e_c + model.h1(model.norm1(e_c - e_c + e_x)) + u2_c,
            "input->m1": e_c + u1_c + model.h2(model.norm2(e_c - e_c + e_x)),
        }
        clean_score = _reduced(_metric(model, r), reduction)
        return {
            name: _reduced(_metric(model, value), reduction) - clean_score
            for name, value in patched.items()
        }


def _audit(table: Any, model: Any = None, clean: Tensor = CLEAN, **kwargs: Any):
    model = Fixture().eval() if model is None else model
    corrupt = kwargs.pop("corrupted", CORRUPT if clean is CLEAN else PAIR_CORRUPT)
    return attribution.causal_audit(
        model, clean, claim=table, corrupted_input=corrupt, metric=fixture_metric, **kwargs
    )


def _shuffled(table: Any) -> Any:
    """The same table with its scores reversed in rank order."""

    rows = table.rows()
    order = sorted(range(len(rows)), key=lambda index: rows[index].score)
    scores = {order[i]: rows[order[-1 - i]].score for i in range(len(rows))}
    swapped = {
        row.key: dataclasses.replace(row, score=scores[index]) for index, row in enumerate(rows)
    }
    return type(table)(swapped, table.provenance, trace=table.trace)


# ----------------------------------------------------------------------------------------------
# The fixture: measured effects against the closed form
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "reduction", "batch"),
    [("eap_ig", "mean", 1), ("eap", "mean", 1), ("eap_ig", "sum", 2), ("eap", "mean", 2)],
)
def test_fixture_measured_effects_equal_the_hand_patch(method: str, reduction: str, batch: int):
    """Every edge's measured effect is its hand-computed patch effect, exactly (float64)."""

    model = Fixture().eval()
    clean, corrupt = (CLEAN, CORRUPT) if batch == 1 else (PAIR_CLEAN, PAIR_CORRUPT)
    table = _table(model, clean, corrupt, method=method, example_reduction=reduction)
    audit = _audit(table, model, clean, corrupted_input=corrupt, k=1, num_random_draws=2)
    graph = torchlens.mechinterp.component_graph(tl.trace(model, clean))
    assert graph.nodes["m0"].module_address == "h1" and graph.nodes["m1"].module_address == "h2"
    want = _hand_effects(model, clean, corrupt, reduction)
    got = {row.component: row.measured_effect for row in audit.units}
    assert got == want
    claimed = {row.address[0]: row.score for row in table.rows()}
    assert {row.component: row.claimed_effect for row in audit.units} == claimed
    assert audit.extra["claim_kind"] == "edge_table"
    assert audit.extra["claim_method"] == method
    assert audit.extra["example_reduction"] == reduction
    assert audit.settings["ablation"] == "edge_patch"
    assert audit.original_score == _reduced(fixture_metric(model(clean).detach()), reduction)
    assert [row.unit for row in audit.units] == [(index,) for index in range(5)]


def test_fixture_a_shuffled_claim_scores_below_the_correct_claim() -> None:
    """Rank agreement of the EAP-IG table beats the same scores reversed in rank."""

    table = _table()
    correct = _audit(table, k=1, num_random_draws=2)
    shuffled = _audit(_shuffled(table), k=1, num_random_draws=2)
    assert correct.rank_agreement is not None and shuffled.rank_agreement is not None
    assert shuffled.rank_agreement < correct.rank_agreement
    assert [row.measured_effect for row in shuffled.units] == [
        row.measured_effect for row in correct.units
    ]


def test_fixture_sets_into_one_receiver_or_several_are_measured() -> None:
    """A set into one receiver is patched together, and so is a set across receivers."""

    model = Fixture().eval()
    table = _table(model)
    into_logits = ["input->logits", "m0->logits", "m1->logits"]
    audit = _audit(table, model, edges=into_logits, k=2, num_random_draws=3)
    assert sorted(row.component for row in audit.units) == sorted(into_logits)
    assert audit.extra["sets_not_measured"] == ()
    name_of = {row.unit: row.component for row in audit.units}
    top = audit.set_of("top")[0]
    with torch.no_grad():
        e_c, e_x = model.embed(CLEAN), model.embed(CORRUPT)
        values = {
            "input->logits": (e_c, e_x),
            "m0->logits": (model.h1(model.norm1(e_c)), model.h1(model.norm1(e_x))),
            "m1->logits": (model.h2(model.norm2(e_c)), model.h2(model.norm2(e_x))),
        }
        r = e_c + values["m0->logits"][0] + values["m1->logits"][0]
        patched = r
        for unit in top.units:
            sender_clean, sender_corrupt = values[name_of[unit]]
            patched = patched - sender_clean + sender_corrupt
        want = float(_metric(model, patched)[0]) - float(_metric(model, r)[0])
    assert top.measured_effect == want
    claimed_of = {row.unit: row.claimed_effect for row in audit.units}
    assert top.claimed_effect == math.fsum(claimed_of[unit] for unit in top.units)
    across = _audit(table, model, k=2, num_random_draws=2)
    assert across.extra["sets_not_measured"] == ()
    assert [item.kind for item in across.sets] == ["top", "bottom", "random", "random"]
    (top_across,) = across.set_of("top")
    assert len({name.split("->")[1] for name in _names(across, top_across)}) == 2


def _names(audit: Any, item: Any) -> list[str]:
    name_of = {row.unit: row.component for row in audit.units}
    return [name_of[unit] for unit in item.units]


def test_fixture_rows_without_a_score_are_reported_apart() -> None:
    """A row the scores could not reach (status not ok) is listed, never patched."""

    table = _table()
    rows = table.rows()
    unreachable = dataclasses.replace(rows[1], status="unreachable", score=None)
    claim = type(table)(
        {row.key: (unreachable if row.key == unreachable.key else row) for row in rows},
        table.provenance,
        trace=table.trace,
    )
    audit = _audit(claim, k=1, num_random_draws=2)
    assert audit.extra["rows_not_audited"] == ((rows[1].address[0], "unreachable"),)
    assert [row.unit for row in audit.units] == [(0,), (2,), (3,), (4,)]


@pytest.mark.heavy
def test_patch_edges_resolves_layer_zero_receivers() -> None:
    """A receiver reading the embedding output (a layer-0 receiver) is patched, not refused."""

    model = Fixture().eval()
    clean = tl.trace(model, CLEAN, capture=_CAPTURE)
    graph = torchlens.mechinterp.component_graph(clean)
    corrupt = torchlens.mechinterp.component_graph(tl.trace(model, CORRUPT, capture=_CAPTURE))
    patched = torchlens.mechinterp.patch_edges(
        clean, [graph.edges["input->m0"]], corrupted_graph=corrupt
    )
    with torch.no_grad():
        want = model.h1(model.norm1(model.embed(CORRUPT)))
    assert torch.equal(patched.modules["h1"].out, want)
    gpt2, ids, other = _tiny_gpt2()
    trace = tl.trace(gpt2, ids, capture=_CAPTURE)
    tiny = torchlens.mechinterp.component_graph(trace)
    tiny_corrupt = torchlens.mechinterp.component_graph(tl.trace(gpt2, other, capture=_CAPTURE))
    head = torchlens.mechinterp.patch_edges(
        trace, [tiny.edges["input->a0.h0<q>"]], corrupted_graph=tiny_corrupt
    )
    assert not torch.equal(
        head.modules["transformer.h.0.attn.c_attn"].out[..., :8],
        trace.modules["transformer.h.0.attn.c_attn"].out[..., :8],
    )


# ----------------------------------------------------------------------------------------------
# Replay and isolation
# ----------------------------------------------------------------------------------------------


def _replayed(audit: Any, pairs: Any, score: Callable[[Any], float]) -> float:
    """Replay recorded pairs on a fresh fork of the audit's trace (edge selections are bound to it)."""

    fork = audit.extra["trace"].fork()
    try:
        with torch.no_grad():
            for selection, value in pairs:
                fork.do(selection, value)
        return score(fork)
    finally:
        fork.cleanup()


def _logits_score(audit: Any, metric: Callable[[Tensor], Tensor]) -> Callable[[Any], float]:
    def _score(run: Any) -> float:
        return _reduced(metric(run[audit.extra["logits_site"]].out), "mean")

    return _score


def test_fixture_recorded_patches_replay_to_their_measured_effects() -> None:
    """Every row's and set's recorded patch, replayed on a fresh fork, is exact (float64)."""

    model = Fixture().eval()
    audit = _audit(_table(model), model, edges=["input->logits", "m1->logits"], k=2)
    score = _logits_score(audit, fixture_metric)
    for item in (*audit.units, *audit.sets):
        replayed = _replayed(audit, item.intervention, score) - audit.original_score
        assert replayed == item.measured_effect


def _tiny_gpt2():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=64, n_positions=32)
    model = transformers.GPT2LMHeadModel(config).to(torch.float64).eval()
    model.set_attn_implementation("eager")
    return model, torch.randint(0, 64, (3, 6)), torch.randint(0, 64, (3, 6))


def _logit_difference(logits: Tensor) -> Tensor:
    return logits[:, -1, 1] - logits[:, -1, 2]


QKV_EDGES = [
    "input->a0.h0<q>",
    "input->a0.h1<k>",
    "a0.h1->a1.h0<v>",
    "m0->a1.h1<q>",
    "a0.h0->a1.h1<q>",
    "m0->m1",
    "b0->m1",
    "a1.h1->logits",
]


@pytest.mark.heavy
def test_tiny_gpt2_q_k_v_patches_replay_to_their_measured_effects() -> None:
    """Head-slice patches (q, k, v), MLP and logits receivers, and a bias row replay exactly."""

    model, clean, corrupt = _tiny_gpt2()
    table = torchlens.mechinterp.edge_scores(
        model, clean, corrupted_input=corrupt, metric=_logit_difference
    )
    audit = attribution.causal_audit(
        model,
        clean,
        claim=table,
        corrupted_input=corrupt,
        metric=_logit_difference,
        edges=QKV_EDGES,
        k=2,
        num_random_draws=4,
    )
    assert sorted(row.component for row in audit.indices) == sorted(QKV_EDGES)
    by_name = {row.component: row for row in audit.indices}
    assert abs(by_name["b0->m1"].measured_effect) <= 1e-12
    assert by_name["b0->m1"].claimed_effect == 0.0
    score = _logits_score(audit, _logit_difference)
    for item in (*audit.indices, *audit.sets):
        replayed = _replayed(audit, item.intervention, score) - audit.original_score
        assert replayed == item.measured_effect, item
    shared = attribution.causal_audit(
        model,
        clean,
        claim=table,
        corrupted_input=corrupt,
        metric=_logit_difference,
        edges=["m0->a1.h1<q>", "a0.h0->a1.h1<q>"],
        k=2,
        controls=(),
    )
    (both,) = shared.sets
    replayed = _replayed(shared, both.build_interventions, score) - shared.original_score
    assert replayed == both.measured_effect


def test_edge_audit_leaves_the_caller_model_and_random_state_untouched() -> None:
    """Parameters, buffers, gradients, training flags and the global RNG are unchanged."""

    model = Fixture().train()
    table = _table(model)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    grads = [parameter.grad.clone() for parameter in model.parameters()]
    rng = torch.get_rng_state()
    _audit(table, model, k=2, num_random_draws=3)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(
        torch.equal(parameter.grad, grad)
        for parameter, grad in zip(model.parameters(), grads, strict=True)
    )
    assert model.training and all(module.training for module in model.modules())
    assert torch.equal(torch.get_rng_state(), rng)


# ----------------------------------------------------------------------------------------------
# Refusals: each names its code and a spelling that runs
# ----------------------------------------------------------------------------------------------


def _scope(model: Any, claim: Any, **names: Any) -> dict[str, Any]:
    return {
        "tl": tl,
        "torchlens": torchlens,
        "model": model,
        "x": CLEAN,
        "claim": claim,
        "corrupted": CORRUPT,
        "metric": fixture_metric,
        **names,
    }


def _misaligned_refusal_runs_its_remedy(model: Any, table: Any, call: dict[str, Any]) -> None:
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.causal_audit(model, CLEAN, claim=table, k=1, **call)
    assert excinfo.value.fields.get("code") == "attribution_audit_claim_misaligned"
    fixed = eval(_remedy(str(excinfo.value)), _scope(model, table))  # noqa: S307
    assert isinstance(fixed, attribution.CausalAudit)
    assert fixed.extra["claim_kind"] == "edge_table" and len(fixed.indices) == 5


# One family per cause, each well inside the 5s tier's aggregate family budget.
@pytest.mark.parametrize("kind", ["no_corrupted", "no_metric"])
def test_edge_missing_corrupted_or_metric_refuses_and_the_remedy_runs(kind: str) -> None:
    """No corrupted inputs, or no metric."""

    model = Fixture().eval()
    call: dict[str, Any] = {"corrupted": CORRUPT, "metric": fixture_metric}
    call.pop("corrupted" if kind == "no_corrupted" else "metric")
    _misaligned_refusal_runs_its_remedy(model, _table(model), call)


@pytest.mark.parametrize(
    "kind", ["corrupted_other_shape", "metric_not_per_example", "target_given"]
)
def test_edge_misaligned_audit_settings_refuse_and_the_remedy_runs(kind: str) -> None:
    """Corrupted inputs of another shape, a metric that is not per example, a target."""

    model = Fixture().eval()
    call: dict[str, Any] = {"corrupted": CORRUPT, "metric": fixture_metric}
    if kind == "corrupted_other_shape":
        call["corrupted"] = torch.tensor([[1, 1]])
    elif kind == "metric_not_per_example":
        call["metric"] = lambda logits: logits[:, -1]
    else:
        call["target"] = 0
    _misaligned_refusal_runs_its_remedy(model, _table(model), call)


@pytest.mark.parametrize("kind", ["other_inputs", "unresolvable_edge"])
def test_edge_misaligned_claim_tables_refuse_and_the_remedy_runs(kind: str) -> None:
    """A table built on other inputs, a table naming an edge the model does not have."""

    model = Fixture().eval()
    if kind == "other_inputs":
        table = _table(model, PAIR_CLEAN, PAIR_CORRUPT)
    else:
        table = _table(model)
        rows = table.rows()
        renamed = dataclasses.replace(rows[0], address=("a3.h1->m9",))
        table = type(table)(
            {**{row.key: row for row in rows[1:]}, renamed.key: renamed},
            table.provenance,
            trace=table.trace,
        )
    _misaligned_refusal_runs_its_remedy(
        model, table, {"corrupted": CORRUPT, "metric": fixture_metric}
    )


@pytest.mark.parametrize("kind", ["mixed_rows", "other_reduction", "edge_inputs_on_a_read"])
def test_edge_unsupported_tables_and_settings_refuse_and_the_remedy_runs(kind: str) -> None:
    """A table mixing ACT and EDGE rows, a foreign reduction, edge keywords on a read table."""

    model = Fixture().eval()
    table = _table(model)
    code = "attribution_audit_claim_unsupported"
    call: dict[str, Any] = {"corrupted": CORRUPT, "metric": fixture_metric}
    if kind == "mixed_rows":
        act = dataclasses.replace(table.rows()[0], kind="ACT", address=("linear_1_3", 1))
        claim = type(table)(
            {**{row.key: row for row in table.rows()}, act.key: act},
            table.provenance,
            trace=table.trace,
        )
    elif kind == "other_reduction":
        claim = type(table)(
            {row.key: dataclasses.replace(row, reduction="mean") for row in table.rows()},
            table.provenance,
            trace=table.trace,
        )
    else:
        target = attribution.seed("linear_3_9", index=(0, 0, 0))
        claim = attribution.read(tl.trace(model, CLEAN), target=target, frozen=None)
        call["target"] = target
        code = "attribution_eval_setting_invalid"
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.causal_audit(model, CLEAN, claim=claim, k=1, **call)
    assert excinfo.value.fields.get("code") == code
    scope = _scope(model, claim, target=call.get("target"))
    fixed = eval(_remedy(str(excinfo.value)), scope)  # noqa: S307
    assert isinstance(fixed, attribution.CausalAudit)


@pytest.mark.parametrize("kind", ["ablation", "unknown_edges", "k_too_large"])
def test_edge_bad_settings_refuse_and_the_remedy_runs(kind: str) -> None:
    """ablation= on an EDGE table, edges= naming no ok row, k above the audited edges."""

    model = Fixture().eval()
    table = _table(model)
    call: dict[str, Any] = {"corrupted": CORRUPT, "metric": fixture_metric, "k": 1}
    if kind == "ablation":
        call["ablation"] = "mean"
    elif kind == "unknown_edges":
        call["edges"] = ["input->m7"]
    else:
        call["k"] = 9
    with pytest.raises(attribution.AttributionError) as excinfo:
        attribution.causal_audit(model, CLEAN, claim=table, **call)
    assert excinfo.value.fields.get("code") == "attribution_eval_setting_invalid"
    fixed = eval(_remedy(str(excinfo.value)), _scope(model, table))  # noqa: S307
    assert isinstance(fixed, attribution.CausalAudit)


def test_many_edges_warn_once_and_the_named_spelling_runs(monkeypatch) -> None:
    """Past the per-unit guard an EDGE audit warns and names edges= (and per_unit=False)."""

    from torchlens.attribution import _causal_audit
    from torchlens.attribution._result import AttributionWarning

    monkeypatch.setattr(_causal_audit, "_PER_UNIT_WARN_COUNT", 3)
    model = Fixture().eval()
    table = _table(model)
    with pytest.warns(AttributionWarning) as record:
        _audit(table, model, k=1, num_random_draws=2)
    (warning,) = [item for item in record if "edges=" in str(item.message)]
    assert warning.message.fields.get("code") == "attribution_audit_units_many"
    assert "per_unit=False" in str(warning.message)
    spelling = _remedy(str(warning.message)).replace("[:200]])", "[:3]], k=1)")
    fixed = eval(spelling, _scope(model, table))  # noqa: S307
    assert len(fixed.indices) == 3
