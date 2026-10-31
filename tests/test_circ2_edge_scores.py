"""Edge attribution patching (``torchlens.mechinterp.edge_scores``): EAP and EAP-IG.

Pre-registered bar CIRC-2b (deterministic nonlinear fixture, float64): the
fixture of the bars file is built as an ``nn.Module`` and scored through the
same entry point the GPT-2 runs use. EAP-IG at ``m`` in {1, 2, 4, 5, 8, 32,
128} equals the closed-form left Riemann sums within 1e-12 per edge; at
``m = 128`` every edge is within 0.01 of the exact integral and the scores
into the receiver sum to ``M(corrupt) - M(clean)`` within 0.01; EAP equals
the one-point closed form within 1e-12.

Fixture mapping onto the component graph: token 0 embeds ``e_clean`` and
token 1 ``e_corrupt`` (the ``input`` node); the senders ``h1`` and ``h2`` are
two bias-free one-layer MLPs that both read the embedding through their own
identity pre-norm (graph names ``m0`` and ``m1``, matched by module
address); the receiver ``R`` is the state entering the identity final norm
before the unembedding ``V`` (graph name ``logits``); the metric is
``M = sum_j w_j tanh((V r)_j)`` at the last position. The identity norms
carry an RMSNorm class name so the graph builder classifies them as the
pre-norms they stand in for.

A tiny random GPT-2 checks the same definitions against an independent
plain-PyTorch implementation (``tests/support/eap_reference.py``) run in a
fresh interpreter that never imports torchlens; the real GPT-2 small bar (CIRC-2a) is in
``tests/real_model/r1/test_circ2_edge_scores_real.py``.
"""

from __future__ import annotations

import pytest
import torch
from support.eap_fixture import CLEAN, CORRUPT, Fixture, fixture_metric

import torchlens as tl
import torchlens.mechinterp

pytestmark = pytest.mark.heavy

#: Closed-form values from the pinned fixture script (``eapig-fixture-values.json``).
METRIC_CLEAN = 2.2684758023857734
METRIC_CORRUPT = 0.8274514781171568
EXACT = {
    "input->R": -0.17654160504493047,
    "h1->R": 0.0684110296513677,
    "h2->R": -1.3328937488750536,
    "input->h1": 0.0684110296513677,
    "input->h2": -1.3328937488750536,
}
EAP_ONE_POINT = {
    "input->R": 0.8871418323909614,
    "h1->R": 1.7788347706825323,
    "h2->R": -1.79248808838436,
    "input->h1": 1.7788347706825323,
    "input->h2": -1.79248808838436,
}
LEFT_RIEMANN = {
    1: {
        "input->R": 0.19877670504511163,
        "h1->R": 0.44007297087867103,
        "h2->R": -0.5676316532440144,
        "input->h1": 0.44007297087867103,
        "input->h2": -0.5676316532440144,
    },
    2: {
        "input->R": -0.428165799398432,
        "h1->R": -0.37252594472694794,
        "h2->R": -1.0788910174828006,
        "input->h1": -0.37252594472694794,
        "input->h2": -1.0788910174828006,
    },
    4: {
        "input->R": -0.23837998895919904,
        "h1->R": -0.06326866895213248,
        "h2->R": -1.1772052579466643,
        "input->h1": -0.06326866895213248,
        "input->h2": -1.1772052579466643,
    },
    5: {
        "input->R": -0.24633008190931596,
        "h1->R": -0.06585652762181762,
        "h2->R": -1.2145543809686252,
        "input->h1": -0.06585652762181762,
        "input->h2": -1.2145543809686252,
    },
    8: {
        "input->R": -0.217771448974816,
        "h1->R": -0.012358813578975905,
        "h2->R": -1.2571934395329925,
        "input->h1": -0.012358813578975905,
        "input->h2": -1.2571934395329925,
    },
}


def _fixture_scores(**kwargs) -> dict[str, float]:
    """Score the fixture and rename edges into the bar's vocabulary by module address."""

    model = Fixture().eval()
    table = torchlens.mechinterp.edge_scores(
        model, CLEAN, corrupted_input=CORRUPT, metric=fixture_metric, **kwargs
    )
    graph = torchlens.mechinterp.component_graph(tl.trace(model, CLEAN))
    names = {node.name: node.module_address or "input" for node in graph.nodes.values()}
    names = {key: {"embed": "input"}.get(value, value) for key, value in names.items()}
    names["logits"] = "R"
    out = {}
    for (_target, kind, address), row in table.items():
        assert kind == "EDGE" and row.status == "ok"
        sender, receiver = address[0].split("->")
        out[f"{names[sender]}->{names[receiver]}"] = row.score
    assert set(out) == set(EXACT)
    return out


def test_fixture_eap_equals_the_one_point_closed_form() -> None:
    """EAP (gradient at the clean input) matches ``eap_one_point_clean`` within 1e-12."""

    scores = _fixture_scores(method="eap")
    for edge, value in EAP_ONE_POINT.items():
        assert abs(scores[edge] - value) <= 1e-12, edge


@pytest.mark.parametrize("steps", sorted(LEFT_RIEMANN))
def test_fixture_eap_ig_equals_the_left_riemann_sum(steps: int) -> None:
    """EAP-IG at ``m`` points equals the closed-form left Riemann sum within 1e-12."""

    scores = _fixture_scores(steps=steps)
    for edge, value in LEFT_RIEMANN[steps].items():
        assert abs(scores[edge] - value) <= 1e-12, (steps, edge)


@pytest.mark.slow
def test_fixture_eap_ig_converges_and_completes_at_many_steps() -> None:
    """m = 32 and 128 match the closed forms; m = 128 converges and is complete within 0.01."""

    m32 = {"input->R": -0.18718271079633553}
    scores32 = _fixture_scores(steps=32)
    assert abs(scores32["input->R"] - m32["input->R"]) <= 1e-12
    scores = _fixture_scores(steps=128)
    for edge, value in EXACT.items():
        assert abs(scores[edge] - value) <= 0.01, edge
    into_r = sum(scores[edge] for edge in ("input->R", "h1->R", "h2->R"))
    assert abs(into_r - (METRIC_CORRUPT - METRIC_CLEAN)) <= 0.01


def _tiny_gpt2():
    """A random float64 eager GPT-2 (2 layers, 2 heads) and clean/corrupt token rows."""

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=64, n_positions=32)
    model = transformers.GPT2LMHeadModel(config).to(torch.float64).eval()
    model.set_attn_implementation("eager")
    return model, torch.randint(0, 64, (3, 6)), torch.randint(0, 64, (3, 6))


def _logit_difference(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, -1, 1] - logits[:, -1, 2]


def test_tiny_gpt2_matches_the_independent_reference(tmp_path) -> None:
    """EAP and EAP-IG (m = 1, 5) equal the plain-PyTorch treeified reference per edge.

    The reference runs in a fresh interpreter that never imports torchlens.
    """

    from support.eap_reference import edge_names, scores_in_subprocess

    model, clean, corrupt = _tiny_gpt2()
    cases = [("eap", 1), ("eap_ig", 1), ("eap_ig", 5)]
    want = scores_in_subprocess(
        model,
        clean,
        corrupt,
        last=torch.full((3,), 5),
        answer=torch.full((3,), 1),
        distractor=torch.full((3,), 2),
        cases=cases,
        workdir=tmp_path,
    )
    assert len(want[cases[0]]) == len(edge_names(2, 2))
    for method, steps in cases:
        extra = {} if method == "eap" else {"steps": steps}
        table = torchlens.mechinterp.edge_scores(
            model, clean, corrupted_input=corrupt, metric=_logit_difference, method=method, **extra
        )
        got = {address[0]: row.score for (_t, _k, address), row in table.items()}
        for edge, value in want[(method, steps)].items():
            assert abs(got[edge] - value) <= 1e-8 * abs(value) + 1e-10, (method, steps, edge)
        assert all(got[e] == 0.0 for e in got if e.startswith("b")), "bias edges score zero"


def test_edge_table_is_scalar_saveable_and_documents_its_reduction(tmp_path) -> None:
    """EDGE rows: address ``(edge name,)``, provenance sample count, scalar save and load."""

    table = torchlens.mechinterp.edge_scores(
        Fixture().eval(),
        torch.tensor([[0], [0]]),
        corrupted_input=torch.tensor([[1], [1]]),
        metric=fixture_metric,
        method="eap",
        example_reduction="sum",
    )
    rows = table.rows()
    assert {row.kind for row in rows} == {"EDGE"}
    assert all(len(row.address) == 1 and "->" in row.address[0] for row in rows)
    assert table.provenance.sample_count == 2
    assert {row.reduction for row in rows} == {"sum_positions_sum_examples"}
    one = _fixture_scores(method="eap")
    by_name = {row.address[0]: row.score for row in rows}
    assert by_name["input->logits"] == pytest.approx(2 * one["input->R"], abs=1e-12)
    loaded = tl.attribution.onebackward.load_read_table(table.save(tmp_path / "edges.json"))
    assert {row.address[0]: row.score for row in loaded.rows()} == by_name


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"method": "acdc"}, "mi_edge_scores_setting_invalid"),
        ({"method": "eap", "steps": 3}, "mi_edge_scores_setting_invalid"),
        ({"steps": 0}, "mi_edge_scores_setting_invalid"),
        ({"example_reduction": "median"}, "mi_edge_scores_setting_invalid"),
        ({"metric": lambda logits: logits[:, -1]}, "mi_edge_scores_metric_invalid"),
        ({"corrupted": torch.tensor([[1, 1]])}, "mi_edge_scores_inputs_mismatched"),
    ],
)
def test_bad_settings_refuse_typed(kwargs, code) -> None:
    """Unknown method, steps with EAP, bad steps or reduction, a non-scalar metric, shapes."""

    call = {"corrupted": CORRUPT, "metric": fixture_metric, **kwargs}
    with pytest.raises(torchlens.errors.MechInterpError) as caught:
        torchlens.mechinterp.edge_scores(Fixture().eval(), CLEAN, **call)
    assert caught.value.fields["code"] == code


def test_shared_norm_and_unshiftable_input_refuse_typed() -> None:
    """A norm feeding two receiver modules, and an embedding write that is no module output."""

    from types import SimpleNamespace

    from torchlens.mechinterp import _eap

    receivers = {
        "a0.h0<q>": SimpleNamespace(norm_address="ln", module_address="attn"),
        "m0": SimpleNamespace(norm_address="ln", module_address="mlp"),
    }
    with pytest.raises(torchlens.errors.MechInterpError) as caught:
        _eap._refuse_shared_norms(SimpleNamespace(receivers=receivers))
    assert caught.value.fields["code"] == "mi_edge_scores_shared_norm"
    with pytest.raises(torchlens.errors.MechInterpError) as caught:
        _eap._input_write_module(None, SimpleNamespace(_input_writes=()))
    assert caught.value.fields["code"] == "mi_edge_scores_input_unresolvable"
