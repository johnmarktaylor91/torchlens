"""The capture benchmark harness: result format, hooks census rules, fixtures and panel.

These pin the harness's own logic (``benchmarks/capture/``), not TorchLens speed.
One test traces a four-operation toy model to pin the operation fields the census
reads, so a change in what ``is_module_output`` or ``input_to_module_calls`` means
breaks here rather than silently shifting the census.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from benchmarks.capture import census, schema


def _op(name: str, **flags: object) -> SimpleNamespace:
    base = {
        "func_name": name,
        "is_input": False,
        "is_output": False,
        "is_buffer": False,
        "is_module_output": False,
        "is_output_parent": False,
        "input_to_module_calls": [],
    }
    base.update(flags)
    return SimpleNamespace(**base)


def test_document_round_trip_is_valid() -> None:
    doc = schema.new_document("A", {"cases": ["single_module_output"]})
    doc["cells"].append(schema.make_cell("A", "single_module_output", "hooks", "mlp", "ok"))
    doc["cells"].append(
        schema.make_cell(
            "A", "single_module_output", "nnsight", "mlp", "unsupported", reason="absent"
        )
    )
    assert schema.validate_document(doc) == []
    assert doc["environment"]["hostname"] == "benchmark-host" or doc["environment"]["hostname"]


def test_document_rejects_silent_gaps() -> None:
    doc = schema.new_document("B", {})
    doc["cells"].append(schema.make_cell("B", "validate", "torchlens", "mlp", "refused"))
    doc["cells"].append(schema.make_cell("B", "save_load", "torchlens", "mlp", "ok"))
    doc["cells"].append(schema.make_cell("B", "draw", "torchlens", "mlp", "bogus", reason="x"))
    problems = schema.validate_document(doc)
    assert any("refused without a reason" in p for p in problems)
    assert any("without 'keeps'" in p for p in problems)
    assert any("unknown status" in p for p in problems)
    with pytest.raises(ValueError):
        schema.new_document("Z", {})


def test_reachability_rules() -> None:
    ops = [
        _op("input", is_input=True),
        _op("linear", is_module_output=True),
        _op("relu"),
        _op("mul", input_to_module_calls=[("lin2", 1)]),
        _op("linear", is_module_output=True, is_output_parent=True),
        _op("lstm"),
        _op("output", is_output=True),
    ]
    counts = census.census_counts(ops)
    assert counts["computations"] == 5
    assert counts["unreachable_output_hooks"] == 3
    assert counts["unreachable_any_hook"] == 2
    assert counts["missed_funcs"] == {"relu": 1, "lstm": 1}
    assert counts["fused_boundaries"]["lstm"][0] == 1
    assert counts["share_unreachable_any_hook"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("mods", "funcs", "family"),
    [
        (["MultiheadAttention", "Linear"], ["linear"], "attention"),
        ([], ["softmax", "bmm", "conv2d"], "attention"),
        (["LSTM", "Conv1d"], ["conv1d", "lstm"], "recurrent"),
        (["Conv2d"], ["conv2d", "linear"], "convolutional"),
        (["Linear"], ["linear"], "dense"),
        ([], ["sigmoid", "__add__"], "other"),
    ],
)
def test_family_priority(mods: list[str], funcs: list[str], family: str) -> None:
    assert census.classify_family(mods, funcs) == family


def test_summary_keeps_failures_visible() -> None:
    ok_counts = census.census_counts([_op("linear", is_module_output=True), _op("relu")])
    cells = [
        schema.make_cell(
            "census",
            "hooks_miss",
            "torchlens",
            "a",
            "ok",
            family="dense",
            counts=ok_counts,
            validation={"passed": True},
        ),
        schema.make_cell(
            "census",
            "hooks_miss",
            "torchlens",
            "b",
            "ok",
            family="dense",
            counts=ok_counts,
            validation={"passed": False, "failure": "x"},
        ),
        schema.make_cell("census", "hooks_miss", "torchlens", "c", "error", reason="boom"),
    ]
    summary = census.summarize(cells)
    assert summary["dense"]["replay_failed"] == ["b"]
    assert summary["dense"]["replay_pass_rate"] == 0.5
    assert summary["not_captured"]["not_captured"] == ["c"]


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.lin1(x))
        return self.lin2(h * 2)


def test_census_fields_on_a_real_trace() -> None:
    import torchlens as tl

    torch.manual_seed(0)
    trace = tl.trace(_Toy(), torch.randn(3, 4))
    counts = census.census_counts(trace.ops)
    assert counts["computations"] == 4
    assert counts["missed_funcs"] == {"relu": 1}
    assert counts["unreachable_output_hooks"] == 2


def test_fixture_names_round_trip_and_build() -> None:
    from benchmarks.capture import fixtures

    names = fixtures.fixture_matrix(smoke=True)
    assert len(names) == len(fixtures.KINDS) * len(fixtures.SMOKE_OP_COUNTS)
    for kind in fixtures.KINDS:
        name = fixtures.fixture_name(kind, 100, "small")
        assert fixtures.parse_fixture_name(name) == (kind, 100, "small")
        model, x, meta = fixtures.build_fixture(name)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape
        assert meta["site"] in dict(model.named_modules())
        assert set(meta["sites"]) <= set(dict(model.named_modules()))


def test_panel_generation_cells_name_entry_points() -> None:
    from benchmarks.capture import panel

    rows = panel.generation_cells(panel.PANEL_BY_NAME["gpt2"])
    assert {r["steps"] for r in rows} == set(panel.GENERATION_STEPS)
    refused = [r for r in rows if r["status"] == "refused"]
    assert refused and all(r["entry_point"] == "full_trace" and r["reason"] for r in refused)
    assert all(r["steps"] > panel.FULL_TRACE_STEP_LIMIT for r in refused)
    assert panel.generation_cells(panel.PANEL_BY_NAME["resnet18"]) == []
    assert {"resnet18", "lstm-tagger", "encoder-small", "dynamic-router"} == set(
        panel.smoke_panel()
    )


def test_matched_contract_compare_and_hooks_reference() -> None:
    from benchmarks.capture import arms, fixtures

    model, x, meta = fixtures.build_fixture("fx-pointwise-100-small")
    subject = arms.prepare_subject("fx-pointwise-100-small", model, x, meta)
    hooks = arms.HooksArm(subject)
    ref = hooks.run("replacement_edit")
    assert arms.compare(ref, ref)["bit_identical"] is True
    assert arms.compare(ref + 1, ref)["matched"] is False
    steered = hooks.run("repeated_steering")
    assert len(steered) == len(arms.STEER_MAGNITUDES)
    assert hooks.run("same_site_gradient").shape == subject.site_shape
    feats = hooks.run("ordered_dataset_features")
    assert feats.shape[0] == arms.DATASET_SIZE


def test_driver_plan_keeps_unsupported_cells_visible() -> None:
    from benchmarks.capture import driver

    cells = driver.plan(smoke=True)
    a_cells = [c for c in cells if c[0] == "A"]
    static = [driver._static_unsupported(c) for c in a_cells]
    unsupported = [row for row in static if row is not None]
    assert unsupported and all(row["reason"] for row in unsupported)
    assert any(c[0] == "B" and c[1] == "bundle_compare" for c in cells)
    assert any(c[0] == "fixtures" for c in cells)
    aa = driver.aa_noise([{"status": "ok", "timing": {"median_ms": 10.0}}] * 2)
    rows = [
        {"cell_id": "A/x/hooks/s", "status": "ok", "timing": {"median_ms": m}} for m in (10.0, 20.0)
    ]
    assert driver.group_cells(rows, aa, 0.1)[0]["noisy"] is True
