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
