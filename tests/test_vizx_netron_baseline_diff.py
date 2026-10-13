"""Per-node diff facts in the Netron baseline slot (ruling U09.5).

``tl.export.netron(subject, path, baseline=reference)`` must carry, per
exported node, exactly the facts ``torchlens.differential.diff_report``
settles for the same pair -- checked here row by row against the diff's own
output, in the artifact's node metadata and in the sidecar.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.differential import diff_report

_PREFIX = "torchlens.diff."


def _mlp(seed: int) -> nn.Module:
    """A three-op MLP with seed-dependent weights."""

    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))


class _Block(nn.Module):
    """Linear then relu inside one module call."""

    def __init__(self) -> None:
        """Build one linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear then relu."""

        return torch.relu(self.fc(x))


class _Deep(nn.Module):
    """Two blocks and a residual add, so module granularity nests calls."""

    def __init__(self) -> None:
        """Build two blocks."""

        super().__init__()
        self.a = _Block()
        self.b = _Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two blocks plus a residual."""

        return self.b(self.a(x)) + x


def _deep(seed: int) -> nn.Module:
    """The nested model with seed-dependent weights."""

    torch.manual_seed(seed)
    return _Deep()


@pytest.fixture(scope="module")
def pair() -> Iterator[tuple[tl.Trace, tl.Trace]]:
    """Subject and reference captures of one MLP architecture, different weights."""

    x = torch.randn(2, 3, generator=torch.Generator().manual_seed(7))
    traces = (tl.trace(_mlp(0), x), tl.trace(_mlp(1), x))
    try:
        yield traces
    finally:
        for trace in traces:
            trace.cleanup()


@pytest.fixture(scope="module")
def deep_pair() -> Iterator[tuple[tl.Trace, tl.Trace]]:
    """Subject and reference captures of the nested model."""

    x = torch.randn(2, 4, generator=torch.Generator().manual_seed(7))
    traces = (tl.trace(_deep(0), x), tl.trace(_deep(1), x))
    try:
        yield traces
    finally:
        for trace in traces:
            trace.cleanup()


def _all_nodes(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every NodeProto row in the root graph and every function body, by name."""

    nodes = {node["name"]: node for node in payload["graph"].get("node", [])}
    for function in payload.get("functions", []):
        for node in function.get("node", []):
            nodes.setdefault(node["name"], node)
    return nodes


def _diff_props(node: dict[str, Any]) -> dict[str, str]:
    """The ``torchlens.diff.*`` metadata rows of one node."""

    return {
        row["key"]: row["value"]
        for row in node.get("metadataProps", [])
        if row["key"].startswith(_PREFIX)
    }


def _export(subject: tl.Trace, reference: Any, path: Path, **kwargs: Any) -> dict[str, Any]:
    """Export with a baseline and return the parsed artifact."""

    written = tl.export.netron(subject, path, baseline=reference, **kwargs)
    return dict(json.loads(written.read_text(encoding="utf-8")))


def _model_props(payload: dict[str, Any]) -> dict[str, str]:
    """The model-level metadataProps as a dict."""

    return {row["key"]: row["value"] for row in payload["metadataProps"]}


def test_op_nodes_carry_exactly_the_diff_rows(
    pair: tuple[tl.Trace, tl.Trace], tmp_path: Path
) -> None:
    """Every op node's diff facts equal the diff_report row for its op."""

    subject, reference = pair
    report = diff_report(subject, reference)
    payload = _export(subject, reference, tmp_path / "op.json", granularity="op")
    nodes = _all_nodes(payload)
    checked = 0
    for row in report.rows:
        if row.subject_label is None:
            continue
        name = row.subject_label.split(":", 1)[0]
        if name not in nodes:
            continue  # graph inputs/outputs are values, carried by the sidecar
        props = _diff_props(nodes[name])
        assert props[f"{_PREFIX}status"] == row.status
        assert float(props[f"{_PREFIX}max_abs_delta"]) == row.max_abs_delta
        assert props[f"{_PREFIX}reference_label"] == row.reference_label
        assert props[f"{_PREFIX}verdict"] == row.verdict
        checked += 1
    assert checked == 3, "linear, relu and linear each carry their diff row"
    assert any(_diff_props(node).get(f"{_PREFIX}status") == "changed" for node in nodes.values())


def test_baseline_slot_holds_the_diff_summary(
    pair: tuple[tl.Trace, tl.Trace], tmp_path: Path
) -> None:
    """The model-level slot carries the diff's own counts and direction."""

    subject, reference = pair
    report = diff_report(subject, reference)
    props = _model_props(_export(subject, reference, tmp_path / "m.json"))
    summary = json.loads(props["torchlens.baseline"])
    assert summary["schema"] == "torchlens.netron_baseline.v1"
    assert summary["source"] == "torchlens.differential.diff_report"
    assert summary["direction"] == "subject - reference"
    assert summary["row_counts"] == dict(sorted(report.counts.items()))
    assert summary["removed_count"] == len(report.rows_with_status("removed"))


def test_sidecar_carries_the_same_facts(pair: tuple[tl.Trace, tl.Trace], tmp_path: Path) -> None:
    """The default sidecar repeats each node's diff facts as node metadata."""

    subject, reference = pair
    path = tmp_path / "op.json"
    payload = _export(subject, reference, path, granularity="op")
    sidecar = json.loads((tmp_path / "op.attachment.json").read_text(encoding="utf-8"))
    rows = {
        (item["target"], item["name"]): item["value"]
        for item in sidecar["metadata"]
        if item["name"].startswith("diff.")
    }
    for name, node in _all_nodes(payload).items():
        for key, value in _diff_props(node).items():
            assert rows[(name, key.removeprefix("torchlens."))] == value
    report = diff_report(subject, reference)
    input_row = next(row for row in report.rows if row.subject_label == "input_1:1")
    assert rows[("input_1", "diff.status")] == input_row.status


def test_module_call_nodes_aggregate_their_ops(
    deep_pair: tuple[tl.Trace, tl.Trace], tmp_path: Path
) -> None:
    """A module-call node carries row counts and the largest delta over its ops."""

    subject, reference = deep_pair
    report = diff_report(subject, reference)
    by_label = {row.subject_label: row for row in report.rows}
    payload = _export(subject, reference, tmp_path / "mod.json", granularity="module")
    nodes = _all_nodes(payload)
    for address in ("a", "b"):
        call_ops = list(subject.module_calls[f"{address}:1"].ops)
        rows = [by_label[label] for label in call_ops if label in by_label]
        assert rows, f"module call {address} has settled rows"
        props = _diff_props(nodes[address])
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.status] = counts.get(row.status, 0) + 1
        assert json.loads(props[f"{_PREFIX}row_counts"]) == counts
        assert float(props[f"{_PREFIX}max_abs_delta"]) == max(
            row.max_abs_delta for row in rows if row.max_abs_delta is not None
        )
    # Leaf nodes inside the function bodies keep their single-row facts.
    leaf = by_label["linear_1_1:1"]
    assert _diff_props(nodes["linear_1_1"])[f"{_PREFIX}status"] == leaf.status


def test_diffed_artifact_passes_the_onnx_checker(
    deep_pair: tuple[tl.Trace, tl.Trace], tmp_path: Path
) -> None:
    """Node metadataProps keep the artifact strict-parseable and checker-green."""

    onnx = pytest.importorskip("onnx", reason="the netron contract needs onnx (test extra)")
    json_format = pytest.importorskip("google.protobuf.json_format")
    subject, reference = deep_pair
    for granularity in ("module", "op", "rolled"):
        path = tl.export.netron(
            subject, tmp_path / f"{granularity}.json", granularity=granularity, baseline=reference
        )
        model = json_format.Parse(path.read_text(encoding="utf-8"), onnx.ModelProto())
        onnx.checker.check_model(model, full_check=True)


def test_identical_weights_settle_to_zero(tmp_path: Path) -> None:
    """Two captures of identical weights carry ``zero`` on every value node."""

    x = torch.randn(2, 3, generator=torch.Generator().manual_seed(3))
    subject, reference = tl.trace(_mlp(5), x), tl.trace(_mlp(5), x)
    payload = _export(subject, reference, tmp_path / "z.json", granularity="op")
    statuses = {
        _diff_props(node)[f"{_PREFIX}status"]
        for node in _all_nodes(payload).values()
        if _diff_props(node)
    }
    assert statuses == {"zero"}


def test_string_baseline_stays_an_identifier(
    pair: tuple[tl.Trace, tl.Trace], tmp_path: Path
) -> None:
    """A string baseline is recorded verbatim with no per-node facts."""

    subject, _reference = pair
    payload = _export(subject, "run-41", tmp_path / "s.json", granularity="op")
    assert _model_props(payload)["torchlens.baseline"] == "run-41"
    assert not any(_diff_props(node) for node in _all_nodes(payload).values())


def test_no_baseline_carries_no_diff_facts(pair: tuple[tl.Trace, tl.Trace], tmp_path: Path) -> None:
    """Without a baseline the artifact has no slot row and no node diff facts."""

    subject, _reference = pair
    written = tl.export.netron(subject, tmp_path / "n.json", granularity="op")
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert "torchlens.baseline" not in _model_props(payload)
    assert not any(_diff_props(node) for node in _all_nodes(payload).values())


def test_non_trace_baseline_refuses_typed(pair: tuple[tl.Trace, tl.Trace], tmp_path: Path) -> None:
    """A baseline that is neither a Trace nor a string refuses before writing."""

    subject, _reference = pair
    with pytest.raises(tl.errors.ArgumentTypeError) as excinfo:
        tl.export.netron(subject, tmp_path / "x.json", baseline=42)
    assert excinfo.value.fields["code"] == "export_expected_trace"
    assert excinfo.value.fields["argument"] == "baseline"
    assert not (tmp_path / "x.json").exists()


def test_self_baseline_refuses(pair: tuple[tl.Trace, tl.Trace], tmp_path: Path) -> None:
    """Diffing a capture against itself refuses through diff_report's own guard."""

    subject, _reference = pair
    with pytest.raises(tl.errors.InvalidArgumentError) as excinfo:
        tl.export.netron(subject, tmp_path / "self.json", baseline=subject)
    assert excinfo.value.fields["code"] == "differential_subject_is_reference"
