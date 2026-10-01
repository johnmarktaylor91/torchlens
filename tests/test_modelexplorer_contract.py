"""B0 vendor contract harness for the Model Explorer export family (F15).

Three of the four data-loss bugs the modelexplorer panel found are invisible
to a dataclass parse and visible only to Model Explorer's REAL graph
processor, so this file pins the contract at three levels: byte-pinned
vendor assets, a strict parse against the pinned pip package's own
``graph_builder``/``node_data_builder`` dataclasses, and the EXECUTED
``dist/worker.js`` oracle asserting NO SILENT NODE LOSS. The semantic
validator's red-team cases prove the tripwires actually fire.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
from test_modelexplorer_assets import harness
from test_modelexplorer_assets.vendor_loader import (
    load_vendor_schema_modules,
    strict_parse,
)
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke

ASSETS_DIR = Path(__file__).parent / "test_modelexplorer_assets"

requires_node = pytest.mark.skipif(
    not harness.node_available(), reason="worker oracle needs a Node runtime on PATH"
)

_VENDOR = load_vendor_schema_modules()
requires_vendor_schema = pytest.mark.skipif(
    _VENDOR is None,
    reason="pinned ai-edge-model-explorer schema modules not installed",
)


class _ReuseNet(nn.Module):
    """Module reuse + duplicated operand toy: the id/port acid case."""

    def __init__(self) -> None:
        """Build one reused linear block."""

        super().__init__()
        self.blk = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the same block twice and cat both results."""

        y = self.blk(x)
        z = self.blk(y)
        return torch.cat([y, z], dim=1)


@pytest.fixture(scope="module")
def reuse_payload() -> Any:
    """Export the reuse toy once for the whole module."""

    log = tl.trace(_ReuseNet().eval(), torch.randn(2, 4))
    try:
        yield tl.export.to_model_explorer_dict(log)
    finally:
        log.cleanup()


def test_vendor_assets_are_byte_pinned() -> None:
    """worker.js / d.ts must match their recorded sha256 pins exactly."""

    pins = {}
    for line in (ASSETS_DIR / "checksums.sha256").read_text().splitlines():
        digest, _, name = line.strip().partition("  ")
        pins[name] = digest
    for name, digest in pins.items():
        actual = hashlib.sha256((ASSETS_DIR / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} drifted from its byte pin"


@requires_vendor_schema
def test_payload_strict_parses_against_vendor_dataclasses(
    reuse_payload: dict[str, Any],
) -> None:
    """Every emitted graph strict-parses as the vendor's own Graph dataclass."""

    graph_builder, _ = _VENDOR
    for graph in reuse_payload["graphs"]:
        parsed = strict_parse(graph_builder.Graph, graph)
        assert parsed.nodes
    collection = strict_parse(graph_builder.GraphCollection, reuse_payload)
    assert collection.graphSorting == "name_asc"


@requires_vendor_schema
def test_node_data_strict_parses_against_vendor_dataclasses() -> None:
    """Overlay sidecar payloads strict-parse as vendor GraphNodeData."""

    _, node_data_builder = _VENDOR
    log = tl.trace(_ReuseNet().eval(), torch.randn(2, 4))
    payload = tl.export.to_model_explorer_dict(log)
    from torchlens.export._model_explorer._overlay import build_node_data

    graphs_data, coverage = build_node_data(log, payload, "bytes")
    assert coverage["resolved"] > 0
    for graph_data in graphs_data.values():
        parsed = strict_parse(node_data_builder.GraphNodeData, graph_data)
        assert parsed.results


def test_outputs_metadata_uses_the_shape_key(reuse_payload: dict[str, Any]) -> None:
    """Edge-shape labels resolve through metadata key ``shape`` (memo D5).

    The vendor bundle's "Tensor shape" edge-label mode reads key ``shape``
    -- writing ``tensor_shape`` renders nothing. Pinned structurally here so
    the contract never rests on a source grep again.
    """

    keys = set()
    for graph in reuse_payload["graphs"]:
        for node in graph["nodes"]:
            for item in node.get("outputsMetadata", []):
                keys.update(attr["key"] for attr in item["attrs"])
    assert "shape" in keys
    assert "tensor_shape" not in keys


@requires_node
def test_worker_oracle_processes_toy_without_loss(reuse_payload: dict[str, Any]) -> None:
    """The real pinned worker keeps every declared node and edge."""

    rows = harness.run_worker_oracle(reuse_payload)
    harness.assert_no_silent_node_loss(reuse_payload, rows)


@requires_node
def test_worker_silently_drops_duplicate_ids() -> None:
    """RED-team: the vendor worker drops a duplicate-id node WITHOUT error.

    This is the measured failure mode the whole id rule exists to prevent;
    if a vendor upgrade ever changes it, this test flags the contract-profile
    update explicitly.
    """

    payload = {
        "label": "dup",
        "graphs": [
            {
                "id": "g",
                "nodes": [
                    {"id": "a", "label": "x", "namespace": ""},
                    {"id": "a", "label": "y", "namespace": ""},
                    {
                        "id": "b",
                        "label": "z",
                        "namespace": "",
                        "incomingEdges": [{"sourceNodeId": "a"}],
                    },
                ],
            }
        ],
    }
    rows = harness.run_worker_oracle(payload)
    stats = rows[0]["stats"]
    assert rows[0]["err"] is None
    assert stats["opNodes"] == 2, "vendor now keeps duplicate ids: contract change"


def test_semantic_validator_catches_planted_defects() -> None:
    """Every validator tripwire fires on a deliberately broken payload."""

    payload = {
        "label": "bad",
        "graphs": [
            {
                "id": "g",
                "nodes": [
                    {"id": "a", "label": "x", "namespace": "m"},
                    {"id": "a", "label": "x", "namespace": "m"},
                    {
                        "id": "c",
                        "label": "y",
                        "namespace": "",
                        "incomingEdges": [{"sourceNodeId": "ghost"}],
                        "inputsMetadata": [{"id": "9", "attrs": []}],
                    },
                ],
                "groupNodeAttributes": {"": {}},
            },
            {"id": "g", "nodes": [], "groupNodeAttributes": {"": {}}},
        ],
    }
    report = tl.export.validate_model_explorer_payload(payload)
    codes = {row["code"] for row in report.failures}
    assert not report.ok
    assert {
        "model_explorer_duplicate_node_id",
        "model_explorer_duplicate_graph_id",
        "model_explorer_edge_unresolved",
        "model_explorer_slot_unreferenced",
        "model_explorer_group_row_missing_namespace",
    } <= codes
    with pytest.raises(Exception) as excinfo:
        tl.export.validate_model_explorer_payload(payload, strict=True)
    assert excinfo.value.fields["code"] == "model_explorer_payload_invalid"


def test_validator_passes_the_shipped_payload(reuse_payload: dict[str, Any]) -> None:
    """The exporter's own output satisfies its semantic floor."""

    report = tl.export.validate_model_explorer_payload(reuse_payload)
    assert report.ok, report.failures
