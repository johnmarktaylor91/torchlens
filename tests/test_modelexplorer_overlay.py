"""Node-data overlay writer tests (memo D8, B4).

The writer is a thin serializer over the existing draw() overlay vocabulary:
missing, nonfinite, and rolled-varying values are omitted and counted by
reason (never a zero heatmap); value-derived providers refuse typed on
public profiles and value-free captures; the callable/field escape hatches
surface their own failures typed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.export._model_explorer._overlay import build_node_data

pytestmark = pytest.mark.smoke


class _Net(nn.Module):
    """Small chain with one nonfinite-producing branch."""

    def __init__(self) -> None:
        """Two layers."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Plain chain."""

        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture(scope="module")
def net_log() -> Any:
    """Trace the toy once per module."""

    log = tl.trace(_Net().eval(), torch.randn(2, 4))
    try:
        yield log
    finally:
        log.cleanup()


@pytest.fixture(scope="module")
def net_payload(net_log: Any) -> dict[str, Any]:
    """Export the toy once per module."""

    return tl.export.to_model_explorer_dict(net_log)


def test_standard_providers_write_sidecars(net_log: Any, tmp_path: Path) -> None:
    """overlays=('standard',) writes the four-provider packet + manifest."""

    path = tl.export.model_explorer(net_log, tmp_path / "m.json", overlays=("standard",))
    manifest = json.loads((tmp_path / "m.manifest.json").read_text(encoding="utf-8"))
    providers = [row["provider"] for row in manifest["node_data"]]
    assert len(providers) == 4
    assert all("per op" in provider for provider in providers)
    for row in manifest["node_data"]:
        sidecar = json.loads((tmp_path / row["file"]).read_text(encoding="utf-8"))
        assert sidecar, "provider files must carry at least one graph's data"
        for graph_data in sidecar.values():
            assert graph_data["results"]
            assert graph_data["gradient"][0]["stop"] == 0
    assert path.exists()


def test_values_join_by_node_id(net_log: Any, net_payload: dict[str, Any]) -> None:
    """Provider results key on the payload's node ids."""

    graphs_data, coverage = build_node_data(net_log, net_payload, "bytes")
    graph_id = next(iter(graphs_data))
    node_ids = {node["id"] for node in net_payload["graphs"][0]["nodes"]}
    assert set(graphs_data[graph_id]["results"]) <= node_ids
    assert coverage["resolved"] > 0
    assert coverage["omitted_nonfinite"] == 0


def test_magnitude_requires_retained_values() -> None:
    """A value-free capture refuses value-derived providers, naming the
    missing retention (composition row 12: structure-only x every surface)."""

    model = _Net().eval()
    log = tl.trace(
        model,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(structure_only=True),
    )
    payload = tl.export.to_model_explorer_dict(log)
    with pytest.raises(Exception) as excinfo:
        build_node_data(log, payload, "magnitude")
    assert excinfo.value.fields["code"] == "model_explorer_value_not_retained"


def test_public_profile_refuses_value_derived(net_log: Any, net_payload: dict[str, Any]) -> None:
    """Public profiles drop value-derived providers by construction (D14)."""

    with pytest.raises(Exception) as excinfo:
        build_node_data(net_log, net_payload, "grad_norm", privacy_profile="public")
    assert excinfo.value.fields["code"] == "model_explorer_overlay_public_conflict"


def test_callable_escape_hatch_and_typed_error(net_log: Any, net_payload: dict[str, Any]) -> None:
    """Callables serialize per node; their exceptions surface typed."""

    def fan_in(node: Any) -> int:
        """Count recorded parents."""

        return len(node.parents or ())

    graphs_data, coverage = build_node_data(net_log, net_payload, fan_in)
    assert coverage["resolved"] > 0
    graph_id = next(iter(graphs_data))
    assert all(isinstance(row["value"], int) for row in graphs_data[graph_id]["results"].values())

    def boom(node: Any) -> int:
        """Always raise."""

        raise RuntimeError("boom")

    with pytest.raises(Exception) as excinfo:
        build_node_data(net_log, net_payload, boom)
    assert excinfo.value.fields["code"] == "model_explorer_overlay_callable_error"


def test_field_source_and_nonnumeric_refusal(net_log: Any, net_payload: dict[str, Any]) -> None:
    """field:<name> resolves record fields; non-numeric values refuse."""

    graphs_data, coverage = build_node_data(net_log, net_payload, "field:num_params")
    assert coverage["resolved"] > 0
    with pytest.raises(Exception) as excinfo:
        build_node_data(net_log, net_payload, "field:layer_type")
    assert excinfo.value.fields["code"] == "model_explorer_overlay_value_invalid"


def test_rolled_varying_values_are_omitted_and_counted() -> None:
    """Per-pass-varying values on rolled nodes are omitted, never averaged
    (composition row 3)."""

    class _Rec(nn.Module):
        """Reused cell whose activations vary across passes."""

        def __init__(self) -> None:
            """One cell."""

            super().__init__()
            self.cell = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Iterate."""

            h = x
            for _ in range(3):
                h = torch.tanh(self.cell(h))
            return h

    log = tl.trace(_Rec().eval(), torch.randn(2, 4))
    payload = tl.export.to_model_explorer_dict(log)
    graphs_data, coverage = build_node_data(log, payload, "magnitude")
    rolled_data = graphs_data.get("01-rolled", {"results": {}})
    rolled_multi_pass_ids = {
        node["id"]
        for node in payload["graphs"][1]["nodes"]
        if any(attr["key"] == "passes" for attr in node["attrs"])
    }
    assert not (set(rolled_data["results"]) & rolled_multi_pass_ids), (
        "a rolled multi-pass node must never carry a single-pass magnitude"
    )
    assert coverage["omitted_rolled_varying"] > 0
