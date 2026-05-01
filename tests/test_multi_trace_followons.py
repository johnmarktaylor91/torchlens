"""Phase 8 multi-trace follow-on tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

import torchlens as tl


class _ModuleModel(nn.Module):
    """Tiny module-nested model for bundle graph rendering."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return torch.sigmoid(self.fc(x))


def _bundle() -> tl.Bundle:
    """Return a deterministic two-member bundle.

    Returns
    -------
    tl.Bundle
        Bundle under test.
    """

    torch.manual_seed(20)
    x = torch.randn(2, 3)
    first = tl.log_forward_pass(_ModuleModel(), x, vis_opt="none", intervention_ready=True)
    second = tl.log_forward_pass(_ModuleModel(), x, vis_opt="none", intervention_ready=True)
    return tl.bundle({"first": first, "second": second})


def test_supergraph_accessor_and_module_type_labels() -> None:
    """Bundle exposes its supergraph with module type metadata."""

    bundle = _bundle()
    supergraph = bundle.supergraph

    assert supergraph.topological_order
    assert any(node.module_type == "Linear" for node in supergraph.nodes.values())


def test_show_bundle_graph_rolled_and_backward_modes(tmp_path: Path) -> None:
    """show_bundle_graph handles rolled and backward bundle render modes."""

    bundle = _bundle()
    rolled_source = tl.show_bundle_graph(
        bundle,
        vis_outpath=str(tmp_path / "rolled_bundle"),
        vis_mode="rolled",
        vis_save_only=True,
        vis_fileformat="svg",
    )
    backward_source = tl.show_bundle_graph(
        bundle,
        vis_outpath=str(tmp_path / "backward_bundle"),
        direction="backward",
        vis_save_only=True,
        vis_fileformat="svg",
    )

    assert rolled_source is not None
    assert "rolled" in rolled_source
    assert "(Linear)" in rolled_source
    assert backward_source is not None
    assert "backward" in backward_source


def test_bundle_graph_style_primitives_and_chrome_trace_diff(tmp_path: Path) -> None:
    """Per-node/edge styles feed rendering and chrome trace diff writes JSON."""

    bundle = _bundle()
    first_node = bundle.supergraph.topological_order[0]
    first_edge = next(iter(bundle.supergraph.edges))
    source = tl.show_bundle_graph(
        bundle,
        vis_outpath=str(tmp_path / "styled_bundle"),
        vis_node_overrides={first_node: {"fillcolor": "#DDEAF7"}},
        vis_edge_overrides={first_edge: {"color": "#0072B2"}},
        vis_save_only=True,
        vis_fileformat="svg",
    )

    trace_path = tl.export.chrome_trace_diff(bundle, tmp_path / "trace_diff.json")
    payload = json.loads(trace_path.read_text(encoding="utf-8"))

    assert source is not None
    assert "#DDEAF7" in source
    assert "#0072B2" in source
    assert payload["metadata"]["schema"] == "torchlens.chrome_trace_diff.v1"
    assert payload["traceEvents"]
