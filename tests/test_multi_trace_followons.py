"""Phase 8 multi-trace follow-on tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.intervention._topology.topology import compare_topology


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


class _ReorderedReluSigmoid(nn.Module):
    """Apply ReLU then sigmoid."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``sigmoid(relu(x))``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Reordered output.
        """

        return torch.sigmoid(torch.relu(x))


class _ReorderedSigmoidRelu(nn.Module):
    """Apply sigmoid then ReLU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``relu(sigmoid(x))``.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Reordered output.
        """

        return torch.relu(torch.sigmoid(x))


class _BufferParamModel(nn.Module):
    """Small model exposing a live parameter and registered buffer."""

    def __init__(self, *, scale: float, weight: float) -> None:
        """Initialize the model with deterministic state.

        Parameters
        ----------
        scale:
            Buffer fill value.
        weight:
            Linear-weight fill value.
        """

        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)
        self.register_buffer("scale", torch.full((3,), scale))
        with torch.no_grad():
            self.linear.weight.fill_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Linear output shifted by the registered buffer.
        """

        return self.linear(x) + self.scale


def _bundle() -> tl.Bundle:
    """Return a deterministic two-member bundle.

    Returns
    -------
    tl.Bundle
        Bundle under test.
    """

    torch.manual_seed(20)
    x = torch.randn(2, 3)
    first = tl.trace(_ModuleModel(), x, capture=tl.options.CaptureOptions(intervention_ready=True))
    second = tl.trace(_ModuleModel(), x, capture=tl.options.CaptureOptions(intervention_ready=True))
    return tl.bundle({"first": first, "second": second})


def test_supergraph_accessor_and_module_type_labels() -> None:
    """Bundle exposes its supergraph with module type metadata."""

    bundle = _bundle()
    supergraph = bundle.supergraph

    assert supergraph.topological_order
    assert any(node.module_type == "Linear" for node in supergraph.nodes.values())


def test_compare_topology_rejects_reordered_graphs() -> None:
    """Reordered unique-fingerprint graphs are not reported identical."""

    x = torch.randn(2, 3)
    first = tl.trace(
        _ReorderedReluSigmoid(), x, capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    second = tl.trace(
        _ReorderedSigmoidRelu(), x, capture=tl.options.CaptureOptions(intervention_ready=True)
    )

    diff = compare_topology(first, second)

    assert diff.is_identical is False
    assert diff.unmatched_a
    assert diff.unmatched_b


def test_supergraph_topological_order_respects_edges_after_reordered_bundle() -> None:
    """Merged bundles with reordered graphs stay acyclic and edge-consistent."""

    x = torch.randn(2, 3)
    bundle = tl.bundle(
        {
            "first": tl.trace(
                _ReorderedReluSigmoid(),
                x,
                capture=tl.options.CaptureOptions(intervention_ready=True),
            ),
            "second": tl.trace(
                _ReorderedSigmoidRelu(),
                x,
                capture=tl.options.CaptureOptions(intervention_ready=True),
            ),
        }
    )

    supergraph = bundle.supergraph
    positions = {name: idx for idx, name in enumerate(supergraph.topological_order)}

    assert len(positions) == len(supergraph.nodes)
    assert all(positions[parent] < positions[child] for parent, child in supergraph.edges)


def test_super_buffer_views_use_public_live_values_and_param_reads_refuse() -> None:
    """Buffer super views expose live values; cross-member param reads refuse.

    The historical assertion here pinned the checkpoint live-ref defect
    (A-CKPT): ``weight_norm_diff`` served live-handle bytes as per-member
    capture values. Cross-member parameter value reads now refuse typed with
    ``checkpoint_series_live_params`` -- the claim is unprovable from live
    handles, regardless of whether the members come from distinct model
    objects (claim-keyed, never identity-keyed).
    """

    import pytest

    from torchlens.errors import CheckpointSeriesLiveParamsError

    x = torch.randn(2, 3)
    model_a = _BufferParamModel(scale=1.0, weight=1.0)
    model_b = _BufferParamModel(scale=2.0, weight=3.0)
    trace_a = tl.trace(model_a, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    trace_b = tl.trace(model_b, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    bundle = tl.bundle({"a": trace_a, "b": trace_b})

    buffer_out = bundle.buffers["scale"].out
    assert torch.equal(buffer_out, torch.cat([model_a.scale, model_b.scale], dim=0))

    with pytest.raises(CheckpointSeriesLiveParamsError) as excinfo:
        _ = bundle.params["linear.weight"].weight_norm_diff
    assert excinfo.value.fields["code"] == "checkpoint_series_live_params"


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


def test_bundle_backward_regression_path(tmp_path: Path) -> None:
    """Bundle backward rendering emits per-member backward subgraphs."""

    bundle = _bundle()
    source = tl.show_bundle_graph(
        bundle,
        vis_outpath=str(tmp_path / "backward_bundle_regression"),
        direction="backward",
        vis_save_only=True,
        vis_fileformat="svg",
    )

    assert source is not None
    assert "cluster_backward_first" in source
    assert "cluster_backward_second" in source
    assert "backward" in source


def test_bundle_backward_no_unified_supergraph(tmp_path: Path) -> None:
    """Bundle backward rendering stays per-member in P6."""

    bundle = _bundle()
    source = tl.show_bundle_graph(
        bundle,
        vis_outpath=str(tmp_path / "backward_bundle_no_supergraph"),
        direction="backward",
        vis_save_only=True,
        vis_fileformat="svg",
    )

    assert source is not None
    assert "cluster_backward" in source
    assert "backward_supergraph" not in source


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
