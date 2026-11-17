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
    first = tl.trace(_ModuleModel(), x, capture=tl.options.CaptureOptions(replayable=True))
    second = tl.trace(_ModuleModel(), x, capture=tl.options.CaptureOptions(replayable=True))
    return tl.Bundle({"first": first, "second": second})


def test_compare_topology_rejects_reordered_graphs() -> None:
    """Reordered unique-fingerprint graphs are not reported identical."""

    x = torch.randn(2, 3)
    first = tl.trace(_ReorderedReluSigmoid(), x, capture=tl.options.CaptureOptions(replayable=True))
    second = tl.trace(
        _ReorderedSigmoidRelu(), x, capture=tl.options.CaptureOptions(replayable=True)
    )

    diff = compare_topology(first, second)

    assert diff.is_identical is False
    assert diff.unmatched_a
    assert diff.unmatched_b


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

    from torchlens.errors import ParamEvidenceMissingError

    x = torch.randn(2, 3)
    model_a = _BufferParamModel(scale=1.0, weight=1.0)
    model_b = _BufferParamModel(scale=2.0, weight=3.0)
    trace_a = tl.trace(model_a, x, capture=tl.options.CaptureOptions(replayable=True))
    trace_b = tl.trace(model_b, x, capture=tl.options.CaptureOptions(replayable=True))
    bundle = tl.Bundle({"a": trace_a, "b": trace_b})

    buffer_out = bundle.buffers["scale"].out
    assert torch.equal(buffer_out, torch.cat([model_a.scale, model_b.scale], dim=0))

    with pytest.raises(ParamEvidenceMissingError) as excinfo:
        _ = bundle.params["linear.weight"].weight_norm_diff
    assert excinfo.value.fields["code"] == "checkpoint_series_live_params"


def test_bundle_draw_rolled_and_backward_graphs(tmp_path: Path) -> None:
    """Bundle.draw handles the rolled and backward bundle pictures."""

    bundle = _bundle()
    rolled_source = bundle.draw(str(tmp_path / "rolled_bundle.svg"), rolled=True).dot
    backward_source = bundle.draw(str(tmp_path / "backward_bundle.svg"), graph="backward").dot

    assert "rolled" in rolled_source
    assert "(Linear)" in rolled_source
    assert "backward" in backward_source


def test_bundle_backward_regression_path(tmp_path: Path) -> None:
    """Bundle backward rendering emits per-member backward subgraphs."""

    bundle = _bundle()
    source = bundle.draw(str(tmp_path / "backward_bundle_regression.svg"), graph="backward").dot

    assert "cluster_backward_first" in source
    assert "cluster_backward_second" in source
    assert "backward" in source


def test_bundle_graph_style_primitives_and_chrome_trace_diff(tmp_path: Path) -> None:
    """Per-node/edge styles feed rendering and chrome trace diff writes JSON."""

    bundle = _bundle()
    first_node = bundle.supergraph.topological_order[0]
    first_edge = next(iter(bundle.supergraph.edges))
    source = bundle.draw(
        str(tmp_path / "styled_bundle.svg"),
        node_overrides={first_node: {"fillcolor": "#DDEAF7"}},
        edge_overrides={first_edge: {"color": "#0072B2"}},
    ).dot

    trace_path = tl.export.chrome_trace_diff(bundle, tmp_path / "trace_diff.json")
    payload = json.loads(trace_path.read_text(encoding="utf-8"))

    assert "#DDEAF7" in source
    assert "#0072B2" in source
    assert payload["metadata"]["schema"] == "torchlens.chrome_trace_diff.v1"
    assert payload["traceEvents"]


class _BranchModel(nn.Module):
    """A model whose forward takes one of two module branches, chosen by ``use_b``."""

    def __init__(self) -> None:
        """Initialize the stem, both branches and the head."""

        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.branch_a = nn.Linear(4, 4)
        self.branch_b = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        self.head = nn.Linear(4, 2)
        self.use_b = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stem, the selected branch, then the head."""

        h = torch.relu(self.stem(x))
        h = self.branch_b(h) if self.use_b else self.branch_a(h)
        return self.head(h)


def test_supergraph_merges_a_branch_the_reference_member_skipped() -> None:
    """Ops that only non-reference members ran become one shared node, not one per member."""

    model = _BranchModel().eval()
    runs = {}
    for name, use_b in (("a0", False), ("b1", True), ("b2", True), ("a3", False), ("b4", True)):
        model.use_b = use_b
        runs[name] = tl.trace(model, torch.randn(1, 4))
    graph = tl.Bundle(runs).supergraph

    def members_in(module: str) -> list[list[str]]:
        """Return each node's sorted member list for the nodes inside ``module``."""

        return [
            sorted(node.traces)
            for node in graph.nodes.values()
            if str(node.module_path or "").startswith(module)
        ]

    # A shared node may keep a member suffix when its label collides with a
    # reference label; what matters is one node per op, not its name.
    branch_b = members_in("branch_b")
    assert len(branch_b) == 3
    assert all(traces == ["b1", "b2", "b4"] for traces in branch_b)
    assert members_in("branch_a") == [["a0", "a3"]]
    head = members_in("head")
    assert head == [["a0", "a3", "b1", "b2", "b4"]]
