"""``Bundle.compare`` on a fork of the same model: recurrence, masks, shapes.

Three regressions from the behaviour net's ``bundle-validate`` cells:

* a recurrent model folds its passes into layers, so the layer-level edge
  set carries feedback edges; the supergraph must still order its nodes
  instead of refusing the member as cyclic;
* two members holding the same ``-inf`` attention mask must read distance
  ``0.0`` at that site, not ``nan``;
* operands of different shapes refuse (typed) naming the site.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.bundle._aligned_sites import site_row_key
from torchlens.bundle._deltas import _distance_value
from torchlens.intervention._topology.topology import (
    _preferential_topological_order,
)
from torchlens.intervention.errors import BundleComparisonError, MultiMatchWarning


class _Recurrent(nn.Module):
    """Three unrolled steps; each step's output feeds the next step's add."""

    def __init__(self) -> None:
        """Initialize modules."""
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three recurrent steps from a single-pass gate.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final hidden state.
        """
        gate = torch.sigmoid(x)
        hidden = gate
        for _ in range(3):
            hidden = torch.tanh(self.lin(gate) + hidden)
        return hidden


def _trace(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    return tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))


def _first_interior_op(trace: tl.Trace) -> str:
    """Return the op label of the first saved single-pass op with parents."""

    for op in trace.saved_ops:
        single_pass = getattr(op.layer, "num_passes", 1) == 1
        if op.parents and op.dtype is not None and op.dtype.is_floating_point and single_pass:
            return op.label
    raise AssertionError("no single-pass interior op")


def _descendants(trace: tl.Trace, label: str) -> set[str]:
    """Row keys of every op downstream of ``label`` (an op label or a row key)."""
    seen: set[str] = set()
    frontier = [label]
    while frontier:
        current = frontier.pop()
        for child in trace[current].children or ():
            child_label = child if isinstance(child, str) else getattr(child, "label", str(child))
            if child_label not in seen:
                seen.add(child_label)
                frontier.append(child_label)
    keys: set[str] = set()
    for name in seen:
        node = trace[name]
        try:
            keys.add(site_row_key(node))
        except InvalidArgumentError:  # a multi-pass Layer: one row key per pass
            keys.update(site_row_key(op) for op in node.ops)
    return keys


def _compare_fork(model: nn.Module, x: torch.Tensor) -> tuple[tl.Trace, str, dict[str, float]]:
    """Zero one interior op on a fork and compare; returns the edited site's ROW KEY."""
    trace = _trace(model, x)
    label = _first_interior_op(trace)
    fork = trace.fork()
    fork.do(tl.label(label), tl.zero_ablate())
    pair = tl.Bundle({"base": trace, "edited": fork}, baseline="base")
    rows = dict(pair.compare("base", "edited", metric="relative_l2"))
    return trace, site_row_key(trace[label]), rows


# --- F5: a recurrent member is not a cyclic supergraph -------------------------


def test_recurrent_compare_gives_one_row_per_pass() -> None:
    """Multi-pass ops get one ``compare`` row per pass, keyed ``layer_label:k``."""

    torch.manual_seed(0)
    trace, _site, rows = _compare_fork(_Recurrent(), torch.randn(2, 4))
    tanh_ops = [op for op in trace.ops if op.func_name == "tanh"]
    assert len(tanh_ops) == 3 and {op.num_passes for op in tanh_ops} == {3}
    layer_label = str(tanh_ops[0].layer_label)
    assert layer_label not in rows
    assert {f"{layer_label}:{k}" for k in (1, 2, 3)} <= set(rows)
    single = [op for op in trace.ops if op.num_passes == 1 and op.func_name == "sigmoid"]
    assert single and str(single[0].layer_label) in rows
    pair = tl.Bundle({"base": trace, "edited": trace.fork()}, baseline="base")
    ranked = dict(pair.rank_changes(k=64))
    assert {f"{layer_label}:{k}" for k in (1, 2, 3)} <= set(ranked)


class _ReusedRelu(nn.Module):
    """A residual block that calls one ReLU module twice, as ResNet blocks do."""

    def __init__(self) -> None:
        """Initialize modules."""
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv, batchnorm and ReLU, add the input back and apply the same ReLU.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, 2, height, width)``.

        Returns
        -------
        torch.Tensor
            Block output.
        """
        return self.relu(self.relu(self.bn(self.conv(x))) + x)


def test_stored_comparison_joins_the_passes_of_a_multi_pass_layer() -> None:
    """A multi-pass supergraph node reads the metric over its passes joined in pass order."""

    torch.manual_seed(0)
    trace = _trace(_Recurrent(), torch.randn(2, 4))
    fork = trace.fork()
    fork.do(tl.label(_first_interior_op(trace)), tl.zero_ablate())
    pair = tl.Bundle({"base": trace, "edited": fork}, baseline="base")

    stored = pair.stored_comparison(pair.store_comparison("relative_l2"))

    tanh_label = str(next(op for op in trace.ops if op.func_name == "tanh").layer_label)
    tanh_nodes = [name for name in stored if name.startswith("tanh_")]
    assert len(tanh_nodes) == 1
    joined = _distance_value(
        torch.cat([trace[f"{tanh_label}:{k}"].out.reshape(-1) for k in (1, 2, 3)]),
        torch.cat([fork[f"{tanh_label}:{k}"].out.reshape(-1) for k in (1, 2, 3)]),
        "relative_l2",
        site=tanh_nodes[0],
    )
    assert stored[tanh_nodes[0]]["base"] == 0.0
    assert stored[tanh_nodes[0]]["edited"] == pytest.approx(joined)
    assert joined > 0.0


def test_diff_drawing_of_a_reused_module_does_not_refuse(tmp_path: Path) -> None:
    """Zero-ablating a ReLU module called twice draws a diff (ResNet bundle-diff case)."""

    if shutil.which("dot") is None:
        pytest.skip("graphviz dot not installed")
    torch.manual_seed(0)
    model = _ReusedRelu().eval()
    trace = _trace(model, torch.randn(1, 2, 4, 4))
    ablated = trace.fork("ablated")
    # The module runs twice, so the selector fans out to both of its sites.
    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        ablated.do(
            tl.module("relu"),
            tl.zero_ablate(),
            intervention=tl.options.InterventionOptions(confirm_mutation=True),
        )
    bundle = tl.Bundle({"clean": trace, "ablated": ablated}, baseline="clean")

    drawing = bundle.draw(graph="diff", path=str(tmp_path / "d"), open_viewer=False)

    assert drawing.dot.startswith("// TorchLens bundle diff")
    assert (tmp_path / "d.svg").exists()
    stored = bundle.stored_comparison(bundle.store_comparison("relative_l2"))
    relu_nodes = [name for name in stored if name.startswith("relu_")]
    assert relu_nodes and all(math.isfinite(stored[n]["ablated"]) for n in relu_nodes)


def test_contradictory_member_orders_refuse_typed() -> None:
    """Members that disagree on operation order refuse with a coded error."""

    edges = {("a", "b"): {"left"}, ("b", "a"): {"right"}}
    with pytest.raises(BundleComparisonError) as info:
        _preferential_topological_order({"a", "b", "c"}, edges, ["a", "b", "c"])

    exc = info.value
    assert isinstance(exc, ValueError)
    assert exc.fields["code"] == "bundle_gate_model_axis_unmet"
    assert exc.fields["operation"] == "supergraph"
    assert exc.fields["cyclic_edges"] == ["a->b (left)", "b->a (right)"]
    assert "Remedy" in str(exc)


# --- F6: identical ``-inf`` payloads read 0.0, not nan ------------------------


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize("metric", ["pearson"])
def test_identical_payload_with_inf_reads_zero(metric: str) -> None:
    """Equal tensors holding ``-inf`` are unchanged under every registered metric."""

    mask = torch.full((4, 4), float("-inf")).triu(1)
    assert _distance_value(mask, mask.clone(), metric) == 0.0
    with_nan = torch.tensor([1.0, float("nan"), 3.0])
    assert _distance_value(with_nan, with_nan.clone(), metric) == 0.0
    changed = _distance_value(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0]), metric)
    assert math.isfinite(changed) and changed > 0.0


def test_shared_non_finite_entries_are_left_out_of_the_metric() -> None:
    """Masked positions both operands agree on do not turn the distance into nan."""

    reference = torch.tensor([1.0, float("-inf"), 3.0])
    candidate = torch.tensor([2.0, float("-inf"), 3.0])
    assert _distance_value(reference, candidate, "relative_l2") == pytest.approx(1 / math.sqrt(10))
    disagreeing = torch.tensor([1.0, 2.0, 3.0])
    assert math.isnan(_distance_value(reference, disagreeing, "relative_l2"))


def test_identical_payload_shortcut_leaves_user_callables_alone() -> None:
    """A user metric is evaluated as given, even on identical operands."""

    calls: list[int] = []

    def constant(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return torch.tensor(0.5)

    assert _distance_value(torch.ones(3), torch.ones(3), constant) == 0.5
    assert calls == [1]


# --- F8: different shapes refuse naming the site ------------------------------


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    ("left", "right"),
    [
        (torch.ones(2, 3), torch.ones(3, 2)),  # equal element counts, different shapes
    ],
)
def test_shape_mismatch_refuses_typed_naming_the_site(
    left: torch.Tensor, right: torch.Tensor
) -> None:
    """Shape-mismatched operands refuse with ``metric_shape_mismatch`` and the site."""

    with pytest.raises(InvalidArgumentError) as info:
        _distance_value(left, right, "cosine", site="view_1_1")

    exc = info.value
    assert exc.fields["code"] == "metric_shape_mismatch"
    assert exc.fields["site"] == "view_1_1"
    assert exc.fields["reference_shape"] == tuple(left.shape)
    assert exc.fields["candidate_shape"] == tuple(right.shape)
    assert "view_1_1" in str(exc)
    assert exc.fields["remedy"]
