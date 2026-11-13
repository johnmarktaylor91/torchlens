"""Bundle comparisons and the diff picture on a layer that runs more than once.

A ReLU module called twice from one loop line (after each of two linears) is
one logical layer with two passes; ``Layer.out`` refuses on it. The diff picture draws one node
per pass, each coloured by its own pass's distance (the per-pass rows of
``Bundle.compare``); a layer-level value (a stored comparison) joins the
passes in pass order.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bundle._aligned_sites import site_row_key
from torchlens.bundle._deltas import _tensor_field, site_deltas
from torchlens.errors import MultiMatchWarning
from torchlens.intervention._metrics import relative_l2


class _SharedRelu(nn.Module):
    """``fc1 -> relu -> fc2 -> relu``: the one ReLU module runs twice from one loop line.

    A stateless module's calls are passes of one layer only when one line makes them;
    called from two lines, the two calls are two layers (``docs/guides/grouping.md``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in (self.fc1, self.fc2):
            x = self.relu(fc(x))
        return x


def _clean_trace() -> tl.Trace:
    torch.manual_seed(0)
    model = _SharedRelu().eval()
    return tl.trace(model, torch.randn(2, 4), capture=tl.options.CaptureOptions(replayable=True))


def _relu_label(trace: tl.Trace) -> str:
    return next(str(op.layer_label) for op in trace._layer_list if op.func_name == "relu")


def _zero_relu_bundle() -> tl.Bundle:
    clean = _clean_trace()
    ablated = clean.fork("ablated")
    # The one module is two sites (its two passes): the selector fans out to both.
    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        ablated.do(
            tl.module("relu"),
            tl.zero_ablate(),
            intervention=tl.options.InterventionOptions(),
        )
    return tl.Bundle({"clean": clean, "ablated": ablated}, baseline="clean")


def _side_node_ids(source: str, side: str) -> list[str]:
    body = source.split(f"subgraph cluster_{side} {{", 1)[1].split("\n\t}", 1)[0]
    return re.findall(r"^\t\t(\w+) \[", body, re.M)


def test_bundle_diff_draws_one_node_per_pass(tmp_path: Path) -> None:
    """A zero-ablated shared ReLU draws as many nodes per column as it has passes."""

    bundle = _zero_relu_bundle()
    clean = bundle["clean"]
    relu = _relu_label(clean)
    num_passes = clean.layers[relu].num_passes
    assert num_passes == 2

    source = bundle.draw(graph="diff", path=str(tmp_path / "shared_relu")).dot

    assert (tmp_path / "shared_relu.svg").exists()
    for side in ("clean", "intervention"):
        relu_nodes = [
            node for node in _side_node_ids(source, side) if node.startswith(f"{side}_{relu}_p")
        ]
        assert len(relu_nodes) == num_passes
    for pass_number in range(1, num_passes + 1):
        assert f'label="{relu}:{pass_number}\n' in source
    # The passes stay apart in the edges: fc2 consumes pass 1 and feeds pass 2.
    linear_2 = next(
        str(op.layer_label) for op in clean._layer_list if f"{relu}:1" in map(str, op.parents)
    )
    assert f"clean_{relu}_p1 -> clean_{linear_2}" in source
    assert f"clean_{linear_2} -> clean_{relu}_p2" in source


def test_bundle_diff_values_are_the_per_pass_compare_rows() -> None:
    """Each drawn op's value equals its row in ``Bundle.compare``, pass by pass."""

    bundle = _zero_relu_bundle()
    rows = dict(bundle.compare("clean", "ablated", metric="relative_l2"))
    deltas = site_deltas(bundle, "relative_l2", "clean", "out", ["clean", "ablated"])

    relu = _relu_label(bundle["clean"])
    assert {f"{relu}:1", f"{relu}:2"} <= set(rows)
    assert relu not in rows
    for op in bundle["ablated"]._layer_list:
        assert deltas.value(op, "ablated") == pytest.approx(rows[site_row_key(op)])
        assert deltas.value(op, "clean") == 0.0


def test_rank_changes_ranks_each_pass() -> None:
    """``rank_changes`` scores every pass of a multi-pass layer as its own site."""

    bundle = _zero_relu_bundle()
    relu = _relu_label(bundle["clean"])

    ranked = dict(bundle.rank_changes(k=20, metric="relative_l2"))

    assert ranked[f"{relu}:1"] == pytest.approx(1.0)
    assert ranked[f"{relu}:2"] == pytest.approx(1.0)
    assert relu not in ranked


def test_store_comparison_joins_passes_in_pass_order() -> None:
    """A layer-level stored value is the metric over both passes joined in pass order.

    The edit sits between the passes (``fc2``), so pass 1 is unchanged and only
    pass 2 moves: a first-pass pick would read 0 and a last-pass pick would
    divide by pass 2's norm alone.
    """

    clean = _clean_trace()
    patched = clean.fork("patched").do(
        tl.module("fc2"),
        tl.scale(0.5),
        intervention=tl.options.InterventionOptions(engine="replay"),
    )
    bundle = tl.Bundle({"clean": clean, "patched": patched}, baseline="clean")
    relu = _relu_label(clean)

    name = bundle.store_comparison(metric="relative_l2")
    stored = bundle.stored_comparison(name)[relu]["patched"]

    clean_passes = [clean.layers[relu].ops[k].out for k in range(2)]
    patched_passes = [patched.layers[relu].ops[k].out for k in range(2)]
    joined = relative_l2(
        torch.cat([t.reshape(-1) for t in clean_passes]),
        torch.cat([t.reshape(-1) for t in patched_passes]),
    )
    last_only = relative_l2(clean_passes[1], patched_passes[1])
    assert torch.equal(clean_passes[0], patched_passes[0])
    assert stored == pytest.approx(float(joined))
    assert stored > 0.0
    assert stored != pytest.approx(float(last_only))


def test_tensor_field_joins_passes_or_reports_none() -> None:
    """A multi-pass layer's field is its passes joined in pass order, ``None`` if one lacks it."""

    first, second = torch.tensor([[1.0, 2.0]]), torch.tensor([3.0])
    layer = SimpleNamespace(
        ops={
            2: SimpleNamespace(pass_number=2, out=second),
            1: SimpleNamespace(pass_number=1, out=first),
        }
    )
    assert torch.equal(_tensor_field(layer, "out"), torch.tensor([1.0, 2.0, 3.0]))

    layer.ops[2].out = None
    assert _tensor_field(layer, "out") is None
