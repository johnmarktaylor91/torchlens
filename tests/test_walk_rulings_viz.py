"""Regression coverage for edge shapes, annotated source and feature optimization."""

from __future__ import annotations

import inspect
import shutil

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._edge_shapes import _with_edge_shapes
from torchlens._visualization.render_ir import build_render_ir
from torchlens._visualization.request import RenderContext


class SourceModel(nn.Module):
    """Keep forward line attribution distinct from a helper in the same file."""

    def helper(self, x):
        """Run an op outside the root forward body."""
        return torch.sigmoid(x)

    def forward(self, x):
        """Run two directly attributable operations and one helper."""
        hidden = torch.relu(x)
        return self.helper(hidden) + hidden


class SplitModel(nn.Module):
    """Produce differently shaped tensor slots from one call."""

    def forward(self, x):
        """Use both split outputs through different consumers."""
        left, right = torch.split(x, [1, 3], dim=1)
        return left.sin(), right.cos()


def test_edge_shapes_default_ir_is_identical():
    """The absent channel returns precisely the original IR, without new attrs."""
    model = SplitModel()
    trace = tl.trace(model, torch.ones(1, 4))
    ir = build_render_ir(trace, collapse_fn=None, repeat_folds={})
    assert _with_edge_shapes(trace, ir) is ir
    options = tl.options.VisualizationOptions()
    assert options.show_edge_shapes is False
    assert tl.options.VisualizationOptions(show_edge_shapes=True).show_edge_shapes is True


def test_edge_shapes_render_specific_output_slots(tmp_path):
    """Each edge leaving a multi-output call names its particular slot shape."""
    if shutil.which("dot") is None:
        pytest.skip("Graphviz dot is not installed")
    model = SplitModel()
    trace = tl.trace(model, torch.ones(1, 4))
    path = str(tmp_path / "graph.svg")
    default = str(trace.draw(path, order_siblings=False))
    explicit = str(trace.draw(path, show_edge_shapes=False, order_siblings=False))
    assert default == explicit
    shaped = str(trace.draw(path, show_edge_shapes=True, order_siblings=False))
    edge_lines = [line for line in shaped.splitlines() if " -> " in line]
    assert any("[1, 1]" in line for line in edge_lines)
    assert any("[1, 3]" in line for line in edge_lines)
    assert shaped != default


def test_edge_shapes_ir_has_labels_without_graphviz():
    """The channel is a renderer-neutral IR decision, also consumed by rank layout."""
    model = SplitModel()
    trace = tl.trace(model, torch.ones(1, 4))
    ir = build_render_ir(
        trace, collapse_fn=None, repeat_folds={}, context=RenderContext(show_edge_shapes=True)
    )
    shaped = _with_edge_shapes(trace, ir)
    labels = [dict(edge.attrs).get("label") for edge in shaped.edges]
    assert "[1, 1]" in labels
    assert "[1, 3]" in labels


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_edge_shapes_refuses_non_bool(value):
    """Options and drawing consistently refuse truthy non-bools."""
    with pytest.raises(tl.errors.InvalidArgumentError):
        tl.options.VisualizationOptions(show_edge_shapes=value)


def test_source_annotated_maps_forward_lines():
    """Only forward lines are shown, with the captured labels on the right lines."""
    model = SourceModel()
    trace = tl.trace(model, torch.ones(1, 3))
    annotated = trace.source.annotated()
    relu = next(op for op in trace.ops if op.func_name == "relu")
    add = next(op for op in trace.ops if op.func_name == "__add__")
    relu_line = next(line for line in annotated.splitlines() if "hidden = torch.relu(x)" in line)
    return_line = next(
        line for line in annotated.splitlines() if "return self.helper(hidden)" in line
    )
    assert relu.label in relu_line
    assert add.label in return_line
    assert "return torch.sigmoid(x)" not in annotated
    assert "input_1" not in annotated
    _, first = inspect.getsourcelines(SourceModel.forward)
    assert annotated.splitlines()[0].lstrip().startswith(f"{first} |")


def test_source_annotated_refuses_missing_snapshot():
    """No source is a typed refusal, never a plausible empty annotation."""
    model = SourceModel()
    trace = tl.trace(model, torch.ones(1, 3))
    trace._source_code_blob = {}
    with pytest.raises(tl.errors.InvalidArgumentError) as caught:
        trace.source.annotated()
    assert caught.value.fields["code"] == "source_unavailable"
    assert caught.value.fields["remedy"]


def test_visualize_feature_increases_channel_and_preserves_parameters():
    """CPU ascent improves a planted conv channel without changing caller input or grads."""
    model = nn.Sequential(nn.Conv2d(1, 2, 1, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.fill_(1)
    model[0].weight.grad = torch.ones_like(model[0].weight)
    inputs = torch.zeros(1, 1, 3, 3)
    before = model[0].weight.detach().clone()
    image, values = tl.mechinterp.visualize_feature(
        model, "0", inputs=inputs, unit=0, steps=4, lr=0.1
    )
    assert len(values) == 5 and values[-1] > values[0] + 0.1
    assert not image.requires_grad
    assert torch.equal(inputs, torch.zeros_like(inputs))
    assert torch.equal(model[0].weight, before)
    assert torch.equal(model[0].weight.grad, torch.ones_like(before))


def test_visualize_feature_op_label_and_sae():
    """A recorded functional label resolves to a structural binding; SAE latents keep gradients."""

    class Encoder:
        def encode(self, out):
            return out * 2

    model = SourceModel()
    trace = tl.trace(model, torch.ones(1, 3))
    relu = next(op for op in trace.ops if op.func_name == "relu")
    image, values = tl.mechinterp.visualize_feature(
        model, relu.label, inputs=torch.ones(1, 3), sae=Encoder(), unit=0, steps=3
    )
    assert values[-1] > values[0]
    assert image.shape == (1, 3)


def test_visualize_feature_seed_and_unit_tuple():
    """Local noise seeds reproduce the input and tuple units address non-batch axes."""
    model = nn.Sequential(nn.Identity())
    rng = torch.get_rng_state().clone()
    image, values = tl.mechinterp.visualize_feature(
        model, "0", input_shape=(1, 2, 3), unit=(0, 1), seed=21, steps=2
    )
    other, repeated = tl.mechinterp.visualize_feature(
        model, "0", input_shape=(1, 2, 3), unit=(0, 1), seed=21, steps=2
    )
    assert torch.equal(image, other) and values == repeated
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"lr": float("nan")}, {"unit": "x"}])
def test_visualize_feature_refuses_invalid_settings(kwargs):
    """Malformed settings refuse before any model execution."""
    model = nn.Sequential(nn.Identity())
    with pytest.raises(tl.errors.InvalidArgumentError) as caught:
        tl.mechinterp.visualize_feature(model, "0", inputs=torch.ones(1, 2), **kwargs)
    assert caught.value.fields["code"] == "feature_visualization_invalid"


def test_visualize_feature_export():
    """The analysis kit advertises the new callable."""
    assert "visualize_feature" in tl.mechinterp.__all__
