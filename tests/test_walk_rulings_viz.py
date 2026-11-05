"""Regression coverage for edge shapes, annotated source and feature optimization."""

from __future__ import annotations

import inspect
import shutil

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import mechinterp
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
    assert any("[1, 1]" in line for line in edge_lines), shaped
    assert any("[1, 3]" in line for line in edge_lines), shaped
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
    image, values = mechinterp.visualize_feature(model, "0", inputs=inputs, unit=0, steps=4, lr=0.1)
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
    image, values = mechinterp.visualize_feature(
        model, relu.label, inputs=torch.ones(1, 3), sae=Encoder(), unit=0, steps=3
    )
    assert values[-1] > values[0]
    assert image.shape == (1, 3)


def test_visualize_feature_seed_and_unit_tuple():
    """Local noise seeds reproduce the input and tuple units address non-batch axes."""
    model = nn.Sequential(nn.Identity())
    rng = torch.get_rng_state().clone()
    image, values = mechinterp.visualize_feature(
        model, "0", input_shape=(1, 2, 3), unit=(0, 1), seed=21, steps=2
    )
    other, repeated = mechinterp.visualize_feature(
        model, "0", input_shape=(1, 2, 3), unit=(0, 1), seed=21, steps=2
    )
    assert torch.equal(image, other) and values == repeated
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"lr": float("nan")}, {"unit": "x"}])
def test_visualize_feature_refuses_invalid_settings(kwargs):
    """Malformed settings refuse before any model execution."""
    model = nn.Sequential(nn.Identity())
    with pytest.raises(tl.errors.InvalidArgumentError) as caught:
        mechinterp.visualize_feature(model, "0", inputs=torch.ones(1, 2), **kwargs)
    assert caught.value.fields["code"] == "feature_visualization_invalid"


def test_visualize_feature_export():
    """The analysis kit advertises the new callable."""
    assert "visualize_feature" in mechinterp.__all__


def test_source_annotated_roundtrip_and_source_privacy(tmp_path):
    """Embedded source annotates after load; opt-out artifacts refuse with the same code."""
    trace = tl.trace(SourceModel(), torch.ones(1, 3))
    source_path = tmp_path / "with-source.tlspec"
    trace.save(source_path)
    loaded = tl.load(source_path)
    assert loaded.source.annotated() == trace.source.annotated()
    private_path = tmp_path / "without-source.tlspec"
    trace.save(private_path, include_source=False)
    with pytest.raises(tl.errors.InvalidArgumentError) as caught:
        tl.load(private_path).source.annotated()
    assert caught.value.fields["code"] == "source_unavailable"


def test_edge_shape_option_group_and_lens_use_the_same_channel():
    """Grouped draw translation and the debug lens with edge shapes preserve the flag."""
    from torchlens._options_validation import visualization_to_render_kwargs
    from torchlens._visualization.lenses import resolve_lens

    options = tl.options.VisualizationOptions(show_edge_shapes=True)
    assert visualization_to_render_kwargs(options)["show_edge_shapes"] is True
    copied = tl.options.VisualizationOptions.from_values(
        options.to_dict(), frozenset({"show_edge_shapes"})
    )
    assert copied.show_edge_shapes and copied.is_field_explicit("show_edge_shapes")
    trace = tl.trace(nn.Sequential(nn.Linear(4, 3), nn.ReLU()), torch.randn(2, 4))
    resolution = resolve_lens(trace, "debug", {"show_edge_shapes": True})
    assert resolution.draw_kwargs["show_edge_shapes"] is True


def test_visualize_feature_rejects_disconnected_objective_and_cleans_hooks():
    """A constant target is a typed failure and leaves no binding hooks installed."""

    class Constant(nn.Module):
        def forward(self, x):
            return x.detach()

    model = nn.Sequential(Constant()).train()
    with pytest.raises(tl.errors.InvalidArgumentError) as caught:
        mechinterp.visualize_feature(model, "0", inputs=torch.ones(1, 2), steps=1)
    assert caught.value.fields["code"] == "feature_visualization_invalid"
    assert model.training and model[0].training
    assert not model[0]._forward_hooks and not model[0]._forward_pre_hooks


def test_visualize_feature_rejects_multiple_fires():
    """A reused module must be narrowed instead of optimizing an arbitrary occurrence."""

    class Twice(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.act(x))

    model = Twice()
    with pytest.raises(tl.errors.InvalidArgumentError) as caught:
        mechinterp.visualize_feature(model, "act", inputs=torch.ones(1, 2), steps=1)
    assert caught.value.fields["code"] == "feature_visualization_invalid"
    assert not model.act._forward_hooks and not model.act._forward_pre_hooks
