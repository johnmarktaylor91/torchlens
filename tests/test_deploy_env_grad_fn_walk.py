"""tl.debug grad_fn walker (lane F37; approved diagnostic, R grad_fn ruling).

Structure-only post-hoc sketching of a tensor's autograd graph: a REAL loss
tensor from a real architecture class walks completely with named parameter
leaves; renders always carry the structure-only legend; graphless tensors
refuse typed with a teaching message; the defensive ceiling discloses
truncation instead of stalling.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from torchlens.debug import GradFnWalkError, sketch_grad_fn, walk_grad_fn


@pytest.mark.smoke
def test_walk_names_leaves_and_orients_edges_forward() -> None:
    """Leaves resolve names from params=, edges run producer -> consumer."""

    w = torch.randn(4, 4, requires_grad=True)
    y = (w @ w).relu().sum()
    sketch = walk_grad_fn(y, params={"w": w})
    kinds = {node.kind for node in sketch.nodes}
    assert kinds == {"root", "op", "leaf"}
    leaves = [node for node in sketch.nodes if node.kind == "leaf"]
    assert [leaf.param_name for leaf in leaves] == ["w"]
    assert leaves[0].shape == (4, 4)
    assert leaves[0].requires_grad is True
    by_id = {node.node_id: node for node in sketch.nodes}
    (root_edge,) = [e for e in sketch.edges if by_id[e[1]].kind == "root"]
    assert by_id[root_edge[0]].name == "SumBackward0"
    assert not sketch.truncated


@pytest.mark.smoke
def test_walk_real_loss_tensor_from_real_architecture() -> None:
    """A real cross-entropy loss from a real GPT-2 class walks completely."""

    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=128, n_positions=64)
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(config).train()
    input_ids = torch.randint(0, 128, (1, 8))
    loss = model(input_ids=input_ids, labels=input_ids).loss
    assert loss.grad_fn is not None  # a REAL loss tensor, not a synthetic scalar

    sketch = walk_grad_fn(loss, model=model)
    assert not sketch.truncated
    named = {node.param_name for node in sketch.nodes if node.param_name}
    assert "transformer.wte.weight" in named
    assert any(node.name == "NllLossBackward0" for node in sketch.nodes)
    op_count = sum(1 for node in sketch.nodes if node.kind == "op")
    leaf_count = sum(1 for node in sketch.nodes if node.kind == "leaf")
    assert op_count > 50 and leaf_count >= 20


@pytest.mark.smoke
def test_walk_double_backprop_gradient_graph() -> None:
    """A create_graph=True gradient's graph (grad-of-grad) walks too."""

    w = torch.randn(4, 4, requires_grad=True)
    y = (w @ w).sum()
    (gradient,) = torch.autograd.grad(y, w, create_graph=True)
    sketch = walk_grad_fn(gradient.sum())
    assert sketch.n_nodes > 3
    assert any(node.kind == "leaf" for node in sketch.nodes)


@pytest.mark.smoke
def test_graphless_tensor_refuses_typed_and_teaches() -> None:
    """No grad_fn anywhere: typed refusal with the stable code and a remedy."""

    with pytest.raises(GradFnWalkError) as excinfo:
        walk_grad_fn(torch.randn(3))
    assert excinfo.value.fields["code"] == "grad_fn_walk_no_graph"
    assert "no_grad" in str(excinfo.value)
    with pytest.raises(GradFnWalkError):
        walk_grad_fn([torch.randn(2).detach(), 7])


@pytest.mark.smoke
def test_truncation_is_disclosed_never_silent() -> None:
    """Hitting max_nodes sets the flag and the render says so."""

    x = torch.randn(2, requires_grad=True)
    y = x
    for _ in range(30):
        y = y * 2.0
    sketch = walk_grad_fn(y.sum(), max_nodes=5)
    assert sketch.truncated
    source = sketch_grad_fn(y.sum(), max_nodes=5)
    assert "WALK TRUNCATED" in source


@pytest.mark.smoke
def test_sketch_dot_source_is_structure_only_with_legend() -> None:
    """DOT source carries the legend and never any tensor values."""

    w = torch.randn(3, 3, requires_grad=True)
    source = sketch_grad_fn((w * 2.0).sum(), params={"w": w})
    assert "structure-only sketch" in source
    assert "no values, timing, or verification" in source
    assert "MulBackward0" in source and "SumBackward0" in source
    assert "w\\n(3, 3)" in source


@pytest.mark.heavy
def test_sketch_renders_bounded_file(tmp_path) -> None:
    """File rendering goes through the bounded runner and publishes the file."""

    pytest.importorskip("graphviz")
    w = torch.randn(3, 3, requires_grad=True)
    target = os.path.join(str(tmp_path), "sketch")
    sketch_grad_fn((w @ w).sum(), target, file_format="svg", params={"w": w})
    rendered = target + ".svg"
    assert os.path.exists(rendered) and os.path.getsize(rendered) > 0
    with open(rendered, encoding="utf-8") as handle:
        svg = handle.read()
    # Hyphens render HTML-escaped (&#45;), so pin an escape-stable slice.
    assert "only sketch of grad_fn graph" in svg


@pytest.mark.smoke
def test_model_and_params_compose_with_params_winning() -> None:
    """model= names leaves; explicit params= wins on collisions."""

    model = nn.Linear(3, 1)
    y = model(torch.randn(2, 3)).sum()
    named_by_model = {
        node.param_name for node in walk_grad_fn(y, model=model).nodes if node.param_name
    }
    assert named_by_model == {"weight", "bias"}
    override = walk_grad_fn(y, model=model, params={"W_custom": model.weight})
    assert "W_custom" in {node.param_name for node in override.nodes if node.param_name}
