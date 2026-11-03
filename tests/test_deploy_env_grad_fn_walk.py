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

from torchlens.debug import sketch_grad_fn, walk_grad_fn
from torchlens.errors import GradFnWalkError


def test_graphless_tensor_refuses_typed_and_teaches() -> None:
    """No grad_fn anywhere: typed refusal with the stable code and a remedy."""

    with pytest.raises(GradFnWalkError) as excinfo:
        walk_grad_fn(torch.randn(3))
    assert excinfo.value.fields["code"] == "grad_fn_walk_no_graph"
    assert "no_grad" in str(excinfo.value)
    with pytest.raises(GradFnWalkError):
        walk_grad_fn([torch.randn(2).detach(), 7])


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
