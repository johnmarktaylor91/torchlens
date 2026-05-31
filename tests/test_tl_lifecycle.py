"""Lifecycle tests for TorchLens private ``._tl`` metadata cleanup."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._tl import (
    TensorMeta,
    get_buffer_address,
    get_module_meta,
    get_tensor_label,
    set_module_meta,
    set_tensor_label,
)
from torchlens.backends.torch.model_prep import _tag_untagged_buffers
from torchlens.intervention.runtime import _copy_tl_replacement_attrs
from torchlens.partial import PartialTrace
from torchlens.utils.introspection import get_vars_of_type_from_obj


class _LifecycleModel(nn.Module):
    """Small model with parameters and a persistent buffer."""

    def __init__(self) -> None:
        """Create the fixture model."""
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0, 3.0, 4.0]))
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a simple buffer and module graph."""
        return torch.relu(self.linear(x * self.scale))


class _FailingLifecycleModel(_LifecycleModel):
    """Lifecycle model that raises after producing partial outputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raise after logging at least one tensor."""
        y = self.linear(x * self.scale)
        raise RuntimeError("intentional lifecycle failure")
        return y


def _assert_tensor_tree_clean(value: Any) -> None:
    """Assert all tensors reachable from a value have no TorchLens metadata."""
    tensors = get_vars_of_type_from_obj(value, torch.Tensor, search_depth=5)
    if isinstance(value, torch.Tensor):
        tensors.append(value)
    for tensor in tensors:
        assert not hasattr(tensor, "_tl")


@pytest.mark.smoke
def test_successful_capture_cleans_session_tensor_and_param_metadata() -> None:
    """Successful captures should clean session metadata while keeping module metadata."""
    model = _LifecycleModel()
    x = torch.randn(2, 3)
    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}

    log = tl.trace(
        model,
        x,
        save_arg_values=True,
        activation_transform=lambda t: t.float().mean(),
    )

    assert not hasattr(x, "_tl")
    assert get_tensor_label(model.scale) is None
    assert get_buffer_address(model.scale) is None
    for name, param in model.named_parameters():
        assert param.requires_grad == original_requires_grad[name]
        assert not hasattr(param, "_tl")

    module_meta = get_module_meta(model.linear)
    assert module_meta is not None
    assert module_meta.address == "linear"
    assert module_meta.module_type == "Linear"

    for layer in log.layer_list:
        _assert_tensor_tree_clean(layer.out)
        _assert_tensor_tree_clean(getattr(layer, "transformed_out", None))
        _assert_tensor_tree_clean(getattr(layer, "saved_args", None))
        _assert_tensor_tree_clean(getattr(layer, "saved_kwargs", None))


@pytest.mark.smoke
def test_failed_capture_cleans_partial_outputs_buffers_and_params() -> None:
    """Failed captures should clean partial output metadata and restore model state."""
    model = _FailingLifecycleModel()
    x = torch.randn(2, 3)
    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}

    with pytest.raises(RuntimeError) as exc_info:
        tl.trace(model, x)

    partial = getattr(exc_info.value, "partial_log", None)
    assert isinstance(partial, PartialTrace)
    for layer in partial.raw_layers:
        _assert_tensor_tree_clean(getattr(layer, "out", None))

    assert get_tensor_label(model.scale) is None
    assert get_buffer_address(model.scale) is None
    for name, param in model.named_parameters():
        assert param.requires_grad == original_requires_grad[name]
        assert not hasattr(param, "_tl")


@pytest.mark.smoke
def test_tag_untagged_buffers_promotes_prior_label_mid_session() -> None:
    """Dynamic buffer tagging should preserve prior label as buffer_parent."""
    module = nn.Module()
    module.register_buffer("buf", torch.ones(2))
    set_module_meta(module, address="block", module_type="Module")
    set_tensor_label(module.buf, "mul_1_2_raw")

    _tag_untagged_buffers(module)

    assert get_tensor_label(module.buf) is None
    assert get_buffer_address(module.buf) == "block.buf"
    assert isinstance(module.buf._tl, TensorMeta)
    assert module.buf._tl.buffer_parent == "mul_1_2_raw"


@pytest.mark.smoke
def test_intervention_replacement_copies_tensor_meta_subclass() -> None:
    """Intervention replacement metadata copying should preserve TensorMeta."""
    source = torch.ones(2)
    replacement = torch.zeros(2)
    set_tensor_label(source, "relu_1_2_raw")

    _copy_tl_replacement_attrs(source, replacement)

    assert isinstance(replacement._tl, TensorMeta)
    assert replacement._tl is not source._tl
    assert replacement._tl.label_raw == "relu_1_2_raw"
