"""Module-held plain tensors read in ``forward`` are captured as buffer sources.

timm's EfficientViT (MSRA) keeps an eval-mode attention-bias cache: a plain tensor
in a dict attribute (``attention_bias_cache["cpu"]``), filled by the first forward
and indexed by ``getitem`` on every later one. On a warmed model the cached tensor
existed before the capture but carried no TorchLens provenance, so the ``getitem``
had no parent and validation called it a dangling node (``graph_connectivity``).
Model preparation already stamped plain tensor attributes and list/tuple items as
buffers; dict values now get the same buffer source (address ``<attr>.<key>``).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


class _DictCachedBias(nn.Module):
    """EfficientViT-shaped: an eval-mode dict cache of an indexed bias table."""

    def __init__(self) -> None:
        super().__init__()
        self.biases = nn.Parameter(torch.randn(3, 16))
        self.register_buffer("idxs", torch.randint(0, 16, (5, 5)), persistent=False)
        self.cache: dict[str, torch.Tensor] = {}

    def get_bias(self, device: torch.device) -> torch.Tensor:
        if self.training:
            return self.biases[:, self.idxs]
        key = str(device)
        if key not in self.cache:
            self.cache[key] = self.biases[:, self.idxs]
        return self.cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.get_bias(x.device)
        out = x
        for head in range(3):
            out = out + bias[head]
        return out


class _AttributeHeld(nn.Module):
    """A plain tensor attribute fed straight into an op with no other operand."""

    def __init__(self) -> None:
        super().__init__()
        self.table = torch.randn(5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.relu(self.table)


class _CreatedInForward(nn.Module):
    """A tensor created inside forward, stored in a dict, and read back."""

    def __init__(self) -> None:
        super().__init__()
        self.scratch: dict[str, Any] = {"note": "not a tensor"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.scratch["t"] = torch.ones(5, 5)
        return x + torch.relu(self.scratch["t"])


class _DictHeldMutated(nn.Module):
    """A dict-held tensor mutated in place inside forward, then read."""

    def __init__(self) -> None:
        super().__init__()
        self.state: dict[str, torch.Tensor] = {"count": torch.zeros(5, 5)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.state["count"].add_(1.0)
        return x + self.state["count"][0]


def _warm(model: nn.Module, x: torch.Tensor) -> nn.Module:
    model.eval()
    with torch.no_grad():
        model(x)
    return model


def _eager(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return copy.deepcopy(model)(x)


def _structure(trace: Any) -> list[tuple[str, str, list[str]]]:
    return [(op.layer_label, op.type, list(op.parents)) for op in trace.layer_list]


def _buffer_addresses(trace: Any) -> list[str]:
    return [trace.layer_dict_all_keys[label].address for label in trace.buffer_layers]


def test_warm_dict_cached_tensor_roots_at_a_buffer_source() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 5)
    model = _warm(_DictCachedBias(), x)
    eager_out = _eager(model, x)

    trace = tl.trace(copy.deepcopy(model), x)
    assert _buffer_addresses(trace) == ["cache.cpu"]
    cache_label = trace.buffer_layers[0]
    getitems = [op for op in trace.layer_list if op.type == "getitem"]
    assert len(getitems) == 3
    for op in getitems:
        assert list(op.parents) == [cache_label]
    assert torch.equal(trace[trace.output_layers[0]].out, eager_out)
    assert tl.validate(copy.deepcopy(model), x, scope="forward") is True


def test_cold_dict_cache_filled_inside_forward_is_not_module_state() -> None:
    # Cold: the cache is empty at preparation and filled by an op inside the
    # forward, so the cached value is an ordinary logged op output, never a buffer.
    torch.manual_seed(0)
    x = torch.randn(2, 5, 5)
    model = _DictCachedBias().eval()
    trace = tl.trace(copy.deepcopy(model), x)
    assert _buffer_addresses(trace) == ["idxs"]
    assert [op.type for op in trace.layer_list] == [
        "input",
        "buffer",
        "getitem",
        "getitem",
        "add",
        "getitem",
        "add",
        "getitem",
        "add",
        "output",
    ]


def test_attribute_held_tensor_fed_straight_into_an_op() -> None:
    torch.manual_seed(0)
    x = torch.randn(5, 5)
    model = _AttributeHeld()
    trace = tl.trace(copy.deepcopy(model), x)
    assert _structure(trace) == [
        ("input_1", "input", []),
        ("buffer_1", "buffer", []),
        ("relu_1_1", "relu", ["buffer_1"]),
        ("add_1_2", "add", ["input_1", "relu_1_1"]),
        ("output_1", "output", ["add_1_2"]),
    ]
    assert _buffer_addresses(trace) == ["table"]
    assert torch.equal(trace[trace.output_layers[0]].out, _eager(model, x))
    assert tl.validate(copy.deepcopy(model), x, scope="forward") is True


def test_tensor_created_in_forward_is_not_misclassified_as_module_state() -> None:
    x = torch.randn(5, 5)
    model = _CreatedInForward()
    trace = tl.trace(model, x)
    assert list(trace.buffer_layers) == []
    assert _structure(trace) == [
        ("input_1", "input", []),
        ("ones_1_1", "ones", []),
        ("relu_1_2", "relu", ["ones_1_1"]),
        ("add_1_3", "add", ["input_1", "relu_1_2"]),
        ("output_1", "output", ["add_1_3"]),
    ]
    # A second capture now finds the stored tensor in module state before the
    # forward, but the forward replaces it before reading, so it is still never read
    # as a buffer.
    second = tl.trace(model, x)
    assert list(second.buffer_layers) == []
    assert _structure(second) == _structure(trace)


def test_dict_held_tensor_mutated_in_place_matches_eager_and_validates() -> None:
    x = torch.randn(5, 5)
    model = _DictHeldMutated()
    eager_model = copy.deepcopy(model)
    with torch.no_grad():
        eager_out = eager_model(x)

    traced_model = copy.deepcopy(model)
    trace = tl.trace(traced_model, x)
    assert "state.count" in _buffer_addresses(trace)
    add_ = next(op for op in trace.layer_list if op.type == "add" and op.parents)
    assert trace.layer_dict_all_keys[add_.parents[0]].type == "buffer"
    assert torch.equal(trace[trace.output_layers[0]].out, eager_out)
    # Capture mutates the held tensor once, as one eager forward does.
    assert torch.equal(traced_model.state["count"], eager_model.state["count"])
    assert tl.validate(copy.deepcopy(model), x, scope="forward") is True


def test_iter_module_held_plain_tensors_holder_shapes_and_unique_names() -> None:
    from torchlens.backends.torch.buffer_writes import iter_module_held_plain_tensors

    module = nn.Linear(2, 2)
    plain = torch.zeros(1)
    listed = torch.ones(1)
    by_str = torch.full((1,), 2.0)
    by_int = torch.full((1,), 3.0)
    clash = torch.full((1,), 4.0)
    by_tuple = torch.full((1,), 5.0)
    module.plain = plain
    module.listed = [listed, "skip", nn.Parameter(torch.zeros(1))]
    module.cache = {"cpu": by_str, 1: by_int, "1": clash, (0, 1): by_tuple, "p": 7}
    module._private = {"hidden": torch.zeros(1)}
    module.tl_meta = torch.zeros(1)
    found = list(iter_module_held_plain_tensors(module))
    names = [name for name, _ in found]
    assert names == ["plain", "listed.0", "cache.cpu", "cache.1", "cache.(0, 1)"]
    tensors = dict(found)
    assert tensors["plain"] is plain
    assert tensors["listed.0"] is listed
    assert tensors["cache.cpu"] is by_str
    # The first key rendering "1" wins; the repeated rendering is skipped.
    assert tensors["cache.1"] is by_int
    assert tensors["cache.(0, 1)"] is by_tuple
    # Registered parameters/buffers live in the private registries, never here.
    assert "weight" not in names and "bias" not in names


@pytest.mark.slow
def test_warm_timm_efficientvit_m1_captures_and_validates() -> None:
    timm = pytest.importorskip("timm")
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    model = _warm(timm.create_model("efficientvit_m1", pretrained=False), x)
    eager_out = _eager(model, x)
    trace = tl.trace(copy.deepcopy(model), x)
    cache_addresses = [a for a in _buffer_addresses(trace) if "attention_bias_cache" in a]
    assert cache_addresses
    assert all(address.endswith("attention_bias_cache.cpu") for address in cache_addresses)
    assert torch.allclose(trace[trace.output_layers[0]].out, eager_out)
    assert tl.validate(copy.deepcopy(model), x, scope="forward") is True
