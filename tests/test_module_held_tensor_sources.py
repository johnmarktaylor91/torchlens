"""Module-held plain tensors read in ``forward`` are captured as buffer sources.

timm's EfficientViT (MSRA) keeps an eval-mode attention-bias cache: a plain tensor
in a dict attribute (``attention_bias_cache["cpu"]``), filled by the first forward
and indexed by ``getitem`` on every later one. On a warmed model the cached tensor
existed before the capture but carried no TorchLens provenance, so the ``getitem``
had no parent and validation called it a dangling node (``graph_connectivity``).
Model preparation already stamped plain tensor attributes and list/tuple items as
buffers; dict values and bounded nested list/tuple/dict values now get the same buffer
source (address ``<attr>[<key>]...``, the key a dot-free rendering of ``repr(key)``).
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
    assert _buffer_addresses(trace) == ["cache['cpu']"]
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
    assert "state['count']" in _buffer_addresses(trace)
    add_ = next(op for op in trace.layer_list if op.type == "add" and op.parents)
    assert trace.layer_dict_all_keys[add_.parents[0]].type == "buffer"
    assert torch.equal(trace[trace.output_layers[0]].out, eager_out)
    # Capture mutates the held tensor once, as one eager forward does.
    assert torch.equal(traced_model.state["count"], eager_model.state["count"])
    assert tl.validate(copy.deepcopy(model), x, scope="forward") is True


class _SameRepr:
    """A hashable key whose distinct instances all render the same."""

    def __repr__(self) -> str:
        return "k"


def test_iter_module_held_plain_tensors_holder_shapes_and_unique_names() -> None:
    from torchlens.backends.torch.buffer_writes import iter_module_held_plain_tensors

    module = nn.Linear(2, 2)
    plain = torch.zeros(1)
    listed = torch.ones(1)
    by_str = torch.full((1,), 2.0)
    by_int = torch.full((1,), 3.0)
    by_str_one = torch.full((1,), 4.0)
    by_tuple = torch.full((1,), 5.0)
    first_k, second_k = torch.full((1,), 6.0), torch.full((1,), 7.0)
    module.plain = plain
    module.listed = [listed, "skip", nn.Parameter(torch.zeros(1))]
    module.cache = {
        "cpu": by_str,
        1: by_int,
        "1": by_str_one,
        (0, 1): by_tuple,
        "p": 7,
        _SameRepr(): first_k,
        _SameRepr(): second_k,
    }
    module._private = {"hidden": torch.zeros(1)}
    module.tl_meta = torch.zeros(1)
    found = list(iter_module_held_plain_tensors(module))
    names = [name for name, _ in found]
    assert names == [
        "plain",
        "listed.0",
        "cache['cpu']",
        "cache[1]",
        "cache['1']",
        "cache[(0, 1)]",
        "cache[k]",
        "cache[k]#2",
    ]
    tensors = dict(found)
    assert tensors["plain"] is plain
    assert tensors["listed.0"] is listed
    assert tensors["cache['cpu']"] is by_str
    # Keys that print alike (1 and "1", two same-repr objects) both keep a source.
    assert tensors["cache[1]"] is by_int
    assert tensors["cache['1']"] is by_str_one
    assert tensors["cache[(0, 1)]"] is by_tuple
    assert tensors["cache[k]"] is first_k
    assert tensors["cache[k]#2"] is second_k
    # Registered parameters/buffers live in the private registries, never here.
    assert "weight" not in names and "bias" not in names


def test_held_key_rendering_is_dot_and_colon_free() -> None:
    from torchlens.backends.torch.buffer_writes import iter_module_held_plain_tensors

    module = nn.Module()
    module.cache = {"sub.cpu": torch.zeros(1), "a:b": torch.zeros(1), 1.5: torch.zeros(1)}
    module.cache["50%"] = torch.zeros(1)
    names = [name for name, _ in iter_module_held_plain_tensors(module)]
    assert names == ["cache['sub%2Ecpu']", "cache['a%3Ab']", "cache[1%2E5]", "cache['50%25']"]
    assert all("." not in name and ":" not in name for name in names)


def test_iter_module_held_plain_tensors_walks_nested_containers() -> None:
    from torchlens.backends.torch.buffer_writes import iter_module_held_plain_tensors

    module = nn.Module()
    deep = torch.zeros(1)
    cycle: list[Any] = [torch.ones(1)]
    cycle.append(cycle)
    module.cache = {"outer": {"cpu": deep}, "lists": [torch.ones(1), (torch.ones(1),)]}
    module.pairs = [(torch.ones(1), torch.ones(1))]
    module.cycle = cycle
    names = [name for name, _ in iter_module_held_plain_tensors(module)]
    assert names == [
        "cache['outer']['cpu']",
        "cache['lists'][0]",
        "cache['lists'][1][0]",
        "pairs.0[0]",
        "pairs.0[1]",
        "cycle.0",
    ]


def test_held_scan_depth_and_object_bounds_are_disclosed() -> None:
    from torchlens.backends.torch.buffer_writes import (
        _HELD_SCAN_MAX_DEPTH,
        _HELD_SCAN_MAX_OBJECTS,
        iter_module_held_plain_tensors,
    )

    assert _HELD_SCAN_MAX_DEPTH == 4
    module = nn.Module()
    # Four bracket levels are reached; the container at level four is not entered.
    module.deep = {"a": {"b": {"c": {"d": torch.zeros(1), "e": {"f": torch.zeros(1)}}}}}
    cut: list[str] = []
    names = [name for name, _ in iter_module_held_plain_tensors(module, cut)]
    assert names == ["deep['a']['b']['c']['d']"]
    assert cut == ["deep['a']['b']['c']['e'] (depth bound 4)"]

    wide = nn.Module()
    wide.cache = {index: torch.zeros(1) for index in range(_HELD_SCAN_MAX_OBJECTS + 3)}
    wide.vocab = {str(index): index for index in range(3 * _HELD_SCAN_MAX_OBJECTS)}
    cut = []
    names = [name for name, _ in iter_module_held_plain_tensors(wide, cut)]
    assert len(names) == _HELD_SCAN_MAX_OBJECTS
    assert names[-1] == f"cache[{_HELD_SCAN_MAX_OBJECTS - 1}]"
    assert cut == [f"cache[{_HELD_SCAN_MAX_OBJECTS}] (object bound {_HELD_SCAN_MAX_OBJECTS})"]

    # Scalar leaves are not counted: a large scalar-only dict never trips the bound.
    vocab_only = nn.Module()
    vocab_only.vocab = wide.vocab
    vocab_only.table = {"t": torch.zeros(1)}
    cut = []
    assert [name for name, _ in iter_module_held_plain_tensors(vocab_only, cut)] == ["table['t']"]
    assert cut == []


class _CollidingKeys(nn.Module):
    """Two cached tensors under keys that printed alike in the old rendering."""

    def __init__(self, first: Any, second: Any) -> None:
        super().__init__()
        self.first, self.second = first, second
        self.cache: dict[Any, torch.Tensor] = {first: torch.randn(5), second: torch.randn(5)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cache[self.first][0] + self.cache[self.second][1]


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (1, "1", ["cache[1]", "cache['1']"]),
        (_SameRepr(), _SameRepr(), ["cache[k]", "cache[k]#2"]),
    ],
    ids=["int_and_str", "same_repr"],
)
def test_colliding_key_renderings_each_get_a_buffer_source(
    first: Any, second: Any, expected: list[str]
) -> None:
    torch.manual_seed(0)
    x = torch.randn(5)
    model = _CollidingKeys(first, second)
    trace = tl.trace(model, x)
    assert sorted(_buffer_addresses(trace)) == sorted(expected)
    getitems = [op for op in trace.layer_list if op.type == "getitem"]
    assert len(getitems) == 2
    assert all(len(op.parents) == 1 for op in getitems)
    assert {trace.layer_dict_all_keys[op.parents[0]].type for op in getitems} == {"buffer"}
    assert torch.equal(trace[trace.output_layers[0]].out, _eager(model, x))
    assert tl.validate(model, x, scope="forward") is True


class _NestedHeld(nn.Module):
    """A dict of dicts and a dict of lists, both read in forward."""

    def __init__(self) -> None:
        super().__init__()
        self.cache: dict[str, Any] = {
            "outer": {"cpu": torch.randn(3, 5)},
            "lists": [torch.randn(5), (torch.randn(5),)],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lists = self.cache["lists"]
        return x + self.cache["outer"]["cpu"][1] + lists[0] * lists[1][0]


def test_nested_held_tensors_root_at_buffer_sources_and_validate() -> None:
    torch.manual_seed(0)
    x = torch.randn(5)
    model = _NestedHeld()
    trace = tl.trace(model, x)
    assert sorted(_buffer_addresses(trace)) == sorted(
        ["cache['outer']['cpu']", "cache['lists'][0]", "cache['lists'][1][0]"]
    )
    assert all(op.parents for op in trace.layer_list if op.type not in {"input", "buffer"})
    assert torch.equal(trace[trace.output_layers[0]].out, _eager(model, x))
    assert tl.validate(model, x, scope="forward") is True


class _TooDeep(nn.Module):
    """A tensor held past the depth bound and never read: disclosed, graph unchanged."""

    def __init__(self) -> None:
        super().__init__()
        self.deep = {"a": {"b": {"c": {"d": {"e": torch.zeros(1)}}}}}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


def test_depth_bound_cut_is_disclosed_at_capture() -> None:
    x = torch.randn(5)
    with pytest.warns(UserWarning, match=r"stopped scanning module-held containers"):
        trace = tl.trace(_TooDeep(), x)
    assert list(trace.buffer_layers) == []
    assert [op.type for op in trace.layer_list] == ["input", "relu", "output"]


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cache = {"sub.cpu": torch.randn(5)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cache["sub.cpu"]


class _Outer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = _Inner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.inner(x))


def test_dotted_key_address_attributes_the_buffer_to_its_module() -> None:
    torch.manual_seed(0)
    x = torch.randn(5)
    model = _Outer()
    trace = tl.trace(model, x)
    address = "inner.cache['sub%2Ecpu']"
    assert _buffer_addresses(trace) == [address]
    assert trace.buffers[address].module_address == "inner"
    assert trace.buffers[address].name == "cache['sub%2Ecpu']"
    assert list(trace.modules["inner"].buffer_layers) == list(trace.buffer_layers)
    assert tl.validate(model, x, scope="forward") is True


def test_runnable_save_refuses_a_dict_held_source_typed(tmp_path: Any) -> None:
    # Pre-existing limit: a module-held plain tensor has no captured state-dict slot or
    # initializer recipe, so the runnable tier refuses it typed instead of replaying it.
    from torchlens.errors import RunnablePreflightError

    torch.manual_seed(0)
    x = torch.randn(2, 5, 5)
    trace = tl.trace(_warm(_DictCachedBias(), x), x)
    with pytest.raises(RunnablePreflightError, match="unsupported_tensor_constant"):
        trace.save(tmp_path / "held.tlspec", level="runnable")


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
    assert all(address.endswith(".attention_bias_cache['cpu']") for address in cache_addresses)
    assert torch.allclose(trace[trace.output_layers[0]].out, eager_out)
    assert tl.validate(copy.deepcopy(model), x, scope="forward") is True
