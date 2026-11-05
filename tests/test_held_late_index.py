"""Pre-forward model-owned tensors the preparation scan leaves unnamed are buffer sources.

The preparation scan stamps registered buffers and the public plain tensors a module
holds. Model state reaches further: a private attribute (torchvision's anchor
generator reads ``self._wh_pairs[k]``, neuraloperator's grid embedding caches
``self._grid``), a dict cache on a plain helper object (CroCo's position getter), and
a plain holder inside a plain holder (``Independent(Normal(loc, scale))``). Each of
those tensors exists before the forward, so its first consumer was recorded as a
parentless op with an unattributed tensor argument and metadata validation failed it
as a dangling node. Such a tensor is now late-indexed as a buffer source at first
use, under an address naming where the model holds it. A tensor first seen
mid-forward, and a tensor reachable only from a forward closure, keep their own
treatment.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import torch
from _stale_holders import OpaqueCallable
from torch import nn

import torchlens as tl


def _buffer_addresses(trace: Any) -> list[str]:
    return [trace._layer_dict_all_keys[label].address for label in trace.buffer_layers]


def _parent_types(trace: Any, op_type: str) -> list[list[str]]:
    return [
        [trace._layer_dict_all_keys[parent].type for parent in op.parents]
        for op in trace._layer_list
        if op.type == op_type
    ]


def _warm(model: nn.Module, x: torch.Tensor) -> nn.Module:
    model.eval()
    with torch.no_grad():
        model(x)
    return model


class _PrivateAnchorPairs(nn.Module):
    """torchvision-anchor-shaped: a private list of tensors built in ``__init__``."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self._wh_pairs = [torch.rand(3, 4), torch.rand(3, 4)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        for pair in self._wh_pairs:
            out = out + pair.clamp(min=0, max=1)
        return out


def test_private_list_attribute_tensors_are_buffer_sources() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    model = _PrivateAnchorPairs()
    trace = tl.trace(copy.deepcopy(model), x)
    assert sorted(_buffer_addresses(trace)) == ["_wh_pairs[0]", "_wh_pairs[1]"]
    assert _parent_types(trace, "clamp") == [["buffer"], ["buffer"]]
    assert tl.validate(model, x, scope="forward") is True


class _GridEmbedding(nn.Module):
    """neuraloperator-shaped: a private grid cache filled by the first forward."""

    def __init__(self) -> None:
        super().__init__()
        self._grid: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._grid is None:
            self._grid = torch.linspace(0, 1, x.shape[-1]).unsqueeze(0)
        return torch.cat([x, self._grid.repeat(x.shape[0], 1)], dim=0)


class _GridModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = _GridEmbedding()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.embed(x))


def test_warm_private_cache_on_a_submodule_is_a_buffer_source() -> None:
    x = torch.randn(2, 5)
    cold = tl.trace(_GridModel(), x)
    assert _buffer_addresses(cold) == []  # filled inside the capture: an ordinary op
    model = _warm(_GridModel(), x)
    trace = tl.trace(copy.deepcopy(model), x)
    assert _buffer_addresses(trace) == ["embed._grid"]
    assert _parent_types(trace, "repeat") == [["buffer"]]
    assert tl.validate(model, x, scope="forward") is True


class _PositionGetter:
    """CroCo-shaped: a plain (non-Module) helper holding a dict cache of positions."""

    def __init__(self) -> None:
        self.cache_positions: dict[tuple[int, int], torch.Tensor] = {}

    def __call__(self, h: int, w: int) -> torch.Tensor:
        if (h, w) not in self.cache_positions:
            self.cache_positions[h, w] = torch.cartesian_prod(torch.arange(h), torch.arange(w))
        return self.cache_positions[h, w]


class _PatchPositions(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.position_getter = _PositionGetter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.position_getter(2, 3).view(1, 6, 2).expand(x.shape[0], -1, 2)
        return x + pos.float()


def test_dict_cache_on_a_plain_helper_object_is_a_buffer_source() -> None:
    x = torch.randn(2, 6, 2)
    model = _warm(_PatchPositions(), x)
    trace = tl.trace(copy.deepcopy(model), x)
    assert _buffer_addresses(trace) == ["position_getter[cache_positions][(2, 3)]"]
    assert _parent_types(trace, "view") == [["buffer"]]
    assert tl.validate(model, x, scope="forward") is True


class _NestedDistribution(nn.Module):
    """flowtorch-shaped: a base distribution nested inside a plain holder."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        normal = torch.distributions.Normal(torch.zeros(4), torch.full((4,), 2.0))
        self.base = torch.distributions.Independent(normal, 1, validate_args=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.log_prob(self.fc(x))


def test_tensors_of_a_nested_plain_holder_are_buffer_sources() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    model = _NestedDistribution()
    trace = tl.trace(copy.deepcopy(model), x)
    assert set(_buffer_addresses(trace)) == {"base[base_dist][loc]", "base[base_dist][scale]"}
    assert all(op.parents for op in trace._layer_list if op.type not in {"input", "buffer"})
    assert tl.validate(model, x, scope="forward") is True


_CLOSURE_SCALE = torch.tensor(3.0)


class _ClosureScaled(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * _CLOSURE_SCALE


def test_forward_global_tensor_stays_a_constant_source() -> None:
    x = torch.randn(4)
    trace = tl.trace(_ClosureScaled(), x)
    assert _buffer_addresses(trace) == []
    assert "constant" in [op.type for op in trace._layer_list]


class _MidForwardEscape(nn.Module):
    """A tensor first made mid-forward by a stale torch reference is never module state."""

    def __init__(self, stale_mul: OpaqueCallable) -> None:
        super().__init__()
        self.stale_mul = stale_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.stale_mul(x, 2.0))


def test_tensor_first_seen_mid_forward_is_not_late_indexed() -> None:
    from torchlens import _state
    from torchlens.backends.torch.wrappers import wrap_torch

    wrap_torch()
    stale_mul = OpaqueCallable(_state._decorated_to_orig.get(id(torch.mul), torch.mul))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the escape's provenance warning is not under test
        trace = tl.trace(_MidForwardEscape(stale_mul), torch.randn(4))
    assert _buffer_addresses(trace) == []
