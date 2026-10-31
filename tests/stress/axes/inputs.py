"""Input-shape axis: how a model is fed, with the small models that accept each shape.

An ``InputSpec`` row names the models it fits; the ``default`` row (``make=None``)
fits every model and uses the model's own example input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..generator import Inputs, InputSpec, ModelSpec

FLOAT_DTYPES = ("f32", "f16", "bf16", "f64")
WIDTH = 8


class Mlp(nn.Module):
    """Two-layer MLP with a keyword scale, fed positionally by default."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(WIDTH, WIDTH)
        self.second = nn.Linear(WIDTH, 4)

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        return self.second(torch.relu(self.first(x))) * scale


@dataclass
class Batch:
    """Dataclass input carrying the features and a mask."""

    x: torch.Tensor
    mask: torch.Tensor


class ContainerNet(nn.Module):
    """Accepts a tensor, a tuple, a dict or a ``Batch`` and reads the features from it."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)

    def forward(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            x, mask = batch, None
        elif isinstance(batch, dict):
            x, mask = batch["x"], batch.get("mask")
        elif isinstance(batch, Batch):
            x, mask = batch.x, batch.mask
        else:
            x, mask = batch
        y = self.linear(x)
        if mask is not None:
            y = torch.where(mask, y, torch.zeros_like(y))
        return y


class ScalarNet(nn.Module):
    """Affine map that accepts any shape, including a rank-0 tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.5))
        self.bias = nn.Parameter(torch.tensor(-0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x * self.weight + self.bias)


class MixedNet(nn.Module):
    """Float features with optional integer ids and an optional bool mask."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, WIDTH)
        self.linear = nn.Linear(WIDTH, 4)

    def forward(
        self, x: torch.Tensor, ids: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if ids is not None:
            x = x + self.embed(ids)
        if mask is not None:
            x = torch.where(mask, x, torch.zeros_like(x))
        return self.linear(x)


def _features(batch: int = 3) -> torch.Tensor:
    return torch.randn(batch, WIDTH)


def _mlp_input(_model: nn.Module) -> Inputs:
    return ((_features(),), {})


def _kwargs_only(_model: nn.Module) -> Inputs:
    return ((), {"x": _features(), "scale": 2.0})


def _empty_batch(_model: nn.Module) -> Inputs:
    return ((torch.empty(0, WIDTH),), {})


def _noncontiguous(_model: nn.Module) -> Inputs:
    return ((torch.randn(WIDTH, 3).t(),), {})


def _nested_tuple(_model: nn.Module) -> Inputs:
    return (((_features(), torch.rand(3, WIDTH) > 0.5),), {})


def _nested_dict(_model: nn.Module) -> Inputs:
    return (({"x": _features(), "mask": torch.rand(3, WIDTH) > 0.5},), {})


def _dataclass(_model: nn.Module) -> Inputs:
    return ((Batch(x=_features(), mask=torch.rand(3, WIDTH) > 0.5),), {})


def _vector(_model: nn.Module) -> Inputs:
    return ((torch.randn(3),), {})


def _rank0(_model: nn.Module) -> Inputs:
    return ((torch.tensor(0.5),), {})


def _int_ids(_model: nn.Module) -> Inputs:
    return ((_features(), torch.randint(0, 16, (3,))), {})


def _bool_mask(_model: nn.Module) -> Inputs:
    return ((_features(),), {"mask": torch.rand(3, WIDTH) > 0.5})


def _model(row_id: str, factory: type[nn.Module], example: Any, reason: str) -> ModelSpec:
    return ModelSpec(
        id=row_id,
        build=factory,
        example_input=example,
        reason=reason,
        dtypes=FLOAT_DTYPES,
        tags=frozenset({"inputs"}),
    )


MODELS: tuple[ModelSpec, ...] = (
    _model("in_mlp", Mlp, _mlp_input, "plain MLP fed in several shapes by the input rows"),
    _model("in_container", ContainerNet, _mlp_input, "reads features from tensor or container"),
    _model("in_scalar", ScalarNet, _vector, "shape-agnostic affine map, carries the rank-0 row"),
    _model("in_mixed", MixedNet, _mlp_input, "float features plus optional int ids and bool mask"),
)

INPUTS: tuple[InputSpec, ...] = (
    InputSpec("default", None, "the model's own example input"),
    InputSpec(
        "kwargs_only",
        _kwargs_only,
        "no positional input; tensor and a float by keyword",
        ("in_mlp",),
    ),
    InputSpec("empty_batch", _empty_batch, "zero-row batch through linear layers", ("in_mlp",)),
    InputSpec(
        "noncontiguous_view", _noncontiguous, "transposed view as the input tensor", ("in_mlp",)
    ),
    InputSpec(
        "nested_tuple", _nested_tuple, "one positional tuple holding two tensors", ("in_container",)
    ),
    InputSpec(
        "nested_dict", _nested_dict, "dict input with a tensor and a bool mask", ("in_container",)
    ),
    InputSpec(
        "dataclass_input", _dataclass, "a dataclass instance as the input", ("in_container",)
    ),
    InputSpec("rank0", _rank0, "a rank-0 tensor as the only input", ("in_scalar",)),
    InputSpec("int_input", _int_ids, "int64 ids beside the float features", ("in_mixed",)),
    InputSpec("bool_input", _bool_mask, "a bool mask passed by keyword", ("in_mixed",)),
)
