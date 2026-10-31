"""Dtype axis on CPU: cast rows, plus the models that carry the non-float rows.

A cast row is a ``DtypeSpec`` whose ``cast`` rewrites the built model and its
inputs. Float rows cast the model's floating parameters and buffers and every
floating input tensor. Integer and bool rows cast only the floating inputs
(the carrying models have no parameters). The complex row casts both.

The corpus sample (the classics smoke tier) joins here as model rows carrying
the float casts, so every corpus entry runs at f16, bf16 and f64 as well.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from tests.classics_corpus._loader import CorpusEntry, build_entry, corpus_entries
from torchlens.utils.arg_handling import normalize_input_args

from ..generator import DtypeSpec, Inputs, ModelSpec

Cast = Callable[[nn.Module, Inputs], tuple[nn.Module, Inputs]]


def _map_floating(value: Any, convert: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Apply ``convert`` to every floating tensor inside ``value``."""

    if isinstance(value, torch.Tensor):
        return convert(value) if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_map_floating(item, convert) for item in value)
    if isinstance(value, list):
        return [_map_floating(item, convert) for item in value]
    if isinstance(value, dict):
        return {key: _map_floating(item, convert) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            setattr(value, name, _map_floating(getattr(value, name), convert))
        return value
    return value


def _cast_inputs(inputs: Inputs, convert: Callable[[torch.Tensor], torch.Tensor]) -> Inputs:
    args, kwargs = inputs
    return _map_floating(args, convert), _map_floating(kwargs, convert)


def _identity(model: nn.Module, inputs: Inputs) -> tuple[nn.Module, Inputs]:
    return model, inputs


def _float_cast(dtype: torch.dtype) -> Cast:
    def cast(model: nn.Module, inputs: Inputs) -> tuple[nn.Module, Inputs]:
        return model.to(dtype), _cast_inputs(inputs, lambda t: t.to(dtype))

    return cast


def _inputs_only_cast(convert: Callable[[torch.Tensor], torch.Tensor]) -> Cast:
    def cast(model: nn.Module, inputs: Inputs) -> tuple[nn.Module, Inputs]:
        return model, _cast_inputs(inputs, convert)

    return cast


DTYPES: tuple[DtypeSpec, ...] = (
    DtypeSpec("f32", _identity, "the identity row every model carries"),
    DtypeSpec("f16", _float_cast(torch.float16), "half precision on CPU kernels"),
    DtypeSpec("bf16", _float_cast(torch.bfloat16), "bfloat16 on CPU kernels"),
    DtypeSpec(
        "f64", _float_cast(torch.float64), "double precision, wider than the capture default"
    ),
    DtypeSpec(
        "complex64", _float_cast(torch.complex64), "complex parameters and inputs through linear"
    ),
    DtypeSpec("bool", _inputs_only_cast(lambda t: t > 0), "bool inputs into integer promotion"),
    DtypeSpec("int8", _inputs_only_cast(lambda t: (t * 4).to(torch.int8)), "signed 8-bit inputs"),
    DtypeSpec(
        "uint8", _inputs_only_cast(lambda t: (t.abs() * 4).to(torch.uint8)), "unsigned 8-bit inputs"
    ),
)

FLOAT_ROWS = ("f32", "f16", "bf16", "f64")


class IntegerOps(nn.Module):
    """Parameter-free arithmetic every integer and bool dtype supports."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x + 1
        y = y * 2 - x
        return y.sum(dim=-1), y.max()


class ComplexLinear(nn.Module):
    """A linear layer that is cast to complex64 by its dtype row."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(self.linear(x)).sum(dim=-1)


def _vector_input(_model: nn.Module) -> Inputs:
    return ((torch.randn(3, 8),), {})


def _corpus_row(entry: CorpusEntry) -> ModelSpec:
    """Wrap one classics smoke entry as a model row carrying the float casts."""

    pending: dict[str, Any] = {}

    def build() -> nn.Module:
        model, inputs = build_entry(entry)
        pending["inputs"] = inputs
        return model

    def example_input(model: nn.Module) -> Inputs:
        inputs = pending.pop("inputs", None)
        if inputs is None:
            inputs = build_entry(entry)[1]
        return tuple(normalize_input_args(inputs, model)), {}

    slug = re.sub(r"[^a-z0-9]+", "_", entry.id.lower()).strip("_")
    return ModelSpec(
        id=f"corpus_{slug}",
        build=build,
        example_input=example_input,
        reason=f"classics smoke entry: {entry.why}"[:200],
        dtypes=FLOAT_ROWS,
        cost=2,
        tags=frozenset({"corpus", *entry.features}),
    )


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="dt_integer_ops",
        build=IntegerOps,
        example_input=_vector_input,
        reason="parameter-free arithmetic that bool, int8 and uint8 inputs all support",
        dtypes=("f32", "bool", "int8", "uint8"),
        tags=frozenset({"dtype"}),
    ),
    ModelSpec(
        id="dt_complex_linear",
        build=ComplexLinear,
        example_input=_vector_input,
        reason="complex64 parameters and inputs through linear and view_as_real",
        dtypes=("f32", "complex64"),
        tags=frozenset({"dtype"}),
    ),
    *(_corpus_row(entry) for entry in corpus_entries("smoke")),
)
