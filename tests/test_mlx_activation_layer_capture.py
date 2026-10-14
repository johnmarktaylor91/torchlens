"""MLX activation layers are captured, and tracing leaves lazy models usable.

Two defects the preview conformance pack found. The activation layers
(``nn.ReLU``, ``nn.GELU``, ...) call the functions of
``mlx.nn.layers.activations`` through that module's globals, most of them
``mx.compile``'d, and only the ``mlx.nn`` attributes were wrapped: the layer's
op was silently missing from the capture, and while ``mx.compile`` traced the
body the wrapped ``mx`` primitives logged tracer arrays, so a model whose
parameters were still unevaluated could not run again after the trace.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends.mlx.wrappers import (  # noqa: E402
    mlx_activation_function_names,
    mlx_activations_module,
)

pytestmark = [pytest.mark.backend_mlx, pytest.mark.heavy]

#: Constructor arguments for activation layers that need them. A layer MLX
#: adds that cannot be built with no arguments fails the coverage test until
#: its arguments are added here, so no layer goes unchecked.
_LAYER_ARGS: dict[str, dict[str, Any]] = {}


def _op_kinds(trace: Any) -> list[str]:
    """The captured ops' kinds in capture order."""

    ops = trace.ops
    rows = ops.values() if hasattr(ops, "values") else ops
    return [str(getattr(op, "func_name", "")) for op in rows]


def _activation_layer_classes() -> list[tuple[str, type[Any]]]:
    """Every nn.Module subclass mlx.nn.layers.activations defines."""

    activations = mlx_activations_module(nn)
    return sorted(
        (name, value)
        for name, value in vars(activations).items()
        if inspect.isclass(value)
        and issubclass(value, nn.Module)
        and value.__module__ == activations.__name__
    )


def test_activation_function_set_is_derived_and_nonempty() -> None:
    """The wrapped set comes from the module itself and includes relu."""

    names = mlx_activation_function_names(mlx_activations_module(nn))
    assert "relu" in names
    assert len(names) >= 10, names


@pytest.mark.parametrize(
    "name", [name for name, _ in _activation_layer_classes()], ids=lambda name: name
)
def test_every_activation_layer_records_an_op(name: str) -> None:
    """Each activation layer MLX ships records at least one op when traced."""

    layer_cls = dict(_activation_layer_classes())[name]
    layer = layer_cls(**_LAYER_ARGS.get(name, {}))
    x = mx.array([[-1.5, -0.25, 0.5, 2.0]])
    trace = tl.trace(layer, x, backend="mlx")
    assert trace.num_ops >= 1, f"nn.{name} recorded no op: {list(trace.layer_labels)}"


def test_sequential_relu_layer_is_captured() -> None:
    """Sequential(Linear, ReLU, Linear) records all three ops in order."""

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    mx.eval(model.parameters())
    trace = tl.trace(model, mx.ones((2, 4)), backend="mlx")
    assert _op_kinds(trace) == ["linear", "relu", "linear"]


def test_lazy_parameter_model_stays_usable_after_trace() -> None:
    """Tracing before the parameters are evaluated leaves the model runnable."""

    mx.random.seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    x = mx.array([[-1.0, -0.5, 0.25, 1.0], [0.5, 1.5, -2.0, 0.75]])
    trace = tl.trace(model, x, backend="mlx")
    direct = model(x)
    mx.eval(direct)
    captured = trace.output_ops[0].out
    assert mx.array_equal(captured, direct).item()
