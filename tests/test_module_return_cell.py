"""Backend-keyed conformance cell: recorded module outputs against each call's actual return.

Every backend that records module returns (``torchlens.backends._module_returns``) runs the
same cases here, live and after save and load:

- ``single``: one computed output; ``Module.out`` equals it.
- ``identity``: the module returns its input unchanged.
- ``tuple`` and ``nested``: container returns; ``output_paths`` names every leaf.
- ``duplicate``: the same computed leaf returned twice.
- ``returned_input``: the input returned beside a computed output.
- ``repeated``: two calls of one module, each checked against its own return.
- ``unused``: a child that is never called gets no output.
- ``unconsumed``: a module output that nothing downstream reads is still recorded; on a lazy
  backend that never computes it (kit ``computes_unread_returns = False``: tinygrad) the call is
  left unrecorded and disclosed with a ``CaptureGapWarning``, never fabricated.

The ground truth is the value the module call actually returned: the parent module stores
it during the traced forward, and every recorded output must match it by value.

Adding a backend: write a kit class with the members of :class:`TorchKit` (``build_module``
wraps a Python callable as one of the framework's modules; ``trace`` runs a TorchLens capture;
``inserts_identity`` says whether the backend inserts an identity op for a returned input, as
torch does, or records the passthrough on the input's op; ``computes_unread_returns`` says
whether a returned value nothing reads is computed; ``saves`` is False only for a backend whose
saved traces are known broken, which then runs the live half alone), then add it to ``KITS``
with its markers. The kit imports its framework with ``pytest.importorskip`` inside ``load``.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends._module_returns import (
    ModuleReturnLedger,
    iter_returned_leaves,
    returned_op_labels,
)
from torchlens.errors import CaptureGapWarning, InvalidArgumentError
from torchlens.intervention.errors import MultiOutputModuleError
from torchlens.validation import check_metadata_invariants

CASES = (
    "single",
    "identity",
    "tuple",
    "nested",
    "duplicate",
    "returned_input",
    "repeated",
    "unused",
    "unconsumed",
)
_PASSTHROUGH_CASES = frozenset({"identity", "returned_input"})
_WIDTH = 4


class TorchKit:
    """Kit for the torch backend (the reference representation)."""

    name = "torch"
    inserts_identity = True
    computes_unread_returns = True
    saves = True

    def load(self) -> None:
        """Import the framework (torch is a hard dependency)."""

        import torch

        self.torch = torch
        torch.manual_seed(0)

    def build_module(self, call: Callable[..., Any], **children: Any) -> Any:
        """Wrap ``call(self, *args)`` as a framework module holding ``children``."""

        nn = self.torch.nn

        class _Cell(nn.Module):
            """Cell module whose forward is the case callable."""

            def __init__(self) -> None:
                """Register the children."""

                super().__init__()
                for name, child in children.items():
                    setattr(self, name, child)

            def forward(self, *args: Any) -> Any:
                """Run the case callable."""

                return call(self, *args)

        return _Cell()

    def linear(self) -> Any:
        """Return a square linear layer."""

        return self.torch.nn.Linear(_WIDTH, _WIDTH)

    def relu(self, value: Any) -> Any:
        """Return the rectified value."""

        return self.torch.relu(value)

    def add(self, left: Any, right: Any) -> Any:
        """Return the elementwise sum."""

        return self.torch.add(left, right)

    def input(self) -> Any:
        """Return a deterministic model input."""

        return self.torch.linspace(-1.0, 1.0, 2 * _WIDTH).reshape(2, _WIDTH)

    def is_leaf(self, value: object) -> bool:
        """Return whether ``value`` is a native array."""

        return isinstance(value, self.torch.Tensor)

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert a native array to numpy."""

        return value.detach().cpu().numpy()

    def set_training(self, model: Any, training: bool) -> None:
        """Put ``model`` in train or eval mode."""

        model.train(training)

    def training(self, module: Any) -> bool:
        """Return the module's own training flag."""

        return bool(module.training)

    def trace(self, model: Any, value: Any) -> Any:
        """Capture ``model(value)``."""

        return tl.trace(model, value)


class MLXKit(TorchKit):
    """Kit for the MLX preview backend."""

    name = "mlx"
    inserts_identity = False

    def load(self) -> None:
        """Import MLX or skip."""

        self.mx = pytest.importorskip("mlx.core")
        self.nn = pytest.importorskip("mlx.nn")
        self.mx.random.seed(0)

    def build_module(self, call: Callable[..., Any], **children: Any) -> Any:
        """Wrap ``call(self, *args)`` as an MLX module holding ``children``."""

        nn = self.nn

        class _Cell(nn.Module):
            """Cell module whose ``__call__`` is the case callable."""

            def __init__(self) -> None:
                """Register the children."""

                super().__init__()
                for name, child in children.items():
                    setattr(self, name, child)

            def __call__(self, *args: Any) -> Any:
                """Run the case callable."""

                return call(self, *args)

        return _Cell()

    def linear(self) -> Any:
        """Return a square linear layer."""

        return self.nn.Linear(_WIDTH, _WIDTH)

    def relu(self, value: Any) -> Any:
        """Return the rectified value."""

        return self.nn.relu(value)

    def add(self, left: Any, right: Any) -> Any:
        """Return the elementwise sum."""

        return self.mx.add(left, right)

    def input(self) -> Any:
        """Return a deterministic model input."""

        return self.mx.linspace(-1.0, 1.0, 2 * _WIDTH).reshape(2, _WIDTH)

    def is_leaf(self, value: object) -> bool:
        """Return whether ``value`` is a native array."""

        return isinstance(value, self.mx.array)

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert a native array to numpy."""

        return np.asarray(value)

    def trace(self, model: Any, value: Any) -> Any:
        """Capture ``model(value)`` on the MLX backend."""

        return tl.trace(model, value, backend="mlx")


class PaddleKit(TorchKit):
    """Kit for the Paddle preview backend."""

    name = "paddle"
    inserts_identity = False

    def load(self) -> None:
        """Import Paddle or skip."""

        self.paddle = pytest.importorskip("paddle")
        self.paddle.seed(0)

    def build_module(self, call: Callable[..., Any], **children: Any) -> Any:
        """Wrap ``call(self, *args)`` as a Paddle layer holding ``children``."""

        nn = self.paddle.nn

        class _Cell(nn.Layer):
            """Cell layer whose forward is the case callable."""

            def __init__(self) -> None:
                """Register the children."""

                super().__init__()
                for name, child in children.items():
                    setattr(self, name, child)

            def forward(self, *args: Any) -> Any:
                """Run the case callable."""

                return call(self, *args)

        return _Cell()

    def linear(self) -> Any:
        """Return a square linear layer."""

        return self.paddle.nn.Linear(_WIDTH, _WIDTH)

    def relu(self, value: Any) -> Any:
        """Return the rectified value."""

        return self.paddle.nn.functional.relu(value)

    def add(self, left: Any, right: Any) -> Any:
        """Return the elementwise sum."""

        return self.paddle.add(left, right)

    def input(self) -> Any:
        """Return a deterministic model input."""

        return self.paddle.linspace(-1.0, 1.0, 2 * _WIDTH, dtype="float32").reshape([2, _WIDTH])

    def is_leaf(self, value: object) -> bool:
        """Return whether ``value`` is a native array."""

        return isinstance(value, self.paddle.Tensor)

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert a native array (or a loaded payload) to numpy."""

        return value.numpy() if hasattr(value, "numpy") else np.asarray(value)

    def set_training(self, model: Any, training: bool) -> None:
        """Put ``model`` in train or eval mode."""

        if training:
            model.train()
        else:
            model.eval()

    def trace(self, model: Any, value: Any) -> Any:
        """Capture ``model(value)`` on the Paddle backend."""

        return tl.trace(model, value, backend="paddle")


class TFKit(TorchKit):
    """Kit for the TensorFlow preview backend (Keras layers)."""

    name = "tf"
    inserts_identity = False

    def load(self) -> None:
        """Import TensorFlow and Keras or skip."""

        self.tf = pytest.importorskip("tensorflow")
        self.keras = pytest.importorskip("keras")
        self.tf.random.set_seed(0)

    def build_module(self, call: Callable[..., Any], **children: Any) -> Any:
        """Wrap ``call(self, inputs)`` as a Keras layer holding ``children``.

        TensorFlow addresses a Keras child by its layer name, so each child is renamed to its
        attribute name.
        """

        keras = self.keras

        class _Cell(keras.layers.Layer):
            """Cell layer whose ``call`` is the case callable."""

            def __init__(self) -> None:
                """Register the children under their attribute names."""

                super().__init__()
                for name, child in children.items():
                    _rename_keras_layer(child, name)
                    setattr(self, name, child)

            def call(self, inputs: Any) -> Any:
                """Run the case callable."""

                return call(self, inputs)

        return _Cell()

    def linear(self) -> Any:
        """Return a square dense layer."""

        return self.keras.layers.Dense(_WIDTH)

    def relu(self, value: Any) -> Any:
        """Return the rectified value."""

        return self.tf.nn.relu(value)

    def add(self, left: Any, right: Any) -> Any:
        """Return the elementwise sum."""

        return self.tf.add(left, right)

    def input(self) -> Any:
        """Return a deterministic model input."""

        return self.tf.reshape(self.tf.linspace(-1.0, 1.0, 2 * _WIDTH), (2, _WIDTH))

    def is_leaf(self, value: object) -> bool:
        """Return whether ``value`` is a native tensor."""

        return isinstance(value, self.tf.Tensor)

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert a native tensor (or a loaded payload) to numpy."""

        return value.numpy() if hasattr(value, "numpy") else np.asarray(value)

    def set_training(self, model: Any, training: bool) -> None:
        """Keras keeps no persistent training mode (it is a call argument); nothing to set."""

        del model, training

    def training(self, module: Any) -> bool:
        """Return the module's training attribute as the TF backend reads it."""

        return bool(getattr(module, "training", False))

    def trace(self, model: Any, value: Any) -> Any:
        """Capture ``model(value)`` on the TensorFlow backend."""

        return tl.trace(model, value, backend="tf")


def _rename_keras_layer(layer: Any, name: str) -> None:
    """Give a Keras layer the name TensorFlow will address it by."""

    try:
        layer.name = name
    except AttributeError:
        layer._name = name


class EquinoxKit(TorchKit):
    """Kit for the JAX preview backend on Equinox modules."""

    name = "equinox"
    inserts_identity = False

    def load(self) -> None:
        """Import JAX and Equinox or skip; define a batch-aware linear module."""

        self.jax = pytest.importorskip("jax")
        self.jnp = pytest.importorskip("jax.numpy")
        self.eqx = pytest.importorskip("equinox")
        self.keys = iter(range(1, 1000))
        jnp = self.jnp

        class _Linear(self.eqx.Module):  # type: ignore[name-defined, misc]
            """Square linear layer applied to the last axis (``eqx.nn.Linear`` is unbatched)."""

            weight: Any
            bias: Any

            def __call__(self, value: Any) -> Any:
                """Return ``value @ weight.T + bias``."""

                return jnp.matmul(value, self.weight.T) + self.bias

        self.linear_class = _Linear

    def build_module(self, call: Callable[..., Any], **children: Any) -> Any:
        """Wrap ``call(self, *args)`` as an Equinox module whose fields are ``children``."""

        def forward(module: Any, *args: Any) -> Any:
            """Run the case callable."""

            return call(module, *args)

        namespace = {"__annotations__": dict.fromkeys(children, Any), "__call__": forward}
        cell_class = type("_Cell", (self.eqx.Module,), namespace)
        return cell_class(**children)

    def linear(self) -> Any:
        """Return a square linear layer with its own random weights."""

        key = self.jax.random.PRNGKey(next(self.keys))
        weight = self.jax.random.normal(key, (_WIDTH, _WIDTH)) / _WIDTH
        return self.linear_class(weight=weight, bias=self.jnp.full((_WIDTH,), 0.1))

    def relu(self, value: Any) -> Any:
        """Return the rectified value."""

        return self.jnp.maximum(value, 0.0)

    def add(self, left: Any, right: Any) -> Any:
        """Return the elementwise sum."""

        return self.jnp.add(left, right)

    def input(self) -> Any:
        """Return a deterministic model input."""

        return self.jnp.linspace(-1.0, 1.0, 2 * _WIDTH).reshape(2, _WIDTH)

    def is_leaf(self, value: object) -> bool:
        """Return whether ``value`` is a native array."""

        return isinstance(value, self.jax.Array)

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert a native array to numpy."""

        return np.asarray(value)

    def set_training(self, model: Any, training: bool) -> None:
        """Equinox modules carry no train/eval flag: nothing to set."""

    def training(self, module: Any) -> bool:
        """Equinox modules carry no train/eval flag; the backend records ``False``."""

        return False

    def trace(self, model: Any, value: Any) -> Any:
        """Capture ``model(value)`` on the JAX backend."""

        return tl.trace(model, value, backend="jax")


class NnxKit(EquinoxKit):
    """Kit for the JAX preview backend on Flax NNX modules."""

    name = "nnx"

    def load(self) -> None:
        """Import JAX and Flax NNX or skip."""

        self.jax = pytest.importorskip("jax")
        self.jnp = pytest.importorskip("jax.numpy")
        self.nnx = pytest.importorskip("flax.nnx")
        self.keys = iter(range(1, 1000))

    def build_module(self, call: Callable[..., Any], **children: Any) -> Any:
        """Wrap ``call(self, *args)`` as an NNX module holding ``children``."""

        class _Cell(self.nnx.Module):  # type: ignore[name-defined, misc]
            """Cell module whose ``__call__`` is the case callable."""

            def __init__(self) -> None:
                """Register the children."""

                for name, child in children.items():
                    setattr(self, name, child)

            def __call__(self, *args: Any) -> Any:
                """Run the case callable."""

                return call(self, *args)

        return _Cell()

    def linear(self) -> Any:
        """Return a square linear layer."""

        return self.nnx.Linear(_WIDTH, _WIDTH, rngs=self.nnx.Rngs(next(self.keys)))


class TinygradKit(TorchKit):
    """Kit for the tinygrad preview backend (plain Python objects are its modules)."""

    name = "tinygrad"
    inserts_identity = False
    computes_unread_returns = False
    # Pre-existing tinygrad save bloat: a 16-op trace saves at about 309 MB and the single case
    # passes the 512 MiB manifest limit with module-return recording on or off. Live half only
    # until that is fixed; flip to True then.
    saves = False

    def load(self) -> None:
        """Import tinygrad or skip."""

        self.Tensor = pytest.importorskip("tinygrad").Tensor
        self.gnn = pytest.importorskip("tinygrad.nn")
        self.Tensor.manual_seed(0)

    def build_module(self, call: Callable[..., Any], **children: Any) -> Any:
        """Wrap ``call(self, *args)`` as a plain object holding ``children``."""

        class _Cell:
            """Cell module whose ``__call__`` is the case callable."""

            def __init__(self) -> None:
                """Register the children."""

                for name, child in children.items():
                    setattr(self, name, child)

            def __call__(self, *args: Any) -> Any:
                """Run the case callable."""

                return call(self, *args)

        return _Cell()

    def linear(self) -> Any:
        """Return a square linear layer."""

        return self.gnn.Linear(_WIDTH, _WIDTH)

    def relu(self, value: Any) -> Any:
        """Return the rectified value."""

        return value.relu()

    def add(self, left: Any, right: Any) -> Any:
        """Return the elementwise sum."""

        return left + right

    def input(self) -> Any:
        """Return a deterministic model input."""

        values = np.linspace(-1.0, 1.0, 2 * _WIDTH, dtype=np.float32).reshape(2, _WIDTH)
        return self.Tensor(values)

    def is_leaf(self, value: object) -> bool:
        """Return whether ``value`` is a native array."""

        return isinstance(value, self.Tensor)

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert a native array to numpy."""

        return np.asarray(value.numpy())

    def set_training(self, model: Any, training: bool) -> None:
        """tinygrad modules carry no train/eval flag: nothing to set."""

    def training(self, module: Any) -> bool:
        """tinygrad modules carry no train/eval flag; the backend records ``False``."""

        return False

    def trace(self, model: Any, value: Any) -> Any:
        """Capture ``model(value)`` on the tinygrad backend."""

        return tl.trace(model, value, backend="tinygrad")


_MLX_MARKS = (pytest.mark.backend_mlx, pytest.mark.optional, pytest.mark.rare)
_PADDLE_MARKS = (pytest.mark.backend_paddle, pytest.mark.optional, pytest.mark.rare)
_TF_MARKS = (pytest.mark.tf_backend, pytest.mark.optional, pytest.mark.rare)
_JAX_MARKS = (pytest.mark.backend_jax, pytest.mark.optional, pytest.mark.rare)
_TINYGRAD_MARKS = (
    pytest.mark.backend_tinygrad,
    pytest.mark.optional,
    pytest.mark.rare,
    pytest.mark.slow,  # tinygrad DEV=PYTHON captures take 25-430 s per case
)
KITS = (
    pytest.param(TorchKit, id="torch"),
    pytest.param(MLXKit, id="mlx", marks=_MLX_MARKS),
    pytest.param(PaddleKit, id="paddle", marks=_PADDLE_MARKS),
    pytest.param(TFKit, id="tf", marks=_TF_MARKS),
    pytest.param(EquinoxKit, id="equinox", marks=_JAX_MARKS),
    pytest.param(NnxKit, id="nnx", marks=_JAX_MARKS),
    pytest.param(TinygradKit, id="tinygrad", marks=_TINYGRAD_MARKS),
)


def _block_body(kit: Any, case: str) -> Callable[..., Any]:
    """Return what the module under test (``block``) computes for one case."""

    def body(module: Any, hidden: Any) -> Any:
        """Compute the block's return from its input."""

        if case == "identity":
            return hidden
        first = kit.relu(module.lin(hidden))
        if case == "tuple":
            return first, kit.add(hidden, first)
        if case == "nested":
            second = kit.add(hidden, first)
            return {"a": first, "b": (second, kit.relu(second))}
        if case == "duplicate":
            return first, first
        if case == "returned_input":
            return hidden, first
        return first

    return body


def _build(kit: Any, case: str) -> tuple[Any, dict[str, list[Any]]]:
    """Build the case model and the store of each block call's actual return."""

    seen: dict[str, list[Any]] = {"block": []}
    block = kit.build_module(_block_body(kit, case), lin=kit.linear())

    def parent(module: Any, value: Any) -> Any:
        """Run the stem, the block (twice when repeated), then reduce the leaves."""

        hidden = module.stem(value)
        out = module.block(hidden)
        seen["block"].append(out)
        if case == "repeated":
            out = module.block(out)
            seen["block"].append(out)
        if case == "unconsumed":
            return kit.relu(hidden)
        leaves = [leaf for _path, leaf in iter_returned_leaves(out, kit.is_leaf)]
        return functools.reduce(kit.add, leaves)

    children = {"stem": kit.linear(), "block": block}
    if case == "unused":
        children["spare"] = kit.linear()
    model = kit.build_module(parent, **children)
    kit.set_training(model, case == "tuple")
    return model, seen


def _match_by_value(kit: Any, trace: Any, labels: list[str], leaves: list[Any]) -> None:
    """Assert recorded op values and returned leaves pair one-to-one by value."""

    remaining = [kit.to_numpy(leaf) for leaf in leaves]
    for label in labels:
        value = kit.to_numpy(trace.ops[label].out)
        hits = [index for index, leaf in enumerate(remaining) if np.allclose(value, leaf)]
        assert hits, f"{label} matches no value the call returned"
        remaining.pop(hits[0])
    assert not remaining, "a returned leaf has no recorded op"


def _check_call(kit: Any, trace: Any, call_label: str, returned: Any, case: str) -> None:
    """Check one module call's recorded outputs against its actual return."""

    call = trace.module_calls[call_label]
    if _discloses_unread(kit, case):
        assert list(call.output_ops) == []
        assert tuple(call.output_paths) == ()
        assert returned_op_labels(trace, call_label) == []
        with pytest.raises(MultiOutputModuleError):
            _ = call.out
        return
    leaves = list(iter_returned_leaves(returned, kit.is_leaf))
    distinct = list({id(leaf): leaf for _path, leaf in leaves}.values())
    assert sorted(map(repr, call.output_paths)) == sorted(repr(path) for path, _ in leaves)
    roster = returned_op_labels(trace, call_label)
    _match_by_value(kit, trace, roster, distinct)
    passthrough = [label for label in roster if label not in call.output_ops]
    expect_passthrough = case in _PASSTHROUGH_CASES and not kit.inserts_identity
    assert bool(passthrough) == expect_passthrough
    assert set(call.output_ops) <= set(roster)
    if len(call.output_ops) == 1 and not passthrough:
        _match_by_value(kit, trace, [call.output_ops[0]], [call.out])
        _match_by_value(kit, trace, list(call.output_ops), distinct)
        return
    with pytest.raises(MultiOutputModuleError) as refusal:
        _ = call.out
    if passthrough:
        assert "returns its input unchanged" in str(refusal.value)
        assert passthrough[0] in str(refusal.value)


def _discloses_unread(kit: Any, case: str) -> bool:
    """Return whether this backend leaves the unconsumed case's return uncomputed (lazy)."""

    return case == "unconsumed" and not kit.computes_unread_returns


def _trace_case(kit: Any, model: Any, case: str) -> Any:
    """Trace one case, asserting the disclosure when the return is never computed."""

    if not _discloses_unread(kit, case):
        return kit.trace(model, kit.input())
    with pytest.warns(
        CaptureGapWarning, match=r"'block:1' returned 1 tensor\(s\) with no recorded op"
    ):
        return kit.trace(model, kit.input())


def _check_trace(kit: Any, trace: Any, model: Any, seen: dict[str, list[Any]], case: str) -> None:
    """Run every cell check on one live or loaded trace."""

    assert check_metadata_invariants(trace) is True
    num_calls = 2 if case == "repeated" else 1
    returns = seen["block"][-num_calls:]
    for call_index, returned in enumerate(returns, start=1):
        _check_call(kit, trace, f"block:{call_index}", returned, case)
    block = trace.modules["block"]
    assert block.training == kit.training(model.block)
    first_ops = list(trace.module_calls["block:1"].output_ops)
    if num_calls == 1 and len(first_ops) == 1 and returned_op_labels(trace, "block:1") == first_ops:
        _match_by_value(kit, trace, first_ops, [block.out])
    elif num_calls == 1:
        with pytest.raises(MultiOutputModuleError):
            _ = block.out
    if case == "unused" and "spare" in trace.modules:
        with pytest.raises((MultiOutputModuleError, InvalidArgumentError, KeyError)):
            _ = trace.modules["spare"].out


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kit_class", KITS)
def test_module_outputs_match_actual_returns(
    kit_class: type[TorchKit], case: str, tmp_path: Path
) -> None:
    """Recorded module outputs equal each call's actual return, live and after load."""

    kit = kit_class()
    kit.load()
    model, seen = _build(kit, case)
    trace = _trace_case(kit, model, case)
    _check_trace(kit, trace, model, seen, case)
    if not kit.saves:
        return
    path = tmp_path / f"{kit.name}_{case}.tlspec"
    trace.save(path)
    _check_trace(kit, tl.load(path), model, seen, case)


@pytest.mark.parametrize("kit_class", KITS)
def test_return_recording_adds_no_ops(
    kit_class: type[TorchKit], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recording module returns adds no ops and no saved activation bytes."""

    kit = kit_class()
    kit.load()
    counts: dict[str, tuple[int, int]] = {}
    for arm in ("off", "on"):
        if arm == "off":
            monkeypatch.setattr(ModuleReturnLedger, "record", lambda *args, **kwargs: None)
        else:
            monkeypatch.undo()
        model, _seen = _build(kit, "nested")
        trace = kit.trace(model, kit.input())
        saved = [op.out for op in trace._layer_list if op.has_saved_activation]
        counts[arm] = (len(trace.ops), sum(kit.to_numpy(value).nbytes for value in saved))
    assert counts["on"] == counts["off"]
    line = (
        f"{kit.name}: ops recorded {counts['off'][0]} -> {counts['on'][0]}, "
        f"activation bytes saved {counts['off'][1]} -> {counts['on'][1]}\n"
    )
    report = os.environ.get("TL_RETURN_CELL_REPORT")
    if report:
        with open(report, "a", encoding="utf-8") as handle:
            handle.write(line)
