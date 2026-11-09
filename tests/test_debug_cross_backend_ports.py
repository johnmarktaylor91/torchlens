"""Module-level comparison of a real torch model against real ports on the other admitted backends.

Each port (Paddle, TensorFlow/Keras, JAX Equinox, JAX Flax NNX) mirrors one small torch model
with copied weights: an input linear, an identity, an eval-mode dropout, two blocks held in a
list, and a head. The correct port must agree everywhere; a port with one injected bug (block 1
adds 0.1) must be named at that block, localized there because its inputs agree; and the same
finding must come back from the saved and reloaded pair.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl

pytest.importorskip("pandas")

_WIDTH = 8
_BUG = 0.1


class _TorchBlock(nn.Module):
    """``act(fc(x))``."""

    def __init__(self) -> None:
        """Build the block."""

        super().__init__()
        self.fc, self.act = nn.Linear(_WIDTH, _WIDTH), nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``act(fc(x))``."""

        return self.act(self.fc(x))


class _TorchNet(nn.Module):
    """``head(blocks(drop(skip(fc_in(x)))))``."""

    def __init__(self) -> None:
        """Build the reference model."""

        super().__init__()
        self.fc_in, self.skip, self.drop = nn.Linear(4, _WIDTH), nn.Identity(), nn.Dropout(0.1)
        self.blocks = nn.ModuleList([_TorchBlock(), _TorchBlock()])
        self.head = nn.Linear(_WIDTH, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the reference forward."""

        h = self.drop(self.skip(self.fc_in(x)))
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def _reference() -> _TorchNet:
    """Return the seeded torch reference in eval mode."""

    torch.manual_seed(0)
    return _TorchNet().eval()


def _linears(model: _TorchNet) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return every linear layer's ``(weight, bias)`` (torch layout: out x in) by address."""

    names = {"fc_in": model.fc_in, "head": model.head}
    names.update({f"blocks.{i}.fc": block.fc for i, block in enumerate(model.blocks)})
    return {
        name: (layer.weight.detach().numpy().copy(), layer.bias.detach().numpy().copy())
        for name, layer in names.items()
    }


# ----------------------------------------------------------------------------- Paddle


def _paddle_port(weights: dict[str, Any], bug_block: int | None, x: np.ndarray) -> tl.Trace:
    """Trace the Paddle port."""

    paddle = pytest.importorskip("paddle")
    pnn = paddle.nn

    class Block(pnn.Layer):
        """``act(fc(x))`` (+ the bug)."""

        def __init__(self, bug: bool) -> None:
            """Build the block."""

            super().__init__()
            self.fc, self.act, self.bug = pnn.Linear(_WIDTH, _WIDTH), pnn.ReLU(), bug

        def forward(self, h: Any) -> Any:
            """Return ``act(fc(h))``."""

            out = self.act(self.fc(h))
            return out + _BUG if self.bug else out

    class Net(pnn.Layer):
        """The Paddle port."""

        def __init__(self) -> None:
            """Build the port."""

            super().__init__()
            self.fc_in, self.skip, self.drop = (
                pnn.Linear(4, _WIDTH),
                pnn.Identity(),
                pnn.Dropout(0.1),
            )
            self.blocks = pnn.LayerList([Block(i == bug_block) for i in range(2)])
            self.head = pnn.Linear(_WIDTH, 2)

        def forward(self, h: Any) -> Any:
            """Run the port forward."""

            h = self.drop(self.skip(self.fc_in(h)))
            for block in self.blocks:
                h = block(h)
            return self.head(h)

    model = Net()
    for name, (weight, bias) in weights.items():
        layer = model
        for part in name.split("."):
            layer = layer[int(part)] if part.isdigit() else getattr(layer, part)
        layer.weight.set_value(weight.T)  # Paddle linear weights are in x out
        layer.bias.set_value(bias)
    model.eval()
    return tl.trace(model, paddle.to_tensor(x), backend="paddle")


# ----------------------------------------------------------------------------- TensorFlow


def _tf_port(weights: dict[str, Any], bug_block: int | None, x: np.ndarray) -> tl.Trace:
    """Trace the Keras port (layers named after the torch attributes)."""

    tf = pytest.importorskip("tensorflow")
    keras = pytest.importorskip("keras")

    class Block(keras.layers.Layer):
        """``act(fc(x))`` (+ the bug)."""

        def __init__(self, bug: bool, **kwargs: Any) -> None:
            """Build the block."""

            super().__init__(**kwargs)
            self.fc = keras.layers.Dense(_WIDTH, name="fc")
            self.act = keras.layers.ReLU(name="act")
            self.bug = bug

        def call(self, h: Any) -> Any:
            """Return ``act(fc(h))``."""

            out = self.act(self.fc(h))
            return out + _BUG if self.bug else out

    class Net(keras.Model):
        """The Keras port."""

        def __init__(self) -> None:
            """Build the port."""

            super().__init__(name="net")
            self.fc_in = keras.layers.Dense(_WIDTH, name="fc_in")
            self.skip = keras.layers.Identity(name="skip")
            self.drop = keras.layers.Dropout(0.1, name="drop")
            self.blocks = [Block(i == bug_block, name=f"blocks.{i}") for i in range(2)]
            self.head = keras.layers.Dense(2, name="head")

        def call(self, h: Any) -> Any:
            """Run the port forward (inference: dropout passes through)."""

            h = self.drop(self.skip(self.fc_in(h)))
            for block in self.blocks:
                h = block(h)
            return self.head(h)

    model = Net()
    inputs = tf.constant(x)
    model(inputs)
    for name, (weight, bias) in weights.items():
        layer = model
        for part in name.split("."):
            layer = layer[int(part)] if part.isdigit() else getattr(layer, part)
        layer.kernel.assign(weight.T)  # Keras kernels are in x out
        layer.bias.assign(bias)
    return tl.trace(model, inputs, backend="tf")


# ----------------------------------------------------------------------------- JAX Equinox


def _equinox_port(weights: dict[str, Any], bug_block: int | None, x: np.ndarray) -> tl.Trace:
    """Trace the Equinox port (blocks in a list field)."""

    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    eqx = pytest.importorskip("equinox")

    class Block(eqx.Module):
        """``act(fc(x))`` (+ the bug)."""

        fc: Any
        act: Any
        bug: bool = eqx.field(static=True)

        def __call__(self, h: Any) -> Any:
            """Return ``act(fc(h))``."""

            out = self.act(self.fc(h))
            return out + _BUG if self.bug else out

    class Net(eqx.Module):
        """The Equinox port."""

        fc_in: Any
        skip: Any
        drop: Any
        blocks: list
        head: Any

        def __call__(self, h: Any) -> Any:
            """Run the port forward."""

            h = self.drop(self.skip(self.fc_in(h)))
            for block in self.blocks:
                h = block(h)
            return self.head(h)

    def linear(name: str, n_in: int, n_out: int) -> Any:
        weight, bias = weights[name]
        layer = eqx.nn.Linear(n_in, n_out, key=jax.random.PRNGKey(0))
        layer = eqx.tree_at(lambda m: m.weight, layer, jnp.asarray(weight))
        return eqx.tree_at(lambda m: m.bias, layer, jnp.asarray(bias))

    blocks = [
        Block(linear(f"blocks.{i}.fc", _WIDTH, _WIDTH), eqx.nn.Lambda(jax.nn.relu), i == bug_block)
        for i in range(2)
    ]
    model = Net(
        linear("fc_in", 4, _WIDTH),
        eqx.nn.Identity(),
        eqx.nn.Dropout(0.1, inference=True),
        blocks,
        linear("head", _WIDTH, 2),
    )
    return tl.trace(model, jnp.asarray(x), backend="jax")


# ----------------------------------------------------------------------------- JAX Flax NNX


def _nnx_port(weights: dict[str, Any], bug_block: int | None, x: np.ndarray) -> tl.Trace:
    """Trace the Flax NNX port (blocks in a list attribute)."""

    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    nnx = pytest.importorskip("flax.nnx")
    rngs = nnx.Rngs(0)
    sequence = getattr(nnx, "List", list)

    class Identity(nnx.Module):
        """Return the input unchanged."""

        def __call__(self, h: Any) -> Any:
            """Return ``h``."""

            return h

    class Relu(nnx.Module):
        """Rectify."""

        def __call__(self, h: Any) -> Any:
            """Return ``relu(h)``."""

            return jax.nn.relu(h)

    class Block(nnx.Module):
        """``act(fc(x))`` (+ the bug)."""

        def __init__(self, bug: bool) -> None:
            """Build the block."""

            self.fc, self.act, self.bug = nnx.Linear(_WIDTH, _WIDTH, rngs=rngs), Relu(), bug

        def __call__(self, h: Any) -> Any:
            """Return ``act(fc(h))``."""

            out = self.act(self.fc(h))
            return out + _BUG if self.bug else out

    class Net(nnx.Module):
        """The NNX port."""

        def __init__(self) -> None:
            """Build the port."""

            self.fc_in = nnx.Linear(4, _WIDTH, rngs=rngs)
            self.skip = Identity()
            self.drop = nnx.Dropout(0.1, deterministic=True, rngs=rngs)
            self.blocks = sequence([Block(i == bug_block) for i in range(2)])
            self.head = nnx.Linear(_WIDTH, 2, rngs=rngs)

        def __call__(self, h: Any) -> Any:
            """Run the port forward."""

            h = self.drop(self.skip(self.fc_in(h)))
            for block in self.blocks:
                h = block(h)
            return self.head(h)

    model = Net()
    for name, (weight, bias) in weights.items():
        layer: Any = model
        for part in name.split("."):
            layer = layer[int(part)] if part.isdigit() else getattr(layer, part)
        layer.kernel.value = jnp.asarray(weight.T)  # NNX kernels are in x out
        layer.bias.value = jnp.asarray(bias)
    return tl.trace(model, jnp.asarray(x), backend="jax")


# ----------------------------------------------------------------------------- the checks

_BATCHED = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 4)
_SINGLE = np.linspace(-1.0, 1.0, 4, dtype=np.float32)
_JAX_MARKS = (pytest.mark.backend_jax, pytest.mark.optional, pytest.mark.rare)
PORTS = (
    pytest.param(
        (_paddle_port, _BATCHED),
        id="paddle",
        marks=(pytest.mark.backend_paddle, pytest.mark.optional, pytest.mark.rare),
    ),
    pytest.param(
        (_tf_port, _BATCHED),
        id="tf",
        marks=(pytest.mark.tf_backend, pytest.mark.optional, pytest.mark.rare),
    ),
    # Equinox and NNX linears map one vector; a batch would need vmap, whose module calls
    # are not recorded, so the JAX ports run one example.
    pytest.param((_equinox_port, _SINGLE), id="equinox", marks=_JAX_MARKS),
    pytest.param((_nnx_port, _SINGLE), id="nnx", marks=_JAX_MARKS),
)

Port = tuple[Callable[[dict[str, Any], int | None, np.ndarray], tl.Trace], np.ndarray]


def _pair(port: Port, bug_block: int | None = None) -> tuple[tl.Trace, tl.Trace]:
    """Trace the torch reference and the port on the same input with the same weights."""

    build, x = port
    reference = _reference()
    port_trace = build(_linears(reference), bug_block, x)
    return tl.trace(reference, torch.from_numpy(x)), port_trace


def _frame_text(trace_a: tl.Trace, trace_b: tl.Trace) -> str:
    """Return the module comparison as text, for assertion messages."""

    frame = tl.debug.compare(trace_a, trace_b).to_pandas()
    columns = [c for c in ("op", "module_b", "alignment", "outcome", "reason") if c in frame]
    return frame[columns].to_string()


def _finding(trace_a: tl.Trace, trace_b: tl.Trace) -> tuple[Any, ...] | None:
    """Return the first divergence as ``(op, module_b, alignment, outcome, reason)``."""

    row = tl.debug.first_divergence(trace_a, trace_b)
    if row is None:
        return None
    return tuple(row[key] for key in ("op", "module_b", "alignment", "outcome", "reason"))


def _reload(trace: tl.Trace, path: Path) -> tl.Trace:
    """Save and reload one trace."""

    trace.save(path)
    return tl.load(path)


@pytest.mark.parametrize("port", PORTS)
def test_correct_port_agrees(port: Port) -> None:
    """The correct port has no divergence, and every module of the reference pairs."""

    torch_trace, port_trace = _pair(port)
    text = _frame_text(torch_trace, port_trace)
    assert tl.debug.first_divergence(torch_trace, port_trace) is None, text
    frame = tl.debug.compare(torch_trace, port_trace).to_pandas().set_index("op")
    for label in ("fc_in:1", "skip:1", "drop:1", "blocks.0:1", "blocks.1:1", "head:1"):
        assert frame.loc[label, "outcome"] == "equal", text
    assert frame.attrs["backends"][0] == "torch"


@pytest.mark.parametrize("port", PORTS)
def test_injected_bug_is_named_at_its_module(port: Port) -> None:
    """A bug in block 1 is the first divergence, localized there."""

    torch_trace, port_trace = _pair(port, bug_block=1)
    finding = _finding(torch_trace, port_trace)
    text = _frame_text(torch_trace, port_trace)
    assert finding is not None, text
    op, module_b, alignment, outcome, reason = finding
    assert (op, module_b) == ("blocks.1:1", "blocks.1:1"), text
    assert (alignment, outcome) == ("supported", "different"), text
    assert reason.startswith("localized inside this module"), text


@pytest.mark.parametrize("port", PORTS)
def test_saved_pair_gives_the_same_finding(port: Port, tmp_path: Path) -> None:
    """The saved and reloaded pair gives the same answer, with and without the bug."""

    for bug_block in (None, 1):
        torch_trace, port_trace = _pair(port, bug_block)
        live = _finding(torch_trace, port_trace)
        loaded_a = _reload(torch_trace, tmp_path / f"torch_{bug_block}.tlspec")
        loaded_b = _reload(port_trace, tmp_path / f"port_{bug_block}.tlspec")
        assert _finding(loaded_a, loaded_b) == live, _frame_text(loaded_a, loaded_b)
