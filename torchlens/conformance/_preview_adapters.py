"""Preview-backend adapters for the conformance runner (F32 conformance packs).

One adapter per preview capture backend (jax, tf, mlx, tinygrad, paddle) on
the same neutral protocol the torch reference uses: ``trace`` goes through the
public ``tl.trace(..., backend=name)`` door, ``reference_output`` calls the
model directly, and ``save``/``load`` use the public persistence doors. The
runner, not the adapter, compares the two arms: :meth:`as_array` only converts
a framework value to a NumPy array.

Each adapter carries its own zero-network roster of two config-built toys (a
function root and a module or object root), built lazily so naming the
backend never imports its framework. An absent framework is reported by
:meth:`unavailable_reason`; the runner turns every model case into a named
skip, which fails the requested cell. Toys earn nothing: these rosters can
never reach a C1/C2 claim (see ``torchlens.conformance``).

Every spelling is DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._adapters import RosterModel

__tl_layer__ = "L9"

#: The preview capture backends a conformance pack exists for, in report order.
PREVIEW_BACKENDS: tuple[str, ...] = ("jax", "tf", "mlx", "tinygrad", "paddle")

_WEIGHTS = [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.9], [1.0, -1.1, 1.2]]
_INPUT = [[-1.0, -0.5, 0.25, 1.0], [0.5, 1.5, -2.0, 0.75]]


@dataclass(frozen=True)
class PreviewFramework:
    """Static facts about one preview backend's framework.

    Parameters
    ----------
    backend:
        TorchLens backend name.
    module:
        Import name probed for availability (never imported to probe).
    extra:
        The TorchLens extra that installs the framework.
    min_python:
        Lowest Python the extra installs on, as ``(major, minor)``.
    """

    backend: str
    module: str
    extra: str
    min_python: tuple[int, int] = (3, 10)


PREVIEW_FRAMEWORKS: dict[str, PreviewFramework] = {
    "jax": PreviewFramework("jax", "jax", "jax"),
    "tf": PreviewFramework("tf", "tensorflow", "tf"),
    "mlx": PreviewFramework("mlx", "mlx.core", "mlx"),
    "tinygrad": PreviewFramework("tinygrad", "tinygrad", "tinygrad", (3, 11)),
    "paddle": PreviewFramework("paddle", "paddle", "paddle"),
}


def framework_unavailable_reason(framework: PreviewFramework) -> str | None:
    """Name why a preview framework cannot run here, or ``None`` when it can.

    Parameters
    ----------
    framework:
        The framework facts to probe.

    Returns
    -------
    str | None
        A named reason (``framework_not_installed`` or
        ``python_below_extra_floor`` with the remedy), else ``None``.
    """

    if sys.version_info[:2] < framework.min_python:
        floor = ".".join(str(part) for part in framework.min_python)
        return (
            f"python_below_extra_floor: the {framework.extra} extra needs Python "
            f">= {floor}; this is {sys.version_info.major}.{sys.version_info.minor}"
        )
    try:
        found = importlib.util.find_spec(framework.module) is not None
    except (ImportError, ValueError):
        found = False
    if found:
        return None
    return (
        f"framework_not_installed: {framework.module} is not importable; "
        f"install torchlens[{framework.extra}]"
    )


def _roster_jax() -> tuple[RosterModel, ...]:
    """Function root plus an equinox MLP for the jax pack."""

    def build_fn() -> tuple[Any, Any]:
        """Build the function root and its example input."""

        import jax.numpy as jnp

        # The jax preview contract is fn(params, *inputs): weights travel as
        # explicit pytree leaves, never as closed-over constants.
        params = {"w": jnp.asarray(_WEIGHTS, dtype=jnp.float32)}

        def matmul_relu(params: Any, x: Any) -> Any:
            """Matmul then relu, the function root's whole computation."""

            return jnp.maximum(jnp.matmul(x, params["w"]), 0.0)

        return matmul_relu, (params, jnp.asarray(_INPUT, dtype=jnp.float32))

    def build_module() -> tuple[Any, Any]:
        """Build the module root and its example input."""

        import equinox as eqx
        import jax
        import jax.numpy as jnp

        class ConformanceMlp(eqx.Module):
            """Two-layer equinox MLP."""

            fc1: eqx.nn.Linear
            fc2: eqx.nn.Linear

            def __init__(self) -> None:
                key1, key2 = jax.random.split(jax.random.PRNGKey(0))
                self.fc1 = eqx.nn.Linear(4, 3, key=key1)
                self.fc2 = eqx.nn.Linear(3, 2, key=key2)

            def __call__(self, x: Any) -> Any:
                return self.fc2(jax.nn.relu(self.fc1(x)))

        return ConformanceMlp(), jnp.asarray(_INPUT[0], dtype=jnp.float32)

    return _rows(
        build_fn,
        build_module,
        fn_ops=("input", "input", "dot_general", "max"),
        module_ops=("input", "dot_general", "add", "max", "dot_general", "add"),
    )


def _roster_tf() -> tuple[RosterModel, ...]:
    """Function root plus a tf.Module for the tf pack."""

    def build_fn() -> tuple[Any, Any]:
        """Build the function root and its example input."""

        import tensorflow as tf

        weights = tf.constant(_WEIGHTS, dtype=tf.float32)

        def matmul_relu(x: Any) -> Any:
            """Matmul then relu, the function root's whole computation."""

            return tf.nn.relu(tf.matmul(x, weights))

        return matmul_relu, tf.constant(_INPUT, dtype=tf.float32)

    def build_module() -> tuple[Any, Any]:
        """Build the module root and its example input."""

        import tensorflow as tf

        class ConformanceMlp(tf.Module):
            """Two-layer tf.Module MLP with fixed weights."""

            def __init__(self) -> None:
                super().__init__()
                self.w1 = tf.constant(_WEIGHTS, dtype=tf.float32)
                self.w2 = tf.constant([[1.0, -1.0], [0.5, 0.5], [-0.25, 2.0]], dtype=tf.float32)

            def __call__(self, x: Any) -> Any:
                return tf.matmul(tf.nn.relu(tf.matmul(x, self.w1)), self.w2)

        return ConformanceMlp(), tf.constant(_INPUT, dtype=tf.float32)

    return _rows(
        build_fn,
        build_module,
        fn_ops=("input", "MatMul", "Relu"),
        module_ops=("input", "MatMul", "Relu", "MatMul"),
    )


def _roster_mlx() -> tuple[RosterModel, ...]:
    """Function root plus an mlx.nn MLP for the mlx pack."""

    def build_fn() -> tuple[Any, Any]:
        """Build the function root and its example input."""

        import mlx.core as mx

        weights = mx.array(_WEIGHTS)

        def matmul_relu(x: Any) -> Any:
            """Matmul then relu, the function root's whole computation."""

            return mx.maximum(mx.matmul(x, weights), 0.0)

        return matmul_relu, mx.array(_INPUT)

    def build_module() -> tuple[Any, Any]:
        """Build the module root and its example input."""

        import mlx.core as mx
        import mlx.nn as mlx_nn

        mx.random.seed(0)
        model = mlx_nn.Sequential(mlx_nn.Linear(4, 3), mlx_nn.ReLU(), mlx_nn.Linear(3, 2))
        return model, mx.array(_INPUT)

    return _rows(
        build_fn,
        build_module,
        fn_ops=("input", "matmul", "maximum"),
        module_ops=("input", "linear", "relu", "linear"),
    )


def _roster_tinygrad() -> tuple[RosterModel, ...]:
    """Function root plus an object-tree model for the tinygrad pack."""

    def build_fn() -> tuple[Any, Any]:
        """Build the function root and its example input."""

        from tinygrad import Tensor

        weights = Tensor(_WEIGHTS)

        def matmul_relu(x: Any) -> Any:
            """Matmul then relu, the function root's whole computation."""

            return (x @ weights).relu()

        return matmul_relu, Tensor(_INPUT)

    def build_module() -> tuple[Any, Any]:
        """Build the module root and its example input."""

        from tinygrad import Tensor, nn

        class ConformanceMlp:
            """Two-layer tinygrad object model with fixed weights."""

            def __init__(self) -> None:
                self.fc1 = nn.Linear(4, 3)
                self.fc1.weight = Tensor([list(row) for row in zip(*_WEIGHTS, strict=True)])
                self.fc1.bias = Tensor([0.0, 0.1, -0.1])
                self.fc2 = nn.Linear(3, 2)
                self.fc2.weight = Tensor([[1.0, 0.5, -0.25], [-1.0, 0.5, 2.0]])
                self.fc2.bias = Tensor([0.0, 0.0])

            def __call__(self, x: Any) -> Any:
                return self.fc2(self.fc1(x).relu())

        return ConformanceMlp(), Tensor(_INPUT)

    # tinygrad records its lazy-graph decomposition, which differs by device, so
    # each inventory is pinned for the device it was observed on. On CPU each
    # matmul is reshape/expand/mul/reduce, relu is cmplt/where, a bias is add,
    # and the const/buffer/copy/_mop ops materialize the fixed weights. On the
    # PYTHON device (the one the replay and derived-grad oracles require, and
    # the one CI runs) list data needs no copy, the movement ops arrive through
    # tinygrad's _mop helper, the reduce through _rop, and mul, add and cmplt
    # through lambdas, which is the name tinygrad hands the observer.
    # Without tinygrad no case runs; the CPU lists keep that scorecard's denominator.
    inventories = _TINYGRAD_INVENTORIES.get(_tinygrad_default_device() or "CPU")
    fn_ops, module_ops = inventories if inventories is not None else (None, None)
    return _rows(build_fn, build_module, fn_ops=fn_ops, module_ops=module_ops)


def _tinygrad_default_device() -> str | None:
    """tinygrad's default device name (``"CPU"``, ``"PYTHON"``, ...), else ``None``."""

    try:
        from tinygrad import Device
    except ImportError:
        return None
    return str(Device.DEFAULT).split(":", 1)[0].upper()


#: Observed tinygrad 0.13 op inventories per default device, as
#: ``(fn_ops, module_ops)``. A device with no entry declares no inventory, so
#: C1 does not run there rather than comparing against another device's list.
_TINYGRAD_INVENTORIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "CPU": (
        (
            "input",
            "const",
            "_mop",
            "const",
            "const",
            "_mop",
            "buffer",
            "const",
            "const",
            "reshape",
            "expand",
            "buffer",
            "reshape",
            "copy",
            "_mop",
            "_mop",
            "_mop",
            "mul",
            "reduce",
            "reshape",
            "cmplt",
            "where",
        ),
        (
            "input",
            "const",
            "_mop",
            "const",
            "const",
            "_mop",
            "buffer",
            "const",
            "const",
            "reshape",
            "expand",
            "buffer",
            "reshape",
            "copy",
            "_mop",
            "_mop",
            "_mop",
            "_mop",
            "mul",
            "reduce",
            "reshape",
            "buffer",
            "copy",
            "_mop",
            "_mop",
            "add",
            "cmplt",
            "where",
            "reshape",
            "expand",
            "buffer",
            "reshape",
            "copy",
            "_mop",
            "_mop",
            "_mop",
            "_mop",
            "mul",
            "reduce",
            "reshape",
            "buffer",
            "copy",
            "_mop",
            "_mop",
            "add",
        ),
    ),
    "PYTHON": (
        (
            "input",
            "const",
            "_mop",
            "const",
            "const",
            "_mop",
            "buffer",
            "const",
            "const",
            "_mop",
            "_mop",
            "buffer",
            "reshape",
            "_mop",
            "_mop",
            "_mop",
            "<lambda>",
            "_rop",
            "_mop",
            "<lambda>",
            "where",
        ),
        (
            "input",
            "const",
            "_mop",
            "const",
            "const",
            "_mop",
            "buffer",
            "const",
            "const",
            "_mop",
            "_mop",
            "buffer",
            "reshape",
            "_mop",
            "_mop",
            "_mop",
            "_mop",
            "<lambda>",
            "_rop",
            "_mop",
            "buffer",
            "_mop",
            "_mop",
            "<lambda>",
            "<lambda>",
            "where",
            "_mop",
            "_mop",
            "buffer",
            "reshape",
            "_mop",
            "_mop",
            "_mop",
            "_mop",
            "<lambda>",
            "_rop",
            "_mop",
            "buffer",
            "_mop",
            "_mop",
            "<lambda>",
        ),
    ),
}


def _roster_paddle() -> tuple[RosterModel, ...]:
    """Function root plus a paddle.nn MLP for the paddle pack."""

    def build_fn() -> tuple[Any, Any]:
        """Build the function root and its example input."""

        import paddle

        weights = paddle.to_tensor(_WEIGHTS, dtype="float32")

        def matmul_relu(x: Any) -> Any:
            """Matmul then relu, the function root's whole computation."""

            return paddle.nn.functional.relu(paddle.matmul(x, weights))

        return matmul_relu, paddle.to_tensor(_INPUT, dtype="float32")

    def build_module() -> tuple[Any, Any]:
        """Build the module root and its example input."""

        import paddle

        paddle.seed(0)
        model = paddle.nn.Sequential(
            paddle.nn.Linear(4, 3), paddle.nn.ReLU(), paddle.nn.Linear(3, 2)
        )
        return model, paddle.to_tensor(_INPUT, dtype="float32")

    return _rows(
        build_fn,
        build_module,
        fn_ops=("input", "matmul", "functional.relu"),
        module_ops=("input", "functional.linear", "functional.relu", "functional.linear"),
    )


def _rows(
    build_fn: Callable[[], tuple[Any, Any]],
    build_module: Callable[[], tuple[Any, Any]],
    *,
    fn_ops: tuple[str, ...] | None,
    module_ops: tuple[str, ...] | None,
) -> tuple[RosterModel, ...]:
    """The two config-built roster rows every preview pack runs.

    ``fn_ops`` and ``module_ops`` are each model's independent op inventory:
    its input ops and the op kinds its definition performs, in capture order,
    in the backend's naming (``Op.func_name``). Each list was checked against
    the model definition; a missing op is a capture defect, never a list edit.
    ``None`` declares no inventory, so C1 does not run for that row.
    """

    return (
        RosterModel(
            family="matmul_relu_function",
            realism="config_built",
            builder=build_fn,
            expected_op_kinds=fn_ops,
        ),
        RosterModel(
            family="mlp_module",
            realism="config_built",
            builder=build_module,
            expected_op_kinds=module_ops,
        ),
    )


_ROSTERS: dict[str, Callable[[], tuple[RosterModel, ...]]] = {
    "jax": _roster_jax,
    "tf": _roster_tf,
    "mlx": _roster_mlx,
    "tinygrad": _roster_tinygrad,
    "paddle": _roster_paddle,
}


class PreviewBackendAdapter:
    """Conformance adapter for one preview capture backend.

    Parameters
    ----------
    backend:
        One of :data:`PREVIEW_BACKENDS`.
    """

    #: Preview backends publish the highest tier they actually earn; the
    #: runner's pretrained-family floor still gates every C1/C2 claim.
    claim_eligible = True

    def __init__(self, backend: str) -> None:
        """Bind the adapter to one preview backend."""

        self.name = backend
        self.framework = PREVIEW_FRAMEWORKS[backend]

    def unavailable_reason(self) -> str | None:
        """Named reason the framework cannot run here, else ``None``."""

        return framework_unavailable_reason(self.framework)

    def default_roster(self) -> tuple[RosterModel, ...]:
        """This backend's zero-network config-built roster (never claim-eligible)."""

        return _ROSTERS[self.name]()

    def capabilities(self) -> dict[str, bool]:
        """Declared capability flags exercised by the packs.

        ``validation_replay`` is ``False``: the packs do not drive a replay
        door for preview backends, so they claim none.
        """

        return {"capture": True, "durable_save": True, "validation_replay": False}

    def trace(self, model: Any, example_input: Any) -> Any:
        """Capture one forward pass through the public preview door."""

        import torchlens as tl

        return tl.trace(model, example_input, backend=self.name)

    def reference_output(self, model: Any, example_input: Any) -> Any:
        """Run the model directly (the parity oracle's independent arm).

        A tuple input is positional arguments, the convention ``tl.trace``
        applies to the same input.
        """

        if isinstance(example_input, tuple):
            return model(*example_input)
        return model(example_input)

    def as_array(self, value: Any) -> Any:
        """Convert one framework value to a NumPy array for the runner's comparison."""

        import numpy as np

        to_numpy = getattr(value, "numpy", None)
        return np.asarray(to_numpy() if callable(to_numpy) else value)

    def save(self, trace: Any, path: Path) -> None:
        """Persist one captured trace as a portable artifact."""

        import torchlens as tl

        tl.save(trace, path)

    def load(self, path: Path) -> Any:
        """Reload one persisted artifact."""

        import torchlens as tl

        return tl.load(str(path))


def _adapter_factory(backend: str) -> Callable[[str | None], Any]:
    """Factory for one preview adapter (plants belong to the fake provider only)."""

    def factory(plant: str | None) -> Any:
        """Build this backend's preview adapter."""

        return PreviewBackendAdapter(backend)

    return factory


#: Preview adapter factories, dispatched as data by ``resolve_adapter``.
PREVIEW_ADAPTER_FACTORIES: dict[str, Callable[[str | None], Any]] = {
    backend: _adapter_factory(backend) for backend in PREVIEW_BACKENDS
}
