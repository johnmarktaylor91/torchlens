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
never reach a C1/C2 claim (see ``tests.conformance``).

Every spelling is DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torchlens as tl
from torchlens.backends.registry import get_backend_spec

from ._adapters import RosterModel
from ._rosters.jax import roster as _jax_roster
from ._rosters.mlx import roster as _mlx_roster
from ._rosters.paddle import roster as _paddle_roster
from ._rosters.tf import roster as _tf_roster
from ._rosters.tinygrad import runner_rows

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
    """The JAX packet's roster (``tests.conformance._rosters.jax``)."""

    return _jax_roster()


def _roster_tf() -> tuple[RosterModel, ...]:
    """The TensorFlow packet's roster (``tests.conformance._rosters.tf``)."""

    return _tf_roster()


def _roster_mlx() -> tuple[RosterModel, ...]:
    """The MLX packet's roster (``tests.conformance._rosters.mlx``)."""

    return _mlx_roster()


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
    # each inventory is pinned for the device it was observed on. Each matmul is
    # reshape/(permute)/expand/mul/reduce/reshape, relu is cmplt/where, a bias is
    # add, and the const/buffer/copy ops materialize the fixed weights. An op is
    # named by its UOp op name when tinygrad hands the observer only an internal
    # helper (_mop for movement ops, _rop for reductions, a lambda for elementwise
    # math). On the PYTHON device (the one the replay and derived-grad oracles
    # require, and the one CI runs) list data needs no copy.
    # Without tinygrad no case runs; the CPU lists keep that scorecard's denominator.
    inventories = _TINYGRAD_INVENTORIES.get(_tinygrad_default_device() or "CPU")
    fn_ops, module_ops = inventories if inventories is not None else (None, None)
    # The backend's own roster (``_rosters/tinygrad.py``) follows the two probe rows.

    return _rows(build_fn, build_module, fn_ops=fn_ops, module_ops=module_ops) + runner_rows()


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
            "reshape",
            "expand",
            "reshape",
            "expand",
            "buffer",
            "reshape",
            "copy",
            "reshape",
            "permute",
            "expand",
            "mul",
            "reduce",
            "reshape",
            "cmplt",
            "where",
        ),
        (
            "input",
            "const",
            "reshape",
            "expand",
            "reshape",
            "expand",
            "buffer",
            "reshape",
            "copy",
            "permute",
            "reshape",
            "permute",
            "expand",
            "mul",
            "reduce",
            "reshape",
            "buffer",
            "copy",
            "reshape",
            "expand",
            "add",
            "cmplt",
            "where",
            "reshape",
            "expand",
            "buffer",
            "reshape",
            "copy",
            "permute",
            "reshape",
            "permute",
            "expand",
            "mul",
            "reduce",
            "reshape",
            "buffer",
            "copy",
            "reshape",
            "expand",
            "add",
        ),
    ),
    "PYTHON": (
        (
            "input",
            "const",
            "reshape",
            "expand",
            "reshape",
            "expand",
            "buffer",
            "reshape",
            "reshape",
            "permute",
            "expand",
            "mul",
            "reduce",
            "reshape",
            "cmplt",
            "where",
        ),
        (
            "input",
            "const",
            "reshape",
            "expand",
            "reshape",
            "expand",
            "buffer",
            "reshape",
            "permute",
            "reshape",
            "permute",
            "expand",
            "mul",
            "reduce",
            "reshape",
            "buffer",
            "reshape",
            "expand",
            "add",
            "cmplt",
            "where",
            "reshape",
            "expand",
            "buffer",
            "reshape",
            "permute",
            "reshape",
            "permute",
            "expand",
            "mul",
            "reduce",
            "reshape",
            "buffer",
            "reshape",
            "expand",
            "add",
        ),
    ),
}


def _roster_paddle() -> tuple[RosterModel, ...]:
    """The Paddle packet's roster (``tests.conformance._rosters.paddle``)."""

    return _paddle_roster()


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

    def protocol(self) -> Any:
        """The backend's registered ``BackendProtocol`` object (``BackendSpec.protocol()``)."""

        factory = get_backend_spec(self.name).protocol
        return factory() if factory is not None else None

    def trace(self, model: Any, example_input: Any) -> Any:
        """Capture one forward pass through the public preview door."""

        return self.trace_with(model, example_input)

    def trace_with(self, model: Any, example_input: Any, **options: Any) -> Any:
        """Capture through the public entry with explicit trace options (refusal cells)."""

        return tl.trace(model, example_input, backend=self.name, **options)

    def reference_output(self, model: Any, example_input: Any) -> Any:
        """Run the model directly (the parity oracle's independent arm).

        A tuple input is positional arguments, the convention ``tl.trace``
        applies to the same input. The protocol's ``reference_output`` is the
        same call; the adapter keeps the tuple convention explicit.
        """

        if isinstance(example_input, tuple):
            return model(*example_input)
        return model(example_input)

    def rng_state(self) -> Any:
        """The framework's RNG state as a host array (``c1_rng_consumed_once``), else ``None``."""

        probe = _RNG_STATE_PROBES.get(self.name)
        return probe() if probe is not None else None

    def side_effect_snapshot(self) -> Any:
        """Framework state a refusal must leave untouched (the RNG state when probeable)."""

        state = self.rng_state()
        return None if state is None else state.tobytes()

    def as_array(self, value: Any) -> Any:
        """Convert one framework value to a NumPy array for the runner's comparison."""

        import numpy as np

        to_numpy = getattr(value, "numpy", None)
        return np.asarray(to_numpy() if callable(to_numpy) else value)

    #: Save level whose bundle keeps the replay recipes validation re-executes
    #: from after load (torch declares the same level for its saved arguments).
    #: Only ``c2_validation_after_load`` saves at it; every other c2 cell keeps
    #: the default level.
    replay_save_level = "executable_with_callables"

    def save(self, trace: Any, path: Path, level: str | None = None) -> None:
        """Persist one captured trace (default level unless ``level`` is given)."""

        if level is None:
            trace.save(path)
        else:
            trace.save(path, level=level)

    def load(self, path: Path) -> Any:
        """Reload one persisted artifact."""

        return tl.load(str(path))


def _mlx_rng_state() -> Any:
    """MLX's global PRNG key as a host array."""

    import mlx.core as mx
    import numpy as np

    key = mx.random.state[0]
    mx.eval(key)
    return np.array(key)


def _tinygrad_rng_state() -> Any:
    """tinygrad's seed and per-device RNG counters as one host array."""

    import numpy as np
    from tinygrad import Tensor

    values = [int(getattr(Tensor, "_seed", 0))]
    counters = getattr(Tensor, "_device_rng_counters", {}) or {}
    for device in sorted(counters):
        values.extend(int(v) for v in np.asarray(counters[device].numpy()).reshape(-1).tolist())
    return np.asarray(values, dtype=np.int64)


#: Backend -> zero-arg probe of the framework RNG state (lazy backends; SPEC 3.4).
_RNG_STATE_PROBES: dict[str, Callable[[], Any]] = {
    "mlx": _mlx_rng_state,
    "tinygrad": _tinygrad_rng_state,
}


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
