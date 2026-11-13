"""Paddle function-call interception for Paddle capture."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

from torchlens.backends._errors import BackendUnsupportedError

from ... import _state
from .._protocols.paddle import (
    PADDLE_REFUSAL_INPLACE,
    PADDLE_REFUSAL_SCALAR_ESCAPE,
    PADDLE_REFUSAL_STOCHASTIC,
)
from ._op_call import PaddleOpCall
from .conv_layers import (
    PADDLE_CONV_LAYER_OPS,
    conv_layer_call,
    conv_layer_input,
    paddle_conv_layer_replay,
)

_ACTIVE_TAP_OBSERVER: object | None = None

_TOP_LEVEL_CORE_OPS = {
    "abs",
    "add",
    "argmax",
    "argmin",
    "assign",
    "cast",
    "clip",
    "concat",
    "cos",
    "divide",
    "einsum",
    "equal",
    "exp",
    "flatten",
    "floor",
    "full",
    "full_like",
    "greater_equal",
    "greater_than",
    "less_equal",
    "less_than",
    "linspace",
    "log",
    "matmul",
    "max",
    "mean",
    "min",
    "mm",
    "multiply",
    "negative",
    "ones",
    "ones_like",
    "pow",
    "prod",
    "reshape",
    "sin",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "subtract",
    "sum",
    "tanh",
    "tile",
    "to_tensor",
    "transpose",
    "unsqueeze",
    "var",
    "where",
    "zeros",
    "zeros_like",
}
_FUNCTIONAL_CORE_OPS = {
    "adaptive_avg_pool1d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "batch_norm",
    "conv1d",
    "conv1d_transpose",
    "conv2d",
    "conv2d_transpose",
    "conv3d_transpose",
    "dropout",
    "gelu",
    "hardsigmoid",
    "hardswish",
    "identity",
    "layer_norm",
    "leaky_relu",
    "linear",
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "relu",
    "relu6",
    "sigmoid",
    "silu",
    "softmax",
    "tanh",
}
_TENSOR_CORE_METHODS = {
    "__add__",
    "__eq__",
    "__ge__",
    "__getitem__",
    "__gt__",
    "__le__",
    "__lt__",
    "__matmul__",
    "__mul__",
    "__ne__",
    "__neg__",
    "__pow__",
    "__radd__",
    "__rmatmul__",
    "__rmul__",
    "__rpow__",
    "__rsub__",
    "__rtruediv__",
    "__sub__",
    "__truediv__",
    "abs",
    "add",
    "astype",
    "cast",
    "clip",
    "contiguous",
    "divide",
    "exp",
    "flatten",
    "log",
    "matmul",
    "max",
    "mean",
    "min",
    "multiply",
    "pow",
    "prod",
    "reshape",
    "rsqrt",
    "scale",
    "sqrt",
    "square",
    "squeeze",
    "std",
    "subtract",
    "sum",
    "t",
    "tile",
    "transpose",
    "unsqueeze",
    "var",
}
_SCALAR_ESCAPE_NAMES = {
    "__array__",
    "__bool__",
    "__dlpack__",
    "__float__",
    "__index__",
    "__int__",
    "item",
    "numpy",
    "tolist",
}
_MUTATOR_NAMES = {
    "__iadd__",
    "__ifloordiv__",
    "__ilshift__",
    "__imatmul__",
    "__imod__",
    "__imul__",
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__itruediv__",
    "__ixor__",
    "__setitem__",
    "copy_",
    "index_put_",
    "scatter_",
    "set_value",
}
_RNG_NAMES = {
    "bernoulli",
    "normal",
    "poisson",
    "rand",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "uniform",
}
_DENY_SUBSTRINGS = (
    "distributed",
    "jit",
    "load",
    "optimizer",
    "save",
    "seed",
    "set_device",
    "static",
)
_ALIAS_ALLOWED_NAMES = {
    "astype",
    "cast",
    "contiguous",
    "dropout",
    "identity",
    "reshape",
    "view",
}


@dataclass(frozen=True)
class PaddleInventory:
    """Deterministic Paddle wrapper inventory.

    Parameters
    ----------
    wrapped
        Qualified operation names wrapped for capture.
    denied
        Qualified operation names wrapped to raise.
    """

    wrapped: tuple[str, ...]
    denied: tuple[str, ...]


class _PaddleWrapperRegistry:
    """Per-process Paddle wrapper registry with clean backend rebinding."""

    def __init__(self) -> None:
        """Initialize an empty wrapper registry."""

        self._originals: dict[tuple[object, str], object] = {}
        self._wrapper_by_original: dict[int, tuple[object, object]] = {}
        self._wrapped = False
        self._inventory = PaddleInventory((), ())
        self.backend: object | None = None

    def wrap(self, backend: object) -> None:
        """Install wrappers bound to ``backend``.

        Parameters
        ----------
        backend
            Active Paddle backend that receives wrapper events.
        """

        if self._wrapped:
            self.unwrap()
        paddle, functional, tensor_cls = _import_paddle()
        wrapped: set[str] = set()
        denied: set[str] = set()
        # R07 (the L4 unwind standard): the install loop mutates process-global
        # Paddle modules and classes; a BaseException escaping mid-install used
        # to strand every wrapper already landed (nothing called ``unwrap``
        # because the capture-side ``finally`` had not been entered yet).
        # ``unwrap`` restores exactly the slots registered so far.
        try:
            for owner, owner_name, name, _original, action in _iter_inventory_candidates(
                paddle, functional, tensor_cls
            ):
                op_name = _op_name(owner_name, name)
                if self.wrap_attr(owner, name, backend, op_name, action=action):
                    if action == "deny":
                        denied.add(op_name)
                    else:
                        wrapped.add(op_name)
            self._wrap_layer_module_aliases(paddle)
            wrapped.update(self._wrap_conv_layers(paddle, backend))
        except BaseException:
            self.unwrap()
            raise
        self._inventory = PaddleInventory(tuple(sorted(wrapped)), tuple(sorted(denied)))
        self._wrapped = True
        self.backend = backend

    def unwrap(self) -> None:
        """Restore all original Paddle callables.

        Every restore is attempted even if one raises; the first failure
        re-raises after the sweep so a single fallible setattr cannot leave
        the remaining process-global wrappers installed. Slots that failed
        to restore stay registered so a retry can restore them.
        """

        first_failure: BaseException | None = None
        for (owner, name), original in list(self._originals.items()):
            try:
                setattr(owner, name, original)
            except BaseException as exc:
                if first_failure is None:
                    first_failure = exc
                continue
            self._originals.pop((owner, name), None)
        if first_failure is not None:
            self._wrapped = bool(self._originals)
            raise first_failure
        self._originals.clear()
        self._wrapper_by_original.clear()
        self._wrapped = False
        self._inventory = PaddleInventory((), ())
        self.backend = None

    def _wrap_layer_module_aliases(self, paddle: object) -> None:
        """Point Paddle's own Layer modules' import-time aliases at the wrappers.

        ``paddle/nn/layer/norm.py`` runs ``from ..functional import
        batch_norm, layer_norm, ...`` at import, so ``nn.BatchNorm2D`` calls
        a module-global bound to the ORIGINAL function and the
        ``paddle.nn.functional`` wrapper never sees it: every BatchNorm output
        was an untracked tensor and validation failed its consumer. Each
        ``paddle.nn.layer.*`` global that IS (by identity) a wrapped original
        now resolves to that original's wrapper; ``unwrap`` restores it.

        Parameters
        ----------
        paddle
            Imported Paddle module (its ``nn.layer`` package lists the Layer
            implementation submodules).
        """

        layer_package = getattr(getattr(paddle, "nn", None), "layer", None)
        submodules = (
            [
                value
                for value in vars(layer_package).values()
                if isinstance(value, ModuleType) and value.__name__.startswith("paddle.nn.layer.")
            ]
            if layer_package is not None
            else []
        )
        for module in submodules:
            for name, value in tuple(vars(module).items()):
                entry = self._wrapper_by_original.get(id(value))
                if entry is None or entry[0] is not value or (module, name) in self._originals:
                    continue
                self._originals[(module, name)] = value
                setattr(module, name, entry[1])

    def _wrap_conv_layers(self, paddle: object, backend: object) -> set[str]:
        """Capture ``Conv1D``/``Conv2D``/``Conv3D`` at their ``forward`` (one op each).

        Parameters
        ----------
        paddle
            Imported Paddle module.
        backend
            Paddle backend receiving wrapper events.

        Returns
        -------
        set[str]
            Inventory names of the wrapped layer forwards (``nn.Conv2D.forward``).
        """

        names: set[str] = set()
        nn = getattr(paddle, "nn", None)
        for class_name, op_name in PADDLE_CONV_LAYER_OPS.items():
            layer_cls = getattr(nn, class_name, None)
            original = getattr(layer_cls, "forward", None) if layer_cls is not None else None
            if layer_cls is None or original is None or (layer_cls, "forward") in self._originals:
                continue
            self._originals[(layer_cls, "forward")] = original
            layer_cls.forward = _conv_layer_wrapper(original, op_name, backend)
            names.add(f"nn.{class_name}.forward")
        return names

    def is_wrapped(self) -> bool:
        """Return whether this registry currently has installed wrappers."""

        return self._wrapped

    def inventory(self) -> PaddleInventory:
        """Return the currently installed wrapper inventory.

        Returns
        -------
        PaddleInventory
            Deterministic wrapped and denied operation names.
        """

        return self._inventory

    def wrap_attr(
        self,
        owner: object,
        name: str,
        backend: object,
        op_name: str,
        *,
        action: str,
    ) -> bool:
        """Replace one Paddle attribute with a logging wrapper.

        Parameters
        ----------
        owner
            Module or class containing the callable.
        name
            Attribute name to wrap.
        backend
            Paddle backend receiving wrapper events.
        op_name
            TorchLens operation type to emit.
        action
            ``"capture"`` or ``"deny"``.

        Returns
        -------
        bool
            True when a wrapper was installed.
        """

        original = getattr(owner, name, None)
        if original is None or not callable(original):
            return False
        key = (owner, name)
        if key in self._originals:
            return False
        self._originals[key] = original

        @functools.wraps(
            original,
            assigned=("__module__", "__name__", "__doc__", "__annotations__"),
        )
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Call a Paddle function and emit an op event when logging is active."""

            observer = _ACTIVE_TAP_OBSERVER
            if observer is not None:
                observe_call = cast(Any, observer).call
                return observe_call(original, op_name, args, kwargs)
            trace, logging_enabled = _state.active_capture()
            if not logging_enabled or trace is None:
                return original(*args, **kwargs)
            depth = int(trace._paddle_capture_depth)
            if action == "deny":
                if depth == 0:
                    _raise_denied(op_name)
                return original(*args, **kwargs)
            if depth > 0:
                return original(*args, **kwargs)
            trace._paddle_capture_depth = depth + 1
            try:
                output = original(*args, **kwargs)
            finally:
                trace._paddle_capture_depth = depth
            emit = cast(Any, backend).emit_paddle_operation
            call = PaddleOpCall(original, args, kwargs, output)
            return emit(trace, op_name, call, module_stack=tuple(trace._paddle_module_stack))

        setattr(owner, name, wrapper)
        self._wrapper_by_original.setdefault(id(original), (original, wrapper))
        return True


def _conv_layer_wrapper(original: Any, op_name: str, backend: object) -> Any:
    """Build the logging ``forward`` for one convolution layer class.

    Parameters
    ----------
    original
        The class's original ``forward``.
    op_name
        ``conv1d``, ``conv2d`` or ``conv3d``.
    backend
        Paddle backend receiving the op event.

    Returns
    -------
    Any
        The replacement ``forward``.
    """

    @functools.wraps(original)
    def forward(layer: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the layer forward and emit one layer-level convolution op."""

        observer = _ACTIVE_TAP_OBSERVER
        if observer is not None and conv_layer_input(args, kwargs) is not None:
            # Derived-grad replay observes the same single layer-level op capture records.
            layer_forward = functools.partial(original, layer)
            return cast(Any, observer).call(layer_forward, op_name, args, kwargs)
        trace, logging_enabled = _state.active_capture()
        if not logging_enabled or trace is None or observer is not None:
            return original(layer, *args, **kwargs)
        depth = int(trace._paddle_capture_depth)
        x = conv_layer_input(args, kwargs)
        if depth > 0 or x is None:
            return original(layer, *args, **kwargs)
        trace._paddle_capture_depth = depth + 1
        try:
            output = original(layer, *args, **kwargs)
        finally:
            trace._paddle_capture_depth = depth
        call_args, call_kwargs = conv_layer_call(layer, op_name, x)
        module_stack = tuple(trace._paddle_module_stack)
        emit = cast(Any, backend).emit_paddle_operation
        call = PaddleOpCall(paddle_conv_layer_replay, call_args, call_kwargs, output)
        return emit(trace, op_name, call, module_stack=module_stack)

    return forward


@contextmanager
def paddle_tap_observer(observer: object) -> Any:
    """Install a Paddle tap observer independently of normal capture logging.

    Parameters
    ----------
    observer
        Backend-owned object with a ``call`` method used by wrappers.

    Yields
    ------
    None
        The observer is active for wrapped Paddle calls inside the context.
    """

    global _ACTIVE_TAP_OBSERVER
    previous = _ACTIVE_TAP_OBSERVER
    _ACTIVE_TAP_OBSERVER = observer
    try:
        yield
    finally:
        _ACTIVE_TAP_OBSERVER = previous


def wrap_paddle(backend: object) -> None:
    """Install Paddle wrappers for ``backend``.

    Parameters
    ----------
    backend
        Active Paddle backend.
    """

    _REGISTRY.wrap(backend)


def unwrap_paddle() -> None:
    """Restore original Paddle callables."""

    _REGISTRY.unwrap()


def bound_paddle_backend() -> object | None:
    """Return the backend the installed wrappers report to.

    Returns
    -------
    object | None
        Active Paddle backend, or ``None`` when the wrappers are not installed.
    """

    return _REGISTRY.backend


def is_paddle_wrapped() -> bool:
    """Return whether Paddle wrappers are currently installed.

    Returns
    -------
    bool
        True when wrappers are active.
    """

    return _REGISTRY.is_wrapped()


def paddle_wrap_inventory() -> PaddleInventory:
    """Return the currently installed Paddle wrapper inventory.

    Returns
    -------
    PaddleInventory
        Deterministic wrapped and denied operation names.
    """

    return _REGISTRY.inventory()


def is_alias_allowed_op(op_name: str) -> bool:
    """Return whether same-object outputs are allowed for ``op_name``.

    Parameters
    ----------
    op_name
        Qualified Paddle wrapper operation name.

    Returns
    -------
    bool
        True when an observed same-object output should be recorded as an alias.
    """

    return op_name.rsplit(".", 1)[-1] in _ALIAS_ALLOWED_NAMES


def _iter_inventory_candidates(
    paddle: object,
    functional: object,
    tensor_cls: type[Any],
) -> Iterable[tuple[object, str, str, object, str]]:
    """Yield deterministic Paddle wrapper candidates.

    Parameters
    ----------
    paddle
        Imported Paddle module.
    functional
        Imported ``paddle.nn.functional`` module.
    tensor_cls
        Paddle tensor class.

    Yields
    ------
    tuple[object, str, str, object, str]
        Owner, owner name, attribute name, original callable, and action.
    """

    for owner, owner_name, curated in (
        (paddle, "paddle", _TOP_LEVEL_CORE_OPS),
        (functional, "paddle.nn.functional", _FUNCTIONAL_CORE_OPS),
        (tensor_cls, "paddle.Tensor", _TENSOR_CORE_METHODS | _SCALAR_ESCAPE_NAMES | _MUTATOR_NAMES),
    ):
        for name in sorted(set(dir(owner))):
            original = getattr(owner, name, None)
            if not callable(original):
                continue
            action = _classify_candidate(owner_name, name, original, curated)
            if action is None:
                continue
            yield owner, owner_name, name, original, action


def _classify_candidate(
    owner_name: str,
    name: str,
    original: object,
    curated: set[str],
) -> str | None:
    """Classify a candidate as capture, deny, or ignored.

    Parameters
    ----------
    owner_name
        Qualified owner name.
    name
        Attribute name.
    original
        Candidate callable.
    curated
        Curated capture names for this owner.

    Returns
    -------
    str | None
        ``"capture"``, ``"deny"``, or ``None``.
    """

    del owner_name
    if name in _SCALAR_ESCAPE_NAMES or name in _MUTATOR_NAMES or _is_mutator_name(name):
        return "deny"
    if name in _RNG_NAMES:
        return "deny"
    lowered = name.lower()
    if any(part in lowered for part in _DENY_SUBSTRINGS):
        return "deny"
    if inspect.isclass(original):
        return None
    if name in curated:
        return "capture"
    return None


def _is_mutator_name(name: str) -> bool:
    """Return whether ``name`` follows Paddle's mutating-name convention.

    Parameters
    ----------
    name
        Candidate attribute name.

    Returns
    -------
    bool
        True for public names ending in one underscore.
    """

    return name.endswith("_") and not (name.startswith("__") and name.endswith("__"))


def _op_name(owner_name: str, name: str) -> str:
    """Return a deterministic operation name.

    Parameters
    ----------
    owner_name
        Qualified owner name.
    name
        Attribute name.

    Returns
    -------
    str
        Operation name for events and inventory.
    """

    if owner_name == "paddle.Tensor":
        return f"tensor.{name}"
    if owner_name == "paddle.nn.functional":
        return f"functional.{name}"
    return name


def _raise_denied(op_name: str) -> None:
    """Raise the canonical preview error for denied Paddle operations.

    Parameters
    ----------
    op_name
        Denied operation name.
    """

    base_name = op_name.rsplit(".", 1)[-1]
    if base_name in _SCALAR_ESCAPE_NAMES:
        raise BackendUnsupportedError(
            "tensor-derived Python scalar/control escape is unsupported in the paddle backend; "
            "keep it as a paddle.Tensor or pass it as an explicit input",
            code=PADDLE_REFUSAL_SCALAR_ESCAPE,
        )
    if base_name in _RNG_NAMES or "rand" in base_name or base_name in {"normal", "uniform"}:
        raise BackendUnsupportedError(
            "Paddle stochastic operations are unsupported in the Paddle backend; pass "
            "stochastic values as explicit inputs.",
            code=PADDLE_REFUSAL_STOCHASTIC,
        )
    raise BackendUnsupportedError(
        "Paddle in-place or global-state operations are unsupported.",
        code=PADDLE_REFUSAL_INPLACE,
    )


def _import_paddle() -> tuple[object, object, type[Any]]:
    """Import Paddle lazily.

    Returns
    -------
    tuple[object, object, type[Any]]
        Paddle module, functional module, and tensor class.
    """

    try:
        import paddle
        import paddle.nn.functional as functional
    except ImportError as exc:
        raise ImportError("Paddle backend requires the optional 'paddlepaddle' package.") from exc
    return paddle, functional, paddle.Tensor


_REGISTRY = _PaddleWrapperRegistry()

__all__ = [
    "PaddleInventory",
    "is_alias_allowed_op",
    "is_paddle_wrapped",
    "paddle_tap_observer",
    "paddle_wrap_inventory",
    "unwrap_paddle",
    "wrap_paddle",
]
