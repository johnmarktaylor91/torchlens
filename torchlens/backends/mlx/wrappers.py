"""MLX function-call interception for technical-preview capture."""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from typing import Any

from ... import _state
from ..._ir.events import ModuleFrame
from .model_prep import MLXModuleTree

_ACTIVE_TAP_OBSERVER: object | None = None


class _MLXWrapperRegistry:
    """Per-process MLX wrapper registry with clean backend rebinding."""

    def __init__(self) -> None:
        """Initialize an empty wrapper registry."""

        self._originals: dict[tuple[object, str], object] = {}
        self._wrapped = False

    def wrap(self, backend: object, module_tree: MLXModuleTree | None = None) -> None:
        """Install wrappers bound to ``backend``.

        Parameters
        ----------
        backend:
            Active MLX backend that receives wrapper events.
        module_tree:
            Optional discovered MLX module tree used for object-module stack
            attribution.
        """

        if self._wrapped:
            self.unwrap()
        mx, nn = _import_mlx()
        # R07 (the L4 unwind standard): the install loops mutate process-global
        # MLX modules and classes; a BaseException escaping mid-install used to
        # strand every wrapper already landed (nothing called ``unwrap`` because
        # the capture-side ``finally`` had not been entered yet). ``unwrap``
        # restores exactly the slots registered so far, so it is the unwind.
        try:
            self._wrap_installed(mx, nn, backend, module_tree)
        except BaseException:
            self.unwrap()
            raise
        self._wrapped = True

    def _wrap_installed(
        self,
        mx: Any,
        nn: Any,
        backend: object,
        module_tree: MLXModuleTree | None,
    ) -> None:
        """Run the wrapper install loops (fenced by :meth:`wrap`).

        Parameters
        ----------
        mx:
            Imported ``mlx.core`` module.
        nn:
            Imported ``mlx.nn`` module.
        backend:
            Active MLX backend that receives wrapper events.
        module_tree:
            Optional discovered MLX module tree.
        """

        for name in (
            "add",
            "matmul",
            "multiply",
            "subtract",
            "divide",
            "maximum",
            "minimum",
            "power",
            "sum",
            "mean",
            "max",
            "min",
            "argmax",
            "argmin",
            "reshape",
            "transpose",
            "concatenate",
            "stack",
            "split",
        ):
            self.wrap_attr(mx, name, backend, name)
        for name in ("relu", "gelu", "sigmoid", "tanh", "softmax", "silu"):
            self.wrap_attr(nn, name, backend, name)
        # Most functions of mlx.nn.layers.activations are mx.compile'd. An
        # unwrapped call traces the body with the wrapped mx primitives logging
        # tracer arrays, which leaves lazy parameters unevaluable. Wrap every
        # one, in that module and under mlx.nn, so the nested-op guard is up
        # while the body traces.
        activations = mlx_activations_module(nn)
        for name in mlx_activation_function_names(activations):
            self.wrap_attr(activations, name, backend, name)
            self.wrap_attr(nn, name, backend, name)
        class_op_names = {
            "Linear": "linear",
            "Conv2d": "conv2d",
            "LayerNorm": "layernorm",
            "RMSNorm": "rmsnorm",
            "BatchNorm": "batchnorm",
            "GroupNorm": "groupnorm",
            "Dropout": "dropout",
            "Embedding": "embedding",
            "MultiHeadAttention": "multiheadattention",
        }
        layer_ops = [(getattr(nn, cls_name, None), op) for cls_name, op in class_op_names.items()]
        # The activation layers (nn.ReLU, ...) reach their function through a
        # closure MLX binds at class creation, so no attribute wrapper sees the
        # call: wrap the layer itself, like nn.Linear.
        layer_ops += [
            (cls, cls.__name__.lower()) for cls in mlx_activation_layer_classes(activations, nn)
        ]
        wrapped_classes: set[type[Any]] = set()
        for cls, op_name in layer_ops:
            if cls is not None and cls not in wrapped_classes:
                self.wrap_attr(
                    cls,
                    "__call__",
                    backend,
                    op_name,
                    module_tree=module_tree,
                    module_instances=_module_instances_for_class(module_tree, cls),
                )
                wrapped_classes.add(cls)
        if module_tree is not None:
            for module_class, instances in module_tree.modules_by_class.items():
                if module_class in wrapped_classes:
                    continue
                self.wrap_attr(
                    module_class,
                    "__call__",
                    backend,
                    None,
                    module_tree=module_tree,
                    module_instances=instances,
                )

    def unwrap(self) -> None:
        """Restore all original MLX callables.

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
        self._wrapped = False

    def is_wrapped(self) -> bool:
        """Return whether this registry currently has installed wrappers."""

        return self._wrapped

    def wrap_attr(
        self,
        owner: object,
        name: str,
        backend: object,
        op_name: str | None,
        *,
        module_tree: MLXModuleTree | None = None,
        module_instances: dict[int, str] | None = None,
    ) -> None:
        """Replace one MLX attribute with a logging wrapper.

        Parameters
        ----------
        owner:
            Module or class containing the callable.
        name:
            Attribute name to wrap.
        backend:
            MLX backend receiving wrapper events.
        op_name:
            TorchLens operation type to emit, or ``None`` for stack-only module wrappers.
        module_tree:
            Optional discovered MLX module tree used to count module calls.
        module_instances:
            Optional mapping from wrapped module instance id to primary address.
        """

        original = getattr(owner, name, None)
        if original is None or not callable(original):
            return
        key = (owner, name)
        if key in self._originals:
            return
        self._originals[key] = original

        @functools.wraps(
            original,
            assigned=("__module__", "__name__", "__doc__", "__annotations__"),
        )
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Call an MLX function and emit an op event when logging is active."""

            observer = _ACTIVE_TAP_OBSERVER
            if observer is not None:
                # The replay observer mirrors capture's module frames, so it
                # needs the same discovered-module address capture would push.
                module_address = (
                    module_instances.get(id(args[0]))
                    if module_tree is not None and module_instances is not None and args
                    else None
                )
                observe_call = getattr(observer, "call")
                return observe_call(original, op_name, args, kwargs, module_address=module_address)
            if not _state._logging_enabled:
                return original(*args, **kwargs)
            trace = _state._active_trace
            if trace is None:
                return original(*args, **kwargs)

            frame = _push_module_frame(trace, module_tree, module_instances, args, kwargs)
            try:
                if op_name is None:
                    return original(*args, **kwargs)
                if getattr(trace, "_mlx_capture_depth", 0) > 0:
                    return original(*args, **kwargs)
                trace._mlx_capture_depth = getattr(trace, "_mlx_capture_depth", 0) + 1
                try:
                    output = original(*args, **kwargs)
                finally:
                    trace._mlx_capture_depth -= 1
                module_stack = tuple(getattr(trace, "_mlx_module_stack", ()))
                emit = getattr(backend, "emit_mlx_operation")
                # emit returns the effective output: intervention-replaced
                # leaves must be what the model consumes downstream.
                return emit(
                    trace, op_name, original, args, kwargs, output, module_stack=module_stack
                )
            finally:
                if frame is not None:
                    getattr(trace, "_mlx_module_stack").pop()

        setattr(owner, name, wrapper)


@contextmanager
def mlx_tap_observer(observer: object) -> Any:
    """Install an MLX tap observer independently of normal capture logging.

    Parameters
    ----------
    observer
        Backend-owned object with a ``call`` method used by wrappers.

    Yields
    ------
    None
        The observer is active for wrapped MLX calls inside the context.
    """

    global _ACTIVE_TAP_OBSERVER
    previous = _ACTIVE_TAP_OBSERVER
    _ACTIVE_TAP_OBSERVER = observer
    try:
        yield
    finally:
        _ACTIVE_TAP_OBSERVER = previous


#: Public names ``mlx.nn.layers.activations`` defines that are not
#: activation functions or layers. Empty today; a helper MLX adds there goes here.
MLX_ACTIVATION_NAME_EXCLUSIONS: frozenset[str] = frozenset()


def mlx_activations_module(nn: Any) -> Any:
    """Return ``mlx.nn.layers.activations`` (``None`` when MLX moved it).

    Parameters
    ----------
    nn
        Imported ``mlx.nn`` module.

    Returns
    -------
    Any
        The activations module, or ``None``.
    """

    return getattr(getattr(nn, "layers", None), "activations", None)


def mlx_activation_function_names(activations: Any) -> tuple[str, ...]:
    """Derive the activation functions ``mlx.nn.layers.activations`` defines.

    Every public plain function defined in that module, and every
    ``mx.compile``'d function bound there, minus
    :data:`MLX_ACTIVATION_NAME_EXCLUSIONS`. Derived, not listed, so an
    activation MLX adds is wrapped without a code change.

    Parameters
    ----------
    activations
        The activations module, or ``None``.

    Returns
    -------
    tuple[str, ...]
        Sorted attribute names to wrap.
    """

    if activations is None:
        return ()
    names = []
    for name, value in vars(activations).items():
        if name.startswith("_") or name in MLX_ACTIVATION_NAME_EXCLUSIONS:
            continue
        if isinstance(value, type) or inspect.ismodule(value):
            continue
        defined_here = inspect.isfunction(value) and value.__module__ == activations.__name__
        if defined_here or type(value).__name__ == "gc_func":
            names.append(name)
    return tuple(sorted(names))


def mlx_activation_layer_classes(activations: Any, nn: Any) -> tuple[type[Any], ...]:
    """Derive the activation layer classes ``mlx.nn.layers.activations`` defines.

    Every ``nn.Module`` subclass defined in that module, minus
    :data:`MLX_ACTIVATION_NAME_EXCLUSIONS`, so a layer MLX adds is wrapped
    without a code change.

    Parameters
    ----------
    activations
        The activations module, or ``None``.
    nn
        Imported ``mlx.nn`` module.

    Returns
    -------
    tuple[type[Any], ...]
        Layer classes in name order.
    """

    module_base = getattr(nn, "Module", None)
    if activations is None or not isinstance(module_base, type):
        return ()
    return tuple(
        value
        for name, value in sorted(vars(activations).items())
        if isinstance(value, type)
        and issubclass(value, module_base)
        and value.__module__ == activations.__name__
        and name not in MLX_ACTIVATION_NAME_EXCLUSIONS
    )


def _module_instances_for_class(
    module_tree: MLXModuleTree | None,
    cls: type[Any],
) -> dict[int, str] | None:
    """Return discovered module instances for ``cls``.

    Parameters
    ----------
    module_tree
        Optional discovered module tree.
    cls
        MLX module class.

    Returns
    -------
    dict[int, str] | None
        Instance-id to primary-address map, if available.
    """

    if module_tree is None:
        return None
    return module_tree.modules_by_class.get(cls)


def _push_module_frame(
    trace: object,
    module_tree: MLXModuleTree | None,
    module_instances: dict[int, str] | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ModuleFrame | None:
    """Push a module-call frame for discovered MLX module instances.

    Parameters
    ----------
    trace
        Active TorchLens trace.
    module_tree
        Discovered MLX module tree.
    module_instances
        Instance-id to primary-address map for the wrapped class.
    args
        Positional call arguments.
    kwargs
        Keyword call arguments.

    Returns
    -------
    ModuleFrame | None
        Pushed frame, or ``None`` when this call is not a discovered module.
    """

    if module_tree is None or module_instances is None or not args:
        return None
    module = args[0]
    address = module_instances.get(id(module))
    if address is None:
        return None
    call_index = module_tree.call_counts.get(address, 0) + 1
    module_tree.call_counts[address] = call_index
    module_tree.forward_args_by_call[(address, call_index)] = (args[1:], dict(kwargs))
    frame = ModuleFrame(
        address=address,
        address_normalized=address,
        call_index=call_index,
        module_type=type(module).__name__,
        fx_qualpath=None,
        entry_argnames=(),
    )
    stack = getattr(trace, "_mlx_module_stack", None)
    if stack is None:
        stack = []
        setattr(trace, "_mlx_module_stack", stack)
    stack.append(frame)
    return frame


_REGISTRY = _MLXWrapperRegistry()


def wrap_mlx(backend: object, module_tree: MLXModuleTree | None = None) -> None:
    """Install MLX wrappers if MLX is available.

    Parameters
    ----------
    backend:
        Active :class:`MLXBackend` instance receiving wrapper events.
    module_tree:
        Optional discovered MLX module tree used for object-module attribution.
    """

    _REGISTRY.wrap(backend, module_tree=module_tree)


def unwrap_mlx() -> None:
    """Restore original MLX callables."""

    _REGISTRY.unwrap()


def is_mlx_wrapped() -> bool:
    """Return whether MLX wrappers are currently installed.

    Returns
    -------
    bool
        ``True`` when wrappers are installed.
    """

    return _REGISTRY.is_wrapped()


def _import_mlx() -> tuple[object, object]:
    """Import MLX lazily.

    Returns
    -------
    tuple[object, object]
        ``mlx.core`` and ``mlx.nn`` modules.
    """

    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise ImportError("MLX backend requires the optional 'mlx' package.") from exc
    return mx, nn


__all__ = ["is_mlx_wrapped", "unwrap_mlx", "wrap_mlx"]
