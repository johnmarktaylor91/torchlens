"""MLX function-call interception for technical-preview capture."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from ... import _state
from ...ir.events import ModuleFrame
from .model_prep import MLXModuleTree


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
        wrapped_classes: set[type[Any]] = set()
        for cls_name, op_name in class_op_names.items():
            cls = getattr(nn, cls_name, None)
            if cls is not None:
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
        self._wrapped = True

    def unwrap(self) -> None:
        """Restore all original MLX callables."""

        for (owner, name), original in list(self._originals.items()):
            setattr(owner, name, original)
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
                emit(trace, op_name, original, args, kwargs, output, module_stack=module_stack)
                return output
            finally:
                if frame is not None:
                    getattr(trace, "_mlx_module_stack").pop()

        setattr(owner, name, wrapper)


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
