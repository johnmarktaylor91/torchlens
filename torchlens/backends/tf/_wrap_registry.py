"""Entry points the TensorFlow intervention layer wraps to present op outputs.

TensorFlow op callbacks are read-only (a returned replacement is ignored), so an
op-level edit happens where Python receives the op's output. Two layers:

1. Curated public entry points (``tf.nn``, ``tf.math``, ``tf.linalg``, top-level
   ``tf``), kept so composite functions present their final output by name.
2. Every generated op wrapper (``gen_*_ops``), plus each internal or public
   namespace binding that refers to one. Every eager op issued from Python runs
   through one of these, so nearly every captured op is presented.

What stays out of reach is the framework limit: ops TensorFlow runs from C++
without returning through a Python wrapper (compiled ``tf.function`` bodies,
fused kernels' internals). The reachability audit refuses a site that matched one.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable, Iterator
from typing import Any

from ._tf_compat import get_tf_gen_op_modules, get_tf_op_namespace_modules

_CURATED_WRAP_ENTRIES: tuple[tuple[str, str], ...] = (
    ("nn", "relu"),
    ("nn", "relu6"),
    ("nn", "leaky_relu"),
    ("nn", "elu"),
    ("nn", "selu"),
    ("nn", "gelu"),
    ("nn", "silu"),
    ("nn", "swish"),
    ("nn", "sigmoid"),
    ("nn", "softmax"),
    ("nn", "log_softmax"),
    ("nn", "softplus"),
    ("nn", "bias_add"),
    ("nn", "conv2d"),
    ("nn", "convolution"),
    ("nn", "depthwise_conv2d"),
    ("nn", "max_pool2d"),
    ("nn", "avg_pool2d"),
    ("nn", "max_pool"),
    ("nn", "avg_pool"),
    ("nn", "batch_normalization"),
    ("nn", "moments"),
    ("", "matmul"),
    ("", "add"),
    ("", "subtract"),
    ("", "multiply"),
    ("", "divide"),
    ("", "tanh"),
    ("", "sigmoid"),
    ("", "exp"),
    ("", "concat"),
    ("", "reduce_sum"),
    ("", "reduce_mean"),
    ("", "reduce_max"),
    ("", "reshape"),
    ("", "transpose"),
    ("", "squeeze"),
    ("", "expand_dims"),
    ("", "pad"),
    ("", "cast"),
    ("", "einsum"),
    ("", "gather"),
    ("", "where"),
    ("math", "add"),
    ("math", "multiply"),
    ("math", "tanh"),
    ("math", "sigmoid"),
    ("math", "exp"),
    ("math", "log"),
    ("math", "rsqrt"),
    ("math", "sqrt"),
    ("math", "square"),
    ("math", "squared_difference"),
    ("math", "maximum"),
    ("math", "minimum"),
    ("math", "erf"),
    ("linalg", "matmul"),
    ("Tensor", "__neg__"),
)
"""Curated (submodule, name) public entry points wrapped for op-level sites.

``Tensor.__neg__`` is TensorFlow's direct reference to the generated ``neg``
wrapper (bound when operators are installed), so unary minus needs its own entry.
"""

_EXTRA_GENERATED_ENTRIES: tuple[tuple[str, str], ...] = (
    ("tensorflow.python.ops.gen_resource_variable_ops", "read_variable_op"),
)
"""Single generated wrappers outside the wrapped ``gen_*_ops`` modules.

``read_variable_op`` presents each variable read (``ReadVariableOp``), so a
parameter's value can be edited where the forward reads it. The rest of that
module (variable creation, assignment) records no forward values and stays out.
"""

#: Public namespaces scanned for direct bindings to generated wrappers.
_PUBLIC_NAMESPACES = ("nn", "math", "linalg", "image")


def iter_wrap_targets(
    tf: Any, curated: Iterable[tuple[str, str]]
) -> Iterator[tuple[Any, str, Any]]:
    """Yield ``(owner, attribute, original)`` for every entry point to wrap.

    Curated entries come first, then generated wrappers and the namespace
    bindings that refer to them. The caller dedups wrappers by original identity.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    curated
        Curated ``(submodule, name)`` public entries (``""`` is top-level ``tf``).

    Yields
    ------
    tuple[Any, str, Any]
        Owner object, attribute name, and the unwrapped callable.
    """

    for submodule_name, attr_name in curated:
        owner = tf if submodule_name == "" else getattr(tf, submodule_name, None)
        original = getattr(owner, attr_name, None) if owner is not None else None
        if original is not None and callable(original):
            yield owner, attr_name, original
    generated: dict[int, Any] = {}
    for module, attr_name, value in _generated_wrappers():
        generated[id(value)] = value
        yield module, attr_name, value
    if not generated:
        return
    owners = (*get_tf_op_namespace_modules(), tf, *_public_namespaces(tf))
    for owner in owners:
        for attr_name, value in list(getattr(owner, "__dict__", {}).items()):
            if id(value) in generated and generated[id(value)] is value:
                yield owner, attr_name, value


def _generated_wrappers() -> Iterator[tuple[Any, str, Any]]:
    """Yield ``(module, attribute, wrapper)`` for every generated op wrapper to wrap."""

    for module in get_tf_gen_op_modules():
        for attr_name, value in list(vars(module).items()):
            if _is_generated_wrapper(module, attr_name, value):
                yield module, attr_name, value
    for module_name, attr_name in _EXTRA_GENERATED_ENTRIES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # feature detection: this TensorFlow lays it out differently
            continue
        value = getattr(module, attr_name, None)
        if _is_generated_wrapper(module, attr_name, value):
            yield module, attr_name, value


def _public_namespaces(tf: Any) -> tuple[Any, ...]:
    """Return the public TensorFlow namespaces that exist on ``tf``."""

    return tuple(
        namespace
        for namespace in (getattr(tf, name, None) for name in _PUBLIC_NAMESPACES)
        if namespace is not None
    )


def _is_generated_wrapper(module: Any, attr_name: str, value: Any) -> bool:
    """Return whether ``value`` is one op wrapper defined in a gen module.

    Single-underscore names are real op wrappers too (``_sum``, ``_any``, ``_max``
    shadow builtins); dunders and the ``*_eager_fallback`` slow paths are not.
    """

    return (
        inspect.isfunction(value)
        and not attr_name.startswith("__")
        and not attr_name.endswith("_eager_fallback")
        and getattr(value, "__module__", None) == module.__name__
    )
