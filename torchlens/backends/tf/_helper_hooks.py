"""TensorFlow applications of the built-in edit helpers.

Each built-in helper (``tl.zero_ablate``, ``tl.scale``, ``tl.add``, ``tl.clamp``,
``tl.mean_ablate``, ``tl.replace_with``) is a portable :class:`Edit` spec whose
own factory builds a torch hook. TensorFlow runs the same arithmetic with
TensorFlow ops instead, so live edits (``intervene=``) and finished-trace edits
(``fork().do``) give the values torch gives. A value from another framework
(a torch tensor, an MLX array) is refused, never converted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..registry import BackendUnsupportedError

#: The built-in helpers TensorFlow applies natively, in documentation order.
TF_SUPPORTED_HELPER_NAMES = (
    "zero_ablate",
    "scale",
    "add",
    "clamp",
    "mean_ablate",
    "replace_with",
)


def tf_helper_hook(spec: Any, tf: Any) -> Callable[[Any], Any]:
    """Return the TensorFlow replacement callable for a built-in helper spec.

    Parameters
    ----------
    spec
        Portable helper spec built by one of :data:`TF_SUPPORTED_HELPER_NAMES`.
    tf
        Imported TensorFlow module.

    Returns
    -------
    Callable[[Any], Any]
        Callable mapping the original output tensor to its replacement.

    Raises
    ------
    BackendUnsupportedError
        For a helper TensorFlow has no application for, or a helper argument
        that is a value from another framework.
    """

    builder = _BUILDERS.get(spec.name)
    if builder is None:
        raise BackendUnsupportedError(
            f"the tf backend has no application for intervention helper {spec.name!r}; "
            f"supported helpers are {', '.join(TF_SUPPORTED_HELPER_NAMES)}, plus callable "
            "actions written against TensorFlow tensors.",
            code="trace_option_unsupported",
            option="intervene",
        )
    return builder(spec, tf)


def _kwargs(spec: Any) -> dict[str, Any]:
    """Return a helper spec's keyword arguments as a dict."""

    return dict(getattr(spec, "kwargs", ()) or ())


def _zero_ablate(spec: Any, tf: Any) -> Callable[[Any], Any]:
    """``tl.zero_ablate``: zeros with the output's shape and dtype."""

    del spec
    return lambda out: tf.fill(_static_shape(out, tf), tf.constant(out.dtype.as_numpy_dtype(0)))


def _static_shape(out: Any, tf: Any) -> Any:
    """``out``'s shape as a host list when fully known, else its runtime shape.

    A host-list shape keeps the replacement a literal producer (``Fill`` with no
    graph parents) that replay validation classifies, unlike ``ZerosLike``.
    """

    return out.shape.as_list() if out.shape.is_fully_defined() else tf.shape(out)


def _scale(spec: Any, tf: Any) -> Callable[[Any], Any]:
    """``tl.scale(factor)``: the output times a scalar factor."""

    factor = _native_operand(spec.args[0], tf, helper="scale")
    return lambda out: out * tf.cast(factor, out.dtype)


def _add(spec: Any, tf: Any) -> Callable[[Any], Any]:
    """``tl.add(delta)``: the output plus a scalar or broadcastable tensor."""

    delta = _native_operand(spec.args[0], tf, helper="add")
    return lambda out: out + tf.cast(delta, out.dtype)


def _clamp(spec: Any, tf: Any) -> Callable[[Any], Any]:
    """``tl.clamp(low=, high=)``: the output limited to the given bounds."""

    kwargs = _kwargs(spec)
    low, high = kwargs.get("low"), kwargs.get("high")

    def _apply(out: Any) -> Any:
        """Clamp one output; an unset bound leaves that side open (torch.clamp)."""

        result = out
        if low is not None:
            result = tf.maximum(result, tf.cast(low, out.dtype))
        if high is not None:
            result = tf.minimum(result, tf.cast(high, out.dtype))
        return result

    return _apply


def _mean_ablate(spec: Any, tf: Any) -> Callable[[Any], Any]:
    """``tl.mean_ablate(source, dim=)``: fill with a mean of the site or a source.

    ``dim=None`` fills with the mean over every axis; an int or tuple fills with
    the mean over those axes, broadcast back (``keepdim``), as torch does. A
    ``Population`` source is a torch-side reduction and is refused.
    """

    source = spec.args[0] if spec.args else None
    if "population_identity" in _kwargs(spec):
        raise BackendUnsupportedError(
            "the tf backend applies tl.mean_ablate to the site itself, a TensorFlow "
            "tensor or a NumPy array; a Population source is averaged by torch hooks. "
            "Pass the members' mean as a TensorFlow tensor instead.",
            code="trace_option_unsupported",
            option="intervene",
        )
    axes = _kwargs(spec).get("dim")
    native_source = (
        None if source is None else _native_operand(_op_value(source), tf, "mean_ablate")
    )

    def _apply(out: Any) -> Any:
        """Return an output-shaped tensor filled with the source mean."""

        values = out if native_source is None else tf.cast(native_source, out.dtype)
        if axes is None:
            return tf.broadcast_to(tf.reduce_mean(values), _static_shape(out, tf))
        reduced = tf.reduce_mean(values, axis=list(_axes_tuple(axes)), keepdims=True)
        return tf.broadcast_to(reduced, _static_shape(out, tf))

    return _apply


def _replace_with(spec: Any, tf: Any) -> Callable[[Any], Any]:
    """``tl.replace_with(value)``: a fixed value, or a zero-argument callable's value."""

    value = spec.args[0]
    if not callable(value):
        _native_operand(value, tf, helper="replace_with")

    def _apply(out: Any) -> Any:
        """Return the replacement cast to the output's dtype, as a new tensor."""

        replacement = value() if callable(value) else value
        native = _native_operand(replacement, tf, helper="replace_with")
        # A fresh tensor per fire: two sites never share one replacement object.
        return tf.identity(tf.cast(native, out.dtype))

    return _apply


def _op_value(source: Any) -> Any:
    """Return an Op-like source's saved output, or the source itself."""

    out = getattr(source, "out", None)
    return source if out is None else out


def _axes_tuple(axes: Any) -> tuple[int, ...]:
    """Return ``mean_ablate``'s ``dim`` as a tuple of axis indexes."""

    return (int(axes),) if isinstance(axes, int) else tuple(int(axis) for axis in axes)


def _native_operand(value: Any, tf: Any, helper: str) -> Any:
    """Return ``value`` when TensorFlow can use it directly, else refuse.

    Python numbers, NumPy arrays and TensorFlow tensors or variables pass; a
    value from another framework (torch, MLX, JAX, Paddle) is refused rather
    than silently converted.
    """

    if isinstance(value, (bool, int, float)) or tf.is_tensor(value):
        return value
    if isinstance(value, tf.Variable):
        return value
    root = type(value).__module__.split(".", maxsplit=1)[0]
    if root == "numpy":
        return value
    raise BackendUnsupportedError(
        f"the tf backend's tl.{helper}(...) takes a Python number, a NumPy array or a "
        f"TensorFlow tensor; got {type(value).__module__}.{type(value).__name__}. "
        "Convert the value to a TensorFlow tensor first.",
        code="trace_option_unsupported",
        option="intervene",
    )


_BUILDERS: dict[str, Callable[[Any, Any], Callable[[Any], Any]]] = {
    "zero_ablate": _zero_ablate,
    "scale": _scale,
    "add": _add,
    "clamp": _clamp,
    "mean_ablate": _mean_ablate,
    "replace_with": _replace_with,
}


__all__ = ["TF_SUPPORTED_HELPER_NAMES", "tf_helper_hook"]
