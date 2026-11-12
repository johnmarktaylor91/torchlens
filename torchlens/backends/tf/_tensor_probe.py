"""TensorFlow tensor probes for the eager op-callback capture (file cap split).

Moved unchanged out of :mod:`.op_callback_capture`: byte sizes, shapes,
identity keys, tensor detection and source classification.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..._ir.refs import TensorRef


def _nbytes_from_tensor(tensor: Any) -> int | None:
    """Return TensorFlow tensor byte size when available.

    Parameters
    ----------
    tensor
        TensorFlow tensor.

    Returns
    -------
    int | None
        Byte size, or ``None``.
    """

    shape = _shape_tuple(tensor)
    dtype = getattr(tensor, "dtype", None)
    size = getattr(dtype, "size", None)
    if shape is None or size is None:
        return None
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel * int(size)


def _shape_tuple(tensor: Any) -> tuple[int, ...] | None:
    """Return a TensorFlow tensor shape tuple.

    Parameters
    ----------
    tensor
        TensorFlow tensor.

    Returns
    -------
    tuple[int, ...] | None
        Shape tuple, or ``None``.
    """

    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _tensor_ref_key(tensor: Any) -> object | None:
    """Return TensorFlow eager tensor identity key from ``tensor.ref()``.

    Parameters
    ----------
    tensor
        TensorFlow tensor.

    Returns
    -------
    object | None
        Hashable TensorFlow reference key.
    """

    ref = getattr(tensor, "ref", None)
    if not callable(ref):
        return None
    try:
        return ref()
    except TypeError:
        return None


def _is_tf_tensor(value: Any, tf: Any) -> bool:
    """Return whether ``value`` is a TensorFlow tensor-like object.

    Parameters
    ----------
    value
        Candidate value.
    tf
        Imported TensorFlow module.

    Returns
    -------
    bool
        True for tensors and variables.
    """

    tensor_type = getattr(tf, "Tensor", None)
    variable_type = getattr(tf, "Variable", None)
    return bool(
        (tensor_type is not None and isinstance(value, tensor_type))
        or (variable_type is not None and isinstance(value, variable_type))
    )


def _iter_tf_tensors(value: Any, tf: Any) -> list[Any]:
    """Return TensorFlow tensor leaves nested inside ``value``.

    Parameters
    ----------
    value
        Candidate nested value.
    tf
        Imported TensorFlow module.

    Returns
    -------
    list[Any]
        Tensor leaves.
    """

    if _is_tf_tensor(value, tf):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors: list[Any] = []
        for item in value:
            tensors.extend(_iter_tf_tensors(item, tf))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_iter_tf_tensors(item, tf))
        return tensors
    return []


def _classify_source_tensor(tensor: Any, consumer_op_type: str) -> str:
    """Classify a non-produced TensorFlow callback input source.

    Parameters
    ----------
    tensor
        Source tensor.
    consumer_op_type
        Consuming op type.

    Returns
    -------
    str
        Typed source kind.
    """

    dtype_text = str(getattr(tensor, "dtype", ""))
    if "resource" in dtype_text:
        return "resource"
    if consumer_op_type in {"Reshape", "Shape", "StridedSlice", "Pack", "ConcatV2"}:
        return "shape/axis"
    if _is_closure_constant(tensor):
        return "constant"
    return "constant/factory"


def _is_closure_constant(tensor: Any) -> bool:
    """Whether a non-produced input is recorded as a ``constant`` source op.

    The shared constants rule (SPEC section 2, rule 1) records every tensor
    the forward used that is neither an input, a parameter nor a captured
    output as a source op with a saved payload. TensorFlow converts Python
    scalars, axes, permutations and shapes to tensors before the callback
    sees them, so the rule is applied to floating or complex array-valued
    constants (rank one and above: closure weights and ``tf.constant([...])``
    literals). Rank-zero scalars and integer or boolean vectors (reduction
    axes, ``perm``, ``dims``, paddings) stay replayable ``constant/factory``
    values recorded on the consuming op's capture record.
    """

    dtype = getattr(tensor, "dtype", None)
    if dtype is None or not (
        bool(getattr(dtype, "is_floating", False)) or bool(getattr(dtype, "is_complex", False))
    ):
        return False
    shape = getattr(tensor, "shape", None)
    rank = getattr(shape, "rank", None)
    if rank is None:
        try:
            rank = len(shape) if shape is not None else None
        except TypeError:
            rank = None
    return rank is not None and rank >= 1


def _is_scalar_bool_tensor_ref(tensor_ref: TensorRef) -> bool:
    """Return whether a tensor ref describes a scalar bool.

    Parameters
    ----------
    tensor_ref
        Tensor metadata.

    Returns
    -------
    bool
        True for scalar bool tensors.
    """

    return tensor_ref.shape == () and "bool" in str(tensor_ref.dtype)


def _bool_value(payload: object | None) -> bool | None:
    """Return scalar bool payload as Python bool when possible.

    Parameters
    ----------
    payload
        Snapshotted payload.

    Returns
    -------
    bool | None
        Boolean value, or ``None``.
    """

    if isinstance(payload, np.ndarray) and payload.shape == () and payload.dtype == np.bool_:
        return bool(payload.item())
    if isinstance(payload, bool):
        return payload
    return None


def _is_scalar_bool_tensor(tensor: Any) -> bool:
    """Return whether a live TensorFlow tensor is a scalar bool.

    Parameters
    ----------
    tensor
        Eager TensorFlow tensor.

    Returns
    -------
    bool
        True for scalar bool tensors.
    """

    return _shape_tuple(tensor) == () and "bool" in str(getattr(tensor, "dtype", ""))


def _scalar_bool_value(tensor: Any) -> bool | None:
    """Return a live scalar bool tensor's concrete value (eager values are concrete).

    ``.numpy()`` on an eager tensor issues no TensorFlow op, so this is safe inside
    an op callback. Non-scalar or non-bool tensors return ``None``, as on torch.

    Parameters
    ----------
    tensor
        Eager TensorFlow tensor.

    Returns
    -------
    bool | None
        The scalar's value, or ``None``.
    """

    if not _is_scalar_bool_tensor(tensor):
        return None
    numpy_method = getattr(tensor, "numpy", None)
    if not callable(numpy_method):
        return None
    try:
        return _bool_value(np.asarray(numpy_method()))
    except Exception:
        return None
