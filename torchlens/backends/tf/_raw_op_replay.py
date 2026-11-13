"""Raw-op replay kernels for TensorFlow validation.

The dispatcher :func:`_replay_raw_op` re-executes one captured raw op from
its rebuilt callback inputs; the ``_replay_*`` kernels below hold the ops
whose attrs need decoding. Split out of :mod:`.validation` for the file cap;
the replay rules themselves are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ... import _state
from .op_callback_capture import TFOpCapture


def _replay_raw_op(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay one TensorFlow raw op from rebuilt callback inputs.

    Parameters
    ----------
    capture
        Callback capture containing op type and attrs.
    inputs
        Rebuilt TensorFlow inputs.

    Returns
    -------
    Any
        Replayed TensorFlow output tensor.
    """

    replay = _raw_op_dispatch().get(capture.op_type)
    if replay is None:
        raise ValueError(f"op is not raw-op replayable: {capture.op_type}")
    with _state.pause_logging():
        return replay(capture, inputs)


def _raw_op_dispatch() -> dict[str, Callable[[TFOpCapture, Sequence[Any]], Any]]:
    """Return the raw-op replay table keyed by TensorFlow op type."""

    return {
        "AddV2": lambda item, args: _raw(item).AddV2(x=args[0], y=args[1]),
        "AvgPool": _replay_pool,
        "BatchMatMulV2": _replay_batch_matmul_v2,
        "BiasAdd": _replay_bias_add,
        "BroadcastTo": lambda item, args: _raw(item).BroadcastTo(input=args[0], shape=args[1]),
        "Cast": _replay_cast,
        "ConcatV2": _replay_concat_v2,
        "Conv2D": _replay_conv2d,
        "DepthwiseConv2dNative": _replay_depthwise_conv2d,
        "Einsum": _replay_einsum,
        "Elu": lambda item, args: _raw(item).Elu(features=args[0]),
        "Erf": lambda item, args: _raw(item).Erf(x=args[0]),
        "Erfc": lambda item, args: _raw(item).Erfc(x=args[0]),
        "Exp": lambda item, args: _raw(item).Exp(x=args[0]),
        "ExpandDims": lambda item, args: _raw(item).ExpandDims(input=args[0], axis=args[1]),
        "Fill": lambda item, args: _raw(item).Fill(dims=args[0], value=args[1]),
        "GatherV2": _replay_gather_v2,
        "Greater": lambda item, args: _raw(item).Greater(x=args[0], y=args[1]),
        "Less": lambda item, args: _raw(item).Less(x=args[0], y=args[1]),
        "Log": lambda item, args: _raw(item).Log(x=args[0]),
        "MatMul": _replay_matmul,
        "Maximum": lambda item, args: _raw(item).Maximum(x=args[0], y=args[1]),
        "Minimum": lambda item, args: _raw(item).Minimum(x=args[0], y=args[1]),
        "Pow": lambda item, args: _raw(item).Pow(x=args[0], y=args[1]),
        "Prod": lambda item, args: _replay_reduction(item, args, "Prod"),
        "SelectV2": lambda item, args: _raw(item).SelectV2(condition=args[0], t=args[1], e=args[2]),
        "Sigmoid": lambda item, args: _raw(item).Sigmoid(x=args[0]),
        "Sin": lambda item, args: _raw(item).Sin(x=args[0]),
        "Slice": lambda item, args: _raw(item).Slice(input=args[0], begin=args[1], size=args[2]),
        "Sum": lambda item, args: _replay_reduction(item, args, "Sum"),
        "Tile": lambda item, args: _raw(item).Tile(input=args[0], multiples=args[1]),
        "MaxPool": _replay_pool,
        "Mean": _replay_mean,
        "Mul": lambda item, args: _raw(item).Mul(x=args[0], y=args[1]),
        "Neg": lambda item, args: _raw(item).Neg(x=args[0]),
        "Pad": lambda item, args: _raw(item).Pad(input=args[0], paddings=args[1]),
        "RealDiv": lambda item, args: _raw(item).RealDiv(x=args[0], y=args[1]),
        "Relu": lambda item, args: _raw(item).Relu(features=args[0]),
        "Relu6": lambda item, args: _raw(item).Relu6(features=args[0]),
        "Reshape": lambda item, args: _raw(item).Reshape(tensor=args[0], shape=args[1]),
        "Rsqrt": lambda item, args: _raw(item).Rsqrt(x=args[0]),
        "Softmax": lambda item, args: _raw(item).Softmax(logits=args[0]),
        "Sqrt": lambda item, args: _raw(item).Sqrt(x=args[0]),
        "SquaredDifference": lambda item, args: _raw(item).SquaredDifference(
            x=args[0],
            y=args[1],
        ),
        "Sub": lambda item, args: _raw(item).Sub(x=args[0], y=args[1]),
        "Squeeze": lambda item, args: _raw(item).Squeeze(
            input=args[0],
            axis=_optional_int_list(item.attrs.get("squeeze_dims")),
        ),
        "Tanh": lambda item, args: _raw(item).Tanh(x=args[0]),
        "Transpose": lambda item, args: _raw(item).Transpose(x=args[0], perm=args[1]),
    }


def _raw(_capture: TFOpCapture) -> Any:
    """Return TensorFlow raw ops module.

    Parameters
    ----------
    _capture
        Unused capture, accepted for dispatch lambdas.

    Returns
    -------
    Any
        ``tf.raw_ops``.
    """

    return _import_tensorflow().raw_ops


def _replay_batch_matmul_v2(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``BatchMatMulV2``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    return _raw(capture).BatchMatMulV2(
        x=inputs[0],
        y=inputs[1],
        adj_x=bool(capture.attrs.get("adj_x", False)),
        adj_y=bool(capture.attrs.get("adj_y", False)),
    )


def _replay_bias_add(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``BiasAdd``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    data_format = capture.attrs.get("data_format")
    kwargs = {} if data_format is None else {"data_format": _attr_str(data_format)}
    return _raw(capture).BiasAdd(value=inputs[0], bias=inputs[1], **kwargs)


def _replay_cast(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``Cast``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    return _raw(capture).Cast(
        x=inputs[0],
        DstT=capture.attrs["DstT"],
        Truncate=bool(capture.attrs.get("Truncate", False)),
    )


def _replay_concat_v2(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``ConcatV2``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    del capture
    return _import_tensorflow().raw_ops.ConcatV2(values=list(inputs[:-1]), axis=inputs[-1])


def _replay_conv2d(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``Conv2D``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    return _raw(capture).Conv2D(
        input=inputs[0],
        filter=inputs[1],
        strides=list(capture.attrs["strides"]),
        padding=_attr_str(capture.attrs["padding"]),
        explicit_paddings=list(capture.attrs.get("explicit_paddings", [])),
        data_format=_attr_str(capture.attrs.get("data_format", "NHWC")),
        dilations=list(capture.attrs.get("dilations", [1, 1, 1, 1])),
        use_cudnn_on_gpu=bool(capture.attrs.get("use_cudnn_on_gpu", True)),
    )


def _replay_depthwise_conv2d(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``DepthwiseConv2dNative``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    # ``tf.nn.depthwise_conv2d`` records unset attrs as ``None``.
    return _raw(capture).DepthwiseConv2dNative(
        input=inputs[0],
        filter=inputs[1],
        strides=list(capture.attrs["strides"]),
        padding=_attr_str(capture.attrs["padding"]),
        explicit_paddings=list(capture.attrs.get("explicit_paddings") or []),
        data_format=_attr_str(capture.attrs.get("data_format") or "NHWC"),
        dilations=list(capture.attrs.get("dilations") or [1, 1, 1, 1]),
    )


def _replay_einsum(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``Einsum``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    return _raw(capture).Einsum(
        inputs=list(inputs),
        equation=_attr_str(capture.attrs["equation"]),
    )


def _replay_matmul(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``MatMul``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    return _raw(capture).MatMul(
        a=inputs[0],
        b=inputs[1],
        transpose_a=bool(capture.attrs.get("transpose_a", False)),
        transpose_b=bool(capture.attrs.get("transpose_b", False)),
    )


def _replay_reduction(capture: TFOpCapture, inputs: Sequence[Any], op_name: str) -> Any:
    """Replay ``Sum`` or ``Prod`` (same raw-op signature as ``Mean``).

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.
    op_name
        ``"Sum"`` or ``"Prod"``.

    Returns
    -------
    Any
        Replayed output.
    """

    return getattr(_raw(capture), op_name)(
        input=inputs[0],
        axis=inputs[1],
        keep_dims=bool(capture.attrs.get("keep_dims", False)),
    )


def _replay_gather_v2(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``GatherV2`` (``tf.gather``; an embedding lookup).

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs: params, indices and axis.

    Returns
    -------
    Any
        Replayed output.
    """

    return _raw(capture).GatherV2(
        params=inputs[0],
        indices=inputs[1],
        axis=inputs[2],
        batch_dims=int(capture.attrs.get("batch_dims", 0)),
    )


def _replay_mean(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``Mean``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    return _raw(capture).Mean(
        input=inputs[0],
        axis=inputs[1],
        keep_dims=bool(capture.attrs.get("keep_dims", False)),
    )


def _replay_pool(capture: TFOpCapture, inputs: Sequence[Any]) -> Any:
    """Replay ``MaxPool`` or ``AvgPool``.

    Parameters
    ----------
    capture
        Callback capture.
    inputs
        Rebuilt inputs.

    Returns
    -------
    Any
        Replayed output.
    """

    raw_op = getattr(_raw(capture), capture.op_type)
    return raw_op(
        input=inputs[0],
        ksize=list(capture.attrs["ksize"]),
        strides=list(capture.attrs["strides"]),
        padding=_attr_str(capture.attrs["padding"]),
        explicit_paddings=list(capture.attrs.get("explicit_paddings", [])),
        data_format=_attr_str(capture.attrs.get("data_format", "NHWC")),
    )


def _attr_str(value: Any) -> str:
    """Return a TensorFlow string attr as text.

    Parameters
    ----------
    value
        TensorFlow callback attr value.

    Returns
    -------
    str
        Decoded text attr.
    """

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _optional_int_list(value: Any) -> list[int]:
    """Return a TensorFlow optional int-list attr.

    Parameters
    ----------
    value
        Callback attr value.

    Returns
    -------
    list[int]
        Empty list for omitted attrs, otherwise the integer list.
    """

    if value is None:
        return []
    return [int(item) for item in value]


def _import_tensorflow() -> Any:
    """Import TensorFlow lazily.

    Returns
    -------
    Any
        Imported TensorFlow module.
    """

    import tensorflow as tf

    return tf
