"""Registry of salient argument extractors for common PyTorch operations.

Salient args are the computation-defining hyperparameters (kernel_size, stride,
num_heads, dropout_p, etc.) that researchers want to see at a glance without
enabling full ``save_function_args=True`` (which deep-copies everything).

The registry maps normalized ``layer_type`` strings to extractor functions.
Each extractor receives the raw args/kwargs and optionally parent_param_shapes,
and returns a ``Dict[str, Any]`` of simple Python types (int, float, str,
tuple, bool, None — never tensors).

Public entry point: ``extract_salient_args()``.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .. import _state as _st

# ---------------------------------------------------------------------------
# Registry: normalized_layer_type -> extractor function
# ---------------------------------------------------------------------------

_EXTRACTORS: Dict[str, Callable] = {}


def _register(*layer_types: str):
    """Decorator to register an extractor for one or more normalized layer types."""

    def decorator(fn: Callable) -> Callable:
        for lt in layer_types:
            _EXTRACTORS[lt] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_arg_name_map(func_name: str, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Map argument names to values using pre-computed ``_state._func_argnames``.

    Returns a dict combining positional args (mapped by name) and kwargs.
    If argnames aren't available, returns just kwargs.
    """
    argnames = _st._func_argnames.get(func_name.strip("_"), ())
    result = dict(kwargs)
    for i, val in enumerate(args):
        if i < len(argnames):
            name = argnames[i]
            if name not in result:  # kwargs take precedence
                result[name] = val
    return result


def _to_simple(val: Any) -> Any:
    """Ensure val is a simple Python type, never a tensor.

    Converts torch.Tensor to its shape tuple, recursively handles
    lists/tuples. Passes through int, float, str, bool, None.
    """
    if isinstance(val, torch.Tensor):
        return tuple(val.shape)
    if isinstance(val, (list, tuple)):
        converted = tuple(_to_simple(v) for v in val)
        return converted
    if isinstance(val, torch.dtype):
        return str(val)
    return val


def _get(mapping: dict, *keys: str, default=None) -> Any:
    """Get the first matching key from mapping."""
    for k in keys:
        if k in mapping:
            return _to_simple(mapping[k])
    return default


def _is_default(val: Any, *defaults) -> bool:
    """Check if val matches any of the given default values."""
    return val in defaults


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def _all_same(val, scalar_default):
    """Check if val is a tuple of all the same scalar_default value, or equals scalar_default."""
    if val == scalar_default:
        return True
    if isinstance(val, tuple) and all(v == scalar_default for v in val):
        return True
    return False


@_register("conv1d", "conv2d", "conv3d", "convolution")
def _conv(named: dict, param_shapes: list) -> dict:
    result = {}
    if len(param_shapes) >= 1:
        w = param_shapes[0]
        if len(w) >= 2:
            result["out_channels"] = w[0]
            result["in_channels"] = w[1]
        if len(w) >= 3:
            result["kernel_size"] = w[2:] if len(w) > 3 else w[2]
    stride = _get(named, "stride")
    if stride is not None and not _all_same(stride, 1):
        result["stride"] = stride
    padding = _get(named, "padding")
    if padding is not None and not _all_same(padding, 0):
        result["padding"] = padding
    dilation = _get(named, "dilation")
    if dilation is not None and not _all_same(dilation, 1):
        result["dilation"] = dilation
    groups = _get(named, "groups")
    if groups is not None and groups != 1:
        result["groups"] = groups
    return result


@_register("convtranspose1d", "convtranspose2d", "convtranspose3d")
def _conv_transpose(named: dict, param_shapes: list) -> dict:
    result = _conv(named, param_shapes)
    output_padding = _get(named, "output_padding")
    if output_padding is not None and not _all_same(output_padding, 0):
        result["output_padding"] = output_padding
    return result


@_register("linear")
def _linear(named: dict, param_shapes: list) -> dict:
    result = {}
    if len(param_shapes) >= 1:
        w = param_shapes[0]
        if len(w) == 2:
            result["out_features"] = w[0]
            result["in_features"] = w[1]
    return result


@_register("batchnorm", "batchnorm1d", "batchnorm2d", "batchnorm3d")
def _batch_norm(named: dict, param_shapes: list) -> dict:
    result = {}
    eps = _get(named, "eps")
    if eps is not None:
        result["eps"] = eps
    momentum = _get(named, "momentum")
    if momentum is not None:
        result["momentum"] = momentum
    return result


@_register("layernorm")
def _layer_norm(named: dict, param_shapes: list) -> dict:
    result = {}
    ns = _get(named, "normalized_shape")
    if ns is not None:
        result["normalized_shape"] = ns
    eps = _get(named, "eps")
    if eps is not None:
        result["eps"] = eps
    return result


@_register("groupnorm")
def _group_norm(named: dict, param_shapes: list) -> dict:
    result = {}
    ng = _get(named, "num_groups")
    if ng is not None:
        result["num_groups"] = ng
    eps = _get(named, "eps")
    if eps is not None:
        result["eps"] = eps
    return result


@_register("instancenorm", "instancenorm1d", "instancenorm2d", "instancenorm3d")
def _instance_norm(named: dict, param_shapes: list) -> dict:
    result = {}
    eps = _get(named, "eps")
    if eps is not None:
        result["eps"] = eps
    momentum = _get(named, "momentum")
    if momentum is not None:
        result["momentum"] = momentum
    return result


@_register("dropout", "dropout1d", "dropout2d", "dropout3d", "alphadropout", "featurealphadropout")
def _dropout(named: dict, param_shapes: list) -> dict:
    p = _get(named, "p")
    return {"p": p} if p is not None else {}


@_register("maxpool1d", "maxpool2d", "maxpool3d")
def _max_pool(named: dict, param_shapes: list) -> dict:
    result = {}
    ks = _get(named, "kernel_size")
    if ks is not None:
        result["kernel_size"] = ks
    stride = _get(named, "stride")
    if stride is not None:
        result["stride"] = stride
    padding = _get(named, "padding")
    if padding is not None and not _all_same(padding, 0):
        result["padding"] = padding
    return result


@_register("avgpool1d", "avgpool2d", "avgpool3d")
def _avg_pool(named: dict, param_shapes: list) -> dict:
    result = {}
    ks = _get(named, "kernel_size")
    if ks is not None:
        result["kernel_size"] = ks
    stride = _get(named, "stride")
    if stride is not None:
        result["stride"] = stride
    padding = _get(named, "padding")
    if padding is not None and not _all_same(padding, 0):
        result["padding"] = padding
    return result


@_register("adaptiveavgpool1d", "adaptiveavgpool2d", "adaptiveavgpool3d")
def _adaptive_avg_pool(named: dict, param_shapes: list) -> dict:
    os = _get(named, "output_size")
    return {"output_size": os} if os is not None else {}


@_register("adaptivemaxpool1d", "adaptivemaxpool2d", "adaptivemaxpool3d")
def _adaptive_max_pool(named: dict, param_shapes: list) -> dict:
    os = _get(named, "output_size")
    return {"output_size": os} if os is not None else {}


@_register("leakyrelu")
def _leaky_relu(named: dict, param_shapes: list) -> dict:
    ns = _get(named, "negative_slope")
    return {"negative_slope": ns} if ns is not None else {}


@_register("elu")
def _elu(named: dict, param_shapes: list) -> dict:
    a = _get(named, "alpha")
    return {"alpha": a} if a is not None else {}


@_register("hardtanh")
def _hardtanh(named: dict, param_shapes: list) -> dict:
    result = {}
    mn = _get(named, "min_val")
    if mn is not None:
        result["min_val"] = mn
    mx = _get(named, "max_val")
    if mx is not None:
        result["max_val"] = mx
    return result


@_register("threshold")
def _threshold(named: dict, param_shapes: list) -> dict:
    result = {}
    t = _get(named, "threshold")
    if t is not None:
        result["threshold"] = t
    v = _get(named, "value")
    if v is not None:
        result["value"] = v
    return result


@_register("softmax", "logsoftmax")
def _softmax(named: dict, param_shapes: list) -> dict:
    dim = _get(named, "dim")
    return {"dim": dim} if dim is not None else {}


@_register("scaleddotproductattention")
def _sdpa(named: dict, param_shapes: list) -> dict:
    result = {}
    dp = _get(named, "dropout_p")
    if dp is not None and dp != 0.0:
        result["dropout_p"] = dp
    ic = _get(named, "is_causal")
    if ic is not None and ic:
        result["is_causal"] = ic
    scale = _get(named, "scale")
    if scale is not None:
        result["scale"] = scale
    return result


@_register("interpolate", "upsample", "upsamplebilinear", "upsamplenearest")
def _interpolate(named: dict, param_shapes: list) -> dict:
    result = {}
    size = _get(named, "size")
    if size is not None:
        result["size"] = size
    sf = _get(named, "scale_factor")
    if sf is not None:
        result["scale_factor"] = sf
    mode = _get(named, "mode")
    if mode is not None:
        result["mode"] = mode
    return result


@_register("embedding")
def _embedding(named: dict, param_shapes: list) -> dict:
    result = {}
    if len(param_shapes) >= 1:
        w = param_shapes[0]
        if len(w) == 2:
            result["num_embeddings"] = w[0]
            result["embedding_dim"] = w[1]
    pi = _get(named, "padding_idx")
    if pi is not None:
        result["padding_idx"] = pi
    return result


@_register("cat")
def _cat(named: dict, param_shapes: list) -> dict:
    dim = _get(named, "dim")
    return {"dim": dim} if dim is not None else {}


@_register("stack", "hstack", "vstack", "dstack")
def _stack(named: dict, param_shapes: list) -> dict:
    dim = _get(named, "dim")
    return {"dim": dim} if dim is not None else {}


@_register("sum", "mean", "prod", "amax", "amin", "nansum", "nanmean")
def _reduction(named: dict, param_shapes: list) -> dict:
    result = {}
    dim = _get(named, "dim")
    if dim is not None:
        result["dim"] = dim
    kd = _get(named, "keepdim")
    if kd is not None and kd:
        result["keepdim"] = kd
    return result


@_register("max", "min", "argmax", "argmin")
def _reduction_simple(named: dict, param_shapes: list) -> dict:
    result = {}
    dim = _get(named, "dim")
    if dim is not None:
        result["dim"] = dim
    kd = _get(named, "keepdim")
    if kd is not None and kd:
        result["keepdim"] = kd
    return result


@_register("permute")
def _permute(named: dict, param_shapes: list) -> dict:
    dims = _get(named, "dims")
    return {"dims": dims} if dims is not None else {}


@_register("transpose")
def _transpose(named: dict, param_shapes: list) -> dict:
    result = {}
    d0 = _get(named, "dim0")
    if d0 is not None:
        result["dim0"] = d0
    d1 = _get(named, "dim1")
    if d1 is not None:
        result["dim1"] = d1
    return result


@_register("pad")
def _pad(named: dict, param_shapes: list) -> dict:
    result = {}
    p = _get(named, "pad")
    if p is not None:
        result["padding"] = p
    mode = _get(named, "mode")
    if mode is not None and mode != "constant":
        result["mode"] = mode
    val = _get(named, "value")
    if val is not None and val != 0:
        result["value"] = val
    return result


@_register("clamp", "clip")
def _clamp(named: dict, param_shapes: list) -> dict:
    result = {}
    mn = _get(named, "min")
    if mn is not None:
        result["min"] = mn
    mx = _get(named, "max")
    if mx is not None:
        result["max"] = mx
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_salient_args(
    layer_type: str,
    func_name: str,
    args: tuple,
    kwargs: dict,
    parent_param_shapes: Optional[List[Tuple]] = None,
) -> Dict[str, Any]:
    """Extract salient hyperparameters for a logged operation.

    Args:
        layer_type: Normalized layer type (lowercase, underscores stripped).
        func_name: Original function name (used for argname lookup).
        args: Positional arguments to the function.
        kwargs: Keyword arguments to the function.
        parent_param_shapes: Shapes of parent parameters (for deriving
            in/out channels, features, etc.).

    Returns:
        Dict of salient args. Empty ``{}`` for unregistered operations.
    """
    extractor = _EXTRACTORS.get(layer_type)
    if extractor is None:
        return {}
    try:
        named = _build_arg_name_map(func_name, args, kwargs)
        return extractor(named, parent_param_shapes or [])
    except Exception:
        # Never crash logging for salient args extraction failure.
        return {}
