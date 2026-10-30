"""Parameter fields of a layer whose passes consume different parameters (step 14)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..quantities import Bytes

if TYPE_CHECKING:
    from ..data_classes.layer import Layer


def union_multipass_params(layer_log: Layer, pass_ops: list[Any]) -> None:
    """Make a layer whose passes consume different parameters list all of them.

    One call site run over different weights (a loop over a ``ParameterList``,
    one hook applied by a loop over distinct modules) is one layer, so its
    parameter fields are the first-seen union over passes, deduplicated by
    parameter identity, with counts and memory summed over that union. Each
    pass keeps its own parameters on ``layer.ops[k]``; the per-pass addresses
    are in ``annotations["varying_across_passes"]["params"]``.
    """

    union: dict[int, Any] = {}
    barcodes: dict[Any, None] = {}
    for op in pass_ops:
        for param_log in op._param_logs or ():
            union.setdefault(id(param_log), param_log)
        for barcode in op._param_barcodes or ():
            barcodes.setdefault(barcode, None)
    params = list(union.values())
    layer_log._param_logs = params
    layer_log._param_barcodes = list(barcodes)
    layer_log.param_shapes = [tuple(param.shape) for param in params]
    layer_log.num_params = sum(int(param.num_params) for param in params)
    layer_log.num_params_trainable = sum(
        int(param.num_params) for param in params if param.is_trainable
    )
    layer_log.num_params_frozen = sum(
        int(param.num_params) for param in params if not param.is_trainable
    )
    layer_log.total_param_memory = Bytes(sum(int(param.param_memory) for param in params))


def pass_param_addresses(pass_ops: list[Any]) -> list[list[str]]:
    """Per pass, the addresses of the parameters that pass consumed."""

    return [[param.address for param in (op._param_logs or ())] for op in pass_ops]
