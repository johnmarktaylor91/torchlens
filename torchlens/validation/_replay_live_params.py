"""Live-parameter restoration for replay arguments.

Replay re-executes an op from its saved arguments. A saved Parameter argument is a
detached snapshot; when the source model is still available and the live
Parameter still equals that snapshot bit for bit, replay consumes the live
Parameter object instead, as the original call did (some kernels, notably
oneDNN convolutions, accumulate differently for a clone), and the autograd
boundary replays need the live Parameters to find the ops that read them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._utils.tensor_utils import tensor_nanequal

if TYPE_CHECKING:
    from .._data_classes.op import Op


def _restore_live_parameter_args_for_replay(input_args: dict[str, Any], layer: Op) -> None:
    """Restore live parameter objects into replay args when snapshots still match.

    Some PyTorch kernels, notably oneDNN-backed convolutions, can replay with
    slightly different accumulation when a model ``nn.Parameter`` is replaced
    by a detached tensor clone. The original call consumed the live parameter
    object, so validation should do the same when the source model is still
    available and the live parameter remains exactly equal to the saved
    argument snapshot. If the parameter has changed or cannot be resolved, the
    saved tensor clone is left in place so stale/corrupt captures still fail.

    Parameters
    ----------
    input_args:
        Replay argument mapping created from saved args/kwargs.
    layer:
        Operation being replayed.

    Returns
    -------
    None
        ``input_args`` is updated in place.
    """

    # An in-place op's receiver holds its PRE-mutation snapshot; the live Parameter
    # must not replace it. The autograd.backward recorder is the one in-place op
    # whose receiver (``accumulate_into``, the previous grad) is a parent path, and
    # its ``inputs`` leaves must be the live Parameters for the subgraph replay to
    # find the ops that read them.
    if getattr(layer, "is_inplace", False) and not _is_autograd_backward_boundary(layer):
        return
    param_logs = tuple(layer._param_logs or ())
    if not param_logs:
        return

    skip_paths = _parent_arg_paths(layer)
    live_params = tuple(_available_live_params(param_logs))
    if not live_params:
        return

    input_args["args"] = [
        _restore_live_parameter_leaf(arg, live_params, skip_paths["args"], (index,))
        for index, arg in enumerate(input_args["args"])
    ]
    input_args["kwargs"] = {
        key: _restore_live_parameter_leaf(value, live_params, skip_paths["kwargs"], (key,))
        for key, value in input_args["kwargs"].items()
    }


def _parent_arg_paths(layer: Op) -> dict[str, set[tuple[Any, ...]]]:
    """Return argument paths occupied by dataflow parent tensors.

    Parameters
    ----------
    layer:
        Operation whose parent argument positions should be skipped.

    Returns
    -------
    dict[str, set[tuple[Any, ...]]]
        Path sets for positional and keyword arguments.
    """

    parent_positions = getattr(layer, "parent_arg_positions", {}) or {}
    paths: dict[str, set[tuple[Any, ...]]] = {"args": set(), "kwargs": set()}
    for arg_type in ("args", "kwargs"):
        for key in parent_positions.get(arg_type, {}):
            paths[arg_type].add(key if isinstance(key, tuple) else (key,))
    return paths


def _available_live_params(param_logs: tuple[Any, ...]) -> list[torch.nn.Parameter]:
    """Resolve live parameters still reachable from a trace's source model.

    Parameters
    ----------
    param_logs:
        Parameter metadata objects attached to an op.

    Returns
    -------
    list[torch.nn.Parameter]
        Live parameters that could be resolved.
    """

    live_params: list[torch.nn.Parameter] = []
    for param_log in param_logs:
        try:
            live_param = getattr(param_log, "handle", None)
        except Exception:
            live_param = None
        if isinstance(live_param, torch.nn.Parameter):
            live_params.append(live_param)
    return live_params


def _restore_live_parameter_leaf(
    value: Any,
    live_params: tuple[torch.nn.Parameter, ...],
    skip_paths: set[tuple[Any, ...]],
    path: tuple[Any, ...],
) -> Any:
    """Replace a saved parameter clone with its matching live parameter.

    Parameters
    ----------
    value:
        Candidate replay argument value.
    live_params:
        Live parameters used by the operation.
    skip_paths:
        Argument paths occupied by graph parents.
    path:
        Current path within args or kwargs.

    Returns
    -------
    Any
        ``value`` with matching parameter leaves restored where unambiguous.
    """

    if path in skip_paths:
        return value
    if isinstance(value, torch.Tensor):
        matches = [
            live_param
            for live_param in live_params
            if _live_parameter_matches_snapshot(live_param, value)
        ]
        if len(matches) == 1:
            return matches[0]
        return value
    if isinstance(value, list):
        return [
            _restore_live_parameter_leaf(item, live_params, skip_paths, (*path, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        restored = [
            _restore_live_parameter_leaf(item, live_params, skip_paths, (*path, index))
            for index, item in enumerate(value)
        ]
        return type(value)(restored)
    if isinstance(value, dict):
        return {
            key: _restore_live_parameter_leaf(item, live_params, skip_paths, (*path, key))
            for key, item in value.items()
        }
    return value


def _live_parameter_matches_snapshot(
    live_param: torch.nn.Parameter,
    snapshot: torch.Tensor,
) -> bool:
    """Return whether a live parameter exactly matches a saved tensor snapshot.

    Parameters
    ----------
    live_param:
        Live model parameter that was used by an operation.
    snapshot:
        Saved replay argument tensor.

    Returns
    -------
    bool
        Whether shape, dtype, device, and values match exactly.
    """

    if tuple(live_param.shape) != tuple(snapshot.shape):
        return False
    if live_param.dtype != snapshot.dtype:
        return False
    if live_param.device != snapshot.device:
        return False
    return tensor_nanequal(live_param, snapshot, allow_tolerance=False)


def _is_autograd_backward_boundary(layer: Op) -> bool:
    """Return whether ``layer`` is a recorded in-forward ``autograd.backward`` op."""

    return getattr(layer, "transform_kind", None) == "autograd.backward"
