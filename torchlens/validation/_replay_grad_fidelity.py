"""Replay arguments that carry the captured call's ``requires_grad`` and grad mode.

ATen can choose a kernel by whether an input requires grad and by grad mode: on
macOS arm64 a depthwise 3x3 conv runs ``Slow2d`` for a grad-requiring weight and
``Winograd3x3Depthwise`` otherwise, and the two accumulate in a different
order. Validation replay therefore hands each op private copies of its saved
arguments that require grad exactly when the captured arguments did, rebuilt
as leaves or non-leaves like them, and builds them outside any ambient
``torch.inference_mode`` (``execute_replay_func`` then restores the op's
recorded grad and inference mode). This makes replay more faithful; it widens
no comparison.

Helpers from ``core`` are imported lazily (``core`` imports this module).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import torch

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace

_R = TypeVar("_R")


def built_outside_inference_mode(build: Callable[..., _R]) -> Callable[..., _R]:
    """Run a replay-argument builder outside any ambient ``torch.inference_mode``.

    Replay tensors must be ordinary tensors whatever mode the caller validates
    under: an inference tensor carries no autograd graph and refuses in-place
    updates once the replay restores the op's captured (non-inference) mode.

    Parameters
    ----------
    build:
        Function that builds replay arguments.

    Returns
    -------
    Callable[..., _R]
        ``build`` wrapped in ``torch.inference_mode(False)``.
    """

    @functools.wraps(build)
    def wrapper(*args: Any, **kwargs: Any) -> _R:
        """Run ``build`` with inference mode off."""
        with torch.inference_mode(False):
            return build(*args, **kwargs)

    return wrapper


def with_slot_grad_flags(layer: Op, arg_type: str, key: Any, parent_layer: Op, value: Any) -> Any:
    """Give a parent payload headed for one replay slot that slot's captured grad flags.

    Parameters
    ----------
    layer:
        Op being replayed.
    arg_type:
        Either ``"args"`` or ``"kwargs"``.
    key:
        Parent-argument position key, possibly nested as a tuple.
    parent_layer:
        Op whose saved output fills the slot.
    value:
        Private replay value (a fresh clone or perturbation of the payload).

    Returns
    -------
    Any
        ``value`` unchanged unless it is a tensor whose captured argument
        required grad, else a grad-requiring copy (see
        ``_mirror_captured_requires_grad``).
    """

    if not isinstance(value, torch.Tensor):
        return value
    return _mirror_captured_requires_grad(
        value, *_captured_slot_grad_flags(layer, arg_type, key, parent_layer)
    )


def _mirror_captured_requires_grad(
    value: torch.Tensor, requires_grad: bool, leaf: bool
) -> torch.Tensor:
    """Give a private replay tensor the ``requires_grad`` its captured argument had.

    ATen's backend choice can depend on whether an input requires grad (on
    macOS arm64 a depthwise 3x3 conv picks ``Slow2d`` for a grad-requiring
    weight and ``Winograd3x3Depthwise`` otherwise; the two accumulate in a
    different order). A replay that hands the op detached copies can therefore
    call a different kernel than the captured run did, so each replay tensor
    requires grad exactly when the captured argument did. This makes replay
    more faithful; it widens no comparison.

    A captured non-leaf (a weight derived from a parameter, any op output) is
    rebuilt as a non-leaf with the same values and strides, so an in-place op
    that was legal on the captured value stays legal on the replay value. The
    autograd graph this creates is owned by the replay arguments and output
    alone and is released when the check that built them returns.

    Parameters
    ----------
    value:
        Replay tensor owned by validation (a fresh clone, never a saved payload).
    requires_grad:
        Whether the captured argument at this slot required grad.
    leaf:
        Whether the captured argument was an autograd leaf.

    Returns
    -------
    torch.Tensor
        ``value`` unchanged when no grad is needed (or its dtype cannot require
        grad), else a grad-requiring leaf or non-leaf with equal values.
    """

    if not requires_grad or value.requires_grad:
        return value
    if not (value.is_floating_point() or value.is_complex()):
        return value
    with torch.enable_grad():
        grad_leaf = value.detach().requires_grad_(True)
        return grad_leaf if leaf else grad_leaf.clone()


def replay_copy_for_slot(
    trace: Trace, layer: Op, arg_type: str, key: Any, value: torch.Tensor
) -> torch.Tensor:
    """Return a private replay copy of ``value`` carrying the slot's captured grad flags.

    Used where a replay splices a stored tensor (an edge-substitution payload)
    into an argument slot after ``_prepare_input_args_for_validating_layer``.

    Parameters
    ----------
    trace:
        Trace that owns ``layer``, used to resolve the slot's parent op.
    layer:
        Op being replayed.
    arg_type:
        Either ``"args"`` or ``"kwargs"``.
    key:
        Argument position key, possibly nested as a tuple.
    value:
        Stored tensor to splice; never mutated or handed to the replay itself.

    Returns
    -------
    torch.Tensor
        A fresh copy that requires grad exactly when the captured argument did.
    """

    from .core import _op_for_validation_label

    parent_label = (getattr(layer, "parent_arg_positions", {}) or {}).get(arg_type, {}).get(key)
    parent_layer = None
    if parent_label is not None:
        try:
            parent_layer = _op_for_validation_label(trace, parent_label)
        except (KeyError, ValueError):
            # An unresolvable parent label gives no producer fact: keep the
            # snapshot-only (non-leaf) reading.
            parent_layer = None
    with torch.inference_mode(False):
        return _mirror_captured_requires_grad(
            value.detach().clone(),
            *_captured_slot_grad_flags(layer, arg_type, key, parent_layer),
        )


def _snapshot_grad_flags(snapshot: Any) -> tuple[bool, bool]:
    """Return the captured ``(requires_grad, is_leaf)`` of one saved argument snapshot.

    Snapshots are copies taken with grad attached, so ``snapshot.requires_grad``
    is the captured argument's flag. Snapshots are copies of copies, so their
    autograd graph cannot tell a captured leaf from a captured user ``clone()``
    of one; a slot without a producing op is therefore rebuilt as a non-leaf,
    the direction in which every in-place op legal at capture stays legal.

    Parameters
    ----------
    snapshot:
        Saved argument value at one slot.

    Returns
    -------
    tuple[bool, bool]
        ``(False, False)`` for a non-tensor or a snapshot that needs no grad.
    """

    if not isinstance(snapshot, torch.Tensor) or not snapshot.requires_grad:
        return False, False
    return True, False


def _captured_slot_grad_flags(
    layer: Op, arg_type: str, key: Any, parent_layer: Op | None = None
) -> tuple[bool, bool]:
    """Return the captured ``(requires_grad, is_leaf)`` of the argument at one slot.

    Parameters
    ----------
    layer:
        Op being replayed; its ``saved_args`` / ``saved_kwargs`` hold the
        snapshots of the arguments its original call received.
    arg_type:
        Either ``"args"`` or ``"kwargs"``.
    key:
        Parent-argument position key, possibly nested as a tuple.
    parent_layer:
        Op whose output fills this slot, when it is a graph parent. Its
        recorded ``grad_fn_class_name`` is the autograd fact for leafness: no
        ``grad_fn`` means the consumed value was a leaf (a factory tensor made with
        ``requires_grad=True`` inside ``forward``; model inputs reach ops as
        TorchLens's grad-attached clones, so they are non-leaves). Any
        recorded node means a non-leaf. A same-object in-place return that
        leaves a leaf (``t.detach().requires_grad_()``) is recorded against
        TorchLens's safe copy (``CloneBackward0``) and so replays as a
        non-leaf: the safe direction, since every in-place op legal on the
        captured leaf stays legal on a non-leaf.

    Returns
    -------
    tuple[bool, bool]
        The captured flags; ``(False, False)`` (the detached-replay behavior)
        when no tensor snapshot sits at this position.
    """

    from .core import _read_replay_arg_value

    try:
        snapshot = _read_replay_arg_value(
            tuple(layer.saved_args or ()), dict(layer.saved_kwargs or {}), arg_type, key
        )
    except (IndexError, KeyError, TypeError):
        # No snapshot at this position (the parent fills a slot the saved args
        # do not carry): there is no captured flag to mirror.
        return False, False
    requires_grad, snapshot_leaf = _snapshot_grad_flags(snapshot)
    if not requires_grad or parent_layer is None:
        return requires_grad, snapshot_leaf
    return True, getattr(parent_layer, "grad_fn_class_name", None) is None


def _deep_clone_tensors(val: Any) -> Any:
    """Recursively clone all tensors in a nested structure of lists/tuples/dicts.

    Non-tensor leaves are returned as-is (shared reference).  Tensor leaves
    are detached and cloned so that in-place ops during validation replay
    don't corrupt the original saved data, then given the snapshot's
    ``requires_grad`` (see ``_mirror_captured_requires_grad``) so the replay
    reaches the same kernel the captured call did.

    Preserves container types: a tuple input produces a tuple output, not a list.
    """
    if isinstance(val, torch.Tensor):
        return _mirror_captured_requires_grad(val.detach().clone(), *_snapshot_grad_flags(val))
    elif isinstance(val, (list, tuple)):
        cloned = [_deep_clone_tensors(v) for v in val]
        # Preserve the original container type (list vs tuple vs namedtuple).
        if isinstance(val, tuple) and hasattr(val, "_fields"):
            return type(val)(*cloned)
        return type(val)(cloned)
    elif isinstance(val, dict):
        return {k: _deep_clone_tensors(v) for k, v in val.items()}
    return val


def _copy_validation_args(input_args: dict[str, Any]) -> dict[str, Any]:
    """Deep-clone replay arguments to avoid in-place mutation during validation.

    Parameters
    ----------
    input_args:
        Dictionary with ``"args"`` and ``"kwargs"`` entries holding replay
        inputs for a layer.

    Returns
    -------
    dict[str, Any]
        Structure-equivalent dictionary with tensor leaves detached, cloned and
        given their snapshot's ``requires_grad``.
    """
    return {
        "args": [_deep_clone_tensors(v) for v in input_args["args"]],
        "kwargs": {k: _deep_clone_tensors(v) for k, v in input_args["kwargs"].items()},
    }


def restore_leaf_for_requires_grad_toggle(layer: Op, input_args: dict[str, Any]) -> None:
    """Hand ``Tensor.requires_grad_`` the autograd leaf its captured call must have had.

    ``requires_grad_(False)`` is legal only on a leaf, so a captured call proves
    its receiver was a leaf (``requires_grad_(True)`` is legal on a leaf too).
    The receiver's producer may be recorded against TorchLens's safe copy
    (``CloneBackward0``), which would otherwise rebuild it as a non-leaf.

    Parameters
    ----------
    layer:
        Op being replayed.
    input_args:
        Prepared replay arguments, updated in place.

    Returns
    -------
    None
        ``input_args["args"][0]`` becomes a leaf with the same values and flag.
    """

    if getattr(layer, "func_name", None) != "requires_grad_" or not input_args["args"]:
        return
    receiver = input_args["args"][0]
    if isinstance(receiver, torch.Tensor) and receiver.requires_grad and not receiver.is_leaf:
        input_args["args"][0] = receiver.detach().requires_grad_(True)
