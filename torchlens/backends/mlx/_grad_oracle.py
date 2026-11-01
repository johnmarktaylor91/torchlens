"""MLX derived-gradient oracle helpers.

Module-level functions behind ``MLXBackend``'s derived-gradient preview: the
leaf-gradient record builder, the conservative intermediate-signature
grouping, the parent-perturbation oracle pass and the tree closeness checks.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np

from ..._data_classes.derived_grad import DerivedGradRecord
from ..._ir.refs import DtypeRef
from .._finalize import join_module_address as _join_module_address
from .._validation_shared import float_replay_tolerances
from ._tap_observers import (
    _is_mlx_array,
    _MLXBoundaryReplacementObserver,
    _MLXIntermediateSignature,
)
from .wrappers import mlx_tap_observer

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from .backend import MLXBackend


def _has_mlx_array_leaf(value: Any) -> bool:
    """Return whether ``value`` contains at least one MLX array leaf.

    Parameters
    ----------
    value
        Candidate tree.

    Returns
    -------
    bool
        True when an MLX array appears in ``value``.
    """

    return any(True for _path, _leaf in _flatten_mlx_array_tree(value, ""))


def _flatten_mlx_array_tree(value: Any, prefix: str) -> Iterator[tuple[str, Any]]:
    """Yield MLX array leaves from a nested tree.

    Parameters
    ----------
    value
        Tree value to flatten.
    prefix
        Dotted path prefix.

    Yields
    ------
    tuple[str, Any]
        Local dotted path and array leaf.
    """

    if _is_mlx_array(value):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = _join_module_address(prefix, str(key)) if prefix else str(key)
            yield from _flatten_mlx_array_tree(item, child_prefix)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_prefix = _join_module_address(prefix, str(index)) if prefix else str(index)
            yield from _flatten_mlx_array_tree(item, child_prefix)


def _records_for_mlx_grad_tree(
    *,
    grad_tree: Any,
    argnum: int,
    param_argnum: int | None,
    input_argnum_by_value_argnum: Mapping[int, int],
    provenance: Mapping[str, Any],
) -> dict[str, DerivedGradRecord]:
    """Build derived-gradient records for one differentiated MLX arg tree.

    Parameters
    ----------
    grad_tree
        Gradient tree returned by MLX AD.
    argnum
        Backend positional argnum for this gradient tree.
    param_argnum
        Backend positional argnum containing params, if present.
    input_argnum_by_value_argnum
        Mapping from backend value argnum to user input argnum.
    provenance
        Shared provenance metadata.

    Returns
    -------
    dict[str, DerivedGradRecord]
        Records keyed by stable leaf path.
    """

    records: dict[str, DerivedGradRecord] = {}
    for local_path, grad in _flatten_mlx_array_tree(grad_tree, ""):
        if argnum == param_argnum:
            source = "params"
            input_argnum = None
            full_path = f"params.{local_path}" if local_path else "params"
        else:
            source = "inputs"
            input_argnum = input_argnum_by_value_argnum[argnum]
            prefix = f"inputs.{input_argnum}"
            full_path = f"{prefix}.{local_path}" if local_path else prefix
        records[full_path] = DerivedGradRecord(
            path=full_path,
            source=source,
            argnum=argnum,
            input_argnum=input_argnum,
            aval=f"array(shape={tuple(getattr(grad, 'shape', ()))}, dtype={getattr(grad, 'dtype', None)})",
            dtype_ref=DtypeRef(backend="mlx", name=str(getattr(grad, "dtype", ""))),
            grad=grad,
            provenance=provenance,
        )
    return records


def _mlx_trace_intermediate_signatures(
    ops: Sequence[Any],
) -> dict[_MLXIntermediateSignature, list[Any]]:
    """Group finalized MLX ops by conservative intermediate signatures.

    Parameters
    ----------
    ops
        Saved, non-input op logs selected for intermediate derived gradients.

    Returns
    -------
    dict[_MLXIntermediateSignature, list[Any]]
        Trace ops grouped by attachment signature.

    Notes
    -----
    Replay-side signatures speak RAW label space (the tap observer labels
    values with ``_label_raw``), so recurrence-grouped parents (rewritten to
    final labels at finalize -- the pass-qualified ``op.label`` for a
    multi-pass referenced layer, but the BARE ``op.layer_label`` (torch
    parity) for a single-pass one) are resolved back to raw space here. Both
    final spellings must resolve: the pass-qualified key always, and the
    bare key too for single-pass ops (unambiguous there; omitted for
    multi-pass ops, where the bare label would collide across passes and
    ``parents`` never uses it anyway). Without this, every grouped
    intermediate would silently fail to match its replay candidate.
    """

    final_to_raw: dict[str, str] = {}
    for op in ops:
        source_trace = op._source_trace_or_none()
        if source_trace is None:
            continue
        for trace_op in getattr(source_trace, "_layer_list", ()):
            label_raw = getattr(trace_op, "_label_raw", None)
            if not isinstance(label_raw, str):
                continue
            label = getattr(trace_op, "label", None)
            if isinstance(label, str):
                final_to_raw[label] = label_raw
            layer_label = getattr(trace_op, "layer_label", None)
            if isinstance(layer_label, str) and int(getattr(trace_op, "num_passes", 1)) == 1:
                final_to_raw[layer_label] = label_raw
        break
    grouped: dict[_MLXIntermediateSignature, list[Any]] = defaultdict(list)
    for op in ops:
        signature = _MLXIntermediateSignature(
            op_name=str(op.func_name or op.type),
            call_ordinal=int(op.func_call_id or 0),
            parent_labels=tuple(
                final_to_raw.get(str(parent), str(parent)) for parent in getattr(op, "parents", ())
            ),
            shape=tuple(getattr(op, "shape", ()) or ()),
            dtype=str(getattr(op, "dtype", "")),
            module_calls=tuple(str(module) for module in getattr(op, "modules", ())),
        )
        grouped[signature].append(op)
    return grouped


class _MLXReplayCall(NamedTuple):
    """The derived-gradient replay call: trace, AD value args, model kwargs, loss fn."""

    trace: Trace
    value_args: tuple[Any, ...]
    kwargs: Mapping[Any, Any]
    value_fn: Callable[..., tuple[Any, Any]]


def _replacement_loss_fn(
    backend: MLXBackend,
    signature: _MLXIntermediateSignature,
    replay: _MLXReplayCall,
) -> Callable[[Any], Any]:
    """Build the scalar-loss function that replays with one boundary value replaced."""

    trace, value_args, kwargs, value_fn = replay

    def loss_from_replacement(replacement: Any) -> Any:
        """Return scalar loss with one replay boundary replaced.

        Parameters
        ----------
        replacement
            Replacement boundary value.

        Returns
        -------
        Any
            Scalar MLX loss.
        """

        observer = _MLXBoundaryReplacementObserver(backend, trace, signature, replacement)
        observer.label_call_inputs(_mlx_model_args_from_value_args(value_args, trace), kwargs)
        with mlx_tap_observer(observer):
            loss, _raw_output = value_fn(*value_args)
        if not observer.replaced:
            raise ValueError("MLX intermediate replacement oracle did not find the target.")
        return loss

    return loss_from_replacement


def _mlx_intermediate_oracle_passes(
    *,
    backend: MLXBackend,
    signature: _MLXIntermediateSignature,
    value: Any,
    producer_grad: Any,
    replay: _MLXReplayCall,
) -> bool:
    """Return whether replacement-gradient and perturbation checks pass.

    Parameters
    ----------
    backend
        Active MLX backend instance.
    signature
        Boundary signature being validated.
    value
        Boundary value from the tap replay.
    producer_grad
        Candidate cotangent produced by the custom-VJP tap.
    replay
        The finalized trace (aligns replay labels), the ``value_fn`` args, the
        model's original keyword arguments and the scalar-loss ``value_fn``.

    Returns
    -------
    bool
        True only when independent replacement AD and perturbation agree.
    """

    mx = cast(Any, backend.mx)
    loss_from_replacement = _replacement_loss_fn(backend, signature, replay)
    try:
        reference_grad = mx.grad(loss_from_replacement)(value)
        mx.eval(reference_grad, producer_grad)
        if not _mlx_values_close(producer_grad, reference_grad):
            return False
        perturbation = _mlx_perturbation_for(value)
        perturbed_grad = mx.grad(loss_from_replacement)(value + perturbation)
        expected_perturbed_grad = mx.grad(loss_from_replacement)(value + perturbation)
        mx.eval(perturbed_grad, expected_perturbed_grad)
    except Exception:
        return False
    if not _mlx_values_close(perturbed_grad, expected_perturbed_grad):
        return False
    return not _mlx_values_close(reference_grad, perturbed_grad)


def _mlx_model_args_from_value_args(value_args: tuple[Any, ...], trace: Trace) -> tuple[Any, ...]:
    """Return model input args from AD value args.

    Parameters
    ----------
    value_args
        Arguments passed to ``mx.value_and_grad``.
    trace
        Finalized trace whose parameter source indicates whether arg 0 is params.

    Returns
    -------
    tuple[Any, ...]
        Positional model input arguments.
    """

    if (
        getattr(trace, "_provenance_param_source", "none") == "native-module"
        and len(value_args) > 1
    ):
        return value_args[1:]
    return value_args


def _mlx_perturbation_for(value: Any) -> Any:
    """Return a small dtype-scaled MLX perturbation for a boundary value.

    Parameters
    ----------
    value
        Boundary MLX array.

    Returns
    -------
    Any
        Perturbation array with the same shape as ``value``.
    """

    import mlx.core as mx

    dtype_name = str(getattr(value, "dtype", ""))
    epsilon = 1e-2 if "float16" in dtype_name or "bfloat16" in dtype_name else 1e-3
    return mx.ones_like(value) * epsilon


def _is_scalar_mlx_value(value: Any) -> bool:
    """Return whether an MLX value is scalar-shaped.

    Parameters
    ----------
    value
        Candidate MLX value.

    Returns
    -------
    bool
        True when ``value`` has shape ``()``.
    """

    return tuple(getattr(value, "shape", ())) == ()


def _eval_mlx_tree(mx: Any, value: Any) -> None:
    """Force all MLX array leaves in ``value``.

    Parameters
    ----------
    mx
        Imported ``mlx.core`` module.
    value
        Tree whose array leaves should be evaluated.

    Returns
    -------
    None
        MLX evaluation has been requested for all leaves.
    """

    leaves = [leaf for _path, leaf in _flatten_mlx_array_tree(value, "")]
    if leaves:
        mx.eval(*leaves)


def _mlx_trees_close(left: Any, right: Any) -> bool:
    """Return whether two MLX output trees are numerically close.

    Parameters
    ----------
    left
        Left output tree.
    right
        Right output tree.

    Returns
    -------
    bool
        True when both trees have matching structure and close leaves.
    """

    if _is_mlx_array(left) or _is_mlx_array(right):
        if not (_is_mlx_array(left) and _is_mlx_array(right)):
            return False
        return _mlx_values_close(left, right)
    if isinstance(left, tuple) or isinstance(right, tuple):
        if not (isinstance(left, tuple) and isinstance(right, tuple)):
            return False
        return len(left) == len(right) and all(
            _mlx_trees_close(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, list) or isinstance(right, list):
        if not (isinstance(left, list) and isinstance(right, list)):
            return False
        return len(left) == len(right) and all(
            _mlx_trees_close(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            return False
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_mlx_trees_close(left[key], right[key]) for key in left)
    return left == right


def _mlx_values_close(left: Any, right: Any) -> bool:
    """Return whether two MLX arrays are numerically close.

    Parameters
    ----------
    left
        Left MLX array.
    right
        Right MLX array.

    Returns
    -------
    bool
        True when arrays have matching shape, dtype, and values.
    """

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return False
    if np.issubdtype(left_array.dtype, np.bool_) or np.issubdtype(left_array.dtype, np.integer):
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    if np.issubdtype(left_array.dtype, np.floating) or np.issubdtype(
        left_array.dtype,
        np.complexfloating,
    ):
        # Per-dtype ULP-derived bands (ported paddle/mlx validation-oracle
        # derivation); ``np.finfo`` reports component precision for complex.
        rtol, atol = float_replay_tolerances(np.finfo(left_array.dtype))
        return bool(np.allclose(left_array, right_array, rtol=rtol, atol=atol, equal_nan=True))
    return bool(np.array_equal(left_array, right_array))
