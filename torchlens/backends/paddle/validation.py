"""Replay-validation helpers for the technical-preview Paddle backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ... import _state
from .._validation_shared import float_replay_tolerances, ops_by_label as _ops_by_label

_FACTORY_OR_SOURCE_OPS = {
    "arange",
    "eye",
    "full",
    "full_like",
    "linspace",
    "ones",
    "ones_like",
    "to_tensor",
    "zeros",
    "zeros_like",
}


@dataclass(frozen=True)
class RebuiltPaddleInputs:
    """Rebuilt Paddle call inputs from a captured argument template.

    Parameters
    ----------
    ok
        Whether the template was reconstructed without a coverage gap.
    args
        Rebuilt positional arguments.
    kwargs
        Rebuilt keyword arguments.
    reason
        Failure reason when ``ok`` is False.
    parent_values
        Distinct replay parent payloads keyed by raw producer label.
    leaf_paths_by_parent
        Template tensor leaf paths keyed by raw producer label.
    param_values
        Replay copies of module parameters keyed by template leaf path.
    """

    ok: bool
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    reason: str | None
    parent_values: dict[str, Any]
    leaf_paths_by_parent: dict[str, tuple[tuple[Any, ...], ...]]
    param_values: dict[tuple[Any, ...], Any] = field(default_factory=dict)


def _is_factory_or_source_capture(capture: Any) -> bool:
    """Return whether a capture may legitimately have no value parents.

    Parameters
    ----------
    capture
        Paddle operation capture record.

    Returns
    -------
    bool
        True for explicit factory/source allowlist operations.
    """

    op_name = str(getattr(capture, "op_name", ""))
    return op_name.rsplit(".", 1)[-1] in _FACTORY_OR_SOURCE_OPS or op_name == "input"


def _capture_output_path(capture: Any) -> tuple[Any, ...]:
    """Return the materialized output path used by a capture.

    Parameters
    ----------
    capture
        Paddle operation capture record.

    Returns
    -------
    tuple[Any, ...]
        Container path for the first tensor output.
    """

    paths = tuple(getattr(capture, "output_leaf_paths", ()))
    if not paths:
        return ()
    return tuple(paths[0])


def _rebuild_inputs(capture: Any, ops_by_label: Mapping[str, Any]) -> RebuiltPaddleInputs:
    """Reconstruct the exact Paddle call inputs from a capture template.

    Parameters
    ----------
    capture
        ``PaddleOpCapture`` emitted by the live wrapped operation.
    ops_by_label
        Materialized trace operations keyed by raw and public labels.

    Returns
    -------
    RebuiltPaddleInputs
        Rebuilt call arguments, or a fail-closed coverage result.
    """

    parent_values: dict[str, Any] = {}
    leaf_paths_by_parent: dict[str, list[tuple[Any, ...]]] = {}
    param_inputs: Mapping[tuple[Any, ...], Any] = getattr(capture, "param_inputs", None) or {}
    param_values: dict[tuple[Any, ...], Any] = {}

    def _rebuild_unlabeled(path: tuple[Any, ...]) -> Any:
        """Rebuild one unlabeled tensor leaf: a module parameter or a factory hole."""

        if path in param_inputs:
            param_values[path] = _parameter_replay_copy(param_inputs[path])
            return param_values[path]
        if _is_factory_or_source_capture(capture):
            return None
        raise ValueError(f"unlabeled tensor input leaf at {path!r}")

    def _rebuild_leaf(value: dict[str, Any], path: tuple[Any, ...]) -> Any:
        """Rebuild one tensor leaf marker from its parent payload."""

        label = value.get("label")
        if label is None:
            return _rebuild_unlabeled(path)
        if not isinstance(label, str) or label not in ops_by_label:
            raise ValueError(f"dangling tensor input label {label!r} at {path!r}")
        parent_value = _saved_payload(ops_by_label[label])
        parent_values.setdefault(label, parent_value)
        leaf_paths_by_parent.setdefault(label, []).append(path)
        return parent_value

    def _rebuild(value: Any, path: tuple[Any, ...]) -> Any:
        """Rebuild one nested template value."""

        if _is_tensor_marker(value):
            return _rebuild_leaf(value, path)
        if isinstance(value, tuple):
            return tuple(_rebuild(item, (*path, index)) for index, item in enumerate(value))
        if isinstance(value, list):
            return [_rebuild(item, (*path, index)) for index, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: _rebuild(item, (*path, key)) for key, item in value.items()}
        return value

    try:
        args = tuple(
            _rebuild(item, ("args", index))
            for index, item in enumerate(getattr(capture, "args_template", ()))
        )
        kwargs = {
            str(key): _rebuild(item, ("kwargs", str(key)))
            for key, item in getattr(capture, "kwargs_template", {}).items()
        }
    except (AttributeError, TypeError, ValueError) as exc:
        return RebuiltPaddleInputs(False, (), {}, str(exc), {}, {})
    return RebuiltPaddleInputs(
        True,
        args,
        kwargs,
        None,
        parent_values,
        {label: tuple(paths) for label, paths in leaf_paths_by_parent.items()},
        param_values,
    )


def _parameter_replay_copy(param: Any) -> Any:
    """Return a detached copy of a live parameter for replay.

    Replay reads the parameter's CURRENT value. A parameter changed since the
    forward makes the replay disagree with the saved output, so a stale value
    can only fail validation, never pass it; the copy keeps a replayed op
    that writes its inputs from mutating the user's model.

    Parameters
    ----------
    param
        Live Paddle parameter recorded by the capture.

    Returns
    -------
    Any
        Detached Paddle tensor with the same value, dtype and place.
    """

    import paddle

    with paddle.no_grad():
        return paddle.assign(param.detach())


def _payloads_close(a: Any, b: Any) -> bool:
    """Return whether two Paddle payloads are exactly or numerically close.

    Parameters
    ----------
    a
        Left Paddle payload.
    b
        Right Paddle payload.

    Returns
    -------
    bool
        True when shape, dtype, and values match within backend tolerances.
    """

    import paddle

    with _state.pause_logging(), paddle.no_grad():
        left = np.asarray(a.numpy())
        right = np.asarray(b.numpy())
    return _arrays_close(left, right)


def _arrays_close(left: np.ndarray, right: np.ndarray) -> bool:
    """Return whether two NumPy payload arrays agree within dtype-aware bands.

    Float tolerances are derived per dtype from its own machine epsilon
    (the ``utils.tensor_utils`` replay error model), replacing the former
    dtype-blind fp32 decimal pair (rtol 1e-5 / atol 1e-6) that was wrong in
    both directions: fp64 corruption thousands of times above fp64 round-off
    read as agreement, while a legitimate one-ULP fp16 storage-rounding
    difference false-failed.

    * Accumulating dtypes (eps <= fp32's): the legacy fp32 relative band
      rescaled by the eps ratio, so every dtype gets the SAME strictness
      measured in its own ULPs (fp32 keeps exactly the historical 1e-5).
    * Storage-rounding dtypes (eps > fp32's, i.e. fp16 transported as such):
      values compute in a wider dtype and round ONCE to storage, so the
      replay difference is a few storage ULPs (4-ULP headroom).
    * The absolute term only absorbs jitter at the bottom of the
      representable range (the relative band applied to the smallest normal
      value); the former 1e-6 floor blessed total corruption of every
      element below it.

    Parameters
    ----------
    left
        Left payload array.
    right
        Right payload array.

    Returns
    -------
    bool
        True when shape, dtype, and values match within backend tolerances.
    """

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.bool_) or np.issubdtype(left.dtype, np.integer):
        return bool(np.array_equal(left, right))
    if np.issubdtype(left.dtype, np.floating) or np.issubdtype(left.dtype, np.complexfloating):
        # The ONE shared eps-derived band (b5-opus R17-1 hoist); complex
        # payloads take the same component-eps band as mlx/jax (bit-exact
        # complex here false-failed legitimate replay jitter). equal_nan
        # matches this backend's own replay oracle (paddle/backend.py) and
        # every sibling: identical NaN patterns are agreement, NaN-vs-number
        # still fails elementwise.
        rtol, atol = float_replay_tolerances(np.finfo(left.dtype))
        return bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))
    return bool(np.array_equal(left, right))


def _perturb_candidates(value: Any) -> tuple[Any, ...]:
    """Return deterministic Paddle perturbation candidates for a parent value.

    Parameters
    ----------
    value
        Paddle tensor payload.

    Returns
    -------
    tuple[Any, ...]
        Perturbed tensors with matching dtype and place.
    """

    import paddle

    with _state.pause_logging(), paddle.no_grad():
        array = np.asarray(value.numpy())
        dtype = getattr(value, "dtype", None)
        place = getattr(value, "place", None)
        candidates: tuple[Any, ...]
        if np.issubdtype(array.dtype, np.bool_):
            candidates = (np.logical_not(array),)
        elif np.issubdtype(array.dtype, np.integer):
            candidates = (array + 1, np.zeros_like(array))
        else:
            magnitude = np.max(np.abs(array)).item() + 1.0 if array.size else 1.0
            candidates = (
                array + magnitude,
                array - magnitude,
                np.zeros_like(array),
            )
        tensors = tuple(
            paddle.to_tensor(candidate, dtype=dtype, place=place) for candidate in candidates
        )
    return tensors


def _parent_perturbations_change_output(
    backend: Any,
    capture: Any,
    ops_by_label: Mapping[str, Any],
    *,
    baseline_output: Any | None = None,
) -> bool:
    """Return whether at least one value-parent perturbation changes output.

    Parameters
    ----------
    backend
        Paddle backend instance.
    capture
        Paddle operation capture record.
    ops_by_label
        Materialized trace operations keyed by raw and public labels.
    baseline_output
        Optional replay baseline. For a corroborated user-intervened capture
        the sensitivity check compares perturbed replays against the pre-hook
        value: comparing against the recorded replacement payload would make
        the check vacuously pass for constant replacements.

    Returns
    -------
    bool
        True when perturbation is non-vacuous or the capture is an allowlisted
        zero-parent factory/source.
    """

    rebuilt = _rebuild_inputs(capture, ops_by_label)
    if not rebuilt.ok:
        return False
    if not rebuilt.parent_values:
        return _is_factory_or_source_capture(capture)
    saved_output = (
        baseline_output
        if baseline_output is not None
        else _saved_payload(ops_by_label[capture.label_raw])
    )
    output_path = _capture_output_path(capture)
    attempted = False
    for parent_label, parent_value in rebuilt.parent_values.items():
        paths = rebuilt.leaf_paths_by_parent.get(parent_label, ())
        if not paths:
            continue
        for candidate in _perturb_candidates(parent_value):
            attempted = True
            args, kwargs = _replace_template_paths(
                tuple(getattr(capture, "args_template", ())),
                dict(getattr(capture, "kwargs_template", {})),
                dict.fromkeys(paths, candidate),
                rebuilt,
            )
            try:
                with _state.pause_logging(), backend.paddle.no_grad():
                    perturbed = capture.func(*args, **kwargs)
            except Exception:
                continue
            perturbed_output = _value_at_path(perturbed, output_path)
            if not _payloads_close(perturbed_output, saved_output):
                return True
    return not attempted and _is_factory_or_source_capture(capture)


def _coverage_oracle(trace: Any) -> bool:
    """Fail closed on Paddle coverage gaps before replay validation.

    Parameters
    ----------
    trace
        Paddle trace containing independent ``PaddleOpCapture`` records.

    Returns
    -------
    bool
        True when capture records conserve tensor inputs and graph parents.
    """

    captures = tuple(getattr(trace, "_paddle_op_captures", ()))
    ops_by_label = _ops_by_label(trace)
    for capture in captures:
        if getattr(capture, "label_raw", None) not in ops_by_label:
            if tuple(getattr(capture, "alias_annotations", ())):
                continue
            return False
        is_factory = _is_factory_or_source_capture(capture)
        if not is_factory:
            for leaf in getattr(capture, "tensor_inputs", ()):
                if getattr(leaf, "label", None) is None and not _is_known_param_leaf(capture, leaf):
                    return False
            if tuple(getattr(capture, "capture_gap_markers", ())) != ():
                return False
        op = ops_by_label.get(getattr(capture, "label_raw", ""))
        if op is None:
            continue
        # Capture records speak RAW label space (their labels are immutable
        # capture identities). Recurrence grouping rewrites graph edges to
        # final pass-qualified labels, so parents are resolved back to raw
        # space before comparison; an unresolvable parent keeps its literal
        # label and fails closed.
        graph_parents: set[str] = set()
        for parent in getattr(op, "parents", ()):
            parent_text = str(parent)
            if parent_text.startswith("input."):
                continue
            parent_op = ops_by_label.get(parent_text)
            parent_raw = getattr(parent_op, "_label_raw", None) if parent_op is not None else None
            graph_parents.add(parent_raw if isinstance(parent_raw, str) else parent_text)
        for label in getattr(capture, "producer_labels", frozenset()):
            if not isinstance(label, str):
                return False
            if not label.startswith("input.") and label not in ops_by_label:
                return False
            if not label.startswith("input.") and label not in graph_parents:
                return False
    return True


def _is_known_param_leaf(capture: Any, leaf: Any) -> bool:
    """Return whether an unlabeled input leaf is a recorded module parameter.

    Parameters
    ----------
    capture
        Paddle operation capture record.
    leaf
        One of the capture's ``tensor_inputs``.

    Returns
    -------
    bool
        True only when the leaf carries a parameter address AND the capture
        holds the parameter object for replay at that path.
    """

    path = tuple(getattr(leaf, "path", ()))
    param_inputs = getattr(capture, "param_inputs", None) or {}
    return getattr(leaf, "param_address", None) is not None and path in param_inputs


def _is_tensor_marker(value: Any) -> bool:
    """Return whether a template value is a tensor leaf marker.

    Parameters
    ----------
    value
        Template value.

    Returns
    -------
    bool
        True when the value is a tensor marker dictionary.
    """

    return isinstance(value, dict) and value.get("kind") == "tensor"


def _saved_payload(op: Any) -> Any:
    """Return an op's saved output payload or raise on missing data.

    Parameters
    ----------
    op
        Materialized operation.

    Returns
    -------
    Any
        Saved Paddle tensor payload.
    """

    if not bool(getattr(op, "has_saved_activation", False)):
        raise ValueError("Paddle validation requires saved activation payloads.")
    output = op.out
    if output is None:
        raise ValueError("Paddle validation found a missing saved payload.")
    return output


def _value_at_path(value: Any, path: tuple[Any, ...]) -> Any:
    """Return a nested value at ``path``.

    Parameters
    ----------
    value
        Root value.
    path
        Container path.

    Returns
    -------
    Any
        Nested value.
    """

    result = value
    for part in path:
        result = result[part]
    return result


def _replace_template_paths(
    args_template: tuple[Any, ...],
    kwargs_template: dict[str, Any],
    replacements: Mapping[tuple[Any, ...], Any],
    rebuilt: RebuiltPaddleInputs,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Rebuild templates while substituting selected leaf paths.

    Parameters
    ----------
    args_template
        Positional argument template.
    kwargs_template
        Keyword argument template.
    replacements
        Replacements keyed by root-prefixed tensor leaf path.
    rebuilt
        Previously rebuilt call used as the source of parent payloads.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Rebuilt positional and keyword arguments.
    """

    parent_values = rebuilt.parent_values

    def _rebuild_leaf(value: dict[str, Any], path: tuple[Any, ...]) -> Any:
        """Rebuild one tensor leaf from a parent payload or a parameter copy."""

        label = value.get("label")
        if isinstance(label, str) and label in parent_values:
            return parent_values[label]
        if label is None and path in rebuilt.param_values:
            return rebuilt.param_values[path]
        raise ValueError(f"cannot rebuild tensor leaf at {path!r}")

    def _rebuild(value: Any, path: tuple[Any, ...]) -> Any:
        """Rebuild one value with replacements."""

        if path in replacements:
            return replacements[path]
        if _is_tensor_marker(value):
            return _rebuild_leaf(value, path)
        if isinstance(value, tuple):
            return tuple(_rebuild(item, (*path, index)) for index, item in enumerate(value))
        if isinstance(value, list):
            return [_rebuild(item, (*path, index)) for index, item in enumerate(value)]
        if isinstance(value, dict):
            return {key: _rebuild(item, (*path, key)) for key, item in value.items()}
        return value

    args = tuple(_rebuild(item, ("args", index)) for index, item in enumerate(args_template))
    kwargs = {
        str(key): _rebuild(item, ("kwargs", str(key))) for key, item in kwargs_template.items()
    }
    return args, kwargs


__all__ = [
    "RebuiltPaddleInputs",
    "_arrays_close",
    "_coverage_oracle",
    "_parent_perturbations_change_output",
    "_payloads_close",
    "_perturb_candidates",
    "_rebuild_inputs",
]
