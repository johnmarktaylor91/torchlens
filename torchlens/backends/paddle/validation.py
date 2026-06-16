"""Replay-validation helpers for the technical-preview Paddle backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ... import _state

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
    """

    ok: bool
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    reason: str | None
    parent_values: dict[str, Any]
    leaf_paths_by_parent: dict[str, tuple[tuple[Any, ...], ...]]


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

    def _rebuild(value: Any, path: tuple[Any, ...]) -> Any:
        """Rebuild one nested template value."""

        if _is_tensor_marker(value):
            label = value.get("label")
            if label is None:
                if _is_factory_or_source_capture(capture):
                    return None
                raise ValueError(f"unlabeled tensor input leaf at {path!r}")
            if not isinstance(label, str) or label not in ops_by_label:
                raise ValueError(f"dangling tensor input label {label!r} at {path!r}")
            parent_op = ops_by_label[label]
            parent_value = _saved_payload(parent_op)
            parent_values.setdefault(label, parent_value)
            leaf_paths_by_parent.setdefault(label, []).append(path)
            return parent_value
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
    )


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
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.bool_) or np.issubdtype(left.dtype, np.integer):
        return bool(np.array_equal(left, right))
    if np.issubdtype(left.dtype, np.floating):
        return bool(np.allclose(left, right, rtol=1e-5, atol=1e-6))
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
    saved_output = _saved_payload(ops_by_label[getattr(capture, "label_raw")])
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
                {path: candidate for path in paths},
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
                if getattr(leaf, "label", None) is None:
                    return False
            if tuple(getattr(capture, "capture_gap_markers", ())) != ():
                return False
        op = ops_by_label.get(getattr(capture, "label_raw", ""))
        if op is None:
            continue
        graph_parents = {
            str(parent)
            for parent in getattr(op, "parents", ())
            if not str(parent).startswith("input.")
        }
        for label in getattr(capture, "producer_labels", frozenset()):
            if not isinstance(label, str):
                return False
            if not label.startswith("input.") and label not in ops_by_label:
                return False
            if not label.startswith("input.") and label not in graph_parents:
                return False
    return True


def _ops_by_label(trace: Any) -> dict[str, Any]:
    """Return materialized trace operations keyed by all known labels.

    Parameters
    ----------
    trace
        Materialized TorchLens trace.

    Returns
    -------
    dict[str, Any]
        Operations keyed by raw, layer, and pass labels.
    """

    result: dict[str, Any] = {}
    for op in getattr(trace, "layer_list", ()):
        for label in (
            getattr(op, "_label_raw", None),
            getattr(op, "layer_label", None),
            getattr(op, "label", None),
        ):
            if isinstance(label, str):
                result[label] = op
    return result


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

    def _rebuild(value: Any, path: tuple[Any, ...]) -> Any:
        """Rebuild one value with replacements."""

        if path in replacements:
            return replacements[path]
        if _is_tensor_marker(value):
            label = value.get("label")
            if isinstance(label, str) and label in parent_values:
                return parent_values[label]
            raise ValueError(f"cannot rebuild tensor leaf at {path!r}")
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
    "_coverage_oracle",
    "_parent_perturbations_change_output",
    "_payloads_close",
    "_perturb_candidates",
    "_rebuild_inputs",
]
