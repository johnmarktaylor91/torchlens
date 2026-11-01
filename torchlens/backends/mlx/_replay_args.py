"""Saved-payload lookup and replay-argument reconstruction for MLX validation.

Extracted from ``validation`` so that module stays under the file cap.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ._capture_records import MLXOpCapture, _is_mlx_array, _ReplaySlot


def _saved_payload(trace: Any, ops_by_label: dict[str, Any], label_raw: str) -> Any:
    """Return the trace's saved payload for one raw label.

    Parameters
    ----------
    trace:
        Trace being validated.
    ops_by_label:
        Label-to-op index from :func:`_ops_by_label`.
    label_raw:
        Raw label whose payload is required.

    Returns
    -------
    Any
        Saved MLX array (public ``op.out`` or the selective-save hidden copy).
    """

    op = ops_by_label.get(label_raw)
    payload = None if op is None else getattr(op, "out", None)
    if payload is None:
        hidden = getattr(trace, "_selective_save_hidden_payloads", {}) or {}
        payload = hidden.get(label_raw)
    if payload is None:
        raise ValueError(f"MLX validation found no saved payload for {label_raw!r}.")
    return payload


def _capture_parent_labels(capture: MLXOpCapture) -> set[str]:
    """Return every recorded parent label across a capture's argument leaves.

    Parameters
    ----------
    capture:
        Captured call.

    Returns
    -------
    set[str]
        Non-``None`` per-leaf parent labels.
    """

    labels = {label for slot in capture.arg_leaf_labels for label in slot if label is not None}
    for slot in capture.kwarg_leaf_labels.values():
        labels.update(label for label in slot if label is not None)
    return labels


def _declared_parents_consistent(
    trace: Any,
    ops_by_label: dict[str, Any],
    capture: MLXOpCapture,
) -> bool:
    """Cross-check a capture's parent labels against the trace's declared graph.

    Parameters
    ----------
    trace:
        Trace being validated.
    ops_by_label:
        Label-to-op index from :func:`_ops_by_label`.
    capture:
        Captured call whose output ops carry the declared parent edges.

    Returns
    -------
    bool
        ``True`` when, for every materialized output op, the declared parent
        op set equals the parent op set implied by the recorded argument
        leaves (the op itself excluded on both sides, so aliasing outputs
        cannot self-shadow).
    """

    implied_ops = {
        id(ops_by_label[label])
        for label in _capture_parent_labels(capture)
        if label in ops_by_label
    }
    for label_raw in capture.labels_raw:
        op = ops_by_label.get(label_raw)
        if op is None:
            continue
        declared = getattr(op, "parents", None)
        if declared is None:
            return False
        declared_ops = {id(ops_by_label[parent]) for parent in declared if parent in ops_by_label}
        declared_ops.discard(id(op))
        implied_without_self = set(implied_ops)
        implied_without_self.discard(id(op))
        if declared_ops != implied_without_self:
            return False
    return True


def _reconstruct_value(
    trace: Any,
    ops_by_label: dict[str, Any],
    value: Any,
    leaf_labels: tuple[str | None, ...],
) -> Any:
    """Rebuild one argument from the trace's saved parent payloads.

    Parameters
    ----------
    trace:
        Trace being validated.
    ops_by_label:
        Label-to-op index from :func:`_ops_by_label`.
    value:
        Emit-time argument value.
    leaf_labels:
        Recorded per-leaf parent labels for ``value`` in flatten order.

    Returns
    -------
    Any
        ``value`` with every labeled array leaf replaced by the saved payload
        the declared graph stores for that label; unlabeled leaves (parameters,
        constants) keep their emit-time values. Emit-time reuse is legitimate
        ONLY because coverage pins each capture's leaf labels to the immutable
        emit-time inventory fingerprint first — a label stripped after capture
        fails coverage instead of laundering the emit-time array through
        replay.
    """

    return _rebuild_node(value, iter(leaf_labels), trace, ops_by_label)


def _rebuild_node(
    node: Any, cursor: Iterator[str | None], trace: Any, ops_by_label: dict[str, Any]
) -> Any:
    """Rebuild one template node, binding each slot to its recorded parent label.

    Raises
    ------
    ValueError
        If a ``_ReplaySlot`` has no recorded label, which means the template and
        the label fingerprint disagree. Fail closed rather than guess.
    """

    if _is_mlx_array(node) or isinstance(node, _ReplaySlot):
        label = next(cursor, None)
        if label is None:
            if isinstance(node, _ReplaySlot):
                # A slot with no recorded label means the template and
                # label fingerprint disagree; fail closed, never guess.
                raise ValueError("MLX replay template slot has no recorded parent label.")
            return node
        return _saved_payload(trace, ops_by_label, label)
    # Exact container types only (see ``build_capture_template``): a
    # module instance replays as itself, never as a plain dict.
    if type(node) is tuple:
        return tuple(_rebuild_node(item, cursor, trace, ops_by_label) for item in node)
    if type(node) is list:
        return [_rebuild_node(item, cursor, trace, ops_by_label) for item in node]
    if type(node) is dict:
        return {key: _rebuild_node(item, cursor, trace, ops_by_label) for key, item in node.items()}
    return node


def _reconstruct_call(
    trace: Any,
    ops_by_label: dict[str, Any],
    capture: MLXOpCapture,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Rebuild a capture's full call material from declared parent payloads.

    Parameters
    ----------
    trace:
        Trace being validated.
    ops_by_label:
        Label-to-op index from :func:`_ops_by_label`.
    capture:
        Captured call.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Positional and keyword arguments with labeled leaves sourced from the
        trace's saved payloads, so replay exercises the recorded graph wiring
        rather than trusting emit-time argument objects.
    """

    args = tuple(
        _reconstruct_value(
            trace,
            ops_by_label,
            value,
            capture.arg_leaf_labels[index] if index < len(capture.arg_leaf_labels) else (),
        )
        for index, value in enumerate(capture.args)
    )
    kwargs = {
        key: _reconstruct_value(
            trace,
            ops_by_label,
            value,
            capture.kwarg_leaf_labels.get(key, ()),
        )
        for key, value in capture.kwargs.items()
    }
    return args, kwargs
