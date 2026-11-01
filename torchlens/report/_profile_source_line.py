"""Source-line grouping for ``Trace.profile(group_by="source_line")``.

The grouped view folds the op-level profile rows into one row per user
source line, so the activation bytes, FLOPs and instrumented time of every
op a line launched read as one figure. It adds no new measurement: each
group is the sum of the op rows it holds, and the groups partition the op
rows exactly (boundary rows fold into one ``boundary`` row), so every
additive column of the grouped view sums to the ungrouped op view.

The line of an op is its source-location witness
(:func:`torchlens._data_classes._site_key.operation_witness`): the deepest
operation frame of its recorded ``code_context``, the same selector the
site-key join uses. Spelling DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

from typing import Any

#: The only grouping key ``profile(group_by=...)`` accepts today.
PROFILE_GROUP_BY_KEYS: tuple[str, ...] = ("source_line",)

#: Row name for input/output boundary rows (they launch no op of their own).
BOUNDARY_GROUP_NAME = "<inputs and outputs>"

#: Row name for ops whose source location was not recorded.
UNKNOWN_SOURCE_GROUP_NAME = "<source line unknown>"

_ADDITIVE_COLUMNS: tuple[str, ...] = ("time", "flops", "flops_pct", "activation_memory")


def source_line_key(row: dict[str, Any], op: Any) -> str:
    """Return the group name of one op-level row.

    Parameters
    ----------
    row:
        The op-level profile row built for ``op``.
    op:
        The op record the row describes.

    Returns
    -------
    str
        ``"<file>:<line>"``, or one of the two sentinel group names.
    """

    from .._data_classes._site_key import operation_witness

    if row.get("kind") == "boundary":
        return BOUNDARY_GROUP_NAME
    witness = operation_witness(op)
    if witness is None or witness[0] is None or witness[1] is None:
        return UNKNOWN_SOURCE_GROUP_NAME
    return f"{witness[0]}:{witness[1]}"


def _sum_or_none(values: list[Any]) -> Any:
    """Sum the available values; ``None`` when none is available."""

    available = [value for value in values if value is not None]
    return sum(available) if available else None


def _fold_saved(values: list[Any]) -> bool | str | None:
    """Fold per-row saved-activation states into one group state."""

    states = {value for value in values if value is not None}
    if not states:
        return None
    if states == {True}:
        return True
    if states == {False}:
        return False
    return "partial"


def _union_values(values: list[Any]) -> str | None:
    """Union comma-separated value sets (dtype, device) across rows."""

    merged = sorted({part for value in values if value for part in str(value).split(", ")})
    return ", ".join(merged) if merged else None


def fold_rows(name: str, members: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold op-level rows into one source-line row.

    Parameters
    ----------
    name:
        The group name.
    members:
        The op-level rows of the group, in execution order.

    Returns
    -------
    dict[str, Any]
        One profile row: additive columns summed, ``param_count`` left
        ``None`` (ops of one line can share parameters, so a sum would
        double-count them).
    """

    is_boundary = name == BOUNDARY_GROUP_NAME
    row: dict[str, Any] = {
        "name": name,
        "kind": "boundary" if is_boundary else "source_line",
        "role": None if is_boundary else "SUBTOTAL",
        "op_count": sum(int(member.get("op_count") or 0) for member in members),
        "saved_activation": _fold_saved([member.get("saved_activation") for member in members]),
        "param_count": None,
        "dtype": _union_values([member.get("dtype") for member in members]),
        "device": _union_values([member.get("device") for member in members]),
    }
    for column in _ADDITIVE_COLUMNS:
        row[column] = _sum_or_none([member.get(column) for member in members])
    return row


def group_rows_by_source_line(
    op_rows: list[dict[str, Any]],
    ops: list[Any],
    row_evidence: list[str | None],
    weakest_evidence: Any,
) -> tuple[list[dict[str, Any]], list[str | None]]:
    """Group op-level rows by source line in first-occurrence order.

    Parameters
    ----------
    op_rows:
        Op-level rows, index-aligned with ``ops``.
    ops:
        The trace's op records (``trace._layer_list``).
    row_evidence:
        Per-row FLOPs evidence labels, index-aligned with ``op_rows``.
    weakest_evidence:
        The profile's weakest-member evidence aggregator (a mixed group
        never upgrades to exact).

    Returns
    -------
    tuple[list[dict[str, Any]], list[str | None]]
        The grouped rows and their FLOPs evidence labels.
    """

    groups: dict[str, list[int]] = {}
    for index, (row, op) in enumerate(zip(op_rows, ops, strict=True)):
        groups.setdefault(source_line_key(row, op), []).append(index)
    grouped: list[dict[str, Any]] = []
    evidence: list[str | None] = []
    for name, members in groups.items():
        grouped.append(fold_rows(name, [op_rows[index] for index in members]))
        labels = [label for index in members if (label := row_evidence[index]) is not None]
        evidence.append(weakest_evidence(labels) if labels else None)
    return grouped, evidence
