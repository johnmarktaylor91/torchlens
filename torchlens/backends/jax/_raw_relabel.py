"""Raw-label helpers for finishing a JAX trace: lineage relabel, equivalence members, sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from ..._utils._record_group_table import record_member, record_storage_attr

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


def _relabel_jax_lineage_fields(op_log: Any, raw_to_final: Mapping[str, str]) -> None:
    """Replace raw labels with final labels in ``op_log``'s lineage sets, in place.

    Parameters
    ----------
    op_log
        Materialized raw JAX op whose lineage sets still hold raw labels.
    raw_to_final
        Mapping from raw op labels to final labels.
    """

    for lineage_field in (
        "input_ancestors",
        "output_descendants",
        "root_ancestors",
        "internal_source_ancestors",
        "internal_source_parents",
    ):
        lineage = record_member(op_log, lineage_field, None)
        if lineage:
            relabeled = (
                raw_to_final.get(item, item) if isinstance(item, str) else item for item in lineage
            )
            setattr(op_log, record_storage_attr(op_log, lineage_field), type(lineage)(relabeled))


def _jax_equivalent_labels_by_key(trace: Trace, raw_labels: Iterable[str]) -> dict[str, set[str]]:
    """Return the op labels of each equivalence class among ``raw_labels``.

    Parameters
    ----------
    trace
        Trace containing materialized raw JAX ops.
    raw_labels
        Raw op labels to group.

    Returns
    -------
    dict[str, set[str]]
        Equivalence class key to the labels of its member ops.
    """

    equivalent_labels_by_key: dict[str, set[str]] = {}
    for label in raw_labels:
        op_log = trace._raw_graph_ws.raw_layer_dict[label]
        equivalent_labels_by_key.setdefault(op_log.graph.equivalence_class, set()).add(op_log.label)
    return equivalent_labels_by_key


def _is_jax_recurrence_source(op_log: Any, data_parents: tuple[str, ...]) -> bool:
    """Return whether a raw JAX op starts a recurrence walk (input, internal source or no data parent).

    Parameters
    ----------
    op_log
        Materialized raw JAX op.
    data_parents
        The op's ordered data-parent labels.

    Returns
    -------
    bool
        True when the op is a source of the recurrence grouping graph.
    """

    return bool(op_log.is_input or op_log.graph.is_internal_source or not data_parents)
