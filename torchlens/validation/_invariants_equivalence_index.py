"""Index builders for the equivalence and loop-detection invariants."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def _expected_equivalence_group(
    ml: Trace,
    op: Any,
    group_by_member: dict[str, set],
    key_by_member: dict[str, object],
) -> Any:
    """Return the trace-level equivalence group an op's ``equivalent_ops`` must equal.

    Parameters
    ----------
    ml:
        Trace whose equivalence classes are checked.
    op:
        Op record being checked.
    group_by_member:
        ``op label -> equivalence group`` from :func:`_equivalence_members`.
    key_by_member:
        ``op label -> equivalence key prefix`` from :func:`_equivalence_members`.

    Returns
    -------
    Any
        The expected group, or ``None`` when the trace records none for the op.
    """

    equivalence_class = record_member(op, "equivalence_class", None)
    expected_group = None
    if isinstance(equivalence_class, str):
        member_key = key_by_member.get(op.label)
        if isinstance(member_key, str) and equivalence_class.startswith(member_key):
            expected_group = group_by_member[op.label]
        else:
            expected_group = ml._graph_op_equivalence_classes.get(equivalence_class)
    if expected_group is None:
        expected_group = group_by_member.get(op.label)
    return expected_group


def _param_barcode_groups(
    ml: Trace, call_site_grouped: bool
) -> dict[tuple[str, tuple[str, ...], str], list[Op]]:
    """Group parameterized ops by ``(func_name, param barcodes, equivalence_class)``.

    Parameters
    ----------
    ml:
        Trace whose loop detection is checked.
    call_site_grouped:
        Whether the trace uses call-site grouping (then no groups are built).

    Returns
    -------
    dict[tuple[str, tuple[str, ...], str], list[Op]]
        Ops per parameter-sharing key.
    """

    param_groups: dict[tuple[str, tuple[str, ...], str], list[Op]] = defaultdict(list)
    for lpl in [] if call_site_grouped else ml._layer_list:
        if lpl.has_params and lpl._param_barcodes:
            key = (lpl.func_name, tuple(sorted(lpl._param_barcodes)), lpl.graph.equivalence_class)
            param_groups[key].append(lpl)
    return param_groups


def _equivalence_key_by_op_label(ml: Trace) -> dict[str, str]:
    """Map each op label to its ``op_equivalence_classes`` key.

    Parameters
    ----------
    ml:
        Trace whose equivalence classes are read.

    Returns
    -------
    dict[str, str]
        ``op label -> equivalence key``.
    """

    op_label_to_equiv_key: dict[str, str] = {}
    for eq_type, equiv_set in ml._graph_op_equivalence_classes.items():
        for label in equiv_set:
            op_label_to_equiv_key[label] = eq_type
    return op_label_to_equiv_key
