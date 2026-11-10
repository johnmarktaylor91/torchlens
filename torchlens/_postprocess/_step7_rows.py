"""Step 7 over op rows: site keys, call-site grouping, pass numbers and the contract.

Recurrence grouping reads and writes ``Op`` fields through one accessor family
(:class:`Step7Rows`). When every raw op shares one row store and the raw label
list names exactly the ops of the raw layer dict (a torch capture after
materialization), a handle is the op's row and the accessors are the column
view's bound per-field readers and writers (:class:`OpColumns`), one call per
cell. Otherwise a handle is the ``Op`` facade ``trace[label]`` returns and the
accessors read and write its attributes. Each algorithm here has one body that
runs over either family and visits ops in the order the facade passes did, so
the stored result is the same.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from operator import itemgetter
from typing import TYPE_CHECKING, Any, NamedTuple

from .._data_classes._site_key import (
    SiteKeyMinter,
    call_instance_id,
    site_axis,
    without_root_frames,
)
from .._record_type import record_type
from .._utils._record_group_table import op_attr, record_member
from ..errors._validation import MetadataInvariantError
from ._call_site_contract import (
    CallSiteEntry,
    call_site_contract_violation,
    remember_call_site_classes,
)
from ._column_passes import _column_lookup

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

#: Marker for "no default": an unset attribute raises.
_NO_DEFAULT: Any = object()

#: Sort and ``min`` key of ``(raw_index, label, handle)`` member tuples.
_RAW_INDEX = itemgetter(0)


class Step7Rows(NamedTuple):
    """The ops of one capture as handles of one accessor family.

    Attributes
    ----------
    getter:
        ``getter(name[, default])`` returning ``get(handle)`` for stored field ``name``.
    setter:
        ``setter(name)`` returning ``put(handle, value)`` for stored field ``name``.
    handles:
        Every raw label of the raw layer dict to its handle, in dict order.
    retained:
        ``(label, handle)`` of every non-orphan op, in raw-label-list order.
    """

    getter: Callable[..., Callable[[Any], Any]]
    setter: Callable[[str], Callable[[Any, Any], None]]
    handles: dict[str, Any]
    retained: list[tuple[str, Any]]


class KeyReaders(NamedTuple):
    """Readers the call-site grouping key needs for an op without a call-site class."""

    is_input: Callable[[Any], Any]
    is_output: Callable[[Any], Any]
    equivalence_class: Callable[[Any], Any]


def _facade_getter(name: str, default: Any = _NO_DEFAULT) -> Callable[[Any], Any]:
    """Return a reader of field ``name`` over ``Op`` facade handles."""

    if name == "type":
        return record_type
    if default is _NO_DEFAULT:
        attr = op_attr(name)
        return lambda node: getattr(node, attr)
    return lambda node: record_member(node, name, default)


def _facade_setter(name: str) -> Callable[[Any, Any], None]:
    """Return a writer of field ``name`` over ``Op`` facade handles."""

    attr = op_attr(name)
    return lambda node, value: setattr(node, attr, value)


def step7_rows(trace: Trace) -> Step7Rows:
    """Return the capture's ops as column rows, or as facade handles when columns do not apply.

    Parameters
    ----------
    trace:
        Trace currently running Step 7 postprocessing.

    Returns
    -------
    Step7Rows
        Accessors, every handle of the raw layer dict, and the retained ops.
    """

    workspace = trace._raw_graph_ws
    labels = workspace.raw_layer_labels_list
    lookup = _column_lookup(trace)
    if lookup is not None and len(lookup[1]) == len(labels) and lookup[1].keys() == set(labels):
        columns, rows = lookup
        is_orphan = columns.getter("is_orphan", False)
        retained = [(label, rows[label]) for label in labels if not is_orphan(rows[label])]
        return Step7Rows(columns.getter, columns.setter, rows, retained)
    handles = {label: trace[label] for label in workspace.raw_layer_dict}
    retained = []
    for label in labels:
        node = trace[label]
        if not record_member(node, "is_orphan", False):
            retained.append((label, node))
    return Step7Rows(_facade_getter, _facade_setter, handles, retained)


def _normalized_site(entries: Any) -> tuple[tuple[str, ...], str]:
    """Return ``(site axis, call instance)`` of one module stack, root frames dropped."""

    kept = without_root_frames(entries)
    return site_axis(kept), call_instance_id(kept)


def mint_site_keys(rows: Step7Rows) -> list[str]:
    """Mint ``site_key_v1`` on every retained op, in raw-label order.

    Equal to ``SiteKeyMinter.mint`` per op; the module-stack normalization is
    computed once per distinct stack. Orphans consume no ordinals.

    Parameters
    ----------
    rows:
        The capture's ops.

    Returns
    -------
    list[str]
        The minted keys, aligned with ``rows.retained``.
    """

    read_modules = rows.getter("modules", None)
    read_type = rows.getter("type")
    read_slot = rows.getter("multi_output_index", None)
    write_key = rows.setter("site_key")
    mint_at = SiteKeyMinter().mint_at
    sites: dict[Any, tuple[tuple[str, ...], str]] = {}
    keys: list[str] = []
    for _label, handle in rows.retained:
        entries = read_modules(handle) or ()
        try:
            site = sites[entries]
        except KeyError:
            site = sites[entries] = _normalized_site(entries)
        except TypeError:  # an unhashable stack is normalized every time
            site = _normalized_site(entries)
        key = mint_at(site[0], site[1], read_type(handle), read_slot(handle))
        write_key(handle, key)
        keys.append(key)
    return keys


def has_call_site_evidence(trace: Trace, rows: Step7Rows) -> bool:
    """Whether any retained op of this capture carries a call-site class."""

    classes = trace._raw_graph_ws._call_site_classes
    if not classes:
        return False
    return any(label in classes for label, _handle in rows.retained)


def call_site_group_key(
    label: str,
    handle: Any,
    site_key: str,
    func_name: Any,
    read: KeyReaders,
    classes: dict[str, str],
) -> tuple[Any, ...]:
    """The grouping key of one retained non-buffer op under call-site grouping.

    An op with a call-site class groups with every op of the same class (same
    innermost module instance, same call-site chain, same function and output
    slot). Inputs and outputs stay single. Anything else (no Python frame inside
    its module call, synthesized nodes) falls back to its ``site_key_v1``
    position within the module call, which never fuses two positions of one
    call and still groups a module called several times; its capture-time
    equivalence class (type, non-tensor arguments, module) joins the fallback
    key so a fallback layer never spans two equivalence sets. Buffer versions
    are keyed by :func:`buffer_group_keys`.
    """

    site_class = classes.get(label)
    if site_class is not None:
        return ("call_site", site_class)
    if read.is_input(handle) or read.is_output(handle):
        return ("single", label)
    return ("position", site_key, func_name, read.equivalence_class(handle))


def buffer_group_keys(
    buffers: list[Any], op_keys: dict[str, tuple[Any, ...]]
) -> list[tuple[Any, tuple[Any, ...]]]:
    """Group the versions of each registered buffer by the place that wrote them.

    A version written by an op takes that op's key, so the versions one write
    line produces across a loop are passes of one buffer layer, and versions
    written at different places are different layers. The entry version (no
    writer) joins the layer of the version written next when that layer
    repeats (the loop-carried state of a loop); otherwise it stays on its own,
    as does every version of a buffer written once.
    """

    keyed: dict[str, tuple[Any, ...]] = {}
    entries: list[Any] = []
    written_by_address: dict[Any, list[Any]] = defaultdict(list)
    for node in buffers:
        writer = next((parent for parent in node.parents if parent in op_keys), None)
        if writer is None:
            entries.append(node)
            continue
        keyed[node._label_raw] = ("buffer", node.buffer_address, op_keys[writer])
        written_by_address[node.buffer_address].append(node)
    sizes: dict[tuple[Any, ...], int] = defaultdict(int)
    for key in keyed.values():
        sizes[key] += 1
    for node in entries:
        later = [
            version
            for version in written_by_address.get(node.buffer_address, ())
            if version.raw_index > node.raw_index
        ]
        first = min(later, key=lambda version: version.raw_index, default=None)
        if first is not None and sizes[keyed[first._label_raw]] > 1:
            keyed[node._label_raw] = keyed[first._label_raw]
        else:
            keyed[node._label_raw] = ("buffer", node.buffer_address, "entry", node._label_raw)
    return [(node, keyed[node._label_raw]) for node in buffers]


def _call_site_partition(
    trace: Trace, rows: Step7Rows, site_keys: list[str]
) -> list[list[tuple[Any, str, Any]]]:
    """Partition the retained ops by call-site key; each member is ``(raw_index, label, handle)``."""

    classes = trace._raw_graph_ws._call_site_classes
    read_buffer = rows.getter("is_buffer")
    read_func = rows.getter("func_name")
    read_index = rows.getter("raw_index")
    read = KeyReaders(
        rows.getter("is_input"), rows.getter("is_output"), rows.getter("equivalence_class")
    )
    groups: dict[tuple[Any, ...], list[tuple[Any, str, Any]]] = defaultdict(list)
    keys: dict[str, tuple[Any, ...]] = {}
    buffers: list[tuple[str, Any]] = []
    for (label, handle), site_key in zip(rows.retained, site_keys):
        # ``buffer_address`` is a facade member, not a column; buffers are few.
        if read_buffer(handle) and trace[label].buffer_address:
            buffers.append((label, handle))
            continue
        key = call_site_group_key(label, handle, site_key, read_func(handle), read, classes)
        keys[label] = key
        groups[key].append((read_index(handle), label, handle))
    if buffers:
        nodes = [trace[label] for label, _handle in buffers]
        for (label, handle), (node, key) in zip(buffers, buffer_group_keys(nodes, keys)):
            groups[key].append((node.raw_index, label, handle))
    return list(groups.values())


def rebuild_passes(rows: Step7Rows, layer_of: dict[str, tuple[Any, str]]) -> None:
    """Rebuild recurrence membership and pass numbers from the layer labels.

    Parameters
    ----------
    rows:
        The capture's ops.
    layer_of:
        ``label -> (raw_index, layer label)`` already known for retained ops;
        any other non-orphan op of the raw layer dict is read from the store.
    """

    is_orphan = rows.getter("is_orphan", False)
    read_index = rows.getter("raw_index")
    read_layer = rows.getter("_layer_label_raw")
    groups: dict[str, list[tuple[Any, str, Any]]] = defaultdict(list)
    for label, handle in rows.handles.items():
        if label in layer_of:
            raw_index, layer = layer_of[label]
        elif is_orphan(handle):
            continue
        else:
            raw_index, layer = read_index(handle), read_layer(handle)
        groups[layer].append((raw_index, label, handle))

    write_group = rows.setter("recurrent_ops")
    write_pass = rows.setter("pass_number")
    write_count = rows.setter("num_passes")
    for members in groups.values():
        members.sort(key=_RAW_INDEX)
        member_labels = [label for _raw_index, label, _handle in members]
        count = len(member_labels)
        for pass_index, (_raw_index, _label, handle) in enumerate(members, start=1):
            write_group(handle, member_labels)
            write_pass(handle, pass_index)
            write_count(handle, count)


def group_call_sites(trace: Trace, rows: Step7Rows, site_keys: list[str]) -> None:
    """Partition retained ops into layers by call-site key (``callsite_v2``).

    Two ops are passes of one layer exactly when their keys are equal
    (:func:`call_site_group_key`). Passes are numbered in capture order by
    :func:`rebuild_passes`. Each op keeps its own ``equivalence_class`` (the
    argument/parameter fingerprint); the call-site class lives only in the
    workspace side table. The finished grouping is checked against the
    call-site contract (``_call_site_contract``) before step 7 returns.

    Parameters
    ----------
    trace:
        Trace currently running Step 7 postprocessing.
    rows:
        The capture's ops.
    site_keys:
        Minted site keys aligned with ``rows.retained``.
    """

    read_layer = rows.getter("_layer_label_raw")
    write_layer = rows.setter("_layer_label_raw")
    layer_of: dict[str, tuple[Any, str]] = {}
    for members in _call_site_partition(trace, rows, site_keys):
        leader_layer = read_layer(min(members, key=_RAW_INDEX)[2])
        for raw_index, label, handle in members:
            write_layer(handle, leader_layer)
            layer_of[label] = (raw_index, leader_layer)

    rebuild_passes(rows, layer_of)
    classes = trace._raw_graph_ws._call_site_classes
    check_call_site_contract(rows, classes)
    remember_call_site_classes(trace, classes)


def check_call_site_contract(rows: Step7Rows, classes: dict[str, str]) -> None:
    """Raise when the stored grouping breaches the call-site contract.

    Reads each op's stored layer and recurrent group, not the grouping key,
    so a defect in the key, the partition or the pass rebuild that merges two
    call sites, splits one, or leaves a recurrent group spanning two sites
    fails here.
    """

    read_layer = rows.getter("_layer_label_raw")
    read_func = rows.getter("func_name")
    read_group = rows.getter("recurrent_ops")
    entries = [
        CallSiteEntry(
            raw_label=label,
            layer=read_layer(handle),
            func_name=read_func(handle),
            group=tuple(read_group(handle) or ()),
        )
        for label, handle in rows.retained
    ]
    violation = call_site_contract_violation(entries, classes)
    if violation is not None:
        raise MetadataInvariantError("call_site_contract", violation)
