"""The output-node copy: build a synthetic output node from its parent op's row.

``_add_output_layers`` clones the op that produced each returned tensor and then
overrides the bookkeeping fields of the clone. The clone reads every field it
does not override straight from the parent's row: each field's data descriptor
(the row-cell fields, relation views, properties such as ``arg_expressions``) is
called directly, and only ``out`` and ``grad`` (the reads ``Op.__getattribute__``
intercepts) go through ``getattr``.
Each read returns exactly what ``getattr(op, op_attr(field), None)`` returns, so
the values are the ones the former whole-schema ``getattr`` loop copied, without
the per-field ``__getattribute__`` frame.

Values keep the former copy policy: fields in the shared set are shared,
immutable values are shared, a flat ``list``, ``set`` or ``dict`` of immutable
members gets a fresh container of the same members (equal to ``deepcopy`` of
it), and anything else goes through ``_op_copy._copy_op_field_value``. The
clone therefore holds no container its parent holds outside the shared set,
so a later edit to the parent or to the model cannot reach it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._constants import LAYER_PASS_LOG_FIELD_ORDER
from .._quantities import Bytes, Duration, Flops
from .._trace_core.alias_registry import register_shared_cells
from .._trace_core.cell_states import LazyCell
from .._trace_core.groups import GroupRef
from .._trace_core.op_store import _CSR, _FACT, _MISSING, OpRowStore, PooledCell, row_clone_scope
from .._utils._record_group_table import op_attr
from . import _op_copy
from ._op_row_release import _OP_FACADE

if TYPE_CHECKING:
    from .op import Op

#: Exact classes shared as they are (the ``_op_copy`` immutable set). A subclass
#: of any of them takes the ``_copy_op_field_value`` path, as before. A frozen
#: dataclass whose every field holds one of these exact classes (the dtype and
#: device references) is immutable too and is shared as well.
_SHARED_CLASSES = frozenset(
    {
        str,
        bytes,
        int,
        float,
        bool,
        type(None),
        torch.dtype,
        torch.device,
        Bytes,
        Duration,
        Flops,
    }
)
_FLAT_CONTAINER_CLASSES = frozenset({list, set})

#: Fields every selective-depth copy shares with its source (the
#: ``fields_not_to_deepcopy`` base set of ``_op_copy._copy_with_shallow_fields``).
_ALWAYS_SHARED_FIELDS = frozenset(
    {
        "func",
        "grad_fn_class_name",
        "grad_fn_handle",
        "source_trace",
        "func_rng_states",
        "saved_args",
        "saved_kwargs",
        "args_template",
        "kwargs_template",
        "parent_params",
        "out",
        "transformed_out",
        "transformed_grad",
        "out_versions_by_child",
        "container_spec",
    }
)

#: The reads ``Op.__getattribute__`` intercepts (``op._INTERCEPTED_READ_FIELDS``,
#: pinned equal by the tests); they keep going through ``getattr``.
_INTERCEPTED_READ_FIELDS = frozenset({"grad", "out"})

#: Raw cell classes a generated field descriptor may transform on read (lazy answer,
#: pooled hydrate, group view). A cell of one of these exact classes, or one of the
#: two sentinels below (CSR dataflow, shared fact), is read through the descriptor;
#: every other set cell is what each generated descriptor returns unchanged.
_TRANSFORMED_CELLS = frozenset({LazyCell, PooledCell, GroupRef})
_TRANSFORMED_SENTINELS = (_CSR, _FACT)

#: ``(op class, shared fields, unread fields)`` -> per-field read plan.
_PLANS: dict[tuple[type, frozenset[str], frozenset[str]], tuple[tuple[Any, ...], ...]] = {}

_GET = object.__getattribute__


def _class_attribute(cls: type, name: str) -> Any:
    """Return ``name`` as the class defines it (first MRO hit), or ``None``."""

    for klass in cls.__mro__:
        found = klass.__dict__.get(name, _MISSING)
        if found is not _MISSING:
            return found
    return None


def _field_plan(
    cls: type, shared: frozenset[str], unread: frozenset[str]
) -> tuple[tuple[Any, ...], ...]:
    """Return ``(field, data descriptor or None, attribute, shared, fid)`` per copied field.

    A field whose attribute is a data descriptor on ``cls`` (and not an
    intercepted read) is read by calling the descriptor, which is what
    ``getattr`` does for it after ``Op.__getattribute__``; every other field
    is read with ``getattr``. ``fid`` is the column id of a generated row-cell
    descriptor (``_OP_FACADE``'s field classes: plain, relation, group, dataflow
    and fact fields), whose read is the raw cell unless that cell is unset, of a
    ``_TRANSFORMED_CELLS`` class or a ``_TRANSFORMED_SENTINELS`` sentinel; else ``None``.
    """

    key = (cls, shared, unread)
    plan = _PLANS.get(key)
    if plan is not None:
        return plan
    rows = []
    for field in LAYER_PASS_LOG_FIELD_ORDER:
        if field in unread:
            continue
        attr = op_attr(field)
        descriptor = _class_attribute(cls, attr)
        descriptor_type = type(descriptor)
        is_data_descriptor = hasattr(descriptor_type, "__get__") and (
            hasattr(descriptor_type, "__set__") or hasattr(descriptor_type, "__delete__")
        )
        if not is_data_descriptor or field in _INTERCEPTED_READ_FIELDS:
            descriptor = None
        plain = descriptor is not None and bool(_OP_FACADE)
        plain = plain and descriptor_type in _OP_FACADE[0][3]
        fid = descriptor._fid if descriptor is not None and plain else None
        rows.append((field, descriptor, attr, field in shared, fid))
    plan = tuple(rows)
    _PLANS[key] = plan
    return plan


def _is_frozen_shared_record(value: Any, value_class: type) -> bool:
    """Return whether ``value`` is a frozen dataclass of shared-class members only."""

    params = getattr(value_class, "__dataclass_params__", None)
    if params is None or not params.frozen:
        return False
    return all(
        getattr(value, name, _MISSING).__class__ in _SHARED_CLASSES
        for name in value_class.__dataclass_fields__  # type: ignore[attr-defined]
    )


def _independent_value(value: Any) -> Any:
    """Return ``value`` or a copy of it that shares no container with it."""

    value_class = value.__class__
    shared_classes = _SHARED_CLASSES
    if value_class in shared_classes:
        return value
    if value_class in _FLAT_CONTAINER_CLASSES and all(
        member.__class__ in shared_classes for member in value
    ):
        return value_class(value)
    if value_class is dict and all(
        key.__class__ in shared_classes and item.__class__ in shared_classes
        for key, item in value.items()
    ):
        return dict(value)
    if _is_frozen_shared_record(value, value_class):
        return value
    return _op_copy._copy_op_field_value(value)


def _plain_cells(op: Op) -> list[Any] | None:
    """Return ``op``'s raw row cells when its store is a plain row-major ``OpRowStore``.

    Only there is a plain descriptor's read exactly the raw cell (audited and
    sealed stores are subclasses and keep the descriptor read).
    """

    # An unbound slot or out-of-range row falls back to the descriptor reads, which
    # handle it field by field exactly as before.
    try:
        store = _GET(op, "_core")
        if store.__class__ is not OpRowStore or store._rows is None:
            return None
        return store._rows[_GET(op, "_row")]
    except (AttributeError, IndexError):
        return None


def _descriptor_read(descriptor: Any, op: Op, cls: type) -> Any:
    """Return ``descriptor``'s read of ``op``, or ``None`` when it raises ``AttributeError``."""

    try:
        return descriptor.__get__(op, cls)
    except AttributeError:
        return None


def _read_parent_fields(op: Op, plan: tuple[tuple[Any, ...], ...]) -> dict[str, Any]:
    """Read every planned field from ``op``'s row, copied by the copy policy."""

    cls = type(op)
    cells = _plain_cells(op)
    shared_classes = _SHARED_CLASSES
    transformed = _TRANSFORMED_CELLS
    csr, fact = _TRANSFORMED_SENTINELS
    fields_dict: dict[str, Any] = {}
    for field, descriptor, attr, shared, fid in plan:
        if cells is not None and fid is not None:
            value = cells[fid]
            if value is _MISSING:
                value = None
            elif value.__class__ in transformed or value is csr or value is fact:
                value = _descriptor_read(descriptor, op, cls)
        elif descriptor is None:
            value = getattr(op, attr, None)
        else:
            value = _descriptor_read(descriptor, op, cls)
        if shared or value.__class__ in shared_classes:
            fields_dict[field] = value
        else:
            fields_dict[field] = _independent_value(value)
    return fields_dict


def _copy_for_output(op: Op, *, _store: Any = None) -> Op:
    """Build the synthetic output node for ``op`` from ``op``'s row.

    Parameters
    ----------
    op:
        The op that produced a returned tensor.
    _store:
        Optional owning row store for the synthesized node.

    Returns
    -------
    Op
        A new op whose fields equal the former whole-schema copy's; the caller
        overrides its bookkeeping fields.
    """

    shared = _ALWAYS_SHARED_FIELDS | _op_copy._OUTPUT_NODE_REPLACED_FIELDS
    plan = _field_plan(type(op), shared, _op_copy._OUTPUT_COPY_UNREAD_DERIVED_FIELDS)
    # A row-clone read under a combined step audit (see ``_copy_with_shallow_fields``).
    with row_clone_scope(_GET(op, "_core")):
        fields_dict = _read_parent_fields(op, plan)
    return register_shared_cells(op, type(op)(fields_dict, _store=_store))
