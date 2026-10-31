"""The recurrence fold: the agent surface's default structural view.

Contract (agent memo 3.4, normative 3-0): a repeated transformer block is
stated ONCE with ``n_instances``, losslessly -- any class whose members
disagree on shape/dtype/num_params is SPLIT into field-uniform subclasses,
other varying fields carry ``varies`` disclosures, and membership reassembles
exactly to the op-row set. The fold key is explicit and versioned; ZERO
render-context inputs participate (criterion vi); ``pass_index`` joins the key
on multi-pass ops so uniformity is never claimed across a pass boundary.

Seam decision (memo D1, rule fixed at panel time): the renderer's collapse
schedule is measured-DEGENERATE at depth (12/24-block models reach only "the
full graph" or "one box"), so the run-aware fold ships agent-side under the
six acceptance criteria, with the FOLD COHERENCE TEST guaranteeing membership
agreement with the collapse plan (tests/test_agent_surface_fold.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Versioned, documented fold key (agent memo 3.4). Recorded in every folded
#: payload so a consumer can branch on the key generation.
FOLD_KEY_VERSION = "torchlens.agent_fold_key.v1"

#: Fields whose disagreement SPLITS a class into field-uniform subclasses.
_SPLIT_FIELDS = ("shape", "dtype", "num_params")

#: Fields disclosed per class with set disclosure when they vary.
_DISCLOSED_FIELDS = ("payload_state", "device", "saved")

#: Ceiling on distinct values listed in one ``varies`` disclosure.
_VARIES_LIST_MAX = 8


def normalize_module_path(stack_entry: str) -> str:
    """Normalize one module-call-stack entry to its recurrence pattern.

    Strips the ``:call`` qualifier and wildcards purely-numeric dotted
    components, so ``"transformer.h.11.mlp:1"`` and ``"transformer.h.3.mlp:1"``
    both normalize to ``"transformer.h.*.mlp"``.

    Parameters
    ----------
    stack_entry:
        One ``module_call_stack`` entry (``address:call_index``).

    Returns
    -------
    str
        Normalized recurrence pattern.
    """

    address = stack_entry.rsplit(":", 1)[0] if ":" in stack_entry else stack_entry
    parts = [("*" if part.isdigit() else part) for part in address.split(".")]
    return ".".join(parts)


def _op_rank(op: Any) -> int | None:
    """Return the op's output tensor rank, or ``None`` when unrecorded."""

    shape = getattr(op, "shape", None)
    if shape is None:
        return None
    try:
        return len(shape)
    except TypeError:
        return None


def _fold_key(op: Any) -> tuple[Any, ...]:
    """Compute the versioned fold key for one op.

    Key = (normalized module_call_stack, func_name, rank, boundary role
    [, pass_index]) -- the boundary role keeps a model's input and output
    nodes in DISTINCT classes even when their shapes match (AUD-CODE
    3.11b; both spell ``func_name="none"`` with an empty stack), and
    ``pass_index`` participates ONLY on multi-pass ops (never claim
    uniformity across a pass boundary).

    Parameters
    ----------
    op:
        Operation record.

    Returns
    -------
    tuple
        Hashable fold key.
    """

    stack = tuple(
        normalize_module_path(str(entry)) for entry in getattr(op, "module_call_stack", ()) or ()
    )
    func_name = str(getattr(op, "func_name", "unknown"))
    key: tuple[Any, ...] = (stack, func_name, _op_rank(op), _boundary_role(op))
    if int(getattr(op, "num_passes", 1) or 1) > 1:
        key = (*key, int(getattr(op, "pass_index", 1) or 1))
    return key


def _boundary_role(op: Any) -> str | None:
    """Return the op's graph-boundary role: input / output / input_output / None."""

    is_input = bool(getattr(op, "is_input", False))
    is_output = bool(getattr(op, "is_output", False))
    if is_input and is_output:
        return "input_output"
    if is_input:
        return "input"
    if is_output:
        return "output"
    return None


def _edge_index(ops: list[Any], label_to_class: dict[str, str]) -> dict[str, str]:
    """Map every edge spelling an op record can carry to its class id.

    ``op.parents``/``op.children`` spell single-pass neighbours by BARE layer
    label (``"linear_1_1"``) and multi-pass neighbours pass-qualified
    (``"linear_1_1:2"``), while ``label_to_class`` is keyed by the
    pass-qualified ``op.label``; without this bridge every single-pass
    trace folded with empty parent/child classes (AUD-CODE 3.11b). A bare
    reference to a MULTI-pass layer stays unresolved: it names several
    classes and claiming any one of them would over-state the relation.

    Parameters
    ----------
    ops:
        Op records.
    label_to_class:
        ``op.label`` -> class id.

    Returns
    -------
    dict[str, str]
        Edge spelling -> class id.
    """

    index = dict(label_to_class)
    for op in ops:
        if int(getattr(op, "num_passes", 1) or 1) == 1:
            bare = str(getattr(op, "layer_label", ""))
            class_id = label_to_class.get(str(getattr(op, "label", "")))
            if bare and class_id is not None:
                index.setdefault(bare, class_id)
    return index


def _split_value(op: Any, field: str) -> Any:
    """Return one split-key field value in hashable, JSON-stable form."""

    if field == "shape":
        shape = getattr(op, "shape", None)
        return tuple(int(dim) for dim in shape) if shape is not None else None
    if field == "dtype":
        dtype = getattr(op, "dtype", None)
        return str(dtype) if dtype is not None else None
    return int(getattr(op, "num_params", 0) or 0)


def _disclosed_value(op: Any, field: str) -> Any:
    """Return one disclosure-field value for a class row."""

    if field == "payload_state":
        from ..report._agent_json import _payload_state

        return _payload_state(op)
    if field == "device":
        device_ref = getattr(op, "device_ref", None)
        return str(getattr(device_ref, "name", device_ref)) if device_ref else None
    return bool(getattr(op, "has_saved_activation", False))


def _disclose(values: list[Any]) -> Any:
    """Fold one field's member values into a uniform value or ``varies`` record."""

    distinct = sorted({repr(value) for value in values})
    if len(distinct) == 1:
        return values[0]
    disclosure: dict[str, Any] = {"varies": True, "n_distinct": len(distinct)}
    if len(distinct) <= _VARIES_LIST_MAX:
        seen: dict[str, Any] = {}
        for value in values:
            seen.setdefault(repr(value), value)
        disclosure["values"] = [seen[key] for key in distinct]
    return disclosure


@dataclass(frozen=True)
class FoldClass:
    """One recurrence class: a set of structurally interchangeable ops."""

    class_id: str
    module_path: tuple[str, ...]
    func_name: str
    rank: int | None
    boundary: str | None
    pass_index: int | None
    n_instances: int
    members: tuple[str, ...]
    representative: str
    fields: dict[str, Any]
    parent_classes: tuple[str, ...]
    child_classes: tuple[str, ...]


@dataclass(frozen=True)
class FoldResult:
    """The complete fold of one trace: classes plus exact membership."""

    key_version: str
    classes: tuple[FoldClass, ...]
    n_ops: int

    def membership(self) -> dict[str, str]:
        """Return the exact op-label -> class_id map (reassembly proof)."""

        return {label: cls.class_id for cls in self.classes for label in cls.members}


def fold_trace(log: Any) -> FoldResult:
    """Fold one finished trace into recurrence classes.

    NEVER declines on a valid Trace (criterion iv): every op lands in exactly
    one class; a trace with no recurrence folds to one class per op.

    Parameters
    ----------
    log:
        Completed ``Trace`` (live or loaded; structure-only traces fold on
        their hypothesis shapes, which the overview banner already discloses).

    Returns
    -------
    FoldResult
        Deterministic fold under :data:`FOLD_KEY_VERSION`.
    """

    ops = list(getattr(log, "layer_list", []) or [])
    groups: dict[tuple[Any, ...], list[Any]] = {}
    order: list[tuple[Any, ...]] = []
    for op in ops:
        key = _fold_key(op)
        split = tuple(_split_value(op, field) for field in _SPLIT_FIELDS)
        full_key = (key, split)
        if full_key not in groups:
            groups[full_key] = []
            order.append(full_key)
        groups[full_key].append(op)

    label_to_class: dict[str, str] = {}
    class_ids: dict[tuple[Any, ...], str] = {}
    for index, full_key in enumerate(order):
        class_id = f"c{index:04d}"
        class_ids[full_key] = class_id
        for op in groups[full_key]:
            label_to_class[str(getattr(op, "label", ""))] = class_id

    edge_index = _edge_index(ops, label_to_class)
    classes: list[FoldClass] = []
    for full_key in order:
        members = groups[full_key]
        (stack, func_name, rank, boundary, *pass_part), split = full_key
        fields: dict[str, Any] = dict(zip(_SPLIT_FIELDS, split, strict=True))
        if fields["shape"] is not None:
            fields["shape"] = list(fields["shape"])
        for field in _DISCLOSED_FIELDS:
            fields[field] = _disclose([_disclosed_value(op, field) for op in members])
        parent_ids = sorted(
            {
                edge_index[str(parent)]
                for op in members
                for parent in getattr(op, "parents", ()) or ()
                if str(parent) in edge_index
            }
        )
        child_ids = sorted(
            {
                edge_index[str(child)]
                for op in members
                for child in getattr(op, "children", ()) or ()
                if str(child) in edge_index
            }
        )
        classes.append(
            FoldClass(
                class_id=class_ids[full_key],
                module_path=stack,
                func_name=func_name,
                rank=rank,
                boundary=boundary,
                pass_index=pass_part[0] if pass_part else None,
                n_instances=len(members),
                members=tuple(str(getattr(op, "label", "")) for op in members),
                representative=str(getattr(members[0], "label", "")),
                fields=fields,
                parent_classes=tuple(parent_ids),
                child_classes=tuple(child_ids),
            )
        )
    return FoldResult(key_version=FOLD_KEY_VERSION, classes=tuple(classes), n_ops=len(ops))


def fold_class_rows(result: FoldResult) -> list[dict[str, Any]]:
    """Serialize fold classes as JSON-safe overview rows.

    Member lists are NOT inlined (the folded overview stays flat in depth);
    exact expansion is served by ``dump(view="graph", class_id=...)``.

    Parameters
    ----------
    result:
        Fold result from :func:`fold_trace`.

    Returns
    -------
    list[dict[str, Any]]
        One row per class, execution-ordered by first member.
    """

    rows: list[dict[str, Any]] = []
    for cls in result.classes:
        row: dict[str, Any] = {
            "class_id": cls.class_id,
            "module_path": list(cls.module_path),
            "func": cls.func_name,
            "rank": cls.rank,
            "n_instances": cls.n_instances,
            "representative": cls.representative,
            "parent_classes": list(cls.parent_classes),
            "child_classes": list(cls.child_classes),
            **cls.fields,
        }
        if cls.boundary is not None:
            row["boundary"] = cls.boundary
        if cls.pass_index is not None:
            row["pass_index"] = cls.pass_index
        rows.append(row)
    return rows
