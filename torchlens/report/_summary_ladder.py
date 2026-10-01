"""The summary auto ladder (F08; summary memo item 11, decisions 3.1/3.4).

One deterministic view resolver: given a finished trace and a body-row
budget, resolve the deepest informative rung of the view ladder --

1. COALESCED HYBRID, full depth: every executed compute op appears exactly
   once; a leaf module owning exactly one op coalesces into a single
   ``fc1 (Linear)`` row; orphan functional ops (the classic torchsummary
   blind spot) get their own rows; multi-op leaf modules expand beneath.
2. MODULE TREE, full depth, strictly folded.
3. MODULE TREE, descending depth (d_max-1 .. 1), strictly folded.
4. ELISION on the appropriate rung: protected, totals-conserving
   accounting rows replace the lowest-share contiguous runs -- never
   truncation, so no model at any budget collapses to opaque containers.

Identity-partition law (memo 3.1): every executed compute-op event and
every parameter identity is owned by exactly one accounting row at every
depth, fold, and elision; fold representatives own all members' events;
elision rows own everything they hide; root-level remainders are owned by
the footer (the root's totals) and disclosed. Alias/boundary rows are
non-accounting and do not enter the ladder.

Every numeric field is a plain int or None (the raw-numbers pin). All
spellings DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

#: The derived default body-row budget (memo 3.4): hybrid(vgg19) exactly.
DEFAULT_ROW_BUDGET = 48

#: Row kinds emitted by the ladder (closed vocabulary).
ROW_KINDS: tuple[str, ...] = (
    "module",
    "coalesced",
    "op",
    "fold",
    "elision",
    "unexecuted",
)

#: Trainability tri-state vocabulary (A12).
TRAIN_STATES: tuple[str, ...] = ("all", "some", "none")


@dataclass(frozen=True)
class ViewRow:
    """One rendered accounting row of a resolved summary view.

    ``params_display``/``flops_display`` are the per-path subtree values a
    reader compares against torchinfo; ``params_owned``/``flops_owned`` are
    the identity-partition shares (tie-deduped, disjoint across rows).
    ``owned_ops``/``owned_params`` make the partition testable.
    """

    row_id: str
    kind: str
    name: str
    class_name: str | None
    address: str | None
    depth: int
    output_shape: tuple[int, ...] | None
    passes: int
    params_display: int | None
    params_owned: int
    flops_display: int | None
    flops_owned: int
    macs_display: int | None
    macs_owned: int
    trainable: str | None
    fold_count: int
    elided_count: int
    evidence: str
    reason: str | None
    tied: bool
    protected: bool
    parent_address: str | None
    owned_ops: tuple[str, ...]
    owned_params: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedView:
    """A resolved rung: rows plus the disclosure facts the renderer prints."""

    rows: tuple[ViewRow, ...]
    rung: str
    depth: int | None
    max_depth: int
    fold_groups: int
    elision_rows: int
    budget: int
    disclosure: str
    root_owned_ops: tuple[str, ...]
    root_owned_params: tuple[str, ...]
    root_flops_owned: int
    root_params_owned: int
    mixed_trainability: bool

    @property
    def body_row_count(self) -> int:
        """Body rows between the header rule and the footer block."""

        return len(self.rows)


@dataclass
class _OpFact:
    """One executed layer-at-address joined across compute face and layer list.

    The multi-pass row model (memo 3.4): a layer owning p passes is ONE
    fact with summed additive values and pass-qualified owned labels; the
    per-pass grain survives on ``pass_facts`` for ``level="op"``.
    """

    label: str
    exec_index: int
    owner_address: str
    func_name: str | None
    shape: tuple[int, ...] | None
    dtype: str | None
    flops: int | None
    macs: int | None
    evidence: str
    reason: str | None
    coverage_class: str
    passes: int
    qualified_labels: tuple[str, ...] = ()


@dataclass
class _ParamFact:
    """One parameter identity with its tie group and owner path."""

    owner_path: str
    all_paths: tuple[str, ...]
    numel: int
    trainable: bool | None


@dataclass
class _ModuleFact:
    """One module-tree node with rollups the rung builders read."""

    address: str
    class_name: str
    depth: int
    children: list[str] = field(default_factory=list)
    direct_ops: list[_OpFact] = field(default_factory=list)
    output_shape: tuple[int, ...] | None = None
    num_calls: int = 1
    executed: bool = True
    first_exec: int = 1 << 30
    subtree_ops: list[_OpFact] = field(default_factory=list)
    subtree_params_display: int = 0
    owned_params: list[_ParamFact] = field(default_factory=list)
    subtree_owned_params: list[_ParamFact] = field(default_factory=list)
    tied: bool = False


def _short_name(address: str) -> str:
    """The attribute name of a module address (last dotted component)."""

    return address.rsplit(".", 1)[-1]


def _stack_owner(stack: tuple[str, ...]) -> str:
    """Innermost containing module address of an op ('' = root level)."""

    if not stack:
        return ""
    return str(stack[-1]).rsplit(":", 1)[0]


class _ModelIndex:
    """Everything the rung builders need, gathered from the trace ONCE.

    Reads the canonical compute aggregation (C02) for op values -- the
    ladder never re-derives compute from raw per-op fields -- and the
    module/param records for structure. Essential complexity: this is the
    one trace-shaped adapter; the rung builders below it are pure.
    """

    def __init__(self, trace: Trace) -> None:
        """Index ops, modules, and parameter identities for one trace."""

        from ._factcore import factcore

        self.core = factcore(trace)
        aggregation = self.core.compute
        entry_meta: dict[tuple[str, int], tuple[int, tuple[str, ...], Any, int]] = {}
        label_seen: dict[str, int] = {}
        for index, op in enumerate(getattr(trace, "layer_list", ()) or ()):
            label = str(getattr(op, "layer_label", "?"))
            occurrence = label_seen.get(label, 0)
            label_seen[label] = occurrence + 1
            shape = getattr(op, "shape", None)
            entry_meta[(label, occurrence)] = (
                index,
                tuple(getattr(op, "module_call_stack", ()) or ()),
                None if shape is None else tuple(int(dim) for dim in shape),
                int(getattr(op, "pass_index", occurrence + 1) or (occurrence + 1)),
            )
        self.pass_facts: list[_OpFact] = []
        row_seen: dict[str, int] = {}
        for row in aggregation.rows:
            if row.kind != "op":
                continue
            occurrence = row_seen.get(row.label, 0)
            row_seen[row.label] = occurrence + 1
            exec_index, stack, shape, pass_index = entry_meta.get(
                (row.label, occurrence), (1 << 30, (), None, occurrence + 1)
            )
            self.pass_facts.append(
                _OpFact(
                    label=row.label,
                    exec_index=exec_index,
                    owner_address=_stack_owner(stack),
                    func_name=row.func_name,
                    shape=shape,
                    dtype=row.dtype,
                    flops=row.flops_fma2,
                    macs=row.fma_macs,
                    evidence=row.evidence,
                    reason=row.reason,
                    coverage_class=row.coverage_class,
                    passes=1,
                    qualified_labels=(f"{row.label}:{pass_index}",),
                )
            )
        self.pass_facts.sort(key=lambda fact: fact.exec_index)
        self.ops = _group_layer_facts(self.pass_facts)
        self.params = _param_facts(trace)
        self.modules = _module_facts(trace, self.ops, self.params)
        self.root_children = _root_children(trace, self.modules)
        self.max_depth = max((mod.depth for mod in self.modules.values()), default=0)
        self.declared_params = self.core.params.total
        self.known_flops_total = int(self.core.compute.partition_total)
        self.root_direct_ops = [op for op in self.ops if op.owner_address == ""]
        self.mixed_trainability = _is_mixed_trainability(self.params)


def _group_layer_facts(pass_facts: list[_OpFact]) -> list[_OpFact]:
    """Group per-pass facts into layer-at-address facts (one row, p passes)."""

    grouped: dict[tuple[str, str], _OpFact] = {}
    ordered: list[tuple[str, str]] = []
    for fact in pass_facts:
        key = (fact.label, fact.owner_address)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = replace(fact)
            ordered.append(key)
            continue
        existing.exec_index = min(existing.exec_index, fact.exec_index)
        existing.shape = fact.shape or existing.shape
        existing.flops = _sum_optional([existing.flops, fact.flops])
        existing.macs = _sum_optional([existing.macs, fact.macs])
        existing.passes += 1
        existing.qualified_labels = existing.qualified_labels + fact.qualified_labels
        order = {"measured": 0, "formula_exact": 1, "estimated": 2, "hypothesis": 3, "unknown": 4}
        if order.get(fact.evidence, 4) > order.get(existing.evidence, 4):
            existing.evidence = fact.evidence
            existing.reason = fact.reason
    return [grouped[key] for key in ordered]


def _param_facts(trace: Trace) -> list[_ParamFact]:
    """Parameter identities with tie groups; owner = first alias path."""

    facts: list[_ParamFact] = []
    seen: set[str] = set()
    for param in getattr(trace, "params", ()) or ():
        paths = tuple(
            str(path) for path in (getattr(param, "all_addresses", None) or [param.address])
        )
        owner = paths[0]
        if owner in seen:
            continue
        seen.add(owner)
        numel = getattr(param, "num_params", None)
        shape = getattr(param, "shape", None)
        if numel is None and shape is not None:
            numel = 1
            for dim in shape:
                numel *= int(dim)
        facts.append(
            _ParamFact(
                owner_path=owner,
                all_paths=paths,
                numel=int(numel or 0),
                trainable=getattr(param, "is_trainable", None),
            )
        )
    return facts


def _is_mixed_trainability(params: list[_ParamFact]) -> bool:
    """True when the model holds both trainable and frozen parameters."""

    states = {bool(fact.trainable) for fact in params if fact.trainable is not None}
    return len(states) > 1


def _module_path_of(param_path: str) -> str:
    """Module address of a parameter path ('fc1.weight' -> 'fc1')."""

    return param_path.rsplit(".", 1)[0] if "." in param_path else ""


def _record_out_shape(record: Any) -> tuple[int, ...] | None:
    """Last-call output shape of a module record (multi-call safe)."""

    source: Any = record
    accessor = getattr(record, "calls", None)
    if accessor is not None:
        for key in (-1, int(getattr(record, "num_calls", 1) or 1)):
            try:
                source = accessor[key]
                break
            except Exception:  # noqa: BLE001, S112 - candidate-key probe over heterogeneous accessors
                continue
    try:
        shapes = getattr(source, "out_shapes", None)
    except Exception:  # noqa: BLE001 - Layer per-pass mirror reads raise ValueError, not AttributeError
        shapes = None
    shape = shapes[-1] if shapes else None
    if shape is None:
        try:
            shape = getattr(source, "out_shape", None)
        except Exception:  # noqa: BLE001 - same heterogeneous-record read; shape degrades to unknown
            shape = None
    if shape is None or not hasattr(shape, "__iter__"):
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _module_facts(
    trace: Trace, ops: list[_OpFact], params: list[_ParamFact]
) -> dict[str, _ModuleFact]:
    """Build the module-fact table, including declared-unexecuted modules."""

    facts: dict[str, _ModuleFact] = {}
    root_address = str(getattr(getattr(trace, "root_module", None), "address", "self"))
    for record in getattr(trace, "modules", ()) or ():
        for address in getattr(record, "all_addresses", None) or [record.address]:
            address = str(address)
            if address == root_address:
                continue
            out_shape = _record_out_shape(record)
            facts[address] = _ModuleFact(
                address=address,
                class_name=str(getattr(record, "class_name", "?")),
                depth=address.count(".") + 1,
                children=[str(child) for child in getattr(record, "address_children", []) or []],
                output_shape=None if out_shape is None else tuple(int(d) for d in out_shape),
                num_calls=int(getattr(record, "num_calls", 1) or 1),
            )
    _add_unexecuted_modules(trace, facts, params)
    _wire_children(facts)
    for op in ops:
        owner = facts.get(op.owner_address)
        if owner is not None:
            owner.direct_ops.append(op)
    _attach_subtrees(facts, ops, params)
    for fact in facts.values():
        if not fact.executed and fact.subtree_ops:
            fact.executed = True
    return facts


def _wire_children(facts: dict[str, _ModuleFact]) -> None:
    """Connect every fact to its structural parent by address prefix.

    Uncalled containers (a ModuleList is never CALLED even when every
    block inside it ran) are absent from declared child lists' records, so
    tree edges are re-derived structurally; declared order is preserved
    where records supplied it and structural extras append after.
    """

    for address, fact in facts.items():
        fact.children = [child for child in fact.children if child in facts]
        if "." not in address:
            continue
        parent = address.rsplit(".", 1)[0]
        parent_fact = facts.get(parent)
        if parent_fact is not None and address not in parent_fact.children:
            parent_fact.children.append(address)


def _param_module_classes(trace: Trace) -> dict[str, str]:
    """Recover module class names from param records (owner address -> name)."""

    classes: dict[str, str] = {}
    for param in getattr(trace, "params", ()) or ():
        module_address = str(getattr(param, "module_address", "") or "")
        try:
            # Param.module_cls resolves through the module accessor, which
            # raises KeyError for exactly the uncalled modules we are here
            # to synthesize -- so the class stays unknown for those.
            module_cls = getattr(param, "module_cls", None)
        except KeyError:
            module_cls = None
        if module_address and module_cls is not None:
            classes.setdefault(module_address, getattr(module_cls, "__name__", str(module_cls)))
    return classes


def _add_unexecuted_modules(
    trace: Trace, facts: dict[str, _ModuleFact], params: list[_ParamFact]
) -> None:
    """Synthesize rows for declared-but-uncalled modules (A3 visibility).

    Uncalled modules are absent from ``trace.modules``; their addresses ride
    ``trace.uncalled_modules`` and their class is recovered from the param
    records where possible. Paramless uncalled modules stay invisible (no
    partition impact: they own no identities).
    """

    classes = _param_module_classes(trace)
    candidates: list[str] = [
        str(address) for address in getattr(trace, "uncalled_modules", ()) or ()
    ]
    # ``uncalled_modules`` reads a WEAKREF to the source model and returns []
    # once the model is collected, so the declared-unexecuted set is ALSO
    # derived structurally: any param owner path without a module record, and
    # every implied ancestor of a known fact (an uncalled ModuleList between
    # called blocks), stays visible on the detached report.
    for fact in params:
        for path in fact.all_paths:
            module_path = _module_path_of(path)
            while module_path:
                candidates.append(module_path)
                module_path = module_path.rsplit(".", 1)[0] if "." in module_path else ""
    for address in list(facts):
        while "." in address:
            address = address.rsplit(".", 1)[0]
            candidates.append(address)
    seen: set[str] = set()
    for address in candidates:
        if not address or address in facts or address in seen:
            continue
        seen.add(address)
        has_params = any(
            _module_path_of(fact.owner_path) == address
            or _module_path_of(fact.owner_path).startswith(address + ".")
            for fact in params
        )
        has_descendants = any(other.startswith(address + ".") for other in facts)
        if not has_params and not has_descendants:
            continue
        facts[address] = _ModuleFact(
            address=address,
            class_name=classes.get(address, ""),
            depth=address.count(".") + 1,
            executed=False,
        )


def _attach_param_rollups(facts: dict[str, _ModuleFact], params: list[_ParamFact]) -> None:
    """Roll one param fact set up into display totals, ties, and ownership."""

    for fact in params:
        owner_module = _module_path_of(fact.owner_path)
        tied = len(fact.all_paths) > 1
        display_modules: set[str] = set()
        for path in fact.all_paths:
            address = _module_path_of(path)
            while address:
                display_modules.add(address)
                address = address.rsplit(".", 1)[0] if "." in address else ""
        for address in display_modules:
            owner = facts.get(address)
            if owner is None:
                continue
            owner.subtree_params_display += fact.numel
            if tied:
                owner.tied = True
        address = owner_module
        if address in facts:
            facts[address].owned_params.append(fact)
        while address:
            owner = facts.get(address)
            if owner is not None:
                owner.subtree_owned_params.append(fact)
            address = address.rsplit(".", 1)[0] if "." in address else ""


def _attach_subtrees(
    facts: dict[str, _ModuleFact], ops: list[_OpFact], params: list[_ParamFact]
) -> None:
    """Fill subtree rollups: ops, first-execution index, params, ties."""

    ordered = sorted(facts, key=len, reverse=True)
    for op in ops:
        address = op.owner_address
        while address:
            owner = facts.get(address)
            if owner is not None:
                owner.subtree_ops.append(op)
                owner.first_exec = min(owner.first_exec, op.exec_index)
            address = address.rsplit(".", 1)[0] if "." in address else ""
    _attach_param_rollups(facts, params)
    for address in ordered:
        facts[address].subtree_ops.sort(key=lambda op: op.exec_index)


def _root_children(trace: Trace, facts: dict[str, _ModuleFact]) -> list[str]:
    """Top-level module addresses in declared order."""

    root = getattr(trace, "root_module", None)
    declared = [str(child) for child in getattr(root, "address_children", []) or []]
    for address, fact in facts.items():
        if fact.depth == 1 and address not in declared:
            declared.append(address)
    return [address for address in declared if address in facts]


def _train_state(params: list[_ParamFact]) -> str | None:
    """A12 tri-state over one row's parameter set."""

    known = [fact.trainable for fact in params if fact.trainable is not None]
    if not known:
        return None
    if all(known):
        return "all"
    if not any(known):
        return "none"
    return "some"


def _sum_optional(values: list[int | None]) -> int | None:
    """Sum where at least one value is known; None when all are None."""

    known = [value for value in values if value is not None]
    if not known:
        return None
    return sum(known)


def _rollup_evidence(ops: list[_OpFact]) -> tuple[str, str | None]:
    """A rollup never upgrades past its weakest member's evidence."""

    order = {"measured": 0, "formula_exact": 1, "estimated": 2, "hypothesis": 3, "unknown": 4}
    worst = "formula_exact"
    reason: str | None = None
    for op in ops:
        if order.get(op.evidence, 4) > order.get(worst, 4):
            worst = op.evidence
            reason = op.reason
    if not ops:
        return "unknown", "no executed ops"
    return worst, reason


def _module_row(index: _ModelIndex, fact: _ModuleFact, *, own_ops: bool) -> ViewRow:
    """One module row; ``own_ops`` rolls the whole subtree into this row."""

    owned_ops = fact.subtree_ops if own_ops else []
    owned_params = fact.subtree_owned_params if own_ops else fact.owned_params
    evidence, reason = _rollup_evidence(fact.subtree_ops)
    if not fact.executed:
        evidence, reason = "measured", "declared; never ran"
    return ViewRow(
        row_id=f"mod:{fact.address}",
        kind="module" if fact.executed else "unexecuted",
        name=_short_name(fact.address),
        class_name=fact.class_name,
        address=fact.address,
        depth=fact.depth,
        output_shape=fact.output_shape,
        passes=fact.num_calls,
        params_display=fact.subtree_params_display or None,
        params_owned=sum(param.numel for param in owned_params),
        flops_display=_sum_optional([op.flops for op in fact.subtree_ops]),
        flops_owned=sum(op.flops or 0 for op in owned_ops),
        macs_display=_sum_optional([op.macs for op in fact.subtree_ops]),
        macs_owned=sum(op.macs or 0 for op in owned_ops),
        trainable=_train_state(fact.subtree_owned_params),
        fold_count=1,
        elided_count=0,
        evidence=evidence,
        reason=reason,
        tied=fact.tied,
        protected=False,
        parent_address=fact.address.rsplit(".", 1)[0] if "." in fact.address else None,
        owned_ops=tuple(label for op in owned_ops for label in op.qualified_labels),
        owned_params=tuple(param.owner_path for param in owned_params),
    )


def _op_row(op: _OpFact, depth: int, parent: str | None) -> ViewRow:
    """One orphan/expanded executed-op row."""

    return ViewRow(
        row_id=f"op:{op.label}",
        kind="op",
        name=op.label,
        class_name=op.func_name,
        address=None,
        depth=depth,
        output_shape=op.shape,
        passes=op.passes,
        params_display=None,
        params_owned=0,
        flops_display=op.flops,
        flops_owned=op.flops or 0,
        macs_display=op.macs,
        macs_owned=op.macs or 0,
        trainable=None,
        fold_count=1,
        elided_count=0,
        evidence=op.evidence,
        reason=op.reason,
        tied=False,
        protected=False,
        parent_address=parent,
        owned_ops=op.qualified_labels,
        owned_params=(),
    )


def _coalesced_row(index: _ModelIndex, fact: _ModuleFact, op: _OpFact) -> ViewRow:
    """A leaf module owning exactly one op coalesces into one row."""

    row = _module_row(index, fact, own_ops=True)
    return replace(row, row_id=f"coal:{fact.address}", kind="coalesced")


def _ordered_items(index: _ModelIndex, address: str | None) -> list[Any]:
    """A container's children (modules + direct orphan ops) by execution."""

    if address is None:
        children = list(index.root_children)
        direct = list(index.root_direct_ops)
    else:
        fact = index.modules[address]
        children = [child for child in fact.children if child in index.modules]
        direct = list(fact.direct_ops)
    items: list[tuple[int, int, Any]] = []
    for position, child in enumerate(children):
        items.append((index.modules[child].first_exec, position, index.modules[child]))
    for op in direct:
        items.append((op.exec_index, 1 << 20, op))
    items.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in items]


def build_hybrid(index: _ModelIndex) -> list[ViewRow]:
    """Rung 1: the coalesced hybrid -- every executed op represented once."""

    rows: list[ViewRow] = []

    def visit(address: str | None, depth: int) -> None:
        """DFS in execution-then-declaration order."""

        for item in _ordered_items(index, address):
            if isinstance(item, _OpFact):
                rows.append(_op_row(item, depth, address))
                continue
            fact = item
            has_children = any(child in index.modules for child in fact.children)
            if not fact.executed:
                rows.append(_module_row(index, fact, own_ops=True))
            elif not has_children and len(fact.subtree_ops) == 1:
                rows.append(_coalesced_row(index, fact, fact.subtree_ops[0]))
            elif not has_children and not fact.direct_ops:
                rows.append(_module_row(index, fact, own_ops=True))
            else:
                rows.append(_module_row(index, fact, own_ops=False))
                visit(fact.address, depth + 1)

    visit(None, 1)
    return rows


def _subtree_signature(index: _ModelIndex, address: str, cache: dict[str, Any]) -> Any:
    """The strict fold signature (memo 3.4), computed recursively.

    Class, ordered child topology, parameter shapes+trainability, executed-op
    schema, output-shape schema, evidence states, and tie ownership -- the
    union signature that measured weaker variants folding into lies.
    """

    if address in cache:
        return cache[address]
    fact = index.modules[address]
    params = tuple(
        sorted(
            (
                fact.owner_path[len(address) :],
                fact_numel,
                bool(trainable) if trainable is not None else None,
            )
            for fact, fact_numel, trainable in (
                (param, param.numel, param.trainable) for param in fact.owned_params
            )
        )
    )
    op_schema = tuple((op.func_name, op.coverage_class, op.passes) for op in fact.direct_ops)
    children = tuple(
        _subtree_signature(index, child, cache) for child in fact.children if child in index.modules
    )
    signature = (
        fact.class_name,
        fact.executed,
        fact.num_calls,
        fact.output_shape,
        params,
        op_schema,
        children,
        fact.tied,
        tuple(sorted({op.evidence for op in fact.subtree_ops})),
    )
    cache[address] = signature
    return signature


def _fold_row(index: _ModelIndex, members: list[_ModuleFact]) -> ViewRow:
    """One representative row owning all fold members' identities."""

    first, last = members[0], members[-1]
    rep = _module_row(index, first, own_ops=True)
    owned_ops: list[str] = []
    owned_params: list[str] = []
    params_display = 0
    flops_display: list[int | None] = []
    macs_display: list[int | None] = []
    flops_owned = 0
    macs_owned = 0
    all_params: list[_ParamFact] = []
    for member in members:
        owned_ops.extend(label for op in member.subtree_ops for label in op.qualified_labels)
        owned_params.extend(param.owner_path for param in member.subtree_owned_params)
        all_params.extend(member.subtree_owned_params)
        params_display += member.subtree_params_display
        flops_display.append(_sum_optional([op.flops for op in member.subtree_ops]))
        macs_display.append(_sum_optional([op.macs for op in member.subtree_ops]))
        flops_owned += sum(op.flops or 0 for op in member.subtree_ops)
        macs_owned += sum(op.macs or 0 for op in member.subtree_ops)
    return replace(
        rep,
        row_id=f"fold:{first.address}..{_short_name(last.address)}",
        kind="fold",
        name=f"{_short_name(first.address)}..{_short_name(last.address)}",
        fold_count=len(members),
        params_display=params_display or None,
        params_owned=sum(param.numel for param in all_params),
        flops_display=_sum_optional(flops_display),
        flops_owned=flops_owned,
        macs_display=_sum_optional(macs_display),
        macs_owned=macs_owned,
        trainable=_train_state(all_params),
        owned_ops=tuple(owned_ops),
        owned_params=tuple(owned_params),
    )


def _fold_siblings(index: _ModelIndex, addresses: list[str]) -> list[Any]:
    """Group contiguous sibling runs (>= 2) of equal strict signatures."""

    cache: dict[str, Any] = {}
    groups: list[list[_ModuleFact]] = []
    for address in addresses:
        fact = index.modules[address]
        signature = _subtree_signature(index, address, cache)
        if groups and cache.get(groups[-1][-1].address) == signature:
            groups[-1].append(fact)
        else:
            groups.append([fact])
    return groups


def _build_tree_interior(index: _ModelIndex, address: str, level: int, cut: int) -> list[ViewRow]:
    """Interior rows of a fold representative (display-only ownership).

    Interior rows under a fold representative display the FIRST
    member's values but own nothing -- the fold row owns all members'
    identities (one ownership rule for all one-to-many constructs).
    """

    interior: list[ViewRow] = []

    def walk(parent: str, walk_level: int) -> None:
        """Emit display-only rows for one interior subtree, folding runs."""

        children = [c for c in index.modules[parent].children if c in index.modules]
        children = sorted(children, key=lambda a: (index.modules[a].first_exec, a))
        for group in _fold_siblings(index, children):
            fact = group[0]
            deeper = walk_level < cut and any(c in index.modules for c in fact.children)
            base = (
                _fold_row(index, group)
                if len(group) >= 2
                else _module_row(index, fact, own_ops=True)
            )
            interior.append(
                replace(
                    base,
                    params_owned=0,
                    flops_owned=0,
                    macs_owned=0,
                    owned_ops=(),
                    owned_params=(),
                )
            )
            if deeper and len(group) < 2:
                interior[-1] = replace(
                    interior[-1],
                    owned_ops=(),
                    owned_params=(),
                )
                walk(fact.address, walk_level + 1)

    walk(address, level)
    return interior


def build_tree(index: _ModelIndex, depth: int, *, fold: bool = True) -> list[ViewRow]:
    """Rungs 2-3: the strictly folded module tree cut at ``depth``."""

    rows: list[ViewRow] = []

    def visit(address: str | None, level: int) -> None:
        """Emit module rows to the depth cut, folding sibling runs."""

        if address is None:
            children = index.root_children
        else:
            children = [c for c in index.modules[address].children if c in index.modules]
        children = sorted(children, key=lambda a: (index.modules[a].first_exec, a))
        groups = _fold_siblings(index, children) if fold else [[index.modules[c]] for c in children]
        for group in groups:
            fact = group[0]
            deeper = level < depth and any(child in index.modules for child in fact.children)
            if len(group) >= 2:
                rows.append(_fold_row(index, group))
                if deeper:
                    # A folded group renders ONE representative interior.
                    rows.extend(_build_tree_interior(index, fact.address, level + 1, depth))
            elif deeper:
                rows.append(_module_row(index, fact, own_ops=False))
                visit(fact.address, level + 1)
            else:
                rows.append(_module_row(index, fact, own_ops=True))

    visit(None, 1)
    return rows


def _mark_protected(rows: list[ViewRow], filter_hits: set[str]) -> list[ViewRow]:
    """Apply the deterministic elision-protection rules (memo 3.4)."""

    marked: list[ViewRow] = []
    by_parent: dict[str | None, list[int]] = {}
    for position, row in enumerate(rows):
        by_parent.setdefault(row.parent_address, []).append(position)
    edge_positions: set[int] = set()
    for positions in by_parent.values():
        edge_positions.add(positions[0])
        edge_positions.add(positions[-1])
    for position, row in enumerate(rows):
        previous = rows[position - 1] if position > 0 else None
        transition = (
            previous is not None
            and previous.parent_address == row.parent_address
            and previous.output_shape is not None
            and row.output_shape is not None
            and previous.output_shape != row.output_shape
        )
        protected = (
            row.kind in ("module", "fold", "elision")
            and any(other.parent_address == row.address for other in rows)
            or position in edge_positions
            or transition
            or row.tied
            or row.trainable == "some"
            or row.evidence in ("unknown", "hypothesis")
            or row.kind == "unexecuted"
            or row.row_id in filter_hits
        )
        marked.append(replace(row, protected=bool(protected)))
    return marked


def _eligible_runs(rows: list[ViewRow]) -> list[tuple[int, int]]:
    """Maximal runs (>= 2) of adjacent unprotected same-parent rows."""

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for position, row in enumerate(rows):
        eligible = not row.protected and row.kind != "elision"
        contiguous = (
            start is not None
            and eligible
            and rows[position - 1].parent_address == row.parent_address
        )
        if contiguous:
            continue
        if start is not None:
            runs.append((start, position))
        start = position if eligible else None
    if start is not None:
        runs.append((start, len(rows)))
    return [(run_start, run_end) for run_start, run_end in runs if run_end - run_start >= 2]


def _elide_once(rows: list[ViewRow], index: _ModelIndex, excess: int) -> list[ViewRow] | None:
    """Replace the lowest-share window of one unprotected run.

    Hides only what the budget requires (``excess + 1`` rows net one
    accounting row), never a whole run when a window suffices -- the
    elision rung is a presentation preference, not a correctness cliff.
    """

    runs = _eligible_runs(rows)
    if not runs:
        return None
    declared = index.declared_params or 0
    flops_total = index.known_flops_total or 0

    def span_score(span: tuple[int, int]) -> float:
        """max(param share, known-FLOP share) of a row span."""

        span_start, span_end = span
        params = sum(rows[i].params_owned for i in range(span_start, span_end))
        flops = sum(rows[i].flops_owned for i in range(span_start, span_end))
        param_share = params / declared if declared else 0.0
        flop_share = flops / flops_total if flops_total else 0.0
        return max(param_share, flop_share) if flops_total else param_share

    def run_key(run: tuple[int, int]) -> tuple[float, int, int]:
        """Lowest score wins; ties break to the longer run, then earlier."""

        return (span_score(run), -(run[1] - run[0]), run[0])

    chosen = min(runs, key=run_key)
    run_start, run_end = chosen
    window = min(run_end - run_start, max(excess + 1, 2))
    candidates = [(start_, start_ + window) for start_ in range(run_start, run_end - window + 1)]
    start_, end_ = min(candidates, key=lambda span: (span_score(span), span[0]))
    members = rows[start_:end_]
    hidden_ops: list[str] = []
    hidden_params: list[str] = []
    for member in members:
        hidden_ops.extend(member.owned_ops)
        hidden_params.extend(member.owned_params)
    first_address = members[0].address or members[0].name
    last_address = members[-1].address or members[-1].name
    elision = ViewRow(
        row_id=f"elide:{first_address}..{last_address}",
        kind="elision",
        name=f"{first_address}..{last_address}",
        class_name=None,
        address=None,
        depth=members[0].depth,
        output_shape=members[-1].output_shape,
        passes=1,
        params_display=_sum_optional([member.params_display for member in members]),
        params_owned=sum(member.params_owned for member in members),
        flops_display=_sum_optional([member.flops_display for member in members]),
        flops_owned=sum(member.flops_owned for member in members),
        macs_display=_sum_optional([member.macs_display for member in members]),
        macs_owned=sum(member.macs_owned for member in members),
        trainable=None,
        fold_count=1,
        elided_count=sum(max(member.fold_count, 1) for member in members),
        evidence="measured",
        reason=None,
        tied=any(member.tied for member in members),
        protected=True,
        parent_address=members[0].parent_address,
        owned_ops=tuple(hidden_ops),
        owned_params=tuple(hidden_params),
    )
    return rows[:start_] + [elision] + rows[end_:]


def elide(rows: list[ViewRow], index: _ModelIndex, budget: int) -> tuple[list[ViewRow], int]:
    """The elision rung: protected, deterministic, totals-conserving."""

    working = _mark_protected(rows, set())
    elisions = 0
    while len(working) > budget:
        result = _elide_once(working, index, len(working) - budget)
        if result is None or len(result) >= len(working):
            break
        working = result
        elisions += 1
    return working, elisions


def _count_folds(rows: list[ViewRow]) -> int:
    """Number of fold-representative rows in a view."""

    return sum(1 for row in rows if row.kind == "fold")


def resolve_auto(
    index: _ModelIndex,
    budget: int = DEFAULT_ROW_BUDGET,
    *,
    fold: bool = True,
) -> ResolvedView:
    """The auto ladder resolver (memo 3.4 pseudocode, verbatim semantics)."""

    rungs: list[tuple[str, int | None, list[ViewRow]]] = []
    rungs.append(("hybrid", None, build_hybrid(index)))
    depth_max = index.max_depth
    for depth in range(depth_max, 0, -1):
        rungs.append(("tree", depth, build_tree(index, depth, fold=fold)))
    fit: tuple[str, int | None, list[ViewRow]] | None = None
    for rung in rungs:
        if len(rung[2]) <= budget:
            fit = rung
            break
    if fit is None:
        base = build_tree(index, 1, fold=fold)
        rows, elisions = elide(base, index, budget)
        return _finish(index, rows, "elided", 1, elisions, budget)
    rung_name, rung_depth, rows = fit
    if rung_name == "hybrid":
        return _finish(index, rows, "hybrid", None, 0, budget)
    if len(rows) >= max(budget // 4, 1):
        return _finish(index, rows, "tree", rung_depth, 0, budget)
    deeper = _one_rung_deeper(rungs, fit)
    if deeper is None:
        return _finish(index, rows, "tree", rung_depth, 0, budget)
    deeper_name, deeper_depth, deeper_rows = deeper
    elided_rows, elisions = elide(list(deeper_rows), index, budget)
    if len(elided_rows) <= budget:
        return _finish(
            index,
            elided_rows,
            "elided" if elisions else deeper_name,
            deeper_depth,
            elisions,
            budget,
        )
    return _finish(index, rows, "tree", rung_depth, 0, budget)


def _one_rung_deeper(
    rungs: list[tuple[str, int | None, list[ViewRow]]],
    fit: tuple[str, int | None, list[ViewRow]],
) -> tuple[str, int | None, list[ViewRow]] | None:
    """The rung immediately deeper than ``fit`` on the ladder."""

    position = rungs.index(fit)
    if position == 0:
        return None
    return rungs[position - 1]


def resolve_view(
    trace: Trace,
    *,
    level: str = "auto",
    depth: Any = "auto",
    max_rows: int = DEFAULT_ROW_BUDGET,
    fold_repeats: Any = "auto",
) -> ResolvedView:
    """Resolve one summary view for a finished trace (the public door)."""

    index = _ModelIndex(trace)
    fold = fold_repeats in ("auto", True, None)
    if level == "op":
        rows = []
        for fact in index.pass_facts:
            row = _op_row(fact, 1, None)
            if len(fact.qualified_labels) == 1 and ":" in fact.qualified_labels[0]:
                pass_index = fact.qualified_labels[0].rsplit(":", 1)[1]
                layer_passes = sum(1 for f in index.pass_facts if f.label == fact.label)
                if layer_passes > 1:
                    row = replace(row, name=f"{fact.label}:{pass_index}")
            rows.append(row)
        return _finish(index, rows, "op", None, 0, max_rows or DEFAULT_ROW_BUDGET)
    if level == "module" or (level == "auto" and depth not in ("auto", None)):
        if depth in ("auto", None, "all"):
            resolved_depth = index.max_depth
        else:
            resolved_depth = max(1, min(int(depth), index.max_depth or 1))
        rows = build_tree(index, resolved_depth, fold=fold)
        return _finish(index, rows, "tree", resolved_depth, 0, max_rows or DEFAULT_ROW_BUDGET)
    return resolve_auto(index, max_rows or DEFAULT_ROW_BUDGET, fold=fold)


def _finish(  # noqa: PLR0913 - the resolved-view assembly facts travel together by design
    index: _ModelIndex,
    rows: list[ViewRow],
    rung: str,
    depth: int | None,
    elisions: int,
    budget: int,
) -> ResolvedView:
    """Assemble the resolved view with its disclosure line."""

    folds = _count_folds(rows)
    owned_ops = {label for row in rows for label in row.owned_ops}
    root_ops = tuple(
        label
        for fact in index.pass_facts
        for label in fact.qualified_labels
        if label not in owned_ops
    )
    owned_params = {path for row in rows for path in row.owned_params}
    root_params = tuple(
        fact.owner_path for fact in index.params if fact.owner_path not in owned_params
    )
    disclosure = _disclosure(index, rows, rung, depth, folds, elisions)
    return ResolvedView(
        rows=tuple(rows),
        rung=rung,
        depth=depth,
        max_depth=index.max_depth,
        fold_groups=folds,
        elision_rows=elisions,
        budget=budget,
        disclosure=disclosure,
        root_owned_ops=root_ops,
        root_owned_params=root_params,
        root_flops_owned=sum(
            fact.flops or 0
            for fact in index.pass_facts
            if fact.qualified_labels[0] not in owned_ops
        ),
        root_params_owned=sum(
            fact.numel for fact in index.params if fact.owner_path not in owned_params
        ),
        mixed_trainability=index.mixed_trainability,
    )


def _disclosure(  # noqa: PLR0913 - mirrors _finish: the disclosure names exactly what resolved
    index: _ModelIndex,
    rows: list[ViewRow],
    rung: str,
    depth: int | None,
    folds: int,
    elisions: int,
) -> str:
    """The always-printed disclosure line naming what resolved."""

    if rung == "hybrid":
        return f"view: hybrid, all {len(index.pass_facts)} ops"
    if rung == "op":
        return f"view: op level, {len(rows)} rows"
    if rung == "elided" and depth is None:
        base = "hybrid"
    else:
        base = f"module tree, depth {depth} of {max(index.max_depth, 1)}"
    parts = [f"view: {base}"]
    if folds:
        parts.append(f"{folds} fold{'s' if folds != 1 else ''}")
    if elisions:
        hidden = sum(row.elided_count for row in rows if row.kind == "elision")
        parts.append(f"{hidden} rows elided")
    if index.root_direct_ops and rung != "hybrid":
        parts.append(f"{len(index.root_direct_ops)} root-level ops in totals")
    return ", ".join(parts)
