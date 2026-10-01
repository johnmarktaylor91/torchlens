"""EPISODE capture export: one collection, full graph + per-step graphs.

Memo D11-D14. Step membership joins through the persisted
``trace.module_calls["<stepped>:<member_call_index>"].ops`` list (KEEP-policy
portable state, measured intact on saved-and-reloaded real episodes) with
the ``module_call_stack[0]`` pass-suffix join as the MANDATORY cross-check
asserted EQUAL -- a lane coding any single-source spelling ships one of the
two measured crashes. Stackless driver ops are assigned exactly ONCE by
execution window and grouped under a ``driver`` namespace. Cross-step edges
get deterministic boundary proxies by default (reserved id prefix, derived
from the SOURCE site key + graph-local ordinal, never an absolute step
number); ``boundary_proxies=False`` gives drop-plus-disclose. The full exact
episode graph is ALWAYS emitted; the 12 MB / 128-graph budget binds the
per-step graphs, and omitted steps are listed with the flag that raises the
budget.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from ._attrs import AttrContext, dtype_string, shape_string
from ._build import BuildResult, GraphSpec, build_graph
from ._edges import build_label_map
from ._errors import ModelExplorerExportError
from ._ids import PROXY_ID_PREFIX, mint_node_ids
from ._options import ModelExplorerOptions

__tl_layer__ = "L8"


@dataclass
class _StepPlan:
    """One ledger row resolved to its member and driver entries."""

    step: int
    role: str
    status: str
    call_label: str | None
    entries: list[Any]
    step_output: Any
    frontier: Any


@dataclass(frozen=True)
class EpisodeExportContext:
    """Per-collection state threaded through the episode graph builders.

    The collection builder constructs it with the first three fields; the
    stepped address and driver labels are derived from the ledger inside
    ``build_episode_graphs`` and threaded onward via ``replace``.
    """

    attr_context: AttrContext
    root_facts: dict[str, str]
    options: ModelExplorerOptions
    stepped_address: str = ""
    driver_labels: frozenset[str] = frozenset()


def build_episode_graphs(
    log: Any,
    entries: list[Any],
    ledger: dict[str, Any],
    context: EpisodeExportContext,
) -> list[dict[str, Any]]:
    """Build the episode collection's graph list (full graph first)."""

    header = ledger.get("header") or {}
    rows = ledger.get("rows") or []
    stepped_address = str(header.get("stepped_module", "") or "")
    if not stepped_address or not isinstance(rows, list):
        raise ModelExplorerExportError(
            "Episode ledger is missing its stepped-module address or rows; the "
            "per-step export cannot derive step membership",
            code="model_explorer_episode_ledger_unavailable",
            remedy="re-capture with episode=EpisodeSpec(...) or export with per_step=False",
        )
    plans, driver_labels = _plan_steps(log, entries, rows, stepped_address)
    episode_facts = {
        **context.root_facts,
        "view": "episode (full exact execution)",
        "episode_id": str(header.get("episode_id", "")),
        "n_steps": str(len(rows)),
        "token_feed": str(header.get("token_feed", "")),
    }
    full = build_graph(
        log,
        entries,
        GraphSpec(
            graph_id="00-episode",
            attr_context=context.attr_context,
            strict_namespace=context.options.strict_namespace,
            driver_labels=driver_labels,
            root_facts=episode_facts,
        ),
    )
    graphs = [full.graph]
    if context.options.per_step is False:
        return graphs
    step_context = replace(context, stepped_address=stepped_address, driver_labels=driver_labels)
    _append_step_graphs(log, plans, graphs, full, step_context)
    return graphs


def _plan_steps(
    log: Any,
    entries: list[Any],
    rows: list[dict[str, Any]],
    stepped_address: str,
) -> tuple[list[_StepPlan], frozenset[str]]:
    """Resolve ledger rows to member entries and window-assign drivers."""

    position = {id(entry): index for index, entry in enumerate(entries)}
    plans: list[_StepPlan] = []
    member_positions: set[int] = set()
    for row in rows:
        coord = row.get("coord") or {}
        call_index = coord.get("member_call_index")
        # pass_range is None on every row of a complete episode; it is
        # deliberately never dereferenced (memo D11 crash 2).
        call_label = f"{stepped_address}:{int(call_index)}" if call_index is not None else None
        members = _step_members(log, entries, call_label, stepped_address)
        for member in members:
            member_positions.add(position[id(member)])
        plans.append(
            _StepPlan(
                step=int(row.get("episode_step", len(plans))),
                role=str(row.get("role", "")),
                status=str(row.get("status", "")),
                call_label=call_label,
                entries=members,
                # Grammar v2 (C07X): tokens/cache_len are gone; step_output
                # is the generic per-step evidence, and no fact may imply
                # an unmeasured cache length.
                step_output=row.get("step_output"),
                frontier=row.get("frontier"),
            )
        )
    driver_labels = _assign_drivers(entries, plans, member_positions, position)
    return plans, driver_labels


def _step_members(
    log: Any, entries: list[Any], call_label: str | None, stepped_address: str
) -> list[Any]:
    """Join one step's member ops, asserting both spellings agree EXACTLY."""

    if call_label is None:
        return []
    try:
        primary = list(log.module_calls[call_label].ops)
    except Exception as exc:
        raise ModelExplorerExportError(
            f"Episode step module call {call_label!r} is missing from the trace's "
            "module-call table; the ledger and the capture disagree",
            code="model_explorer_episode_ledger_unavailable",
            remedy="re-capture the episode, or export with per_step=False",
        ) from exc
    cross_check = {
        str(getattr(entry, "label", "")) for entry in entries if _stack_head(entry) == call_label
    }
    if set(primary) != cross_check:
        raise ModelExplorerExportError(
            f"Episode step membership joins disagree for {call_label!r}: the "
            f"module-call ops list has {len(primary)} labels, the stack join "
            f"{len(cross_check)}; exporting either silently would misattribute ops",
            code="model_explorer_episode_join_mismatch",
            remedy=(
                "this indicates a TorchLens episode-capture bug; report it with the "
                "trace's summary() and this call label"
            ),
            call_label=call_label,
            stepped_address=stepped_address,
        )
    entry_by_label = {str(getattr(entry, "label", "")): entry for entry in entries}
    return [entry_by_label[label] for label in primary if label in entry_by_label]


def _stack_head(entry: Any) -> str | None:
    """Return the first module-call-stack entry (the stack join spelling)."""

    stack = list(getattr(entry, "module_call_stack", ()) or ())
    return str(stack[0]) if stack else None


def _assign_drivers(
    entries: list[Any],
    plans: list[_StepPlan],
    member_positions: set[int],
    position: dict[int, int],
) -> frozenset[str]:
    """Window-assign stackless driver ops exactly once (memo D11)."""

    # Essential complexity (CC>10 named): window assignment, boundary
    # exemption, and driver labeling are one execution-order sweep; the
    # exactly-once guarantee lives in its shared state.
    spans = [
        (min(position[id(entry)] for entry in plan.entries), plan_index)
        for plan_index, plan in enumerate(plans)
        if plan.entries
    ]
    driver_labels: set[str] = set()
    for entry in entries:
        entry_position = position[id(entry)]
        if entry_position in member_positions:
            continue
        if getattr(entry, "module_call_stack", ()) or ():
            continue
        is_boundary = getattr(entry, "is_input", False) or getattr(entry, "is_output", False)
        if not is_boundary:
            driver_labels.add(str(getattr(entry, "label", "")))
        target = _window_step(entry_position, spans)
        if target is not None:
            plans[target].entries.append(entry)
    for plan in plans:
        plan.entries.sort(key=lambda entry: position[id(entry)])
    return frozenset(driver_labels)


def _window_step(entry_position: int, spans: list[tuple[int, int]]) -> int | None:
    """Return the plan index owning one execution-window position.

    Preamble positions belong to the first step; every later position
    belongs to the last step whose members started at or before it
    (between-calls windows attach to the PRECEDING step; the tail to the
    final step).
    """

    if not spans:
        return None
    owner = spans[0][1]
    for start, plan_index in spans:
        if start <= entry_position:
            owner = plan_index
    return owner


def _append_step_graphs(
    log: Any,
    plans: list[_StepPlan],
    graphs: list[dict[str, Any]],
    full_graph: BuildResult,
    context: EpisodeExportContext,
) -> None:
    """Append budgeted per-step graphs, disclosing every omission."""

    width = max(2, len(str(len(plans) + 1)))
    spent = sum(len(json.dumps(graph)) for graph in graphs)
    omitted: list[int] = []
    for plan in plans:
        if (
            len(graphs) - 1 >= context.options.max_step_graphs
            or spent >= context.options.step_budget_bytes
        ):
            omitted.append(plan.step)
            continue
        graph = _build_step_graph(
            log,
            plan,
            f"{plan.step + 1:0{width}d}-step-{plan.step}",
            context,
        )
        spent += len(json.dumps(graph))
        graphs.append(graph)
    if omitted:
        disclosure = (
            f"steps {omitted[0]}-{omitted[-1]} omitted by the per-collection budget "
            "(raise step_budget_bytes / max_step_graphs to include them)"
        )
        full_graph.graph["groupNodeAttributes"][""]["omitted_step_graphs"] = disclosure


def _build_step_graph(
    log: Any,
    plan: _StepPlan,
    graph_id: str,
    context: EpisodeExportContext,
) -> dict[str, Any]:
    """Build one per-step graph with boundary proxies or drop-plus-disclose."""

    facts = {
        **context.root_facts,
        "view": f"episode step {plan.step}",
        "step": str(plan.step),
        "role": plan.role,
        "status": plan.status,
    }
    if plan.step_output and context.options.privacy_profile != "public":
        facts["step_output"] = json.dumps(plan.step_output, default=str)[:500]
    if plan.frontier is not None:
        facts["frontier"] = json.dumps(plan.frontier, default=str)[:500]
    if not plan.entries:
        return {
            "id": graph_id,
            "nodes": [],
            "groupNodeAttributes": {"": {**facts, "nodes": "0", "edges": "0"}},
        }
    proxy_map: dict[str, str] = {}
    proxy_nodes: tuple[dict[str, Any], ...] = ()
    if context.options.boundary_proxies:
        proxy_map, proxy_nodes = _boundary_proxies(log, plan.entries)
    result = build_graph(
        log,
        plan.entries,
        GraphSpec(
            graph_id=graph_id,
            attr_context=context.attr_context,
            strict_namespace=context.options.strict_namespace,
            strip_prefix_entry=plan.call_label,
            driver_labels=context.driver_labels,
            extra_label_map=proxy_map,
            extra_nodes=proxy_nodes,
            resolve_missing="raise" if context.options.boundary_proxies else "skip",
            root_facts=facts,
        ),
    )
    if proxy_nodes:
        result.graph["groupNodeAttributes"][""]["boundary_proxies"] = str(len(proxy_nodes))
    return result.graph


def _boundary_proxies(
    log: Any, step_entries: list[Any]
) -> tuple[dict[str, str], tuple[dict[str, Any], ...]]:
    """Mint deterministic boundary-proxy nodes for cross-step parents.

    One proxy per out-of-step source op; ids derive from the SOURCE site key
    plus a graph-local ordinal (memo D12), so per-step id sets stay
    step-stable. Proxies carry ``kind=step_boundary`` and shape/dtype
    metadata, NO measured values.
    """

    in_step = build_label_map(step_entries, [""] * len(step_entries))
    source_map = _source_map(_all_entries(log))
    proxy_map: dict[str, str] = {}
    proxy_nodes: list[dict[str, Any]] = []
    proxy_entries: list[Any] = []
    node_by_source: dict[int, str] = {}
    for entry in step_entries:
        for ref in _entry_parent_refs(entry):
            if ref in in_step or ref in proxy_map:
                continue
            source = source_map.get(ref)
            if source is None:
                continue
            if id(source) not in node_by_source:
                proxy_entries.append(source)
                node_by_source[id(source)] = ""
            proxy_map[ref] = ""
    if not proxy_entries:
        return {}, ()
    base_ids, _legacy = mint_node_ids(proxy_entries)
    for source, base_id in zip(proxy_entries, base_ids, strict=True):
        proxy_id = f"{PROXY_ID_PREFIX}{base_id}"
        node_by_source[id(source)] = proxy_id
        proxy_nodes.append(_proxy_node(source, proxy_id))
    for ref in list(proxy_map):
        source = source_map[ref]
        proxy_map[ref] = node_by_source[id(source)]
    return proxy_map, tuple(proxy_nodes)


def _source_map(all_entries: list[Any]) -> dict[str, Any]:
    """Index episode entries by BOTH ref spellings (bare only when unambiguous)."""

    source_map: dict[str, Any] = {}
    for entry in all_entries:
        source_map[str(getattr(entry, "label", ""))] = entry
        if int(getattr(entry, "num_passes", 1) or 1) <= 1:
            source_map.setdefault(str(getattr(entry, "layer_label", "")), entry)
    return source_map


def _proxy_node(source: Any, proxy_id: str) -> dict[str, Any]:
    """Build one step-boundary proxy node from its source op's geometry."""

    attrs = [{"key": "kind", "value": "step_boundary"}]
    shape_value = shape_string(getattr(source, "shape", None))
    if shape_value is not None:
        attrs.append({"key": "shape", "value": shape_value})
    dtype_value = dtype_string(source)
    if dtype_value is not None:
        attrs.append({"key": "dtype", "value": dtype_value})
    attrs.append({"key": "source_step_op", "value": str(getattr(source, "label", ""))})
    node: dict[str, Any] = {
        "id": proxy_id,
        "label": "step boundary",
        "namespace": "",
        "attrs": attrs,
    }
    outputs = []
    if shape_value is not None:
        outputs.append({"key": "shape", "value": shape_value})
    if dtype_value is not None:
        outputs.append({"key": "dtype", "value": dtype_value})
    if outputs:
        node["outputsMetadata"] = [{"id": "0", "attrs": outputs}]
    return node


def _all_entries(log: Any) -> list[Any]:
    """Return every layer-pass entry of the episode trace."""

    from .._common import _iter_layers

    return _iter_layers(log)


def _entry_parent_refs(entry: Any) -> list[str]:
    """Return one entry's recorded parent refs (positions first)."""

    positions = getattr(entry, "parent_arg_positions", None) or {}
    refs = [str(ref) for ref in (positions.get("args") or {}).values()]
    refs.extend(str(ref) for ref in (positions.get("kwargs") or {}).values())
    if refs:
        return refs
    return [str(ref) for ref in (getattr(entry, "parents", ()) or ())]
