"""The four cards: Trace, pass-aware Layer, Op, failure-first PartialTrace.

Treescope memo decision 3 (B2, lane F16), one grammar per card: collapsed
identity + honesty badge in the always-visible header; identity / value /
context zones expanded; disclosure footer; and the PROOF row on every
card -- capture outcome, replay-verification badge, nonfinite coverage
basis, save policy -- the category treescope structurally cannot enter
(our card asserts "this is a verified execution", not "this is an
object").

Hard card rules implemented here (each earned by a measurement):

- R-NEVER-RAISE: every entry point renders behind
  :func:`torchlens.notebook.cardtree.safe_card_html`.
- Honesty states (partial, halted, poisoned, unknown coverage, unsaved,
  truncated) are NEVER folded.
- Per-pass fields read via ``layer.ops[k]``, never off the Layer (per-pass
  reads RAISE on reused layers, so pass rows are the card's SHAPE).
- Display paths never materialize disk payloads (a non-resident value
  degrades to a one-line reason), never do unbounded device transfers
  (grids ride the ported truncation budgets), never recompute stats
  (the C02 kernel memoizes), and pass Flops/Duration/Bytes wrappers
  through UNTOUCHED via ``str()`` -- renderer unit arithmetic measurably
  produced plausible wrong numbers.
- Escape once at the typed leaf boundary (CardTree owns HTML safety).

Every spelling is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from typing import Any

from ._access import CopyRoot, resolve_copy_root
from ._axis_labels import sdpa_attention_hint
from ._grid import GridBudgets, array_grid_html
from .cardtree import (
    Card,
    CardCollection,
    CardHtml,
    CardKey,
    CardNode,
    CardSection,
    CardText,
    safe_card_html,
)

__all__ = [
    "BRIDGE_SENTINEL_HTML",
    "layer_card",
    "layer_repr_html",
    "op_card",
    "op_repr_html",
    "partial_trace_card",
    "partial_repr_html",
    "trace_card",
    "trace_repr_html",
]

#: One-line sentinel returned instead of the full card when the treescope
#: bridge just rendered this object (memo 3.6; wording is [UI-SPRINT]).
BRIDGE_SENTINEL_HTML = (
    '<div class="tl-card-sentinel">card omitted: rendered by the treescope bridge above</div>'
)

#: Value-grid budget inside a card (smaller than the standalone default so
#: an Op card stays comfortably in the KB band).
_CARD_GRID_CELL_BUDGET = 1_000


def _bridge_sentinel(obj: Any) -> str | None:
    """Return the suppression sentinel when the bridge claims this render.

    Fails SAFE: any error (bridge absent, treescope absent, predicate
    fault) returns ``None`` and the full card renders -- a broken sniff
    may never suppress the user's only good rendering.
    """

    try:
        from ..bridge.treescope import consume_repr_suppression

        if consume_repr_suppression(obj):
            return BRIDGE_SENTINEL_HTML
    except Exception:  # noqa: BLE001 - fail-safe is the contract (memo 3.6)
        return None
    return None


def _stats_line(tensor: Any) -> str | None:
    """Render the lovely core stats line for one resident tensor payload.

    The stats package resolves DYNAMICALLY: cards are reachable from
    ``Trace._repr_html_`` and the stats package's aggregate module reads
    the root facade, so a static edge here closes an import cycle that
    breaks type analysis of ``torchlens.stats._aggregate``.
    """

    try:
        import importlib

        stats = importlib.import_module("torchlens.stats")
        return str(stats.render_core_line(stats.tensor_stats(tensor)))
    except Exception:  # noqa: BLE001 - stats are cosmetic on a card; degrade
        return None


def _resident_tensor(owner: Any, attribute: str) -> Any | None:
    """Read a tensor attribute without materializing non-resident payloads.

    Returns the tensor only when the read succeeds and yields a real,
    non-meta ``torch.Tensor``; anything else (disk-deferred payload, typed
    refusal, exotic subclass) degrades to ``None`` -- display paths never
    silently reload from disk (composition row X6).
    """

    try:
        import torch

        value = getattr(owner, attribute, None)
        if type(value) is torch.Tensor and not value.is_meta:
            return value
    except Exception:  # noqa: BLE001 - a display read may never raise
        return None
    return None


def _proof_section(trace_like: Any) -> CardSection:
    """Build the PROOF row: outcome, verification, coverage, save policy.

    Rendered OPEN (honesty is never folded).
    """

    facts: list[CardNode] = []
    outcome = getattr(trace_like, "outcome", None)
    status = getattr(getattr(outcome, "status", None), "name", None)
    facts.append(CardText(f"capture outcome: {status or 'UNKNOWN'}"))
    verified = getattr(trace_like, "capture_verified", None)
    if verified is False:
        reason = getattr(trace_like, "capture_verification_reason", None) or "unverified"
        facts.append(CardText(f"capture NOT verified: {reason}", role="notice"))
    elif verified is True:
        facts.append(CardText("capture verified"))
    coverage = getattr(trace_like, "nonfinite_coverage", None)
    if coverage is not None:
        facts.append(CardText(f"nonfinite coverage: {coverage}"))
    total, saved = _save_counts(trace_like)
    if total is not None and saved is not None:
        facts.append(CardText(f"save policy: {saved}/{total} op records saved"))
    return CardSection(title="capture proof", children=tuple(facts), folded=False)


def _save_counts(trace_like: Any) -> tuple[int | None, int | None]:
    """(total op records, saved op records) on ONE consistent basis.

    ``num_ops`` counts only real computation ops while ``num_saved_ops``
    counts every saved record including input/output boundaries -- mixing
    them printed ``saved 4/2``. The record count is the honest denominator.
    """

    try:
        total: int | None = len(tuple(trace_like.op_labels or ()))
    except Exception:  # noqa: BLE001 - unfinished/legacy logs may lack labels
        total = None
    saved = getattr(trace_like, "num_saved_ops", None)
    return total, saved


def _trace_of(record: Any) -> Any | None:
    """Owning trace of a Layer/Op record, when reachable."""

    try:
        return record.source_trace
    except Exception:  # noqa: BLE001 - detached records have no trace
        return None


def _root_for(record: Any, explicit_root: str | None) -> CopyRoot:
    """Resolve the copy root against the record's OWNING TRACE identity.

    A detached record (no trace backref) disables copy-access rather than
    scanning for the record itself -- ``record['key']`` is not a valid
    access expression, and a wrong expression is worse than none.
    """

    trace = _trace_of(record)
    if trace is None and explicit_root is None:
        return CopyRoot(None, "disabled", "detached record: no owning trace to index")
    return resolve_copy_root(trace, explicit_root=explicit_root)


def _identity_zone(op: Any, root: CopyRoot) -> tuple[CardNode, ...]:
    """Op identity: label, pass, function, module, site key, access key."""

    label = str(getattr(op, "label", "?"))
    pass_index = getattr(op, "pass_index", None)
    num_passes = getattr(op, "num_passes", None)
    identity = f"{getattr(op, 'func_name', '?')}"
    if pass_index is not None and num_passes is not None and num_passes > 1:
        identity += f" (pass {pass_index}/{num_passes})"
    module = getattr(op, "atomic_module_address", None) or getattr(op, "module", None)
    nodes: list[CardNode] = [CardText(identity)]
    if module:
        nodes.append(CardText(f"module: {module}"))
    site_key = getattr(op, "site_key", None)
    if site_key:
        nodes.append(CardText(f"site: {site_key}", role="muted"))
    nodes.append(CardKey(root.key_expression(label)))
    if root.expression is None and root.reason:
        nodes.append(CardText(f"copy-access disabled: {root.reason}", role="muted"))
    return tuple(nodes)


def _value_zone(op: Any) -> tuple[CardNode, ...]:
    """Op value zone: stats line + budgeted grid, or the honest reason."""

    if not getattr(op, "has_saved_activation", False):
        return (CardText("value not saved (capture save= policy)", role="muted"),)
    tensor = _resident_tensor(op, "out")
    if tensor is None:
        return (CardText("value not resident (disk/offloaded payload)", role="muted"),)
    nodes: list[CardNode] = []
    line = _stats_line(tensor)
    if line:
        nodes.append(CardText(line))
    try:
        grid = array_grid_html(tensor, budgets=GridBudgets(cells=_CARD_GRID_CELL_BUDGET))
        nodes.append(CardHtml(grid.html))
    except Exception:  # noqa: BLE001 - the grid is optional; the line is not
        nodes.append(CardText("array view unavailable", role="muted"))
    return tuple(nodes)


def _grad_zone(op: Any) -> tuple[CardNode, ...]:
    """Grad stats appear ONLY when a gradient was captured (row X7)."""

    if not getattr(op, "has_saved_gradient", False):
        return ()
    tensor = _resident_tensor(op, "grad")
    if tensor is None:
        return ()
    line = _stats_line(tensor)
    return (CardText(f"grad: {line}"),) if line else ()


def _context_zone(op: Any, root: CopyRoot) -> tuple[CardNode, ...]:
    """Graph neighbors as copyable keys plus cost wrappers UNTOUCHED."""

    nodes: list[CardNode] = []
    parents = tuple(getattr(op, "parents", ()) or ())
    children = tuple(getattr(op, "children", ()) or ())
    if parents:
        nodes.append(CardText("parents:", role="muted"))
        nodes.extend(CardKey(root.key_expression(str(p))) for p in parents[:8])
    if children:
        nodes.append(CardText("children:", role="muted"))
        nodes.extend(CardKey(root.key_expression(str(c))) for c in children[:8])
    costs: list[str] = []
    for attribute, prefix in (
        ("num_params", "params"),
        ("flops_forward", "flops fwd"),
        ("func_duration", "time"),
        ("activation_memory", "memory"),
    ):
        value = getattr(op, attribute, None)
        if value is not None:
            costs.append(f"{prefix}: {value}")
    if costs:
        nodes.append(CardText(" | ".join(costs)))
    return tuple(nodes)


def _intervention_badges(op: Any) -> tuple[CardNode, ...]:
    """Intervention disclosures (row X2): shown whenever evidence exists."""

    nodes: list[CardNode] = []
    if getattr(op, "intervention_replaced", False):
        nodes.append(CardText("intervened: value replaced", role="notice"))
    if getattr(op, "edge_substitutions", None):
        nodes.append(CardText("intervened: edge substitution", role="notice"))
    return tuple(nodes)


def op_card(op: Any, *, root: str | None = None, kind: str = "op") -> Card:
    """Assemble the Op card (memo section 5)."""

    resolved_root = _root_for(op, root)
    trace = _trace_of(op)
    children: tuple[CardNode, ...] = (
        *_identity_zone(op, resolved_root),
        *_intervention_badges(op),
        *_value_zone(op),
        *_grad_zone(op),
        *_context_zone(op, resolved_root),
    )
    if trace is not None:
        children += (_proof_section(trace),)
    else:
        children += (CardText("capture facts unavailable (detached record)", role="muted"),)
    shape = getattr(op, "shape", None)
    dtype = getattr(op, "dtype", None)
    title = f"TorchLens Op: {getattr(op, 'label', '?')}"
    badge = f"{tuple(shape)} {dtype}" if shape is not None else None
    return Card(title=title, badge=badge, kind=kind, children=children)


def layer_card(layer: Any, *, root: str | None = None) -> Card:
    """Assemble the pass-aware Layer card.

    Pass rows are the card's SHAPE: per-pass facts read via ``ops[k]``
    only. A single-pass layer degenerates to the Op card (ONE code path);
    pooling across passes is forbidden -- it describes no executed value.
    """

    num_passes = int(getattr(layer, "num_passes", 1) or 1)
    if num_passes <= 1:
        # ``OpAccessor`` indexes by 0-based POSITION (its repr teaches the
        # basis); the single pass is position 0.
        return op_card(layer.ops[0], root=root, kind="layer")
    resolved_root = _root_for(layer, root)
    rows: list[CardNode] = []
    for pass_index in range(1, num_passes + 1):
        rows.append(_pass_row(layer, pass_index, num_passes, resolved_root))
    label = str(getattr(layer, "layer_label", "?"))
    return Card(
        title=f"TorchLens Layer: {label}",
        badge=f"{num_passes} passes",
        kind="layer",
        children=tuple(rows) + _layer_proof(layer),
    )


def _pass_row(layer: Any, pass_index: int, num_passes: int, root: CopyRoot) -> CardSection:
    """One per-pass row, reading exclusively through ``layer.ops[k]``."""

    try:
        op = layer.ops[pass_index - 1]
    except Exception:  # noqa: BLE001 - a missing pass renders as data, not a crash
        return CardSection(
            title=f"pass {pass_index}/{num_passes}: unavailable",
            children=(CardText("pass record unavailable", role="notice"),),
            folded=False,
        )
    label = str(getattr(op, "label", "?"))
    facts: list[CardNode] = [CardKey(root.key_expression(label))]
    tensor = _resident_tensor(op, "out") if getattr(op, "has_saved_activation", False) else None
    line = _stats_line(tensor) if tensor is not None else None
    if line:
        facts.append(CardText(line))
    else:
        shape = getattr(op, "shape", None)
        dtype = getattr(op, "dtype", None)
        facts.append(CardText(f"shape {tuple(shape) if shape else '?'} {dtype or ''}"))
    return CardSection(
        title=f"pass {pass_index}/{num_passes}: {label}",
        children=tuple(facts),
        folded=False,
    )


def _layer_proof(layer: Any) -> tuple[CardNode, ...]:
    """PROOF row for a Layer card, read from the owning trace."""

    trace = _trace_of(layer)
    if trace is None:
        return (CardText("capture facts unavailable (detached record)", role="muted"),)
    return (_proof_section(trace),)


def trace_card(trace: Any, *, root: str | None = None) -> Card:
    """Assemble the Trace overview card (replaces the 525-byte box)."""

    resolved_root = resolve_copy_root(trace, explicit_root=root)
    layers = list(getattr(trace, "layer_logs", {}) or {})
    total, saved = _save_counts(trace)
    facts: list[CardNode] = [
        CardText(
            f"Layers: {len(layers)} | Ops: {getattr(trace, 'num_ops', '?')} | "
            f"backend: {getattr(trace, 'backend', '?')} | "
            f"modules: {getattr(trace, 'num_modules', '?')}"
        ),
        CardText(
            f"params: {getattr(trace, 'num_params', '?')} | "
            f"flops fwd: {getattr(trace, 'total_flops_forward', '?')}"
        ),
        CardText(f"saved: {saved if saved is not None else '?'}/{total or '?'} op records"),
    ]
    nonfinite = _first_nonfinite_text(trace)
    if nonfinite:
        facts.append(CardText(f"NaN/Inf: {nonfinite}"))
    hint = sdpa_attention_hint(trace)
    if hint:
        facts.append(CardText(hint, role="notice"))
    if resolved_root.source == "namespace":
        facts.append(CardText(f"as: {resolved_root.expression}", role="muted"))
    key_index = CardCollection(
        children=tuple(CardKey(resolved_root.key_expression(str(k))) for k in layers[:64]),
        budget=20,
    )
    modules = _module_index(trace)
    children: tuple[CardNode, ...] = (
        *facts,
        CardSection(title="lookup keys", children=(key_index,), folded=True),
        *modules,
        _proof_section(trace),
    )
    title = str(getattr(trace, "trace_label", None) or getattr(trace, "model_label", "Trace"))
    status = getattr(getattr(trace, "outcome", None), "status", None)
    return Card(
        title=f"TorchLens Trace: {title}",
        badge=getattr(status, "name", None),
        kind="trace",
        children=children,
    )


def _first_nonfinite_text(trace: Any) -> str | None:
    """Text-format first-nonfinite disclosure, degrade-safe."""

    try:
        return str(trace.first_nonfinite(link_format="text"))
    except Exception:  # noqa: BLE001 - the card renders on unfinished logs too
        return None


def _module_index(trace: Any) -> tuple[CardNode, ...]:
    """Budgeted mini module table (top collapse-order rows)."""

    try:
        order = list(getattr(trace, "module_collapse_order", ()) or ())[:10]
    except Exception:  # noqa: BLE001 - collapse metadata is optional
        order = []
    if not order:
        return ()
    rows = tuple(CardText(str(address), role="muted") for address in order)
    return (CardSection(title="modules", children=(CardCollection(rows, budget=10),)),)


def partial_trace_card(partial: Any) -> Card:
    """Assemble the failure-first PartialTrace card.

    The card treescope structurally cannot have: it renders an execution
    that did not finish -- persistent failure banner, phase/reason,
    escaped exception, committed prefix and frontier, known/unknown
    evidence, and what remains safe to inspect.
    """

    outcome = getattr(partial, "outcome", None)
    phase = getattr(getattr(outcome, "phase", None), "name", None)
    facts: list[CardNode] = []
    if phase:
        facts.append(CardText(f"failed during: {phase}", role="notice"))
    error = getattr(partial, "original_exception", None)
    if error is not None:
        facts.append(CardText(f"error: {type(error).__name__}: {error}", role="notice"))
    raw_layers = list(getattr(partial, "raw_layers", ()) or ())
    facts.append(CardText(f"committed prefix: {len(raw_layers)} raw layers"))
    frontier = [str(getattr(r, "layer_label_raw", None) or r) for r in raw_layers[-3:]]
    if frontier:
        facts.append(CardText(f"frontier (last committed): {', '.join(frontier)}"))
    facts.append(CardText(_partial_nonfinite(partial)))
    facts.append(
        CardText(
            "safe to inspect: raw_layers, audit(), draw(); "
            "replay/validation refuse on failed captures",
            role="muted",
        )
    )
    facts.append(_proof_section(partial))
    return Card(
        title="TorchLens PartialTrace",
        badge="FAILED CAPTURE",
        kind="partial",
        children=tuple(facts),
    )


def _partial_nonfinite(partial: Any) -> str:
    """Nonfinite evidence line for a partial capture, degrade-safe."""

    try:
        return f"NaN/Inf: {partial.first_nonfinite()}"
    except Exception:  # noqa: BLE001 - evidence may be unavailable mid-failure
        return "NaN/Inf: unknown (capture did not finish)"


def trace_repr_html(trace: Any) -> str:
    """``Trace._repr_html_`` body: sniff-aware, never-raise."""

    sentinel = _bridge_sentinel(trace)
    if sentinel is not None:
        return sentinel
    return safe_card_html(lambda: trace_card(trace))


def layer_repr_html(layer: Any) -> str:
    """``Layer._repr_html_`` body: sniff-aware, never-raise."""

    sentinel = _bridge_sentinel(layer)
    if sentinel is not None:
        return sentinel
    return safe_card_html(lambda: layer_card(layer))


def op_repr_html(op: Any) -> str:
    """``Op._repr_html_`` body: sniff-aware, never-raise."""

    sentinel = _bridge_sentinel(op)
    if sentinel is not None:
        return sentinel
    return safe_card_html(lambda: op_card(op))


def partial_repr_html(partial: Any) -> str:
    """``PartialTrace._repr_html_`` body: sniff-aware, never-raise."""

    sentinel = _bridge_sentinel(partial)
    if sentinel is not None:
        return sentinel
    return safe_card_html(lambda: partial_trace_card(partial))
