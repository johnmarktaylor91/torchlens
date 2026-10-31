"""Grounding an EDGE read table (edge attribution patching scores) for the causal audit.

``torchlens.mechinterp.edge_scores`` scores every edge ``s->t`` of the component
graph by edge attribution patching: ``(s_corrupt - s_clean) . dM/d(input of
t)``, summed over positions and reduced over examples, with the gradient
taken at the clean input (EAP: Syed et al., "Attribution Patching
Outperforms Automated Circuit Discovery", 2023) or averaged along the path
from the corrupted to the clean input (EAP-IG: Hanna et al., "Have Faith in
Faithfulness", 2024). Both estimate one quantity: what patching the edge
does to the metric, ``M(patched) - M(clean)``, when receiver ``t`` reads
``clean input - s_clean + s_corrupt``.

The audit does the patch. Each ``ok`` row's edge is patched alone with
``torchlens.mechinterp.patch_edges`` on a fork of the audit's trace (the clean run,
with the sender's value from the corrupted run), and the metric the table
was scored with is read from the unembedding output of the fork and reduced
over examples as the table was (mean or sum). The measured effect is
``M(patched) - M(clean)``, the sign of the score. A set of edges is patched
together in one ``patch_edges`` call, never estimated as a sum of single-edge
effects; when its edges enter several receivers, each later receiver reads
the forward as patched so far (the MIB circuit-track convention), and the
recorded edits are the values that fork wrote, one per receiver.

An EDGE table records neither its corrupted inputs nor its metric (its rows
are not re-scorable), so the caller passes the ``corrupted=`` and
``metric=`` it was scored with, keeping ``edge_scores``' meanings; the table's
example count, reduction, method, edge names and sender shapes are checked
against the component graph rebuilt on the audit's trace. Rows whose status
is not ``ok`` carry no score and are reported apart.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from torchlens.attribution._audit_claims import _AUDIT_DOCS_POINTER, _AuditRequest
from torchlens.attribution._audit_sites import _SitesPlan, _SiteUnit
from torchlens.attribution._core import _temporarily_eval
from torchlens.attribution._result import AttributionError

# The reductions ``edge_scores`` records, mapped to the reduction over examples.
_REDUCTIONS = {"sum_positions_mean_examples": "mean", "sum_positions_sum_examples": "sum"}
_METHODS = ("eap", "eap_ig")
_AUDIT_CALL = (
    "tl.attribution.causal_audit(model, x, claim=claim, corrupted=corrupted, metric=metric, k=1)"
)
_RESCORED_CALL = (
    "tl.attribution.causal_audit(model, x, claim=torchlens.mechinterp.edge_scores(model, x, "
    "corrupted=corrupted, metric=metric), corrupted=corrupted, metric=metric, k=1)"
)
#: The cheaper spelling the many-units warning names for an EDGE table.
EDGES_REMEDY = (
    "pass per_unit=False to measure only the sets, or edges= to audit the largest rows only, "
    "e.g. tl.attribution.causal_audit(model, x, claim=claim, corrupted=corrupted, "
    "metric=metric, edges=[row.address[0] for row in sorted((row for row in claim.rows() "
    "if row.status == 'ok'), key=lambda row: -abs(row.score))[:200]])"
)


def _is_edge_table(claim: Any) -> bool:
    """Return whether ``claim`` is a one-backward ``ReadTable`` holding ``EDGE`` rows."""

    from torchlens.attribution.onebackward._table import ReadTable

    return isinstance(claim, ReadTable) and any(row.kind == "EDGE" for row in claim.rows())


def _edge_refusal(why: str, remedy: str, *, code: str) -> AttributionError:
    """Build a coded refusal for an EDGE table the audit cannot measure."""

    return AttributionError(
        f"causal_audit cannot audit this EDGE table: {why}. Remedy: {remedy}. "
        f"{_AUDIT_DOCS_POINTER}",
        code=code,
    )


@dataclass(frozen=True)
class _EdgesPlan(_SitesPlan):
    """What an EDGE-table audit measures: one edge per unit, patched on a fork."""

    graph: Any = None
    corrupt: Any = None
    names: dict[tuple[int, ...], str] = field(default_factory=dict)
    #: Each measured key's recorded edits, read from the fork that measured it.
    _recorded: dict[tuple[tuple[int, ...], ...], tuple[tuple[Any, Tensor], ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def effect(self, original: float, scored: float) -> float:
        """Return ``M(patched) - M(clean)``, the quantity an edge score estimates."""

        return scored - original

    def _edges(self, keys: tuple[tuple[int, ...], ...]) -> list[Any]:
        """Return the component-graph edges of ``keys``."""

        return [self.graph.edges[self.names[key]] for key in keys]

    def _patched(self, keys: tuple[tuple[int, ...], ...]) -> Any:
        """Return a fork of the trace with the edges of ``keys`` patched."""

        from torchlens.mechinterp import patch_edges

        with torch.no_grad():
            return patch_edges(self.trace, self._edges(keys), corrupted_graph=self.corrupt)

    def ablated_score(self, keys: tuple[tuple[int, ...], ...]) -> float:
        """Score a fork of the trace with the edges of ``keys`` patched together.

        The recorded edits are read from this fork and kept, so
        ``intervention`` does not patch the edges again; each is one receiver
        input per receiver, small beside the saved trace the audit holds.
        """

        fork = self._patched(keys)
        try:
            scored = self.score(fork)
            self._recorded[keys] = self._pairs(fork, keys)
            return scored
        finally:
            fork.cleanup()

    def unit_scores(self) -> dict[tuple[int, ...], float]:
        """Score every edge patched alone, one ``patch_edges`` fork each.

        ``site_sweep`` takes one edit for every candidate; each edge's patch
        writes its own value, so the edges are not swept.
        """

        return {key: self.ablated_score((key,)) for key in self.units}

    def intervention(self, keys: tuple[tuple[int, ...], ...]) -> tuple[tuple[Any, Any], ...]:
        """Return the ``(selection, value)`` pairs ``patch_edges`` wrote for ``keys``.

        One pair per receiver, in the order ``patch_edges`` writes them
        (topological stages), read from the patched fork itself (for a measured
        row or set, the fork that measured it), so a later receiver's value is
        the one computed from the partly patched forward.
        For an ``m{L}`` or ``logits`` receiver the selection is the EDGE
        selection of its pre-norm input edge and the value is what the fork
        substituted on that edge (``input - sum(s) + sum(s_corrupt)``); for a
        q, k or v receiver it is head ``H``'s role slice of the projection
        output and the value is the fork's projection output (only the slice
        is written). An EDGE selection addresses its edge by call id, so the
        pairs replay on forks of the audit's own trace (``extra["trace"]``):
        ``fork = audit.extra["trace"].fork()``, then ``fork.do(*pair)`` for
        each pair in order.
        """

        if keys in self._recorded:
            return tuple((selection, value.clone()) for selection, value in self._recorded[keys])
        fork = self._patched(keys)
        try:
            return self._pairs(fork, keys)
        finally:
            fork.cleanup()

    def _pairs(
        self, fork: Any, keys: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[Any, Tensor], ...]:
        """Return one ``(selection, value)`` pair per receiver of ``keys``, as ``fork`` wrote it."""

        from torchlens.mechinterp._edge_patch import _stage_key, receiver_selection

        receivers = {edge.receiver.name: edge.receiver for edge in self._edges(keys)}
        ordered = sorted(receivers.values(), key=_stage_key)
        return tuple(
            (receiver_selection(self.trace, receiver), _written(fork, receiver))
            for receiver in ordered
        )


def _written(fork: Any, receiver: Any) -> Tensor:
    """Return the value a ``patch_edges`` fork wrote for ``receiver``."""

    from torchlens.mechinterp._edge_patch import _head_spec, _input_edge_record

    if receiver.kind in ("q", "k", "v"):
        return fork.ops[str(_head_spec(fork, receiver).home_label)].out.detach().clone()
    record = _input_edge_record(fork, receiver)
    store = getattr(fork.ops[str(record.child_label)], "edge_substitutions", None) or {}
    (value,) = [
        entry["value"] for entry in store.values() if entry["parent_label"] == record.parent_label
    ]
    return value.detach().clone()


def _edge_trace(request: _AuditRequest, inputs: Any) -> Any:
    """Trace the model on ``inputs`` with every op saved, intervention-ready, in eval mode."""

    import torchlens as tl

    positional = [inputs] if isinstance(inputs, Tensor) else list(inputs)
    with _temporarily_eval(request.model), torch.no_grad():
        return tl.trace(
            request.model,
            positional,
            input_kwargs=request.input_kwargs or None,
            capture=tl.options.CaptureOptions(layers_to_save="all", intervention_ready=True),
        )


def _mixed_remedy(claim: Any, rows: tuple[Any, ...]) -> str:
    """Return the audit of a mixed table's site reads (seeded), else of its edges rescored."""

    from torchlens.attribution._audit_reads import _read_remedy, _recorded_target, _seed_spelling

    act = next(row for row in rows if row.kind != "EDGE")
    seed = _seed_spelling(_recorded_target(claim, act.target_id))
    return _RESCORED_CALL if seed == "target" else _read_remedy(seed)


def _check_request(request: _AuditRequest, rows: tuple[Any, ...]) -> str:
    """Refuse what an EDGE audit cannot take; return the reduction over examples."""

    if any(row.kind != "EDGE" for row in rows):
        raise _edge_refusal(
            "it mixes EDGE rows with ACT rows; an edge is tested by patching it and a site by "
            "ablating it, so audit each table on its own",
            _mixed_remedy(request.claim, rows),
            code="attribution_audit_claim_unsupported",
        )
    if request.corrupted is None or request.metric is None:
        missing = [
            name
            for name, value in (("corrupted", request.corrupted), ("metric", request.metric))
            if value is None
        ]
        raise _edge_refusal(
            f"{' and '.join(f'{name}=' for name in missing)} not given; an EDGE table records "
            "neither the corrupted inputs nor the metric it was scored with (its rows cannot be "
            "re-scored), so pass the ones torchlens.mechinterp.edge_scores was given",
            _AUDIT_CALL,
            code="attribution_audit_claim_misaligned",
        )
    if request.target is not None:
        raise _edge_refusal(
            "target= was given; an EDGE table's target is its metric, so leave target= out",
            _AUDIT_CALL,
            code="attribution_audit_claim_misaligned",
        )
    if not (isinstance(request.ablation, str) and request.ablation == "zero"):
        raise _edge_refusal(
            f"ablation={request.ablation!r} was given; an edge is tested by patching in the "
            "sender's corrupted value, so ablation= does not apply",
            _AUDIT_CALL,
            code="attribution_eval_setting_invalid",
        )
    ok = [row for row in rows if row.status == "ok"]
    methods = sorted({str(row.method) for row in ok})
    reductions = sorted({str(row.reduction) for row in ok})
    if (
        any(method not in _METHODS for method in methods)
        or len(reductions) > 1
        or any(reduction not in _REDUCTIONS for reduction in reductions)
    ):
        raise _edge_refusal(
            f"its rows are {methods!r} scores reduced by {reductions!r}; the audit compares "
            f"patches with one torchlens.mechinterp.edge_scores table (methods {_METHODS!r})",
            _RESCORED_CALL,
            code="attribution_audit_claim_unsupported",
        )
    return _REDUCTIONS[reductions[0]] if reductions else "mean"


def _graphs(request: _AuditRequest) -> tuple[Any, Any, Any]:
    """Return the audit's trace, its component graph and the corrupted run's graph."""

    from torchlens.mechinterp import component_graph
    from torchlens.mechinterp._errors import MechInterpError

    trace = _edge_trace(request, request.inputs)
    try:
        graph = component_graph(trace)
        corrupt = component_graph(_edge_trace(request, request.corrupted))
    except MechInterpError as exc:
        raise _edge_refusal(
            "the component graph of this model cannot be built on these inputs, so the table "
            f"was not scored on them ({exc.fields.get('code')})",
            _RESCORED_CALL,
            code="attribution_audit_claim_misaligned",
        ) from exc
    for name, node in graph.nodes.items():
        other = corrupt.nodes.get(name)
        if other is None or tuple(other.value.shape) != tuple(node.value.shape):
            found = None if other is None else tuple(other.value.shape)
            raise _edge_refusal(
                f"node {name!r} is {tuple(node.value.shape)} on the clean inputs and {found} on "
                "corrupted=; edge scores compare one node on two inputs of one shape",
                "pass the corrupted= torchlens.mechinterp.edge_scores was given, the same shape as x, "
                f"e.g. {_AUDIT_CALL}",
                code="attribution_audit_claim_misaligned",
            )
    return trace, graph, corrupt


def _metric_scorer(
    trace: Any, metric: Callable[[Tensor], Tensor], reduction: str, n_examples: int
) -> tuple[str, Callable[[Any], float]]:
    """Return the logits op label and ``run -> metric reduced over examples``."""

    from torchlens.mechinterp._anchors import resolve_lm_head

    head = resolve_lm_head(trace).head
    label = str(trace.ops[str(list(head.output_ops)[-1])].label)
    values = metric(trace.ops[label].out.detach())
    shape = tuple(values.shape) if isinstance(values, Tensor) else None
    if (
        shape is None
        or not values.is_floating_point()
        or len(shape) > 1
        or (len(shape) == 1 and shape[0] != n_examples)
    ):
        raise _edge_refusal(
            f"metric returned {type(values).__name__}{'' if shape is None else shape} on the "
            f"logits; torchlens.mechinterp.edge_scores takes a metric with one real value per example "
            f"(shape ({n_examples},)) or one scalar, so this is not the metric the table was "
            "scored with",
            "pass the metric torchlens.mechinterp.edge_scores was given, e.g. metric=lambda logits: "
            f"logits[:, -1, 0] - logits[:, -1, 1], as in {_AUDIT_CALL}",
            code="attribution_audit_claim_misaligned",
        )
    divisor = n_examples if reduction == "mean" else 1

    def _score(run: Any) -> float:
        """The metric on a trace or fork, summed over examples (divided for a mean)."""

        out = metric(run.ops[label].out.detach())
        return math.fsum(out.detach().to(torch.float64).flatten().tolist()) / divisor

    return label, _score


def _write_site(trace: Any, receiver: Any) -> str:
    """Return the op label a patch into ``receiver`` writes."""

    if receiver.kind in ("q", "k", "v"):
        from torchlens.mechinterp._edge_patch import _head_spec

        return str(_head_spec(trace, receiver).home_label)
    return str(receiver.input_label)


def _chosen(request: _AuditRequest, ok: dict[str, int]) -> set[str] | None:
    """Return the edge names ``edges=`` keeps, or ``None`` for every ``ok`` row."""

    if request.edges is None:
        return None
    if isinstance(request.edges, str):
        names = [request.edges]
    else:
        names = [str(getattr(item, "name", item)) for item in request.edges]
    unknown = [name for name in names if name not in ok]
    if unknown or not names:
        raise _edge_refusal(
            f"edges= names {unknown[:4]!r}, which are not ok rows of the table"
            if unknown
            else "edges= is empty",
            "pass names of the table's ok rows, e.g. tl.attribution.causal_audit(model, x, "
            "claim=claim, corrupted=corrupted, metric=metric, k=1, edges=[row.address[0] for "
            "row in claim.rows() if row.status == 'ok'][:2])",
            code="attribution_eval_setting_invalid",
        )
    return set(names)


def _edge_units(
    request: _AuditRequest, rows: tuple[Any, ...], trace: Any, graph: Any
) -> tuple[list[_SiteUnit], dict[tuple[int, ...], str], list[tuple[str, str]]]:
    """Return the audited units, their edge names, and the rows reported apart."""

    apart: list[tuple[str, str]] = []
    ok: dict[str, int] = {}
    for index, row in enumerate(rows):
        name = str(row.address[0])
        if row.status != "ok":
            apart.append((name, str(row.status)))
            continue
        edge = graph.edges.get(name)
        sender = None if edge is None else edge.sender.value
        if sender is None or (row.shape is not None and tuple(row.shape) != tuple(sender.shape)):
            found = "no such edge" if sender is None else f"sender shape {tuple(sender.shape)}"
            raise _edge_refusal(
                f"row {name!r} (sender shape {row.shape}) does not resolve on the component "
                f"graph of these inputs ({found}), so the table was scored on another model or "
                "other inputs",
                _RESCORED_CALL,
                code="attribution_audit_claim_misaligned",
            )
        ok[name] = index
    chosen = _chosen(request, ok)
    units: list[_SiteUnit] = []
    names: dict[tuple[int, ...], str] = {}
    for name, index in ok.items():
        if chosen is not None and name not in chosen:
            continue
        receiver = graph.edges[name].receiver
        units.append(
            _SiteUnit((index,), _write_site(trace, receiver), (), name, float(rows[index].score))
        )
        names[(index,)] = name
    return units, names, apart


def _plan_edge_audit(request: _AuditRequest) -> _EdgesPlan:
    """Ground an EDGE table on the audit's own trace and the corrupted run's graph."""

    claim = request.claim
    rows = claim.rows()
    reduction = _check_request(request, rows)
    trace, graph, corrupt = _graphs(request)
    n_examples = int(graph.nodes["input"].value.shape[0])
    if claim.provenance.sample_count != n_examples:
        raise _edge_refusal(
            f"it was scored on {claim.provenance.sample_count} examples and these inputs hold "
            f"{n_examples}",
            _RESCORED_CALL,
            code="attribution_audit_claim_misaligned",
        )
    units, names, apart = _edge_units(request, rows, trace, graph)
    logits_site, score = _metric_scorer(trace, request.metric, reduction, n_examples)
    methods = sorted({str(rows[key[0]].method) for key in names})
    return _EdgesPlan(
        trace=trace,
        units=tuple(unit.key for unit in units),
        rows={unit.key: unit for unit in units},
        shapes={},
        ablation_name="edge_patch",
        edits={},
        score=score,
        claim_kind="edge_table",
        claim_method=methods[0] if len(methods) == 1 else ",".join(methods),
        target_repr=str(claim.provenance.target_reprs[0] if claim.provenance.target_reprs else ""),
        extra={
            "example_reduction": reduction,
            "logits_site": logits_site,
            "rows_not_audited": tuple(apart),
            "trace": trace,
        },
        graph=graph,
        corrupt=corrupt,
        names=names,
    )
