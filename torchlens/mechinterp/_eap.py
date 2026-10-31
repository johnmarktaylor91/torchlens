"""Edge attribution patching (EAP and EAP-IG) over the component edge graph.

The score of edge ``s->t`` is the first-order estimate of patching it:
``(s_corrupt - s_clean) . dM/d(input of t)``, summed over positions and
averaged over examples (the MIB circuit track and EAP-IG convention). EAP
takes the gradient once, at the clean input. EAP-IG averages it over ``m``
points on the straight line between the corrupted and the clean ``input``
node activations, ``e(alpha) = e_corrupt + alpha (e_clean - e_corrupt)`` at
``alpha = k/m`` for ``k = 0..m-1`` (a left Riemann sum).

Every edge is scored from one backward per evaluation point. The gradient
of a receiver is the gradient through that receiver alone, never the total
gradient at the shared residual state: the one backward stops at each
receiver's own read (the output of an MLP or logits receiver's pre-norm,
and head ``H``'s q, k or v slice of the projection output for an attention
receiver), and a local pull-back through that read (the norm, plus the
projection for q, k and v) carries it to the receiver's pre-norm input.
The local pull-backs reuse the captured autograd graph; they never re-run
the model.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from ._edge_patch import _head_spec
from ._edges import ComponentGraph, ComponentReceiver, component_graph
from ._errors import refuse

__all__ = ["edge_scores"]

#: The scoring methods, in vocabulary order.
METHODS: tuple[str, ...] = ("eap", "eap_ig")

#: EAP-IG's default number of evaluation points (the MIB reference default).
DEFAULT_STEPS = 5

#: Accepted values of ``example_reduction=``.
EXAMPLE_REDUCTIONS: tuple[str, ...] = ("mean", "sum")


@dataclass(frozen=True)
class _Read:
    """Where one receiver reads the residual stream, in one trace."""

    seed_label: str
    input_label: str
    spec: Any


def _validate(method: str, steps: int | None, example_reduction: str) -> int:
    """Check the how-arguments and return the number of evaluation points."""

    if method not in METHODS:
        refuse(
            code="mi_edge_scores_setting_invalid",
            message=f"edge_scores got method={method!r}; the methods are {METHODS}.",
            remedy="pass method='eap_ig' (the default) or method='eap'",
            argument="method",
            value=method,
        )
    if example_reduction not in EXAMPLE_REDUCTIONS:
        refuse(
            code="mi_edge_scores_setting_invalid",
            message=f"edge_scores got example_reduction={example_reduction!r}; the choices "
            f"are {EXAMPLE_REDUCTIONS}.",
            remedy="pass example_reduction='mean' (the default) or example_reduction='sum'",
            argument="example_reduction",
            value=example_reduction,
        )
    if method == "eap":
        if steps is not None:
            refuse(
                code="mi_edge_scores_setting_invalid",
                message=f"edge_scores got steps={steps!r} with method='eap'; EAP takes the "
                "gradient once, at the clean input, so it has no integration steps.",
                remedy="drop steps=, or pass method='eap_ig' with steps=",
                argument="steps",
                value=steps,
            )
        return 1
    if steps is None:
        return DEFAULT_STEPS
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        refuse(
            code="mi_edge_scores_setting_invalid",
            message=f"edge_scores got steps={steps!r}; steps is the number of EAP-IG "
            "evaluation points, a positive integer.",
            remedy="pass steps=5 (the default) or another positive integer",
            argument="steps",
            value=steps,
        )
    return int(steps)


def _capture(model: Any, inputs: Any, *, intervene: Any = None) -> Any:
    """Capture ``model`` on ``inputs`` with every op saved and autograd on."""

    from ..user_funcs import trace as capture_trace

    with torch.enable_grad():
        return capture_trace(model, inputs, intervene=intervene)


def _input_write_module(trace: Any, graph: ComponentGraph) -> str:
    """Return the module whose output is the first embedding write (EAP-IG's shift site)."""

    writes = graph._input_writes
    if writes:
        label, address = writes[0]
        outputs = {
            str(trace.ops[str(out)].label)
            for out in getattr(trace.modules[address], "output_ops", ()) or ()
        }
        if str(trace.ops[label].label) in outputs:
            return address
    refuse(
        code="mi_edge_scores_input_unresolvable",
        message="The first embedding write into the residual stream is not the output of "
        "its embedding module, so EAP-IG has no capture-time site to shift the input node.",
        remedy="use method='eap', or a model whose embedding modules write the residual "
        "stream directly",
    )
    raise AssertionError("unreachable")


def _node_deltas(clean: ComponentGraph, corrupt: ComponentGraph) -> dict[str, torch.Tensor]:
    """Return ``corrupt - clean`` for every source node, checked for matching geometry."""

    deltas: dict[str, torch.Tensor] = {}
    for name, node in clean.nodes.items():
        other = corrupt.nodes.get(name)
        if other is None or tuple(other.value.shape) != tuple(node.value.shape):
            refuse(
                code="mi_edge_scores_inputs_mismatched",
                message=f"Node {name!r} is "
                f"{'missing' if other is None else tuple(other.value.shape)} on the corrupted "
                f"input and {tuple(node.value.shape)} on the clean input; edge scores compare "
                "the same node on two inputs of one shape.",
                remedy="pass corrupted inputs with the same batch size and the same number of "
                "tokens per row as the clean inputs",
                node=name,
            )
        deltas[name] = (other.value.detach() - node.value.detach()).to(node.value.dtype)
    return deltas


def _refuse_shared_norms(graph: ComponentGraph) -> None:
    """Refuse when one norm feeds two receiver modules outside the attention projection."""

    owners: dict[str, set[str]] = {}
    for receiver in graph.receivers.values():
        owners.setdefault(receiver.norm_address, set()).add(receiver.module_address)
    for norm, modules in owners.items():
        if len(modules) > 1:
            refuse(
                code="mi_edge_scores_shared_norm",
                message=f"Norm {norm!r} feeds {len(modules)} receiver modules "
                f"({sorted(modules)}), so the gradient at its output mixes their paths and "
                "a per-receiver gradient cannot be read there.",
                remedy="use a model whose attention and MLP each read the residual stream "
                "through their own norm (GPT-2, GPT-NeoX/Pythia, Llama-class)",
                norm_address=norm,
            )


def _norm_output_label(trace: Any, receiver: ComponentReceiver) -> str:
    """Return the label of the op that leaves a receiver's pre-norm."""

    outputs = list(getattr(trace.modules[receiver.norm_address], "output_ops", ()) or ())
    if len(outputs) != 1:
        refuse(
            code="mi_edge_selection_unresolvable",
            message=f"Norm {receiver.norm_address!r} of receiver {receiver.name!r} has "
            f"{len(outputs)} output ops; one is needed to read the receiver's gradient.",
            remedy="capture with default options (layers_to_save='all')",
            receiver=receiver.name,
        )
    return str(trace.ops[str(outputs[0])].label)


def _reads(trace: Any, graph: ComponentGraph) -> dict[str, _Read]:
    """Resolve every receiver's own read (the seed of its local pull-back) in ``trace``."""

    reads: dict[str, _Read] = {}
    for receiver in graph.receivers.values():
        if receiver.input_label not in trace.ops:
            refuse(
                code="mi_edge_scores_inputs_mismatched",
                message=f"Receiver {receiver.name!r} reads {receiver.input_label!r}, which "
                "the evaluation-point capture does not contain.",
                remedy="pass corrupted inputs with the same shape as the clean inputs",
                receiver=receiver.name,
            )
        if receiver.kind in ("q", "k", "v"):
            spec = _head_spec(trace, receiver)
            reads[receiver.name] = _Read(str(spec.home_label), receiver.input_label, spec)
        else:
            reads[receiver.name] = _Read(
                _norm_output_label(trace, receiver), receiver.input_label, None
            )
    return reads


def _output_cotangent(trace: Any, metric: Callable[..., Any]) -> Any:
    """Return ``(logits label, dM/dlogits, per-example M)`` for one evaluation point."""

    from ._anchors import resolve_lm_head

    head = resolve_lm_head(trace).head
    outputs = list(getattr(head, "output_ops", ()) or ())
    label = str(trace.ops[str(outputs[-1])].label)
    logits = trace.ops[label].out
    if not isinstance(logits, torch.Tensor):
        refuse(
            code="mi_payload_missing",
            message=f"The unembedding output {label!r} was not saved.",
            remedy="capture with default options (layers_to_save='all')",
            analysis="edge_scores",
        )
    leaf = logits.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        values = metric(leaf)
    if not isinstance(values, torch.Tensor) or values.dim() > 1 or not values.is_floating_point():
        refuse(
            code="mi_edge_scores_metric_invalid",
            message="metric must return a real tensor with one value per example (shape "
            f"[batch]) or one scalar; it returned {type(values).__name__}"
            f"{'' if not isinstance(values, torch.Tensor) else tuple(values.shape)}.",
            remedy="return per-example values, for example "
            "lambda logits: logits[:, -1, correct] - logits[:, -1, wrong]",
        )
    (cotangent,) = torch.autograd.grad(values.sum(), leaf, allow_unused=True)
    if cotangent is None:
        refuse(
            code="mi_edge_scores_metric_invalid",
            message="metric does not depend on the logits it is given.",
            remedy="compute the metric from the logits argument",
        )
    return label, cotangent.detach(), values.detach()


def _grad_rows(table: Any) -> dict[str, torch.Tensor]:
    """Map each read row's pass-qualified label to its dense gradient (``None`` if unreachable)."""

    out: dict[str, torch.Tensor] = {}
    for row in table.rows():
        layer_label, pass_index = row.address
        out.setdefault(f"{layer_label}:{pass_index}", row.value)
        out.setdefault(str(layer_label), row.value)
    return out


def _site_gradients(trace: Any, seed_label: str, cotangent: Any, sites: list[str]) -> dict:
    """One backward from ``seed_label`` (cotangent given) to every site, element grain."""

    from ..attribution.onebackward import read, seed

    population = None
    for site in sites:
        selection = trace.ops[site].__selection__()
        population = selection if population is None else population | selection
    table = read(
        trace,
        target=seed(seed_label, cotangent=cotangent),
        within=population,
        frozen=None,
        method="grad",
        reduce=None,
    )
    return _grad_rows(table)


def _local_pullbacks(
    trace: Any, seeds: list[tuple[str, torch.Tensor]], input_label: str
) -> list[torch.Tensor | None]:
    """Pull several seed cotangents back to one receiver input (local, batched)."""

    from ..attribution.onebackward import read, seed

    targets = [seed(label, cotangent=cot) for label, cot in seeds]
    element_bytes = sum(int(cot.numel()) * cot.element_size() for _label, cot in seeds)
    table = read(
        trace,
        target=targets,
        within=trace.ops[input_label].__selection__(),
        frozen=None,
        method="grad",
        reduce=None,
        result_byte_budget=max(1, 4 * element_bytes),
    )
    by_target: dict[str, torch.Tensor | None] = {}
    for row in table.rows():
        by_target[str(row.target_id)] = row.value
    return [by_target.get(f"t{index}") for index in range(len(seeds))]


def _point_gradients(
    trace: Any, graph: ComponentGraph, metric: Callable[..., Any]
) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
    """Every receiver's own gradient at one evaluation point."""

    reads = _reads(trace, graph)
    logits_label, cotangent, values = _output_cotangent(trace, metric)
    seed_labels = sorted({read.seed_label for read in reads.values()})
    at_seeds = _site_gradients(trace, logits_label, cotangent, seed_labels)
    grouped: dict[tuple[str, str], list[tuple[str, torch.Tensor]]] = {}
    names: dict[tuple[str, str], list[str]] = {}
    unreached: list[str] = []
    for name, read in reads.items():
        grad = at_seeds.get(read.seed_label)
        if grad is None:
            unreached.append(name)
            continue
        if read.spec is not None:
            grad = read.spec.scatter_update(
                torch.zeros_like(grad), read.spec.apply_transforms(grad)
            )
        key = (read.input_label, read.seed_label)
        grouped.setdefault(key, []).append((read.seed_label, grad))
        names.setdefault(key, []).append(name)
    gradients: dict[str, torch.Tensor | None] = dict.fromkeys(unreached)
    for key, seeds in grouped.items():
        pulled = _local_pullbacks(trace, seeds, key[0])
        gradients.update(zip(names[key], pulled, strict=True))
    return gradients, values


def _interpolation_points(
    model: Any, inputs: Any, site_module: str, difference: torch.Tensor, steps: int
) -> Iterator[Any]:
    """Captures at ``e(k/m)`` for ``k = 0..m-1``, one at a time; the shift
    ``(1 - k/m) * difference`` rides the output of ``site_module`` (the first embedding
    write)."""

    from ..intervention.predicates import add
    from ..intervention.selectors import module as module_selector
    from ..intervention.spec import when

    site = module_selector(site_module)
    for k in range(steps):
        shift = (1.0 - k / steps) * difference
        yield _capture(model, inputs, intervene=when(site, add(shift)))


def _edge_rows(
    graph: ComponentGraph, scores: dict[str, float], statuses: dict[str, str]
) -> list[dict[str, Any]]:
    """One EDGE-table row mapping per graph edge, in graph order."""

    rows = []
    for name, edge in graph.edges.items():
        value = edge.sender.value
        rows.append(
            {
                "name": name,
                "score": scores.get(name),
                "status": statuses.get(name, "ok"),
                "shape": tuple(int(n) for n in value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        )
    return rows


def _accumulate(
    total: dict[str, torch.Tensor | None], point: dict[str, torch.Tensor | None]
) -> None:
    """Add one point's receiver gradients into the running sum (``None`` stays unreachable)."""

    for name, grad in point.items():
        if grad is None:
            total.setdefault(name, None)
            continue
        previous = total.get(name)
        total[name] = grad.detach().clone() if previous is None else previous + grad.detach()


def edge_scores(  # noqa: PLR0913 -- the public door: five keyword-only spec-pinned knobs
    model: Any,
    inputs: Any,
    *,
    corrupted_input: Any,
    metric: Callable[[torch.Tensor], torch.Tensor],
    method: str = "eap_ig",
    steps: int | None = None,
    example_reduction: str = "mean",
) -> Any:
    """Score every edge of the component graph by edge attribution patching.

    Parameters
    ----------
    model:
        A pre-norm additive-residual language model (the models
        :func:`component_graph` accepts; eager attention).
    inputs:
        The clean inputs (token ids ``[batch, pos]``).
    corrupted_input:
        Keyword-only. The corrupted inputs, the same shape as ``inputs``.
    metric:
        Keyword-only. ``logits -> values``: the metric ``M`` from the
        unembedding output, one value per example (``[batch]``) or one
        scalar.
    method:
        Keyword-only. ``"eap_ig"`` (default; the gradient averaged over
        ``steps`` points between the corrupted and the clean input
        embedding, left Riemann sum) or ``"eap"`` (one gradient, at the
        clean input).
    steps:
        Keyword-only. EAP-IG's number of evaluation points ``m``; ``None``
        means 5 (the MIB reference default). Refused with ``method="eap"``.
    example_reduction:
        Keyword-only. ``"mean"`` (default) averages the per-example scores
        over the batch; ``"sum"`` adds them.

    Returns
    -------
    ReadTable
        One ``EDGE`` row per graph edge, addressed ``(edge name,)`` in the
        component-graph vocabulary (``"a9.h9->logits"``); ``score`` is
        ``sum over positions of (s_corrupt - s_clean) . dM/d(input of t)``,
        reduced over examples. Bias-sender edges score exactly zero (their
        corrupted and clean values are equal).
    """

    started = time.perf_counter()
    m = _validate(method, steps, example_reduction)
    clean_trace = _capture(model, inputs)
    clean = component_graph(clean_trace)
    _refuse_shared_norms(clean)
    corrupt = component_graph(_capture(model, corrupted_input))
    deltas = _node_deltas(clean, corrupt)
    total: dict[str, torch.Tensor | None] = {}
    if method == "eap":
        point, _values = _point_gradients(clean_trace, clean, metric)
        _accumulate(total, point)
    else:
        site_module = _input_write_module(clean_trace, clean)
        difference = (corrupt.nodes["input"].value - clean.nodes["input"].value).detach()
        for point_trace in _interpolation_points(model, inputs, site_module, difference, m):
            point, _values = _point_gradients(point_trace, component_graph(point_trace), metric)
            _accumulate(total, point)
            point_trace.cleanup()
    n_examples = int(clean.nodes["input"].value.shape[0])
    scores: dict[str, float] = {}
    statuses: dict[str, str] = {}
    for name, edge in clean.edges.items():
        grad = total.get(edge.receiver.name)
        if grad is None:
            statuses[name] = "unreachable"
            continue
        per_example = (deltas[edge.sender.name] * (grad / m)).flatten(1).sum(dim=1)
        reduced = per_example.mean() if example_reduction == "mean" else per_example.sum()
        scores[name] = float(reduced)
    from ..attribution.onebackward._table import edge_read_table

    return edge_read_table(
        _edge_rows(clean, scores, statuses),
        method=method,
        reduction=f"sum_positions_{example_reduction}_examples",
        target_repr=f"metric via {method}" + (f" (steps={m})" if method == "eap_ig" else ""),
        sample_count=n_examples,
        timing_s=time.perf_counter() - started,
        trace=clean_trace,
    )
