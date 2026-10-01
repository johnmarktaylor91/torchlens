"""The one-backward product engine (M(reads) items 2/4/6: D3, D11, D12).

ONE ``autograd.grad`` call pattern serves every read site at once: inputs are
the alias-deduped GradientEdges of the whole read population (one gradient
per unique ``(node, slot)``, D11), ``allow_unused=True``,
``materialize_grads=False`` ALWAYS (torch itself REFUSES ``True`` on
GradientEdge inputs -- the band probe pinned it -- and silent
zero-materialization would fabricate rows), ``retain_graph=True`` always
(repeated reads and a later ordinary ``log_backward`` must work).

Target batching (D12) is chunked ``is_grads_batched=True`` VJPs with a
plan-vs-refusal posture -- BUT chunks group by CONE: a batched call returns
ZERO rows (never ``None``) for a (target, input) pair whose input is outside
that target's cone while inside a chunk-mate's, fabricating exactly the
silent zeros D3 forbids (measured in this lane's band probe). Targets share
a chunk ONLY when they share a cone key (same seed ``(node, slot)``, or the
same expression tensor), where per-input ``None`` is exact for every row.

Streaming reductions: each chunk's gradients are handed to the consumer
callback and released before the next chunk runs.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch

from ._accessor import SiteEdge
from ._errors import ReadError, ReadInternalError
from ._frozen import FrozenPlan, install_freeze_hooks
from ._suppress import read_suppressed
from ._targets import NormalizedTarget

__all__ = ["BatchPlan", "EngineReport", "EngineSpec", "run_engine", "resolve_batch_plan"]

# Auto plan defaults. CPU batching is a measured single-digit win (1.7-4.3x
# across labs and substrates), so auto engages a modest chunk; the CUDA
# crossover is UNMEASURED (every published ratio is CPU) -- the CUDA auto
# default stays sequential until the C-READ cluster row lands (D13/D12).
_AUTO_CPU_BATCH = 32
_AUTO_CPU_REASON = "cpu_batched_vjp_single_digit_gain_measured"
_AUTO_CUDA_REASON = "cuda_envelope_unmeasured_pending_c_read"


@dataclass(frozen=True)
class BatchPlan:
    """The disclosed target-batching plan (D12).

    Attributes
    ----------
    requested:
        The user request (``'auto'`` or an explicit positive int).
    batch_size:
        The resolved per-chunk target count (B=1 is a legitimate measured
        plan on CPU, not a degrade).
    reason:
        Why this size was chosen; stamped on the table.
    device:
        Device the plan was resolved for.
    """

    requested: Any
    batch_size: int
    reason: str
    device: str


@dataclass(frozen=True)
class EngineReport:
    """What the engine actually did (exact, deterministic, gate-able)."""

    autograd_calls: int
    chunk_count: int
    cone_group_count: int
    timing_s: float
    fired_freeze_labels: frozenset[str]
    plan: BatchPlan


def resolve_batch_plan(requested: Any, device: str) -> BatchPlan:
    """Resolve ``target_batch_size=`` to a disclosed plan.

    Parameters
    ----------
    requested:
        ``'auto'`` or a positive int.
    device:
        Device type string the read runs on.

    Raises
    ------
    ReadError
        Code ``read_option_invalid`` on anything else -- never a silent
        fallback.
    """

    if requested == "auto":
        if device == "cpu":
            return BatchPlan(
                requested="auto",
                batch_size=_AUTO_CPU_BATCH,
                reason=_AUTO_CPU_REASON,
                device=device,
            )
        return BatchPlan(requested="auto", batch_size=1, reason=_AUTO_CUDA_REASON, device=device)
    if isinstance(requested, int) and not isinstance(requested, bool) and requested >= 1:
        return BatchPlan(
            requested=requested, batch_size=requested, reason="explicit", device=device
        )
    raise ReadError(
        f"target_batch_size must be 'auto' or a positive int; got "
        f"{requested!r}. Remedy: pass 'auto' or an explicit chunk size",
        code="read_option_invalid",
        option="target_batch_size",
        value=repr(requested),
    )


def _group_by_cone(targets: tuple[NormalizedTarget, ...]) -> list[list[NormalizedTarget]]:
    """Group targets by cone key, order-preserving within and across groups."""

    groups: dict[tuple[Any, ...], list[NormalizedTarget]] = {}
    for target in targets:
        groups.setdefault(target.cone_key, []).append(target)
    return list(groups.values())


def _chunk_output(chunk: list[NormalizedTarget]) -> Any:
    """Return the single shared seed output for a cone-keyed chunk."""

    first = chunk[0]
    if first.family == "edge":
        # The normalizer guarantees an edge for the edge family.
        return cast(SiteEdge, first.edge).gradient_edge()
    return first.tensor


def _translate_engine_error(
    error: RuntimeError, *, batched: bool, chunk_size: int
) -> ReadError | None:
    """Translate a torch engine failure to the typed read refusal, if known."""

    text = str(error)
    if "backward through the graph a second time" in text or "Saved intermediate" in text:
        return ReadError(
            "The trace's autograd graph has been freed (a backward without "
            "retain_graph ran on it). Remedy: re-capture the model and read "
            "before any graph-freeing backward",
            code="read_addressing_unavailable",
            reason="graph_freed",
        )
    lowered = text.lower()
    if batched and (
        "vmap" in lowered or "batching rule" in lowered or "is_grads_batched" in lowered
    ):
        return ReadError(
            "A batched VJP hit an operator without a batching rule "
            f"(chunk size {chunk_size}): {text.splitlines()[0]} "
            "Remedy: pass target_batch_size=1 to run sequentially",
            code="batched_attribution_unsupported",
            chunk_size=chunk_size,
        )
    return None


@dataclass(frozen=True)
class EngineSpec:
    """The engine's per-read configuration bundle.

    Attributes
    ----------
    input_edges:
        Alias-deduped read sites: ONE representative :class:`SiteEdge` per
        unique ``(node, slot)``.
    frozen_plan:
        Resolved freeze plan; hooks installed around the whole run and
        removed in ``finally``.
    plan:
        Resolved batch plan.
    """

    input_edges: list[SiteEdge]
    frozen_plan: FrozenPlan
    plan: BatchPlan


def _chunk_grad_outputs(chunk: list[NormalizedTarget]) -> tuple[list[torch.Tensor], bool]:
    """Return (grad_outputs, batched) for one cone-keyed chunk."""

    # Every normalized target carries a cotangent by construction (the
    # normalizer builds or validates one per family).
    cotangents = [cast(torch.Tensor, entry.cotangent) for entry in chunk]
    if len(chunk) == 1:
        return [cotangents[0]], False
    return [torch.stack(cotangents)], True


def _execute_chunk(
    chunk: list[NormalizedTarget], edge_inputs: list[Any]
) -> tuple[tuple[torch.Tensor | None, ...], bool]:
    """Run one suppressed chunk VJP; translate known engine failures typed."""

    output = _chunk_output(chunk)
    grad_outputs, batched = _chunk_grad_outputs(chunk)
    try:
        grads = torch.autograd.grad(
            [output],
            edge_inputs,
            grad_outputs=grad_outputs,
            retain_graph=True,
            allow_unused=True,
            materialize_grads=False,
            is_grads_batched=batched,
        )
    except RuntimeError as error:
        translated = _translate_engine_error(error, batched=batched, chunk_size=len(chunk))
        if translated is not None:
            raise translated from error
        raise
    return grads, batched


def _scatter_chunk(
    chunk: list[NormalizedTarget],
    grads: tuple[torch.Tensor | None, ...],
    batched: bool,
    alias_keys: list[tuple[int, int]],
    consume: Callable[[str, tuple[int, int], torch.Tensor | None], None],
) -> None:
    """Hand each (target, unique-site) gradient to the streaming sink."""

    for input_position, alias_key in enumerate(alias_keys):
        grad = grads[input_position]
        if grad is None:
            for entry in chunk:
                consume(entry.target_id, alias_key, None)
            continue
        if batched:
            for row, entry in enumerate(chunk):
                consume(entry.target_id, alias_key, grad[row].detach())
        else:
            consume(chunk[0].target_id, alias_key, grad.detach())


def run_engine(
    trace: Any,
    targets: tuple[NormalizedTarget, ...],
    spec: EngineSpec,
    consume: Callable[[str, tuple[int, int], torch.Tensor | None], None],
) -> EngineReport:
    """Run the suppressed, frozen, chunked VJP engine over all targets.

    Parameters
    ----------
    trace:
        Live trace (suppression context target).
    targets:
        Normalized targets, order preserved.
    spec:
        Input edges, freeze plan, and batch plan for this read.
    consume:
        Streaming sink called as ``consume(target_id, alias_key, grad)``
        with a DETACHED gradient (or ``None`` for autograd-unreachable
        pairs) for every (target, unique site) pair; chunk tensors are
        released after consumption.

    Returns
    -------
    EngineReport
        Exact call counts, timing, plan, and fired freeze labels.
    """

    plan = spec.plan
    edge_inputs = [edge.gradient_edge() for edge in spec.input_edges]
    alias_keys = [edge.alias_key for edge in spec.input_edges]
    groups = _group_by_cone(targets)
    expected_calls = sum(math.ceil(len(group) / plan.batch_size) for group in groups)
    calls = 0
    started = time.perf_counter()
    with read_suppressed(trace), install_freeze_hooks(spec.frozen_plan) as hooks:
        for group in groups:
            for start in range(0, len(group), plan.batch_size):
                chunk = group[start : start + plan.batch_size]
                grads, batched = _execute_chunk(chunk, edge_inputs)
                calls += 1
                _scatter_chunk(chunk, grads, batched, alias_keys, consume)
                del grads
    elapsed = time.perf_counter() - started
    if calls != expected_calls:
        raise ReadInternalError(
            f"Engine call-count drift: expected {expected_calls} autograd "
            f"calls, made {calls}. This is a TorchLens contract breach. "
            "Remedy: report this as a bug",
            code="read_engine_call_drift",
            expected=expected_calls,
            made=calls,
        )
    return EngineReport(
        autograd_calls=calls,
        chunk_count=calls,
        cone_group_count=len(groups),
        timing_s=elapsed,
        fired_freeze_labels=frozenset(hooks.fired_for_plan),
        plan=plan,
    )
