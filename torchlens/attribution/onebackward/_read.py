"""The one-backward read orchestrator (M(reads) item 2: D3/D4/D9/D10/D14/D15).

``read(trace, target=..., within=..., frozen=..., method=..., reduce=...,
target_batch_size=..., result_byte_budget=...)`` is the one door. Population
law (D9, verbatim): explicit enumeration is a contract; implicit population
is a filter. An explicit ``within=`` naming an unretained site refuses at
preflight with the first missing addresses and a concrete recapture recipe;
an implicit population serves the eligible sites and records excluded counts
by reason. Under an intermediate target the implicit population defaults to
the target's ancestor cone (D10) -- rows autograd returns ``None`` for are
excluded WITH COUNTS, while explicitly-named sites keep honest
``unreachable`` rows. A numeric zero stays ``ok``; ``None`` never becomes a
zero.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from typing import Any

import torch

from ._accessor import ReadEdgeIndex, SiteEdge, read_edge_index
from ._engine import EngineSpec, resolve_batch_plan, run_engine
from ._errors import ReadError
from ._frozen import DEFAULT, FrozenPlan, _DefaultFrozenSentinel, resolve_frozen
from ._table import ReadRow, ReadTable, TableProvenance
from ._targets import NormalizedTarget, normalize_targets

__all__ = ["read", "METHODS", "REDUCTIONS"]

# Closed method vocabulary; the tuple is the item-9 extension seam (R4
# position targets and later methods extend it in their own lanes).
METHODS: tuple[str, ...] = ("activation_x_grad", "grad", "activation")

_GRADIENT_METHODS: frozenset[str] = frozenset({"activation_x_grad", "grad"})


def _reduce_sum(value: torch.Tensor) -> torch.Tensor:
    """Signed sum."""

    return value.sum()


def _reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Signed mean over the reduced elements."""

    return value.mean()


def _reduce_sum_of_abs(value: torch.Tensor) -> torch.Tensor:
    """Sum of absolute values -- DISTINCT from abs_of_sum (D15)."""

    return value.abs().sum()


def _reduce_abs_of_sum(value: torch.Tensor) -> torch.Tensor:
    """Absolute value of the signed sum (Molchanov/VISCNN Taylor importance)."""

    return value.sum().abs()


def _reduce_l2(value: torch.Tensor) -> torch.Tensor:
    """Euclidean norm."""

    return torch.linalg.vector_norm(value.reshape(-1))


def _reduce_max_abs(value: torch.Tensor) -> torch.Tensor:
    """Largest magnitude."""

    return value.abs().max()


REDUCTIONS: dict[str, Any] = {
    "sum": _reduce_sum,
    "mean": _reduce_mean,
    "sum_of_abs": _reduce_sum_of_abs,
    "abs_of_sum": _reduce_abs_of_sum,
    "l2": _reduce_l2,
    "max_abs": _reduce_max_abs,
}


def _digest_of(parts: tuple[str, ...]) -> str:
    """Short stable digest over sorted string parts."""

    hasher = hashlib.sha256()
    for part in sorted(parts):
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


def _capture_outcome_status(trace: Any) -> str | None:
    """Best-effort settled capture-outcome status string."""

    try:
        outcome = trace.outcome
    except Exception:  # noqa: BLE001 - disclosure only, never a gate here
        return None
    status = getattr(outcome, "status", None)
    return getattr(status, "name", None) or (str(status) if status is not None else None)


def _trace_label(trace: Any) -> str | None:
    """Best-effort human identity for provenance."""

    name = getattr(trace, "model_name", None)
    return str(name) if name else None


def _validate_method(method: str) -> None:
    """Refuse unknown methods with the closed vocabulary."""

    if method not in METHODS:
        raise ReadError(
            f"method {method!r} is not in the closed vocabulary {METHODS}. "
            "Remedy: pick a listed method",
            code="read_option_invalid",
            option="method",
            value=method,
        )


def _validate_reduce(reduce: str | None) -> None:
    """Refuse unknown reductions with the closed vocabulary."""

    if reduce is not None and reduce not in REDUCTIONS:
        raise ReadError(
            f"reduce {reduce!r} is not a named reduction "
            f"({sorted(REDUCTIONS)}) or None (element grain). Note that "
            "sum_of_abs and abs_of_sum are DISTINCT reductions. Remedy: "
            "pick a named reduction or None",
            code="read_option_invalid",
            option="reduce",
            value=reduce,
        )


class _SitePayloads:
    """Per-site payload access with save-mode honesty (D14)."""

    def __init__(self, trace: Any) -> None:
        """Index ops by pass-qualified label for payload reads."""

        self._ops = {op.label: op for op in trace.ops}
        self._save_mode = getattr(trace, "save_mode", "copy")
        self._cache: dict[str, torch.Tensor | None] = {}

    @property
    def save_mode(self) -> str:
        """The trace's payload retention mode."""

        return str(self._save_mode)

    def retention(self, label: str) -> str:
        """Return ``'saved'`` / ``'unsaved'`` for a site."""

        op = self._ops.get(label)
        if op is None:
            return "unsaved"
        return "saved" if bool(getattr(op, "has_saved_activation", False)) else "unsaved"

    def payload(self, label: str) -> torch.Tensor | None:
        """Return the saved payload tensor, or ``None`` when unretained.

        ``save_mode='reference'`` mutation validation runs inside the ``out``
        read and its ``MutatedReferenceError`` propagates unchanged (D14).
        """

        if label in self._cache:
            return self._cache[label]
        op = self._ops.get(label)
        value: torch.Tensor | None = None
        if op is not None and bool(getattr(op, "has_saved_activation", False)):
            out = op.out
            value = out if isinstance(out, torch.Tensor) else None
        self._cache[label] = value
        return value


def _preflight_view_mode(payloads: _SitePayloads, method: str) -> None:
    """Refuse activation methods on ``save_mode='view'`` captures, fail-closed.

    A view payload can be silently mutated after capture with no recorded
    baseline version to prove it unchanged, so activation-bearing methods
    refuse rather than multiply untrustworthy values (D14).
    """

    if method in ("activation_x_grad", "activation") and payloads.save_mode == "view":
        raise ReadError(
            "save_mode='view' payloads cannot be proven unmutated, so "
            f"method={method!r} refuses on this capture. Remedy: re-capture "
            "with save_mode='copy' (or use method='grad', which needs no "
            "payload)",
            code="read_payload_untrustworthy",
            save_mode="view",
            method=method,
        )


def _resolve_within_impl(
    trace: Any,
    index: ReadEdgeIndex,
    within: Any,
) -> tuple[dict[str, torch.Tensor | None], bool]:
    """Resolve ``within=`` to ``label -> element mask (or None)``.

    Returns the population mapping and whether it was EXPLICIT. Implicit
    (``within=None``) serves every addressable site. Explicit resolution
    reuses the selection algebra (``_lift_within``), so PARAM/EDGE
    populations refuse through the same typed door as every other ACT value
    producer.
    """

    if within is None:
        return (dict.fromkeys(index.edges), False)
    from ...selection import ResolvedSelection, Selection
    from ...selection_values import _lift_within

    population: dict[str, torch.Tensor | None] = {}
    if isinstance(within, ResolvedSelection):
        resolved = within
    else:
        lifted = _lift_within(within, "read")
        if isinstance(lifted, Selection):
            resolved = lifted.resolve(trace)
        else:
            raise ReadError(
                "within= did not lift to a selection. Remedy: pass None, a "
                "site label, or a selection-shaped producer",
                code="read_option_invalid",
                option="within",
            )
    if resolved.kind != "ACT":
        raise ReadError(
            f"within= must be an ACT selection; got {resolved.kind!r}. "
            "Remedy: select activation sites",
            code="read_option_invalid",
            option="within",
            kind=resolved.kind,
        )
    for entry in resolved:
        site_key = entry.site_key
        label = f"{site_key[0]}:{site_key[1]}" if len(site_key) == 2 else str(site_key[0])
        mask = entry.mask
        population[label] = None if bool(mask.all()) else mask
    return (population, True)


def _explicit_preflight(
    index: ReadEdgeIndex,
    payloads: _SitePayloads,
    population: dict[str, torch.Tensor | None],
    method: str,
) -> None:
    """Enforce the D9 explicit-population contract at preflight.

    Explicit sites must exist on the trace; for activation-bearing methods
    they must also be retained -- the refusal names the FIRST missing
    addresses and the concrete recapture recipe.
    """

    unknown = [
        label
        for label in population
        if label not in index.edges and label not in index.unaddressable
    ]
    if unknown:
        raise ReadError(
            f"within= names sites absent from this trace: {unknown[:5]}. "
            "Remedy: pick pass-qualified op labels from the trace",
            code="read_population_invalid",
            missing=unknown[:20],
        )
    if method in ("activation_x_grad", "activation"):
        unretained = [label for label in population if payloads.retention(label) != "saved"]
        if unretained:
            raise ReadError(
                f"within= explicitly names {len(unretained)} sites without "
                f"retained payloads (first: {unretained[:5]}) and "
                f"method={method!r} multiplies payloads. Explicit enumeration "
                "is a contract. Remedy: re-capture with save= covering these "
                "sites, e.g. tl.trace(model, x, save=[...labels...]), or use "
                "method='grad' (no payload needed)",
                code="read_payload_unretained",
                missing=unretained[:20],
                method=method,
            )


def _estimate_element_bytes(
    index: ReadEdgeIndex, population: dict[str, torch.Tensor | None], n_targets: int
) -> int:
    """Upper-bound the dense result bytes for ``reduce=None``."""

    total = 0
    for label in population:
        edge = index.edges.get(label)
        if edge is None or edge.shape is None:
            continue
        numel = 1
        for extent in edge.shape:
            numel *= int(extent)
        dtype = edge.dtype if isinstance(edge.dtype, torch.dtype) else torch.float32
        total += numel * dtype.itemsize
    return total * max(1, n_targets)


def _prepare_targets(
    trace: Any, index: ReadEdgeIndex, method: str, target: Any
) -> tuple[NormalizedTarget, ...]:
    """Validate target presence per method and normalize (order preserved)."""

    if method == "activation":
        if target is not None:
            raise ReadError(
                "method='activation' reads saved payloads and takes no "
                "target. Remedy: drop target= or pick a gradient-bearing "
                "method",
                code="read_option_invalid",
                option="target",
            )
        return ()
    if target is None:
        raise ReadError(
            f"method={method!r} needs target=. Remedy: pass "
            "seed(site, index=...) / seed(site, cotangent=...), a "
            "graph-connected scalar Tensor, or a Trace -> Tensor callable",
            code="read_target_invalid",
        )
    return normalize_targets(trace, index, target)


def _prepare_frozen(
    trace: Any, index: ReadEdgeIndex, method: str, frozen: Any
) -> tuple[FrozenPlan, tuple[str, ...]]:
    """Resolve frozen= per method; activation reads run no backward."""

    if method in _GRADIENT_METHODS:
        return resolve_frozen(trace, index, frozen, method=method)
    if frozen is not None and not isinstance(frozen, _DefaultFrozenSentinel):
        raise ReadError(
            "method='activation' runs no backward, so frozen= has no "
            "effect. Remedy: drop frozen= or pick a gradient-bearing method",
            code="read_option_invalid",
            option="frozen",
        )
    return (
        FrozenPlan(
            policy="none",
            sites={},
            masks={},
            digest="none",
            alias_disclosures={},
            disclosures=(),
        ),
        (),
    )


def _filter_implicit_population(
    index: ReadEdgeIndex,
    population: dict[str, torch.Tensor | None],
    method: str,
    exclude: Any,
) -> dict[str, torch.Tensor | None]:
    """Filter an implicit population to the eligible set, counting exclusions."""

    filtered: dict[str, torch.Tensor | None] = {}
    for label, mask in population.items():
        edge = index.edges.get(label)
        if edge is None:
            exclude(index.unaddressable.get(label, "no_grad_fn"))
            continue
        dtype = edge.dtype if isinstance(edge.dtype, torch.dtype) else None
        if method in _GRADIENT_METHODS and dtype is not None and not dtype.is_floating_point:
            exclude("not_differentiable")
            continue
        filtered[label] = mask
    return filtered


def _enforce_result_budget(
    index: ReadEdgeIndex,
    population: dict[str, torch.Tensor | None],
    targets: tuple[NormalizedTarget, ...],
    reduce: str | None,
    result_byte_budget: int | None,
) -> None:
    """Refuse an unbudgeted (or over-budget) multi-target element-grain read."""

    if reduce is not None or len(targets) <= 1:
        return
    estimate = _estimate_element_bytes(index, population, len(targets))
    if result_byte_budget is None:
        raise ReadError(
            f"reduce=None with {len(targets)} targets would retain an "
            f"estimated {estimate} bytes of dense values. Remedy: pass "
            "result_byte_budget=<bytes> explicitly, or use a named "
            "scalar reduction (reduce='sum' etc.)",
            code="read_result_budget_required",
            estimated_bytes=estimate,
            n_targets=len(targets),
        )
    if estimate > result_byte_budget:
        raise ReadError(
            f"reduce=None with {len(targets)} targets is estimated at "
            f"{estimate} bytes, over the result_byte_budget of "
            f"{result_byte_budget}. Remedy: raise the budget, narrow "
            "within=, or use a named scalar reduction",
            code="read_result_budget_required",
            estimated_bytes=estimate,
            budget=result_byte_budget,
            n_targets=len(targets),
        )


def _plan_device(targets: tuple[NormalizedTarget, ...]) -> str:
    """Pick the device the batching plan is resolved for."""

    for entry in targets:
        candidate = entry.cotangent if entry.cotangent is not None else entry.tensor
        if isinstance(candidate, torch.Tensor) and candidate.device.type != "cpu":
            return candidate.device.type
    return "cpu"


class _RowBuilder:
    """Builds honesty rows for one read call (shared skeleton + fold logic)."""

    def __init__(  # noqa: PLR0913 -- one binder for the read call's full shared state
        self,
        *,
        index: ReadEdgeIndex,
        payloads: _SitePayloads,
        population: dict[str, torch.Tensor | None],
        explicit: bool,
        method: str,
        reduce: str | None,
        device: str,
        frozen_plan: FrozenPlan,
        capture_status: str | None,
        exclude: Any,
    ) -> None:
        """Bind the per-call state every row shares."""

        self._index = index
        self._payloads = payloads
        self._population = population
        self._explicit = explicit
        self._method = method
        self._reduce = reduce
        self._reducer = REDUCTIONS.get(reduce) if reduce is not None else None
        self._device = device
        self._frozen_plan = frozen_plan
        self._capture_status = capture_status
        self._exclude = exclude
        self._group_ordinals: dict[tuple[int, int], int] = {
            key: ordinal for ordinal, key in enumerate(sorted(index.alias_groups, key=repr))
        }
        self._labels_by_alias: dict[tuple[int, int], list[str]] = {}
        for label in population:
            edge = index.edges.get(label)
            if edge is not None:
                self._labels_by_alias.setdefault(edge.alias_key, []).append(label)
        self.rows: dict[tuple[str | None, str, tuple[Any, ...]], ReadRow] = {}

    @property
    def capture_status(self) -> str | None:
        """The capture's settled outcome status string."""

        return self._capture_status

    def row_base(self, label: str, target_id: str | None) -> dict[str, Any]:
        """Build the shared honesty-field skeleton for one row."""

        edge = self._index.edges[label]
        return {
            "target_id": target_id,
            "kind": "ACT",
            "address": (edge.layer_label, edge.pass_index),
            "site_key": edge.site_key,
            "alias_group": self._group_ordinals.get(edge.alias_key),
            "method": self._method,
            "reduction": self._reduce,
            "grain": "site" if self._reduce is not None else "element",
            "score": None,
            "value": None,
            "shape": edge.shape,
            "dtype": str(edge.dtype) if edge.dtype is not None else None,
            "device": self._device,
            "status": "ok",
            "status_reason": None,
            "differentiable": True,
            "retention": self._payloads.retention(label),
            "capture_status": self._capture_status,
            "resolution": "exact",
            "frozen_requested": label in self._frozen_plan.sites,
            "frozen_resolved": label in self._frozen_plan.sites,
            "frozen_reached": None,
            "policy_digest": self._frozen_plan.digest,
            "detached": True,
            "sample_id": None,
            "rescorable": True,
        }

    def _emit(self, base: dict[str, Any]) -> None:
        """Freeze one row dict into the table rows."""

        row = ReadRow(**base)
        self.rows[row.key] = row

    def _fold_value(self, base: dict[str, Any], value: torch.Tensor, mask: Any) -> None:
        """Apply the mask and the named reduction (or carry the dense value)."""

        if mask is not None:
            value = value * mask.to(value.dtype)
        if self._reducer is not None:
            selected = value[mask] if mask is not None else value
            base["score"] = float(self._reducer(selected))
        else:
            base["value"] = value.detach().clone()
        self._emit(base)

    def build_activation_rows(self) -> None:
        """Fill rows for ``method='activation'`` (no backward, payloads only)."""

        for label, mask in self._population.items():
            base = self.row_base(label, None)
            payload = self._payloads.payload(label)
            if payload is None:
                base.update(status="unavailable", status_reason="payload_unretained")
                self._emit(base)
                continue
            self._fold_value(base, payload.detach(), mask)

    def consume(
        self, target_id: str, alias_key: tuple[int, int], grad: torch.Tensor | None
    ) -> None:
        """Fold one (target, unique-site) gradient into rows, streaming."""

        for label in self._labels_for(alias_key):
            base = self.row_base(label, target_id)
            if grad is None:
                if self._explicit:
                    base.update(status="unreachable", status_reason="not_upstream_of_target")
                    self._emit(base)
                else:
                    self._exclude("not_upstream_of_target")
                continue
            value = self._gradient_value(label, grad, base)
            if value is not None:
                self._fold_value(base, value, self._population.get(label))

    def _labels_for(self, alias_key: tuple[int, int]) -> tuple[str, ...]:
        """Member labels of one unique ``(node, slot)`` in this population."""

        return tuple(self._labels_by_alias.get(alias_key, ()))

    def _gradient_value(
        self, label: str, grad: torch.Tensor, base: dict[str, Any]
    ) -> torch.Tensor | None:
        """Return the method's value tensor, or emit the payload refusal row."""

        if self._method != "activation_x_grad":
            return grad
        payload = self._payloads.payload(label)
        if payload is None:
            base.update(status="unavailable", status_reason="payload_unretained")
            self._emit(base)
            return None
        if payload.device != grad.device:
            payload = payload.to(grad.device)
        return payload.detach() * grad

    def stamp_frozen_reached(self, fired: frozenset[str]) -> None:
        """Stamp per-row freeze-fired honesty after the engine run."""

        if not self.rows or not fired:
            return
        self.rows = {
            key: replace(row, frozen_reached=(_label_of(row) in fired))
            if row.frozen_resolved
            else row
            for key, row in self.rows.items()
        }


def read(  # noqa: PLR0913 -- the M(reads) public door: eight keyword-only spec-pinned knobs
    trace: Any,
    *,
    target: Any = None,
    within: Any = None,
    frozen: Any = DEFAULT,
    method: str = "activation_x_grad",
    reduce: str | None = "sum",
    target_batch_size: Any = "auto",
    result_byte_budget: int | None = None,
) -> ReadTable:
    """One backward pass; a table of how much each site matters.

    Parameters
    ----------
    trace:
        Live finished torch trace.
    target:
        Keyword-only: a ``seed(site, index=|cotangent=)`` edge-seeded target
        (primary), a graph-connected scalar Tensor, a pure
        ``Trace -> Tensor`` callable (1-D result = target batch), or a list
        of these. Order preserved. Required for gradient-bearing methods;
        forbidden for ``method='activation'``.
    within:
        Population scoping (same seam as every value producer): ``None`` =
        implicit eligible population; a site label / Selection = explicit
        contract population.
    frozen:
        Omitted -> the default MLP-output linearization; ``None`` -> total
        derivative; explicit ACT selection/site specs -> exactly that set,
        replacing the default.
    method:
        ``'activation_x_grad'`` (default, signed), ``'grad'``,
        ``'activation'``.
    reduce:
        Named scalar reduction (site grain) or ``None`` (element grain).
    target_batch_size:
        ``'auto'`` (plan disclosed; CUDA default stays sequential pending
        C-READ) or an explicit positive int.
    result_byte_budget:
        Required for multi-target element-grain reads; refused with an
        estimate when exceeded.

    Returns
    -------
    ReadTable
        Immutable carrier with per-row honesty and table provenance.
    """

    started = time.perf_counter()
    _validate_method(method)
    _validate_reduce(reduce)
    index = read_edge_index(trace)
    payloads = _SitePayloads(trace)
    _preflight_view_mode(payloads, method)
    targets = _prepare_targets(trace, index, method, target)
    frozen_plan, warning_codes = _prepare_frozen(trace, index, method, frozen)

    population, explicit = _resolve_within_impl(trace, index, within)
    if explicit:
        _explicit_preflight(index, payloads, population, method)
    excluded_counts: dict[str, int] = {}

    def _exclude(reason: str) -> None:
        """Count one implicit-population exclusion under a closed reason."""

        excluded_counts[reason] = excluded_counts.get(reason, 0) + 1

    if not explicit:
        population = _filter_implicit_population(index, population, method, _exclude)
    _enforce_result_budget(index, population, targets, reduce, result_byte_budget)

    plan = resolve_batch_plan(target_batch_size, _plan_device(targets))
    builder = _RowBuilder(
        index=index,
        payloads=payloads,
        population=population,
        explicit=explicit,
        method=method,
        reduce=reduce,
        device=plan.device,
        frozen_plan=frozen_plan,
        capture_status=_capture_outcome_status(trace),
        exclude=_exclude,
    )
    if method == "activation":
        builder.build_activation_rows()
        report = None
    else:
        input_edges: list[SiteEdge] = []
        seen_alias: set[tuple[int, int]] = set()
        for label in population:
            edge = index.edges.get(label)
            if edge is None or edge.alias_key in seen_alias:
                continue
            seen_alias.add(edge.alias_key)
            input_edges.append(edge)
        report = run_engine(
            trace,
            targets,
            EngineSpec(input_edges=input_edges, frozen_plan=frozen_plan, plan=plan),
            builder.consume,
        )
        builder.stamp_frozen_reached(report.fired_freeze_labels)

    rows = builder.rows
    result_bytes = sum(
        row.value.numel() * row.value.element_size()
        for row in rows.values()
        if isinstance(row.value, torch.Tensor)
    )
    provenance = TableProvenance(
        trace_label=_trace_label(trace),
        capture_outcome=builder.capture_status,
        target_reprs=tuple(entry.source_repr for entry in targets),
        within_digest=_digest_of(tuple(population)) if explicit else None,
        population_size=len(population),
        excluded_counts=dict(sorted(excluded_counts.items())),
        frozen_policy=frozen_plan.policy,
        frozen_digest=frozen_plan.digest,
        frozen_requested_count=len(frozen_plan.sites),
        frozen_resolved_count=len(frozen_plan.sites),
        frozen_fired_count=len(report.fired_freeze_labels) if report is not None else 0,
        alias_group_count=len(index.alias_groups),
        aliased_site_count=sum(len(members) for members in index.alias_groups.values()),
        batching_plan={
            "requested": plan.requested,
            "batch_size": plan.batch_size,
            "reason": plan.reason,
            "device": plan.device,
        },
        autograd_calls=report.autograd_calls if report is not None else 0,
        timing_s=time.perf_counter() - started,
        result_bytes=result_bytes,
        sample_count=1,
        warnings=tuple(warning_codes) + frozen_plan.disclosures,
        rescorable=True,
    )
    from ._accessor import _validity_token

    return ReadTable(rows, provenance, trace=trace, trace_token=_validity_token(trace))


def _label_of(row: ReadRow) -> str:
    """Rebuild the pass-qualified label from an ACT row address."""

    return f"{row.address[0]}:{row.address[1]}"
