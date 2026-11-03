"""The private candidate engine + the ``site_sweep`` face (F03 items 6-8).

ONE engine, three customers: the ``site_sweep`` face, the
tier-c selector conversion (:meth:`EffectsView.to_selection`), and the shipped
occlusion attribution loop it generalizes. Given a baseline trace, an
ordered candidate plan, a lane, a metric, and a retention policy, the engine
preflights ONCE, executes candidates SERIALLY in stable order ("batched"
means a batch of CANDIDATES; tensor-axis vectorization is a proof-gated
later strategy swap), measures each member BEFORE releasing it (cleanup()
husks the trace — every logged field dies with it), retains or releases
under policy, and appends one effect row per candidate — including released,
refused, and failed candidates, which are table rows and NEVER falsely
members (D3c).

Lane ruling (D3d): default ``engine="live_hook"`` — fork + _attach_hooks +
run(model, x), a real forward with hooks attached. ``engine="replay"`` is
the cone-recompute opt-in (fork + do on saved payloads). The lane is stamped
in every event envelope and effect table; escalation is never silent.

Required arguments (D3b): ``edit=``, ``metric=``, and ``retain=`` have NO
defaults — which ablation is sound is a live scientific argument, the metric
IS the experiment, and retention decides what evidence is DESTROYED.

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ..bundle import Bundle
from ..bundle._bytes import preflight_retention_projection
from ..bundle._lineage import MemberEffectRow, MemberEffectTable
from ..errors.episode import BundleExperimentError

if TYPE_CHECKING:
    from torch import nn

    from .._data_classes.trace import Trace

__all__ = ["TopK", "sweep_sites", "top_k"]

_ENGINE_LANES = ("live_hook", "replay")


@dataclass(frozen=True)
class TopK:
    """Retention policy: keep the k largest-|effect| members (rolling)."""

    k: int

    def __post_init__(self) -> None:
        if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k < 0:
            raise BundleExperimentError(
                f"top_k retention needs a non-negative int, got {self.k!r}",
                code="site_sweep_retain_invalid",
            )

    def __repr__(self) -> str:
        return f"top_k({self.k})"


def top_k(k: int) -> TopK:
    """The ``retain=top_k(k)`` policy constructor."""

    return TopK(k)


@dataclass(frozen=True)
class _Candidate:
    """One preflighted candidate: id, WHERE, resolution + execution terms.

    ``execution_where`` is the SPANNING-COORDINATE lowering (Build 0):
    label/Selection candidates are ENUMERATED however the user spelled them
    and EXECUTED as structural ``tl.site(key)`` terms, which are valid in
    every lane — the live lane never sees a finalized label.
    """

    candidate_id: str
    where: Any
    execution_where: Any
    resolved_site_count: int | None
    resolution_digest: str
    #: Resolved entries of a unit selection (a partial mask at some site); each
    #: executes as its structural site key plus the edit masked to its units.
    unit_entries: tuple[Any, ...] | None = None
    unit_digest: str = ""
    #: The resolved site labels (empty for a facet selector, which resolves
    #: through the intervention mutators, never post-hoc).
    labels: tuple[str, ...] = ()
    #: The edit this candidate executes (the shared edit for ``sweep_sites``;
    #: per cell when a grid carries an edit axis).
    edit: Any = None


def _candidate_pairs(candidates: Any) -> list[tuple[str, Any]]:
    """Normalize the candidate plan to ordered (id, where) pairs."""

    if isinstance(candidates, Mapping):
        pairs = [(str(key), value) for key, value in candidates.items()]
    elif isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        pairs = [(f"c{index}", value) for index, value in enumerate(candidates)]
    else:
        raise BundleExperimentError(
            "site_sweep candidates= must be a mapping of candidate_id -> site "
            f"or an ordered sequence of sites, got {type(candidates).__name__}",
            code="site_sweep_candidates_invalid",
            received_type=type(candidates).__name__,
        )
    if not pairs:
        raise BundleExperimentError(
            "site_sweep received an empty candidate plan; enumerate at least one candidate site.",
            code="site_sweep_candidates_invalid",
        )
    if "baseline" in {name for name, _ in pairs}:
        raise BundleExperimentError(
            "site_sweep reserves the member name 'baseline' for the un-edited "
            "member; rename the candidate.",
            code="site_sweep_candidates_invalid",
        )
    return pairs


def _require_resolved(candidate_id: str, where: Any, labels: tuple[str, ...]) -> None:
    """Refuse a candidate that resolves to no site (a silent no-op otherwise).

    A zero-site candidate would execute nothing and measure the baseline
    value under its own name: the quiet null result the engine exists to
    prevent, so it refuses typed before any candidate runs.
    """

    if labels:
        return
    raise BundleExperimentError(
        f"candidate {candidate_id!r} ({where!r}) resolves to no site on the "
        "baseline. Check the spelling against baseline.ops (labels renumber "
        "between captures) or pass a selector that matches.",
        code="site_sweep_candidate_unresolved",
        candidate_id=candidate_id,
        where=repr(where),
    )


def _resolve_candidate(baseline: Trace, candidate_id: str, where: Any) -> _Candidate:
    """Resolve one candidate's site count + canonical digest on the baseline.

    Resolution ladder: Selection/`__selection__` producers resolve to their
    touched-site families; a fully-scoped FacetSelector (facet name + head +
    module address) is structurally ONE site; everything else resolves
    through ``_resolve_sites``. An UNKNOWN count (an unscoped facet selector —
    the measured ``tl.head(5)`` 36-site broadcast) is treated as multi-site,
    never as one.
    """

    from .._selection.selection import Selection, Selector
    from .._selection.selectors import FacetSelector

    if isinstance(where, FacetSelector):
        # Checked BEFORE the Selection lift: facet selectors resolve through
        # the intervention mutators, never post-hoc site resolution.
        return _facet_candidate(candidate_id, where)
    lifted = where
    converter = getattr(lifted, "__selection__", None)
    if not isinstance(lifted, (Selector, Selection)) and callable(converter):
        lifted = converter()
    if isinstance(lifted, (Selector, Selection)):
        resolved = lifted._resolve(baseline) if isinstance(lifted, Selector) else lifted
        # ResolvedSelection is a Sequence of Site rows. Entry site keys
        # are (layer_label, pass_index) tuples; the structural L1 key rides
        # structural_site_key and the label backstop derives it.
        labels = tuple(str(entry.key[0]) for entry in resolved)
        _require_resolved(candidate_id, where, labels)
        parts = tuple(_unit_digest_part(entry) for entry in resolved)
        units_selected = any(isinstance(part, tuple) for part in parts)
        return _Candidate(
            candidate_id=candidate_id,
            where=where,
            execution_where=_site_key_term(baseline, candidate_id, labels),
            resolved_site_count=len(labels),
            resolution_digest=_digest(("selection",) + tuple(sorted(parts, key=repr))),
            # A unit selection's edit must reach only the selected units, exactly
            # as ``fork.do(selection, edit)`` masks it; whole-site entries keep
            # the plain structural lowering.
            unit_entries=tuple(resolved) if units_selected else None,
            unit_digest=str(getattr(resolved, "resolve_digest", "")),
            labels=labels,
        )
    return _sites_candidate(baseline, candidate_id, where)


def _facet_candidate(candidate_id: str, where: Any) -> _Candidate:
    """A facet selector: ONE site when fully scoped (name, head, module), else unknown."""

    fully_scoped = (
        where.module_address is not None and where.head_index is not None and where.name is not None
    )
    return _Candidate(
        candidate_id=candidate_id,
        where=where,
        execution_where=where,
        resolved_site_count=1 if fully_scoped else None,
        resolution_digest=_digest(("facet", where.name, where.head_index, where.module_address)),
    )


def _sites_candidate(baseline: Trace, candidate_id: str, where: Any) -> _Candidate:
    """Resolve a label string or structural spelling through ``_resolve_sites``."""

    from .._selection.resolver import _resolve_sites

    table = _resolve_sites(baseline, where, max_fanout=100_000)
    labels = tuple(str(getattr(site, "label", site)) for site in table)
    _require_resolved(candidate_id, where, labels)
    # A label string lowers to its structural site key (valid in every lane);
    # callable/selector spellings are structural already and execute as spelled.
    execution_where = (
        _site_key_term(baseline, candidate_id, labels) if isinstance(where, str) else where
    )
    return _Candidate(
        candidate_id=candidate_id,
        where=where,
        execution_where=execution_where,
        resolved_site_count=len(labels),
        resolution_digest=_digest(("sites",) + tuple(sorted(labels))),
        labels=labels,
    )


def _candidate_spec(baseline: Trace, candidate: _Candidate, edit: Any) -> Any:
    """The intervention spec one candidate executes: its WHERE term with the edit.

    A unit candidate gets one rule per resolved site: the site's structural key
    with the edit masked to the selected units by the same derivation
    ``fork.do(selection, edit)`` uses, so the sweep's member equals a direct
    fork of the same selection edit.
    """

    from ..intervention.spec import Recipe, when

    if candidate.unit_entries is None:
        return when(candidate.execution_where, edit)
    from ..intervention.masked_edit import _derive_masked_edit

    rules: list[Any] = []
    for entry in candidate.unit_entries:
        label = str(entry.key[0])
        term = _site_key_term(baseline, candidate.candidate_id, (label,))
        op_label = str(baseline.ops[label].label)
        masked = _derive_masked_edit(edit, entry, candidate.unit_digest, op_label)
        rules.extend(when(term, masked).rules)
    return Recipe(rules=tuple(rules))


def _unit_digest_part(entry: Any) -> Any:
    """Digest part of one resolved selection entry: its label, plus its units if partial.

    A whole-site entry contributes its label alone, so a whole-site unit
    selection stays the same candidate as the bare site. A partial mask also
    contributes its shape and a digest of its bytes, so distinct units at one
    site are distinct candidates and identical units are still duplicates.
    """

    label = str(entry.key[0])
    mask = entry.mask
    if bool(mask.all()):
        return label
    packed = mask.reshape(-1).cpu().numpy().tobytes()
    return (label, tuple(mask.shape), hashlib.sha256(packed).hexdigest())


def _site_key_term(baseline: Trace, candidate_id: str, labels: tuple[str, ...]) -> Any:
    """Lower enumerated labels to the OR of their structural site keys.

    The Build-0 spanning coordinate: labels renumber, site keys do not, and
    ``tl.site(key)`` is valid in every lane — so a label-enumerated candidate
    executes identically on live_hook and replay.
    """

    from .._selection.selectors import site as site_selector

    keys: list[str] = []
    for label in labels:
        op = baseline.ops[label]
        key = getattr(op, "site_key", None)
        if not key:
            raise BundleExperimentError(
                f"candidate {candidate_id!r} site {label!r} carries no "
                "structural site key (legacy capture); the engine executes in "
                "site-key coordinates so label-enumerated candidates need a "
                "keyed baseline. Re-capture with a current TorchLens.",
                code="site_sweep_site_key_unavailable",
                candidate_id=candidate_id,
                label=label,
            )
        keys.append(str(key))
    term: Any = site_selector(keys[0])
    for key in keys[1:]:
        term = term | site_selector(key)
    return term


def _digest(parts: tuple[Any, ...]) -> str:
    """Stable sha256 digest over the repr-joined parts (resolution identity)."""

    return hashlib.sha256("|".join(repr(part) for part in parts).encode("utf-8")).hexdigest()


def _member_bytes(member: Any) -> int:
    """Committed retained-activation bytes of one member (0 when unbudgeted)."""

    accountant = getattr(member, "_save_budget_accountant", None)
    ledgers = getattr(accountant, "ledgers", None) or {}
    return int(sum(int(getattr(ledger, "committed_bytes", 0)) for ledger in ledgers.values()))


def _run_live_candidate(
    baseline: Trace,
    candidate: _Candidate,
    spec: Any,
    *,
    model: Any,
    x: Any,
) -> tuple[Any, str]:
    """Run one candidate on the live_hook lane, with the DISCLOSED door split.

    Structural terms (facet selectors, func/in_module, predicates) ride the
    rerun door (fork + _attach_hooks + run). Site-key terms cannot be matched
    mid-attach (final structural numbering is absent), so they ride the
    CAPTURE door (``trace(model, x, intervene=spec)``, which arms the live
    site-key minter) — the same real forward with the edit installed, and the
    member's own envelope EVENT row stamps lane="capture"; escalation is
    never silent.
    """

    from ..intervention.errors import SelectorCapabilityError

    try:
        import time as _time

        member = baseline.fork(name=candidate.candidate_id)
        member._attach_hooks(spec)
        started = _time.monotonic()
        member.run(model, x)
        _record_run_envelope(member, spec, started=started)
        return member, "live_hook"
    except SelectorCapabilityError:
        from .._user_funcs import trace as _trace

        member = _trace(
            model,
            x,
            capture=_capture_options_like(baseline),
            intervene=spec,
        )
        return member, "capture"


def _record_run_envelope(member: Any, spec: Any, *, started: float) -> None:
    """Write the run-door transaction envelope (the historical fourth-door gap).

    ``_attach_hooks`` stages the spec and ``run()`` fires it, but the run door
    wrote no InterventionEvent before this engine — exactly the design memo's
    3.1 finding ("three of the four write doors record nothing"). The engine
    is the run door's first experiment customer, so it writes the envelope
    through the ONE shared writer with the fires this run minted.
    """

    from ..intervention.audit import (
        fire_records_since,
        record_intervention_event,
        rules_payload,
        site_keys_for_labels,
    )

    fires = fire_records_since(member, started)
    fired_labels = tuple(
        dict.fromkeys(record.call_label or record.target_label for record in fires)
    )
    rules = rules_payload(spec)
    record_intervention_event(
        member,
        lane="live_hook",
        door="run",
        edit_names=tuple(str(rule["action"]) for rule in rules),
        selection_repr=" | ".join(str(rule["where"]) for rule in rules),
        status="fired" if fires else "no_fire",
        fire_count=len(fires),
        site_keys=site_keys_for_labels(member, fired_labels),
        rules=rules,
        append_audit_row=False,
        audit_event_row=True,
    )


def _capture_options_like(baseline: Trace) -> Any:
    """Capture options for a capture-door candidate (intervention-ready)."""

    from ..options import CaptureOptions

    return CaptureOptions(replayable=True)


def _measure(metric: Callable[[Any], Any], member: Any, candidate_id: str) -> float:
    """Apply the user metric to one member and refuse non-finite results typed."""

    value = metric(member)
    number = float(value)
    if not math.isfinite(number):
        raise BundleExperimentError(
            f"site_sweep metric returned a non-finite value {value!r} for "
            f"candidate {candidate_id!r}; the metric must return one finite "
            "real scalar per member.",
            code="site_sweep_metric_invalid",
            candidate_id=candidate_id,
        )
    return number


def _validate_sweep_entry(
    *,
    engine: str,
    metric: Any,
    retain: Any,
    model: nn.Module | None,
    x: Any,
) -> None:
    """Typed entry refusals (D3b: edit/metric/retain have NO defaults)."""

    if engine not in _ENGINE_LANES:
        raise BundleExperimentError(
            f"site_sweep engine {engine!r} is outside the closed lane "
            f"vocabulary {list(_ENGINE_LANES)}",
            code="site_sweep_engine_invalid",
            engine=engine,
        )
    if not callable(metric):
        raise BundleExperimentError(
            "site_sweep metric= must be a callable(member_trace) -> scalar; "
            "the metric IS the experiment and has no default.",
            code="site_sweep_metric_invalid",
        )
    if not (retain in ("all", "none") or isinstance(retain, TopK)):
        raise BundleExperimentError(
            f"site_sweep retain= must be 'all', 'none', or top_k(k); got "
            f"{retain!r}. Retention decides what evidence is destroyed and "
            "has no default.",
            code="site_sweep_retain_invalid",
            received=repr(retain),
        )
    if engine == "live_hook" and (model is None or x is None):
        raise BundleExperimentError(
            "site_sweep on the live_hook lane runs a REAL forward per "
            "candidate and needs model= and inputs=; pass both, or use "
            "engine='replay' to recompute from saved payloads.",
            code="site_sweep_inputs_missing",
            engine=engine,
        )


def _declared_multi_ids(pairs: list[tuple[str, Any]], multi_site: Any) -> set[str]:
    """The candidate ids declared multi-site (``True`` = every candidate)."""

    if multi_site is True:
        return {str(name) for name, _ in pairs}
    if isinstance(multi_site, Collection) and not isinstance(multi_site, (str, bytes)):
        return {str(name) for name in multi_site}
    return set()


def _edit_identity(edit: Any) -> str:
    """Identity of an edit for duplicate detection (helper edits by content)."""

    from ..intervention.types import Edit

    if isinstance(edit, Edit):
        return _digest(("edit", edit.name, repr(edit.args), repr(edit.kwargs)))
    return _digest(("edit", repr(edit)))


def _preflight_candidates(
    baseline: Trace,
    plan: list[tuple[str, Any, Any]],
    declared_multi: set[str],
    replicates: bool,
) -> tuple[list[_Candidate], list[tuple[_Candidate, str]]]:
    """Resolve every candidate ONCE, before any candidate runs.

    ``plan`` rows are ``(candidate_id, where, edit)``. Undeclared multi-site
    candidates become REFUSED rows; undeclared duplicate resolutions refuse
    the whole sweep typed.
    """

    resolved: list[_Candidate] = []
    refused: list[tuple[_Candidate, str]] = []
    for candidate_id, where, edit in plan:
        candidate = replace(_resolve_candidate(baseline, candidate_id, where), edit=edit)
        reason = _multi_site_reason(candidate, declared_multi)
        if reason is not None:
            refused.append((candidate, reason))
            continue
        resolved.append(candidate)
    _refuse_duplicates(resolved, replicates)
    return resolved, refused


def _multi_site_reason(candidate: _Candidate, declared_multi: set[str]) -> str | None:
    """The refusal text for an undeclared multi-site candidate, else None."""

    count = candidate.resolved_site_count
    if (count is None or count > 1) and candidate.candidate_id not in declared_multi:
        shown = "unprovable" if count is None else str(count)
        return (
            f"resolved_site_count={shown} without a multi_site "
            "declaration (an undeclared broadcast is the measured "
            "tl.head(5) 36-site defect class)"
        )
    return None


def _refuse_duplicates(resolved: list[_Candidate], replicates: bool) -> None:
    """Refuse two candidates with one identity (sites, units AND edit) typed."""

    seen: dict[tuple[str, str], str] = {}
    for candidate in resolved:
        key = (candidate.resolution_digest, _edit_identity(candidate.edit))
        prior = seen.get(key)
        if prior is not None and not replicates:
            raise BundleExperimentError(
                f"site_sweep candidates {prior!r} and {candidate.candidate_id!r} "
                "resolve to the same sites and units; pass replicates=True if the "
                "duplication is a deliberate replicate.",
                code="site_sweep_duplicate_candidates",
                first=prior,
                second=candidate.candidate_id,
            )
        seen.setdefault(key, candidate.candidate_id)


def _preflight_retention(
    baseline: Trace,
    retain: Any,
    retain_ceiling_bytes: int | None,
    n_resolved: int,
) -> None:
    """Byte preflight BEFORE the first candidate runs (typed refusal)."""

    if retain_ceiling_bytes is None or retain == "none":
        return
    per_candidate_bytes = _member_bytes(baseline)
    preflight_retention_projection(
        per_candidate_bytes=per_candidate_bytes,
        retained_candidates=(
            retain.k if isinstance(retain, TopK) else n_resolved if retain == "all" else 1
        ),
        ceiling_bytes=int(retain_ceiling_bytes),
        current_bytes=per_candidate_bytes,  # the baseline member itself
        operation="site_sweep",
    )


@dataclass
class _SweepState:
    """Mutable per-sweep accumulator: bundle, effect rows, retention."""

    bundle: Bundle
    baseline: Trace
    lane: str
    edit_repr: str
    metric: Callable[[Any], Any]
    model: nn.Module | None
    x: Any
    retain: Any
    baseline_value: float
    attempted: int
    extra_params: dict[str, Any] = field(default_factory=dict)
    rows: list[MemberEffectRow] = field(default_factory=list)
    completed_rows: dict[str, MemberEffectRow] = field(default_factory=dict)
    retained: list[tuple[float, str]] = field(default_factory=list)
    session_selections: dict[str, Any] = field(default_factory=dict)

    def release(self, name: str) -> None:
        """Remove one retained member from the bundle and free its payloads."""

        member = self.bundle._members.get(name)
        if member is not None:
            self.bundle.remove(name)
            member.cleanup()

    def record_refusal(self, candidate: _Candidate, reason: str) -> None:
        """Append a REFUSED effect row (the candidate never executed)."""

        self.rows.append(
            MemberEffectRow(
                candidate_id=candidate.candidate_id,
                status="refused",
                resolved_site_count=candidate.resolved_site_count,
                error=reason,
            )
        )

    def record_failure(self, candidate: _Candidate, exc: BaseException, member: Any) -> None:
        """Append a FAILED effect row and free the partial member, if any."""

        self.rows.append(
            MemberEffectRow(
                candidate_id=candidate.candidate_id,
                status="failed",
                resolved_site_count=candidate.resolved_site_count,
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        if member is not None:
            member.cleanup()

    def retain_or_release(
        self, candidate: _Candidate, member: Any, value: float, delta: float
    ) -> None:
        """Record a completed candidate, retaining or releasing per the policy."""

        row = MemberEffectRow(
            candidate_id=candidate.candidate_id,
            status="completed",
            member_name=candidate.candidate_id,
            resolved_site_count=candidate.resolved_site_count,
            value=value,
            retained=True,
        )
        if self.retain == "none":
            member.cleanup()
            self.rows.append(
                MemberEffectRow(
                    candidate_id=candidate.candidate_id,
                    status="released",
                    resolved_site_count=candidate.resolved_site_count,
                    value=value,
                    retained=False,
                )
            )
            return
        self.bundle.add(member, names=candidate.candidate_id)
        self.bundle._member_construction[candidate.candidate_id] = {
            "origin": "swept",
            "source_member": "baseline",
        }
        self.completed_rows[candidate.candidate_id] = row
        self.retained.append((delta, candidate.candidate_id))
        if isinstance(self.retain, TopK) and len(self.retained) > self.retain.k:
            self._evict_smallest()

    def _evict_smallest(self) -> None:
        """Evict the smallest-|effect| retained member (rolling top-k, disclosed)."""

        # Rolling top-k by |effect|: evict the smallest, disclosed.
        self.retained.sort(key=lambda item: (item[0], item[1]))
        _evict_delta, evict_id = self.retained.pop(0)
        self.release(evict_id)
        evicted = self.completed_rows.pop(evict_id)
        self.rows.append(
            MemberEffectRow(
                candidate_id=evicted.candidate_id,
                status="released",
                resolved_site_count=evicted.resolved_site_count,
                value=evicted.value,
                retained=False,
            )
        )
        self.session_selections.pop(evict_id, None)


def _execute_and_measure(state: _SweepState, candidate: _Candidate, spec: Any) -> tuple[Any, float]:
    """Run one candidate on the stamped lane; EXTRACT BEFORE CLEANUP.

    The metric, the byte figure, and the envelope EVENT row are read while
    the member is alive (the measured engine contract).
    """

    from ..intervention.audit import append_event_audit_row, event_audit_rows

    if state.lane == "live_hook":
        member, actual_lane = _run_live_candidate(
            state.baseline, candidate, spec, model=state.model, x=state.x
        )
        del actual_lane
    else:
        member = state.baseline.fork(name=candidate.candidate_id)
        member.do(spec)
    value = _measure(state.metric, member, candidate.candidate_id)
    envelope_rows = [
        row
        for row in getattr(member, "state_history", ())
        if isinstance(row, dict) and row.get("op") == "intervention_event"
    ]
    if envelope_rows and not event_audit_rows(member):
        append_event_audit_row(member, envelope_rows[-1])
    return member, value


def _record_sweep_operation(state: _SweepState) -> Any:
    """Record the ONE chronology row and stamp retained anchors with its id."""

    released = sum(1 for r in state.rows if r.status == "released")
    operation = state.bundle._record_bundle_operation(
        "site_sweep",
        member_names=tuple(state.bundle._members),
        params={
            "lane": state.lane,
            "retain": repr(state.retain),
            "attempted": state.attempted,
            "completed": len(state.completed_rows) + released,
            "retained": len(state.completed_rows),
            "released": released,
            "refused": sum(1 for r in state.rows if r.status == "refused"),
            "failed": sum(1 for r in state.rows if r.status == "failed"),
            "edit_repr": state.edit_repr,
            "baseline_value": state.baseline_value,
            **state.extra_params,
        },
    )
    for name in state.completed_rows:
        state.bundle._member_construction[name]["operation_id"] = operation.operation_id
    return operation


def sweep_sites(
    baseline: Trace,
    *,
    candidates: Any,
    edit: Any,
    metric: Callable[[Any], Any],
    retain: Any,
    engine: str = "live_hook",
    model: nn.Module | None = None,
    inputs: Any = None,
    multi_site: bool | Collection[str] = False,
    replicates: bool = False,
    retain_ceiling_bytes: int | None = None,
) -> Bundle:
    """Ablate every candidate site, one member per candidate, baseline first.

    Parameters
    ----------
    baseline:
        The un-edited capture (``replayable``); becomes the
        ``"baseline"`` member and the comparison anchor.
    candidates:
        Ordered candidate plan: ``{candidate_id: site}`` or a sequence of
        sites (ids minted ``c0..cN``). A site is a Selection / region
        producer / scoped facet selector / label / selector. A multi-site
        Selection is ONE candidate knockout set only when DECLARED via
        ``multi_site=``.
    edit:
        REQUIRED. The edit applied at each candidate site (which ablation is
        sound is a live scientific argument — no default).
    metric:
        REQUIRED. ``callable(member_trace) -> finite real scalar`` — the
        metric IS the experiment; readouts resolve to SITE TENSORS on the
        member (``reconstruct_output`` is not in this path).
    retain:
        REQUIRED. ``"all"`` | ``"none"`` | ``top_k(k)`` — retention decides
        what evidence is DESTROYED. Pair any policy with
        ``retain_ceiling_bytes=`` for the typed BEFORE-the-first-candidate
        byte preflight.
    engine:
        ``"live_hook"`` (default; fork + _attach_hooks + run(model, x) — a
        real forward) or ``"replay"`` (fork + do on saved payloads). The
        lane is stamped in the envelope and the effect table.
    model, x:
        REQUIRED on the live_hook lane (the real forward's module and
        input); unused on replay.
    multi_site:
        ``True`` (every candidate may span sites) or a collection of
        candidate ids declared multi-site. An undeclared candidate whose
        resolved site count exceeds one — or cannot be proven one — REFUSES.
    replicates:
        Whether duplicate resolved candidates are deliberate replicates.
    retain_ceiling_bytes:
        Optional byte ceiling for the retention preflight and rolling
        release (lower-bound basis).

    Returns
    -------
    Bundle
        Baseline + retained members; the COMPLETE per-candidate effect table
        (released/refused/failed rows included) persists in the bundle
        artifact keyed by operation id and survives discard/save/load.

    Examples
    --------
    >>> base = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    >>> bundle = tl.experiment.sweep_sites(
    ...     base, candidates={"relu": "relu_1_2"}, edit=tl.zero_ablate(),
    ...     metric=lambda t: float(t[t.output_layers[0]].out.sum()), retain="all",
    ...     model=model, inputs=x)
    >>> bundle.names
    ['baseline', 'relu']
    """

    _validate_sweep_entry(engine=engine, metric=metric, retain=retain, model=model, x=inputs)

    # ---- preflight (once, before any candidate runs) ----------------------
    pairs = _candidate_pairs(candidates)
    declared_multi = _declared_multi_ids(pairs, multi_site)
    plan = [(candidate_id, where, edit) for candidate_id, where in pairs]
    resolved, refused = _preflight_candidates(baseline, plan, declared_multi, replicates)
    run = _SweepRun(
        attempted=len(pairs),
        metric=metric,
        retain=retain,
        engine=engine,
        model=model,
        x=inputs,
        retain_ceiling_bytes=retain_ceiling_bytes,
        edit_repr=repr(edit),
    )
    return _run_plan(baseline, resolved, refused, run)


@dataclass(frozen=True)
class _SweepRun:
    """The run settings one sweep face hands the shared engine.

    ``extra_params`` ride the recorded ``site_sweep`` operation (a grid's
    axes), so a loaded artifact can rebuild the caller's shape.
    """

    attempted: int
    metric: Callable[[Any], Any]
    retain: Any
    engine: str
    model: nn.Module | None
    x: Any
    retain_ceiling_bytes: int | None
    edit_repr: str
    extra_params: Mapping[str, Any] | None = None


def _run_plan(
    baseline: Trace,
    resolved: list[_Candidate],
    refused: list[tuple[_Candidate, str]],
    run: _SweepRun,
) -> Bundle:
    """Execute a preflighted plan: the ONE engine behind both sweep faces.

    ``resolved`` candidates run serially in order, each with its own
    ``edit``; ``refused`` candidates land as refused rows without running.
    """

    _preflight_retention(baseline, run.retain, run.retain_ceiling_bytes, len(resolved))
    state = _open_sweep(baseline, run)
    baseline_value = state.baseline_value

    # ---- serial execution, stable order -----------------------------------
    for candidate, reason in refused:
        state.record_refusal(candidate, reason)
    for candidate in resolved:
        spec = _candidate_spec(baseline, candidate, candidate.edit)
        member: Any = None
        try:
            member, value = _execute_and_measure(state, candidate, spec)
        except BundleExperimentError:
            raise
        except Exception as exc:  # noqa: BLE001 - per-candidate outcome, disclosed
            state.record_failure(candidate, exc, member)
            continue
        delta = abs(value - baseline_value)
        state.session_selections[candidate.candidate_id] = candidate.where
        state.retain_or_release(candidate, member, value, delta)
    state.rows.extend(state.completed_rows.values())
    return _close_sweep(state)


def _close_sweep(state: _SweepState) -> Bundle:
    """Record the operation and persist its effect table on the Bundle."""

    operation = _record_sweep_operation(state)
    table = MemberEffectTable(
        operation_id=operation.operation_id,
        rows=tuple(state.rows),
        baseline_member="baseline",
        metric_repr=getattr(state.metric, "__qualname__", repr(state.metric)),
        edit_repr=state.edit_repr,
        lane=state.lane,
        retain_policy=repr(state.retain),
    )
    bundle = state.bundle
    bundle._effect_tables[operation.operation_id] = table
    bundle._session_effect_selections = dict(state.session_selections)  # type: ignore[attr-defined]
    return bundle


def _open_sweep(baseline: Trace, run: _SweepRun) -> _SweepState:
    """Measure the baseline and open the Bundle + state every sweep row joins."""

    baseline_value = _measure(run.metric, baseline, "baseline")
    bundle = Bundle({"baseline": baseline}, baseline="baseline")
    state = _SweepState(
        bundle=bundle,
        baseline=baseline,
        lane=run.engine,
        edit_repr=run.edit_repr,
        metric=run.metric,
        model=run.model,
        x=run.x,
        retain=run.retain,
        baseline_value=baseline_value,
        attempted=run.attempted,
        extra_params=dict(run.extra_params or {}),
    )
    state.rows.append(
        MemberEffectRow(
            candidate_id="__baseline__",
            status="completed",
            member_name="baseline",
            resolved_site_count=0,
            value=baseline_value,
            retained=True,
        )
    )
    return state
