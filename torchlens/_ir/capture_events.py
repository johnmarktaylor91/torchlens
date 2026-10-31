"""Mutable capture event accumulator for one forward pass."""

from __future__ import annotations

import itertools
import warnings
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from ..errors._base import TorchLensWarning
from .events import (
    BackwardCoverageGap,
    BackwardPassEnd,
    BackwardPassStart,
    BufferWriteEvent,
    CheckpointInvocationObserved,
    GradFnDiscovered,
    GradFnFired,
    InterventionAppliedEvent,
    ModuleEnterEvent,
    ModuleExitEvent,
    ModulePrepEvent,
    OpEvent,
    OpGradObserved,
    OutputVersionEvent,
    ParamGradObserved,
    PreHookProvenanceEvent,
    _AtenCallEvent,
    _ModePausedInteriorEvent,
)
from .live_index import LiveIndex
from .op_record import (
    OpAmendment,
    OpRecord,
    apply_patch_items,
    validate_amendment,
)
from .predicate import RecordContext
from .refs import ParamRef, ReservedLabel

# Op-record count at which a running capture DISCLOSES its growth exactly
# once (r8 R60-1). Purely informational -- capture continues -- sized from
# the measured ~31KB/op journal+row cost (a 50k-op forward holds roughly
# 1.5GB of op records before postprocess).
OP_COUNT_DISCLOSURE_THRESHOLD = 50_000

# The journal op lane's record union: compat flat events (legacy producer,
# preview backends) and decomposed records (torch decomposed producer). The
# compat member dies in S15.
JournalOp = OpEvent | OpRecord


# Declared merge law: how each journal lane combines when one run's stream is
# folded into an accumulating journal (multi-pass recording, failed-partial
# recovery). ``CaptureEvents.concat`` is the ONLY sanctioned way to combine two
# streams; ad-hoc lane splicing is forbidden.
#
# - ``append_restamp``: events join the target journal and are re-stamped into
#   its sequence domain by the single writer.
# - ``first_run_only``: merged only while the target lane is empty (module
#   structure repeats identically per pass; one non-duplicated set is kept).
# - ``run_local``: never merged — the lane's facts are scoped to their own run
#   (per-pass replay snapshots, buffer writes predicate capture does not track,
#   and backward events, which append to the ACCUMULATING stream directly).
#
# Dict order is the stamping order for one concat call.
LANE_MERGE_POLICIES: dict[str, str] = {
    "module_prep_events": "first_run_only",
    "module_enter_events": "first_run_only",
    "module_exit_events": "first_run_only",
    "pre_hook_events": "append_restamp",
    "op_events": "append_restamp",
    "intervention_events": "append_restamp",
    "op_amendments": "append_restamp",
    "output_version_events": "run_local",
    "buffer_write_events": "run_local",
    "backward_events": "run_local",
    "aten_events": "append_restamp",
}

# Lanes whose events carry a ``target_seq`` reference into the op lane's seq
# domain: concat rebinds those references through the merge seq map so a
# genuinely-bound record follows its target into the combined journal (DoR
# 4.6). Any new target-carrying lane MUST register here or its references
# dangle silently after a merge.
REBINDABLE_TARGET_LANES: frozenset[str] = frozenset({"intervention_events", "op_amendments"})

_LANE_APPENDERS: dict[str, str] = {
    "op_events": "append",
    "module_prep_events": "append_module_prep",
    "module_enter_events": "append_module_enter",
    "module_exit_events": "append_module_exit",
    "pre_hook_events": "append_pre_hook",
    "intervention_events": "append_intervention",
    "op_amendments": "append_amendment",
    "aten_events": "append_aten",
}


class SealedJournalWriteError(RuntimeError):
    """Base for every refused write to a sealed source journal.

    ``CaptureSession.seal()`` freezes a journal into its ``CapturedRunCore``:
    the op spine is snapshotted (folded through the amendment reducer) and the
    forward sibling lanes are snapshotted onto the pre-seal projection clone.
    A write to any of those lanes AFTER the seal silently diverges the live
    journal from the artifact every consumer projects from, so the whole class
    fails closed here rather than at the far end of a projection.
    """


class SealedJournalAmendmentError(SealedJournalWriteError):
    """A typed amendment was appended to a sealed source journal.

    After ``CaptureSession.seal()`` folds a journal into its
    ``CapturedRunCore``, the sealed source refuses further amendment appends
    fail-closed (DoR 4.5.5): post-seal knowledge must route through a working
    projection (``copy_for_replay``), whose carried amendment lane the
    watermark filter keeps double-application-free.
    """


class SealedJournalAppendError(SealedJournalWriteError):
    """A forward-lane event was appended to a sealed source journal.

    Finding B1-15: sealed-journal enforcement was HALF-HEIGHT -- only the
    amendment lane checked ``amendments_sealed``, while the op lane's
    :meth:`CaptureEvents.append` and every forward sibling appender wrote
    unconditionally. Code holding a sealed journal could therefore diverge the
    live stream from its sealed ``CapturedRunCore`` snapshot with no signal at
    all. The op lane and the forward sibling lanes are now the same fence the
    amendment lane always had; post-seal work belongs on a working projection
    (``copy_for_replay``).

    The backward lane is deliberately NOT fenced: backward capture runs after
    the forward seal by design (``Trace.log_backward`` /
    ``Recording.log_backward`` both append to the sealed journal), and
    ``backward_events`` is not part of the sealed core. Measured: across the
    smoke tier plus the fastlog/producer suites, ``append_backward`` is the ONLY
    appender that ever fires on a sealed journal.
    """


class AmendmentTargetError(RuntimeError):
    """An amendment's target cannot be resolved in this journal.

    Raised when ``target_label_raw`` names no committed op, or (in a
    single-seq-domain journal) when the resolved op's ``seq`` disagrees with
    ``target_seq``. Multi-domain journals (multi-pass fastlog projections have
    no single seq domain) resolve by label only, so the seq cross-check is
    skipped there by design (reviewer note O-N6).
    """


class LaneMergePolicyError(RuntimeError):
    """A lane declares a merging policy but has no registered appender.

    Raised by :meth:`CaptureEvents.concat` BEFORE any event moves, so a
    declared-but-unwired lane fails closed instead of silently skipping (or
    crashing halfway through a merge and corrupting the target journal).
    """


class SourceSequencingError(RuntimeError):
    """A concat source's global seq domain is invalid.

    Raised by :meth:`CaptureEvents.concat` BEFORE any event moves when the
    source stream holds an unstamped event, a duplicate cross-lane seq, a
    non-monotone lane, or a stamp beyond the source's writer counter. Sorting
    such a stream on its seq domain would substitute dict-lane-order
    tie-breaking for real chronology, laundering a producer defect (an
    unstamped, duplicate, reordered, or counter-bypassing writer) into a
    merged journal that then passes the seq invariants the source itself
    would have failed.
    """


# Process-monotonic run-nonce source: every CaptureEvents stream is one
# capture run's journal, and intervention-edit records are causally bound to
# their run through this token (streams never serialize, so an in-process
# counter is collision-free for the token's whole lifetime).
_RUN_NONCE_COUNTER = itertools.count(1)


def _clone_op_event_for_replay(event: Any) -> Any:
    """Return a projection copy of ``event`` with independent mutable state.

    ``OpEvent`` is a frozen dataclass, but two of its fields are live dicts:
    ``transform_config`` and ``parent_arg_positions``. A ``copy_for_replay``
    projection must not be able to mutate those dicts on the sealed source
    stream, so they are duplicated here. Every other field is immutable (or an
    intentionally shared tensor payload / opaque handle), so the clone stays
    cheap and never copies activations.

    Polymorphic from P3 (reviewer note O-N5): a decomposed ``OpRecord`` clones
    the facets that carry live dicts (``graph.parent_arg_positions``,
    ``transform.transform_config``, ``annotations.annotations``) — the same
    shared-mutable-state guarantee, NOT a blanket identity return, because the
    record's facets are frozen but those three payloads are not. The clone
    ALWAYS owns a fresh ``core``: the journal ``append`` path stamps ``seq``
    on the record's core (the one slot append mutates), so a shared core
    would let re-stamping a merged clone silently rewrite the sealed source
    stream's seq facts.

    Parameters
    ----------
    event
        Sealed source operation event or decomposed op record.

    Returns
    -------
    OpEvent | OpRecord
        Event/record whose reachable mutable dicts are independent copies.
    """

    if isinstance(event, OpRecord):
        record_changes: dict[str, Any] = {"core": replace(event.core)}
        graph = event.graph
        if graph is not None:
            record_changes["graph"] = replace(
                graph,
                parent_arg_positions={
                    domain: dict(positions)
                    for domain, positions in graph.parent_arg_positions.items()
                },
            )
        transform = event.transform
        if transform is not None:
            record_changes["transform"] = replace(
                transform, transform_config=dict(transform.transform_config)
            )
        annotations_facet = event.annotations_facet
        if annotations_facet is not None:
            record_changes["annotations_facet"] = replace(
                annotations_facet, annotations=dict(annotations_facet.annotations)
            )
        return replace(event, **record_changes)
    return replace(
        event,
        parent_arg_positions={
            domain: dict(positions) for domain, positions in event.parent_arg_positions.items()
        },
        transform_config=dict(event.transform_config),
    )


@dataclass(slots=False)
class CaptureEvents:
    """Mutable event buffer allocated once per capture."""

    op_events: list[JournalOp] = field(default_factory=list)
    module_prep_events: list[ModulePrepEvent] = field(default_factory=list)
    module_enter_events: list[ModuleEnterEvent] = field(default_factory=list)
    module_exit_events: list[ModuleExitEvent] = field(default_factory=list)
    pre_hook_events: list[PreHookProvenanceEvent] = field(default_factory=list)
    output_version_events: list[OutputVersionEvent] = field(default_factory=list)
    buffer_write_events: list[BufferWriteEvent] = field(default_factory=list)
    intervention_events: list[InterventionAppliedEvent] = field(default_factory=list)
    backward_events: list[
        BackwardPassStart
        | OpGradObserved
        | ParamGradObserved
        | BackwardPassEnd
        | GradFnDiscovered
        | GradFnFired
        | BackwardCoverageGap
        | CheckpointInvocationObserved
    ] = field(default_factory=list)
    aten_events: list[_AtenCallEvent | _ModePausedInteriorEvent] = field(default_factory=list)
    aten_recording_enabled: bool = False
    # Typed post-commit knowledge lane (producer unification P4): the op lane
    # is genuinely append-only and every post-commit mutation is an
    # ``OpAmendment`` folded by the canonical reducer. Amendments are stamped
    # from their OWN per-journal monotone counter (``amendment_seq``), NOT the
    # shared event counter: their seq domain is lane-local so that appending
    # an amendment never shifts the Tier-F ``seq`` facts on subsequent events
    # (the temporal byte-identity baselines pin those), while the watermark
    # filter and fold order only ever need lane-local monotonicity.
    op_amendments: list[OpAmendment] = field(default_factory=list)
    amendment_seq: int = 0
    # Highest amendment seq folded into this journal's sealed core, stamped by
    # ``CaptureSession.seal()`` (and mirrored onto the pre-seal projection
    # clone — reviewer note S-N2). ``copy_for_replay`` preserves it and, when
    # already-folded events seed the copy, filters the carried lane to
    # ``seq > core_seal_watermark`` so double-application is impossible.
    core_seal_watermark: int | None = None
    # A sealed SOURCE journal refuses further amendment appends fail-closed;
    # working projections reset this and accept appends.
    amendments_sealed: bool = False
    # False only for projections seeded from multiple concatenated seq
    # domains (multi-pass fastlog); gates the target_seq cross-check (O-N6).
    single_seq_domain: bool = True
    _amended_fold_cache: tuple[tuple[int, int], list[JournalOp], dict[str, JournalOp]] | None = (
        field(default=None, repr=False)
    )
    param_refs: dict[str, ParamRef] = field(default_factory=dict)
    raw_layer_counter: int = 0
    raw_layer_type_counter: dict[str, int] = field(default_factory=dict)
    func_call_id_counter: int = 0
    recent_events: deque[RecordContext] = field(default_factory=deque)
    backend_session: object | None = None
    live_index: LiveIndex = field(default_factory=LiveIndex)
    grad_fn_handles_by_label_raw: dict[str, Any] = field(default_factory=dict)
    # ONE run-monotonic sequence counter spanning every event kind and phase
    # (forward ops, module/prehook/output-version siblings, buffer writes, and
    # the whole backward family). The append methods below are the single
    # sequencing authority: every event receives ``seq`` at append time, so
    # cross-kind and forward/backward ordering is an exact recorded fact.
    event_seq: int = 0
    backward_revision: int = 0
    # Run identity token for causal binding of intervention-edit records: the
    # observing site stamps it onto each edit, and validation accepts an edit
    # only when its token matches the validated stream's nonce. Working
    # projections of the SAME run (``copy_for_replay``) preserve the nonce;
    # detached streams and fresh captures get their own.
    run_nonce: int = field(default_factory=lambda: next(_RUN_NONCE_COUNTER))
    # Detached-stream baseline: event streams never serialize and forks never
    # share a stream, so a stream installed on a trace that ALREADY carries a
    # materialized backward projection records the projection it extends.
    # ``pass_index_base`` is the number of backward passes materialized before
    # this stream existed; every event appended here carries a strictly
    # greater pass index, and the projection/invariant layers treat passes at
    # or below the base as preserved facts outside this stream's window. The
    # ``base_*`` constants seed the cumulative counters a full scratch rebuild
    # would otherwise recompute from the (dropped) pre-detach events. All five
    # are written only by :meth:`detached_from` at the two detach sites
    # (pickle restore and fork) and stay 0/empty for live capture streams.
    pass_index_base: int = 0
    base_total_gradient_memory: int = 0
    base_total_backward_memory: int = 0
    base_saved_grad_labels: frozenset[str] = frozenset()
    base_root_grad_fn_object_ids: tuple[int, ...] = ()

    @classmethod
    def detached_from(cls, trace: Any) -> CaptureEvents:
        """Return a fresh stream extending ``trace``'s materialized projection.

        Used when a trace keeps its portable backward projection but must
        drop or replace its event stream (pickle restore, ``Trace.fork()``).
        The new stream starts empty with the projection baseline recorded so
        later full rebuilds preserve the pre-detach passes instead of
        silently erasing them, and so backward pass numbering stays dense
        from ``pass_index_base + 1`` within this stream.

        Parameters
        ----------
        trace
            Trace whose current backward projection this stream extends.

        Returns
        -------
        CaptureEvents
            Empty event buffer carrying the projection baseline.
        """

        return cls(
            pass_index_base=int(getattr(trace, "num_backward_passes", 0) or 0),
            base_total_gradient_memory=int(getattr(trace, "total_gradient_memory", 0) or 0),
            base_total_backward_memory=int(getattr(trace, "total_backward_memory", 0) or 0),
            base_saved_grad_labels=frozenset(getattr(trace, "_saved_grad_labels", ()) or ()),
            base_root_grad_fn_object_ids=tuple(
                getattr(trace, "backward_root_grad_fn_object_ids", ()) or ()
            ),
        )

    @property
    def op_event_by_label_raw(self) -> dict[str, OpEvent]:
        """Return the label lookup derived from the shared live index.

        Returns
        -------
        dict[str, OpEvent]
            Canonical live label-to-event mapping.
        """

        return self.live_index.by_raw_label

    @op_event_by_label_raw.setter
    def op_event_by_label_raw(self, events_by_label: dict[str, OpEvent]) -> None:
        """Replace the canonical label lookup and synchronize live edges.

        Parameters
        ----------
        events_by_label
            Replacement mapping, normally derived from ``op_events`` by a
            compatibility projector.
        """

        self.live_index.by_raw_label = events_by_label
        self.live_index.labels = list(events_by_label)
        self.live_index.rebuild_edges()

    def copy_for_replay(
        self,
        *,
        projected_op_events: Iterable[OpEvent] | None = None,
    ) -> CaptureEvents:
        """Return a structural working projection for postprocess mutation.

        Postprocess consumes the journal through an independent projection so no
        step can mutate the sealed capture lanes: the op lane is append-only
        (P4 deleted in-place entry replacement -- post-commit knowledge rides
        the typed amendment lane), and a projection is where a step may still
        append, drop, or clear its own working containers. This is also used
        when a long-lived, frozen ``Recording`` cooks itself into a ``Trace`` so
        repeated projections cannot alter its event stream.

        Every mutable container is duplicated into a fresh object (nested list
        values included where they are rebuilt in place). The ``OpEvent`` objects
        are re-created with independent copies of their mutable dict fields
        (``transform_config`` and ``parent_arg_positions``) so a projection can
        never mutate those dicts on the sealed source stream; the same cloned
        events back ``op_events``, ``op_event_by_label_raw``, and the projected
        ``live_index`` so the projection stays internally consistent. Tensor
        payloads and every other (immutable) event field are shared by
        reference, so the copy is cheap and does not clone activations. Scalars
        and the opaque ``backend_session`` are copied by value / reference.

        A caller that already projected independent ``OpEvent`` objects may
        provide them through ``projected_op_events``. Those events are installed
        directly while every mutable container and live-index lane is still
        copied. The caller must own independent ``transform_config`` and
        ``parent_arg_positions`` dictionaries on each supplied event.

        Parameters
        ----------
        projected_op_events
            Already-projected operation events whose mutable dictionaries are
            independent from this source stream. ``None`` clones this source's
            operation events as usual.

        Returns
        -------
        CaptureEvents
            Independent event buffer over the same underlying events.
        """

        if projected_op_events is None:
            replay_op_events = [_clone_op_event_for_replay(event) for event in self.op_events]
            # Same-run structural copy of a possibly-unfolded journal: the
            # carried amendment lane moves whole so the reducer keeps folding
            # the same knowledge on the copy.
            replay_amendments = list(self.op_amendments)
            replay_single_domain = self.single_seq_domain
        else:
            replay_op_events = list(projected_op_events)
            # Already-folded events seed the copy (a sealed core's fold, or a
            # multi-pass concatenation of sealed folds). The carried lane is
            # filtered to seq > core_seal_watermark so knowledge the seal
            # already folded can never apply twice (DoR 4.5.3.ii); an unsealed
            # source (no watermark) carries everything — its seeds are raw.
            watermark = self.core_seal_watermark
            if watermark is None:
                replay_amendments = list(self.op_amendments)
            else:
                replay_amendments = [
                    amendment for amendment in self.op_amendments if amendment.seq > watermark
                ]
            seed_seqs = [int(getattr(event, "seq", 0) or 0) for event in replay_op_events]
            replay_single_domain = all(
                later > earlier for earlier, later in zip(seed_seqs, seed_seqs[1:])
            )
        replay_by_label = {event.label_raw: event for event in replay_op_events}
        projected_index = self.live_index.copy()
        if projected_op_events is None:
            projected_index.by_raw_label = {
                label: replay_by_label.get(label, event)
                for label, event in projected_index.by_raw_label.items()
            }
        else:
            projected_index.by_raw_label = dict(replay_by_label)
            projected_index.labels = list(replay_by_label)
            projected_index.rebuild_edges()

        return CaptureEvents(
            op_events=replay_op_events,
            op_amendments=replay_amendments,
            amendment_seq=self.amendment_seq,
            core_seal_watermark=self.core_seal_watermark,
            # A working projection accepts amendment appends even when its
            # source was sealed; the watermark filter above keeps the carried
            # lane double-application-free.
            amendments_sealed=False,
            single_seq_domain=replay_single_domain,
            module_prep_events=list(self.module_prep_events),
            module_enter_events=list(self.module_enter_events),
            module_exit_events=list(self.module_exit_events),
            pre_hook_events=list(self.pre_hook_events),
            output_version_events=list(self.output_version_events),
            buffer_write_events=list(self.buffer_write_events),
            intervention_events=list(self.intervention_events),
            backward_events=list(self.backward_events),
            aten_events=list(self.aten_events),
            aten_recording_enabled=self.aten_recording_enabled,
            param_refs=dict(self.param_refs),
            raw_layer_counter=self.raw_layer_counter,
            raw_layer_type_counter=dict(self.raw_layer_type_counter),
            func_call_id_counter=self.func_call_id_counter,
            recent_events=deque(self.recent_events),
            backend_session=self.backend_session,
            live_index=projected_index,
            grad_fn_handles_by_label_raw=dict(self.grad_fn_handles_by_label_raw),
            event_seq=self.event_seq,
            backward_revision=self.backward_revision,
            run_nonce=self.run_nonce,
            pass_index_base=self.pass_index_base,
            base_total_gradient_memory=self.base_total_gradient_memory,
            base_total_backward_memory=self.base_total_backward_memory,
            base_saved_grad_labels=self.base_saved_grad_labels,
            base_root_grad_fn_object_ids=self.base_root_grad_fn_object_ids,
        )

    def release_working_projection(self) -> None:
        """Release mutable projector lanes without touching the sealed source.

        Returns
        -------
        None
            Drops working-container and runtime-handle references after Step 0.
        """

        self.op_events.clear()
        self.op_amendments.clear()
        self._amended_fold_cache = None
        self.module_prep_events.clear()
        self.module_enter_events.clear()
        self.module_exit_events.clear()
        self.pre_hook_events.clear()
        self.output_version_events.clear()
        self.buffer_write_events.clear()
        self.intervention_events.clear()
        self.aten_events.clear()
        self.live_index.clear()
        self.grad_fn_handles_by_label_raw.clear()

    def release_runtime_sidecars(self) -> None:
        """Detach payload and runtime handles while retaining structural facts.

        Operation entries are rebuilt as payload-free immutable facts, and every
        runtime-handle sidecar the buffer holds is dropped so the advertised
        release boundary really frees backend / autograd / runtime-context object
        graphs: the backend session (``backend_session``), the per-label autograd
        ``grad_fn`` handles (``grad_fn_handles_by_label_raw``, also cleared by the
        sibling :meth:`release_working_projection`), and the runtime record-context
        deque (``recent_events``). Structural event facts (op/module/prep/enter/
        exit/output-version lanes with payloads stripped) are retained.

        Returns
        -------
        None
            Replaces operation entries with payload-free immutable facts and
            detaches all runtime-handle sidecars.
        """

        # Terminal structural projection: fold the amendment lane first, then
        # strip payloads from the FOLDED facts. Amendments may carry payload
        # refs (lookback/boundary retention outputs), so retaining the raw
        # lane after stripping would smuggle payloads past the release
        # boundary; the folded facts ARE the amended structural truth.
        structural_events: list[JournalOp] = []
        for event in self.amended_op_records():
            tensor = replace(event.output.tensor, payload=None)
            transformed = event.output.transformed_tensor
            if transformed is not None:
                transformed = replace(transformed, payload=None)
            child_versions = tuple(
                (label, replace(child_tensor, payload=None))
                for label, child_tensor in event.output.child_versions
            )
            output = replace(
                event.output,
                tensor=tensor,
                transformed_tensor=transformed,
                child_versions=child_versions,
                activation_transform=None,
            )
            templates = event.templates
            if templates is not None:
                templates = replace(
                    templates,
                    saved_args=None,
                    saved_kwargs=None,
                    args_template=None,
                    kwargs_template=None,
                )
            if isinstance(event, OpRecord):
                structural_events.append(
                    replace(
                        event,
                        core=replace(event.core, output=output),
                        templates=templates,
                    )
                )
            else:
                structural_events.append(
                    replace(
                        event,
                        output=output,
                        templates=templates,
                        source_trace=None,
                    )
                )
        self.op_events = structural_events
        self.op_amendments.clear()
        self._amended_fold_cache = None
        self.module_prep_events = [
            replace(
                event,
                forward_pre_hooks=None,
                forward_hooks=None,
                backward_pre_hooks=None,
                backward_hooks=None,
                full_backward_pre_hooks=None,
                full_backward_hooks=None,
            )
            for event in self.module_prep_events
        ]
        self.module_enter_events = [
            replace(
                event,
                forward_args=None,
                forward_kwargs=None,
                forward_args_template=None,
                forward_kwargs_template=None,
            )
            for event in self.module_enter_events
        ]
        self.pre_hook_events = [
            replace(
                event,
                inputs_before_pre_hooks=None,
                inputs_after_pre_hooks=None,
            )
            for event in self.pre_hook_events
        ]
        self.output_version_events = [
            replace(event, payload=None, transform_state=None)
            for event in self.output_version_events
        ]
        self.live_index.clear()
        self.live_index.by_raw_label = {event.label_raw: event for event in structural_events}
        self.backend_session = None
        self.grad_fn_handles_by_label_raw.clear()
        self.recent_events.clear()

    def amended_op_records(self) -> list[JournalOp]:
        """Return the canonical folded view of the op lane.

        The ONE reducer every amended-state consumer reads (producer
        unification P2/P4). The raw ``op_events`` list is genuinely
        append-only; typed amendments fold here in amendment-seq order with
        last-wins-per-path semantics, on both journal shapes (decomposed
        records via facet ``dataclasses.replace``, compat events via the 1:1
        ``PATH_TO_FLAT`` table). With no amendments this is the raw list
        itself (byte-identical passthrough). Raw ``op_events`` reads for
        amended semantics are forbidden from P2 on.
        """

        if not self.op_amendments:
            return self.op_events
        return self._amended_fold()[0]

    def amended_op_record(self, label_raw: str) -> JournalOp | None:
        """Return one op record through the folded view."""

        if not self.op_amendments:
            return self.op_event_by_label_raw.get(label_raw)
        return self._amended_fold()[1].get(label_raw)

    def _amended_fold(self) -> tuple[list[JournalOp], dict[str, JournalOp]]:
        """Compute (and cache) the amendment fold over the op lane.

        On a single-seq-domain journal the (concat-rebound) ``target_seq`` is
        the exact binding key: raw labels repeat across recorder passes once
        journals merge, so when the SAME label names several events each
        amendment folds onto the occurrence carrying its ``target_seq``
        (refusing fail-closed when no occurrence of that label carries it).
        Multi-domain projections have no single seq domain (O-N6): there
        resolution is by ``target_label_raw`` and the fold lands on the LAST
        occurrence, matching the live index's last-wins label semantics.
        Every amendment re-validates against the closed registry at fold
        (review 2.4). The cache keys on both lane lengths: each lane is
        append-only, so growth is the only invalidation signal.
        """

        cache = self._amended_fold_cache
        key = (len(self.op_events), len(self.op_amendments))
        if cache is not None and cache[0] == key:
            return cache[1], cache[2]
        last_position: dict[str, int] = {
            event.label_raw: index for index, event in enumerate(self.op_events)
        }
        position_by_seq: dict[int, int] | None = None
        if self.single_seq_domain:
            position_by_seq = {event.seq: index for index, event in enumerate(self.op_events)}
        folded_list: list[JournalOp] = list(self.op_events)
        for amendment in self.op_amendments:
            validate_amendment(amendment)
            label_raw = amendment.target_label_raw
            position = last_position.get(label_raw)
            if position is None:
                raise AmendmentTargetError(
                    f"amendment target {label_raw!r} names no committed op in this journal"
                )
            if position_by_seq is not None and folded_list[position].seq != amendment.target_seq:
                seq_position = position_by_seq.get(amendment.target_seq)
                if seq_position is None or folded_list[seq_position].label_raw != label_raw:
                    raise AmendmentTargetError(
                        f"amendment {amendment.family!r} target_seq "
                        f"{amendment.target_seq} names no committed occurrence "
                        f"of {label_raw!r} (single-seq-domain cross-check)"
                    )
                position = seq_position
            folded_list[position] = apply_patch_items(folded_list[position], amendment.patch)
        folded_by_label = {event.label_raw: event for event in folded_list}
        self._amended_fold_cache = (key, folded_list, folded_by_label)
        return folded_list, folded_by_label

    def append_amendment(self, amendment: OpAmendment) -> JournalOp:
        """Append one typed amendment, stamping its lane-local seq + nonce.

        The single writer for the amendment lane: validates against the
        closed family registry, resolves the target fail-closed, stamps
        ``seq`` from the lane-local monotone counter and ``run_nonce`` from
        this journal, and incrementally folds the patch into the live index
        so hot-path label reads keep seeing amended state. The raw
        ``op_events`` list is never touched.

        Returns
        -------
        OpEvent | OpRecord
            The folded target record after this amendment.
        """

        if self.amendments_sealed:
            raise SealedJournalAmendmentError(
                f"journal sealed at watermark {self.core_seal_watermark}: "
                f"amendment {amendment.family!r} targeting "
                f"{amendment.target_label_raw!r} must route through a working "
                "projection (copy_for_replay), never the sealed source"
            )
        validate_amendment(amendment)
        live_target = self.live_index.by_raw_label.get(amendment.target_label_raw)
        if live_target is None:
            raise AmendmentTargetError(
                f"amendment {amendment.family!r} targets unknown op {amendment.target_label_raw!r}"
            )
        target = live_target
        update_live_index = True
        if self.single_seq_domain and live_target.seq != amendment.target_seq:
            # Raw labels repeat across recorder passes once journals merge
            # (each pass re-emits the same label_raw), so the label's LIVE
            # (last) occurrence is not necessarily the bound target. In a
            # single seq domain the (concat-rebound) target_seq is the exact
            # key: bind to the earlier committed occurrence of the SAME label
            # carrying that seq; anything else stays refused fail-closed.
            target = next(
                (
                    event
                    for event in reversed(self.op_events)
                    if event.seq == amendment.target_seq
                    and event.label_raw == amendment.target_label_raw
                ),
                None,
            )
            if target is None:
                raise AmendmentTargetError(
                    f"amendment {amendment.family!r} target_seq "
                    f"{amendment.target_seq} names no committed occurrence of "
                    f"{amendment.target_label_raw!r} (live occurrence has seq "
                    f"{live_target.seq}; single-seq-domain cross-check)"
                )
            # The live index tracks only the label's LAST occurrence, and its
            # incrementally-folded state belongs to that op: folding an
            # earlier occurrence must not clobber it. The returned fold below
            # is best-effort for this path (raw op + this patch); the reducer
            # (``amended_op_records``) stays the canonical folded read.
            update_live_index = False
        self.amendment_seq += 1
        object.__setattr__(amendment, "seq", self.amendment_seq)
        object.__setattr__(amendment, "run_nonce", self.run_nonce)
        self.op_amendments.append(amendment)
        self._amended_fold_cache = None
        folded = apply_patch_items(target, amendment.patch)
        if update_live_index:
            self.live_index.replace(folded)
        return folded

    def next_seq(self) -> int:
        """Return the next value of the one run-monotonic event sequence."""

        self.event_seq += 1
        return self.event_seq

    def _refuse_sealed_append(self, lane_name: str) -> None:
        """Refuse a forward-lane append on a sealed source journal (B1-15).

        Parameters
        ----------
        lane_name
            Lane the caller is trying to write.

        Raises
        ------
        SealedJournalAppendError
            When this journal was sealed into a ``CapturedRunCore``. Every
            forward lane the seal snapshotted is fenced; the backward lane is
            legitimately still open (see :class:`SealedJournalAppendError`).
        """

        if self.amendments_sealed:
            raise SealedJournalAppendError(
                f"journal sealed at watermark {self.core_seal_watermark}: "
                f"appending to {lane_name!r} would diverge the live stream from "
                "its sealed CapturedRunCore snapshot; route post-seal work "
                "through a working projection (copy_for_replay), never the "
                "sealed source"
            )

    def append(self, event: JournalOp) -> None:
        """Append a single operation event/record, stamping the global seq.

        The seq slot lives on the flat event for compat ``OpEvent``s and on
        ``core`` for decomposed ``OpRecord``s; both are frozen dataclasses and
        this append path is their single sequencing authority.

        Refuses on a sealed journal (B1-15): the op spine IS the sealed core's
        ``events`` tuple.
        """
        self._refuse_sealed_append("op_events")
        seq = self.next_seq()
        if isinstance(event, OpRecord):
            object.__setattr__(event.core, "seq", seq)
        else:
            object.__setattr__(event, "seq", seq)
        self.op_events.append(event)
        if len(self.op_events) == OP_COUNT_DISCLOSURE_THRESHOLD:
            # Capture-side op-count disclosure (r8 R60-1): the save budget
            # bounds retained activation BYTES only, so a million-op forward
            # with nothing saved grew op records unbounded with no
            # torchlens-side signal until the OOM killer fired. This is a
            # disclosure, not a ceiling -- huge captures stay legal -- fired
            # exactly once per journal at the threshold crossing (one int
            # compare per op on the hot path).
            warnings.warn(
                f"TorchLens has recorded {OP_COUNT_DISCLOSURE_THRESHOLD} ops in "
                "this forward and capture is still running. Op records grow "
                "memory per op regardless of save=; a very deep or long-loop "
                "forward can exhaust host RAM. Consider tracing a smaller "
                "submodule, reducing loop iterations, or using "
                "tl.record(model, x, save=...) for sparse recording.",
                TorchLensWarning,
                stacklevel=2,
            )
        self.live_index.append(event)

    def append_module_prep(self, event: ModulePrepEvent) -> None:
        """Append a module-prep sibling event, stamping the global seq."""
        self._refuse_sealed_append("module_prep_events")
        object.__setattr__(event, "seq", self.next_seq())
        self.module_prep_events.append(event)

    def append_module_enter(self, event: ModuleEnterEvent) -> None:
        """Append a module-entry sibling event, stamping the global seq."""
        self._refuse_sealed_append("module_enter_events")
        object.__setattr__(event, "seq", self.next_seq())
        self.module_enter_events.append(event)

    def append_module_exit(self, event: ModuleExitEvent) -> None:
        """Append a module-exit sibling event, stamping the global seq."""
        self._refuse_sealed_append("module_exit_events")
        object.__setattr__(event, "seq", self.next_seq())
        self.module_exit_events.append(event)

    def append_pre_hook(self, event: PreHookProvenanceEvent) -> None:
        """Append a pre-hook provenance sibling event, stamping the global seq."""
        self._refuse_sealed_append("pre_hook_events")
        object.__setattr__(event, "seq", self.next_seq())
        self.pre_hook_events.append(event)

    def append_buffer_write(self, event: BufferWriteEvent) -> None:
        """Append a registered-buffer write event, stamping the global seq."""
        self._refuse_sealed_append("buffer_write_events")
        object.__setattr__(event, "seq", self.next_seq())
        self.buffer_write_events.append(event)

    def append_intervention(self, event: InterventionAppliedEvent) -> None:
        """Append an intervention edit record, stamping the global seq."""
        self._refuse_sealed_append("intervention_events")
        object.__setattr__(event, "seq", self.next_seq())
        self.intervention_events.append(event)

    def append_aten(self, event: _AtenCallEvent | _ModePausedInteriorEvent) -> None:
        """Append one ATen-profile event under the journal sequencing authority.

        Parameters
        ----------
        event
            Value-free primitive-call or observer-gap event.
        """

        object.__setattr__(event, "seq", self.next_seq())
        self.aten_events.append(event)

    def concat(self, other: CaptureEvents, *, lanes: Iterable[str] | None = None) -> None:
        """Merge another stream's lanes into this journal under the merge law.

        This is the ONLY sanctioned way to combine two capture streams. Each
        lane follows its declared :data:`LANE_MERGE_POLICIES` entry. Merging
        events keep the SOURCE stream's cross-lane chronological order (its
        ``seq`` domain is the sort key) and are re-stamped into THIS journal's
        sequence domain by the single-writer append methods, so the combined
        journal keeps unique seq values that preserve the source's recorded
        chronology. Every merged event is a CLONE: re-stamping never mutates
        the sealed source stream. Counters, param refs, and runtime sidecars
        stay the target's own (they are run state, not journal facts).

        Raises
        ------
        LaneMergePolicyError
            When a requested lane declares a merging (non-``run_local``)
            policy but has no registered single-writer appender. The check
            runs before any event moves, so a declared-but-unwired lane fails
            closed even while empty.
        SourceSequencingError
            When the source's global seq domain is invalid: an unstamped
            event, a duplicate cross-lane seq, a non-monotone lane, or a
            stamp beyond the source's writer counter. The check runs before
            any event moves, so an invalid source never partially merges.

        Parameters
        ----------
        other
            Source stream whose lanes should fold into this journal.
        lanes
            Optional restriction to a subset of lane names. ``None`` merges
            every declared lane under its policy. A caller may restrict lanes
            (failed-partial recovery keeps only op and pre-hook facts from the
            failing pass) but never override a lane's declared policy.
        """

        if other is self:
            return
        lane_names = tuple(lanes) if lanes is not None else tuple(LANE_MERGE_POLICIES)
        # Target-rebinding precondition (B1-15 companion): every amendment's
        # ``target_seq`` is rebound through the merge seq map built from the OP
        # lane. Merging the amendment lane WITHOUT the op lane leaves that map
        # empty, so each rebind silently falls back to the source-domain value
        # and binds against this journal's unrelated seq domain. Refuse the lane
        # selection instead.
        if "op_amendments" in lane_names and "op_events" not in lane_names and other.op_amendments:
            raise LaneMergePolicyError(
                "concat cannot merge lane 'op_amendments' without 'op_events': "
                "amendment target_seq values rebind through the op lane's merge "
                "seq map, which would be empty"
            )
        for lane_name in lane_names:
            if LANE_MERGE_POLICIES[lane_name] != "run_local" and lane_name not in _LANE_APPENDERS:
                raise LaneMergePolicyError(
                    f"lane {lane_name!r} declares merge policy "
                    f"{LANE_MERGE_POLICIES[lane_name]!r} but has no registered appender; "
                    "wire it into _LANE_APPENDERS before it can merge"
                )
        # Source seq-domain gate: sorting an invalid domain would replace real
        # chronology with dict-lane-order tie-breaking, so the source must
        # PROVE its stamps are unique, per-lane monotone, and counter-covered
        # before anything moves. This mirrors the journal seq invariants a
        # standalone stream is held to; concat must not launder a stream that
        # validation would reject. Skipped (``first_run_only``-satisfied)
        # lanes still validate: a corrupt lane in the source is a producer
        # defect regardless of whether its events merge this round.
        seen_lane_by_seq: dict[int, str] = {}
        for lane_name in lane_names:
            if LANE_MERGE_POLICIES[lane_name] == "run_local":
                continue
            if lane_name == "op_amendments":
                # Lane-local seq domain: amendments never share the event
                # counter (their stamps must not shift Tier-F event seqs), so
                # they validate against their own writer counter and merge in
                # a dedicated pass below instead of the shared sorted merge.
                previous_amendment_seq = 0
                for amendment in other.op_amendments:
                    if amendment.seq < 1:
                        raise SourceSequencingError(
                            "concat source lane 'op_amendments' holds an "
                            f"unstamped amendment (seq {amendment.seq})"
                        )
                    if amendment.seq <= previous_amendment_seq:
                        raise SourceSequencingError(
                            f"concat source lane 'op_amendments' seq "
                            f"{amendment.seq} does not increase past "
                            f"{previous_amendment_seq}"
                        )
                    previous_amendment_seq = amendment.seq
                if previous_amendment_seq > int(other.amendment_seq or 0):
                    raise SourceSequencingError(
                        f"concat source amendment seq {previous_amendment_seq} "
                        f"exceeds the source writer counter {other.amendment_seq}"
                    )
                continue
            previous_seq = 0
            for event in getattr(other, lane_name):
                seq = int(getattr(event, "seq", 0) or 0)
                if seq < 1:
                    raise SourceSequencingError(
                        f"concat source lane {lane_name!r} holds an unstamped event "
                        f"(seq {seq}): only single-writer-stamped streams may merge"
                    )
                if seq <= previous_seq:
                    raise SourceSequencingError(
                        f"concat source lane {lane_name!r} seq {seq} does not "
                        f"increase past {previous_seq}"
                    )
                previous_seq = seq
                duplicate_lane = seen_lane_by_seq.get(seq)
                if duplicate_lane is not None:
                    raise SourceSequencingError(
                        f"concat source seq {seq} appears in both "
                        f"{duplicate_lane!r} and {lane_name!r}"
                    )
                seen_lane_by_seq[seq] = lane_name
        if seen_lane_by_seq and max(seen_lane_by_seq) > int(other.event_seq or 0):
            raise SourceSequencingError(
                f"concat source seq {max(seen_lane_by_seq)} exceeds the source "
                f"writer counter {other.event_seq}: an event bypassed the "
                "single-writer append path"
            )
        merge_rows: list[tuple[int, str, Any]] = []
        for lane_name in lane_names:
            policy = LANE_MERGE_POLICIES[lane_name]
            if policy == "run_local" or lane_name == "op_amendments":
                continue
            source_events = list(getattr(other, lane_name))
            if not source_events:
                continue
            if policy == "first_run_only" and getattr(self, lane_name):
                continue
            merge_rows.extend((event.seq, lane_name, event) for event in source_events)
        # Sort on the source seq domain: cross-lane chronology is preserved
        # exactly (the gate above proved every stamp unique and monotone, so
        # the sort never has to tie-break).
        merge_rows.sort(key=lambda row: row[0])
        seq_map: dict[int, int] = {}
        for source_seq, lane_name, event in merge_rows:
            if lane_name == "op_events":
                clone = _clone_op_event_for_replay(event)
            else:
                clone = replace(event)
            # Sanctioned chain of custody: an edit genuinely bound to the
            # SOURCE run re-binds to this journal (its run token and the
            # re-stamped seq of its already-merged target op, which sorts
            # earlier by chronology). A forged/unbound edit or one bound to
            # a foreign run keeps its stale binding and stays refused by
            # validation.
            if (
                lane_name == "intervention_events"
                and lane_name in REBINDABLE_TARGET_LANES
                and getattr(event, "run_token", None) == other.run_nonce
            ):
                object.__setattr__(clone, "run_token", self.run_nonce)
                object.__setattr__(
                    clone, "target_seq", seq_map.get(getattr(event, "target_seq", 0), 0)
                )
            getattr(self, _LANE_APPENDERS[lane_name])(clone)
            if source_seq:
                seq_map[source_seq] = clone.seq
        # Dedicated amendment pass (REBINDABLE_TARGET_LANES): the lane rides
        # its own seq domain, so it merges after the event lanes — every
        # amendment's target op has already merged (or the append below
        # refuses fail-closed). ``target_seq`` rebinds through the merge seq
        # map into this journal's event domain; the appender re-stamps the
        # lane-local seq and run nonce and re-folds the live index.
        if "op_amendments" in lane_names and LANE_MERGE_POLICIES["op_amendments"] != "run_local":
            for amendment in other.op_amendments:
                clone = replace(amendment)
                source_target_seq = int(getattr(amendment, "target_seq", 0) or 0)
                if source_target_seq >= 1 and source_target_seq not in seq_map:
                    # The target op did not merge, so no rebinding is possible.
                    # Keeping the source-domain value would point the amendment
                    # at whatever this journal happens to hold at that seq.
                    raise LaneMergePolicyError(
                        f"concat cannot rebind amendment {amendment.family!r}: its "
                        f"target_seq {source_target_seq} has no merged op in this "
                        "journal's seq domain"
                    )
                object.__setattr__(
                    clone,
                    "target_seq",
                    seq_map.get(amendment.target_seq, amendment.target_seq),
                )
                self.append_amendment(clone)

    def append_backward(
        self,
        event: BackwardPassStart
        | OpGradObserved
        | ParamGradObserved
        | BackwardPassEnd
        | GradFnDiscovered
        | GradFnFired
        | BackwardCoverageGap
        | CheckpointInvocationObserved,
    ) -> None:
        """Append a backward sidecar event, stamping the global backward seq.

        The append path is the single writer for the backward stream, so it is
        also the single sequencing authority: every appended event of every
        kind receives the next value of one run-monotonic counter, making
        cross-kind ordering an exact recorded fact rather than an inference
        from timestamps or list positions.

        It is also the single freezing authority for nested mutable event
        state: ``GradFnDiscovered.source`` is the one nested container the
        projection copies BY VALUE at materialize time, so an in-place
        mutation of it would diverge a guarded (already-folded) projection
        from a scratch rebuild without moving ``backward_revision``. The
        writer therefore snapshots it into a read-only mapping over a PRIVATE
        dict copy here — unconditionally, because a caller-supplied
        ``MappingProxyType`` still aliases the caller's mutable backing dict,
        which would reintroduce the exact bypass the freeze exists to close.
        Every other nested reference (payload refs, ``engine_flags``,
        ``root_meta`` elements) is shared BY REFERENCE between the event and
        both projection paths, so mutating it cannot make folded and scratch
        state diverge.
        """

        if isinstance(event, GradFnDiscovered):
            object.__setattr__(event, "source", MappingProxyType(dict(event.source)))
        object.__setattr__(event, "seq", self.next_seq())
        self.backward_events.append(event)
        self.backward_revision += 1

    def note_backward_event_removal(self) -> None:
        """Advance the backward revision after a sanctioned event removal.

        Event count alone cannot distinguish an add-then-remove from an
        unchanged stream, so every mutation of ``backward_events`` must move
        the revision forward for the projection guard to stay sound.
        """

        self.backward_revision += 1

    def extend(self, events: tuple[OpEvent, ...] | list[OpEvent]) -> None:
        """Append multiple operation events in order, re-stamping seq.

        Extending moves events into THIS buffer's sequence domain (the
        recorder's multi-pass accumulation), so each event receives a fresh
        ``seq`` from this buffer's counter.
        """
        for event in events:
            self.append(event)

    def append_output_version(self, event: OutputVersionEvent) -> None:
        """Append a parent output-version sibling event, stamping the global seq."""
        self._refuse_sealed_append("output_version_events")
        object.__setattr__(event, "seq", self.next_seq())
        self.output_version_events.append(event)

    def reserve_label(self, layer_type: str) -> ReservedLabel:
        """Reserve the next raw label for a single output site."""
        return self.reserve_label_block(layer_type, 1)[0]

    def reserve_label_block(self, layer_type: str, n: int) -> tuple[ReservedLabel, ...]:
        """Reserve a contiguous block of raw labels for output sites."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return ()

        type_counter = self.raw_layer_type_counter.get(layer_type, 0)
        labels: list[ReservedLabel] = []
        for _ in range(n):
            self.raw_layer_counter += 1
            type_counter += 1
            label_raw = f"{layer_type}_{type_counter}_{self.raw_layer_counter}_raw"
            labels.append(
                ReservedLabel(
                    label=label_raw,
                    label_raw=label_raw,
                    raw_index=self.raw_layer_counter,
                    type_index=type_counter,
                    layer_type=layer_type,
                    site=label_raw,
                )
            )
        self.raw_layer_type_counter[layer_type] = type_counter
        return tuple(labels)


def register_live_event(trace: Any, event: OpEvent) -> None:
    """Register an emitted operation event on a trace.

    Appends ``event`` to ``trace.capture_events`` (allocating the buffer on
    first use) and records its grad-fn handle when present. This function has no
    callers in the tree; the live hot path appends events directly through
    :meth:`CaptureEvents.append`.

    Parameters
    ----------
    trace
        Active trace receiving capture events.
    event
        Operation event emitted for the new raw label.

    Returns
    -------
    None
        Mutates ``trace.capture_events``.
    """

    events = getattr(trace, "capture_events", None)
    if events is None:
        events = CaptureEvents()
        trace.capture_events = events
    events.append(event)
    if event.grad_fn_handle is not None:
        events.grad_fn_handles_by_label_raw[event.label_raw] = event.grad_fn_handle


# ``replace_op_event`` is DELETED (producer unification P4): the op lane is
# genuinely append-only and every post-commit mutation routes through the
# typed amendment lane (``CaptureEvents.append_amendment`` + the nine-family
# registry in ``op_record.py``).
