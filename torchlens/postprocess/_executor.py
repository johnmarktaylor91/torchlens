"""Postprocess dependency derivation: declared contracts -> execution order.

The pipeline order is DERIVED, not hand-maintained (design-ppdag-v3): each
step's ``PostprocessStepContract`` declares its op-column writes/reads,
trace-state tokens, row effects, and barrier; ``derive_edges`` orients every
conflict by the frozen ``LEGACY_STEP_RANK`` (never by registry position),
and ``execution_order`` runs Kahn's algorithm with a min-heap keyed by rank.
Ranks are unique, so no secondary tie-break exists.

Day-1 identity theorem (verified independently by both design reviewers):
every edge rule orients rank-ascending, so the graph is acyclic by
construction and the rank-keyed Kahn emits exactly rank order — which R1
pins to the registry order. The derivation therefore detects ARTIFACT DRIFT
(registry, rank, or declarations disagreeing), never reorder legality; the
semantic direction authority is the reason-bearing ``PINNED_ORDER_PAIRS``
corpus (checks K1 at import in ``_contracts``, K2 test-side).

Import-time structural checks (plain ``raise``, never ``assert`` —
``python -O`` strips asserts): the 7.1 family below. The read-before-write
findings channel is a TEST REPORT over the recording matrix, never an
import gate — import must not depend on recordings.
"""

from __future__ import annotations

import heapq
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..utils.display import _vtimed
from ._contracts import (
    CAPTURE_BASELINE_TOKENS,
    LEGACY_STEP_RANK,
    POSTPROCESS_STEP_CONTRACTS,
    PostprocessStepContract,
)

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


def _pp() -> Any:
    """Return the live ``torchlens.postprocess`` module namespace.

    Every step body resolves its callable through this namespace AT CALL
    TIME (design-ppdag-v3 §5.2, tripwire integrity): tests monkeypatch step
    functions on the module — a registry holding imported references would
    break the in-place-mutation tripwire test and the streaming/step-0
    seams silently.
    """

    return sys.modules["torchlens.postprocess"]


@dataclass
class StepContext:
    """Mutable per-run context threaded through the step registry."""

    trace: Trace
    output_tensors: list[Any]
    output_tensor_addresses: list[str]
    output_parent_labels: list[Any]
    capture_session: Any | None
    #: Step-18-position snapshot (design §5.4): 18's should_run IS the
    #: snapshot point — 18 clears _out_writer, so a re-evaluated 19
    #: predicate would always be false and streamed outs would never be
    #: evicted. Context-writing, never trace-writing.
    finalize_streaming: bool | None = None


def _always(ctx: StepContext) -> bool:
    """Default should_run: the step always executes."""

    return True


@dataclass(frozen=True)
class StepSpec:
    """One executable pipeline step.

    ``should_run`` is evaluated EXACTLY ONCE per step per run, in registry
    order (executor invariant, counting-tested). ``assert_when_skipped``
    mirrors the historical per-step assert placement: steps 11.75 and 13
    assert unconditionally (their gates are inside), steps 4/18/19 assert
    only when their body ran.
    """

    step: str
    run: Callable[[StepContext], None]
    should_run: Callable[[StepContext], bool] = _always
    assert_when_skipped: bool = False


def _run_step_1(ctx: StepContext) -> None:
    """Step 1: create dedicated output nodes for each attributable output tensor."""

    with _vtimed(ctx.trace, "  Step 1: Add output layers"):
        _pp()._add_output_layers(
            ctx.trace,
            ctx.output_tensors,
            ctx.output_tensor_addresses,
            ctx.output_parent_labels,
        )


def _run_step_2(ctx: StepContext) -> None:
    """Step 2: mark every node that is an ancestor of a model output."""

    with _vtimed(ctx.trace, "  Step 2: Trace output ancestors"):
        _pp()._find_output_ancestors(ctx.trace)


def _run_step_3(ctx: StepContext) -> None:
    """Step 3: drop raw nodes that reach no model output."""

    with _vtimed(ctx.trace, "  Step 3: Remove orphan nodes"):
        _pp()._remove_orphan_nodes(ctx.trace)


def _run_step_4(ctx: StepContext) -> None:
    """Step 4: write the optional input/output distance metadata."""

    with _vtimed(ctx.trace, "  Step 4: Input/output distances"):
        _pp()._mark_layer_depths(ctx.trace)


def _should_run_step_4(ctx: StepContext) -> bool:
    """Gate step 4 on the opt-in ``Trace.mark_layer_depths`` flag."""

    return bool(ctx.trace.mark_layer_depths)


def _run_step_5(ctx: StepContext) -> None:
    """Step 5: attribute conditional branches (AST, bool, event, and edge evidence)."""

    with _vtimed(ctx.trace, "  Step 5: Mark conditional branches"):
        _pp()._mark_conditional_branches(ctx.trace)


def _run_step_6(ctx: StepContext) -> None:
    """Step 6: deduplicate buffer layers and reconnect their edges."""

    with _vtimed(ctx.trace, "  Step 6: Fix buffer layers"):
        _pp()._fix_buffer_layers(ctx.trace)


def _run_step_7(ctx: StepContext) -> None:
    """Step 7: group recurrent calls into multi-pass layers.

    With ``recurrence_detection`` off this degrades to shared-parameter
    grouping only, which is the historical single-pass layout.
    """

    trace = ctx.trace
    loop_desc = (
        "  Step 7: Loop detection (full)"
        if trace.recurrence_detection
        else "  Step 7: Loop detection (params only)"
    )
    with _vtimed(trace, loop_desc):
        if trace.recurrence_detection:
            _pp()._detect_and_label_loops(trace)
        else:
            _pp()._group_by_shared_params(trace)


def _run_step_8(ctx: StepContext) -> None:
    """Step 8: build the raw-label to final-label map."""

    with _vtimed(ctx.trace, "  Step 8: Map labels"):
        _pp()._map_raw_labels_to_final_labels(ctx.trace)


def _run_step_9(ctx: StepContext) -> None:
    """Step 9: write the final per-layer and per-module fields."""

    with _vtimed(ctx.trace, "  Step 9: Log final info"):
        _pp()._log_final_info_for_layers(ctx.trace)


def _run_step_10(ctx: StepContext) -> None:
    """Step 10: rename global label references to their final names."""

    with _vtimed(ctx.trace, "  Step 10: Rename labels"):
        _pp()._rename_model_history_layer_names(ctx.trace)


def _run_step_11(ctx: StepContext) -> None:
    """Step 11: build lookup keys, finalize retained layers, warn on unattributed args.

    The three calls share one step because the saved-summary refresh and the
    unattributed-arg warning both read the retained-layer lists this step
    finalizes.
    """

    module = _pp()
    with _vtimed(ctx.trace, "  Step 11: Build lookup keys"):
        module._build_lookup_keys_and_finalize_retained_layers(ctx.trace)
        module._refresh_fast_saved_summary(ctx.trace)
        module._warn_unattributed_tensor_args(ctx.trace)


def _run_step_11_5(ctx: StepContext) -> None:
    """Step 11.5: resolve source assignment variable names from the AST index."""

    with _vtimed(ctx.trace, "  Step 11.5: Populate source var names"):
        _pp()._populate_var_names(ctx.trace)


def _run_step_11_75(ctx: StepContext) -> None:
    """Step 11.75: let the capture session resolve deferred retention decisions.

    Runs only when a ``CaptureSession`` is attached; ``should_run`` gates on
    that, so the body may assert it.
    """

    capture_session = ctx.capture_session
    assert capture_session is not None  # should_run gates on this
    with _vtimed(ctx.trace, "  Step 11.75: Resolve deferred retention"):
        capture_session.resolve_deferred_retention(ctx.trace, list(ctx.output_tensors))


def _run_step_12(ctx: StepContext) -> None:
    """Step 12: strip TorchLens ``tl_*`` attributes from every saved tensor."""

    with _vtimed(ctx.trace, "  Step 12: Undecorate tensors"):
        _pp()._undecorate_all_saved_tensors(ctx.trace)


def _run_step_13(ctx: StepContext) -> None:
    """Step 13: clear the CUDA caching allocator for CUDA-touching captures."""

    # Unwrapped by _vtimed (historical); the CUDA availability test lives
    # in the body so the contract assert fires unconditionally (§5.4).
    # R36-3: additionally gated on the CAPTURE having touched CUDA (the
    # trace-level forward_memory_backend fact) so a CPU-only trace inside a
    # GPU training loop no longer flushes the caller's allocator.
    from ..utils.tensor_utils import capture_touched_cuda

    module = _pp()
    if module._is_cuda_available() and capture_touched_cuda(ctx.trace):
        module.torch.cuda.empty_cache()


def _run_step_14(ctx: StepContext) -> None:
    """Step 14: record elapsed capture timing on the trace."""

    with _vtimed(ctx.trace, "  Step 14: Log timing"):
        _pp()._log_time_elapsed(ctx.trace)


def _run_step_15(ctx: StepContext) -> None:
    """Step 15: build and complete the ``Param`` records."""

    with _vtimed(ctx.trace, "  Step 15: Finalize params"):
        _pp()._finalize_param_logs(ctx.trace)


def _run_step_15_5(ctx: StepContext) -> None:
    """Step 15.5: build aggregate ``Layer`` records and the pass-index index.

    ``Trace.by_pass`` maps each pass index to the ``layer_list`` positions of
    the layers recorded at that pass; layers with no ``pass_index`` are absent
    rather than bucketed.
    """

    trace = ctx.trace
    with _vtimed(trace, "  Step 15.5: Build layer logs"):
        _pp()._build_layer_logs(trace)
        trace.by_pass = {}
        for index, op in enumerate(trace.layer_list):
            pass_index = getattr(op, "pass_index", None)
            if pass_index is not None:
                trace.by_pass.setdefault(pass_index, []).append(index)


def _run_step_16(ctx: StepContext) -> None:
    """Step 16: build ``Module`` records and refresh the saved module-call count."""

    module = _pp()
    with _vtimed(ctx.trace, "  Step 16: Build module logs"):
        module._build_module_logs(ctx.trace)
        module.refresh_saved_module_call_count(ctx.trace)


def _run_step_16_5(ctx: StepContext) -> None:
    """Step 16.5: normalize layer addresses, then hash the graph shape.

    The hash is taken here, before step 17 flips the trace to user-facing
    behavior and changes what the traversal observes.
    """

    module = _pp()
    trace = ctx.trace
    with _vtimed(trace, "  Step 16.5: Graph shape hash"):
        module.populate_normalized_layer_addresses(trace)
        trace.graph_shape_hash = module.compute_graph_shape_hash(trace)


def _run_step_17(ctx: StepContext) -> None:
    """Step 17: switch the trace from build-time to user-facing behavior."""

    with _vtimed(ctx.trace, "  Step 17: Mark pass finished"):
        _pp()._set_tracing_finished(ctx.trace)


def _run_step_17_5(ctx: StepContext) -> None:
    """Step 17.5: adopt container records, then drop the per-phase workspaces.

    This is the contracted terminal consume of ``RawGraphWorkspace``,
    ``ModuleCaptureWorkspace``, and ``WrapperRuntimeWorkspace``: nothing after
    this step may read them. The container registry's records are copied onto
    the trace before its live state is cleared.
    """

    # Unwrapped by _vtimed (historical). Adopt container records, then drop
    # the per-phase workspaces (terminal consumes).
    trace = ctx.trace
    wrapper_ws = trace.__dict__.get("_wrapper_runtime_ws")
    if wrapper_ws is not None:
        registry = getattr(wrapper_ws, "container_registry", None)
        if registry is not None:
            if registry.records:
                trace.__dict__["_containers"] = dict(registry.records)
            registry.clear_live_state()
    for field_name in (
        "_raw_graph_ws",
        "_module_capture_ws",
        "_wrapper_runtime_ws",
        "capture_events",
        "_output_container_specs_by_raw_label",
    ):
        trace.__dict__.pop(field_name, None)


def _should_run_step_18(ctx: StepContext) -> bool:
    """THE streaming snapshot point (design §5.4, deliberately context-writing)."""

    trace = ctx.trace
    ctx.finalize_streaming = getattr(trace, "_out_writer", None) is not None and not getattr(
        trace, "_defer_streaming_bundle_finalization", False
    )
    return ctx.finalize_streaming


def _run_step_18(ctx: StepContext) -> None:
    """Step 18: finalize the streamed out bundle."""

    with _vtimed(ctx.trace, "  Step 18: Finalize streamed bundle"):
        _pp()._finalize_streamed_bundle(ctx.trace)


def _should_run_step_19(ctx: StepContext) -> bool:
    """Gate step 19 on step 18 having run with in-memory outs not requested."""

    return bool(ctx.finalize_streaming) and not ctx.trace._keep_outs_in_memory


def _run_step_19(ctx: StepContext) -> None:
    """Step 19: evict streamed outs from memory."""

    with _vtimed(ctx.trace, "  Step 19: Evict streamed outs"):
        _pp()._evict_streamed_outs(ctx.trace)


def _run_step_20(ctx: StepContext) -> None:
    """Step 20: release live parameter references held for finalization."""

    with _vtimed(ctx.trace, "  Step 20: Release param refs"):
        ctx.trace.release_param_refs(allow_iter_rehydrate=True)


#: The executable registry, in the canonical hand-established order. R1
#: pins it strictly rank-ascending; R2 pins the derived order equal to it.
#: Step "0" is the fenced prologue (producer lane) and deliberately absent.
STEP_REGISTRY: tuple[StepSpec, ...] = (
    StepSpec("1", _run_step_1),
    StepSpec("2", _run_step_2),
    StepSpec("3", _run_step_3),
    StepSpec("4", _run_step_4, should_run=_should_run_step_4),
    StepSpec("5", _run_step_5),
    StepSpec("6", _run_step_6),
    StepSpec("7", _run_step_7),
    StepSpec("8", _run_step_8),
    StepSpec("9", _run_step_9),
    StepSpec("10", _run_step_10),
    StepSpec("11", _run_step_11),
    StepSpec("11.5", _run_step_11_5),
    StepSpec(
        "11.75",
        _run_step_11_75,
        should_run=lambda ctx: ctx.capture_session is not None,
        assert_when_skipped=True,
    ),
    StepSpec("12", _run_step_12),
    StepSpec("13", _run_step_13, assert_when_skipped=True),
    StepSpec("14", _run_step_14),
    StepSpec("15", _run_step_15),
    StepSpec("15.5", _run_step_15_5),
    StepSpec("16", _run_step_16),
    StepSpec("16.5", _run_step_16_5),
    StepSpec("17", _run_step_17),
    StepSpec("17.5", _run_step_17_5),
    StepSpec("18", _run_step_18, should_run=_should_run_step_18),
    StepSpec("19", _run_step_19, should_run=_should_run_step_19),
    StepSpec("20", _run_step_20),
)

REGISTRY_ORDER: tuple[str, ...] = tuple(spec.step for spec in STEP_REGISTRY)


def run_pipeline(ctx: StepContext) -> None:
    """Execute steps 1-20 with explicit per-step audit windows.

    Window protocol (design §2.5): begin -> run -> end (in ``finally``, so
    a raising step can no longer leave the store class-swapped with a live
    collector) -> contract check -> postconditions OUTSIDE any window. No
    window is open while harness code runs, and none survives the loop —
    the freeze seam after step 20 runs unaudited by construction (review
    note N11 replaces the historical trailing-window discard).
    """

    module = _pp()
    audits_enabled = module._postprocess_assertions_enabled()
    for spec in STEP_REGISTRY:
        should = spec.should_run(ctx)
        if not should and not spec.assert_when_skipped:
            continue
        if not audits_enabled:
            if should:
                spec.run(ctx)
            continue
        try:
            # B1-23b: the OPEN is inside the try. ``begin_cell_write_audit``
            # registers the collectors and only then swaps the store class, and
            # its fingerprint sweep runs arbitrary ``__repr__`` code at the
            # leaves -- so a raise mid-arm used to escape with collectors
            # registered under ``id(store)`` and the class possibly swapped,
            # with nothing to unwind it. ``end_cell_write_audit`` is pop-based
            # and only un-swaps an actually-audited class, so closing an
            # unopened (or half-opened) window is safe.
            module._open_step_write_audit(ctx.trace)
            if should:
                spec.run(ctx)
        finally:
            audit_result = module._close_step_write_audit(ctx.trace)
        module._check_postprocess_contract(ctx.trace, spec.step, audit_result)
    if audits_enabled:
        module._assert_no_open_window(ctx.trace)


@dataclass(frozen=True)
class DerivedEdge:
    """One derived ordering edge with its provenance.

    ``kind`` is one of ``raw`` / ``ww`` / ``war`` / ``row_barrier`` /
    ``token_raw`` / ``token_ww`` / ``token_war`` / ``barrier``; ``carrier``
    names the column or token (empty for structural edges).
    """

    src: str
    dst: str
    kind: str
    carrier: str


def _registry_contracts() -> dict[str, PostprocessStepContract]:
    """Return the registry-step contracts (step "0" excluded)."""

    return {step: contract for step, contract in POSTPROCESS_STEP_CONTRACTS.items() if step != "0"}


def _op_touching(contract: PostprocessStepContract) -> bool:
    """Whether a step reads or writes ANY op column (edge rule 4 operand).

    Row-set membership is an implicit input of every per-op iteration and
    an implicit precondition of every cell write, so row-effect steps pin
    two-sidedly against every op-touching step — including each other.
    """

    return bool(
        contract.writes or contract.reads or contract.placeholder_probes or contract.row_effects
    )


def derive_edges(rank_override: dict[str, int] | None = None) -> list[DerivedEdge]:
    """Derive every ordering edge from the declared contracts.

    All rules orient by ``LEGACY_STEP_RANK`` (design-ppdag-v3 §2.3):

    1. RAW: reader depends on EVERY lower-rank writer of the column
       (refine-in-place semantics).
    2. WW: co-writers order pairwise by rank.
    3. WAR: a lower-rank reader pins before every higher-rank writer.
       Declared placeholder probes generate WAR edges too — a probe's
       correctness depends on observing "not yet set", so the probing step
       stays pinned before the writer (probes are exempt from FINDINGS,
       never from edges).
    4. Row structure: a step with row_effects is a two-sided barrier with
       respect to every op-touching step.
    5. Tokens: RAW/WW/WAR over ``r:<t>``/``w:<t>``.
    6. Barrier: a ``barrier=True`` step orders against every other step.
    """

    contracts = _registry_contracts()
    rank = rank_override if rank_override is not None else dict(LEGACY_STEP_RANK)
    edges: list[DerivedEdge] = []

    def _directed(low: str, high: str) -> tuple[str, str] | None:
        """Order one step pair rank-ascending, or ``None`` when both are the same step.

        Ranks are unique across distinct steps, so the ``None`` result means
        ``low is high`` (a self-conflict, which generates no edge).
        """

        if rank[low] < rank[high]:
            return (low, high)
        if rank[high] < rank[low]:
            return (high, low)
        return None

    # Rules 1-3: column conflicts.
    writers: dict[str, list[str]] = {}
    readers: dict[str, list[str]] = {}
    for step, contract in contracts.items():
        for column in contract.writes:
            writers.setdefault(column, []).append(step)
        for column in contract.reads | contract.placeholder_probes:
            readers.setdefault(column, []).append(step)
    for column, column_writers in writers.items():
        by_rank = sorted(column_writers, key=rank.__getitem__)
        for i, low in enumerate(by_rank):
            for high in by_rank[i + 1 :]:
                edges.append(DerivedEdge(low, high, "ww", column))
        for reader in readers.get(column, ()):  # rules 1 and 3
            for writer in column_writers:
                if writer == reader:
                    continue
                if rank[writer] < rank[reader]:
                    edges.append(DerivedEdge(writer, reader, "raw", column))
                else:
                    edges.append(DerivedEdge(reader, writer, "war", column))

    # Rule 5: token conflicts.
    token_writers: dict[str, list[str]] = {}
    token_readers: dict[str, list[str]] = {}
    for step, contract in contracts.items():
        for entry in contract.trace_state:
            prefix, _, token = entry.partition(":")
            target = token_writers if prefix == "w" else token_readers
            target.setdefault(token, []).append(step)
    for token, its_writers in token_writers.items():
        by_rank = sorted(its_writers, key=rank.__getitem__)
        for i, low in enumerate(by_rank):
            for high in by_rank[i + 1 :]:
                edges.append(DerivedEdge(low, high, "token_ww", token))
        for reader in token_readers.get(token, ()):
            for writer in its_writers:
                if writer == reader:
                    continue
                if rank[writer] < rank[reader]:
                    edges.append(DerivedEdge(writer, reader, "token_raw", token))
                else:
                    edges.append(DerivedEdge(reader, writer, "token_war", token))

    # Rule 4: two-sided row-structure barriers.
    row_steps = [s for s, c in contracts.items() if c.row_effects]
    op_steps = [s for s, c in contracts.items() if _op_touching(c)]
    for row_step in row_steps:
        for other in op_steps:
            if other == row_step:
                continue
            pair = _directed(row_step, other)
            if pair is not None:
                edges.append(DerivedEdge(pair[0], pair[1], "row_barrier", ""))

    # Rule 6: full barriers.
    barrier_steps = [s for s, c in contracts.items() if c.barrier]
    for barrier_step in barrier_steps:
        for other in contracts:
            if other == barrier_step:
                continue
            pair = _directed(barrier_step, other)
            if pair is not None:
                edges.append(DerivedEdge(pair[0], pair[1], "barrier", ""))

    return edges


def derived_pinnable_pairs() -> dict[tuple[str, str], set[str]]:
    """Return the RAW/WW pair corpus view: (producer, consumer) -> carriers.

    This is check K2's left-hand side (and the one-time corpus seed input):
    every derived producer->consumer fact, as step pairs with the column or
    token names that carry it. WAR pairs are excluded — a WAR edge is not a
    producer->consumer fact, but hoisting the reader past the writer under
    a coordinated reversal MANUFACTURES a new RAW pair, which lands here
    and turns K2 red until pinned with a reason.
    """

    pairs: dict[tuple[str, str], set[str]] = {}
    for edge in derive_edges():
        if edge.kind in ("raw", "ww"):
            pairs.setdefault((edge.src, edge.dst), set()).add(edge.carrier)
        elif edge.kind in ("token_raw", "token_ww"):
            # Token carriers are namespaced so a trace-state token can never
            # collide with (or launder through) an op-column name.
            pairs.setdefault((edge.src, edge.dst), set()).add(f"token:{edge.carrier}")
    return pairs


def classify_declared_reads(
    noop_writers: Mapping[str, frozenset[str]] | None = None,
) -> dict[tuple[str, str], str]:
    """Classify every declared op-column read (design-ppdag-v3 §2.4).

    Returns ``(step, column) -> category`` over the declared reads:
    ``baseline`` (capture-populated), ``probe`` (reviewed placeholder
    probe), ``self_write`` (designed intra-step read-modify-write: the
    step reads a column it also writes; the window-granular audit cannot
    order the read against the write, so the discharge rests on
    effectiveness evidence — guard 2 — plus review), ``earlier_writer``
    (RAW dependency on a lower-rank writer), or ``finding`` — a read of a
    manifest-excluded column with no content-effective writer to discharge
    it: an undeclared dependency or phantom read, the real latent-bug
    class. Findings are PINNED by name in ``test_postprocess_dag.py``; a
    new one fails there and is root-caused, never silenced.

    ``noop_writers`` is guard 2's ledger: the reviewed pinned table of
    writers whose intercepted writes are never content-effective on any
    matrix axis (``PINNED_NOOP_WRITERS``, the static mirror of
    ``RECORDED_STEP_EFFECTIVE_WRITES``). A pinned no-op writer CANNOT
    discharge a read — neither as the step's own ``self_write`` nor as an
    ``earlier_writer`` — because a write that never changes the cell
    anywhere is exactly the laundering path the design names (§2.4
    guard 2: "a writer that never changes c anywhere in the matrix is a
    finding"). ``None`` (the default) skips the ledger: classification is
    then purely structural, which is LENIENT — the pinned-findings test
    always passes the ledger.
    """

    from ._contracts import CAPTURE_BASELINE_COLUMNS

    contracts = _registry_contracts()
    rank = LEGACY_STEP_RANK
    writers: dict[str, list[str]] = {}
    for step, contract in contracts.items():
        for column in contract.writes:
            writers.setdefault(column, []).append(step)

    def _discharges(writer: str, column: str) -> bool:
        """Whether ``writer``'s declared write of ``column`` can discharge a read.

        A writer recorded as a permanent no-op for that column never discharges:
        that is exactly the laundering path guard 2 names. With no ledger
        supplied every declared write discharges, which is the LENIENT direction.
        """

        if noop_writers is None:
            return True
        return column not in noop_writers.get(writer, frozenset())

    classified: dict[tuple[str, str], str] = {}
    for step, contract in contracts.items():
        for column in contract.reads:
            if column in CAPTURE_BASELINE_COLUMNS:
                classified[(step, column)] = "baseline"
            elif column in contract.placeholder_probes:
                classified[(step, column)] = "probe"
            elif column in contract.writes and _discharges(step, column):
                classified[(step, column)] = "self_write"
            elif any(
                rank[w] < rank[step] and _discharges(w, column) for w in writers.get(column, ())
            ):
                classified[(step, column)] = "earlier_writer"
            else:
                classified[(step, column)] = "finding"
        for column in contract.placeholder_probes - contract.reads:
            classified[(step, column)] = "probe"
    return classified


def execution_order(rank: dict[str, int] | None = None) -> tuple[str, ...]:
    """Derive the execution order: Kahn with a min-heap keyed by rank.

    Ranks are unique integers, so no secondary key exists (the historical
    "registry-index tie-break" formulation is gone); R2 asserts the emitted
    order equals ``REGISTRY_ORDER``. ``rank`` exists for the swap tests.
    """

    if rank is None:
        rank = dict(LEGACY_STEP_RANK)
    contracts = _registry_contracts()
    indegree: dict[str, int] = dict.fromkeys(contracts, 0)
    successors: dict[str, set[str]] = {step: set() for step in contracts}
    for edge in derive_edges(rank):
        if edge.dst not in successors[edge.src]:
            successors[edge.src].add(edge.dst)
            indegree[edge.dst] += 1
    heap = [(rank[step], step) for step, degree in indegree.items() if degree == 0]
    heapq.heapify(heap)
    order: list[str] = []
    while heap:
        _, step = heapq.heappop(heap)
        order.append(step)
        for successor in successors[step]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(heap, (rank[successor], successor))
    if len(order) != len(contracts):
        cyclic = sorted(step for step, degree in indegree.items() if degree > 0)
        raise ValueError(
            f"Postprocess dependency derivation found a cycle involving "
            f"{cyclic}; declared contracts contradict LEGACY_STEP_RANK."
        )
    return tuple(order)


def _iter_structural_violations(
    registry_order: tuple[str, ...] | None = None,
    rank: dict[str, int] | None = None,
) -> Iterator[str]:
    """Yield every 7.1-family structural violation (import checks).

    ``registry_order``/``rank`` exist for the swap regression tests, which
    prove a permuted registry (and a coordinated rank+registry reversal)
    refuse by name; production imports pass nothing.
    """

    contracts = _registry_contracts()
    if registry_order is None:
        registry_order = REGISTRY_ORDER
    if rank is None:
        rank = dict(LEGACY_STEP_RANK)
    # 7.1-1: registry <-> contracts bijection (step "0" exempted).
    registry_set = set(registry_order)
    contract_set = set(contracts)
    if registry_set != contract_set:
        yield (
            f"registry/contract mismatch: contracts without registry entry "
            f"{sorted(contract_set - registry_set)}, registry steps without "
            f"contract {sorted(registry_set - contract_set)}"
        )
        return
    # 7.1-3 (R1): registry strictly rank-ascending.
    registry_ranks = [rank[step] for step in registry_order]
    for i in range(1, len(registry_ranks)):
        if registry_ranks[i - 1] >= registry_ranks[i]:
            yield (
                f"registry order contradicts LEGACY_STEP_RANK at steps "
                f"{registry_order[i - 1]!r}, {registry_order[i]!r} (R1)"
            )
    # 7.1-5: token read-before-write analogue.
    token_writer_ranks: dict[str, int] = {}
    for step, contract in contracts.items():
        for entry in contract.trace_state:
            prefix, _, token = entry.partition(":")
            if prefix == "w":
                current = token_writer_ranks.get(token)
                step_rank = rank[step]
                if current is None or step_rank < current:
                    token_writer_ranks[token] = step_rank
    for step, contract in contracts.items():
        for entry in contract.trace_state:
            prefix, _, token = entry.partition(":")
            if prefix != "r" or token in CAPTURE_BASELINE_TOKENS:
                continue
            earliest = token_writer_ranks.get(token)
            if earliest is None or earliest >= rank[step]:
                yield (
                    f"step {step} declares r:{token} with no lower-rank "
                    f"w:{token} and {token!r} is not capture-baseline "
                    "(7.1-5)"
                )
    # 7.1-4 (R2): the derived order reproduces the registry.
    derived = execution_order(rank)
    if derived != registry_order:
        yield (f"derived execution order diverges from the registry (R2): derived={derived!r}")


def _validate_derivation() -> None:
    """Run the 7.1 checks; refuse import on the first violation, by name."""

    violations = list(_iter_structural_violations())
    if violations:
        raise ValueError("Postprocess DAG structural checks failed:\n- " + "\n- ".join(violations))


_validate_derivation()
