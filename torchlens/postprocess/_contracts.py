"""Postprocess step contracts, frozen rank, and the pinned-pair corpus.

The three frozen artifacts of the postprocess dependency derivation
(design-ppdag-v3):

- ``POSTPROCESS_STEP_CONTRACTS`` — each step's declared writes/reads/
  probes/row_effects/trace_state (the derivation's inputs).
- ``LEGACY_STEP_RANK`` — key 1 of the two-key direction authority: the
  historically-established order as a frozen semantic constant. Every edge
  orients by rank; registry position is never an input to derivation.
- ``PINNED_ORDER_PAIRS`` — key 2: the named producer->consumer pair corpus.
  Reordering steps mechanically requires editing a reason-bearing corpus
  entry (import check K1), and every newly derived RAW/WW edge must be
  pinned before CI passes (test-side check K2).

The read-triggers-write class (design-ppdag-v3 §5.6): exactly one member is
live inside steps 1-20 — reading ``op.arg_expressions`` writes
``_arg_expressions_cache`` (a lazy cache behind a property). The ``_CSR``/
``_FACT`` fact-block hydration writes can only fire after the freeze
installs their sentinels, which happens in the epilogue AFTER step 20 —
out-of-window by construction. PRECOMMIT: a write-audit trip on a
``*_cache`` column from a read site is ROOT-CAUSED (is the read intended?
is the cache column declared for that step?), never resolved by
reflexively widening a declared set.

Named model limits (unchanged by the derivation): warnings and
first-exception order are pinned solely by day-1 order identity — any
future reorder adds a warnings/exception-order review gate; deferred
gradient streaming re-runs step-18/19-equivalent code after backward,
outside the pipeline and its windows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from torchlens.ir.op_record_manifest import CELL_SOURCE_MANIFEST

#: Closed trace-state token vocabulary (design-ppdag-v3 §2.2). Each token
#: names one non-column state surface a step may consume or produce; the
#: derivation orients token conflicts exactly like column conflicts. The
#: three workspace tokens are capture-produced (legal to read with no
#: earlier pipeline writer — the token analogue of the capture baseline).
_TRACE_STATE_TOKENS: frozenset[str] = frozenset(
    (
        "raw_graph_ws",
        "module_capture_ws",
        "wrapper_runtime_ws",
        "label_maps",
        "lookup_containers",
        "conditional_records",
        "layer_logs",
        "module_logs",
        "module_build",
        "param_logs_kind",
        "stream_writer",
        "payload_tensors",
        "timing",
        "warnings",
        "cuda_cache",
        "finished_flag",
        "saved_summary",
        "graph_hash",
        "containers",
        "stream_lifecycle",
    )
)

#: Tokens produced by capture/step 0 itself: a declared read with no earlier
#: pipeline writer is legal for exactly these (import check 7.1-5). Beyond
#: the three per-phase workspaces: param_logs_kind because the raw ParamLog
#: kind records are built during capture (step 15 finalizes pre-existing
#: state — its declared rw: read has no earlier pipeline writer), and
#: stream_writer because disk streaming creates the writer and streams
#: payloads DURING the forward (step 18's rw: read consumes capture-created
#: writer state).
CAPTURE_BASELINE_TOKENS: frozenset[str] = frozenset(
    (
        "raw_graph_ws",
        "module_capture_ws",
        "wrapper_runtime_ws",
        "param_logs_kind",
        "stream_writer",
    )
)


def tokens(*declarations: str) -> frozenset[str]:
    """Normalize trace-state token declarations to the stored vocabulary.

    The stored vocabulary is ``r:<token>`` / ``w:<token>`` ONLY. ``rw:<token>``
    is construction-time shorthand expanded to both entries; any other prefix
    or unknown token raises at import time (design-ppdag-v3 §2.2).
    """

    normalized: set[str] = set()
    for declaration in declarations:
        prefix, _, token = declaration.partition(":")
        if token not in _TRACE_STATE_TOKENS:
            raise ValueError(
                f"Unknown trace-state token {token!r} in {declaration!r}; the "
                "closed vocabulary lives in _TRACE_STATE_TOKENS and growing it "
                "is a reviewed contract diff."
            )
        if prefix == "rw":
            normalized.add(f"r:{token}")
            normalized.add(f"w:{token}")
        elif prefix in ("r", "w"):
            normalized.add(declaration)
        else:
            raise ValueError(
                f"Invalid trace-state prefix {prefix!r} in {declaration!r}; "
                "only r:/w: are stored (rw: is construction-time shorthand)."
            )
    return frozenset(normalized)


@dataclass(frozen=True)
class PostprocessStepContract:
    """Declared contract for one postprocess pipeline step.

    Parameters
    ----------
    step:
        Stable step identifier used by the pipeline.
    name:
        Human-readable step name.
    contract:
        Short consumes/produces/mutation contract.
    writes:
        Declared op-store COLUMN write set for this step (M10): the exact
        cell columns the step may write or delete, enforced under
        ``TORCHLENS_POSTPROCESS_ASSERTIONS`` by the zero-cost-when-off
        write audit (``op_store.begin_cell_write_audit``). The former
        ``None`` wildcard is DELETED (design-ppdag-v3): every step declares
        an exact set. A step writing an undeclared column fails the
        tripwire; widening a set is a REVIEWED schema-contract diff, never
        a silent drift.
    reads:
        Declared op-store COLUMN read set. Authority for the dependency
        derivation: seeded from the recording matrix, hand-reviewed against
        the step source, shipped as a reviewed diff (recordings are
        evidence, never auto-regenerated declarations).
    placeholder_probes:
        Reviewed reads that legally observe the step-0 schema placeholder
        ("has this been set yet"). Exempt from read-before-write findings,
        NEVER from WAR edges — a probe's correctness depends on staying
        pinned before the column's writer.
    row_effects:
        Whole-row lifecycle sanctions: ``"creates"`` (the step may build op
        rows — also the legality condition for row-clone reads) and/or
        ``"deletes"`` (removal husking releases every cell of the row).
        Row lifecycle is audited separately from column writes, so an
        unsanctioned removal fails with a precise message instead of a
        wall of column names. A step with either effect is a two-sided
        barrier against every op-column-touching step (edge rule 4).
    trace_state:
        Non-column state tokens, ``r:<token>``/``w:<token>`` over the
        closed ``_TRACE_STATE_TOKENS`` vocabulary. Construct with
        ``tokens()`` so ``rw:`` shorthand normalizes and typos refuse at
        import.
    barrier:
        Full ordering barrier (step 17 only: ``_tracing_finished`` flips
        global facade behavior).
    """

    step: str
    name: str
    contract: str
    writes: frozenset[str]
    reads: frozenset[str]
    placeholder_probes: frozenset[str] = frozenset()
    row_effects: frozenset[str] = frozenset()
    trace_state: frozenset[str] = frozenset()
    barrier: bool = False

    def __post_init__(self) -> None:
        """Refuse malformed contracts at construction (plain raise, not assert)."""

        if self.writes is None or self.reads is None:
            raise ValueError(
                f"Step {self.step}: writes/reads must be exact frozensets; the "
                "None wildcard is deleted (design-ppdag-v3 defect 1)."
            )
        unknown_effects = self.row_effects - {"creates", "deletes"}
        if unknown_effects:
            raise ValueError(
                f"Step {self.step}: unknown row_effects {sorted(unknown_effects)}; "
                "the vocabulary is {'creates', 'deletes'}."
            )
        for entry in self.trace_state:
            prefix, _, token = entry.partition(":")
            if prefix not in ("r", "w") or token not in _TRACE_STATE_TOKENS:
                raise ValueError(
                    f"Step {self.step}: invalid trace_state entry {entry!r}; "
                    "construct with tokens() (stored vocabulary is r:/w: over "
                    "_TRACE_STATE_TOKENS)."
                )


#: FROZEN semantic constant (design-ppdag-v3 §2.1, key 1 of the two-key
#: direction authority): the historically-established producer/consumer
#: order as ground truth. Every RAW/WW/WAR/row/token edge orients by this
#: rank; registry position is NOT an input to derivation. Editing it is a
#: reviewed semantic diff under the same governance as widening a write
#: set. It is NOT derived from the step registry and is NOT regenerated by
#: any tool. Step "0" is deliberately absent (fenced prologue, producer
#: lane); step "17.5" is the contracted container-adoption seam.
LEGACY_STEP_RANK: Mapping[str, int] = MappingProxyType(
    {
        "1": 10,
        "2": 20,
        "3": 30,
        "4": 40,
        "5": 50,
        "6": 60,
        "7": 70,
        "8": 80,
        "9": 90,
        "10": 100,
        "11": 110,
        "11.5": 115,
        "11.75": 118,
        "12": 120,
        "13": 130,
        "14": 140,
        "15": 150,
        "15.5": 155,
        "16": 160,
        "16.5": 165,
        "17": 170,
        "17.5": 175,
        "18": 180,
        "19": 190,
        "20": 200,
    }
)


POSTPROCESS_STEP_CONTRACTS: dict[str, PostprocessStepContract] = {
    # Step 0 runs before any audit window can arm (the store is born inside
    # it), so its writes line is vacuously green; it exists for the
    # producer-lane fence and is JOINT-SIGNOFF with that lane.
    "0": PostprocessStepContract(
        "0",
        "Materialize capture events",
        "Consumes capture events; rebuilds raw Op state; mutates Trace in place.",
        writes=frozenset(),
        reads=frozenset(),
        row_effects=frozenset(("creates",)),
        trace_state=tokens("w:raw_graph_ws"),
    ),
    "1": PostprocessStepContract(
        "1",
        "Add output layers",
        "Consumes model outputs and parent labels; produces output Ops in raw state.",
        writes=frozenset(
            (
                "_arg_expressions_cache",
                "_edge_uses",
                "_label_raw",
                "_layer_label_raw",
                "_param_barcodes",
                "_param_logs",
                "activation_memory",
                # The synthetic output node RESETS the clone-inherited per-op
                # annotations namespace: inheriting it duplicated a collective
                # boundary payload (incl. its correlation key) onto the output
                # node whenever a collective's result was returned directly
                # (fix/distributed-r3 R18).
                "annotations",
                "arg_names",
                "atomic_module_call",
                "autograd_memory",
                "bytes_delta_at_call",
                "bytes_peak_at_call",
                "children",
                "code_context",
                "container_path",
                "container_spec",
                "dropped_edge_tensor_args",
                "dtype",
                "equivalence_class",
                "equivalent_ops",
                # A1: output nodes reset clone-inherited compute (alias rows own none).
                "flops_backward",
                "flops_forward",
                "func",
                "func_config",
                "func_duration",
                "func_name",
                "func_non_tensor_args",
                "func_rng_states",
                "grad_fn_class_name",
                "has_children",
                "has_out_variations",
                "has_output_descendant",
                "input_to_module_calls",
                "intervention_replaced",
                "interventions",
                "io_role",
                "is_atomic_module",
                "is_buffer",
                "is_final_output",
                "is_input",
                "is_internal_source",
                "is_module_output",
                "is_output",
                "is_transform",
                # The synthetic output node RE-DERIVES this direct-parent relation
                # instead of inheriting it from the clone source, which named labels
                # that were not its parents at all (B3 R05 output-node ISP fix).
                "internal_source_parents",
                "module",
                "module_call_stack",
                "modules",
                "non_tensor_kwargs",
                "non_tensor_pos_args",
                "num_args_total",
                "num_autograd_tensors",
                "num_kwargs",
                "num_params",
                "num_params_frozen",
                "num_params_trainable",
                "num_passes",
                "num_pos_args",
                "out",
                "out_versions_by_child",
                "output_descendants",
                "output_of_module_calls",
                "output_of_modules",
                "param_memory",
                "param_shapes",
                "parent_arg_positions",
                "parent_param_ops",
                "parent_params",
                "parents",
                "pass_index",
                "raw_index",
                "recurrent_ops",
                # The synthetic output node RE-DERIVES root_ancestors as
                # input_ancestors | internal_source_ancestors over its ONE parent
                # (the armed ancestry-closure convention): the wholesale clone
                # inherited the empty set when the direct parent is a parentless
                # factory source (b9 R71-1 ancestry_closure fix).
                "root_ancestors",
                "saved_args",
                "saved_kwargs",
                "shape",
                "transform_chain",
                "transform_config",
                "transform_fn_name",
                "transform_fn_qualname",
                "transform_fn_source",
                "transform_kind",
                "transformed_activation_memory",
                "transformed_out",
                "transformed_out_dtype",
                "transformed_out_shape",
                "type",
                "unattributed_tensor_args",
                "var_names",
            )
        ),
        reads=frozenset(
            (
                "_label_raw",
                "_layer_label_raw",
                "_source_trace_ref",
                "_tracing_finished",
                "children",
                "dtype",
                "func_name",
                # Deriving the output node's internal_source_parents asks its parent
                # whether it carries internal-source ancestry (B3 R05 fix).
                "has_internal_source_ancestor",
                "has_saved_activation",
                # Deriving the output node's root_ancestors reads its parent's
                # ancestry closure sets (b9 R71-1 ancestry_closure fix).
                "input_ancestors",
                "internal_source_ancestors",
                "label",
                "layer_label",
                "modules",
                "out",
                "out_ref",
                "output_device",
                "shape",
                "transformed_out",
            )
        ),
        # Probes (reviewed): the transform/streaming path resolves
        # _streaming_label via the label -> layer_label -> _label_raw
        # fallback chain, and out_ref gates payload-ref reuse — all
        # three intentionally observe the not-yet-set placeholder and
        # fall back (op.py _streaming_label; graph_traversal _apply_transform).
        placeholder_probes=frozenset(
            (
                "label",
                "layer_label",
                "out_ref",
            )
        ),
        # Row creation carries the row-clone read legality (design-ppdag-v3
        # §2.4d): step 1 clones the output node via Op.copy(), whose
        # whole-schema getattr loop is a mechanical row_clone access kind,
        # not a per-column dependency.
        row_effects=frozenset(("creates",)),
        trace_state=tokens("rw:raw_graph_ws", "w:lookup_containers"),
    ),
    "2": PostprocessStepContract(
        "2",
        "Trace output ancestors",
        "Consumes raw graph links; mutates output-descendant ancestry flags in place.",
        # Reviewed widening (sol finding 6 in-place audit): the traversal
        # mutates each op's output_descendants staging SET in place.
        writes=frozenset(
            (
                "has_output_descendant",
                "output_descendants",
            )
        ),
        reads=frozenset(
            (
                "children",
                "has_output_descendant",
                "output_descendants",
            )
        ),
        trace_state=tokens("r:raw_graph_ws"),
    ),
    "3": PostprocessStepContract(
        "3",
        "Remove orphan nodes",
        "Consumes ancestry flags; removes or records orphan raw Ops in place.",
        # Design-ppdag-v3 defect 4, evidence-narrowed: step 3's undeclared
        # writes are two config-gated sets. (4a) keep_orphans=True on an
        # orphan-bearing model writes is_orphan on every retained orphan and
        # returns before the batch removal. (4b) default keep_orphans=False
        # runs the removal scrub (_batch_remove_log_entries with
        # remove_references=True), which rebinds SURVIVING rows'
        # equivalent_ops when an orphan shared an equivalence class
        # (verified live on the OrphanTensors fixture). Why the design's
        # wider parents/children scrub columns are NOT declared (opus
        # impl-review F3 corrected the original disconnected-components
        # claim): the undirected flood alone IS closed (every popped node
        # pushes children+parents), but
        # _expand_seen_nodes_to_complete_func_call_groups
        # (graph_traversal.py) then adds func-call-group siblings WITHOUT
        # flooding from them, so an expansion-added survivor can hold edges
        # to orphans and the scrub would rebind its parents/children. The
        # trigger needs a multi-output call sharing NO parent with any
        # flooded member — standard torch ops cannot produce it (siblings
        # share the call's inputs) — and if it ever fires, the write audit
        # fails LOUD on the undeclared column; declaring parents/children
        # today would be phantom declarations (the review-4 anti-laundering
        # guard). recurrent_ops additionally cannot fire: recurrence groups
        # are not built until step 7.
        writes=frozenset(
            (
                "_edge_uses",
                "args_template",
                "conditional_arm_children",
                "conditional_elif_children",
                "conditional_else_children",
                "conditional_entry_children",
                "conditional_then_children",
                "equivalent_ops",
                "interventions",
                "is_internal_sink",
                "is_orphan",
                "is_terminal_bool",
                "kwargs_template",
            )
        ),
        reads=frozenset(
            (
                "_edge_uses",
                "_label_raw",
                "_source_trace_ref",
                "_tracing_finished",
                "args_template",
                "buffer_write_kind",
                "children",
                "conditional_arm_children",
                "conditional_elif_children",
                "conditional_else_children",
                "conditional_entry_children",
                "conditional_then_children",
                "equivalent_ops",
                "func_call_id",
                "func_id",
                "has_saved_activation",
                "input_ancestors",
                "internal_source_ancestors",
                "internal_source_parents",
                "interventions",
                "is_output",
                "is_scalar_bool",
                "kwargs_template",
                "label",
                "layer_label",
                "out",
                "out_ref",
                "out_versions_by_child",
                "output_descendants",
                "parent_arg_positions",
                "parents",
                "recurrent_ops",
                "root_ancestors",
                "unattributed_tensor_args",
            )
        ),
        # Probes (reviewed): out_ref gates load/stream-rehydrated
        # payload refs in orphan_records; the five
        # conditional child views, and recurrent_ops are read by the
        # removal-reference scrub (cleanup.py), which observes their
        # not-yet-populated placeholders and tolerates them by design —
        # the writers run later (5/9 for conditional views, 7 for
        # recurrence groups). These six were
        # previously laundered through no-op self-writes / a no-op step-1
        # writer (opus impl-review B1); probe-blessing is the same
        # reviewed treatment used by the identical code path. layer_label is the
        # _label_for_reference_removal fallback (layer_label -> _label_raw;
        # cleanup.py) — probe-blessed per the opus impl-review §4 split
        # after confirming the second reader
        # (_materialize_layer_mirrors_for_removed) is benign here: it
        # early-returns until layer_logs exist (built at step 15.5), and if
        # that guard ever moved earlier, the mirror materialization's ~86
        # undeclared column reads would trip the enforcement axes long
        # before this probe could mask anything. The label read is NOT a
        # probe: orphan_records stores the never-populated placeholder as
        # DATA — the pinned day-1 category-(c) finding (design-ppdag-v3
        # §2.4), reported for root-cause, never silenced; the root-cause
        # fix (record _label_raw instead) changes a serialized public
        # field's content and is deferred to a maintainer decision by name.
        placeholder_probes=frozenset(
            (
                "conditional_arm_children",
                "conditional_elif_children",
                "conditional_else_children",
                "conditional_entry_children",
                "conditional_then_children",
                "layer_label",
                "out_ref",
                "recurrent_ops",
            )
        ),
        row_effects=frozenset(("deletes",)),
        trace_state=tokens("rw:raw_graph_ws"),
    ),
    "4": PostprocessStepContract(
        "4",
        "Input/output distances",
        "Consumes orphan-free graph; mutates distance fields in place.",
        writes=frozenset(
            (
                "has_input_ancestor",
                "has_output_descendant",
                # Reviewed widening (sol finding 6 in-place audit): the
                # distance traversal mutates input_ancestors sets in place.
                "input_ancestors",
                "max_distance_from_input",
                "max_distance_to_output",
                "min_distance_from_input",
                "min_distance_to_output",
            )
        ),
        reads=frozenset(
            (
                "children",
                "input_ancestors",
                "max_distance_from_input",
                "max_distance_to_output",
                "min_distance_from_input",
                "min_distance_to_output",
                "output_descendants",
                "parents",
            )
        ),
        trace_state=tokens("r:raw_graph_ws"),
    ),
    "5": PostprocessStepContract(
        "5",
        "Mark conditional branches",
        "Consumes orphan-free graph; mutates conditional metadata in place.",
        writes=frozenset(
            (
                "_is_in_conditional_body",
                "conditional_arm_children",
                "conditional_branch_depth",
                "conditional_branch_stack",
                "conditional_context_kind",
                "conditional_elif_children",
                "conditional_else_children",
                "conditional_entry_children",
                "conditional_then_children",
                "conditional_wrapper_kind",
                # Reviewed widening (sol finding 6 in-place audit): terminal
                # scalar-bool classification writes is_terminal_bool.
                "is_terminal_bool",
                "is_terminal_conditional_bool",
                "terminal_conditional_id",
            )
        ),
        reads=frozenset(
            (
                "_label_raw",
                "children",
                "code_context",
                "conditional_arm_children",
                "conditional_branch_stack",
                "conditional_entry_children",
                "has_output_descendant",
                "is_orphan",
                "is_scalar_bool",
                "is_terminal_conditional_bool",
                "parents",
                "pass_index",
                "terminal_conditional_id",
            )
        ),
        trace_state=tokens("r:raw_graph_ws", "w:conditional_records"),
    ),
    "6": PostprocessStepContract(
        "6",
        "Fix buffer layers",
        "Consumes buffer events and graph links; mutates buffer metadata in place.",
        # The buffer-merge write family (opus impl-review B2, evidence:
        # the buffer_duplicate matrix axis fires _merge_buffer_entries):
        # the merge rewires children/parents/parent_arg_positions across
        # surviving rows, rebinds internal_source_ancestors closure sets,
        # repoints scalar buffer_source cells at the survivor, and its
        # _remove_log_entry(remove_references=True) husking runs the same
        # reference scrub as step 3 (_edge_uses, equivalent_ops,
        # interventions, args/kwargs_template, conditional child views).
        # Correctly NOT declared: the guarded saved_args.append fires only
        # when the buffer row's own saved_args is non-None, and source rows
        # materialize it as None (backends/torch/sources.py) — same
        # unreachability class as internal_source_parents below, without
        # the code-real write sites that keep that one declared.
        writes=frozenset(
            (
                "_edge_uses",
                # B2 residual closure: the None-address recovery/anonymous
                # fallback (control_flow.py, from d8155654) assigns a
                # display address when a buffer row reaches step 6 without
                # one. No known capture path materializes such a row (every
                # buffer source-logging call site records an address, and
                # step 0 resolves it through the registered pool, the
                # equivalence-class recovery, and the recorded I/O address
                # in turn — bounded empirical sweep: registered read,
                # journal write, dynamic register_buffer, cooked recording,
                # top-level read). Kept declared as a named phantom
                # exemption; retires loudly the day a path produces one.
                "address",
                "args_template",
                "buffer_pass",
                "buffer_replay_validated",
                "buffer_source",
                "children",
                "conditional_arm_children",
                "conditional_elif_children",
                "conditional_else_children",
                "conditional_entry_children",
                "conditional_then_children",
                "equivalent_ops",
                "func",
                "func_name",
                "has_children",
                # The step-6 ancestry cone re-derivation keeps this flag in step with
                # the closure it recomputes (F-R05-3: descendants of a rewired buffer
                # kept pre-edge ancestry, so an op that demonstrably depended on the
                # model input reported no input ancestry at all).
                "has_internal_source_ancestor",
                "has_input_ancestor",
                # Reviewed widening (r3settle e12aa996): the duplicate-buffer
                # merge unions the removed duplicate's child-direction reach
                # into the survivor, and the ancestor-cone re-derivation
                # recomputes output reach in reverse topological order —
                # output_descendants/has_output_descendant were computed at
                # step 2 from PRE-MERGE edges and shipped stale on the
                # survivor and every ancestor otherwise.
                "has_output_descendant",
                "output_descendants",
                # B2 residual closure: the buffer-source ancestry fallback
                # (control_flow.py lines 827-828) copies the source's
                # input_ancestors onto journaled buffer version rows. Step 4
                # pre-propagates transitive ancestry whenever it runs — and
                # it runs on default captures — so this write is only
                # content-effective with layer depths OFF; the
                # buffer_from_input axis is exactly that configuration and
                # retired the former ("6", "has_input_ancestor") permanent
                # no-op row.
                "input_ancestors",
                "internal_source_ancestors",
                # Buffer merging repoints direct internal-source parents
                # materialized from the capture journal at step 0.
                "internal_source_parents",
                "interventions",
                "kwargs_template",
                "parent_arg_positions",
                "parents",
                # Reviewed widening (sol finding 6 in-place audit): buffer
                # rewiring mutates root_ancestors closure sets in place.
                "root_ancestors",
            )
        ),
        reads=frozenset(
            (
                "_edge_uses",
                "_label_raw",
                "_source_trace_ref",
                "_tracing_finished",
                "address",
                "args_template",
                "buffer_source",
                "children",
                "conditional_arm_children",
                "conditional_elif_children",
                "conditional_else_children",
                "conditional_entry_children",
                "conditional_then_children",
                # B2 residual closure: read only on the None-address
                # recovery branch (the buffer_ prefix strip), which no
                # known capture path reaches — the named phantom-READ
                # exemption paired with the address write above.
                "equivalence_class",
                "equivalent_ops",
                "has_input_ancestor",
                # Reviewed widening (r3settle e12aa996): the merge consults the
                # removed duplicate's child-direction reach flag, and the cone
                # re-derivation reads is_output plus each child's
                # output_descendants (declared below) to rebuild the closure.
                "has_output_descendant",
                "input_ancestors",
                "internal_source_ancestors",
                "internal_source_parents",
                "interventions",
                "is_output",
                "kwargs_template",
                "layer_label",
                "modules",
                "out",
                "out_versions_by_child",
                "output_descendants",
                "parent_arg_positions",
                "parents",
                "recurrent_ops",
                "root_ancestors",
                # B3 R05: the ancestry cone re-derivation reads the role flags plus the
                # parents' own closure sets (F-R05-3).
                "has_internal_source_ancestor",
                "is_input",
                "is_internal_source",
                "saved_args",
            )
        ),
        # Probes (reviewed): the merge-husking path observes three columns
        # in their placeholder state and tolerates it by design —
        # layer_label via _label_for_reference_removal's
        # layer_label -> _label_raw fallback (the second reader,
        # _materialize_layer_mirrors_for_removed, early-returns until
        # layer_logs exist at step 15.5, so it never fires here);
        # recurrent_ops via the removal scrub's group rebind (groups are
        # built at step 7).
        placeholder_probes=frozenset(
            (
                "layer_label",
                "recurrent_ops",
            )
        ),
        # Buffer dedup removes merged duplicate rows through the same husking
        # path as orphan removal (_remove_log_entry at control_flow.py:951);
        # previously unsanctioned — a latent released-row trip on any
        # buffer-merging axis (design-ppdag-v3 inventory row 6). Exercised
        # by the buffer_duplicate axis.
        row_effects=frozenset(("deletes",)),
        trace_state=tokens("rw:raw_graph_ws"),
    ),
    "7": PostprocessStepContract(
        "7",
        "Loop detection",
        "Consumes final raw graph structure; mutates recurrence/equivalence metadata.",
        writes=frozenset(
            (
                "_layer_label_raw",
                "equivalence_class",
                "num_passes",
                "pass_index",
                "recurrent_ops",
                # site_key_v1 minting: policy-independent structural-position
                # identity, minted here on BOTH the full-detection and the
                # degraded (recurrence_detection=False) paths.
                "site_key",
            )
        ),
        reads=frozenset(
            (
                "_label_raw",
                "_layer_label_raw",
                "_param_barcodes",
                "children",
                "equivalence_class",
                "equivalent_ops",
                "func_name",
                "is_buffer",
                "is_orphan",
                "modules",
                "multi_output_index",
                "non_tensor_kwargs",
                "non_tensor_pos_args",
                "parents",
                "raw_index",
                "recurrent_ops",
                # Same-step read-after-own-write: the grouping-graph builder
                # copies each op's just-minted key onto its RecurrenceNode.
                "site_key",
                # site-key type axis (the normalized op type token).
                "type",
            )
        ),
        trace_state=tokens("r:raw_graph_ws"),
    ),
    "8": PostprocessStepContract(
        "8",
        "Map labels",
        "Consumes raw labels and recurrence metadata; produces raw-to-final maps.",
        writes=frozenset(
            (
                "label",
                "label_short",
                "layer_label",
                "layer_label_short",
                "step_index",
                "type_index",
            )
        ),
        reads=frozenset(
            (
                "_label_raw",
                "label",
                "layer_label",
                "num_passes",
                "pass_index",
                "recurrent_ops",
                "step_index",
                "type",
                "type_index",
            )
        ),
        trace_state=tokens("r:raw_graph_ws", "w:label_maps"),
    ),
    "9": PostprocessStepContract(
        "9",
        "Log final info",
        "Consumes mapped labels; mutates final Op metadata and module build data.",
        writes=frozenset(
            (
                "_edge_uses",
                # Reviewed widening (closure review, enforcement leg over the
                # intervention/observer suites): the final-label rename in
                # _replace_layer_names_for_layer_entry rewrites raw parent
                # refs inside replay templates and intervention records —
                # cells that exist only on intervention-ready captures, an
                # axis absent from the six surface-oracle recording models.
                "args_template",
                "atomic_module_call",
                "children",
                "conditional_arm_children",
                "conditional_elif_children",
                "conditional_else_children",
                "conditional_entry_children",
                "conditional_then_children",
                # B3 R05: buffer_source holds a RAW label and is renamed here with
                # the rest of the label-bearing fields, so it no longer survives
                # into finished traces as a dangling raw label.
                "buffer_source",
                "equivalent_ops",
                "fx_call_index",
                "fx_qualpath",
                "input_ancestors",
                "internal_source_ancestors",
                # Direct internal-source parent references are renamed from
                # raw to final labels with the other graph edges.
                "internal_source_parents",
                # Same reviewed widening as args_template above.
                "interventions",
                "is_buffer",
                "is_input",
                "is_output",
                # Same reviewed widening as args_template above.
                "kwargs_template",
                "output_descendants",
                # Reviewed widening (sol finding 6 in-place audit): final-info
                # logging mutates parent_arg_positions dicts in place.
                "parent_arg_positions",
                "parents",
                "recurrent_ops",
                "root_ancestors",
                "step_index",
            )
        ),
        reads=frozenset(
            (
                "_edge_uses",
                # Reviewed widening (8ba75e99 superlinear-rename fix): the
                # recurrent_ops rename memo is indexed by every raw member
                # label, so the sweep reads each entry's own _label_raw as
                # the memo lookup key.
                "_label_raw",
                "_param_barcodes",
                "activation_memory",
                "args_template",
                "atomic_module_call",
                # B3 R05: buffer_source holds a raw label and is renamed here with
                # the rest of the label-bearing fields.
                "buffer_source",
                "children",
                "conditional_arm_children",
                "conditional_entry_children",
                "equivalent_ops",
                "func_duration",
                "func_name",
                "fx_call_index",
                "input_ancestors",
                "internal_source_ancestors",
                "internal_source_parents",
                "interventions",
                "kwargs_template",
                "label",
                "layer_label",
                "modules",
                # A07: the per-op num_params/param_memory reads left this step
                # -- trace/module parameter tallies now derive from the
                # object-deduplicated param_logs inventory, not per-layer sums.
                "out_versions_by_child",
                "output_descendants",
                "parent_arg_positions",
                "parents",
                "recurrent_ops",
                "root_ancestors",
                "type",
            )
        ),
        trace_state=tokens("r:label_maps", "rw:raw_graph_ws", "w:module_build"),
    ),
    "10": PostprocessStepContract(
        "10",
        "Rename labels",
        "Consumes raw-to-final maps; mutates graph references to final labels.",
        writes=frozenset(),
        reads=frozenset(("layer_label",)),
        trace_state=tokens("r:label_maps", "r:raw_graph_ws", "w:lookup_containers"),
    ),
    "11": PostprocessStepContract(
        "11",
        "Build lookup keys",
        "Consumes final labels; rebuilds final lookup containers in place.",
        writes=frozenset(
            (
                "input_to_module_calls",
                "lookup_keys",
                "module",
                # B3R7-R05-1: relabeled to canonical ModuleCall labels in
                # lockstep with ``modules`` (same containment fact).
                "module_call_stack",
                "modules",
                "ordinal_index",
                "output_of_module_calls",
            )
        ),
        reads=frozenset(
            (
                "_label_raw",
                # F20 D-17: the saved-summary refresh derives
                # saved_activation_memory from PHYSICAL retained payloads
                # (out/transformed_out + the facade's resolution guards).
                "_source_trace_ref",
                "_tracing_finished",
                "activation_memory",
                "address",
                "buffer_pass",
                "fx_call_index",
                "fx_qualpath",
                "has_saved_activation",
                "input_to_module_calls",
                "io_role",
                "is_buffer",
                "is_input",
                "is_orphan",
                "is_output",
                "label",
                "label_short",
                "layer_label",
                "layer_label_short",
                "module",
                "module_call_stack",
                "modules",
                "num_passes",
                "out",
                # F20 D-17: reading ``out`` on a payload-absent op (lookback
                # windows, disk-only routes) probes the ``out_ref`` slot via
                # the accessor's lazy-materialization guard, so the refresh's
                # byte-model read carries this companion column.
                "out_ref",
                "output_of_module_calls",
                "raw_index",
                "transformed_activation_memory",
                "transformed_out",
                "type",
                "unattributed_tensor_args",
            )
        ),
        # r:module_build makes the 9 -> 11 edge derivable: step 11 reads
        # module_build_data["module_num_calls"] (labeling.py:847).
        trace_state=tokens(
            "r:label_maps",
            "r:module_build",
            "rw:lookup_containers",
            "w:saved_summary",
            "w:warnings",
        ),
        # Probe (reviewed, F20 D-17): the saved-summary refresh reads ``out``
        # per saved op, and the accessor's lazy-materialization guard checks
        # ``out_ref`` whenever ``out`` is empty (lookback windows, disk-only
        # routes) — in-pipeline it always observes the not-yet-attached
        # placeholder (refs attach at step 18) and falls through, the same
        # reviewed shape as step 12's undecorate probe.
        placeholder_probes=frozenset(("out_ref",)),
    ),
    # Step 11.5 previously declared an EMPTY write set, silently wrong under
    # save_code_context=True where it assigns op.var_names on every op
    # (design-ppdag-v3 defect 3). The audit never tripped because no recorded
    # enforcement axis enabled save_code_context. Note: the design's expected
    # _arg_expressions_cache companion write does NOT fire here —
    # resolve_var_names never reads op.arg_expressions (verified), so
    # declaring it would be a phantom declaration.
    "11.5": PostprocessStepContract(
        "11.5",
        "Populate source var names",
        "Consumes code context; mutates Op var_names in place.",
        writes=frozenset(("var_names",)),
        reads=frozenset(
            (
                "code_context",
                "func_name",
                "type",
            )
        ),
        trace_state=tokens("r:raw_graph_ws"),
    ),
    # Step 11.75 previously had NO contract boundary, so its writes were
    # misattributed to step 12's window and only surfaced on the selective/
    # fastlog axis (deferred retention runs only with a capture session) —
    # an axis absent from the recorded contract runs (closure review,
    # enforcement leg). The declared set is retention's payload family:
    # saving a deferred out writes the payload cells and their derived
    # shape/dtype/memory metadata.
    "11.75": PostprocessStepContract(
        "11.75",
        "Resolve deferred retention",
        "Consumes deferred retention decisions; saves selected payloads.",
        writes=frozenset(
            (
                "activation_memory",
                "annotations",
                "dtype",
                "has_saved_activation",
                "out",
                "saved_args",
                "saved_kwargs",
                "shape",
                "transformed_activation_memory",
                "transformed_out",
                "transformed_out_dtype",
                "transformed_out_shape",
            )
        ),
        reads=frozenset(
            (
                "_layer_label_raw",
                "_source_trace_ref",
                "_tracing_finished",
                # F20 D-17: ``activation_memory`` left this read set — the
                # refresh here reads live payloads (every op saved by deferred
                # resolution holds its payload until step-19 eviction), so the
                # byte model's raw-field fallback is unreachable at this step.
                "annotations",
                "detach_saved_activations",
                "dtype",
                "func_name",
                "has_saved_activation",
                "is_inplace",
                "is_orphan",
                "label",
                "layer_label",
                "out",
                "output_device",
                "parents",
                "raw_index",
                "shape",
                "transformed_out",
                "type",
            )
        ),
        trace_state=tokens("w:payload_tensors"),
    ),
    "12": PostprocessStepContract(
        "12",
        "Undecorate tensors",
        "Consumes saved tensors; mutates payload wrappers in place.",
        writes=frozenset(),
        reads=frozenset(
            (
                "_source_trace_ref",
                "_tracing_finished",
                "has_saved_activation",
                "is_orphan",
                "layer_label",
                "out",
                "out_ref",
                "saved_args",
                "saved_kwargs",
                "transformed_out",
            )
        ),
        # Probe (reviewed): undecorate checks out_ref to decide whether
        # a payload lives behind a streamed ref.
        placeholder_probes=frozenset(("out_ref",)),
        trace_state=tokens("w:payload_tensors"),
    ),
    "13": PostprocessStepContract(
        "13",
        "Clear CUDA cache",
        "Runs optional CUDA allocator cleanup; leaves Trace metadata unchanged.",
        writes=frozenset(),
        reads=frozenset(),
        trace_state=tokens("r:payload_tensors", "w:cuda_cache"),
    ),
    "14": PostprocessStepContract(
        "14",
        "Log timing",
        "Consumes capture timestamps; mutates duration fields in place.",
        writes=frozenset(),
        reads=frozenset(),
        trace_state=tokens("w:timing"),
    ),
    "15": PostprocessStepContract(
        "15",
        "Finalize params",
        "Consumes Op param references; mutates Param reverse mappings.",
        # Reviewed widening (sol finding 6 in-place audit): param
        # finalization mutates the _param_logs containers in place.
        writes=frozenset(
            (
                "_param_logs",
                "parent_params",
            )
        ),
        reads=frozenset(
            (
                "_param_logs",
                "label",
                "layer_label",
            )
        ),
        trace_state=tokens("rw:param_logs_kind"),
    ),
    "15.5": PostprocessStepContract(
        "15.5",
        "Build layer logs",
        "Consumes final Op list; rebuilds aggregate Layer logs and pass index.",
        writes=frozenset(
            (
                "in_conditionals",
                "terminal_bool_for",
            )
        ),
        reads=frozenset(
            (
                "_is_in_conditional_body",
                "_param_logs",
                "_source_trace_ref",
                "activation_memory",
                # Reviewed widening (fix/mergedannot 130b3f80): Layer
                # construction seeds layer_log.annotations["collective"] as an
                # independent deep copy of the first pass's reserved
                # collective_boundary_v1 payload; only that reserved key is
                # consumed, other op annotation keys stay op-only.
                "annotations",
                "autograd_memory",
                "bool_value",
                "conditional_arm_children",
                "conditional_branch_stack",
                "conditional_entry_children",
                "device_ref",
                "dtype",
                "flops_backward",
                "flops_forward",
                "has_input_ancestor",
                "has_output_descendant",
                "in_conditionals",
                "io_role",
                "is_atomic_module",
                "label",
                "layer_label",
                "modules",
                "num_autograd_tensors",
                "parents",
                "pass_index",
                "shape",
                "terminal_bool_for",
                "terminal_conditional_id",
                "transformed_activation_memory",
                "transformed_out_dtype",
                "transformed_out_shape",
            )
        ),
        trace_state=tokens("r:conditional_records", "w:layer_logs"),
    ),
    "16": PostprocessStepContract(
        "16",
        "Build module logs",
        "Consumes module build data and layer logs; rebuilds Module/ModuleCall logs.",
        # Reviewed widening (sol finding 6 in-place audit): module-log
        # building mutates the _param_logs containers in place.
        #
        # Reviewed widening (r7 b1-opus R04-2): on any module CALL with more
        # than one output entry (LSTM/GRU cells, tuple-returning submodules)
        # ``_assign_output_roles`` reads ``multi_output_name`` (role dedup
        # against the ingest-scattered value) and ``_source_trace_ref``
        # (``output.source_trace`` to mirror the role onto the parent layer).
        # Both columns are written at capture ingress (step-0 ingest scatter /
        # op adoption), never by a pipeline step, so the widen adds NO
        # ordering edges -- it makes the derivation authority match the read
        # the executor's own audit proved. The defensive None-branch WRITE of
        # ``multi_output_name`` at finalization._assign_output_roles stays
        # deliberately UNDECLARED: no constructed axis reaches it (round-6
        # instrumentation intercepted zero writes), so if one ever does, the
        # write audit must fire as a fresh finding rather than be pre-blessed.
        writes=frozenset(("_param_logs",)),
        reads=frozenset(
            (
                "_grad_records",
                "_source_trace_ref",
                "address",
                "container_spec",
                "has_saved_activation",
                "input_to_module_calls",
                "is_buffer",
                "is_module_output",
                "is_orphan",
                "label",
                "layer_label",
                # B3R7-R05-1 note: ``module_call_stack`` is no longer read
                # here -- the module-log builder keys pass-input membership on
                # ``input_to_module_calls`` alone (the old emptiness guard was
                # an enter-lane proxy for the same fact).
                "multi_output_name",
                "output_of_module_calls",
                "raw_index",
            )
        ),
        # Probe (reviewed, manifest-swap 2026-08-13): module-log grad
        # summaries read the backward-phase _grad_records channel, which
        # holds the step-0 constant seed on every in-pipeline axis (grads
        # are written post-backward, outside the pipeline and its windows)
        # — observe-empty-and-fall-through.
        placeholder_probes=frozenset(("_grad_records",)),
        trace_state=tokens(
            "r:layer_logs",
            "r:module_build",
            "r:module_capture_ws",
            "rw:param_logs_kind",
            "rw:saved_summary",
            "w:module_logs",
        ),
    ),
    "16.5": PostprocessStepContract(
        "16.5",
        "Graph shape hash",
        "Consumes final graph; mutates normalized addresses and graph hash.",
        writes=frozenset(("_address_normalized",)),
        reads=frozenset(
            (
                "container_path",
                "container_spec",
                "func_name",
                "is_buffer",
                "is_input",
                "is_output",
                "label",
                "layer_label",
                "module",
                "num_passes",
                "parents",
                "type",
            )
        ),
        trace_state=tokens("r:lookup_containers", "w:graph_hash"),
    ),
    "17": PostprocessStepContract(
        "17",
        "Mark pass finished",
        "Consumes finalized containers; mutates Trace to user-facing finished state.",
        writes=frozenset(("_tracing_finished",)),
        reads=frozenset(),
        trace_state=tokens("w:finished_flag"),
        barrier=True,
    ),
    # Step 17.5: the container-adoption + workspace-drop seam, contracted at
    # its exact historical position between 17 and the streaming snapshot
    # (design-ppdag-v3 §5.3). Trace-side only: adopts the wrapper runtime
    # registry's container records into trace._containers and drops all
    # three per-phase workspaces (terminal consumes). Step 18 declares
    # r:containers, which is the edge that makes this seam's position
    # derivable. Unwrapped by _vtimed today — stays unwrapped.
    "17.5": PostprocessStepContract(
        "17.5",
        "Adopt containers, drop workspaces",
        "Consumes wrapper runtime registry; adopts container records; drops workspaces.",
        writes=frozenset(),
        reads=frozenset(),
        trace_state=tokens(
            "r:wrapper_runtime_ws",
            "w:containers",
            "w:raw_graph_ws",
            "w:module_capture_ws",
            "w:wrapper_runtime_ws",
        ),
    ),
    # Steps 18/19 previously declared writes=None (wildcard), which the audit
    # skipped entirely — the streaming axis ran unaudited (design-ppdag-v3
    # defect 1). The sets below are HAND-DERIVED from finalization.py and
    # verified by a streaming recording run: step 18's only live-op writes are
    # the LazyActivationRef attachments in _attach_streamed_tensor_refs
    # (out_ref always; grad_ref only when grads streamed), and step 19's are
    # the _internal_set evictions (out always; transformed_out only when a
    # transformed payload was streamed). The grad/transform halves are
    # config-gated, not phantom.
    "18": PostprocessStepContract(
        "18",
        "Finalize streamed bundle",
        "Consumes stream writer state; finalizes bundle metadata in place.",
        writes=frozenset(
            (
                "grad_ref",
                "out_ref",
            )
        ),
        reads=frozenset(
            (
                "_address_normalized",
                "_arg_expressions_cache",
                "_construction_done",
                "_edge_uses",
                "_facets_cache",
                "_grad_records",
                "_is_in_conditional_body",
                "_label_raw",
                "_layer_label_raw",
                "_param_barcodes",
                "_param_logs",
                "_pending_blob_id",
                "_pending_grad_blob_id",
                "_pending_transformed_grad_blob_id",
                "_pending_transformed_out_blob_id",
                "_projective_field_cache",
                "_receptive_field_cache",
                "_source_trace_ref",
                "_tracing_finished",
                "activation_memory",
                "activation_transform",
                "address",
                "annotations",
                "arg_names",
                "args_template",
                "atomic_module_call",
                "autograd_memory",
                "backend_address",
                "bool_value",
                "buffer_pass",
                "buffer_replay_validated",
                "buffer_source",
                "buffer_source_func_name",
                "buffer_value_changed",
                "buffer_write_kind",
                "bytes_delta_at_call",
                "bytes_peak_at_call",
                "children",
                "code_context",
                "conditional_arm_children",
                "conditional_branch_depth",
                "conditional_branch_stack",
                "conditional_context_kind",
                "conditional_elif_children",
                "conditional_else_children",
                "conditional_entry_children",
                "conditional_then_children",
                "conditional_wrapper_kind",
                "container_path",
                "container_spec",
                "detach_saved_activations",
                "device_ref",
                "dropped_edge_tensor_args",
                "dtype",
                "dtype_ref",
                # tlspec v9 entry-dark KEEP columns (episode_step,
                # injection_provenance, tl_authored_root): the streamed-bundle
                # writer reads every declared portable field to apply its
                # policy, so the v9 additions join the hand-derived set as a
                # reviewed contract diff.
                "episode_step",
                "equivalence_class",
                "equivalent_ops",
                "flops_backward",
                "flops_forward",
                "func",
                "func_autocast_state",
                "func_call_id",
                "func_config",
                "func_duration",
                "func_id",
                "func_name",
                "func_non_tensor_args",
                "func_qualname",
                "func_rng_states",
                "fx_call_index",
                "fx_qualpath",
                "grad",
                "grad_dtype",
                "grad_fn",
                "grad_fn_class_name",
                "grad_fn_class_qualname",
                "grad_fn_handle",
                "grad_fn_object_id",
                "grad_ref",
                "grad_shape",
                "gradient_memory",
                "has_children",
                "has_grad",
                "has_input_ancestor",
                "has_internal_source_ancestor",
                "has_out_variations",
                "has_output_descendant",
                "has_saved_activation",
                "has_saved_args",
                "in_conditionals",
                "in_multi_output",
                "injection_provenance",
                "input_ancestors",
                "input_to_module_calls",
                "input_was_parameter",
                "internal_source_ancestors",
                "internal_source_parents",
                "intervention_replaced",
                "interventions",
                "io_role",
                "is_atomic_module",
                "is_buffer",
                "is_final_output",
                "is_inplace",
                "is_input",
                "is_internal_sink",
                "is_internal_source",
                "is_module_output",
                "is_orphan",
                "is_output",
                "is_output_parent",
                "is_scalar_bool",
                "is_terminal_bool",
                "is_terminal_conditional_bool",
                "is_transform",
                "kwargs_template",
                "label",
                "label_short",
                "layer_label",
                "layer_label_short",
                "lookup_keys",
                "max_distance_from_input",
                "max_distance_to_output",
                "min_distance_from_input",
                "min_distance_to_output",
                "module",
                "module_call_stack",
                "module_entry_arg_keys",
                "modules",
                "multi_output_index",
                "multi_output_name",
                "non_tensor_kwargs",
                "non_tensor_pos_args",
                "num_args_total",
                "num_autograd_tensors",
                "num_kwargs",
                "num_params",
                "num_params_frozen",
                "num_params_trainable",
                "num_passes",
                "num_pos_args",
                "ordinal_index",
                "out",
                "out_ref",
                "out_versions_by_child",
                "output_descendants",
                "output_device",
                "output_of_module_calls",
                "output_of_modules",
                "param_memory",
                "param_shapes",
                "parent_arg_positions",
                "parent_param_ops",
                "parent_params",
                "parents",
                "pass_index",
                "raw_index",
                "recurrent_ops",
                "resolver_status",
                "root_ancestors",
                "save_grads",
                "saved_args",
                "saved_kwargs",
                "shape",
                # Whole-record scrub: the streamed-bundle writer reads every
                # declared portable field to apply its policy (site_key is
                # DROP under v7 and read-then-dropped here).
                "site_key",
                "step_index",
                "terminal_bool_for",
                "terminal_conditional_id",
                "tl_authored_root",
                "transform_chain",
                "transform_config",
                "transform_fn_name",
                "transform_fn_qualname",
                "transform_fn_source",
                "transform_kind",
                "transformed_activation_memory",
                "transformed_grad",
                "transformed_grad_dtype",
                "transformed_grad_shape",
                "transformed_gradient_memory",
                "transformed_out",
                "transformed_out_dtype",
                "transformed_out_shape",
                "type",
                "type_index",
                "unattributed_tensor_args",
                "var_names",
                "visualizer_path",
            )
        ),
        # Probes (reviewed): the whole-row portable scrub serializes
        # every set cell, legally observing unset lazy caches. The grad
        # family (manifest-swap 2026-08-13) is the backward-phase channel:
        # in-pipeline every one of these holds the step-0 constant seed
        # (grads are written post-backward, when the deferred-grad
        # streaming re-run executes this same body OUTSIDE the pipeline
        # and its windows — the fact already carried by the
        # ('18','grad_ref') phantom-write exemption); the reads
        # observe-placeholder-and-fall-through. The L6 tier-(ii) pair
        # (persisted since the v8 bump; written only by session-time
        # fork.do() on a FINISHED trace, so the scrub always observes the
        # step-0 empty-dict seed) is reviewed in PROBES_GOLDEN. The tlspec
        # v9 entry-dark trio (episode_step / injection_provenance /
        # tl_authored_root) is the same reviewed shape: their Phase-3
        # writers (F-EPISODE / F01 / F41) run outside this pipeline, so
        # in-pipeline the scrub always observes the step-0 None seed.
        placeholder_probes=frozenset(
            (
                "_facets_cache",
                "_grad_records",
                "_pending_grad_blob_id",
                "_pending_transformed_grad_blob_id",
                "_projective_field_cache",
                "_receptive_field_cache",
                "edge_replacement_stamps",
                "edge_substitutions",
                "episode_step",
                "grad",
                "grad_dtype",
                "grad_fn",
                "grad_shape",
                "gradient_memory",
                "has_grad",
                "injection_provenance",
                "tl_authored_root",
                "transformed_grad",
                "transformed_grad_dtype",
                "transformed_grad_shape",
                "transformed_gradient_memory",
            )
        ),
        trace_state=tokens(
            "r:containers",
            "r:payload_tensors",
            "rw:stream_writer",
            "w:stream_lifecycle",
        ),
    ),
    "19": PostprocessStepContract(
        "19",
        "Evict streamed outs",
        "Consumes finalized stream state; drops in-memory output payloads.",
        writes=frozenset(
            (
                "out",
                "transformed_out",
            )
        ),
        reads=frozenset(
            (
                "_pending_transformed_out_blob_id",
                "out_ref",
            )
        ),
        trace_state=tokens("r:stream_writer", "rw:stream_lifecycle"),
    ),
    "20": PostprocessStepContract(
        "20",
        "Release param refs",
        "Consumes finalized Param logs; drops live parameter references in place.",
        writes=frozenset(),
        reads=frozenset(),
        # r:stream_lifecycle is the explicit hand-declared token that makes
        # the 18/19 -> 20 ordering derivable (release after optional stream
        # finalization, AGENTS.md).
        trace_state=tokens("rw:param_logs_kind", "r:stream_lifecycle"),
    ),
}


#: The manifest source classes whose columns are capture-populated (legal
#: to read with no earlier pipeline writer) vs excluded (placeholder until
#: a pipeline step writes). Vocabulary of the FROZEN CellSourceManifest v1
#: (producer-seam-v1: ``CORE | FACET:<name> | JOIN:<lane> | STEP /
#: DERIVED:init | DEFAULT | EXTRAS:<key> | NO_PRODUCER``); the projection
#: refuses unknown classes at call time so a manifest v2 with a new class
#: (e.g. an aten lane) must extend this table explicitly (review note N5).
#: NO_PRODUCER (pure lazy caches) is excluded per design-ppdag-v3 §2.4.
_MANIFEST_BASELINE_CLASSES: frozenset[str] = frozenset(("CORE", "FACET", "JOIN", "EXTRAS"))
_MANIFEST_EXCLUDED_CLASSES: frozenset[str] = frozenset(("STEP", "DEFAULT", "NO_PRODUCER"))

#: INTEGRATION FINDING (2026-08-13, manifest-swap diff): manifest v1's
#: ``DERIVED:init`` class CONFLATES two baseline behaviors and cannot
#: project as a class. Five columns are derived inside ``Op.__init__``
#: FROM CAPTURE FIELDS during step-0 row construction (op.py: dtype_ref
#: <- dtype, device_ref <- out/output_device, backend_address <- address,
#: resolver_status <- "resolved" disposition, _source_trace_ref <- the
#: source_trace weakref), so they hold real content at step-0 exit and ARE
#: baseline. Two (``out_ref``/``grad_ref``) are bare ``None`` defaults
#: until step 18 or artifact load writes them — placeholder, NOT baseline.
#: A NEW DERIVED:init column refuses projection until classified into one
#: of these two tables; manifest v2 should split the class (cross-lane).
_MANIFEST_DERIVED_INIT_BASELINE: frozenset[str] = frozenset(
    ("_source_trace_ref", "backend_address", "device_ref", "dtype_ref", "resolver_status")
)
_MANIFEST_DERIVED_INIT_EXCLUDED: frozenset[str] = frozenset(("out_ref", "grad_ref"))

#: INTEGRATION FINDING (2026-08-13, manifest-swap diff): the manifest
#: generator flattened per-row override lanes to the scatter's default
#: class. These four columns are classed ``DEFAULT`` but step-0 ingest
#: fills them with REAL event content on the rows that have it
#: (code-verified: buffer-write sibling fields at _materialize.py's
#: buffer-event builder; ``multi_output_name`` from ``event.output_names``
#: — the scatter spec's own ``buffer_write_kind`` row carries the
#: "JOIN:buffer_write override when present" comment). Union-over-rows
#: baseline semantics therefore include them. Each row here must still be
#: classed in an EXCLUDED head by the manifest — the projection refuses a
#: stale promotion row the day manifest v2 reclassifies the column.
_MANIFEST_FLATTENED_OVERRIDE_BASELINE: frozenset[str] = frozenset(
    ("buffer_source_func_name", "buffer_value_changed", "buffer_write_kind", "multi_output_name")
)

#: Manifest rows that are step-0 INPUT CHANNELS, not readable op-store
#: columns (``source_trace`` is consumed into the ``_source_trace_ref``
#: cell; ``_materialized_backend_address`` is an extra-key channel applied
#: to the buffer address). They never appear in the op-store layout, so
#: the baseline (a set of readable columns) skips them.
_MANIFEST_INPUT_CHANNEL_ROWS: frozenset[str] = frozenset(
    ("source_trace", "_materialized_backend_address")
)


def capture_baseline_from_manifest(source_classes: Mapping[str, str]) -> frozenset[str]:
    """Project a CellSourceManifest column->class map onto the baseline.

    Total over the frozen class vocabulary: an unknown class raises rather
    than silently classifying (fail-closed against manifest growth). The
    class argument is the manifest's per-column source-class NAME (the part
    before any parameter, e.g. ``FACET:control`` -> ``FACET``).
    """

    baseline: set[str] = set()
    for column, source_class in source_classes.items():
        if column in _MANIFEST_INPUT_CHANNEL_ROWS:
            continue
        head = source_class.split(":", 1)[0].split("(", 1)[0]
        if head == "DERIVED":
            if column in _MANIFEST_DERIVED_INIT_BASELINE:
                baseline.add(column)
            elif column not in _MANIFEST_DERIVED_INIT_EXCLUDED:
                raise ValueError(
                    f"Unclassified DERIVED:init column {column!r}: manifest "
                    "v1's DERIVED:init class conflates init-derived real "
                    "content with step-18/load placeholders; classify the "
                    "new column into _MANIFEST_DERIVED_INIT_BASELINE or "
                    "_MANIFEST_DERIVED_INIT_EXCLUDED explicitly."
                )
        elif column in _MANIFEST_FLATTENED_OVERRIDE_BASELINE:
            if head in _MANIFEST_BASELINE_CLASSES:
                raise ValueError(
                    f"Stale flattened-override promotion for {column!r}: the "
                    f"manifest now classes it {source_class!r} (baseline on "
                    "its own); delete the row from "
                    "_MANIFEST_FLATTENED_OVERRIDE_BASELINE."
                )
            baseline.add(column)
        elif head in _MANIFEST_BASELINE_CLASSES:
            baseline.add(column)
        elif head not in _MANIFEST_EXCLUDED_CLASSES:
            raise ValueError(
                f"Unknown CellSourceManifest source class {source_class!r} "
                f"for column {column!r}; extend the projection table "
                "explicitly (frozen vocabulary: CORE|FACET|JOIN|EXTRAS in, "
                "STEP|DEFAULT|NO_PRODUCER out, DERIVED:init split by named "
                "table)."
            )
    return frozenset(baseline)


#: Capture baseline (design-ppdag-v3 §2.4): op-store columns legal to read
#: with NO earlier pipeline writer because step 0 populates them from
#: capture-event data. PROJECTED from the jointly frozen CellSourceManifest
#: v1 (the provisional hand-derived literal was swapped out at integration,
#: 2026-08-13). The swap diff against the provisional literal was
#: root-caused column-by-column: 13 grad-family columns the hand-derivation
#: over-included (constant seeds on every step-0 path — ``grad``/
#: ``grad_fn``/``_grad_records``/``_pending_*_grad_blob_id``/... are
#: written post-backward, OUT of the pipeline) are now correctly excluded;
#: the named adjustment tables above carry the two directions in which the
#: manifest itself is imprecise (DERIVED:init conflation, flattened
#: override lanes). Granularity disclosure: per-column and static — a
#: union over configurations and rows. Per-row provenance is out of scope.
#: Fail-closed: a NEW schema column is NOT baseline until the manifest
#: classifies it (its reads report as findings).
CAPTURE_BASELINE_COLUMNS: frozenset[str] = capture_baseline_from_manifest(CELL_SOURCE_MANIFEST)


@dataclass(frozen=True)
class PinnedPair:
    """One reviewed semantic producer->consumer fact (corpus key 2).

    Parameters
    ----------
    carrier:
        What carries the dependency: ``"columns"`` (op-store columns),
        ``"tokens"`` (trace-state tokens), or ``"structure"`` (row-effects
        barriers / the step-17 barrier — prose invariants absorbed from the
        docstring-invariant map, whose ``carriers`` set may then be empty of
        columns; review note N16 keeps K2's subset check well-typed).
    carriers:
        The column or token names carrying the dependency (may be empty for
        ``"structure"`` entries).
    reason:
        One-line reviewed semantic justification. Deleting or editing an
        entry is THE reviewed act that blesses a reorder — no tooling
        regenerates this corpus.
    """

    carrier: str
    carriers: frozenset[str]
    reason: str

    def __post_init__(self) -> None:
        """Refuse malformed corpus entries at construction."""

        if self.carrier not in ("columns", "tokens", "structure"):
            raise ValueError(
                f"PinnedPair carrier must be columns/tokens/structure, got {self.carrier!r}."
            )
        if self.carrier != "structure" and not self.carriers:
            raise ValueError("A columns/tokens PinnedPair must name its carriers.")


#: Key 2 of the two-key direction authority: the semantic producer->consumer
#: corpus. Seeded ONCE from the day-1 derived RAW and WW edge sets, then
#: hand-reviewed; NEVER regenerated by tooling. Import check K1 refuses any
#: entry contradicting LEGACY_STEP_RANK by name; test-side check K2 refuses
#: any derived RAW/WW edge not pinned here. Content lands with the
#: declaration freeze (implementation plan step 5); the empty corpus is the
#: pre-seed state, not a steady state.
PINNED_ORDER_PAIRS: Mapping[tuple[str, str], PinnedPair] = MappingProxyType(
    {
        ("1", "2"): PinnedPair(
            "columns",
            frozenset(
                (
                    "children",
                    "has_output_descendant",
                    "output_descendants",
                    "token:raw_graph_ws",
                )
            ),
            "step 2 consumes/refines children, has_output_descendant, output_descendants, token:raw_graph_ws after step 1 writes",
        ),
        ("1", "3"): PinnedPair(
            "columns",
            frozenset(
                (
                    "internal_source_parents",
                    "_edge_uses",
                    "_label_raw",
                    "children",
                    "equivalent_ops",
                    "interventions",
                    "is_output",
                    "out",
                    "out_versions_by_child",
                    "output_descendants",
                    "parent_arg_positions",
                    "parents",
                    "recurrent_ops",
                    # b9 R71-1: step 1's output-node minting writes the
                    # root_ancestors closure the removal scrub rebinds.
                    "root_ancestors",
                    "token:raw_graph_ws",
                )
            ),
            "orphan removal floods and scrubs the COMPLETE raw graph: step 1 "
            "must have added the output rows (else outputs read as orphans) "
            "and initialized the relation/equivalence/edge-use state the "
            "removal scrub rebinds; the recurrent_ops carrier is the B1 "
            "probe-blessed placeholder observe (step 1's write is a pinned "
            "no-op, the real writer is step 7)",
        ),
        ("1", "4"): PinnedPair(
            "columns",
            frozenset(
                (
                    "children",
                    "has_output_descendant",
                    "output_descendants",
                    "parents",
                    "token:raw_graph_ws",
                )
            ),
            "step 4 consumes/refines children, has_output_descendant, output_descendants, parents, ... after step 1 writes",
        ),
        ("1", "5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_label_raw",
                    "children",
                    "code_context",
                    "has_output_descendant",
                    "parents",
                    "pass_index",
                    "token:raw_graph_ws",
                )
            ),
            "step 5 consumes/refines _label_raw, children, code_context, has_output_descendant, ... after step 1 writes",
        ),
        ("1", "6"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_edge_uses",
                    "_label_raw",
                    "children",
                    "equivalence_class",
                    "equivalent_ops",
                    "func",
                    "func_name",
                    "has_children",
                    # r3settle e12aa996: the merge's child-direction reach
                    # repair reads/rewrites the reach step 1's output-node
                    # minting seeded (is_output marks the minted rows the
                    # cone re-derivation anchors on).
                    "has_output_descendant",
                    "internal_source_parents",
                    "interventions",
                    "is_input",
                    "is_internal_source",
                    "is_output",
                    "modules",
                    "out",
                    "out_versions_by_child",
                    "output_descendants",
                    "parent_arg_positions",
                    "parents",
                    "recurrent_ops",
                    # b9 R71-1: step 1's output-node minting writes the
                    # root_ancestors closure step 6's rewiring re-derives.
                    "root_ancestors",
                    "saved_args",
                    "token:raw_graph_ws",
                )
            ),
            "buffer connect/merge rewires the raw graph only after step 1 has "
            "added the output rows, and the merge husking re-scrubs the "
            "step-1-seeded relation, template, and edge-use state",
        ),
        ("1", "7"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_label_raw",
                    "_layer_label_raw",
                    "_param_barcodes",
                    "children",
                    "equivalence_class",
                    "equivalent_ops",
                    "func_name",
                    "is_buffer",
                    "modules",
                    "non_tensor_kwargs",
                    "non_tensor_pos_args",
                    "num_passes",
                    "parents",
                    "pass_index",
                    "raw_index",
                    "recurrent_ops",
                    "token:raw_graph_ws",
                    # site_key_v1 minting reads the type axis of step-1 output
                    # rows (every retained op, output nodes included, gets a
                    # structural-position key at step 7).
                    "type",
                )
            ),
            "loop detection consumes the completed raw graph incl. output rows",
        ),
        ("1", "8"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_label_raw",
                    "num_passes",
                    "pass_index",
                    "recurrent_ops",
                    "token:raw_graph_ws",
                    "type",
                )
            ),
            "step 8 consumes/refines _label_raw, num_passes, pass_index, recurrent_ops, ... after step 1 writes",
        ),
        ("1", "9"): PinnedPair(
            "columns",
            frozenset(
                (
                    "internal_source_parents",
                    "_edge_uses",
                    # Reviewed widening (8ba75e99 superlinear-rename fix):
                    # step 9's recurrent_ops rename memo reads each entry's
                    # _label_raw as its lookup key, which step 1 writes when
                    # minting output rows.
                    "_label_raw",
                    "_param_barcodes",
                    "activation_memory",
                    "atomic_module_call",
                    "children",
                    "equivalent_ops",
                    "func_duration",
                    "func_name",
                    "interventions",
                    "is_buffer",
                    "is_input",
                    "is_output",
                    "modules",
                    "num_params",
                    "num_params_frozen",
                    "num_params_trainable",
                    "out_versions_by_child",
                    "output_descendants",
                    "param_memory",
                    "parent_arg_positions",
                    "parents",
                    "recurrent_ops",
                    # b9 R71-1: step 1's output-node minting and step 9's
                    # scrub both write the root_ancestors closure.
                    "root_ancestors",
                    "token:raw_graph_ws",
                    "type",
                )
            ),
            "step 9 consumes/refines _edge_uses, _param_barcodes, activation_memory, atomic_module_call, ... after step 1 writes",
        ),
        ("1", "10"): PinnedPair(
            "tokens",
            frozenset(
                (
                    "token:lookup_containers",
                    "token:raw_graph_ws",
                )
            ),
            "step 10 depends on token:lookup_containers, token:raw_graph_ws produced by step 1",
        ),
        ("1", "11"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_label_raw",
                    "activation_memory",
                    # F20 D-17: step 11 derives saved_activation_memory from
                    # the physical payloads step 1 ingests (byte model).
                    "out",
                    "transformed_activation_memory",
                    "transformed_out",
                    "input_to_module_calls",
                    "io_role",
                    "is_buffer",
                    "is_input",
                    "is_output",
                    "module",
                    # B3R7-R05-1: step 11 relabels module_call_stack to
                    # canonical ModuleCall labels in lockstep with modules;
                    # step 1 seeds it empty on synthetic output nodes.
                    "module_call_stack",
                    "modules",
                    "num_passes",
                    "output_of_module_calls",
                    "raw_index",
                    "token:lookup_containers",
                    "type",
                    "unattributed_tensor_args",
                )
            ),
            "step 11 consumes/refines _label_raw, activation_memory, input_to_module_calls, io_role, ... after step 1 writes",
        ),
        ("1", "11.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "code_context",
                    "func_name",
                    "token:raw_graph_ws",
                    "type",
                    "var_names",
                )
            ),
            "step 11.5 consumes/refines code_context, func_name, token:raw_graph_ws, type, ... after step 1 writes",
        ),
        ("1", "11.75"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_layer_label_raw",
                    "activation_memory",
                    "annotations",
                    "dtype",
                    "func_name",
                    "out",
                    "parents",
                    "raw_index",
                    "saved_args",
                    "saved_kwargs",
                    "shape",
                    "transformed_activation_memory",
                    "transformed_out",
                    "transformed_out_dtype",
                    "transformed_out_shape",
                    "type",
                )
            ),
            "step 11.75 consumes/refines _layer_label_raw, activation_memory, dtype, func_name, ... after step 1 writes",
        ),
        ("1", "12"): PinnedPair(
            "columns",
            frozenset(
                (
                    "out",
                    "saved_args",
                    "saved_kwargs",
                    "transformed_out",
                )
            ),
            "step 12 consumes/refines out, saved_args, saved_kwargs, transformed_out after step 1 writes",
        ),
        ("1", "15"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_param_logs",
                    "parent_params",
                )
            ),
            "step 15 consumes/refines _param_logs, parent_params after step 1 writes",
        ),
        ("1", "15.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_param_logs",
                    "activation_memory",
                    "annotations",
                    "autograd_memory",
                    "dtype",
                    "flops_backward",
                    "flops_forward",
                    "has_output_descendant",
                    "io_role",
                    "is_atomic_module",
                    "modules",
                    "num_autograd_tensors",
                    "parents",
                    "pass_index",
                    "shape",
                    "transformed_activation_memory",
                    "transformed_out_dtype",
                    "transformed_out_shape",
                )
            ),
            "step 15.5 consumes/refines _param_logs, activation_memory, autograd_memory, dtype, ... after step 1 writes",
        ),
        ("1", "16"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_param_logs",
                    "container_spec",
                    "input_to_module_calls",
                    "is_buffer",
                    "is_module_output",
                    "module_call_stack",
                    "output_of_module_calls",
                    "raw_index",
                )
            ),
            "step 16 consumes/refines _param_logs, container_spec, input_to_module_calls, is_buffer, ... after step 1 writes",
        ),
        ("1", "16.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "container_path",
                    "container_spec",
                    "func_name",
                    "is_buffer",
                    "is_input",
                    "is_output",
                    "module",
                    "num_passes",
                    "parents",
                    "token:lookup_containers",
                    "type",
                )
            ),
            "step 16.5 consumes/refines container_path, container_spec, func_name, is_buffer, ... after step 1 writes",
        ),
        ("1", "17.5"): PinnedPair(
            "tokens",
            frozenset(("token:raw_graph_ws",)),
            "step 17.5 depends on token:raw_graph_ws produced by step 1",
        ),
        ("1", "18"): PinnedPair(
            "columns",
            frozenset(
                (
                    "internal_source_parents",
                    "_arg_expressions_cache",
                    "annotations",
                    "_edge_uses",
                    "_label_raw",
                    "_layer_label_raw",
                    "_param_barcodes",
                    "_param_logs",
                    "activation_memory",
                    "arg_names",
                    "atomic_module_call",
                    "autograd_memory",
                    "bytes_delta_at_call",
                    "bytes_peak_at_call",
                    "children",
                    "code_context",
                    "container_path",
                    "container_spec",
                    "dropped_edge_tensor_args",
                    "dtype",
                    "equivalence_class",
                    "equivalent_ops",
                    # A1: the output node's boundary compute reset (alias rows
                    # own no compute) rides the same step-1 write set.
                    "flops_backward",
                    "flops_forward",
                    "func",
                    "func_config",
                    "func_duration",
                    "func_name",
                    "func_non_tensor_args",
                    "func_rng_states",
                    "grad_fn_class_name",
                    "has_children",
                    "has_out_variations",
                    "has_output_descendant",
                    "input_to_module_calls",
                    "intervention_replaced",
                    "interventions",
                    "io_role",
                    "is_atomic_module",
                    "is_buffer",
                    "is_final_output",
                    "is_input",
                    "is_internal_source",
                    "is_module_output",
                    "is_output",
                    "is_transform",
                    "module",
                    "module_call_stack",
                    "modules",
                    "non_tensor_kwargs",
                    "non_tensor_pos_args",
                    "num_args_total",
                    "num_autograd_tensors",
                    "num_kwargs",
                    "num_params",
                    "num_params_frozen",
                    "num_params_trainable",
                    "num_passes",
                    "num_pos_args",
                    "out",
                    "out_versions_by_child",
                    "output_descendants",
                    "output_of_module_calls",
                    "output_of_modules",
                    "param_memory",
                    "param_shapes",
                    "parent_arg_positions",
                    "parent_param_ops",
                    "parent_params",
                    "parents",
                    "pass_index",
                    "raw_index",
                    "recurrent_ops",
                    # b9 R71-1: step 1's output-node minting now writes the
                    # re-derived root_ancestors closure the finalize path reads.
                    "root_ancestors",
                    "saved_args",
                    "saved_kwargs",
                    "shape",
                    "transform_chain",
                    "transform_config",
                    "transform_fn_name",
                    "transform_fn_qualname",
                    "transform_fn_source",
                    "transform_kind",
                    "transformed_activation_memory",
                    "transformed_out",
                    "transformed_out_dtype",
                    "transformed_out_shape",
                    "type",
                    "unattributed_tensor_args",
                    "var_names",
                )
            ),
            "step 18 consumes/refines _arg_expressions_cache, _edge_uses, _label_raw, _layer_label_raw, ... after step 1 writes",
        ),
        ("1", "19"): PinnedPair(
            "columns",
            frozenset(
                (
                    "out",
                    "transformed_out",
                )
            ),
            "step 19 consumes/refines out, transformed_out after step 1 writes",
        ),
        ("2", "3"): PinnedPair(
            "columns",
            frozenset(("output_descendants",)),
            "step 3 consumes/refines output_descendants after step 2 writes",
        ),
        ("2", "4"): PinnedPair(
            "columns",
            frozenset(
                (
                    "has_output_descendant",
                    "output_descendants",
                )
            ),
            "step 4 consumes/refines has_output_descendant, output_descendants after step 2 writes",
        ),
        ("2", "5"): PinnedPair(
            "columns",
            frozenset(("has_output_descendant",)),
            "step 5 consumes/refines has_output_descendant after step 2 writes",
        ),
        ("2", "9"): PinnedPair(
            "columns",
            frozenset(("output_descendants",)),
            "step 9 consumes/refines output_descendants after step 2 writes",
        ),
        ("2", "15.5"): PinnedPair(
            "columns",
            frozenset(("has_output_descendant",)),
            "step 15.5 consumes/refines has_output_descendant after step 2 writes",
        ),
        ("2", "18"): PinnedPair(
            "columns",
            frozenset(
                (
                    "has_output_descendant",
                    "output_descendants",
                )
            ),
            "step 18 consumes/refines has_output_descendant, output_descendants after step 2 writes",
        ),
        ("3", "4"): PinnedPair(
            "tokens",
            frozenset(("token:raw_graph_ws",)),
            "step 4 depends on token:raw_graph_ws produced by step 3",
        ),
        ("3", "6"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_edge_uses",
                    "args_template",
                    "conditional_arm_children",
                    "conditional_elif_children",
                    "conditional_else_children",
                    "conditional_entry_children",
                    "conditional_then_children",
                    "equivalent_ops",
                    "interventions",
                    "kwargs_template",
                    "token:raw_graph_ws",
                )
            ),
            "orphan removal settles the graph before buffer dedup re-walks "
            "it; both steps remove rows through the same husking scrub, so "
            "the shared reference state (edge uses, equivalence groups, "
            "interventions, replay templates, conditional child views) "
            "orders their removals (also a two-sided row barrier)",
        ),
        ("3", "7"): PinnedPair(
            "columns",
            frozenset(
                (
                    "equivalent_ops",
                    "is_orphan",
                    "token:raw_graph_ws",
                )
            ),
            "step 7 consumes/refines equivalent_ops, is_orphan, token:raw_graph_ws after step 3 writes",
        ),
        ("3", "8"): PinnedPair(
            "tokens",
            frozenset(("token:raw_graph_ws",)),
            "step 8 depends on token:raw_graph_ws produced by step 3",
        ),
        ("3", "9"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_edge_uses",
                    "args_template",
                    "conditional_arm_children",
                    "conditional_elif_children",
                    "conditional_else_children",
                    "conditional_entry_children",
                    "conditional_then_children",
                    "equivalent_ops",
                    "interventions",
                    "kwargs_template",
                    "token:raw_graph_ws",
                )
            ),
            "step 9 consumes/refines _edge_uses, args_template, conditional_arm_children, conditional_elif_children, ... after step 3 writes",
        ),
        ("3", "10"): PinnedPair(
            "tokens",
            frozenset(("token:raw_graph_ws",)),
            "step 10 depends on token:raw_graph_ws produced by step 3",
        ),
        ("3", "11"): PinnedPair(
            "columns",
            frozenset(("is_orphan",)),
            "retained-layer finalization consumes step 3's orphan verdicts",
        ),
        ("3", "11.5"): PinnedPair(
            "tokens",
            frozenset(("token:raw_graph_ws",)),
            "step 11.5 depends on token:raw_graph_ws produced by step 3",
        ),
        ("3", "11.75"): PinnedPair(
            "columns",
            frozenset(("is_orphan",)),
            "step 11.75 consumes/refines is_orphan after step 3 writes",
        ),
        ("3", "12"): PinnedPair(
            "columns",
            frozenset(("is_orphan",)),
            "step 12 consumes/refines is_orphan after step 3 writes",
        ),
        ("3", "15.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "conditional_arm_children",
                    "conditional_entry_children",
                )
            ),
            "step 15.5 consumes/refines conditional_arm_children, conditional_entry_children after step 3 writes",
        ),
        ("3", "16"): PinnedPair(
            "columns",
            frozenset(("is_orphan",)),
            "step 16 consumes/refines is_orphan after step 3 writes",
        ),
        ("3", "17.5"): PinnedPair(
            "tokens",
            frozenset(("token:raw_graph_ws",)),
            "step 17.5 depends on token:raw_graph_ws produced by step 3",
        ),
        ("3", "18"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_edge_uses",
                    "args_template",
                    "conditional_arm_children",
                    "conditional_elif_children",
                    "conditional_else_children",
                    "conditional_entry_children",
                    "conditional_then_children",
                    "equivalent_ops",
                    "interventions",
                    "is_internal_sink",
                    "is_orphan",
                    "is_terminal_bool",
                    "kwargs_template",
                )
            ),
            "step 18 consumes/refines _edge_uses, args_template, conditional_arm_children, conditional_elif_children, ... after step 3 writes",
        ),
        ("4", "5"): PinnedPair(
            "columns",
            frozenset(("has_output_descendant",)),
            "step 5 consumes/refines has_output_descendant after step 4 writes",
        ),
        ("4", "6"): PinnedPair(
            "columns",
            frozenset(
                (
                    "has_input_ancestor",
                    # r3settle e12aa996: step 6's reach repair rewrites the
                    # flag step 4's distance marking also maintains.
                    "has_output_descendant",
                    "input_ancestors",
                )
            ),
            "step 6 consumes/refines has_input_ancestor, input_ancestors after step 4 writes",
        ),
        ("4", "9"): PinnedPair(
            "columns",
            frozenset(("input_ancestors",)),
            "step 9 consumes/refines input_ancestors after step 4 writes",
        ),
        ("4", "15.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "has_input_ancestor",
                    "has_output_descendant",
                )
            ),
            "step 15.5 consumes/refines has_input_ancestor, has_output_descendant after step 4 writes",
        ),
        ("4", "18"): PinnedPair(
            "columns",
            frozenset(
                (
                    "has_input_ancestor",
                    "has_output_descendant",
                    "input_ancestors",
                    "max_distance_from_input",
                    "max_distance_to_output",
                    "min_distance_from_input",
                    "min_distance_to_output",
                )
            ),
            "step 18 consumes/refines has_input_ancestor, has_output_descendant, input_ancestors, max_distance_from_input, ... after step 4 writes",
        ),
        ("5", "9"): PinnedPair(
            "columns",
            frozenset(
                (
                    "conditional_arm_children",
                    "conditional_elif_children",
                    "conditional_else_children",
                    "conditional_entry_children",
                    "conditional_then_children",
                )
            ),
            "step 9 consumes/refines conditional_arm_children, conditional_elif_children, conditional_else_children, conditional_entry_children, ... after step 5 writes",
        ),
        ("5", "15.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_is_in_conditional_body",
                    "conditional_arm_children",
                    "conditional_branch_stack",
                    "conditional_entry_children",
                    "terminal_conditional_id",
                    "token:conditional_records",
                )
            ),
            "step 15.5 consumes/refines _is_in_conditional_body, conditional_arm_children, conditional_branch_stack, conditional_entry_children, ... after step 5 writes",
        ),
        ("5", "18"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_is_in_conditional_body",
                    "conditional_arm_children",
                    "conditional_branch_depth",
                    "conditional_branch_stack",
                    "conditional_context_kind",
                    "conditional_elif_children",
                    "conditional_else_children",
                    "conditional_entry_children",
                    "conditional_then_children",
                    "conditional_wrapper_kind",
                    "is_terminal_bool",
                    "is_terminal_conditional_bool",
                    "terminal_conditional_id",
                )
            ),
            "step 18 consumes/refines _is_in_conditional_body, conditional_arm_children, conditional_branch_depth, conditional_branch_stack, ... after step 5 writes",
        ),
        ("6", "7"): PinnedPair(
            "columns",
            frozenset(
                (
                    "children",
                    "equivalent_ops",
                    "func_name",
                    "parents",
                    "token:raw_graph_ws",
                )
            ),
            "loop detection walks the MERGED buffer graph: step 6 settles "
            "children/parents rewires, equivalence-group membership after "
            "duplicate removal, and identity func names before recurrence "
            "grouping reads them",
        ),
        ("6", "8"): PinnedPair(
            "tokens",
            frozenset(("token:raw_graph_ws",)),
            "step 8 depends on token:raw_graph_ws produced by step 6",
        ),
        ("6", "9"): PinnedPair(
            "columns",
            frozenset(
                (
                    "buffer_source",
                    "_edge_uses",
                    "args_template",
                    "children",
                    "conditional_arm_children",
                    "conditional_elif_children",
                    "conditional_else_children",
                    "conditional_entry_children",
                    "conditional_then_children",
                    "equivalent_ops",
                    "func_name",
                    "input_ancestors",
                    "internal_source_ancestors",
                    "internal_source_parents",
                    "interventions",
                    "kwargs_template",
                    # r3settle e12aa996: step 6's merge repairs child-direction
                    # reach; step 9 relabels the closure into final-label space.
                    "output_descendants",
                    "parent_arg_positions",
                    "parents",
                    "root_ancestors",
                    "token:raw_graph_ws",
                )
            ),
            "final-info logging rewrites step 6's merged relation state — "
            "children/parents/arg positions after duplicate-buffer rewiring, "
            "the scrubbed templates and conditional views, ancestry closures "
            "— into final-label space",
        ),
        ("6", "10"): PinnedPair(
            "tokens",
            frozenset(("token:raw_graph_ws",)),
            "step 10 depends on token:raw_graph_ws produced by step 6",
        ),
        ("6", "11"): PinnedPair(
            "columns",
            frozenset(
                (
                    "address",
                    "buffer_pass",
                )
            ),
            "step 11 consumes/refines address, buffer_pass after step 6 writes",
        ),
        ("6", "11.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "func_name",
                    "token:raw_graph_ws",
                )
            ),
            "step 11.5 consumes/refines func_name, token:raw_graph_ws after step 6 writes",
        ),
        ("6", "11.75"): PinnedPair(
            "columns",
            frozenset(
                (
                    "func_name",
                    "parents",
                )
            ),
            "deferred retention resolves saved-layer selectors against the "
            "merged graph: step 6's parents rewires and identity func names "
            "must be settled first",
        ),
        ("6", "15.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "conditional_arm_children",
                    "conditional_entry_children",
                    "has_input_ancestor",
                    # r3settle e12aa996: aggregation reads output reach as
                    # step 6's merge repair left it.
                    "has_output_descendant",
                    "parents",
                )
            ),
            "layer-log aggregation reads per-op ancestry, parents, and "
            "conditional child views as step 6's merge left them",
        ),
        ("6", "16"): PinnedPair(
            "columns",
            frozenset(("address",)),
            "module-log construction reads per-op display addresses, which "
            "for buffer rows are final only after step 6's recovery / "
            "anonymous-fallback assignment",
        ),
        ("6", "16.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "func_name",
                    "parents",
                )
            ),
            "the graph-shape hash digests topology (parents) and identity "
            "func names as the buffer merge finalized them",
        ),
        ("6", "17.5"): PinnedPair(
            "tokens",
            frozenset(("token:raw_graph_ws",)),
            "step 17.5 depends on token:raw_graph_ws produced by step 6",
        ),
        ("6", "18"): PinnedPair(
            "columns",
            frozenset(
                (
                    "has_internal_source_ancestor",
                    "_edge_uses",
                    "address",
                    "args_template",
                    "buffer_pass",
                    "buffer_replay_validated",
                    "buffer_source",
                    "children",
                    "conditional_arm_children",
                    "conditional_elif_children",
                    "conditional_else_children",
                    "conditional_entry_children",
                    "conditional_then_children",
                    "equivalent_ops",
                    "func",
                    "func_name",
                    "has_children",
                    "has_input_ancestor",
                    # r3settle e12aa996: the bundle persists output reach as
                    # step 6's merge repair finalized it.
                    "has_output_descendant",
                    "input_ancestors",
                    "internal_source_ancestors",
                    "internal_source_parents",
                    "interventions",
                    "kwargs_template",
                    "output_descendants",
                    "parent_arg_positions",
                    "parents",
                    "root_ancestors",
                )
            ),
            "the streamed bundle persists per-op metadata — buffer "
            "pass/source/validation state, merged relations, scrubbed "
            "templates and conditional views — exactly as step 6 finalized "
            "them",
        ),
        ("7", "8"): PinnedPair(
            "columns",
            frozenset(
                (
                    "num_passes",
                    "pass_index",
                    "recurrent_ops",
                )
            ),
            "label generation consumes step 7's recurrence groups and pass indexes",
        ),
        ("7", "9"): PinnedPair(
            "columns",
            frozenset(("recurrent_ops",)),
            "step 9 consumes/refines recurrent_ops after step 7 writes",
        ),
        ("7", "11"): PinnedPair(
            "columns",
            frozenset(("num_passes",)),
            "step 11 consumes/refines num_passes after step 7 writes",
        ),
        ("7", "11.75"): PinnedPair(
            "columns",
            frozenset(("_layer_label_raw",)),
            "step 11.75 consumes/refines _layer_label_raw after step 7 writes",
        ),
        ("7", "15.5"): PinnedPair(
            "columns",
            frozenset(("pass_index",)),
            "step 15.5 consumes/refines pass_index after step 7 writes",
        ),
        ("7", "16.5"): PinnedPair(
            "columns",
            frozenset(("num_passes",)),
            "step 16.5 consumes/refines num_passes after step 7 writes",
        ),
        ("7", "18"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_layer_label_raw",
                    "equivalence_class",
                    "num_passes",
                    "pass_index",
                    "recurrent_ops",
                    # step 18's whole-record scrub reads the step-7-minted
                    # site_key (DROP under v7, read-then-dropped).
                    "site_key",
                )
            ),
            "step 18 consumes/refines _layer_label_raw, equivalence_class, num_passes, pass_index, ... after step 7 writes",
        ),
        ("8", "9"): PinnedPair(
            "columns",
            frozenset(
                (
                    "label",
                    "layer_label",
                    "step_index",
                    "token:label_maps",
                )
            ),
            "final-info logging consumes step 8's raw-to-final label maps",
        ),
        ("8", "10"): PinnedPair(
            "columns",
            frozenset(
                (
                    "layer_label",
                    "token:label_maps",
                )
            ),
            "step 10 consumes/refines layer_label, token:label_maps after step 8 writes",
        ),
        ("8", "11"): PinnedPair(
            "columns",
            frozenset(
                (
                    "label",
                    "label_short",
                    "layer_label",
                    "layer_label_short",
                    "token:label_maps",
                )
            ),
            "step 11 consumes/refines label, label_short, layer_label, layer_label_short, ... after step 8 writes",
        ),
        ("8", "11.75"): PinnedPair(
            "columns",
            frozenset(
                (
                    "label",
                    "layer_label",
                )
            ),
            "step 11.75 consumes/refines label, layer_label after step 8 writes",
        ),
        ("8", "12"): PinnedPair(
            "columns",
            frozenset(("layer_label",)),
            "step 12 consumes/refines layer_label after step 8 writes",
        ),
        ("8", "15"): PinnedPair(
            "columns",
            frozenset(
                (
                    "label",
                    "layer_label",
                )
            ),
            "step 15 consumes/refines label, layer_label after step 8 writes",
        ),
        ("8", "15.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "label",
                    "layer_label",
                )
            ),
            "step 15.5 consumes/refines label, layer_label after step 8 writes",
        ),
        ("8", "16"): PinnedPair(
            "columns",
            frozenset(
                (
                    "label",
                    "layer_label",
                )
            ),
            "step 16 consumes/refines label, layer_label after step 8 writes",
        ),
        ("8", "16.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "label",
                    "layer_label",
                )
            ),
            "step 16.5 consumes/refines label, layer_label after step 8 writes",
        ),
        ("8", "18"): PinnedPair(
            "columns",
            frozenset(
                (
                    "label",
                    "label_short",
                    "layer_label",
                    "layer_label_short",
                    "step_index",
                    "type_index",
                )
            ),
            "step 18 consumes/refines label, label_short, layer_label, layer_label_short, ... after step 8 writes",
        ),
        ("9", "10"): PinnedPair(
            "tokens",
            frozenset(("token:raw_graph_ws",)),
            "step 10 depends on token:raw_graph_ws produced by step 9",
        ),
        ("9", "11"): PinnedPair(
            "columns",
            frozenset(
                (
                    "fx_call_index",
                    "fx_qualpath",
                    "is_buffer",
                    "is_input",
                    "is_output",
                    "token:module_build",
                )
            ),
            "lookup keys consume step 9's module hierarchy/build data",
        ),
        ("9", "11.5"): PinnedPair(
            "tokens",
            frozenset(("token:raw_graph_ws",)),
            "step 11.5 depends on token:raw_graph_ws produced by step 9",
        ),
        ("9", "11.75"): PinnedPair(
            "columns",
            frozenset(("parents",)),
            "step 11.75 consumes/refines parents after step 9 writes",
        ),
        ("9", "15.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "conditional_arm_children",
                    "conditional_entry_children",
                    "parents",
                )
            ),
            "step 15.5 consumes/refines conditional_arm_children, conditional_entry_children, parents after step 9 writes",
        ),
        ("9", "16"): PinnedPair(
            "columns",
            frozenset(
                (
                    "is_buffer",
                    "token:module_build",
                )
            ),
            "step 16 consumes/refines is_buffer, token:module_build after step 9 writes",
        ),
        ("9", "16.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "is_buffer",
                    "is_input",
                    "is_output",
                    "parents",
                )
            ),
            "step 16.5 consumes/refines is_buffer, is_input, is_output, parents after step 9 writes",
        ),
        ("9", "17.5"): PinnedPair(
            "tokens",
            frozenset(("token:raw_graph_ws",)),
            "step 17.5 depends on token:raw_graph_ws produced by step 9",
        ),
        ("9", "18"): PinnedPair(
            "columns",
            frozenset(
                (
                    "buffer_source",
                    "_edge_uses",
                    "args_template",
                    "atomic_module_call",
                    "children",
                    "conditional_arm_children",
                    "conditional_elif_children",
                    "conditional_else_children",
                    "conditional_entry_children",
                    "conditional_then_children",
                    "equivalent_ops",
                    "fx_call_index",
                    "fx_qualpath",
                    "input_ancestors",
                    "internal_source_ancestors",
                    "internal_source_parents",
                    "interventions",
                    "is_buffer",
                    "is_input",
                    "is_output",
                    "kwargs_template",
                    "output_descendants",
                    "parent_arg_positions",
                    "parents",
                    "recurrent_ops",
                    "root_ancestors",
                    "step_index",
                )
            ),
            "step 18 consumes/refines _edge_uses, args_template, atomic_module_call, children, ... after step 9 writes",
        ),
        ("10", "11"): PinnedPair(
            "tokens",
            frozenset(("token:lookup_containers",)),
            "lookup keys are built over step 10's renamed references",
        ),
        ("10", "16.5"): PinnedPair(
            "tokens",
            frozenset(("token:lookup_containers",)),
            "step 16.5 depends on token:lookup_containers produced by step 10",
        ),
        ("11", "15.5"): PinnedPair(
            "columns",
            frozenset(("modules",)),
            "step 15.5 consumes/refines modules after step 11 writes",
        ),
        ("11", "16"): PinnedPair(
            "columns",
            frozenset(
                (
                    "input_to_module_calls",
                    "output_of_module_calls",
                    "token:saved_summary",
                )
            ),
            "step 16 consumes/refines input_to_module_calls, output_of_module_calls, token:saved_summary after step 11 writes",
        ),
        ("11", "16.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "module",
                    "token:lookup_containers",
                )
            ),
            "step 16.5 consumes/refines module, token:lookup_containers after step 11 writes",
        ),
        ("11", "18"): PinnedPair(
            "columns",
            frozenset(
                (
                    "input_to_module_calls",
                    "lookup_keys",
                    "module",
                    # B3R7-R05-1: step 18's streamed-bundle serialization reads
                    # the stack after step 11's canonical-label relabel.
                    "module_call_stack",
                    "modules",
                    "ordinal_index",
                    "output_of_module_calls",
                )
            ),
            "step 18 consumes/refines input_to_module_calls, lookup_keys, module, modules, ... after step 11 writes",
        ),
        ("11.5", "18"): PinnedPair(
            "columns",
            frozenset(("var_names",)),
            "step 18 consumes/refines var_names after step 11.5 writes",
        ),
        ("11.75", "12"): PinnedPair(
            "columns",
            frozenset(
                (
                    "has_saved_activation",
                    "out",
                    "saved_args",
                    "saved_kwargs",
                    "token:payload_tensors",
                    "transformed_out",
                )
            ),
            "step 12 consumes/refines has_saved_activation, out, saved_args, saved_kwargs, ... after step 11.75 writes",
        ),
        ("11.75", "13"): PinnedPair(
            "tokens",
            frozenset(("token:payload_tensors",)),
            "step 13 depends on token:payload_tensors produced by step 11.75",
        ),
        ("11.75", "15.5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "activation_memory",
                    # Layer construction mirrors the reserved "collective"
                    # annotations key from its first pass (fix/mergedannot
                    # 130b3f80), so the layer build must see step 11.75's
                    # settled annotation writes.
                    "annotations",
                    "dtype",
                    "shape",
                    "transformed_activation_memory",
                    "transformed_out_dtype",
                    "transformed_out_shape",
                )
            ),
            "step 15.5 consumes/refines activation_memory, dtype, shape, transformed_activation_memory, ... after step 11.75 writes",
        ),
        ("11.75", "16"): PinnedPair(
            "columns",
            frozenset(("has_saved_activation",)),
            "step 16 consumes/refines has_saved_activation after step 11.75 writes",
        ),
        ("11.75", "18"): PinnedPair(
            "columns",
            frozenset(
                (
                    "activation_memory",
                    "annotations",
                    "dtype",
                    "has_saved_activation",
                    "out",
                    "saved_args",
                    "saved_kwargs",
                    "shape",
                    "token:payload_tensors",
                    "transformed_activation_memory",
                    "transformed_out",
                    "transformed_out_dtype",
                    "transformed_out_shape",
                )
            ),
            "step 18 consumes/refines activation_memory, annotations, dtype, has_saved_activation, ... after step 11.75 writes",
        ),
        ("11.75", "19"): PinnedPair(
            "columns",
            frozenset(
                (
                    "out",
                    "transformed_out",
                )
            ),
            "step 19 consumes/refines out, transformed_out after step 11.75 writes",
        ),
        ("12", "13"): PinnedPair(
            "tokens",
            frozenset(("token:payload_tensors",)),
            "step 13 depends on token:payload_tensors produced by step 12",
        ),
        ("12", "18"): PinnedPair(
            "tokens",
            frozenset(("token:payload_tensors",)),
            "step 18 depends on token:payload_tensors produced by step 12",
        ),
        ("15", "15.5"): PinnedPair(
            "columns",
            frozenset(("_param_logs",)),
            "step 15.5 consumes/refines _param_logs after step 15 writes",
        ),
        ("15", "16"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_param_logs",
                    "token:param_logs_kind",
                )
            ),
            "step 16 consumes/refines _param_logs, token:param_logs_kind after step 15 writes",
        ),
        ("15", "18"): PinnedPair(
            "columns",
            frozenset(
                (
                    "_param_logs",
                    "parent_params",
                )
            ),
            "step 18 consumes/refines _param_logs, parent_params after step 15 writes",
        ),
        ("15", "20"): PinnedPair(
            "tokens",
            frozenset(("token:param_logs_kind",)),
            "step 20 depends on token:param_logs_kind produced by step 15",
        ),
        ("15.5", "16"): PinnedPair(
            "tokens",
            frozenset(("token:layer_logs",)),
            "Module.layers references the Layer keys step 15.5 builds",
        ),
        ("15.5", "18"): PinnedPair(
            "columns",
            frozenset(
                (
                    "in_conditionals",
                    "terminal_bool_for",
                )
            ),
            "step 18 consumes/refines in_conditionals, terminal_bool_for after step 15.5 writes",
        ),
        ("16", "18"): PinnedPair(
            "columns",
            frozenset(("_param_logs",)),
            "step 18 consumes/refines _param_logs after step 16 writes",
        ),
        ("16", "20"): PinnedPair(
            "tokens",
            frozenset(("token:param_logs_kind",)),
            "step 20 depends on token:param_logs_kind produced by step 16",
        ),
        ("16.5", "18"): PinnedPair(
            "columns",
            frozenset(("_address_normalized",)),
            "step 18 consumes/refines _address_normalized after step 16.5 writes",
        ),
        ("17", "18"): PinnedPair(
            "columns",
            frozenset(("_tracing_finished",)),
            "step 18 consumes/refines _tracing_finished after step 17 writes",
        ),
        ("17.5", "18"): PinnedPair(
            "tokens",
            frozenset(("token:containers",)),
            "the streamed bundle persists the containers step 17.5 adopts",
        ),
        ("18", "19"): PinnedPair(
            "columns",
            frozenset(
                (
                    "out_ref",
                    "token:stream_lifecycle",
                    "token:stream_writer",
                )
            ),
            "step 19 consumes/refines out_ref, token:stream_lifecycle, token:stream_writer after step 18 writes",
        ),
        ("18", "20"): PinnedPair(
            "tokens",
            frozenset(("token:stream_lifecycle",)),
            "param-ref release must follow optional stream finalization",
        ),
        ("19", "20"): PinnedPair(
            "tokens",
            frozenset(("token:stream_lifecycle",)),
            "param-ref release must follow optional out eviction",
        ),
        ("3", "5"): PinnedPair(
            "columns",
            frozenset(
                (
                    "conditional_arm_children",
                    "conditional_elif_children",
                    "conditional_else_children",
                    "conditional_entry_children",
                    "conditional_then_children",
                    "is_orphan",
                    "is_terminal_bool",
                    "token:raw_graph_ws",
                )
            ),
            "conditional attribution requires the orphan-free graph and step 3's "
            "orphan/terminal-bool verdicts (also a two-sided row barrier)",
        ),
        ("17", "17.5"): PinnedPair(
            "structure",
            frozenset(),
            "workspace drops happen only after the finished-flag barrier flips facade behavior",
        ),
        ("2", "6"): PinnedPair(
            "columns",
            # r3settle e12aa996 added has_output_descendant: the merge and
            # cone re-derivation rewrite the reach flag step 2 seeded.
            frozenset(("has_output_descendant", "output_descendants")),
            "the buffer-merge scrub filters removed labels out of the "
            "output-descendant closures step 2 marks",
        ),
        ("5", "6"): PinnedPair(
            "columns",
            frozenset(
                (
                    "conditional_arm_children",
                    "conditional_elif_children",
                    "conditional_else_children",
                    "conditional_entry_children",
                    "conditional_then_children",
                )
            ),
            "the buffer-merge husking scrubs removed buffer labels out of "
            "the conditional child views step 5 attributes",
        ),
    }
)


def iter_corpus_violations(rank: dict[str, int]) -> list[str]:
    """Return every K1 violation of the corpus against ``rank``.

    Injectable for the coordinated-reversal regression test; production
    import passes the frozen rank.
    """

    violations: list[str] = []
    for (producer, consumer), pair in PINNED_ORDER_PAIRS.items():
        if producer not in rank or consumer not in rank:
            violations.append(
                f"PINNED_ORDER_PAIRS[({producer!r}, {consumer!r})] names a "
                "step absent from LEGACY_STEP_RANK."
            )
            continue
        if rank[producer] >= rank[consumer]:
            violations.append(
                f"Reordering {consumer} before {producer} contradicts "
                f"PINNED_ORDER_PAIRS[({producer!r}, {consumer!r})]: "
                f"{pair.reason}"
            )
    return violations


def _validate_contract_artifacts() -> None:
    """Import-time structural binding of contracts and the frozen rank.

    Check 7.1-0 (design-ppdag-v3): the rank's key set equals the contract
    key set minus the fenced step "0", refused by name — a step
    insertion/removal diff hits this first, never a bare ``KeyError``
    inside derivation. Plain ``raise`` (``python -O`` strips asserts).
    """

    contract_steps = set(POSTPROCESS_STEP_CONTRACTS) - {"0"}
    rank_steps = set(LEGACY_STEP_RANK)
    if contract_steps != rank_steps:
        missing_rank = sorted(contract_steps - rank_steps)
        missing_contract = sorted(rank_steps - contract_steps)
        raise ValueError(
            "POSTPROCESS_STEP_CONTRACTS and LEGACY_STEP_RANK disagree: "
            f"steps missing a rank: {missing_rank}; ranks missing a "
            f"contract: {missing_contract}. Adding or removing a pipeline "
            "step edits both artifacts (and the pinned-pair corpus) in one "
            "reviewed diff."
        )
    ranks = [LEGACY_STEP_RANK[step] for step in LEGACY_STEP_RANK]
    if len(set(ranks)) != len(ranks):
        raise ValueError("LEGACY_STEP_RANK ranks must be unique integers.")
    # K1: every pinned pair is rank-consistent, refused BY NAME with its
    # reason — a coordinated rank+registry reversal re-orients every derived
    # edge and passes R1/R2, but cannot pass this without editing the named
    # reason-bearing corpus entry (design-ppdag-v3 §2.1).
    for violation in iter_corpus_violations(dict(LEGACY_STEP_RANK)):
        raise ValueError(violation)


_validate_contract_artifacts()
