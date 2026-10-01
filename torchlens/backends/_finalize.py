"""Shared preview-backend trace finalization helpers."""

from __future__ import annotations

import inspect
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal, TypeAlias, cast

from ..data_classes._compaction import compact_op_metadata
from ..data_classes._site_key import SiteKeyMinter
from ..data_classes.layer import Layer
from ..data_classes.module import ModuleAccessor
from ..data_classes.trace import Trace
from ..ir.op_record import amend_preview_output_parent_mark
from ..ir.workspaces import _init_module_hierarchy_data
from ..postprocess._grouping_stamp import build_grouping_policy_stamp
from ..postprocess._recurrence import compute_preview_recurrence_assignments, relabel_edge_metadata
from ..postprocess.finalization import _build_module_logs, _build_root_module_log
from ..postprocess.loop_grouping_adapter import RecurrenceAssignment, strip_raw_label_suffix
from ..quantities import Bytes
from .registry import BackendName

OpHook: TypeAlias = Callable[[Any, Trace, set[str]], None]
OpEnrichmentHook: TypeAlias = Callable[[Any], None]
LayerEnrichmentHook: TypeAlias = Callable[[Any, Any], None]
SidecarRelabelHook: TypeAlias = Callable[[dict[str, str]], None]
ModuleCallNormalizer: TypeAlias = Callable[[Any], tuple[tuple[str, int], ...]]
MetadataTopLevelPredicate: TypeAlias = Callable[
    [str, dict[str, Any], dict[str, dict[str, Any]]], bool
]
OpTopLevelPredicate: TypeAlias = Callable[[str], bool]
TrainingModeResolver: TypeAlias = Callable[[dict[str, Any]], bool]


def finalize_single_pass_trace(
    trace: Trace,
    *,
    backend_name: str,
    module_tree: Any | None,
    attach_function_root_module: Callable[[Trace], None],
    attach_object_module_logs: Callable[[Trace, Any], None],
    attach_op_params: OpHook | None = None,
    enrich_op: OpEnrichmentHook | None = None,
    enrich_layer: LayerEnrichmentHook | None = None,
    update_param_usage: bool = True,
    update_param_totals_from_layers: bool = False,
    count_layers_with_attached_params: bool = False,
    finish_before_module_logs: bool = True,
    compute_input_output_distances: bool | None = None,
    recurrence_detection: bool = False,
    relabel_sidecar_labels: SidecarRelabelHook | None = None,
) -> None:
    """Finalize raw single-pass op logs into public trace accessors.

    Parameters
    ----------
    trace:
        Trace whose ``_raw_layer_dict`` contains backend-created op logs.
    backend_name:
        Canonical backend name written to ``trace.backend``.
    module_tree:
        Optional object-module discovery tree for object-module traces.
    attach_function_root_module:
        Callback used when no object module tree is active.
    attach_object_module_logs:
        Callback used when object module attribution is active.
    attach_op_params:
        Optional hook that attaches backend parameter logs to one op.
    enrich_op:
        Optional hook for backend-specific op metadata before layer creation.
    enrich_layer:
        Optional hook for backend-specific layer metadata copied from an op.
    update_param_usage:
        Whether attached op params should receive ``used_by_*`` cross-links.
    update_param_totals_from_layers:
        Whether trace parameter counters should be recomputed from finalized layers.
    count_layers_with_attached_params:
        Whether to compute ``trace.num_layers_with_params`` from attached params.
    finish_before_module_logs:
        Whether to set ``_tracing_finished`` before module logs are attached.
    compute_input_output_distances:
        Whether to run the neutral input/output distance flood (torch Step 4)
        over the finalized single-pass graph. ``None`` reads the request the
        backend stored on ``trace.mark_layer_depths``; the effective value is
        written back to ``trace.mark_layer_depths`` either way.
    recurrence_detection:
        Whether to run the neutral recurrence grouper over the raw graph and
        apply its assignments (multi-pass layers, pass-qualified op labels,
        relabeled edges). ``False`` preserves the historical ungrouped
        single-pass layout.
    relabel_sidecar_labels:
        Backend hook receiving the COMPLETE ``{raw_label: final_op_label}``
        mapping after per-op final labels are assigned. Backends holding
        label-keyed sidecar state (validation replay inventories, intervention
        records) must remap it atomically here; capture-index-keyed sidecars
        may ignore the hook. Called unconditionally (N5: every op's raw label
        differs from its final label now, not only multi-pass group members),
        not only when ``recurrence_detection`` is on.

    Returns
    -------
    None
        The trace is updated in place.
    """

    _mint_preview_site_keys(trace)
    assignments: dict[str, RecurrenceAssignment] | None = None
    if recurrence_detection:
        assignments = compute_preview_recurrence_assignments(trace, backend_name=backend_name)

    seen_param_barcodes: set[str] = set()
    layers_with_params_seen: set[str] = set()
    param_usage_membership: dict[int, tuple[set[str], set[str], set[str]]] = {}
    for raw_index, (label, op_log) in enumerate(trace._raw_graph_ws.raw_layer_dict.items()):
        assignment = assignments.get(label) if assignments is not None else None
        _finalize_single_op(trace, op_log, label, raw_index, assignment)
        if enrich_op is not None:
            enrich_op(op_log)
        if attach_op_params is not None:
            attach_op_params(op_log, trace, seen_param_barcodes)
            if not isinstance(op_log.param_memory, Bytes):
                op_log.param_memory = Bytes(int(op_log.param_memory))
        if getattr(op_log, "_param_logs", []):
            layers_with_params_seen.add(op_log.layer_label)
        if update_param_usage:
            _attach_param_usage(trace, op_log, param_usage_membership)
        layer_log = trace.layer_logs.get(op_log.layer_label)
        layer_created = layer_log is None
        if layer_log is None:
            layer_log = Layer(op_log)
        layer_log.ops[op_log.pass_index] = op_log
        layer_log.call_labels.append(op_log.label)
        if op_log.num_passes != 1:
            layer_log.num_passes = op_log.num_passes
        if layer_created:
            if enrich_layer is not None:
                enrich_layer(layer_log, op_log)
            trace.layer_logs[op_log.layer_label] = layer_log

    trace.num_ops = sum(
        1
        for op_log in trace.layer_list
        if not (op_log.is_input or op_log.is_output or op_log.is_buffer)
    )
    if compute_input_output_distances is None:
        compute_input_output_distances = bool(getattr(trace, "mark_layer_depths", False))
    trace.mark_layer_depths = bool(compute_input_output_distances)
    if compute_input_output_distances:
        # The flood runs BEFORE the relabel epilogue: edges and the trace-side
        # input/output label lists are still uniformly raw here, so every seed
        # resolves to its exact op (a layer label would resolve to pass 1 and
        # mis-seed distances for outputs produced by a later pass).
        compute_preview_input_output_distances(trace)
    # N5: always relabel, even when grouping did not run -- every op's final
    # identity differs from its raw identity (the ``_raw`` capture sentinel
    # is always stripped), not just multi-pass group members.
    _apply_recurrence_relabel_epilogue(trace, assignments, relabel_sidecar_labels)
    # The stored flag is the EFFECTIVE value: ``True`` only when the neutral
    # grouper actually ran over this graph, so an ungrouped finalize can never
    # claim grouping that never happened. (JAX finalizes through its own
    # recurrence-grouping path and keeps the request.)
    trace.recurrence_detection = assignments is not None
    trace.grouping_policy = build_grouping_policy_stamp(
        ran_recurrence_grouping=assignments is not None,
        requested=getattr(trace, "grouping", "structural"),
    )
    if update_param_totals_from_layers:
        _update_param_totals_from_layers(trace)
    if count_layers_with_attached_params:
        trace.num_layers_with_params = len(layers_with_params_seen)
    trace._layers_logged = True
    trace._layers_saved = True
    trace.has_backward_pass = False
    trace.capture_end_time = time.time()
    trace.backend = cast(BackendName, backend_name)
    if finish_before_module_logs:
        trace._tracing_finished = True
        _set_per_op_tracing_finished(trace)
    if module_tree is None:
        trace.module_identity_mode = "function_root"
        attach_function_root_module(trace)
    else:
        trace.module_identity_mode = "object_module"
        attach_object_module_logs(trace, module_tree)
    if not finish_before_module_logs:
        trace._tracing_finished = True
        _set_per_op_tracing_finished(trace)
    compact_op_metadata(trace)


def _set_per_op_tracing_finished(trace: Trace) -> None:
    """Flip ``_tracing_finished`` on every retained op, mirroring the torch path.

    Parameters
    ----------
    trace:
        Trace whose ``_tracing_finished`` flag was just set at the trace level.

    Returns
    -------
    None
        Every op reachable via ``trace.layer_dict_main_keys`` is mutated in place.

    Notes
    -----
    Mirrors ``torchlens.postprocess.finalization._set_tracing_finished``, which
    also flips this flag on every retained ``OpLog`` (not just the trace). Without
    this, preview-backend ops stay stuck on the "mid-capture" ``__str__``/repr
    branch, which assumes torch-only attributes (e.g. ``grad_fn``) and crashes
    on non-torch tensors.
    """

    for layer_label in trace.layer_dict_main_keys:
        op_log = trace.layer_dict_main_keys[layer_label]
        op_log._tracing_finished = True


def attach_function_root_module(trace: Trace) -> None:
    """Attach the shared ``self`` module log for function-root traces.

    Parameters
    ----------
    trace:
        Trace receiving root-module metadata.

    Returns
    -------
    None
        ``trace._module_logs`` is populated with a single ``self`` module.
    """

    mbd = trace._module_capture_ws.module_build_data
    mbd["top_level_modules"] = ["self"]
    mbd["top_level_module_ops"] = ["self:1"]
    trace._module_capture_ws.module_metadata = {
        "self": {
            "cls": None,
            "class_name": trace.model_class_name,
            "class_qualname": trace.model_class_qualname,
            "all_addresses": ["self"],
            "training": False,
        }
    }
    root = _build_root_module_log(trace, {}, mbd)
    trace._module_logs = ModuleAccessor({"self": root})


def attach_object_module_logs(
    trace: Trace,
    tree: Any,
    *,
    normalize_module_calls: ModuleCallNormalizer,
    metadata_top_level: MetadataTopLevelPredicate,
    op_top_level: OpTopLevelPredicate,
    training_mode: TrainingModeResolver,
) -> None:
    """Build module logs for preview backends with object-module attribution.

    Parameters
    ----------
    trace:
        Trace receiving module build-data and public module logs.
    tree:
        Backend module tree exposing ``metadata``, ``forward_args_by_call``, and
        ``call_counts`` attributes.
    normalize_module_calls:
        Backend normalizer for raw per-op module-call records.
    metadata_top_level:
        Predicate for top-level modules discovered from module metadata.
    op_top_level:
        Predicate for top-level modules observed in finalized op call stacks.
    training_mode:
        Resolver for backend-specific module training metadata.

    Returns
    -------
    None
        ``trace.modules`` and transient module build-data are populated.
    """

    trace._module_capture_ws.module_build_data = _init_module_hierarchy_data()
    trace._module_capture_ws.module_forward_args = dict(tree.forward_args_by_call)
    trace._module_capture_ws.module_metadata = tree.metadata
    mbd = trace._module_capture_ws.module_build_data
    metadata_by_address = cast(dict[str, dict[str, Any]], tree.metadata)
    call_counts = cast(dict[str, int], tree.call_counts)
    for address, metadata in metadata_by_address.items():
        if address not in mbd["addresses"]:
            mbd["addresses"].append(address)
        mbd["module_types"][address] = str(metadata.get("class_name", ""))
        mbd["module_training_modes"][address] = training_mode(metadata)
        mbd["module_num_calls"][address] = max(1, call_counts.get(address, 1))
        for child_address in metadata.get("address_children", []):
            if child_address not in mbd["module_children"][address]:
                mbd["module_children"][address].append(child_address)
        if metadata_top_level(address, metadata, metadata_by_address):
            mbd["top_level_modules"].append(address)

    for param in trace.param_logs:
        owner = param.module_address
        mbd["module_nparams"][owner] += param.num_params
        if param.is_trainable:
            mbd["module_nparams_trainable"][owner] += param.num_params
        else:
            mbd["module_nparams_frozen"][owner] += param.num_params

    populate_object_module_build_data(
        trace,
        normalize_module_calls=normalize_module_calls,
        op_top_level=op_top_level,
    )
    _build_module_logs(trace)


def populate_object_module_build_data(
    trace: Trace,
    *,
    normalize_module_calls: ModuleCallNormalizer,
    op_top_level: OpTopLevelPredicate,
) -> None:
    """Populate module hierarchy side channels from attributed op logs.

    Parameters
    ----------
    trace:
        Trace whose finalized ops carry module-call records.
    normalize_module_calls:
        Backend normalizer for raw per-op module-call records.
    op_top_level:
        Predicate for top-level modules observed in op call stacks.

    Returns
    -------
    None
        ``trace._module_capture_ws.module_build_data`` is updated in place.
    """

    mbd = trace._module_capture_ws.module_build_data
    seen_layers: dict[str, set[str]] = defaultdict(set)
    seen_pass_layers: dict[str, set[str]] = defaultdict(set)
    seen_module_ops: set[str] = set()
    seen_top_level_ops: set[str] = set()
    seen_pass_children: dict[str, set[str]] = defaultdict(set)
    seen_addresses = set(mbd["addresses"])
    for op_log in trace.layer_list:
        normalized_calls = normalize_module_calls(op_log.modules)
        op_log.modules = [f"{address}:{call_index}" for address, call_index in normalized_calls]
        op_log.module = op_log.modules[-1] if op_log.modules else None
        parent_call_label: str | None = None
        for module_index, (address, call_index) in enumerate(normalized_calls):
            call_label = f"{address}:{call_index}"
            if call_label not in mbd["module_call_stacks"]:
                mbd["module_call_stacks"][call_label] = [
                    f"{ancestor_address}:{ancestor_call_index}"
                    for ancestor_address, ancestor_call_index in normalized_calls[:module_index]
                    if ancestor_address != "self"
                ]
            if mbd["module_num_calls"][address] < call_index:
                mbd["module_num_calls"][address] = call_index
            mbd["module_num_tensors"][address] += 1
            mbd["module_call_index_tensors"][call_label] += 1
            if op_log.layer_label not in seen_layers[address]:
                seen_layers[address].add(op_log.layer_label)
                mbd["module_layers"][address].append(op_log.layer_label)
            if op_log.label not in seen_pass_layers[call_label]:
                seen_pass_layers[call_label].add(op_log.label)
                mbd["module_pass_layers"][call_label].append(op_log.label)
            if address not in seen_addresses:
                seen_addresses.add(address)
                mbd["addresses"].append(address)
            if call_label not in seen_module_ops:
                seen_module_ops.add(call_label)
                mbd["module_ops"].append(call_label)
            if module_index == 0:
                if call_label not in seen_top_level_ops:
                    seen_top_level_ops.add(call_label)
                    mbd["top_level_module_ops"].append(call_label)
                if op_top_level(address) and address not in mbd["top_level_modules"]:
                    mbd["top_level_modules"].append(address)
            elif (
                parent_call_label is not None
                and call_label not in seen_pass_children[parent_call_label]
            ):
                seen_pass_children[parent_call_label].add(call_label)
                mbd["module_pass_children"][parent_call_label].append(call_label)
            parent_call_label = call_label


def compute_preview_input_output_distances(trace: Trace) -> None:
    """Run torch's Step-4 depth flood over a finalized preview graph.

    Parameters
    ----------
    trace:
        Finalized preview trace whose ops carry label-based ``parents``/
        ``children`` edges and populated ``input_layers``/``output_layers``.

    Returns
    -------
    None
        ``min/max_distance_from_input``, ``min/max_distance_to_output``,
        ``input_ancestors``, and ``output_descendants`` are populated in place.

    Notes
    -----
    This mirrors ``postprocess.graph_traversal._mark_layer_depths`` but
    resolves labels through an explicit op index instead of
    ``Trace.__getitem__`` (whose finished-mode lookup returns ``Layer``
    objects, not the ops the flood must mutate). Lineage sets are normalized
    first because backends may emit immutable placeholders (e.g. TF's
    ``input_ancestors=()``).
    """

    ops_by_label: dict[str, Any] = {}
    for op_log in trace.layer_list:
        for field_name in ("input_ancestors", "output_descendants"):
            value = getattr(op_log, field_name, None)
            if not isinstance(value, set):
                setattr(op_log, field_name, set(value or ()))
        for key in (
            getattr(op_log, "_label_raw", None),
            getattr(op_log, "label", None),
            getattr(op_log, "layer_label", None),
        ):
            if key is not None:
                ops_by_label.setdefault(key, op_log)

    ordered_ops = [
        ops_by_label[label]
        for label in trace._raw_graph_ws.raw_layer_labels_list
        if label in ops_by_label
    ]
    for mode, starting_labels, min_field, max_field, marker_field, lineage_field, edge_field in (
        (
            "input",
            trace.input_layers,
            "min_distance_from_input",
            "max_distance_from_input",
            "has_input_ancestor",
            "input_ancestors",
            "children",
        ),
        (
            "output",
            trace.output_layers,
            "min_distance_to_output",
            "max_distance_to_output",
            "has_output_descendant",
            "output_descendants",
            "parents",
        ),
    ):
        traversal = ordered_ops if mode == "input" else list(reversed(ordered_ops))
        for starting_label in starting_labels:
            starting_op = ops_by_label.get(starting_label)
            if starting_op is None:
                continue
            _update_distance(starting_op, min_field, max_field, 0)
            setattr(starting_op, marker_field, True)
            getattr(starting_op, lineage_field).add(starting_label)
        for op_log in traversal:
            current_min = getattr(op_log, min_field, None)
            current_max = getattr(op_log, max_field, None)
            if current_min is None or current_max is None:
                continue
            lineage = getattr(op_log, lineage_field)
            for next_label in getattr(op_log, edge_field, ()) or ():
                next_op = ops_by_label.get(next_label)
                if next_op is None:
                    continue
                _update_distance(next_op, min_field, max_field, current_min + 1)
                _update_distance(next_op, min_field, max_field, current_max + 1)
                setattr(next_op, marker_field, True)
                getattr(next_op, lineage_field).update(lineage)


def _update_distance(op_log: Any, min_field: str, max_field: str, hops: int) -> None:
    """Fold one candidate hop count into an op's min/max distance fields.

    Parameters
    ----------
    op_log:
        Op receiving the update.
    min_field, max_field:
        Distance attribute names for the active flood direction.
    hops:
        Candidate hop count from the flood frontier.

    Returns
    -------
    None
        The op's distance fields are widened in place.
    """

    current_min = getattr(op_log, min_field, None)
    current_max = getattr(op_log, max_field, None)
    setattr(op_log, min_field, hops if current_min is None else min(current_min, hops))
    setattr(op_log, max_field, hops if current_max is None else max(current_max, hops))


def _mint_preview_site_keys(trace: Trace) -> None:
    """Mint the policy-independent ``site_key_v1`` on every retained preview op.

    Runs before recurrence assignments are computed (site keys are structural
    facts from raw records, identical whether grouping runs or not -- P4), so
    the preview node builder copies the minted key into its
    ``RecurrenceNode`` and the ungrouped path carries keys all the same.
    Orphan ops consume no ordinals and keep ``site_key=None`` (SF-63).

    Parameters
    ----------
    trace:
        Trace whose ``_raw_graph_ws.raw_layer_dict`` holds materialized
        preview ops in execution order.
    """

    minter = SiteKeyMinter()
    for op_log in trace._raw_graph_ws.raw_layer_dict.values():
        if getattr(op_log, "is_orphan", False):
            continue
        op_log.site_key = minter.mint(
            getattr(op_log, "modules", None) or (),
            str(getattr(op_log, "type", "") or ""),
            (
                getattr(op_log, "multi_output_index", None)
                if getattr(op_log, "in_multi_output", False)
                else None
            ),
        )


def _finalize_single_op(
    trace: Trace,
    op_log: Any,
    label: str,
    raw_index: int,
    assignment: RecurrenceAssignment | None = None,
) -> None:
    """Attach final labels and lookup keys to one op log.

    Parameters
    ----------
    trace:
        Trace receiving lookup indexes.
    op_log:
        Operation log being finalized.
    label:
        Raw backend label.
    raw_index:
        Zero-based raw op index.
    assignment:
        Recurrence assignment for this op when grouping ran. ``None`` (and any
        singleton assignment) reproduces the historical single-pass layout:
        the raw label, with its internal ``_raw`` capture sentinel stripped
        (``strip_raw_label_suffix``), becomes the layer label; the RAW label
        (``_raw`` suffix intact) stays the main lookup key. Multi-pass
        members become pass-qualified: ``label`` is ``layer_label:pass_index``,
        and the main key is the pass label. The bare shared layer label lands
        in ``layer_dict_all_keys`` as an INCIDENTAL raw-index artifact (each
        pass overwrites it, so it resolves to the LAST pass, matching the
        torch backend) — it is NOT a contract; bare-label addressing of
        multi-pass layers refuses on every path that matters
        (``multipass_bare_label_ambiguous``).

    Returns
    -------
    None
        ``op_log`` and trace lookup dictionaries are mutated in place.
    """

    layer_label = strip_raw_label_suffix(
        assignment.layer_label if assignment is not None else label
    )
    pass_index = assignment.pass_index if assignment is not None else 1
    num_passes = assignment.num_passes if assignment is not None else 1
    pass_label = f"{layer_label}:{pass_index}"
    op_log._label_raw = label
    op_log._layer_label_raw = layer_label
    op_log.label = pass_label
    op_log.label_short = pass_label
    op_log.layer_label = layer_label
    op_log.layer_label_short = layer_label
    op_log.lookup_keys = [label, pass_label]
    op_log.pass_index = pass_index
    op_log.num_passes = num_passes
    if assignment is not None:
        op_log.equivalence_class = assignment.equivalence_key
    trace.layer_list.append(op_log)
    trace.layer_dict_main_keys[label if num_passes == 1 else pass_label] = op_log
    trace.layer_dict_all_keys[label] = op_log
    trace.layer_dict_all_keys[pass_label] = op_log
    if num_passes > 1:
        # Incidental, not a contract: the bare layer label is a raw-index
        # artifact that every pass overwrites (last pass wins, torch parity).
        trace.layer_dict_all_keys[layer_label] = op_log
        op_log.lookup_keys.append(layer_label)
    trace.op_labels.append(pass_label)
    if layer_label not in trace.layer_num_calls:
        trace.layer_labels.append(layer_label)
    trace.layer_num_calls[layer_label] = num_passes
    trace._lookup_keys_to_layer_num_dict[label] = raw_index
    trace._layer_num_to_lookup_keys_dict[raw_index].append(label)


def _apply_recurrence_relabel_epilogue(
    trace: Trace,
    assignments: dict[str, RecurrenceAssignment] | None,
    relabel_sidecar_labels: SidecarRelabelHook | None,
) -> None:
    """Relabel graph metadata after per-op final labels were assigned.

    Parameters
    ----------
    trace:
        Trace whose ops already carry final (possibly pass-qualified) labels.
    assignments:
        Recurrence assignments keyed by raw label, or ``None`` when grouping
        did not run (every op is its own singleton, single-pass "group").
    relabel_sidecar_labels:
        Backend hook receiving the complete raw-to-final label mapping so
        label-keyed sidecar state can be remapped atomically.

    Returns
    -------
    None
        Edge metadata, trace-side label lists, equivalence metadata, and
        backend sidecars are updated in place.

    Notes
    -----
    N5: every op's final identity now differs from its raw identity (the raw
    label's internal ``_raw`` capture sentinel is always stripped,
    ``strip_raw_label_suffix``), not just multi-pass group members, so every
    label-bearing edge (``parents``, ``children``, trace-side input/output
    lists, backend sidecars) must be relabeled unconditionally -- this runs
    whether or not recurrence grouping ran (``assignments`` may be ``None``).
    Raw labels stay resolvable through ``lookup_keys`` either way.
    """

    raw_dict = trace._raw_graph_ws.raw_layer_dict
    raw_to_final = {label: raw_dict[label].label for label in raw_dict}
    changed = {label: final for label, final in raw_to_final.items() if final != label}
    if changed:
        for op_log in raw_dict.values():
            relabel_edge_metadata(op_log, changed)
        # Trace-side input/output/source lists speak OP space: each entry must
        # resolve to the specific pass that produced the value (``output_ops``
        # reads them through ``trace[label]``). Inputs and internal sources are
        # pseudo-ops and never group; only lists naming grouped computational
        # ops (an output produced by a later pass) are rewritten, to the
        # pass-qualified final label. The module-log builders map these to
        # layer space at their own boundary.
        for attr_name in (
            "input_layers",
            "output_layers",
            "internal_source_layers",
            "internal_source_ops",
            "buffer_layers",
        ):
            labels = getattr(trace, attr_name, None)
            if isinstance(labels, list):
                setattr(
                    trace,
                    attr_name,
                    [changed.get(item, item) if isinstance(item, str) else item for item in labels],
                )

    equivalent_labels_by_key: dict[str, set[str]] = {}
    for op_log in raw_dict.values():
        equivalent_labels_by_key.setdefault(op_log.equivalence_class, set()).add(op_log.label)
    for label, op_log in raw_dict.items():
        op_log.equivalent_ops = equivalent_labels_by_key[op_log.equivalence_class]
        recurrent_labels = (
            assignments[label].recurrent_labels if assignments is not None else (label,)
        )
        op_log.recurrent_ops = [
            raw_to_final[member] for member in recurrent_labels if member in raw_to_final
        ]
    trace.op_equivalence_classes.clear()
    trace.op_equivalence_classes.update(equivalent_labels_by_key)

    if relabel_sidecar_labels is not None:
        relabel_sidecar_labels(dict(raw_to_final))


def _attach_param_usage(
    trace: Trace,
    op_log: Any,
    membership_by_param: dict[int, tuple[set[str], set[str], set[str]]] | None = None,
) -> None:
    """Update parameter usage cross-links for params attached to one op.

    Parameters
    ----------
    trace:
        Trace receiving ``layers_with_params`` entries.
    op_log:
        Finalized op log that may carry ``_param_logs``.
    membership_by_param:
        Per-pass membership memo shared across ops (back-port of the torch
        path's ``finalization.py`` set-based membership). Without it, the
        bare ``label not in param.used_by_ops`` list scans made a parameter
        consumed by ``m`` ops cost O(m^2) on EVERY preview backend. The
        lists stay authoritative and keep first-seen order; the sets only
        replace the growing-list scans.

    Returns
    -------
    None
        Parameter logs are mutated in place.
    """

    if membership_by_param is None:
        membership_by_param = {}
    for param in getattr(op_log, "_param_logs", []):
        membership = membership_by_param.get(id(param))
        if membership is None:
            membership = (
                set(param.used_by_ops),
                set(param.used_by_layers),
                set(trace.layers_with_params[param.barcode]),
            )
            membership_by_param[id(param)] = membership
        used_by_ops, used_by_layers, layer_membership = membership
        if op_log.label not in used_by_ops:
            param.used_by_ops.append(op_log.label)
            used_by_ops.add(op_log.label)
        if op_log.layer_label not in used_by_layers:
            param.used_by_layers.append(op_log.layer_label)
            used_by_layers.add(op_log.layer_label)
        if op_log.layer_label not in layer_membership:
            trace.layers_with_params[param.barcode].append(op_log.layer_label)
            layer_membership.add(op_log.layer_label)


def _update_param_totals_from_layers(trace: Trace) -> None:
    """Recompute trace parameter totals from finalized single-pass layer logs.

    Parameters
    ----------
    trace:
        Trace whose layer logs already carry per-layer parameter counters.

    Returns
    -------
    None
        Trace-level parameter counters are updated when parameters are present.
    """

    seen_layers: set[str] = set()
    num_param_tensors = 0
    num_params = 0
    num_params_trainable = 0
    for op_log in trace.layer_list:
        if op_log.layer_label in seen_layers:
            continue
        seen_layers.add(op_log.layer_label)
        num_param_tensors += op_log.num_param_tensors
        num_params += op_log.num_params
        num_params_trainable += op_log.num_params_trainable
    if trace.param_source != "none":
        trace.num_param_tensors = num_param_tensors
        trace.num_params = num_params
        trace.num_params_trainable = num_params_trainable
        trace.num_params_frozen = num_params - num_params_trainable
        trace.num_layers_with_params = len(
            {op.layer_label for op in trace.layer_list if op.uses_params}
        )


def numel_from_shape(shape: Any) -> int:
    """Return the number of elements implied by ``shape``.

    Parameters
    ----------
    shape:
        Shape sequence; empty means scalar.

    Returns
    -------
    int
        Product of dimensions (``1`` for a scalar shape).
    """

    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def value_nbytes(value: object) -> int | None:
    """Return byte size for any preview-backend tensor-like value.

    One neutral ladder covering every preview backend's native spelling:
    ``nbytes`` attribute (jax/mlx) or method (tinygrad), ``size * itemsize``
    (mlx fallback), ``numel() * element_size()`` (paddle),
    ``numel() * dtype.itemsize`` (tinygrad fallback), and
    ``shape x dtype.size`` (tf). Each rung is guarded, so a backend value
    settles on exactly the rung its API supports.

    Parameters
    ----------
    value:
        Backend tensor/array-like value.

    Returns
    -------
    int | None
        Byte size when any rung resolves, else ``None``.
    """

    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        try:
            return int(nbytes() if callable(nbytes) else nbytes)
        except Exception:
            pass
    size = getattr(value, "size", None)
    itemsize = getattr(value, "itemsize", None)
    if size is not None and itemsize is not None and not callable(size):
        try:
            return int(size) * int(itemsize)
        except (TypeError, ValueError):
            pass
    numel = getattr(value, "numel", None)
    if callable(numel):
        element_size = getattr(value, "element_size", None)
        if callable(element_size):
            try:
                return int(numel()) * int(element_size())
            except (AttributeError, TypeError, ValueError):
                pass
        dtype_itemsize = getattr(getattr(value, "dtype", None), "itemsize", None)
        if dtype_itemsize is not None:
            try:
                return int(numel()) * int(dtype_itemsize)
            except (TypeError, ValueError):
                pass
    dtype_size = getattr(getattr(value, "dtype", None), "size", None)
    if dtype_size is not None:
        try:
            shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
        except (TypeError, ValueError):
            return None
        return numel_from_shape(shape) * int(dtype_size)
    return None


def session_callable_identity(fn: Callable[..., Any] | None) -> str | None:
    """Return a session-unique best-effort callable identity.

    Includes ``id(fn)``, so the string distinguishes two callables with equal
    qualified names within one process but is NOT stable across sessions.
    Use :func:`stable_callable_name` for persisted provenance.

    Parameters
    ----------
    fn:
        Callable or ``None``.

    Returns
    -------
    str | None
        Identity string used in fingerprints and session provenance.
    """

    if fn is None:
        return None
    return f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', repr(fn))}:{id(fn)}"


def stable_callable_name(fn: Callable[..., Any] | None) -> str | None:
    """Return a stable human-readable callable name.

    No ``id()`` component: equal across sessions for importable callables,
    which is what persisted provenance needs.

    Parameters
    ----------
    fn:
        Callable or ``None``.

    Returns
    -------
    str | None
        Qualified name, or ``repr`` when module/qualname are unavailable.
    """

    if fn is None:
        return None
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return repr(fn)


def mirror_param_derived_grads(trace: Trace, records: Any) -> None:
    """Mirror unambiguous param derived gradients onto param records.

    The ONE five-backend implementation (R17-3): every backend records the
    full superset metadata -- payload, record path, ``has_grad``,
    ``grad_shape``, ``grad_dtype``, and ``gradient_memory``. (mlx/paddle/tf
    historically stopped at ``grad_shape``; that drift is exactly why this
    body is hoisted.)

    Parameters
    ----------
    trace:
        Trace containing backend-derived params.
    records:
        Derived gradient records keyed by leaf path (``params.<address>``).

    Returns
    -------
    None
        Matching ``trace.params`` entries receive the same gradient payload.
    """

    for address, param in trace.params.items():
        record = records.get(f"params.{address}")
        if record is None:
            continue
        param._derived_grad_payload = record.grad
        param._derived_grad_record_path = record.path
        param.has_grad = True
        param.grad_shape = tuple(getattr(record.grad, "shape", ()))
        param.grad_dtype = cast(Any, str(getattr(record.grad, "dtype", "")))
        param.gradient_memory = value_nbytes(record.grad) or 0


def normalize_op_module_calls(value: Any) -> tuple[tuple[str, int], ...]:
    """Normalize an op's raw module-call records to ``(address, call_index)``.

    The ONE five-backend normalizer (R17-6). Accepted spellings:
    ``(address, call_index)`` tuples, single-element tuples (call index
    defaults to ``1``), ``"address:index"`` strings, and bare address strings
    (legacy, call index ``1``). A string with a ``":"`` whose tail is not a
    digit is REFUSED -- module attribution is a correctness surface, and the
    historical per-backend copies silently DROPPED such entries (four
    backends) or crashed on multi-colon strings (jax's first-colon split).

    Parameters
    ----------
    value:
        Materialized op ``modules`` field entries.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Normalized module-call pairs.

    Raises
    ------
    ValueError
        On an entry no accepted spelling matches (malformed capture-side
        module attribution must fail loudly, never vanish from attribution).
    """

    calls: list[tuple[str, int]] = []
    for item in value:
        if isinstance(item, tuple):
            if len(item) >= 2:
                calls.append((str(item[0]), int(item[1])))
                continue
            if len(item) == 1:
                calls.append((str(item[0]), 1))
                continue
            raise ValueError("module-call entry is an empty tuple")
        text = str(item)
        address, separator, index_text = text.rpartition(":")
        if separator:
            if not index_text.isdigit():
                raise ValueError(
                    f"unparseable module-call entry {text!r}: expected 'address:index' "
                    "with a digit index"
                )
            calls.append((address, int(index_text)))
            continue
        calls.append((text, 1))
    return tuple(calls)


def nearest_metadata_parent(address: str, metadata: dict[str, dict[str, Any]]) -> str | None:
    """Return the closest existing parent address for ``address``.

    Parameters
    ----------
    address:
        Child module address.
    metadata:
        Module metadata keyed by address.

    Returns
    -------
    str | None
        Parent address, or ``None`` for root.
    """

    if address == "self":
        return None
    parts = address.split(".")
    while len(parts) > 1:
        parts.pop()
        candidate = ".".join(parts)
        if candidate in metadata:
            return candidate
    return "self" if "self" in metadata else None


def mark_output_label(trace: Trace, label: str) -> None:
    """Mark one resolved output label on a preview trace.

    The shared tail of every preview backend's output marking: append the
    label to ``output_layers`` and amend the producing op event's
    ``is_output_parent`` flag. Backends keep only their native output-tensor
    iteration and label resolution.

    Parameters
    ----------
    trace:
        Trace with live capture events.
    label:
        Resolved raw producer label for one output tensor.

    Returns
    -------
    None
        Mutates ``trace.output_layers`` and the event stream.
    """

    trace.output_layers.append(label)
    event = trace.capture_events.op_event_by_label_raw.get(label)
    if event is None:
        return
    trace.capture_events.append_amendment(
        amend_preview_output_parent_mark(event.seq, label, is_output_parent=True)
    )


def attach_module_owned_op_params(
    op_log: Any,
    trace: Trace,
    seen_param_barcodes: set[str],
) -> None:
    """Attach module-owned parameters to one finalized op log.

    The ONE implementation of the mlx/tf/paddle triplet: the op's owning
    module is its innermost normalized module call, and each parameter
    barcode attaches to the FIRST op of its owner.

    Parameters
    ----------
    op_log:
        Operation log being finalized.
    trace:
        Trace whose parameter accessor owns the backend param logs.
    seen_param_barcodes:
        Mutable set of parameter barcodes already attached to earlier ops.

    Returns
    -------
    None
        Mutates ``op_log`` in place when new params are attached.
    """

    module_calls = normalize_op_module_calls(getattr(op_log, "modules", ()))
    if not module_calls:
        return
    owner = module_calls[-1][0]
    params = [
        param
        for param in trace.param_logs
        if param.module_address == owner and param.barcode not in seen_param_barcodes
    ]
    if not params:
        return
    op_log._param_logs = params
    op_log._param_barcodes = [param.barcode for param in params]
    op_log.param_shapes = [param.shape for param in params]
    op_log.num_params = sum(param.num_params for param in params)
    op_log.num_params_trainable = sum(param.num_params for param in params if param.is_trainable)
    op_log.num_params_frozen = sum(param.num_params for param in params if not param.is_trainable)
    op_log.param_memory = sum(int(param.param_memory) for param in params)
    seen_param_barcodes.update(param.barcode for param in params)


def new_preview_function_trace(
    *,
    backend_name: str,
    model: Callable[..., Any],
    keep_orphans: bool,
    num_context_lines: int,
    recurrence_detection: bool,
    verbose: bool,
    name: str | None,
    raw_input: object | None,
    save_raw_input: str | bool,
    batch_render: str,
    output_transform: object | None,
    save_raw_output: str | bool,
    param_source: str,
    compute_input_output_distances: bool = True,
) -> Trace:
    """Construct an empty function-root preview trace shell.

    The shared jax/tinygrad constructor (their copies differed only in
    ``param_source``); tf keeps its own richer shell because it honors more
    public options.

    Parameters
    ----------
    backend_name:
        Canonical backend name.
    model:
        Captured callable.
    keep_orphans:
        Whether orphan ops are retained.
    num_context_lines:
        Source context line count.
    recurrence_detection:
        Recurrence-detection setting.
    verbose:
        Verbose flag.
    name:
        Optional trace label.
    raw_input:
        Original user input.
    save_raw_input:
        Raw-input save policy.
    batch_render:
        Raw-input render policy.
    output_transform:
        Optional output transform.
    save_raw_output:
        Raw-output save policy.
    param_source:
        Backend parameter provenance (``"pytree-derived"``, ``"none"``, ...).
    compute_input_output_distances:
        Whether the layer-depth flood is requested.

    Returns
    -------
    Trace
        Empty trace initialized for the preview backend.
    """

    trace = Trace(
        model_class_name=getattr(model, "__name__", type(model).__name__),
        output_device="same",
        activation_transform=None,
        grad_transform=None,
        save_raw_activations=True,
        save_raw_gradients=True,
        keep_orphans=keep_orphans,
        save_arg_values=False,
        save_grads=None,
        detach_saved_activations=False,
        mark_layer_depths=compute_input_output_distances,
        num_context_lines=num_context_lines,
        optimizer=None,
        save_code_context=False,
        save_rng_states=False,
        recurrence_detection=recurrence_detection,
        verbose=verbose,
        backward_ready=False,
        module_filter=None,
        emit_nvtx=False,
        transform=None,
        raw_input=raw_input,
        save_raw_input=save_raw_input,
        batch_render=batch_render,
        output_transform=cast("Callable[[Any], Any] | None", output_transform),
        save_raw_output=save_raw_output,
        layer_visualizers=None,
        save_visualizations=False,
    )
    trace.trace_label = name
    trace.backend = cast(BackendName, backend_name)
    trace.module_identity_mode = "function_root"
    trace.param_source = cast("Literal['native-module', 'pytree-derived', 'none']", param_source)
    trace.model_label = trace.model_class_name
    trace.model_class_qualname = getattr(model, "__qualname__", trace.model_class_name)
    # Root entry-point identity fact (C07X item (iv)): preview backends
    # capture a function/module-like root under module_identity_mode
    # "function_root", so the honest closed-kind token is function_call.
    trace.root_entry_point = f"function_call:{trace.model_class_qualname}"
    trace._pre_forward_rng_states = None
    return trace


def module_source_metadata(module: Any) -> dict[str, Any]:
    """Return best-effort source metadata for a preview module-like object.

    Parameters
    ----------
    module
        Module-like object.

    Returns
    -------
    dict[str, Any]
        Source metadata compatible with TorchLens module logs.
    """

    cls = type(module)
    init = getattr(cls, "__init__", None)
    call = getattr(cls, "__call__", None)  # noqa: B004 - fetches the __call__ object, not a callability test
    return {
        "class_source_file": safe_source_file(cls),
        "classsource_line": source_line(cls),
        "init_source_file": safe_source_file(init) if init is not None else None,
        "initsource_line": source_line(init),
        "forward_source_file": safe_source_file(call) if call is not None else None,
        "forwardsource_line": source_line(call),
        "class_docstring": inspect.getdoc(cls),
        "init_signature": signature_string(init),
        "init_docstring": inspect.getdoc(init) if init is not None else None,
        "forward_signature": signature_string(call),
        "forward_docstring": inspect.getdoc(call) if call is not None else None,
    }


def safe_source_file(obj: Any) -> str | None:
    """Return the source file for ``obj`` when inspectable.

    Parameters
    ----------
    obj
        Object to inspect.

    Returns
    -------
    str | None
        Source file path, or ``None`` when ``obj`` is not inspectable (e.g.
        a class defined without a backing source file, such as one built
        via ``exec``/``compile`` or implemented as a builtin).
    """

    try:
        return inspect.getsourcefile(obj)
    except (OSError, TypeError):
        return None


def source_line(obj: Any) -> int | None:
    """Return the first source line for ``obj`` when available.

    Parameters
    ----------
    obj
        Object to inspect.

    Returns
    -------
    int | None
        First source line, or ``None``.
    """

    if obj is None:
        return None
    try:
        return inspect.getsourcelines(obj)[1]
    except (OSError, TypeError):
        return None


def signature_string(obj: Any) -> str | None:
    """Return ``obj``'s signature string when inspectable.

    Parameters
    ----------
    obj
        Callable object.

    Returns
    -------
    str | None
        Signature string, or ``None``.
    """

    if obj is None:
        return None
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def join_module_address(parent: str, child_name: str) -> str:
    """Return a TorchLens child module address.

    Parameters
    ----------
    parent
        Parent module address.
    child_name
        Child field name.

    Returns
    -------
    str
        Joined child address.
    """

    return child_name if parent == "self" else f"{parent}.{child_name}"
