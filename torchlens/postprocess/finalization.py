"""Steps 12-20: Tensor undecoration, timing, param logs, layer/module logs, streaming.

Step 12 (_undecorate_all_saved_tensors): Removes TorchLens tensor metadata from
    all saved tensors and their creation args/kwargs.
Step 13: torch.cuda.empty_cache() — handled inline in __init__.py.
Step 14 (_log_time_elapsed): Records wall-clock timing for cleanup and overall pass.
Step 15 (_finalize_param_logs): Populates Param reverse mappings (used_by_ops,
    used_by_layers, co_parent_params), num_calls, and clears Parameter tensor references.
Step 15.5 (_build_layer_logs): Groups Op entries by layer_label to
    create aggregate Layer objects. Static identity fields still use first-pass
    values, while aggregate fields merge across ops, including conditional
    branch signatures, pass maps, and pass-stripped child views.
Step 16 (_build_module_logs): Builds structured Module/ModuleCall objects from
    _module_build_data and _module_metadata. MUST NOT run in fast mode — _module_build_data
    isn't repopulated in fast mode (Step 9 is skipped). Existing module logs from the
    exhaustive pass remain valid.
Step 17 (_set_tracing_finished): Marks Trace and all OpLogs as finished, switching
    to user-facing mode for display and access custom_methods.
Step 18 (_finalize_streamed_bundle): Finalizes any streamed out bundle.
Step 19 (_evict_streamed_outs): Optionally drops in-memory outs after streaming refs attach.
Step 20 (release_param_refs): Drops live parameter references after finalization.
"""

import time
from collections import defaultdict, deque
from dataclasses import replace
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import torch

from .._io import BlobRef, TorchLensIOError
from .._io.accessor_rebuild import rebuild_trace_accessors
from .._io.lazy import LazyActivationRef
from .._io.manifest import sha256_of_file
from .._io.scrub import BlobSpec
from .._io.streaming import BundleStreamWriter
from ..backends.torch._tl import clear_meta
from ..data_classes._module_role_hints import (
    multi_output_role_from_path,
    role_hints_for_module_class,
)
from ..data_classes._state_adapter import state_items
from ..data_classes._summary import format_call_arg
from ..data_classes.module import Module, ModuleCall
from ..data_classes.prehook import ModuleInputSnapshot, PreHookEffect
from ..ir.container_registry import _iter_tensor_leaves
from ..quantities import Bytes, Duration, Flops
from ..utils._torch_symbols import torch_attr

if TYPE_CHECKING:
    from ..data_classes.layer import Layer
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace


def _undecorate_all_saved_tensors(self: "Trace") -> None:
    """Step 12: Remove TorchLens metadata from all saved tensors.

    During logging, tensors are tagged with private ``_tl`` metadata for
    tracking. This function strips that metadata from saved outs and any
    tensors embedded in saved_args/saved_kwargs, so tensors returned to the
    user are clean.
    """
    tensors_to_undecorate = []
    for layer_entry in [*self.layer_list, *self.orphans.values()]:
        if layer_entry.has_saved_activation and layer_entry.out is not None:
            tensors_to_undecorate.append(layer_entry.out)
        if getattr(layer_entry, "transformed_out", None) is not None:
            tensors_to_undecorate.append(layer_entry.transformed_out)

        if layer_entry.saved_args:
            tensors_to_undecorate.extend(_iter_tensor_leaves(layer_entry.saved_args))
        if layer_entry.saved_kwargs:
            tensors_to_undecorate.extend(_iter_tensor_leaves(layer_entry.saved_kwargs))

    for t in tensors_to_undecorate:
        clear_meta(t)


def _log_time_elapsed(self: "Trace") -> None:
    """Step 14: Record wall-clock timing for the cleanup phase and overall pass.

    Computes cleanup time as the residual after subtracting setup and forward
    pass times from total elapsed time. Also computes torchlens_logging overhead
    as total time minus actual function call time.
    """
    self.capture_end_time = time.time()
    self.cleanup_duration = Duration(
        self.capture_end_time
        - self.capture_start_time
        - self.setup_duration
        - self.forward_duration
    )


def _finalize_param_logs(self: "Trace") -> None:
    """Step 15: Populate Param reverse mappings, co-parent params, and num_calls.

    For each Op with _param_logs:
    - Adds the Op label to each Param's used_by_ops list.
    - Adds the Layer label to each Param's used_by_layers list.
    - Links params that co-occur in the same operation (co_parent_params).
    - Sets num_calls = max(1, len(used_by_ops)).

    Then clears actual Parameter tensor references (parent_params) from
    Op entries to reduce memory. Param._param_ref is released after the
    full finalization pipeline, once all finalization-time param reads finish.
    """
    from ._lazy_param_geometry import _finalize_lazy_param_geometry

    _finalize_lazy_param_geometry(self)

    # Lists remain authoritative and preserve first-seen order. Local sets avoid
    # repeatedly scanning those growing lists for high-arity parameterized ops.
    membership_by_param: dict[int, tuple[set[str], set[str], set[str]]] = {}

    # Build used_by_ops, used_by_layers, and co_parent_params from Op entries
    for layer_entry in self.layer_list:
        if not layer_entry._param_logs:
            continue
        addresses_in_op = [pl.address for pl in layer_entry._param_logs]
        for pl in layer_entry._param_logs:
            membership = membership_by_param.get(id(pl))
            if membership is None:
                membership = (
                    set(pl.used_by_ops),
                    set(pl.used_by_layers),
                    set(pl.co_parent_params),
                )
                membership_by_param[id(pl)] = membership
            used_by_ops, used_by_layers, co_parent_params = membership
            if layer_entry.label not in used_by_ops:
                pl.used_by_ops.append(layer_entry.label)
                used_by_ops.add(layer_entry.label)
            layer_label = layer_entry.layer_label
            if layer_label not in used_by_layers:
                pl.used_by_layers.append(layer_label)
                used_by_layers.add(layer_label)
            # Link to other params in the same operation
            for other_addr in addresses_in_op:
                if other_addr != pl.address and other_addr not in co_parent_params:
                    pl.co_parent_params.append(other_addr)
                    co_parent_params.add(other_addr)

    # Populate num_calls: how many times this parameter was used in the forward pass
    for pl in self.param_logs:
        pl.num_calls = max(1, len(pl.used_by_ops))
        pl.source_trace = self

    # Adopt the finished Param rows into the per-trace kind table (M8): each
    # record's detached single row moves into ONE shared store owned by the
    # trace core; facade behavior is unchanged.
    _core = self.__dict__.get("_trace_core")
    if _core is not None:
        from .._trace_core.record_rows import adopt_records

        adopt_records(_core, "param", self.param_logs)

    # Param grad metadata is populated lazily via backward hooks in _log_tensor_grad.
    # Param._param_ref is intentionally released later, after module finalization has
    # consumed param metadata; Param re-fetches from Trace._source_model_ref on demand.
    # _param_logs (GC-9) and func (GC-10) are still needed by validation.

    # Clear actual Parameter tensor references from Op entries to save memory.
    for layer_entry in self.layer_list:
        layer_entry.parent_params = []


def _build_root_module_log(
    self: "Trace", pass_dict: dict[str, "ModuleCall"], mbd: dict[str, Any]
) -> "Module":
    """Build the root Module ("self") representing the model itself.

    The root module encomops all layers and params. Its address_children are
    only direct children (no dots in address), while call_children may include
    grandchildren called directly (e.g., self.level21.level12(x)).
    """
    from ..data_classes.param import ParamAccessor

    module_metadata = cast(dict[str, dict[str, Any]], self._module_capture_ws.module_metadata)
    root_meta = module_metadata.get("self", {})
    root_layers = list(self.layer_logs.keys())

    root_param_dict = {pl.address: pl for pl in self.param_logs}
    root_params = ParamAccessor(root_param_dict)
    root_num_params = sum(pl.num_params for pl in self.param_logs)
    root_num_trainable = sum(pl.num_params for pl in self.param_logs if pl.is_trainable)
    root_num_frozen = sum(pl.num_params for pl in self.param_logs if not pl.is_trainable)
    root_fsize = Bytes(sum(int(pl.param_memory) for pl in self.param_logs))
    root_meta_children = root_meta.get("address_children")
    if root_meta_children is None:
        address_children = [m for m in mbd["top_level_modules"] if m != "self" and "." not in m]
    else:
        address_children = [
            child for child in root_meta_children if child != "self" and "." not in child
        ]

    root_module = Module(
        address="self",
        all_addresses=root_meta.get("all_addresses", ["self"]),
        name="self",
        cls=root_meta.get("cls"),
        class_name=root_meta.get("class_name", self.model_class_name),
        class_qualname=root_meta.get("class_qualname", ""),
        class_source_file=root_meta.get("class_source_file"),
        class_source_line=root_meta.get("class_source_line"),
        init_source_file=root_meta.get("init_source_file"),
        init_source_line=root_meta.get("init_source_line"),
        forward_source_file=root_meta.get("forward_source_file"),
        forward_source_line=root_meta.get("forward_source_line"),
        class_docstring=root_meta.get("class_docstring"),
        init_signature=root_meta.get("init_signature"),
        init_docstring=root_meta.get("init_docstring"),
        forward_signature=root_meta.get("forward_signature"),
        forward_docstring=root_meta.get("forward_docstring"),
        address_parent=None,
        # address_children: only direct children (no dots in address).
        # top_level_modules may include grandchildren called directly
        # (e.g., self.level21.level12(x)), which belong in call_children
        # but not in the static address hierarchy.
        address_children=address_children,
        address_depth=0,
        call_parent=None,
        call_children=[m for m in mbd["top_level_modules"] if m != "self"],
        call_depth=0,
        num_calls=1,
        ops={},
        call_labels=["self:1"],
        layer_labels=root_layers,
        params=root_params,
        num_params=root_num_params,
        num_params_trainable=root_num_trainable,
        num_params_frozen=root_num_frozen,
        param_memory=root_fsize,
        buffer_layers=list(self.buffer_layers),
        training=root_meta.get("training", True),
        forward_pre_hooks=root_meta.get("forward_pre_hooks"),
        forward_hooks=root_meta.get("forward_hooks"),
        backward_pre_hooks=root_meta.get("backward_pre_hooks"),
        backward_hooks=root_meta.get("backward_hooks"),
        full_backward_pre_hooks=root_meta.get("full_backward_pre_hooks"),
        full_backward_hooks=root_meta.get("full_backward_hooks"),
        custom_attributes=root_meta.get("custom_attributes", {}),
        custom_methods=root_meta.get("custom_methods", []),
        _source_trace=self,
    )

    root_pre_hook_provenance = _pre_hook_provenance_for_call(self, "self:1")
    root_pass = ModuleCall(
        address="self",
        call_index=1,
        call_label="self:1",
        ops=root_layers,
        input_layers=_layer_space_labels(self, self.input_layers),
        output_layers=_layer_space_labels(self, self.output_layers),
        output_ops=list(self.output_layers),
        output_structure=_first_output_structure(self, list(self.output_layers)),
        inputs_before_pre_hooks=root_pre_hook_provenance[0],
        inputs_after_pre_hooks=root_pre_hook_provenance[1],
        forward_pre_hook_effects=root_pre_hook_provenance[2],
        call_parent=None,
        call_children=_root_call_children(mbd),
        all_addresses=root_meta.get("all_addresses", ["self"]),
        cls=root_meta.get("cls"),
        class_name=root_meta.get("class_name", self.model_class_name),
        class_qualname=root_meta.get("class_qualname", ""),
        ordinal_index=0,
        forward_duration=float(self.forward_duration or 0.0),
        code_context=list(getattr(self, "code_context", [])),
        module_call_stack=[],
        _source_trace=self,
    )
    root_module.ops[1] = root_pass
    root_module._sync_boundary_fields_from_calls()
    pass_dict["self:1"] = root_pass

    return root_module


def _pre_hook_provenance_for_call(
    trace: "Trace", call_label: str
) -> tuple[ModuleInputSnapshot | None, ModuleInputSnapshot | None, tuple[PreHookEffect, ...]]:
    """Return provenance for one call with raw producer labels finalized.

    Parameters
    ----------
    trace:
        Trace whose Step-8 raw-to-final operation map is available.
    call_label:
        Pass-qualified module call label.

    Returns
    -------
    tuple[ModuleInputSnapshot | None, ModuleInputSnapshot | None, tuple[PreHookEffect, ...]]
        Before snapshot, after snapshot, and ordered effects.
    """

    values = trace._module_capture_ws.module_build_data.get("module_pre_hook_provenance", {}).get(
        call_label
    )
    if values is None:
        return None, None, ()
    before, after, effects = values
    return (
        _finalize_snapshot_source_labels(trace, before),
        _finalize_snapshot_source_labels(trace, after),
        tuple(effects),
    )


def _finalize_snapshot_source_labels(
    trace: "Trace", snapshot: ModuleInputSnapshot | None
) -> ModuleInputSnapshot | None:
    """Replace raw tensor producer labels in one immutable snapshot."""

    if snapshot is None:
        return None
    mapping = trace._raw_to_final_op_labels
    observations = tuple(
        replace(
            observation,
            source_op=(
                mapping.get(observation.source_op, observation.source_op)
                if observation.source_op is not None
                else None
            ),
        )
        for observation in snapshot.tensor_observations
    )
    return replace(snapshot, tensor_observations=observations)


def _root_call_children(mbd: dict[str, Any]) -> list[str]:
    """Resolve the root ``self:1`` ModuleCall's direct call children.

    Two capture shapes feed this:

    * Function-root traces (``self`` is synthetic and never appears in an op's
      module stack): the direct children are the outermost real module calls,
      recorded in ``top_level_module_ops`` while ``module_pass_children['self:1']``
      stays empty.
    * Object-module / explicit-self traces (``self`` is itself a traced module,
      so ops carry ``['self:1', child:1, ...]`` stacks): the outermost call IS
      ``self:1`` itself, so ``top_level_module_ops`` collapses to ``['self:1']``
      and the real direct children land in ``module_pass_children['self:1']``.

    Unioning both sources (preserving order, dropping the self-reference) yields
    the correct children in both shapes. Previously only ``top_level_module_ops``
    was consulted, so the explicit-self shape silently dropped every child of
    ``self:1`` -- an asymmetric call-tree the module-hierarchy invariant rightly
    flags (parent lists no children while children point back to ``self:1``).
    """

    children: list[str] = []
    for label in mbd.get("top_level_module_ops", []):
        if label != "self:1" and label not in children:
            children.append(label)
    for label in mbd.get("module_pass_children", {}).get("self:1", []):
        if label != "self:1" and label not in children:
            children.append(label)
    return children


def _compute_call_depths(module_dict: dict[str, "Module"], root_module: "Module") -> None:
    """Assign call_depth to each Module via BFS from the root.

    Root ("self") has depth 0. Each level of call_children nesting adds 1.
    """
    visited = {"self": 0}
    queue: deque[str] = deque()
    for child_addr in root_module.call_children:
        if child_addr in module_dict:
            module_dict[child_addr].call_depth = 1
            visited[child_addr] = 1
            queue.append(child_addr)

    while queue:
        addr = queue.popleft()
        ml = module_dict[addr]
        for child_addr in ml.call_children:
            if child_addr not in visited and child_addr in module_dict:
                depth = visited[addr] + 1
                module_dict[child_addr].call_depth = depth
                visited[child_addr] = depth
                queue.append(child_addr)


def _append_unique_child_label(
    child_labels: list[str], seen_labels: set[str], child_label: str
) -> None:
    """Append a child label if it has not been seen yet.

    Parameters
    ----------
    child_labels:
        Ordered list being accumulated.
    seen_labels:
        Membership set mirroring ``child_labels`` (the bare list scan made
        each aggregate projection O(k^2) in its child count -- R52).
    child_label:
        Child label to append.
    """
    if child_label not in seen_labels:
        seen_labels.add(child_label)
        child_labels.append(child_label)


def _strip_pass_suffix(layer_label: str) -> str:
    """Strip the ``:call_index`` suffix from a layer label.

    Parameters
    ----------
    layer_label:
        Pass-qualified or pass-free layer label.

    Returns
    -------
    str
        Layer label without any pass suffix.
    """
    return layer_label.split(":", 1)[0]


def _layer_space_labels(self: "Trace", labels: list[str]) -> list[str]:
    """Map op-space labels to layer space for module-log containers.

    Module-log ``input_layers``/``output_layers`` speak layer-label space (the
    backend-neutral module invariants require them to be a subset of
    ``trace.layer_labels``), while the trace-side lists speak op space so each
    entry resolves to the specific producing pass. For torch the two spaces
    coincide (inputs and outputs are dedicated single-pass nodes); they diverge
    only for preview backends whose recurrence grouping made an output the
    later pass of a multi-pass layer.

    Parameters
    ----------
    self:
        Trace owning the label lookup tables.
    labels:
        Op-space labels (raw, pass-qualified, or layer labels).

    Returns
    -------
    list[str]
        Layer labels, preserving order; unresolvable labels pass through.
    """

    layer_labels: list[str] = []
    for label in labels:
        op = self.layer_dict_all_keys.get(label)
        resolved = getattr(op, "layer_label", None) if op is not None else None
        layer_labels.append(resolved if isinstance(resolved, str) else label)
    return layer_labels


def _first_output_structure(self: "Trace", output_layers: list[str]) -> Any | None:
    """Return the first non-empty container spec for output layers.

    Parameters
    ----------
    self:
        Trace containing the output layer records.
    output_layers:
        Pass-qualified layer labels to inspect.

    Returns
    -------
    Any | None
        The first container spec found on the output OpLogs, or ``None``.
    """

    for layer_label in output_layers:
        entry = self.layer_dict_all_keys.get(layer_label)
        if entry is not None and entry.container_spec is not None:
            return entry.container_spec
    return None


def _assign_output_roles(
    output_entries: list["Op"],
    *,
    hints: Any | None,
) -> None:
    """Populate semantic multi-output roles on output entries.

    Parameters
    ----------
    output_entries:
        Output OpLogs from one module call.
    hints:
        Optional role-hint mapping keyed by primitive output paths.
    """

    if len(output_entries) <= 1:
        return
    for index, output in enumerate(output_entries):
        if output.multi_output_name is not None:
            parent_layer = output.source_trace.layer_logs.get(output.layer_label)
            if parent_layer is not None:
                parent_layer.multi_output_name = output.multi_output_name
            continue
        output.multi_output_name = multi_output_role_from_path(
            tuple(output.container_path or ()),
            output.multi_output_index if output.multi_output_index is not None else index,
            hints=hints,
        )
        parent_layer = output.source_trace.layer_logs.get(output.layer_label)
        if parent_layer is not None:
            parent_layer.multi_output_name = output.multi_output_name


def _merge_layer_log_conditional_fields(
    layer_log: "Layer",
    pass_log: "Op",
) -> None:
    """Merge one pass's conditional metadata into an aggregate ``Layer``.

    Parameters
    ----------
    layer_log:
        Aggregate layer entry being updated.
    pass_log:
        Per-pass layer entry contributing conditional metadata.
    """
    if pass_log.is_in_conditional_body:
        layer_log.is_in_conditional_body = True

    stack_signature = tuple(pass_log.conditional_branch_stack)
    if stack_signature not in layer_log.conditional_branch_stack_ops:
        layer_log.conditional_role_stacks.append(list(pass_log.conditional_branch_stack))
        layer_log.conditional_branch_stack_ops[stack_signature] = []

    signature_ops = layer_log.conditional_branch_stack_ops[stack_signature]
    if pass_log.pass_index not in signature_ops:
        signature_ops.append(pass_log.pass_index)
        signature_ops.sort()

    for conditional_id, branch_children in pass_log.conditional_arm_children.items():
        merged_branch_children = layer_log.conditional_arm_children.setdefault(
            conditional_id,
            {},
        )
        for branch_kind, child_labels in branch_children.items():
            merged_child_labels = merged_branch_children.setdefault(branch_kind, [])
            merged_seen = set(merged_child_labels)
            for child_label in child_labels:
                _append_unique_child_label(
                    merged_child_labels,
                    merged_seen,
                    _strip_pass_suffix(child_label),
                )


def _rebuild_layer_log_conditional_views(layer_log: "Layer") -> None:
    """Recompute aggregate conditional compatibility views for one ``Layer``.

    Parameters
    ----------
    layer_log:
        Aggregate layer entry whose derived conditional fields are refreshed.
    """
    layer_log.is_in_conditional_body = any(
        len(branch_stack) > 0 for branch_stack in layer_log.conditional_role_stacks
    )

    conditional_entry_children: list[str] = []
    entry_seen: set[str] = set()
    for _, pass_log in sorted(layer_log.ops.items()):
        for child_label in pass_log.conditional_entry_children:
            _append_unique_child_label(
                conditional_entry_children,
                entry_seen,
                _strip_pass_suffix(child_label),
            )
    layer_log.conditional_entry_children = conditional_entry_children

    conditional_then_children: list[str] = []
    then_seen: set[str] = set()
    conditional_elif_children: dict[int, list[str]] = {}
    elif_seen_by_index: dict[int, set[str]] = {}
    conditional_else_children: list[str] = []
    else_seen: set[str] = set()
    for branch_children in layer_log.conditional_arm_children.values():
        for child_label in branch_children.get("then", []):
            _append_unique_child_label(conditional_then_children, then_seen, child_label)
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            elif_children = conditional_elif_children.setdefault(elif_index, [])
            elif_seen = elif_seen_by_index.setdefault(elif_index, set())
            for child_label in child_labels:
                _append_unique_child_label(elif_children, elif_seen, child_label)
        for child_label in branch_children.get("else", []):
            _append_unique_child_label(conditional_else_children, else_seen, child_label)

    layer_log.conditional_then_children = conditional_then_children
    layer_log.conditional_elif_children = conditional_elif_children
    layer_log.conditional_else_children = conditional_else_children


def _rebuild_conditional_edge_call_indices(self: "Trace") -> None:
    """Recompute rolled conditional edge-pass metadata from arm-entry edges.

    Parameters
    ----------
    self:
        Model log being finalized.
    """
    if getattr(self, "conditional_edge_call_indices", None):
        self.conditional_edge_call_indices = {
            edge_key: sorted(dict.fromkeys(call_indexs))
            for edge_key, call_indexs in self.conditional_edge_call_indices.items()
        }
        return

    conditional_edge_call_indices: dict[tuple[str, str, int, str], list[int]] = defaultdict(list)
    for (conditional_id, branch_kind), edge_list in self.conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            parent_no_pass = _strip_pass_suffix(parent_label)
            child_no_pass = _strip_pass_suffix(child_label)
            child_call_index = _get_label_call_index(child_label)
            edge_key = (
                parent_no_pass,
                child_no_pass,
                conditional_id,
                branch_kind,
            )
            if child_call_index not in conditional_edge_call_indices[edge_key]:
                conditional_edge_call_indices[edge_key].append(child_call_index)

    self.conditional_edge_call_indices = {
        edge_key: sorted(call_indexs)
        for edge_key, call_indexs in conditional_edge_call_indices.items()
    }


def _get_label_call_index(layer_label: str) -> int:
    """Extract the pass number encoded in a layer label.

    Parameters
    ----------
    layer_label:
        Layer label that may include a ``:call_index`` suffix.

    Returns
    -------
    int
        Parsed pass number, or ``1`` for pass-free labels.
    """
    label_parts = layer_label.rsplit(":", 1)
    if len(label_parts) == 2 and label_parts[1].isdigit():
        return int(label_parts[1])
    return 1


def _build_submodule_call_logs(
    self: "Trace",
    address: str,
    num_calls: int,
    pass_dict: dict[str, "ModuleCall"],
    mbd: dict[str, Any],
    _child_to_parent_pass: dict[str, str] | None = None,
    all_addresses: list[str] | None = None,
    module_class: type[torch.nn.Module] | None = None,
    pass_input_layers_by_call: dict[str, list[str]] | None = None,
) -> tuple[dict[int, "ModuleCall"], list[str]]:
    """Build ModuleCall objects for all ops of a single submodule.

    For each pass, derives input/output layers from Op fields,
    retrieves forward args, and resolves call parent/children relationships.

    Returns:
        Tuple of (ops dict {call_index: ModuleCall}, call_labels list).
    """
    ops = {}
    call_labels_list = []
    for call_index in range(1, num_calls + 1):
        call_label = f"{address}:{call_index}"
        call_labels_list.append(call_label)

        pass_layers = list(mbd["module_pass_layers"].get(call_label, []))

        # Derive input/output layers per pass from Op fields
        pass_input_layers = list((pass_input_layers_by_call or {}).get(call_label, []))
        pass_output_layers = []
        pass_output_ops = []
        pass_outputs = []
        for op_label in pass_layers:
            try:
                te = self.ops[op_label]
            except (KeyError, TypeError):
                continue
            # B3R7-R05-1: ``input_to_module_calls`` is the authoritative
            # fed-call fact. The old ``len(module_call_stack) > 0`` companion
            # test was a proxy for the same thing only while the enter lane
            # wrote both together; under containment semantics it would drop
            # top-level-created inputs (e.g. the model input) from the call's
            # input roster.
            if call_label in te.input_to_module_calls and te.layer_label not in pass_input_layers:
                pass_input_layers.append(te.layer_label)
            if te.is_module_output and call_label in te.output_of_module_calls:
                pass_output_layers.append(te.layer_label)
                pass_output_ops.append(te.label)
                pass_outputs.append(te)

        role_hints = role_hints_for_module_class(module_class)
        _assign_output_roles(pass_outputs, hints=role_hints)

        # Forward args for this pass
        module_forward_args = cast(
            dict[tuple[str, int], tuple[Any, Any]], self._module_capture_ws.module_forward_args
        )
        fwd_args = module_forward_args.get((address, call_index))
        fwd_positional = fwd_args[0] if fwd_args else None
        fwd_kwargs = fwd_args[1] if fwd_args else None
        fwd_templates = mbd.get("module_forward_templates", {}).get(call_label)
        fwd_args_template = fwd_templates[0] if fwd_templates else None
        fwd_kwargs_template = fwd_templates[1] if fwd_templates else None

        # Call children for this pass
        call_children_pass = list(mbd["module_pass_children"].get(call_label, []))

        # Call parent for this pass: look up from pre-computed reverse mapping
        call_parent_pass = _child_to_parent_pass.get(call_label) if _child_to_parent_pass else None
        if call_parent_pass is None and call_label in mbd["top_level_module_ops"]:
            call_parent_pass = "self:1"

        pre_hook_provenance = _pre_hook_provenance_for_call(self, call_label)
        module_call_log = ModuleCall(
            address=address,
            call_index=call_index,
            call_label=call_label,
            ops=pass_layers,
            input_layers=pass_input_layers,
            output_layers=pass_output_layers,
            output_ops=pass_output_ops,
            output_structure=mbd.get("module_output_structures", {}).get(
                call_label, _first_output_structure(self, pass_output_layers)
            ),
            output_paths=mbd.get("module_output_paths", {}).get(call_label, ()),
            forward_args=fwd_positional,
            forward_kwargs=fwd_kwargs,
            inputs_before_pre_hooks=pre_hook_provenance[0],
            inputs_after_pre_hooks=pre_hook_provenance[1],
            forward_pre_hook_effects=pre_hook_provenance[2],
            forward_args_template=fwd_args_template,
            forward_kwargs_template=fwd_kwargs_template,
            forward_arg_names=[
                str(arg_key)
                for _layer_label, arg_key in mbd["module_layer_argnames"].get(call_label, [])
            ],
            forward_duration=mbd.get("module_forward_durations", {}).get(call_label, 0.0),
            code_context=mbd.get("module_code_contexts", {}).get(call_label, []),
            module_call_stack=mbd.get("module_call_stacks", {}).get(call_label, []),
            call_parent=call_parent_pass,
            call_children=call_children_pass,
            all_addresses=all_addresses,
            cls=module_class,
            class_name=getattr(module_class, "__name__", "") if module_class is not None else "",
            class_qualname=(
                f"{module_class.__module__}.{module_class.__qualname__}"
                if module_class is not None
                else ""
            ),
            ordinal_index=len(pass_dict),
            _source_trace=self,
        )
        ops[call_index] = module_call_log
        pass_dict[call_label] = module_call_log

    return ops, call_labels_list


def _resolve_call_hierarchy(
    ops: dict[int, "ModuleCall"],
) -> tuple[list[str], str | None]:
    """Derive module-level call_children and call_parent from per-pass data.

    Unions call_children across all ops (stripping pass suffixes to get
    addresses). call_parent is taken from the first pass and converted from
    pass-label to address.

    Returns:
        Tuple of (call_children_all: list of addresses, call_parent_addr or None).
    """
    call_children_all = []
    for module_call_log in ops.values():
        for child_call_label in module_call_log.call_children:
            cc_addr = child_call_label.rsplit(":", 1)[0]
            if cc_addr not in call_children_all:
                call_children_all.append(cc_addr)

    call_parent_addr = None
    if ops:
        first_pass = next(iter(ops.values()))
        if first_pass.call_parent and first_pass.call_parent != "self:1":
            call_parent_addr = first_pass.call_parent.rsplit(":", 1)[0]
        elif first_pass.call_parent == "self:1":
            call_parent_addr = "self"

    return call_children_all, call_parent_addr


class ModuleParamInfo(NamedTuple):
    """Parameter and buffer info for a single module."""

    params: object  # ParamAccessor
    num_params: int
    num_trainable: int
    num_frozen: int
    memory: Bytes
    buffer_layers: list[str]


def _build_module_param_info(
    self: "Trace",
    address: str,
    mbd: dict[str, Any],
    _buffer_layers_by_module: dict[str, list[str]] | None = None,
) -> ModuleParamInfo:
    """Gather parameter counts, sizes, and buffer layers for a single module."""
    from ..data_classes.param import ParamAccessor

    module_param_dict = {pl.address: pl for pl in self._param_logs_by_module.get(address, [])}
    module_params = ParamAccessor(module_param_dict)
    m_num_params = mbd["module_nparams"].get(address, 0)
    m_num_trainable = mbd["module_nparams_trainable"].get(address, 0)
    m_num_frozen = mbd["module_nparams_frozen"].get(address, 0)
    m_fsize = Bytes(sum(int(pl.param_memory) for pl in module_param_dict.values()))

    if _buffer_layers_by_module is not None:
        module_buffer_layers = list(_buffer_layers_by_module.get(address, []))
    else:
        module_buffer_layers = [
            bl
            for bl in self.buffer_layers
            if bl in self.layer_dict_all_keys
            and hasattr(self.layer_dict_all_keys[bl], "address")
            and self.layer_dict_all_keys[bl].address is not None
            and self.layer_dict_all_keys[bl].address.rsplit(".", 1)[0] == address
        ]

    return ModuleParamInfo(
        module_params, m_num_params, m_num_trainable, m_num_frozen, m_fsize, module_buffer_layers
    )


def _build_module_logs(self: "Trace") -> None:
    """Step 16: Build structured Module/ModuleCall objects.

    Constructs the module hierarchy from _module_build_data (populated in Step 9)
    and _module_metadata (captured during model preparation). Creates:
    - A root Module for "self" (the model itself).
    - ModuleLogs for each submodule with ModuleCallLogs for each pass.
    - ModuleAccessor and BufferAccessor for user-facing access.

    Handles shared modules (same nn.Module registered under multiple addresses)
    via an alias-to-metadata map. Computes nesting depths via BFS from root.
    Clears temporary state (_module_metadata, _module_forward_args, _module_build_data)
    after building.
    """
    mbd = self._module_capture_ws.module_build_data
    module_dict = {}  # address -> Module
    pass_dict: dict[str, ModuleCall] = {}  # "addr:pass" -> ModuleCall
    module_order = []  # ordered by first appearance

    # --- Build root Module ("self") ---
    root_module = _build_root_module_log(self, pass_dict, mbd)
    module_dict["self"] = root_module
    module_order.append(root_module)

    # Pre-compute param_logs grouped by module address
    self._param_logs_by_module = defaultdict(list)
    for pl in self.param_logs:
        for module_address in pl.all_module_addresses:
            self._param_logs_by_module[module_address].append(pl)

    # Pre-compute reverse mapping: child_call_label -> parent_call_label
    _child_to_parent_pass = {}
    for parent_call_label, children in mbd["module_pass_children"].items():
        for child_label in children:
            _child_to_parent_pass[child_label] = parent_call_label

    # Build alias-to-metadata map for shared modules (same nn.Module instance
    # registered under multiple addresses). _module_metadata stores metadata
    # under the FIRST address visited by _capture_module_metadata, but
    # Module addresses may be overwritten to a LATER address by
    # _prepare_model_once. This map ensures all aliases resolve to the same meta.
    _metadata_by_alias: dict[str, dict[str, Any]] = {}
    for _primary_addr, _meta in self._module_capture_ws.module_metadata.items():
        for _alias in _meta.get("all_addresses", [_primary_addr]):
            _metadata_by_alias[_alias] = _meta

    # Pre-compute buffer layers grouped by parent module address (O6).
    _buffer_layers_by_module = defaultdict(list)
    for bl in self.buffer_layers:
        if bl in self.layer_dict_all_keys:
            bl_entry = self.layer_dict_all_keys[bl]
            if hasattr(bl_entry, "address") and bl_entry.address is not None:
                module_addr = bl_entry.address.rsplit(".", 1)[0]
                _buffer_layers_by_module[module_addr].append(bl)

    # Pre-compute input layers for each module call.  Large traces may have
    # tens of thousands of ModuleCalls; scanning every Op once per call is
    # quadratic and dominates validation postprocessing.
    _pass_input_layers_by_call: dict[str, list[str]] = defaultdict(list)
    for te in self.layer_list:
        # B3R7-R05-1: iterate the fed-call fact directly; the old
        # ``module_call_stack`` emptiness guard was an enter-lane proxy.
        for call_label in te.input_to_module_calls:
            _pass_input_layers_by_call[call_label].append(te.layer_label)

    # --- Build ModuleLogs for each submodule ---
    for address in mbd["addresses"]:
        if address == "self":
            continue
        meta = _metadata_by_alias.get(address, {})
        num_calls = mbd["module_num_calls"].get(address, 1)

        name = address.rsplit(".", 1)[-1] if "." in address else address
        address_parent = address.rsplit(".", 1)[0] if "." in address else "self"
        address_depth = address.count(".") + 1
        layers_raw = list(mbd["module_layers"].get(address, []))
        seen = set()
        layers = []
        for label in layers_raw:
            entry = self.layer_dict_all_keys.get(label)
            if entry is not None:
                no_pass = entry.layer_label
                if no_pass not in seen:
                    seen.add(no_pass)
                    layers.append(no_pass)

        all_addresses = meta.get("all_addresses", [address])
        ops, call_labels_list = _build_submodule_call_logs(
            self,
            address,
            num_calls,
            pass_dict,
            mbd,
            _child_to_parent_pass,
            all_addresses=all_addresses,
            module_class=meta.get("cls"),
            pass_input_layers_by_call=_pass_input_layers_by_call,
        )
        call_children_all, call_parent_addr = _resolve_call_hierarchy(ops)
        param_info = _build_module_param_info(self, address, mbd, _buffer_layers_by_module)

        # address_children from metadata may have a different address prefix
        # when the metadata was captured for a shared module under a different
        # alias.  Extract child names and reconstruct with current address.
        meta_children = meta.get("address_children")
        if meta_children is not None:
            meta_primary = meta.get("all_addresses", [address])[0]
            if meta_primary == address:
                address_children = meta_children
            else:
                # Rewrite: strip old prefix, prepend current address
                prefix = meta_primary + "."
                address_children = []
                for child_addr in meta_children:
                    if child_addr.startswith(prefix):
                        child_name = child_addr[len(prefix) :]
                        address_children.append(f"{address}.{child_name}")
                    else:
                        address_children.append(child_addr)
        else:
            address_children = list(mbd["module_children"].get(address, []))

        ml = Module(
            address=address,
            all_addresses=meta.get("all_addresses", [address]),
            name=name,
            cls=meta.get("cls"),
            class_name=meta.get("class_name", mbd["module_types"].get(address, "")),
            class_qualname=meta.get("class_qualname", ""),
            class_source_file=meta.get("class_source_file"),
            class_source_line=meta.get("class_source_line"),
            init_source_file=meta.get("init_source_file"),
            init_source_line=meta.get("init_source_line"),
            forward_source_file=meta.get("forward_source_file"),
            forward_source_line=meta.get("forward_source_line"),
            class_docstring=meta.get("class_docstring"),
            init_signature=meta.get("init_signature"),
            init_docstring=meta.get("init_docstring"),
            forward_signature=meta.get("forward_signature"),
            forward_docstring=meta.get("forward_docstring"),
            address_parent=address_parent,
            address_children=address_children,
            address_depth=address_depth,
            call_parent=call_parent_addr,
            call_children=call_children_all,
            call_depth=0,  # computed below
            num_calls=num_calls,
            ops=ops,
            call_labels=call_labels_list,
            layer_labels=layers,
            params=param_info.params,  # type: ignore[arg-type]
            num_params=param_info.num_params,
            num_params_trainable=param_info.num_trainable,
            num_params_frozen=param_info.num_frozen,
            param_memory=param_info.memory,
            buffer_layers=param_info.buffer_layers,
            training=mbd["module_training_modes"].get(address, meta.get("training", True)),
            forward_pre_hooks=meta.get("forward_pre_hooks"),
            forward_hooks=meta.get("forward_hooks"),
            backward_pre_hooks=meta.get("backward_pre_hooks"),
            backward_hooks=meta.get("backward_hooks"),
            full_backward_pre_hooks=meta.get("full_backward_pre_hooks"),
            full_backward_hooks=meta.get("full_backward_hooks"),
            custom_attributes=meta.get("custom_attributes", {}),
            custom_methods=meta.get("custom_methods", []),
            _source_trace=self,
        )
        module_dict[address] = ml
        module_order.append(ml)

    # --- Compute nesting depths ---
    _compute_call_depths(module_dict, root_module)

    # Adopt the finished Module/ModuleCall rows into the per-trace kind
    # tables (M8) before the accessors are rebuilt around them.
    _core = self.__dict__.get("_trace_core")
    if _core is not None:
        from .._trace_core.record_rows import adopt_records

        adopt_records(_core, "module", module_dict.values())
        adopt_records(_core, "module_call", pass_dict.values())

    rebuild_trace_accessors(self, module_dict, module_order, pass_dict)

    # Clean up temporary build state to free memory. These dicts are only
    # needed during construction and are not part of the user-facing API.
    self._module_capture_ws.module_metadata = {}
    self._module_capture_ws.module_forward_args = {}
    from ..ir.workspaces import _init_module_hierarchy_data

    self._module_capture_ws.module_build_data = _init_module_hierarchy_data()

    # GC-11: Clear forward_args/kwargs from ModuleCallLogs to release tensor references.
    # These can hold large tensors from the model's forward() call args.
    keep_forward_args = getattr(self, "module_identity_mode", None) in {
        "pytree_module",
        "object_module",
    }
    for pass_log in pass_dict.values():
        pass_log.forward_args_summary = format_call_arg(pass_log.forward_args)
        pass_log.forward_kwargs_summary = format_call_arg(pass_log.forward_kwargs)
        if not keep_forward_args:
            pass_log.forward_args = None
            pass_log.forward_kwargs = None


# Aggregate Layer fields that may legitimately DIVERGE across genuine recurrent
# passes (variable-length recurrence, autocast dtype shifts, tied-site module
# attribution) and therefore must never be silently projected from pass 1.
# Fields absent here are structurally uniform per the grouping identity
# (function, parameters, output slot, module address, non-tensor args).
_MULTIPASS_SHAPE_FIELDS = ("shape", "transformed_out_shape")
_MULTIPASS_BYTES_FIELDS = ("activation_memory", "transformed_activation_memory")
_MULTIPASS_FLOPS_FIELDS = ("flops_forward", "flops_backward")
_MULTIPASS_MARKER_ONLY_FIELDS = ("dtype", "transformed_out_dtype", "device_ref")


def _honest_multipass_shape(values: list[Any]) -> tuple:
    """Return an honest aggregate shape for divergent per-pass shapes.

    Constant dimensions keep their integer; divergent dimensions become an
    explicit ``"lo..hi"`` range token. Rank divergence collapses to
    ``("varies",)`` — never a silently wrong pass-1 tuple.

    Parameters
    ----------
    values:
        Per-pass shape values in pass order.

    Returns
    -------
    tuple
        Honest aggregate shape.
    """

    tuples = [tuple(value) for value in values if isinstance(value, (tuple, list))]
    if len(tuples) != len(values) or len({len(t) for t in tuples}) != 1:
        return ("varies",)
    dims: list[Any] = []
    # The guard above already returned ("varies",) unless every tuple has the
    # same length, so this transpose is exact rather than truncating.
    for dim_values in zip(*tuples, strict=True):
        distinct = set(dim_values)
        if len(distinct) == 1:
            dims.append(dim_values[0])
        elif all(isinstance(dim, int) for dim in dim_values):
            dims.append(f"{min(dim_values)}..{max(dim_values)}")
        else:
            dims.append("varies")
    return tuple(dims)


def _reconcile_multipass_layer_fields(layer_log: "Layer") -> None:
    """Make multi-pass aggregate Layer fields honest about divergent passes.

    ``Layer`` construction projects pass-1 values onto 78+ aggregate fields on the
    premise that grouping guarantees uniform structural metadata. That premise is
    FALSE for genuine variable-length recurrence: a 3-step recurrent Linear over a
    shrinking sequence has per-pass shapes ``[(4, 4), (3, 4), (2, 4)]``, yet the
    rolled view published ``(4, 4), 64 B`` for every call — silent metadata and
    rolled-render corruption that forward validation cannot catch.

    For every divergent field this records the explicit machine-readable marker
    ``layer_log.annotations["varying_across_passes"]`` (field name -> per-pass
    values in pass order) and, for the display-critical fields, replaces the
    pass-1 projection with an honest aggregate: per-dimension range tuples for
    shapes, and the per-pass MAXIMUM (a documented upper bound, not a call-1
    sample) for byte and FLOP counts. ``Layer.ops`` remains the exact per-pass
    truth. Uniform layers — every non-recurrent model and every fixed-shape
    loop — are left byte-identical.

    The byte/FLOP aggregates stay plain ``Bytes``/``Flops``: the ``.tlspec``
    metadata unpickler admits types by a frozen default-deny allowlist, so a
    range-displaying quantity subclass would fail every load of a trace with a
    varying-shape recurrence until ``quantities`` grows a vetted first-class
    range type (deferred to the owner).

    Parameters
    ----------
    layer_log:
        Aggregate layer with its ``ops`` accessor fully populated.
    """

    if len(layer_log.ops) <= 1:
        return
    pass_ops = list(layer_log.ops.values())
    varying: dict[str, list[Any]] = {}

    def record_if_varying(field_name: str, values: list[Any]) -> bool:
        """Record ``field_name`` as varying when its per-pass values are not all equal.

        Returns ``True`` when the values differ, so the caller replaces the
        representative field with an honest multi-pass summary.
        """

        distinct_count = len({repr(value) for value in values})
        if distinct_count > 1:
            varying[field_name] = values
            return True
        return False

    for field_name in _MULTIPASS_SHAPE_FIELDS:
        values = [getattr(op, field_name, None) for op in pass_ops]
        if record_if_varying(field_name, values):
            setattr(layer_log, field_name, _honest_multipass_shape(values))

    for field_name, quantity_type in (
        *((name, Bytes) for name in _MULTIPASS_BYTES_FIELDS),
        *((name, Flops) for name in _MULTIPASS_FLOPS_FIELDS),
    ):
        values = [getattr(op, field_name, None) for op in pass_ops]
        if record_if_varying(field_name, values):
            numeric = [value for value in values if isinstance(value, (int, float))]
            if numeric and len(numeric) == len(values):
                setattr(layer_log, field_name, quantity_type(int(max(numeric))))

    for field_name in _MULTIPASS_MARKER_ONLY_FIELDS:
        record_if_varying(field_name, [getattr(op, field_name, None) for op in pass_ops])

    module_stacks = [
        [module_pass[0] for module_pass in (getattr(op, "modules", None) or ())] for op in pass_ops
    ]
    record_if_varying("modules", module_stacks)

    if varying:
        layer_log.annotations["varying_across_passes"] = varying


def _build_layer_logs(self: "Trace") -> None:
    """Step 15.5: Build aggregate Layer objects from per-pass Op entries.

    Groups layer_list entries by layer_label and creates a Layer for
    each unique layer. For single-pass layers, the Layer is a thin wrapper
    delegating attribute access to the sole Op.

    MERGE RULES for multi-pass layers:
    1. has_input_ancestor: OR across ops (True if ANY pass has an input ancestor).
    2. io_role: Character-wise merge with '*' for differing characters
       (e.g., "output.0" + "output.1" -> "output.*").
    3. is_atomic_module: OR across ops.
    4. is_in_conditional_body: OR across ops.
    5. conditional_role_stacks / conditional_branch_stack_ops:
       unique stack signatures in first-seen order plus sorted pass maps.
    6. conditional_arm_children: pass-stripped union across ops with
       insertion-order preservation.

    Derived aggregate views (cond_branch_then/elif/else/start_children) are
    recomputed after the merge from the cond-id-aware primary structures.

    All other 78+ fields use first-pass values only. This is correct because
    same-layer grouping requires same structural position, so module containment,
    function info, param info, etc. are identical across ops.

    Note: output_of_modules/output_of_module_calls are NOT updated during merge
    (same structural position implies same modules).
    """
    from collections import OrderedDict

    from ..data_classes.layer import Layer

    layer_logs = OrderedDict()

    for pass_log in self.layer_list:
        no_call_label = pass_log.layer_label

        if no_call_label not in layer_logs:
            # First pass: create Layer from this pass's data.
            layer_log = Layer(pass_log)
            layer_logs[no_call_label] = layer_log
        else:
            # Subsequent pass: merge only the 3 aggregate fields.
            layer_log = layer_logs[no_call_label]
            # Merge field 1/3: has_input_ancestor (OR across ops).
            if pass_log.has_input_ancestor:
                layer_log.has_input_ancestor = True
            # Merge field 2/3: io_role (char-merge with '*').
            if layer_log.io_role is not None and pass_log.io_role is not None:
                merged = "".join(
                    c if c == s else "*"
                    for c, s in zip_longest(
                        layer_log.io_role,
                        pass_log.io_role,
                        fillvalue="",
                    )
                )
                if merged.endswith("."):
                    merged = merged[:-1]
                if merged.endswith("*"):
                    merged = merged.rstrip("*") + "*"
                layer_log.io_role = merged
            # Merge field 3/3: is_atomic_module (OR across ops).
            if pass_log.is_atomic_module:
                layer_log.is_atomic_module = True

        layer_log.ops[pass_log.pass_index] = pass_log
        layer_log.call_labels.append(pass_log.label)
        _merge_layer_log_conditional_fields(layer_log, pass_log)

    autograd_memory = 0
    has_autograd_saved_value = False
    for layer_log in layer_logs.values():
        for pass_log in layer_log.ops.values():
            linked_labels = []
            for param_log in getattr(pass_log, "_param_logs", []):
                # Tied/shared parameters expose every aliasing address through
                # ``all_addresses`` (the primary is ``param_log.address`` == all_addresses[0]).
                # Emit ``primary -> alias`` for each additional address that shares the
                # underlying tensor storage. (Co-occurrence of distinct tensors in one op
                # is tracked separately by ``co_parent_params`` and is NOT tying.)
                for alias_address in getattr(param_log, "all_addresses", []):
                    if alias_address == param_log.address:
                        continue
                    linked_labels.append(f"{param_log.address} → {alias_address}")
            if linked_labels:
                pass_log.annotations["tied_parameter_notation"] = linked_labels
        pass_autograd_bytes = [
            pass_log.autograd_memory
            for pass_log in layer_log.ops.values()
            if pass_log.autograd_memory is not None
        ]
        pass_autograd_tensor_counts = [
            pass_log.num_autograd_tensors
            for pass_log in layer_log.ops.values()
            if pass_log.num_autograd_tensors is not None
        ]
        if pass_autograd_bytes:
            layer_log.autograd_memory = Bytes(sum(int(value) for value in pass_autograd_bytes))
            layer_log.total_autograd_memory = layer_log.autograd_memory
            layer_log.num_autograd_tensors = sum(pass_autograd_tensor_counts)
            autograd_memory += layer_log.autograd_memory
            has_autograd_saved_value = True
        else:
            layer_log.autograd_memory = None
            layer_log.total_autograd_memory = None
            layer_log.num_autograd_tensors = None
        _reconcile_multipass_layer_fields(layer_log)
        _rebuild_layer_log_conditional_views(layer_log)

    self.total_autograd_memory = Bytes(autograd_memory) if has_autograd_saved_value else None
    self.layer_logs = layer_logs
    _rebuild_conditional_edge_call_indices(self)
    _build_conditional_records(self)


def _build_conditional_records(self: "Trace") -> None:
    """Build the public ConditionalAccessor from conditional postprocess data."""

    from ..data_classes.trace import (
        Conditional,
        ConditionalAccessor,
        ConditionalArm,
        ConditionalRoleRef,
    )

    conditionals: list[Conditional] = []
    event_by_id = {event.id: event for event in self.conditional_records}
    conditional_id_by_event_id = _make_public_conditional_ids(self.conditional_records)
    role_labels_by_cond_arm: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for event in self.conditional_records:
        terminal_bool_label = event.bool_layers[0] if event.bool_layers else str(event.id)
        conditional_id = conditional_id_by_event_id[event.id]
        # Use AST-derived branch_ranges so every static arm is materialized,
        # regardless of which arm fired at runtime. fired-vs-not is captured
        # per-arm via ConditionalArm.fired below.
        ast_branch_kinds = list(event.branch_ranges)
        ordered_branch_kinds = ["then"]
        ordered_branch_kinds.extend(
            sorted(
                (kind for kind in ast_branch_kinds if kind.startswith("elif_")),
                key=lambda kind: int(kind.split("_", 1)[1]),
            )
        )
        if "else" in ast_branch_kinds:
            ordered_branch_kinds.append("else")

        # Per-arm test identity: phase 5c stashes ``_arm_bool_indices``
        # (branch kind -> indices into ``bool_layers`` for bools whose runtime
        # consumption evaluated THAT arm's test), ``_arm_test_structures``
        # (branch kind -> "bare"/"negated"/"compound" bool-value semantics),
        # and ``_bool_layers_raw`` (index-aligned raw labels resolving the
        # exact per-pass op of rolled multi-pass bools). Without the stash
        # (legacy/degraded events) every then/elif arm falls back to the
        # historical whole-event bool list.
        arm_bool_indices = getattr(event, "_arm_bool_indices", None)
        arm_test_structures = getattr(event, "_arm_test_structures", None) or {}
        bool_layers_raw = getattr(event, "_bool_layers_raw", None)

        arms: list[ConditionalArm] = []
        for branch_kind in ordered_branch_kinds:
            kind = "elif" if branch_kind.startswith("elif_") else branch_kind
            edge_list = list(self.conditional_arm_entry_edges.get((event.id, branch_kind), []))
            execution_labels = list(dict.fromkeys(child for _parent, child in edge_list))
            if kind not in {"then", "elif"}:
                evaluation_labels = []
            elif arm_bool_indices is not None:
                # An arm whose test never ran (short-circuited elif) has no
                # witnessed consumption, so its evaluation set is honestly
                # empty and ``condition_evaluated`` below reports False.
                evaluation_labels = [
                    event.bool_layers[bool_index]
                    for bool_index in arm_bool_indices.get(branch_kind, [])
                    if bool_index < len(event.bool_layers)
                ]
            else:
                evaluation_labels = list(event.bool_layers)
            if kind == "then" and evaluation_labels:
                terminal_bool = evaluation_labels[0]
            elif kind == "then" and arm_bool_indices is None and event.bool_layers:
                terminal_bool = terminal_bool_label
            else:
                terminal_bool = None
            bool_value = None
            witnessed_indices = (
                list(arm_bool_indices.get(branch_kind, [])) if arm_bool_indices is not None else []
            )
            if (
                terminal_bool is not None
                and arm_bool_indices is not None
                and (len(witnessed_indices) > 1)
            ):
                # N witnessed evaluations of this arm's test (rolled or
                # unrolled multi-pass loops) cannot be represented by ONE
                # scalar: promoting an arbitrary evaluation's value
                # contradicts ``fired`` whenever the values differ across
                # passes (round-24 condbranch seal, S3). Mirror the
                # compound-test refusal and stay None; per-pass values remain
                # queryable on the bool ops themselves.
                bool_value = None
            elif terminal_bool is not None:
                value_op = _resolve_conditional_bool_op(
                    self, terminal_bool, witnessed_indices, bool_layers_raw
                )
                raw_bool_value = getattr(value_op, "bool_value", None)
                # ``bool_value_at_run`` must never contradict ``fired``: apply
                # the test expression's statically-known polarity, and refuse a
                # single-bool value for compound (``and``/``or``) tests where
                # no one consumed bool determines the outcome.
                test_structure = arm_test_structures.get(branch_kind, "bare")
                if test_structure == "compound":
                    bool_value = None
                elif test_structure == "negated":
                    bool_value = (not raw_bool_value) if raw_bool_value is not None else None
                else:
                    bool_value = raw_bool_value
            arm = ConditionalArm(
                kind=kind,  # type: ignore[arg-type]
                terminal_bool_op_label=terminal_bool,
                bool_value_at_run=bool_value,
                condition_evaluated=bool(evaluation_labels) or kind == "else",
                evaluation_entry_edge=_find_conditional_evaluation_entry_edge(
                    self, evaluation_labels
                )
                if evaluation_labels and kind in {"then", "elif"}
                else None,
                fired=bool(execution_labels),
                execution_entry_edge=edge_list[0] if edge_list else None,
            )
            arm_index = len(arms)
            _merge_conditional_role_labels(
                role_labels_by_cond_arm,
                (conditional_id, arm_index, "evaluation"),
                evaluation_labels,
            )
            _merge_conditional_role_labels(
                role_labels_by_cond_arm,
                (conditional_id, arm_index, "body"),
                execution_labels,
            )
            arms.append(arm)

        fired_arm_indices = [index for index, arm in enumerate(arms) if arm.fired]
        fired_arm_index = fired_arm_indices[0] if len(fired_arm_indices) == 1 else None
        fired_arm_kind = arms[fired_arm_index].kind if fired_arm_index is not None else None
        conditional = Conditional(
            id=conditional_id,
            arms=arms,
            fired_arm_index=fired_arm_index,
            fired_arm_kind=fired_arm_kind,
            source_file=event.source_file,
            source_line=event.if_stmt_span[0] if event.if_stmt_span else None,
        )
        for arm_index, arm in enumerate(conditional.arms):
            arm._bind(self, conditional.id, arm_index)
        conditionals.append(conditional)

    # Inverted role index (r8 R60-7): the per-layer triple loop with an
    # innermost list-membership check was O(N_layers x Sum|role lists|)
    # whenever a conditional's entry-edge children were numerous (an ``if``
    # in a hot unrolled loop). One pass over the role buckets builds the
    # same per-layer role lists in the same (conditional, arm,
    # evaluation-then-body) order; per-bucket label dedup mirrors the old
    # one-role-per-membership semantics.
    roles_by_label: dict[str, list[ConditionalRoleRef]] = {}
    for conditional in conditionals:
        for arm_index, arm in enumerate(conditional.arms):
            for role in ("evaluation", "body"):
                bucket = role_labels_by_cond_arm.get((conditional.id, arm_index, role), [])
                bucket_seen: set[str] = set()
                for label in bucket:
                    if label in bucket_seen:
                        continue
                    bucket_seen.add(label)
                    roles_by_label.setdefault(label, []).append(
                        ConditionalRoleRef(conditional.id, arm_index, arm.kind, role)
                    )
    for layer in self.layer_list:
        layer.in_conditionals = roles_by_label.get(layer.layer_label, [])
        if (
            layer.terminal_conditional_id is not None
            and layer.terminal_conditional_id in event_by_id
        ):
            layer.terminal_bool_for = (
                conditional_id_by_event_id[layer.terminal_conditional_id],
                0,
            )
        else:
            layer.terminal_bool_for = None

    self.conditionals = ConditionalAccessor(conditionals)


def _make_public_conditional_ids(condition_events: list[Any]) -> dict[int, str]:
    """Build unique public ids for conditional records.

    Parameters
    ----------
    condition_events:
        Conditional events in trace order.

    Returns
    -------
    dict[int, str]
        Mapping from dense ``ConditionalEvent.id`` to the public conditional id
        exposed through ``Trace.conditionals`` and role refs.
    """

    id_counts: dict[str, int] = defaultdict(int)
    base_ids_by_event_id: dict[int, str] = {}
    for event in condition_events:
        terminal_bool_label = event.bool_layers[0] if event.bool_layers else str(event.id)
        base_id = f"cond_{terminal_bool_label}"
        base_ids_by_event_id[event.id] = base_id
        id_counts[base_id] += 1

    return {
        event.id: (
            base_ids_by_event_id[event.id]
            if id_counts[base_ids_by_event_id[event.id]] == 1
            else f"{base_ids_by_event_id[event.id]}__event_{event.id}"
        )
        for event in condition_events
    }


def _resolve_conditional_bool_op(
    self: "Trace",
    terminal_bool: str,
    witnessed_indices: list[int],
    bool_layers_raw: list[str] | None,
) -> Any | None:
    """Resolve the op whose ``bool_value`` backs one arm's single evaluation.

    Parameters
    ----------
    self:
        Trace whose lookup keys resolve conditional bool ops.
    terminal_bool:
        Renamed public label of the arm's terminal bool layer.
    witnessed_indices:
        Indices into the owning event's ``bool_layers`` witnessed for this
        arm's test. Only single-witness arms reach this resolver.
    bool_layers_raw:
        Index-aligned RAW labels stashed by phase 5c, or ``None`` for
        legacy/degraded events.

    Returns
    -------
    Any | None
        The exact evaluating op. The raw label is preferred because it is
        unique per pass: a rolled multi-pass bool layer renames every pass to
        ONE base label, and an unqualified ``layer_dict_all_keys`` lookup
        resolves last-writer-wins to an arbitrary pass (round-24 condbranch
        seal, S3). Falls back to the renamed label for legacy events.
    """

    if (
        bool_layers_raw is not None
        and len(witnessed_indices) == 1
        and 0 <= witnessed_indices[0] < len(bool_layers_raw)
    ):
        raw_op = self.layer_dict_all_keys.get(bool_layers_raw[witnessed_indices[0]])
        if raw_op is not None:
            return raw_op
    return self.layer_dict_all_keys.get(terminal_bool)


def _merge_conditional_role_labels(
    role_labels_by_cond_arm: dict[tuple[str, int, str], list[str]],
    role_key: tuple[str, int, str],
    labels: list[str],
) -> None:
    """Merge one conditional-role label list without dropping prior members.

    Parameters
    ----------
    role_labels_by_cond_arm:
        Mapping from ``(conditional_id, arm_index, role)`` to layer labels.
    role_key:
        Key naming the arm-role bucket to update.
    labels:
        Labels to merge into the bucket.
    """

    merged = dict.fromkeys([*role_labels_by_cond_arm.get(role_key, []), *labels])
    role_labels_by_cond_arm[role_key] = list(merged)


def _find_conditional_evaluation_entry_edge(
    self: "Trace",
    bool_layers: list[str],
) -> tuple[str, str] | None:
    """Return the main-graph entry edge for a conditional evaluation chain.

    Parameters
    ----------
    self:
        Trace whose finalized layer graph contains the conditional evaluation.
    bool_layers:
        Bool-producing layer labels for one conditional event, in evaluation order.

    Returns
    -------
    tuple[str, str] | None
        Edge from the upstream main-graph parent into the first bool-producing
        layer, or ``None`` when no distinct upstream parent is available.
    """

    if not bool_layers:
        return None

    first_bool_label = bool_layers[0]
    if first_bool_label not in self.layer_dict_all_keys:
        return None

    visited_labels: set[str] = set()
    parent_queue: deque[str] = deque(self.layer_dict_all_keys[first_bool_label].parents)
    while parent_queue:
        parent_label = parent_queue.popleft()
        if parent_label in visited_labels:
            continue
        visited_labels.add(parent_label)
        if parent_label not in self.layer_dict_all_keys:
            continue

        parent_layer = self.layer_dict_all_keys[parent_label]
        if parent_label != first_bool_label and parent_layer.has_output_descendant:
            return (parent_label, first_bool_label)
        parent_queue.extend(parent_layer.parents)

    return None


def _set_tracing_finished(self: "Trace") -> None:
    """Step 17: Mark the Trace and all OpLogs as pass-finished.

    Sets ``_tracing_finished = True`` on the Trace and every retained
    Op entry. This flag switches various custom_methods (e.g., __str__,
    __getitem__) from their "realtime debugging" behavior to their
    "user-facing" behavior.

    Note: _tracing_finished is NOT reset between exhaustive and fast ops.
    This is intentional — it enables fast-path postprocessing to use
    fully-populated lookup dicts from the exhaustive pass.
    """
    for layer_label in self.layer_dict_main_keys:
        tensor = self.layer_dict_main_keys[layer_label]
        tensor._tracing_finished = True
    self._tracing_finished = True


def _finalize_streamed_bundle(self: "Trace") -> None:
    """Step 18: Finalize any in-progress streamed tensor bundle.

    Streamed bundles ENTER SETTLEMENT (WT1 A-IV item 18, lane A08): an
    UNSETTLED trace (the ordinary step-18 call, which runs before
    ``settle_completed``) only STAGES the bundle -- blobs and manifest land in
    the temp directory and the writer stays on ``self._out_writer`` -- and the
    tmp->final publish happens at the settlement seam in
    ``capture/trace.py``, carrying the settled capture-outcome attestation.
    A SETTLED trace (the deferred grad-streaming tail in
    ``backends/torch/backward.py``) stages and publishes in one pass.
    Historical behavior published here unconditionally, so a live COMPLETE
    capture loaded UNATTESTED (no attestation existed yet to persist) and a
    postprocess failure AFTER this step left a published artifact the N1
    export gate would refuse.

    Parameters
    ----------
    self:
        Model log whose streamed bundle should be finalized.

    Raises
    ------
    TorchLensIOError
        If portable scrubbing or bundle staging fails.
    """

    writer = self._out_writer
    if writer is None:
        return

    from .._io.scrub import scrub_for_save
    from ..capture.outcome import outcome_for
    from ..intervention.injection import refuse_injection_logged_stream_finalize

    # F44: the streamed writer stages its own bundle and never passes the
    # analysis save door that appends the injected family; refusing here
    # keeps a streamed artifact from silently dropping recorded computation.
    refuse_injection_logged_stream_finalize(self)
    try:
        scrubbed_state, blob_specs, unsupported_tensor_records = scrub_for_save(
            self,
            include_outs=True,
            include_grads=_has_retained_gradient_payloads(self),
            include_saved_args=self.save_arg_values,
            include_rng_states=self.save_rng_states,
            include_custom_attributes=getattr(writer, "include_custom_attributes", True),
            include_buffer_values=getattr(writer, "include_buffer_values", True),
        )
        scrubbed_state, blob_specs = _reuse_streamed_blob_ids(
            self,
            scrubbed_state=scrubbed_state,
            blob_specs=blob_specs,
            writer=writer,
        )
        final_path = writer.stage_for_settlement(
            scrubbed_state=scrubbed_state,
            blob_specs=blob_specs,
            unsupported=unsupported_tensor_records,
            trace=self,
        )
        settled = outcome_for(self)
        if settled is not None:
            writer.publish_staged(settled.to_payload())
    except BaseException as exc:
        if not getattr(writer, "_closed", False):
            writer.abort(str(exc))
        raise
    setattr(self, "_source_bundle_path", Path(final_path))
    setattr(
        self,
        "_source_bundle_manifest_sha256",
        sha256_of_file(
            Path(writer.tmp_path if not writer._finalized else final_path) / "manifest.json"
        ),
    )
    _attach_streamed_tensor_refs(
        self, scrubbed_state=scrubbed_state, writer=writer, final_path=final_path
    )
    if writer._finalized:
        self._out_writer = None
    self._defer_streaming_bundle_finalization = False


def _has_retained_gradient_payloads(self: "Trace") -> bool:
    """Return whether any completed backward pass retained gradient payloads.

    Parameters
    ----------
    self:
        Trace whose projected operation and parameter gradients should be
        inspected.

    Returns
    -------
    bool
        Whether portable scrubbing must include gradient payloads.

    Notes
    -----
    The projection is the authority here, rather than ``Trace.save_grads``:
    ``log_backward(save_grads=...)`` may override that capture-time default on
    each pass.
    """

    for op in getattr(self, "layer_list", ()):
        for record in getattr(op, "grads", ()):
            if record.grad is not None or record.transformed_grad is not None:
                return True
    for param in getattr(self, "param_logs", {}).values():
        if any(record.grad is not None for record in getattr(param, "grads", ())):
            return True
    return False


def _evict_streamed_outs(self: "Trace") -> None:
    """Step 19: Drop in-memory outs once streaming refs have been attached.

    Parameters
    ----------
    self:
        Model log whose streamed outs should be evicted.
    """

    for layer_entry in self.layer_list:
        if getattr(layer_entry, "out_ref", None) is not None:
            layer_entry._internal_set("out", None)
        if getattr(layer_entry, "_pending_transformed_out_blob_id", None) is not None:
            layer_entry._internal_set("transformed_out", None)


def _evict_streamed_grads(self: "Trace") -> None:
    """Drop in-memory grads once streaming refs have been attached.

    Parameters
    ----------
    self:
        Model log whose streamed grads should be evicted.
    """

    for layer_entry in self.layer_list:
        if getattr(layer_entry, "grad_ref", None) is not None:
            layer_entry._internal_set("grad", None)
        if getattr(layer_entry, "_pending_transformed_grad_blob_id", None) is not None:
            layer_entry._internal_set("transformed_grad", None)


def _reuse_streamed_blob_ids(
    trace: "Trace",
    *,
    scrubbed_state: dict[str, Any],
    blob_specs: list[BlobSpec],
    writer: BundleStreamWriter,
) -> tuple[dict[str, Any], list[BlobSpec]]:
    """Patch scrubbed tensor refs to reuse blob ids written during capture.

    Parameters
    ----------
    trace:
        Live model log containing pending streamed blob ids.
    scrubbed_state:
        Scrubbed portable metadata state produced by ``scrub_for_save``.
    blob_specs:
        Scrub-generated blob specs.
    writer:
        Streaming writer holding already-written out entries.

    Returns
    -------
    tuple[dict, list[BlobSpec]]
        Patched scrubbed state and the filtered blob spec list.
    """

    scrubbed_layers = scrubbed_state.get("layer_list")
    if not isinstance(scrubbed_layers, list):
        raise TorchLensIOError(
            "Streaming finalize expected scrubbed_state['layer_list'] to be a list."
        )
    # The scrubbed layers are a 1:1 scrub of trace.layer_list. If they ever
    # diverge, pairing them positionally would silently finalize only the
    # shorter prefix -- writing a bundle that looks complete but is missing
    # blobs for every layer past the truncation point.
    if len(scrubbed_layers) != len(trace.layer_list):
        raise TorchLensIOError(
            "Streaming finalize expected one scrubbed layer per live layer, got "
            f"{len(scrubbed_layers)} scrubbed vs {len(trace.layer_list)} live."
        )

    skipped_blob_ids: set[str] = set()
    for live_layer, scrubbed_layer in zip(trace.layer_list, scrubbed_layers, strict=True):
        # One lazy full-schema snapshot per layer (r8 R29): the old loop
        # rebuilt the ~200-300-field portable state dict on EVERY field
        # iteration and BEFORE the pending short-circuit, so every layer of
        # every streamed save paid 4 full dict builds to read one key. Each
        # field key is read once and the four keys are distinct, so the
        # snapshot cannot go stale across this layer's own setattrs.
        scrubbed_items: dict[str, Any] | None = None
        for tensor_field, pending_field in (
            ("out", "_pending_blob_id"),
            ("transformed_out", "_pending_transformed_out_blob_id"),
            ("grad", "_pending_grad_blob_id"),
            ("transformed_grad", "_pending_transformed_grad_blob_id"),
        ):
            pending_blob_id = getattr(live_layer, pending_field, None)
            if pending_blob_id is None:
                continue
            if scrubbed_items is None:
                scrubbed_items = dict(state_items(scrubbed_layer))
            tensor_blob = scrubbed_items.get(tensor_field)
            if not isinstance(tensor_blob, BlobRef):
                setattr(
                    scrubbed_layer,
                    tensor_field,
                    BlobRef(blob_id=pending_blob_id, kind=tensor_field),
                )
                writer.relabel_blob(pending_blob_id, live_layer._streaming_label)
                continue
            skipped_blob_ids.add(tensor_blob.blob_id)
            setattr(
                scrubbed_layer,
                tensor_field,
                BlobRef(blob_id=pending_blob_id, kind=tensor_blob.kind),
            )
            writer.relabel_blob(pending_blob_id, live_layer._streaming_label)
            _reuse_streamed_grad_record_blob_ids(
                scrubbed_layer,
                tensor_field=tensor_field,
                pending_blob_id=pending_blob_id,
                skipped_blob_ids=skipped_blob_ids,
            )

    filtered_blob_specs = [spec for spec in blob_specs if spec.blob_id not in skipped_blob_ids]
    return scrubbed_state, filtered_blob_specs


def _reuse_streamed_grad_record_blob_ids(
    scrubbed_layer: Any,
    *,
    tensor_field: str,
    pending_blob_id: str,
    skipped_blob_ids: set[str],
) -> None:
    """Rewrite per-pass gradient record blob ids to the streamed direct blob id.

    Parameters
    ----------
    scrubbed_layer:
        Scrubbed Op whose gradient records should be patched.
    tensor_field:
        Direct tensor field being reused.
    pending_blob_id:
        Blob id already written by the streaming writer.
    skipped_blob_ids:
        Blob ids that should be filtered from scrub-generated blob specs.
    """

    record_field = {
        "grad": "grad",
        "transformed_grad": "transformed_grad",
    }.get(tensor_field)
    if record_field is None:
        return
    for grad_record in getattr(scrubbed_layer, "_grad_records", ()):
        record_blob = dict(state_items(grad_record)).get(record_field)
        if not isinstance(record_blob, BlobRef):
            continue
        skipped_blob_ids.add(record_blob.blob_id)
        setattr(
            grad_record,
            record_field,
            BlobRef(blob_id=pending_blob_id, kind=record_blob.kind),
        )


def _attach_streamed_tensor_refs(
    trace: "Trace",
    *,
    scrubbed_state: dict[str, Any],
    writer: BundleStreamWriter,
    final_path: str | Path,
) -> None:
    """Attach ``LazyActivationRef`` placeholders for persisted direct tensor blobs.

    Parameters
    ----------
    trace:
        Live model log receiving the refs.
    scrubbed_state:
        Scrubbed metadata state whose out ``BlobRef`` values now match the
        final persisted blob ids.
    writer:
        Streaming writer containing manifest entries for all saved blobs.
    final_path:
        Final bundle directory path after the temp-dir rename.
    """

    scrubbed_layers = scrubbed_state.get("layer_list")
    if not isinstance(scrubbed_layers, list):
        raise TorchLensIOError(
            "Streaming finalize expected scrubbed_state['layer_list'] to be a list."
        )
    # The scrubbed layers are a 1:1 scrub of trace.layer_list. If they ever
    # diverge, pairing them positionally would silently finalize only the
    # shorter prefix -- writing a bundle that looks complete but is missing
    # blobs for every layer past the truncation point.
    if len(scrubbed_layers) != len(trace.layer_list):
        raise TorchLensIOError(
            "Streaming finalize expected one scrubbed layer per live layer, got "
            f"{len(scrubbed_layers)} scrubbed vs {len(trace.layer_list)} live."
        )

    for live_layer, scrubbed_layer in zip(trace.layer_list, scrubbed_layers, strict=True):
        for tensor_field, ref_field, kind in (
            ("out", "out_ref", "out"),
            ("grad", "grad_ref", "grad"),
        ):
            tensor_blob = getattr(scrubbed_layer, tensor_field, None)
            if not isinstance(tensor_blob, BlobRef):
                continue
            manifest_entry = writer.get_entry(tensor_blob.blob_id)
            setattr(
                live_layer,
                ref_field,
                LazyActivationRef(
                    blob_id=manifest_entry.blob_id,
                    shape=tuple(manifest_entry.shape),
                    dtype=_dtype_from_manifest_string(manifest_entry.dtype),
                    device_at_save=manifest_entry.device_at_save,
                    source_bundle_path=Path(final_path),
                    relative_path=manifest_entry.relative_path,
                    kind=cast(Literal["out", "grad"], kind),
                    expected_sha256=manifest_entry.sha256,
                ),
            )


def _dtype_from_manifest_string(dtype_name: str) -> torch.dtype:
    """Resolve a manifest dtype string back into a ``torch.dtype``.

    Parameters
    ----------
    dtype_name:
        Manifest dtype name without the ``torch.`` prefix.

    Returns
    -------
    torch.dtype
        Resolved dtype object.

    Raises
    ------
    TorchLensIOError
        If the dtype name is unknown to the current runtime.
    """

    dtype_obj = torch_attr(dtype_name)  # r47 secD_1: no lazy ``torch.__getattr__``
    if not isinstance(dtype_obj, torch.dtype):
        raise TorchLensIOError(f"Unsupported dtype string in manifest: {dtype_name}.")
    return dtype_obj
