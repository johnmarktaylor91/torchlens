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
Step 20 (_release_param_refs): Drops live parameter references after finalization.
"""

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import torch

from .._capture.outcome import outcome_for
from .._data_classes._module_facade_fields import summarize_call_args
from .._data_classes._op_column_scan import op_labels
from .._data_classes._state_adapter import state_items
from .._data_classes.layer import Layer
from .._data_classes.module import Module, ModuleAccessor, ModuleCall, set_attribute_writes
from .._data_classes.param import ParamAccessor, param_key
from .._data_classes.trace import (
    Conditional,
    ConditionalAccessor,
    ConditionalArm,
    ConditionalRoleRef,
)
from .._io import ArtifactError, BlobRef
from .._io.accessor_rebuild import rebuild_trace_accessors
from .._io.format3.writer import settle_live_side
from .._io.lazy import LazyActivationRef
from .._io.scrub import BlobSpec, scrub_for_save
from .._io.streaming import BundleStreamWriter
from .._ir.container_registry import _iter_tensor_leaves
from .._quantities import Bytes, Duration, Flops
from .._utils._record_group_table import layer_attr, op_attr, record_member, record_storage_attr
from .._utils._tensor_label import clear_meta
from .._utils.hashing import sha256_file
from . import _column_passes_finalize
from ._conditional_records import (
    _find_conditional_evaluation_entry_edge,
    _make_public_conditional_ids,
    _merge_conditional_role_labels,
    _ordered_branch_kinds,
    _resolve_conditional_bool_op,
    _roles_by_label,
    rebind_op_conditional_roles,
)
from ._finished_marks import mark_ops_finished
from ._layer_conditional_merge import _rebuild_conditional_edge_call_indices
from ._module_census import (
    _call_io_layers,
    bind_layer_conditional_roles,
    module_source_and_hooks,
    op_space_labels,
    param_totals,
)
from ._module_pending import PendingModules, ensure_modules_built, seal_module_facts
from ._multipass_params import pass_param_addresses, union_multipass_params
from ._pre_hook_provenance import _pre_hook_provenance_for_call
from ._root_module_log import (
    _recorded_root_calls,
    _root_address_children,
    _root_call_build_inputs,
    _root_call_children,
)
from ._streamed_tensor_types import _dtype_from_manifest_string, _streamed_entry
from .saved_summary import _has_retained_gradient_payloads

if TYPE_CHECKING:
    from .._data_classes.layer import Layer
    from .._data_classes.trace import Trace


def _undecorate_all_saved_tensors(self: "Trace") -> None:
    """Step 12: Remove TorchLens metadata from all saved tensors.

    During logging, tensors are tagged with private ``_tl`` metadata for
    tracking. This function strips that metadata from saved outs and any
    tensors embedded in saved_args/saved_kwargs, so tensors returned to the
    user are clean.
    """
    tensors_to_undecorate = []
    for layer_entry in [*self._layer_list, *self.graph.orphans.values()]:
        if layer_entry.has_saved_activation and layer_entry.out is not None:
            tensors_to_undecorate.append(layer_entry.out)
        if record_member(layer_entry, "transformed_out", None) is not None:
            tensors_to_undecorate.append(layer_entry._payload_transformed_out)

        if layer_entry._call_saved_args:
            tensors_to_undecorate.extend(_iter_tensor_leaves(layer_entry._call_saved_args))
        if layer_entry._call_saved_kwargs:
            tensors_to_undecorate.extend(_iter_tensor_leaves(layer_entry._call_saved_kwargs))

    for t in tensors_to_undecorate:
        clear_meta(t)


def _log_time_elapsed(self: "Trace") -> None:
    """Step 14: Record wall-clock timing for the cleanup phase and overall pass.

    Computes cleanup time as the residual after subtracting setup and forward
    pass times from total elapsed time. Also computes torchlens_logging overhead
    as total time minus actual function call time.
    """
    self._provenance_end_time = end_time = time.time()
    elapsed = end_time - self._provenance_start_time
    self._provenance_cleanup_duration = Duration(
        elapsed - self._provenance_setup_duration - self._cost_forward_duration
    )


def _finalize_param_logs(self: "Trace") -> None:
    """Step 15: Populate Param reverse mappings, co-parent params, and num_calls.

    For each Op with _param_logs:
    - Adds the Op label to each Param's used_by_ops list.
    - Adds the Layer label to each Param's used_by_layers list.
    - Links params that co-occur in the same operation (co_parent_params).
    - Sets num_calls = len(used_by_ops) (0 for a parameter no op consumed).

    Then clears actual Parameter tensor references (parent_params) from
    Op entries to reduce memory. Param._param_ref is released after the
    full finalization pipeline, once all finalization-time param reads finish.
    """
    from ._lazy_param_geometry import _finalize_lazy_param_geometry

    _finalize_lazy_param_geometry(self)

    # Lists remain authoritative and preserve first-seen order. Local sets avoid
    # repeatedly scanning those growing lists for high-arity parameterized ops.
    membership_by_param: dict[Any, tuple[set[str], set[str], set[str]]] = {}

    # Build used_by_ops, used_by_layers, and co_parent_params from Op entries
    for layer_entry in self._layer_list:
        if not layer_entry._param_logs:
            continue
        addresses_in_op = [pl.address for pl in layer_entry._param_logs]
        for pl in layer_entry._param_logs:
            membership = membership_by_param.get(param_key(pl))
            if membership is None:
                membership = (
                    set(pl.used_by_ops),
                    set(pl.used_by_layers),
                    set(pl.co_parent_params),
                )
                membership_by_param[param_key(pl)] = membership
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
    for pl in self._param_logs:
        pl.num_calls = len(pl.used_by_ops)
        pl.trace = self

    # Adopt the finished Param rows into the per-trace kind table: each
    # record's detached single row moves into ONE shared store owned by the
    # trace core; facade behavior is unchanged.
    _core = self.__dict__.get("_trace_core")
    if _core is not None:
        from .._trace_core.record_rows import adopt_records

        adopt_records(_core, "param", self._param_logs)

    # Param grad metadata is populated lazily via backward hooks in _log_tensor_grad.
    # Param._param_ref is intentionally released later, after module finalization has
    # consumed param metadata; Param re-fetches from Trace._source_model_ref on demand.
    # _param_logs and func are still needed by validation.

    # Clear actual Parameter tensor references from Op entries to save memory.
    for layer_entry in self._layer_list:
        layer_entry._graph_parent_params = []


def _build_root_module_log(
    self: "Trace", pass_dict: dict[str, "ModuleCall"], mbd: dict[str, Any]
) -> "Module":
    """Build the root Module ("self") representing the model itself.

    The root encompasses all layers and params; address_children are the direct
    children (no dots), call_children may include grandchildren called directly.
    Its calls: one synthesized ``self:1`` spanning the whole capture when the
    root's forward ran undecorated (a plain module root), else the recorded
    ``self:1`` .. ``self:N`` of a bound-method root called N times.
    """

    module_metadata = cast(dict[str, dict[str, Any]], self._module_capture_ws.module_metadata)
    root_meta = module_metadata.get("self", {})
    root_layers = list(self._layer_logs.keys())
    recorded_calls = _recorded_root_calls(mbd)

    root_param_dict = {pl.address: pl for pl in self._param_logs}
    root_params = ParamAccessor(root_param_dict)
    root_num_params, root_num_trainable, root_num_frozen, root_memory = param_totals(
        self._param_logs
    )
    root_fsize = Bytes(root_memory)
    address_children = _root_address_children(root_meta, mbd)

    root_ops: dict[int, ModuleCall] = {}
    # The labels are the calls' own string objects (the synthesized call's literal below), so
    # the saved metadata pickles each label once and refers back to it.
    root_call_labels = ["self:1"]
    if recorded_calls:
        child_to_parent, input_layers_by_call = _root_call_build_inputs(self, mbd)
        root_ops, root_call_labels = _build_submodule_call_logs(
            self,
            "self",
            recorded_calls,
            pass_dict,
            mbd,
            child_to_parent,
            all_addresses=list(root_meta.get("all_addresses", ["self"])),
            module_class=root_meta.get("cls"),
            pass_input_layers_by_call=input_layers_by_call,
        )
        for root_call in root_ops.values():
            root_call.call_parent = None  # the root's own calls have no calling module

    root_module = Module(
        address="self",
        all_addresses=root_meta.get("all_addresses", ["self"]),
        name="self",
        cls=root_meta.get("cls"),
        class_name=root_meta.get("class_name", self._source_class_name),
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
        num_calls=max(1, recorded_calls),
        ops=root_ops,
        call_labels=root_call_labels,
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

    if not recorded_calls:
        root_pass = _synthesized_root_call(self, mbd, root_meta)
        root_module.ops[1] = root_pass  # int keys read positionally: keep the object
        pass_dict["self:1"] = root_pass
    root_module._sync_boundary_fields_from_calls()
    return root_module


def _synthesized_root_call(
    self: "Trace", mbd: dict[str, Any], root_meta: dict[str, Any]
) -> ModuleCall:
    """The synthesized ``self:1`` spanning the whole capture of a plain module root."""

    root_pre_hook_provenance = _pre_hook_provenance_for_call(self, "self:1")
    root_args, root_kwargs = self._module_capture_ws.root_forward_args or (None, None)
    return ModuleCall(
        address="self",
        call_index=1,
        label="self:1",
        ops=op_labels(self),  # Op space, as on every other call
        input_layers=_layer_space_labels(self, self.input_layers),
        output_layers=_layer_space_labels(self, self.output_layers),
        output_ops=op_space_labels(self, self.output_layers),
        output_structure=_first_output_structure(self, list(self.output_layers)),
        forward_args=root_args,
        forward_kwargs=root_kwargs,
        inputs_before_pre_hooks=root_pre_hook_provenance[0],
        inputs_after_pre_hooks=root_pre_hook_provenance[1],
        forward_pre_hook_effects=root_pre_hook_provenance[2],
        call_parent=None,
        call_children=_root_call_children(mbd),
        all_addresses=root_meta.get("all_addresses", ["self"]),
        cls=root_meta.get("cls"),
        class_name=root_meta.get("class_name", self._source_class_name),
        class_qualname=root_meta.get("class_qualname", ""),
        ordinal_index=0,
        forward_duration=float(self._cost_forward_duration or 0.0),
        code_context=list(self._source_frames),
        module_call_stack=[],
        _source_trace=self,
    )


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
        op = self._layer_dict_all_keys.get(label)
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
        entry = self._layer_dict_all_keys.get(layer_label)
        if entry is not None and entry.container_spec is not None:
            return entry.container_spec
    return None


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
    outputs_by_call: dict[str, dict[str, str]] | None = None,
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

        # Derive input/output layers per pass from the trace-wide rosters
        pass_input_layers, pass_output_layers, pass_output_ops = _call_io_layers(
            call_label, pass_layers, pass_input_layers_by_call or {}, outputs_by_call or {}
        )

        # Forward args for this pass
        fwd_args = self._module_capture_ws.module_forward_args.get((address, call_index))
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
            label=call_label,
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
        first_parent = next(iter(ops.values())).call_parent
        if first_parent and first_parent != "self:1":
            call_parent_addr = first_parent.rsplit(":", 1)[0]
        elif first_parent == "self:1":
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
            if bl in self._layer_dict_all_keys
            and hasattr(self._layer_dict_all_keys[bl], "buffer_address")
            and self._layer_dict_all_keys[bl].buffer_address is not None
            and self._layer_dict_all_keys[bl].buffer_address.rsplit(".", 1)[0] == address
        ]

    return ModuleParamInfo(
        module_params, m_num_params, m_num_trainable, m_num_frozen, m_fsize, module_buffer_layers
    )


def _seal_module_logs(self: "Trace") -> None:
    """Step 16: seal the module facts; the records are built on first lookup."""

    seal_module_facts(self, _build_pending_module_logs)


def _build_module_logs_now(self: "Trace") -> None:
    """Seal and build at once: other backends do not stub call arguments at entry."""

    seal_module_facts(self, _build_pending_module_logs)
    ensure_modules_built(self)


def _build_pending_module_logs(self: "Trace", pending: PendingModules) -> None:
    """Build the records from the sealed facts, then attach the forward's attribute writes."""

    # Build as step 16 did, before step 20 released the live parameter
    # references: iterating the params must not restore them from the model.
    params: Any = self.__dict__.get("_param_logs")
    rehydrate = getattr(params, "_rehydrate_on_iter", False)
    if rehydrate:
        params._rehydrate_on_iter = False
    try:
        _build_module_logs(self, pending)
    finally:
        if rehydrate:
            params._rehydrate_on_iter = rehydrate
    records = self.modules  # the door the eager capture wrote through
    for address, writes in (pending.attribute_writes or {}).items():
        if writes is not None and address in records:
            set_attribute_writes(records[address], writes)


def _build_module_logs(self: "Trace", pending: PendingModules) -> None:
    """Build structured Module/ModuleCall objects from a pending trace's sealed facts.

    Constructs the module hierarchy from _module_build_data (populated in Step 9)
    and _module_metadata (captured during model preparation). Creates:
    - A root Module for "self" (the model itself).
    - ModuleLogs for each submodule with ModuleCallLogs for each pass.
    - ModuleAccessor and BufferAccessor for user-facing access.

    Handles shared modules (same nn.Module registered under multiple addresses)
    via an alias-to-metadata map. Computes nesting depths via BFS from root.
    Runs inside ``PendingModules.ensure``, which installs the sealed facts
    as the module workspace, so the builders read what step 16 captured.
    """
    self.__dict__.update(_module_logs=ModuleAccessor({}), _buffer_accessor=None)  # read mid-build
    mbd = self._module_capture_ws.module_build_data
    module_dict = {}  # address -> Module
    pass_dict: dict[str, ModuleCall] = {}  # "addr:pass" -> ModuleCall
    module_order = []  # ordered by first appearance

    root_module = _build_root_module_log(self, pass_dict, mbd)
    module_dict["self"] = root_module
    # Pre-compute reverse mapping: child_call_label -> parent_call_label
    _child_to_parent_pass = {}
    for parent_call_label, children in mbd["module_pass_children"].items():
        for child_label in children:
            _child_to_parent_pass[child_label] = parent_call_label

    # Pre-compute input layers for each module call.  Large traces may have
    # tens of thousands of ModuleCalls; scanning every Op once per call is
    # quadratic and dominates validation postprocessing.
    _pass_input_layers_by_call = pending.input_layers_by_call  # read at the seal
    _outputs_by_call = pending.outputs_by_call

    # --- Build root Module ("self") ---
    module_order.append(root_module)

    # Pre-compute param_logs grouped by module address
    self._param_logs_by_module = defaultdict(list)
    for pl in self._param_logs:
        for module_address in pl.all_module_addresses:
            self._param_logs_by_module[module_address].append(pl)

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
        if bl in self._layer_dict_all_keys:
            bl_entry = self._layer_dict_all_keys[bl]
            if hasattr(bl_entry, "buffer_address") and bl_entry.buffer_address is not None:
                module_addr = bl_entry.buffer_address.rsplit(".", 1)[0]
                _buffer_layers_by_module[module_addr].append(bl)

    # --- Build ModuleLogs for each submodule ---
    _all_keys = self._layer_dict_all_keys
    _layer_label_of: dict[str, str] = {}
    for address in mbd["addresses"]:
        if address == "self":
            continue
        meta = _metadata_by_alias.get(address, {})
        num_calls = mbd["module_num_calls"].get(address, 1)

        name = address.rsplit(".", 1)[-1] if "." in address else address
        address_parent = address.rsplit(".", 1)[0] if "." in address else "self"
        address_depth = address.count(".") + 1
        layers_seen: dict[str, None] = {}  # insertion-ordered, deduplicated
        for label in mbd["module_layers"].get(address, []):
            if label not in _layer_label_of:  # nested modules list the same ops again
                entry = _all_keys.get(label)
                if entry is None:
                    continue
                _layer_label_of[label] = entry.layer_label
            layers_seen[_layer_label_of[label]] = None
        layers = list(layers_seen)

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
            outputs_by_call=_outputs_by_call,
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
            **module_source_and_hooks(meta),
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
            custom_attributes=meta.get("custom_attributes", {}),
            custom_methods=meta.get("custom_methods", []),
            _source_trace=self,
        )
        module_dict[address] = ml
        module_order.append(ml)

    # --- Compute nesting depths ---
    _compute_call_depths(module_dict, root_module)

    # Summarize the calls' forward args, then drop them (unless the identity mode
    # keeps them) to release tensor references from the model's forward() call args.
    keep_forward_args = self._options_module_identity_mode in {"pytree_module", "object_module"}
    summarize_call_args(pass_dict.values(), keep_forward_args)
    if pending.root_summaries is not None and not _recorded_root_calls(mbd):
        # The root call's live arguments were summarised at the seal.
        root_call = pass_dict["self:1"]
        root_call.forward_args_summary, root_call.forward_kwargs_summary = pending.root_summaries

    # The rebuild adopts the finished Module/ModuleCall rows into the per-trace
    # kind tables and puts the accessors around them.
    rebuild_trace_accessors(
        self,
        module_dict,
        module_order,
        pass_dict,
        sealed_buffer_versions=pending.buffer_versions,
    )


# Aggregate Layer fields that may legitimately DIVERGE across genuine recurrent
# passes (variable-length recurrence, autocast dtype shifts, tied-site module
# attribution) and therefore must never be silently projected from pass 1.
# Fields absent here are structurally uniform per the grouping identity
# (function, call site, output slot); parameters are unioned (_multipass_params).
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


def _reconcile_multipass_layer_fields(layer_log: "Layer", read: Any = getattr) -> None:
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
    read:
        ``getattr``-shaped per-pass field reader (step 15.5 binds column getters).
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
        values = [read(op, field_name, None) for op in pass_ops]
        if record_if_varying(field_name, values):
            setattr(layer_log, layer_attr(field_name), _honest_multipass_shape(values))

    for field_name, quantity_type in (
        *((name, Bytes) for name in _MULTIPASS_BYTES_FIELDS),
        *((name, Flops) for name in _MULTIPASS_FLOPS_FIELDS),
    ):
        values = [read(op, field_name, None) for op in pass_ops]
        if record_if_varying(field_name, values):
            numeric = [value for value in values if isinstance(value, (int, float))]
            if numeric and len(numeric) == len(values):
                setattr(layer_log, layer_attr(field_name), quantity_type(int(max(numeric))))

    for field_name in _MULTIPASS_MARKER_ONLY_FIELDS:
        record_if_varying(field_name, [read(op, field_name, None) for op in pass_ops])

    module_stacks = [[pair[0] for pair in (read(op, "modules", None) or ())] for op in pass_ops]
    record_if_varying("modules", module_stacks)

    if record_if_varying("params", pass_param_addresses(pass_ops)):
        union_multipass_params(layer_log, pass_ops)

    if varying:
        layer_log.annotations["varying_across_passes"] = varying


def _build_layer_logs(self: "Trace") -> None:
    """Step 15.5: group per-pass Op entries into aggregate Layers, built on first lookup.

    Groups layer_list entries by layer_label; each Layer is built the first time
    it is looked up. A single-pass Layer is a thin wrapper over its sole Op.

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

    All other 78+ fields use first-pass values only: passes of one layer share
    the call site, function and output slot; fields that can still differ (shapes,
    memory, modules, parameters) are reconciled by _reconcile_multipass_layer_fields.

    Note: output_of_modules/output_of_module_calls are NOT updated during merge
    (same structural position implies same modules).
    """
    rebound = _column_passes_finalize.build_layer_logs(
        self, Layer, _reconcile_multipass_layer_fields
    )
    _rebuild_conditional_edge_call_indices(self)
    _build_conditional_records(self, rebound)


def _build_conditional_records(self: "Trace", rebound: bool = False) -> None:
    """Build the public ConditionalAccessor (``rebound``: the ops' roles are already reset)."""

    conditionals: list[Conditional] = []
    event_by_id = {event.id: event for event in self._graph_conditional_records}
    conditional_id_by_event_id = _make_public_conditional_ids(self._graph_conditional_records)
    role_labels_by_cond_arm: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    arm_stack_entries: dict[tuple[str, int], tuple[int, str]] = {}
    for event in self._graph_conditional_records:
        terminal_bool_label = event.bool_layers[0] if event.bool_layers else str(event.id)
        conditional_id = conditional_id_by_event_id[event.id]
        # AST-derived branch_ranges materialize every static arm regardless of which
        # fired at runtime; fired-vs-not is captured per-arm via ConditionalArm.fired.
        ordered_branch_kinds = _ordered_branch_kinds(event)

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
            edge_list = list(
                self._graph_conditional_arm_entry_edges.get((event.id, branch_kind), [])
            )
            execution_labels = list(dict.fromkeys(child for _parent, child in edge_list))
            if kind not in {"then", "elif"}:
                evaluation_labels = []
            elif arm_bool_indices is not None:
                # A short-circuited elif has no witnessed consumption: its evaluation
                # set is honestly empty and ``condition_evaluated`` below reports False.
                evaluation_labels = [
                    event.bool_layers[bool_index]
                    for bool_index in arm_bool_indices.get(branch_kind, [])
                    if bool_index < len(event.bool_layers)
                ]
            else:
                evaluation_labels = list(event.bool_layers)
            # A witnessed elif names its bool too (a pure-Python then-test leaves none).
            witnessed_arm = kind == "then" or (kind == "elif" and arm_bool_indices is not None)
            if witnessed_arm and evaluation_labels:
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
                # passes. Mirror the
                # compound-test refusal and stay None; per-pass values remain
                # queryable on the bool ops themselves.
                bool_value = None
            elif terminal_bool is not None:
                value_op = _resolve_conditional_bool_op(
                    self, terminal_bool, witnessed_indices, bool_layers_raw
                )
                raw_bool_value = record_member(value_op, "bool_value", None)
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
                condition_evaluated=bool(evaluation_labels) or kind in {"else", "fallthrough"},
                evaluation_entry_edge=_find_conditional_evaluation_entry_edge(
                    self, evaluation_labels
                )
                if evaluation_labels and kind in {"then", "elif"}
                else None,
                fired=bool(execution_labels),
                execution_entry_edge=edge_list[0] if edge_list else None,
            )
            arm_index = len(arms)
            arm_stack_entries[(conditional_id, arm_index)] = (event.id, branch_kind)
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

    roles_by_label = _roles_by_label(conditionals, role_labels_by_cond_arm, ConditionalRoleRef)
    if not rebound:
        rebind_op_conditional_roles(
            self, roles_by_label, arm_stack_entries, event_by_id, conditional_id_by_event_id
        )
        bind_layer_conditional_roles(self, roles_by_label)

    self._graph_conditionals = ConditionalAccessor(conditionals)


def _set_tracing_finished(self: "Trace") -> None:
    """Step 17: Mark Trace and retained Ops finished and bind display-only keys.

    This switches display and lookup to user-facing behavior. The flag stays
    true between exhaustive and fast passes so fast postprocessing can use the
    exhaustive pass's populated lookup dictionaries.
    """
    mark_ops_finished(self)
    self._tracing_finished = True


def _finalize_streamed_bundle(self: "Trace") -> None:
    """Step 18: Finalize any in-progress streamed tensor bundle.

    Streamed bundles ENTER SETTLEMENT: an UNSETTLED trace (the
    step-18 call, before ``settle_completed``) only STAGES the bundle in the temp
    directory, and the publish happens at the settlement seam in ``_capture/trace.py``
    with the settled capture-outcome attestation. A SETTLED trace (the deferred
    grad-streaming tail in ``backends/torch/backward.py``) stages and publishes in one
    pass. Live records pair with their scrubbed copies so live cells match ``save()``.

    Parameters
    ----------
    self:
        Model log whose streamed bundle should be finalized.

    Raises
    ------
    ArtifactError
        If portable scrubbing or bundle staging fails.
    """

    writer = self._out_writer
    if writer is None:
        return

    from ..intervention._injection import refuse_injection_logged_stream_finalize

    # the streamed writer skips the save door that appends the injected family.
    refuse_injection_logged_stream_finalize(self)
    settle_live_side(self)
    record_pairs: list[tuple[Any, Any]] = []
    try:
        scrubbed_state, blob_specs, unsupported_tensor_records = _scrub_streamed_trace(
            self, writer, record_pairs
        )
        scrubbed_state, blob_specs = _reuse_streamed_blob_ids(
            self, scrubbed_state=scrubbed_state, blob_specs=blob_specs, writer=writer
        )
        writer.hold_live_records(record_pairs)
        final_path = writer.stage_for_settlement(
            scrubbed_state=scrubbed_state,
            blob_specs=blob_specs,
            unsupported=unsupported_tensor_records,
            trace=self,
            payload_policy=_streamed_payload_flags(self, writer),
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
        sha256_file(
            Path(writer.tmp_path if not writer._finalized else final_path) / "manifest.json"
        ),
    )
    _attach_streamed_tensor_refs(
        self, scrubbed_state=scrubbed_state, writer=writer, final_path=final_path
    )
    if writer._finalized:
        self._out_writer = None
    self._defer_streaming_bundle_finalization = False


def _streamed_payload_flags(self: "Trace", writer: BundleStreamWriter) -> dict[str, Any]:
    """Return the include flags a streamed save scrubs with (and records as its policy)."""

    return {
        "include_outs": True,
        "include_grads": _has_retained_gradient_payloads(self),
        "include_saved_args": self._options_save_args,
        "include_rng_states": self._options_save_rng,
        "include_custom_attributes": getattr(writer, "include_custom_attributes", True),
        "include_buffer_values": getattr(writer, "include_buffer_values", True),
    }


def _scrub_streamed_trace(
    self: "Trace", writer: BundleStreamWriter, record_pairs: list[tuple[Any, Any]]
) -> tuple[dict[str, Any], list[BlobSpec], list[dict[str, str]]]:
    """Scrub a streamed trace for staging, numbering its blobs past the stream's ids.

    Parameters
    ----------
    self:
        Model log whose streamed bundle is being finalized.
    writer:
        The capture's streaming bundle writer.
    record_pairs:
        Filled with ``(scrubbed copy, live record)`` pairs for the writer's live cells.

    Returns
    -------
    tuple[dict[str, Any], list[BlobSpec], list[dict[str, str]]]
        Scrubbed state, blob specs and unsupported tensor audit records.
    """

    return scrub_for_save(
        self,
        **_streamed_payload_flags(self, writer),
        record_pairs=record_pairs,
        # Scrub ids start past every streamed id: else a blobless tensor field takes an id
        # a pruned streamed op already wrote, and its ref resolves to that op's payload.
        blob_id_floor=writer._issued_blob_count(),
    )


def _evict_streamed_outs(self: "Trace") -> None:
    """Step 19: Drop in-memory outs once streaming refs have been attached.

    Parameters
    ----------
    self:
        Model log whose streamed outs should be evicted.
    """

    for layer_entry in self._layer_list:
        if record_member(layer_entry, "out_ref", None) is not None:
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

    for layer_entry in self._layer_list:
        if record_member(layer_entry, "grad_ref", None) is not None:
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

    scrubbed_layers = scrubbed_state.get("_layer_list")
    if not isinstance(scrubbed_layers, list):
        raise ArtifactError(
            "Streaming finalize expected scrubbed_state['_layer_list'] to be a list."
        )
    # The scrubbed layers are a 1:1 scrub of trace._layer_list; pairing diverged lists
    # positionally would silently drop the blobs of every layer past the shorter one.
    if len(scrubbed_layers) != len(trace._layer_list):
        raise ArtifactError(
            "Streaming finalize expected one scrubbed layer per live layer, got "
            f"{len(scrubbed_layers)} scrubbed vs {len(trace._layer_list)} live."
        )

    skipped_blob_ids: set[str] = set()
    for live_layer, scrubbed_layer in zip(trace._layer_list, scrubbed_layers, strict=True):
        # One lazy snapshot per layer; the four keys are distinct, so it never goes stale.
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
                    op_attr(tensor_field),
                    BlobRef(blob_id=pending_blob_id, kind=tensor_field),
                )
                writer.relabel_blob(pending_blob_id, live_layer._streaming_label)
                continue
            skipped_blob_ids.add(tensor_blob.blob_id)
            setattr(
                scrubbed_layer,
                op_attr(tensor_field),
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

    scrubbed_layers = scrubbed_state.get("_layer_list")
    if not isinstance(scrubbed_layers, list):
        raise ArtifactError(
            "Streaming finalize expected scrubbed_state['_layer_list'] to be a list."
        )
    # The scrubbed layers are a 1:1 scrub of trace._layer_list; pairing diverged lists
    # positionally would silently drop the blobs of every layer past the shorter one.
    if len(scrubbed_layers) != len(trace._layer_list):
        raise ArtifactError(
            "Streaming finalize expected one scrubbed layer per live layer, got "
            f"{len(scrubbed_layers)} scrubbed vs {len(trace._layer_list)} live."
        )

    for live_layer, scrubbed_layer in zip(trace._layer_list, scrubbed_layers, strict=True):
        for tensor_field, ref_field, kind in (
            ("out", "out_ref", "out"),
            ("grad", "grad_ref", "grad"),
        ):
            tensor_blob = getattr(scrubbed_layer, tensor_field, None)
            if not isinstance(tensor_blob, BlobRef):
                continue
            manifest_entry, key = _streamed_entry(writer, tensor_blob.blob_id)
            setattr(
                live_layer,
                record_storage_attr(live_layer, ref_field),
                LazyActivationRef(
                    blob_id=manifest_entry.blob_id,
                    shape=tuple(manifest_entry.shape),
                    dtype=_dtype_from_manifest_string(manifest_entry.dtype),
                    device_at_save=manifest_entry.device_at_save,
                    source_bundle_path=Path(final_path),
                    relative_path=manifest_entry.relative_path,
                    kind=cast(Literal["out", "grad"], kind),
                    expected_sha256=manifest_entry.sha256,
                    key=key,
                ),
            )
