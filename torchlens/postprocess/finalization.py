"""Steps 13-18: Tensor undecoration, timing, param logs, layer logs, module logs, pass finish.

Step 13 (_undecorate_all_saved_tensors): Removes tl_tensor_label_raw attribute from
    all saved tensors and their creation args/kwargs.
Step 14: torch.cuda.empty_cache() — handled inline in __init__.py.
Step 15 (_log_time_elapsed): Records wall-clock timing for cleanup and overall pass.
Step 16 (_finalize_param_logs): Populates ParamLog reverse mappings (used_by_layers,
    linked_params), num_passes, and clears Parameter tensor references.
Step 16.5 (_build_layer_logs): Groups LayerPassLog entries by layer_label_no_pass to
    create aggregate LayerLog objects. Static identity fields still use first-pass
    values, while aggregate fields merge across passes, including conditional
    branch signatures, pass maps, and pass-stripped child views.
Step 17 (_build_module_logs): Builds structured ModuleLog/ModulePassLog objects from
    _module_build_data and _module_metadata. MUST NOT run in fast mode — _module_build_data
    isn't repopulated in fast mode (Step 10 is skipped). Existing module logs from the
    exhaustive pass remain valid.
Step 18 (_set_pass_finished): Marks ModelLog and all LayerPassLogs as finished, switching
    to user-facing mode for display and access methods.
"""

import time
from collections import defaultdict, deque
from typing import Dict, List, NamedTuple, Optional, TYPE_CHECKING, Tuple

import torch

from .._io.accessor_rebuild import rebuild_model_log_accessors
from ..data_classes._summary import format_call_arg
from ..data_classes.module_log import ModuleLog, ModulePassLog
from ..utils.introspection import get_vars_of_type_from_obj

if TYPE_CHECKING:
    from ..data_classes.layer_log import LayerLog
    from ..data_classes.layer_pass_log import LayerPassLog
    from ..data_classes.model_log import ModelLog


def _undecorate_all_saved_tensors(self) -> None:
    """Step 13: Remove tl_tensor_label_raw from all saved tensors.

    During logging, tensors are "decorated" with a tl_tensor_label_raw attribute
    for tracking. This function strips that attribute from saved activations and
    any tensors embedded in captured_args/captured_kwargs, so that tensors
    returned to the user are clean.
    """
    tensors_to_undecorate = []
    for layer_label in self.layer_labels:
        layer_entry = self.layer_dict_main_keys[layer_label]
        if layer_entry.activation is not None:
            tensors_to_undecorate.append(layer_entry.activation)

        if layer_entry.captured_args:
            tensors_to_undecorate.extend(
                get_vars_of_type_from_obj(layer_entry.captured_args, torch.Tensor, search_depth=2)
            )
        if layer_entry.captured_kwargs:
            tensors_to_undecorate.extend(
                get_vars_of_type_from_obj(layer_entry.captured_kwargs, torch.Tensor, search_depth=2)
            )

    for t in tensors_to_undecorate:
        if hasattr(t, "tl_tensor_label_raw"):
            delattr(t, "tl_tensor_label_raw")


def _log_time_elapsed(self) -> None:
    """Step 15: Record wall-clock timing for the cleanup phase and overall pass.

    Computes cleanup time as the residual after subtracting setup and forward
    pass times from total elapsed time. Also computes torchlens_logging overhead
    as total time minus actual function call time.
    """
    self.pass_end_time = time.time()
    self.time_cleanup = (
        self.pass_end_time - self.pass_start_time - self.time_setup - self.time_forward_pass
    )


def _finalize_param_logs(self: "ModelLog") -> None:
    """Step 16: Populate ParamLog reverse mappings, linked params, and num_passes.

    For each LayerPassLog with parent_param_logs:
    - Adds the layer's label to each ParamLog's used_by_layers list.
    - Links params that co-occur in the same operation (linked_params).
    - Sets num_passes = max(1, len(used_by_layers)).

    Then clears actual Parameter tensor references (parent_params) from
    LayerPassLog entries to reduce memory, while preserving ParamLog._param_ref
    for potential backward() calls by the user.
    """
    # Build used_by_layers and linked_params from LayerPassLog entries
    for layer_entry in self.layer_list:
        if not layer_entry.parent_param_logs:
            continue
        addresses_in_op = [pl.address for pl in layer_entry.parent_param_logs]
        for pl in layer_entry.parent_param_logs:
            if layer_entry.layer_label not in pl.used_by_layers:
                pl.used_by_layers.append(layer_entry.layer_label)
            # Link to other params in the same operation
            for other_addr in addresses_in_op:
                if other_addr != pl.address and other_addr not in pl.linked_params:
                    pl.linked_params.append(other_addr)

    # Populate num_passes: how many times this parameter was used in the forward pass
    for pl in self.param_logs:
        pl.num_passes = max(1, len(pl.used_by_layers))

    # ParamLog gradient metadata is populated lazily via backward hooks in _log_tensor_grad.
    # Each ParamLog holds a _param_ref to the actual nn.Parameter, and _update_grad_from_param()
    # reads param.grad after backward is called.

    # Note: _param_ref (GC-1) is NOT cleared here because the user may call backward()
    # after postprocessing to populate gradients. It's cleared in cleanup() instead.
    # Same for parent_param_logs (GC-9) and func_applied (GC-10) — needed by validation.

    # Clear actual Parameter tensor references from LayerPassLog entries to save memory.
    for layer_entry in self.layer_list:
        layer_entry.parent_params = []


def _build_root_module_log(self: "ModelLog", pass_dict: dict, mbd: dict) -> "ModuleLog":
    """Build the root ModuleLog ("self") representing the model itself.

    The root module encompasses all layers and params. Its address_children are
    only direct children (no dots in address), while call_children may include
    grandchildren called directly (e.g., self.level21.level12(x)).
    """
    from ..data_classes.param_log import ParamAccessor

    root_meta = self._module_metadata.get("self", {})
    root_all_layers = list(self.layer_logs.keys())

    root_param_dict = {pl.address: pl for pl in self.param_logs}
    root_params = ParamAccessor(root_param_dict)
    root_num_params = sum(pl.num_params for pl in self.param_logs)
    root_num_trainable = sum(pl.num_params for pl in self.param_logs if pl.trainable)
    root_num_frozen = sum(pl.num_params for pl in self.param_logs if not pl.trainable)
    root_fsize = sum(pl.memory for pl in self.param_logs)

    root_module = ModuleLog(
        address="self",
        all_addresses=root_meta.get("all_addresses", ["self"]),
        name="self",
        module_class_name=root_meta.get("module_class_name", self.model_name),
        source_file=root_meta.get("source_file"),
        source_line=root_meta.get("source_line"),
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
        address_children=[m for m in mbd["top_level_modules"] if "." not in m],
        address_depth=0,
        call_parent=None,
        call_children=mbd["top_level_modules"][:],
        nesting_depth=0,
        num_passes=1,
        passes={},
        pass_labels=["self:1"],
        all_layers=root_all_layers,
        params=root_params,
        num_params=root_num_params,
        num_params_trainable=root_num_trainable,
        num_params_frozen=root_num_frozen,
        params_memory=root_fsize,
        requires_grad=root_num_trainable > 0,
        buffer_layers=list(self.buffer_layers),
        is_training=root_meta.get("is_training", True),
        has_forward_hooks=root_meta.get("has_forward_hooks", False),
        has_backward_hooks=root_meta.get("has_backward_hooks", False),
        extra_attributes=root_meta.get("extra_attributes", {}),
        methods=root_meta.get("methods", []),
        _source_model_log=self,
    )

    root_pass = ModulePassLog(
        module_address="self",
        pass_num=1,
        pass_label="self:1",
        layers=root_all_layers,
        input_layers=list(self.input_layers),
        output_layers=list(self.output_layers),
        call_parent=None,
        call_children=mbd["top_level_module_passes"][:],
        all_module_addresses=root_meta.get("all_addresses", ["self"]),
    )
    root_module.passes = {1: root_pass}
    pass_dict["self:1"] = root_pass

    return root_module


def _compute_nesting_depths(module_dict: dict, root_module: "ModuleLog") -> None:
    """Assign nesting_depth to each ModuleLog via BFS from the root.

    Root ("self") has depth 0. Each level of call_children nesting adds 1.
    """
    visited = {"self": 0}
    queue: deque = deque()
    for child_addr in root_module.call_children:
        if child_addr in module_dict:
            module_dict[child_addr].nesting_depth = 1
            visited[child_addr] = 1
            queue.append(child_addr)

    while queue:
        addr = queue.popleft()
        ml = module_dict[addr]
        for child_addr in ml.call_children:
            if child_addr not in visited and child_addr in module_dict:
                depth = visited[addr] + 1
                module_dict[child_addr].nesting_depth = depth
                visited[child_addr] = depth
                queue.append(child_addr)


def _append_unique_child_label(child_labels: List[str], child_label: str) -> None:
    """Append a child label if it has not been seen yet.

    Parameters
    ----------
    child_labels:
        Ordered list being accumulated.
    child_label:
        Child label to append.
    """
    if child_label not in child_labels:
        child_labels.append(child_label)


def _strip_pass_suffix(layer_label: str) -> str:
    """Strip the ``:pass_num`` suffix from a layer label.

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


def _merge_layer_log_conditional_fields(
    layer_log: "LayerLog",
    pass_log: "LayerPassLog",
) -> None:
    """Merge one pass's conditional metadata into an aggregate ``LayerLog``.

    Parameters
    ----------
    layer_log:
        Aggregate layer entry being updated.
    pass_log:
        Per-pass layer entry contributing conditional metadata.
    """
    if pass_log.in_cond_branch:
        layer_log.in_cond_branch = True

    stack_signature = tuple(pass_log.conditional_branch_stack)
    if stack_signature not in layer_log.conditional_branch_stack_passes:
        layer_log.conditional_branch_stacks.append(list(pass_log.conditional_branch_stack))
        layer_log.conditional_branch_stack_passes[stack_signature] = []

    signature_passes = layer_log.conditional_branch_stack_passes[stack_signature]
    if pass_log.pass_num not in signature_passes:
        signature_passes.append(pass_log.pass_num)
        signature_passes.sort()

    for conditional_id, branch_children in pass_log.cond_branch_children_by_cond.items():
        merged_branch_children = layer_log.cond_branch_children_by_cond.setdefault(
            conditional_id,
            {},
        )
        for branch_kind, child_labels in branch_children.items():
            merged_child_labels = merged_branch_children.setdefault(branch_kind, [])
            for child_label in child_labels:
                _append_unique_child_label(
                    merged_child_labels,
                    _strip_pass_suffix(child_label),
                )


def _rebuild_layer_log_conditional_views(layer_log: "LayerLog") -> None:
    """Recompute aggregate conditional compatibility views for one ``LayerLog``.

    Parameters
    ----------
    layer_log:
        Aggregate layer entry whose derived conditional fields are refreshed.
    """
    layer_log.in_cond_branch = any(
        len(branch_stack) > 0 for branch_stack in layer_log.conditional_branch_stacks
    )

    cond_branch_start_children: List[str] = []
    for pass_num in sorted(layer_log.passes):
        pass_log = layer_log.passes[pass_num]
        for child_label in pass_log.cond_branch_start_children:
            _append_unique_child_label(
                cond_branch_start_children,
                _strip_pass_suffix(child_label),
            )
    layer_log.cond_branch_start_children = cond_branch_start_children

    cond_branch_then_children: List[str] = []
    cond_branch_elif_children: Dict[int, List[str]] = {}
    cond_branch_else_children: List[str] = []
    for branch_children in layer_log.cond_branch_children_by_cond.values():
        for child_label in branch_children.get("then", []):
            _append_unique_child_label(cond_branch_then_children, child_label)
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            elif_children = cond_branch_elif_children.setdefault(elif_index, [])
            for child_label in child_labels:
                _append_unique_child_label(elif_children, child_label)
        for child_label in branch_children.get("else", []):
            _append_unique_child_label(cond_branch_else_children, child_label)

    layer_log.cond_branch_then_children = cond_branch_then_children
    layer_log.cond_branch_elif_children = cond_branch_elif_children
    layer_log.cond_branch_else_children = cond_branch_else_children


def _rebuild_conditional_edge_passes(self: "ModelLog") -> None:
    """Recompute rolled conditional edge-pass metadata from arm-entry edges.

    Parameters
    ----------
    self:
        Model log being finalized.
    """
    conditional_edge_passes: Dict[Tuple[str, str, int, str], List[int]] = defaultdict(list)
    for (conditional_id, branch_kind), edge_list in self.conditional_arm_edges.items():
        for parent_label, child_label in edge_list:
            parent_no_pass = _strip_pass_suffix(parent_label)
            child_no_pass = _strip_pass_suffix(child_label)
            child_pass_num = _get_label_pass_num(child_label)
            edge_key = (
                parent_no_pass,
                child_no_pass,
                conditional_id,
                branch_kind,
            )
            if child_pass_num not in conditional_edge_passes[edge_key]:
                conditional_edge_passes[edge_key].append(child_pass_num)

    self.conditional_edge_passes = {
        edge_key: sorted(pass_nums) for edge_key, pass_nums in conditional_edge_passes.items()
    }


def _get_label_pass_num(layer_label: str) -> int:
    """Extract the pass number encoded in a layer label.

    Parameters
    ----------
    layer_label:
        Layer label that may include a ``:pass_num`` suffix.

    Returns
    -------
    int
        Parsed pass number, or ``1`` for pass-free labels.
    """
    label_parts = layer_label.rsplit(":", 1)
    if len(label_parts) == 2 and label_parts[1].isdigit():
        return int(label_parts[1])
    return 1


def _build_submodule_pass_logs(
    self: "ModelLog",
    address: str,
    num_passes: int,
    pass_dict: dict,
    mbd: dict,
    _child_to_parent_pass: Optional[dict] = None,
    all_module_addresses: Optional[list] = None,
) -> tuple:
    """Build ModulePassLog objects for all passes of a single submodule.

    For each pass, derives input/output layers from LayerPassLog fields,
    retrieves forward args, and resolves call parent/children relationships.

    Returns:
        Tuple of (passes dict {pass_num: ModulePassLog}, pass_labels list).
    """
    passes = {}
    pass_labels_list = []
    for pass_num in range(1, num_passes + 1):
        pass_label = f"{address}:{pass_num}"
        pass_labels_list.append(pass_label)

        pass_layers = list(mbd["module_pass_layers"].get(pass_label, []))

        # Derive input/output layers per pass from LayerPassLog fields
        pass_input_layers = []
        pass_output_layers = []
        for layer_label in pass_layers:
            if layer_label in self.layer_dict_all_keys:
                te = self.layer_dict_all_keys[layer_label]
                if te.is_submodule_input and pass_label in te.module_passes_entered:
                    pass_input_layers.append(layer_label)
                if te.is_submodule_output and pass_label in te.module_passes_exited:
                    pass_output_layers.append(layer_label)

        # Forward args for this pass
        fwd_args = self._module_forward_args.get((address, pass_num))
        fwd_positional = fwd_args[0] if fwd_args else None
        fwd_kwargs = fwd_args[1] if fwd_args else None

        # Call children for this pass
        call_children_pass = list(mbd["module_pass_children"].get(pass_label, []))

        # Call parent for this pass: look up from pre-computed reverse mapping
        call_parent_pass = _child_to_parent_pass.get(pass_label) if _child_to_parent_pass else None
        if call_parent_pass is None and pass_label in mbd["top_level_module_passes"]:
            call_parent_pass = "self:1"

        module_pass_log = ModulePassLog(
            module_address=address,
            pass_num=pass_num,
            pass_label=pass_label,
            layers=pass_layers,
            input_layers=pass_input_layers,
            output_layers=pass_output_layers,
            forward_args=fwd_positional,
            forward_kwargs=fwd_kwargs,
            call_parent=call_parent_pass,
            call_children=call_children_pass,
            all_module_addresses=all_module_addresses,
        )
        passes[pass_num] = module_pass_log
        pass_dict[pass_label] = module_pass_log

    return passes, pass_labels_list


def _resolve_call_hierarchy(passes: dict) -> tuple:
    """Derive module-level call_children and call_parent from per-pass data.

    Unions call_children across all passes (stripping pass suffixes to get
    addresses). call_parent is taken from the first pass and converted from
    pass-label to address.

    Returns:
        Tuple of (call_children_all: list of addresses, call_parent_addr or None).
    """
    call_children_all = []
    for module_pass_log in passes.values():
        for child_pass_label in module_pass_log.call_children:
            cc_addr = child_pass_label.split(":")[0]
            if cc_addr not in call_children_all:
                call_children_all.append(cc_addr)

    call_parent_addr = None
    if passes:
        first_pass = passes[1]
        if first_pass.call_parent and first_pass.call_parent != "self:1":
            call_parent_addr = first_pass.call_parent.split(":")[0]
        elif first_pass.call_parent == "self:1":
            call_parent_addr = "self"

    return call_children_all, call_parent_addr


class ModuleParamInfo(NamedTuple):
    """Parameter and buffer info for a single module."""

    params: object  # ParamAccessor
    num_params: int
    num_trainable: int
    num_frozen: int
    memory: int
    buffer_layers: list


def _build_module_param_info(
    self: "ModelLog", address: str, mbd: dict, _buffer_layers_by_module: Optional[dict] = None
) -> ModuleParamInfo:
    """Gather parameter counts, sizes, and buffer layers for a single module."""
    from ..data_classes.param_log import ParamAccessor

    module_param_dict = {pl.address: pl for pl in self._param_logs_by_module.get(address, [])}  # type: ignore[attr-defined]
    module_params = ParamAccessor(module_param_dict)
    m_num_params = mbd["module_nparams"].get(address, 0)
    m_num_trainable = mbd["module_nparams_trainable"].get(address, 0)
    m_num_frozen = mbd["module_nparams_frozen"].get(address, 0)
    m_fsize = sum(pl.memory for pl in module_param_dict.values())

    if _buffer_layers_by_module is not None:
        module_buffer_layers = list(_buffer_layers_by_module.get(address, []))
    else:
        module_buffer_layers = [
            bl
            for bl in self.buffer_layers
            if bl in self.layer_dict_all_keys
            and hasattr(self.layer_dict_all_keys[bl], "buffer_address")
            and self.layer_dict_all_keys[bl].buffer_address is not None
            and self.layer_dict_all_keys[bl].buffer_address.rsplit(".", 1)[0] == address
        ]

    return ModuleParamInfo(
        module_params, m_num_params, m_num_trainable, m_num_frozen, m_fsize, module_buffer_layers
    )


def _build_module_logs(self: "ModelLog") -> None:
    """Step 17: Build structured ModuleLog/ModulePassLog objects.

    Constructs the module hierarchy from _module_build_data (populated in Step 10)
    and _module_metadata (captured during model preparation). Creates:
    - A root ModuleLog for "self" (the model itself).
    - ModuleLogs for each submodule with ModulePassLogs for each pass.
    - ModuleAccessor and BufferAccessor for user-facing access.

    MUST NOT be called in fast mode (postprocess_fast) because _module_build_data
    is not repopulated when Step 10 is skipped. Existing module logs from the
    exhaustive pass remain valid (#108).

    Handles shared modules (same nn.Module registered under multiple addresses)
    via an alias-to-metadata map. Computes nesting depths via BFS from root.
    Clears temporary state (_module_metadata, _module_forward_args, _module_build_data)
    after building.
    """
    mbd = self._module_build_data
    module_dict = {}  # address -> ModuleLog
    pass_dict: Dict[str, ModulePassLog] = {}  # "addr:pass" -> ModulePassLog
    module_order = []  # ordered by first appearance

    # --- Build root ModuleLog ("self") ---
    root_module = _build_root_module_log(self, pass_dict, mbd)
    module_dict["self"] = root_module
    module_order.append(root_module)

    # Pre-compute param_logs grouped by module address
    self._param_logs_by_module = defaultdict(list)  # type: ignore[attr-defined]
    for pl in self.param_logs:
        self._param_logs_by_module[pl.module_address].append(pl)  # type: ignore[attr-defined]

    # Pre-compute reverse mapping: child_pass_label -> parent_pass_label
    _child_to_parent_pass = {}
    for parent_pass_label, children in mbd["module_pass_children"].items():
        for child_label in children:
            _child_to_parent_pass[child_label] = parent_pass_label

    # Build alias-to-metadata map for shared modules (same nn.Module instance
    # registered under multiple addresses). _module_metadata stores metadata
    # under the FIRST address visited by _capture_module_metadata, but
    # tl_module_address may be overwritten to a LATER address by
    # _prepare_model_once. This map ensures all aliases resolve to the same meta.
    _metadata_by_alias: Dict[str, dict] = {}
    for _primary_addr, _meta in self._module_metadata.items():
        for _alias in _meta.get("all_addresses", [_primary_addr]):
            _metadata_by_alias[_alias] = _meta

    # Pre-compute buffer layers grouped by parent module address (O6).
    _buffer_layers_by_module = defaultdict(list)
    for bl in self.buffer_layers:
        if bl in self.layer_dict_all_keys:
            bl_entry = self.layer_dict_all_keys[bl]
            if hasattr(bl_entry, "buffer_address") and bl_entry.buffer_address is not None:
                module_addr = bl_entry.buffer_address.rsplit(".", 1)[0]
                _buffer_layers_by_module[module_addr].append(bl)

    # --- Build ModuleLogs for each submodule ---
    for address in mbd["module_addresses"]:
        meta = _metadata_by_alias.get(address, {})
        num_passes = mbd["module_num_passes"].get(address, 1)

        name = address.rsplit(".", 1)[-1] if "." in address else address
        address_parent = address.rsplit(".", 1)[0] if "." in address else "self"
        address_depth = address.count(".") + 1
        all_layers_raw = list(mbd["module_layers"].get(address, []))
        seen = set()
        all_layers = []
        for label in all_layers_raw:
            entry = self.layer_dict_all_keys.get(label)
            if entry is not None:
                no_pass = entry.layer_label_no_pass
                if no_pass not in seen:
                    seen.add(no_pass)
                    all_layers.append(no_pass)

        all_addresses = meta.get("all_addresses", [address])
        passes, pass_labels_list = _build_submodule_pass_logs(
            self,
            address,
            num_passes,
            pass_dict,
            mbd,
            _child_to_parent_pass,
            all_module_addresses=all_addresses,
        )
        call_children_all, call_parent_addr = _resolve_call_hierarchy(passes)
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

        ml = ModuleLog(
            address=address,
            all_addresses=meta.get("all_addresses", [address]),
            name=name,
            module_class_name=meta.get("module_class_name", mbd["module_types"].get(address, "")),
            source_file=meta.get("source_file"),
            source_line=meta.get("source_line"),
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
            nesting_depth=0,  # computed below
            num_passes=num_passes,
            passes=passes,
            pass_labels=pass_labels_list,
            all_layers=all_layers,
            params=param_info.params,  # type: ignore[arg-type]
            num_params=param_info.num_params,
            num_params_trainable=param_info.num_trainable,
            num_params_frozen=param_info.num_frozen,
            params_memory=param_info.memory,
            requires_grad=param_info.num_trainable > 0,
            buffer_layers=param_info.buffer_layers,
            is_training=mbd["module_training_modes"].get(address, meta.get("is_training", True)),
            has_forward_hooks=meta.get("has_forward_hooks", False),
            has_backward_hooks=meta.get("has_backward_hooks", False),
            extra_attributes=meta.get("extra_attributes", {}),
            methods=meta.get("methods", []),
            _source_model_log=self,
        )
        module_dict[address] = ml
        module_order.append(ml)

    # --- Compute nesting depths ---
    _compute_nesting_depths(module_dict, root_module)

    rebuild_model_log_accessors(self, module_dict, module_order, pass_dict)

    # Clean up temporary build state to free memory. These dicts are only
    # needed during construction and are not part of the user-facing API.
    self._module_metadata = {}
    self._module_forward_args = {}
    from ..data_classes.model_log import _init_module_build_data

    self._module_build_data = _init_module_build_data()

    # GC-11: Clear forward_args/kwargs from ModulePassLogs to release tensor references.
    # These can hold large tensors from the model's forward() call args.
    for pass_log in pass_dict.values():
        pass_log.forward_args_summary = format_call_arg(pass_log.forward_args)
        pass_log.forward_kwargs_summary = format_call_arg(pass_log.forward_kwargs)
        pass_log.forward_args = None
        pass_log.forward_kwargs = None


def _build_layer_logs(self: "ModelLog") -> None:
    """Step 16.5: Build aggregate LayerLog objects from per-pass LayerPassLog entries.

    Groups layer_list entries by layer_label_no_pass and creates a LayerLog for
    each unique layer. For single-pass layers, the LayerLog is a thin wrapper
    delegating attribute access to the sole LayerPassLog.

    MERGE RULES for multi-pass layers:
    1. has_input_ancestor: OR across passes (True if ANY pass has an input ancestor).
    2. io_role: Character-wise merge with '*' for differing characters
       (e.g., "output.0" + "output.1" -> "output.*").
    3. is_leaf_module_output: OR across passes.
    4. in_cond_branch: OR across passes.
    5. conditional_branch_stacks / conditional_branch_stack_passes:
       unique stack signatures in first-seen order plus sorted pass maps.
    6. cond_branch_children_by_cond: pass-stripped union across passes with
       insertion-order preservation.

    Derived aggregate views (cond_branch_then/elif/else/start_children) are
    recomputed after the merge from the cond-id-aware primary structures.

    All other 78+ fields use first-pass values only. This is correct because
    same-layer grouping requires same structural position, so module containment,
    function info, param info, etc. are identical across passes.

    Note: modules_exited/module_passes_exited are NOT updated during merge
    (same structural position implies same modules). The comment in layer_log.py:107
    ("may be updated") is misleading — no such update occurs.
    """
    from collections import OrderedDict

    from ..data_classes.layer_log import LayerLog

    layer_logs = OrderedDict()

    for pass_log in self.layer_list:
        no_pass_label = pass_log.layer_label_no_pass

        if no_pass_label not in layer_logs:
            # First pass: create LayerLog from this pass's data.
            layer_log = LayerLog(pass_log)
            layer_logs[no_pass_label] = layer_log
        else:
            # Subsequent pass: merge only the 3 aggregate fields.
            layer_log = layer_logs[no_pass_label]
            # Merge field 1/3: has_input_ancestor (OR across passes).
            if pass_log.has_input_ancestor:
                layer_log.has_input_ancestor = True
            # Merge field 2/3: io_role (char-merge with '*').
            if layer_log.io_role is not None and pass_log.io_role is not None:
                merged = "".join(
                    c if c == s else "*"
                    for c, s in zip(
                        layer_log.io_role,
                        pass_log.io_role,
                    )
                )
                if merged.endswith("."):
                    merged = merged[:-1]
                if merged.endswith("*"):
                    merged = merged.rstrip("*") + "*"
                layer_log.io_role = merged
            # Merge field 3/3: is_leaf_module_output (OR across passes).
            if pass_log.is_leaf_module_output:
                layer_log.is_leaf_module_output = True

        layer_log.passes[pass_log.pass_num] = pass_log
        layer_log.pass_labels.append(pass_log.layer_label)
        pass_log.parent_layer_log = layer_log  # type: ignore[assignment]
        _merge_layer_log_conditional_fields(layer_log, pass_log)

    for layer_log in layer_logs.values():
        _rebuild_layer_log_conditional_views(layer_log)

    self.layer_logs = layer_logs
    _rebuild_conditional_edge_passes(self)


def _set_pass_finished(self) -> None:
    """Step 18: Mark the ModelLog and all LayerPassLogs as pass-finished.

    Sets ``_pass_finished = True`` on the ModelLog and every retained
    LayerPassLog entry. This flag switches various methods (e.g., __str__,
    __getitem__) from their "realtime debugging" behavior to their
    "user-facing" behavior.

    Note: _pass_finished is NOT reset between exhaustive and fast passes.
    This is intentional — it enables fast-path postprocessing to use
    fully-populated lookup dicts from the exhaustive pass.
    """
    for layer_label in self.layer_dict_main_keys:
        tensor = self.layer_dict_main_keys[layer_label]
        tensor._pass_finished = True
    self._pass_finished = True
