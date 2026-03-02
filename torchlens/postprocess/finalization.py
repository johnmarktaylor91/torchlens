"""Steps 13-18: Tensor undecoration, timing, param logs, module logs, pass finish, graph rolling."""

import time
from collections import deque
from typing import TYPE_CHECKING

import torch

from ..data_classes.module_log import ModuleAccessor, ModuleLog, ModulePassLog
from ..data_classes.tensor_log import RolledTensorLog
from ..helper_funcs import get_vars_of_type_from_obj, human_readable_size

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def _undecorate_all_saved_tensors(self):
    """Utility function to undecorate all saved tensors."""
    tensors_to_undecorate = []
    for layer_label in self.layer_labels:
        tensor_entry = self.layer_dict_main_keys[layer_label]
        if tensor_entry.tensor_contents is not None:
            tensors_to_undecorate.append(tensor_entry.tensor_contents)

        tensors_to_undecorate.extend(
            get_vars_of_type_from_obj(tensor_entry.creation_args, torch.Tensor, search_depth=2)
        )
        tensors_to_undecorate.extend(
            get_vars_of_type_from_obj(tensor_entry.creation_kwargs, torch.Tensor, search_depth=2)
        )

    for t in tensors_to_undecorate:
        if hasattr(t, "tl_tensor_label_raw"):
            delattr(t, "tl_tensor_label_raw")


def _log_time_elapsed(self):
    self.pass_end_time = time.time()
    self.elapsed_time_cleanup = (
        self.pass_end_time
        - self.pass_start_time
        - self.elapsed_time_setup
        - self.elapsed_time_forward_pass
    )
    self.elapsed_time_total = self.pass_end_time - self.pass_start_time
    self.elapsed_time_torchlens_logging = self.elapsed_time_total - self.elapsed_time_function_calls


def _finalize_param_logs(self: "ModelLog"):
    """Populate ParamLog reverse mappings, linked params, num_passes, and gradient metadata."""
    from ..helper_funcs import get_tensor_memory_amount, human_readable_size

    # Build tensor_log_entries and linked_params from TensorLogEntries
    for tensor_entry in self.layer_list:
        if not tensor_entry.parent_param_logs:
            continue
        addresses_in_op = [pl.address for pl in tensor_entry.parent_param_logs]
        for pl in tensor_entry.parent_param_logs:
            if tensor_entry.layer_label not in pl.tensor_log_entries:
                pl.tensor_log_entries.append(tensor_entry.layer_label)
            # Link to other params in the same operation
            for other_addr in addresses_in_op:
                if other_addr != pl.address and other_addr not in pl.linked_params:
                    pl.linked_params.append(other_addr)

    # Populate num_passes: how many times this parameter was used in the forward pass
    for pl in self.param_logs:
        pl.num_passes = max(1, len(pl.tensor_log_entries))

    # ParamLog gradient metadata is populated lazily via backward hooks in _log_tensor_grad.
    # Each ParamLog holds a _param_ref to the actual nn.Parameter, and _update_grad_from_param()
    # reads param.grad after backward is called.

    # Clear actual Parameter tensor references from TensorLogEntries to save memory
    for tensor_entry in self.layer_list:
        tensor_entry.parent_params = None


def _build_module_logs(self: "ModelLog"):
    """Build structured ModuleLog/ModulePassLog objects from raw module_* dicts and _module_metadata.

    Called as Step 17 of postprocess(), after all raw module data has been populated
    and layer labels have been finalized.
    """
    from ..data_classes.param_log import ParamAccessor

    module_dict = {}  # address -> ModuleLog
    pass_dict = {}  # "addr:pass" -> ModulePassLog
    module_order = []  # ordered by first appearance

    # --- Build root ModuleLog ("self") ---
    root_meta = self._module_metadata.get("self", {})

    # Collect all layers belonging to root (= all layers in the model)
    root_all_layers = list(self.layer_labels)

    # Build per-module ParamAccessor for root: all params
    root_param_dict = {pl.address: pl for pl in self.param_logs}
    root_params = ParamAccessor(root_param_dict)
    root_num_params = sum(pl.num_params for pl in self.param_logs)
    root_num_trainable = sum(pl.num_params for pl in self.param_logs if pl.trainable)
    root_num_frozen = sum(pl.num_params for pl in self.param_logs if not pl.trainable)
    root_fsize = sum(pl.fsize for pl in self.param_logs)

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
        address_children=self.top_level_modules[:],
        address_depth=0,
        call_parent=None,
        call_children=self.top_level_modules[:],
        nesting_depth=0,
        num_passes=1,
        passes={},
        pass_labels=["self:1"],
        all_layers=root_all_layers,
        params=root_params,
        num_params=root_num_params,
        num_params_trainable=root_num_trainable,
        num_params_frozen=root_num_frozen,
        params_fsize=root_fsize,
        params_fsize_nice=human_readable_size(root_fsize),
        requires_grad=root_num_trainable > 0,
        buffer_layers=list(self.buffer_layers),
        training_mode=root_meta.get("training_mode", True),
        has_forward_hooks=root_meta.get("has_forward_hooks", False),
        has_backward_hooks=root_meta.get("has_backward_hooks", False),
        extra_attributes=root_meta.get("extra_attributes", {}),
        methods=root_meta.get("methods", []),
        _source_model_log=self,
    )

    # Build root ModulePassLog
    root_pass = ModulePassLog(
        module_address="self",
        pass_num=1,
        pass_label="self:1",
        layers=root_all_layers,
        input_layers=list(self.input_layers),
        output_layers=list(self.output_layers),
        call_parent=None,
        call_children=self.top_level_module_passes[:],
    )
    root_module.passes = {1: root_pass}
    module_dict["self"] = root_module
    pass_dict["self:1"] = root_pass
    module_order.append(root_module)

    # --- Build ModuleLogs for each submodule ---
    for address in self.module_addresses:
        meta = self._module_metadata.get(address, {})
        num_passes = self.module_num_passes.get(address, 1)

        # Name = last segment of address
        name = address.rsplit(".", 1)[-1] if "." in address else address

        # Address parent
        if "." in address:
            address_parent = address.rsplit(".", 1)[0]
        else:
            address_parent = "self"

        # Address depth: number of dots + 1
        address_depth = address.count(".") + 1

        # All layers across all passes
        all_layers = list(self.module_layers.get(address, []))

        # Build ModulePassLogs
        passes = {}
        pass_labels_list = []
        for pn in range(1, num_passes + 1):
            pass_label = f"{address}:{pn}"
            pass_labels_list.append(pass_label)

            pass_layers = list(self.module_pass_layers.get(pass_label, []))

            # Derive input/output layers per pass from TensorLog fields
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
            fwd_args = self._module_forward_args.get((address, pn))
            fwd_positional = fwd_args[0] if fwd_args else None
            fwd_kwargs = fwd_args[1] if fwd_args else None

            # Call children for this pass
            call_children_pass = list(self.module_pass_children.get(pass_label, []))

            # Call parent for this pass: find which module:pass contains this one
            call_parent_pass = None
            for parent_pass_label, children in self.module_pass_children.items():
                if pass_label in children:
                    call_parent_pass = parent_pass_label
                    break
            # If not found in module_pass_children, check if it's top-level
            if call_parent_pass is None and pass_label in self.top_level_module_passes:
                call_parent_pass = "self:1"

            mpl = ModulePassLog(
                module_address=address,
                pass_num=pn,
                pass_label=pass_label,
                layers=pass_layers,
                input_layers=pass_input_layers,
                output_layers=pass_output_layers,
                forward_args=fwd_positional,
                forward_kwargs=fwd_kwargs,
                call_parent=call_parent_pass,
                call_children=call_children_pass,
            )
            passes[pn] = mpl
            pass_dict[pass_label] = mpl

        # Call children (union across all passes, addresses only)
        call_children_all = []
        for pn, mpl in passes.items():
            for cc in mpl.call_children:
                cc_addr = cc.split(":")[0]
                if cc_addr not in call_children_all:
                    call_children_all.append(cc_addr)

        # Call parent (address only, from first pass)
        call_parent_addr = None
        if passes:
            first_pass = passes[1]
            if first_pass.call_parent and first_pass.call_parent != "self:1":
                call_parent_addr = first_pass.call_parent.split(":")[0]
            elif first_pass.call_parent == "self:1":
                call_parent_addr = "self"

        # Build per-module ParamAccessor
        module_param_dict = {
            pl.address: pl for pl in self.param_logs if pl.module_address == address
        }
        module_params = ParamAccessor(module_param_dict)
        m_num_params = self.module_nparams.get(address, 0)
        m_num_trainable = self.module_nparams_trainable.get(address, 0)
        m_num_frozen = self.module_nparams_frozen.get(address, 0)
        m_fsize = sum(pl.fsize for pl in module_param_dict.values())

        # Buffer layers belonging to this module
        module_buffer_layers = [
            bl
            for bl in self.buffer_layers
            if bl in self.layer_dict_all_keys
            and hasattr(self.layer_dict_all_keys[bl], "buffer_address")
            and self.layer_dict_all_keys[bl].buffer_address is not None
            and self.layer_dict_all_keys[bl].buffer_address.rsplit(".", 1)[0] == address
        ]

        ml = ModuleLog(
            address=address,
            all_addresses=meta.get("all_addresses", [address]),
            name=name,
            module_class_name=meta.get("module_class_name", self.module_types.get(address, "")),
            source_file=meta.get("source_file"),
            source_line=meta.get("source_line"),
            class_docstring=meta.get("class_docstring"),
            init_signature=meta.get("init_signature"),
            init_docstring=meta.get("init_docstring"),
            forward_signature=meta.get("forward_signature"),
            forward_docstring=meta.get("forward_docstring"),
            address_parent=address_parent,
            address_children=meta.get(
                "address_children", list(self.module_children.get(address, []))
            ),
            address_depth=address_depth,
            call_parent=call_parent_addr,
            call_children=call_children_all,
            nesting_depth=0,  # computed below
            num_passes=num_passes,
            passes=passes,
            pass_labels=pass_labels_list,
            all_layers=all_layers,
            params=module_params,
            num_params=m_num_params,
            num_params_trainable=m_num_trainable,
            num_params_frozen=m_num_frozen,
            params_fsize=m_fsize,
            params_fsize_nice=human_readable_size(m_fsize),
            requires_grad=m_num_trainable > 0,
            buffer_layers=module_buffer_layers,
            training_mode=self.module_training_modes.get(address, meta.get("training_mode", True)),
            has_forward_hooks=meta.get("has_forward_hooks", False),
            has_backward_hooks=meta.get("has_backward_hooks", False),
            extra_attributes=meta.get("extra_attributes", {}),
            methods=meta.get("methods", []),
            _source_model_log=self,
        )
        module_dict[address] = ml
        module_order.append(ml)

    # --- Compute nesting_depth via BFS from root using call_children ---
    # Root is depth 0, top-level modules are depth 1, etc.
    visited = {"self": 0}
    queue = deque()
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

    # --- Build ModuleAccessor and assign to ModelLog ---
    self._module_logs = ModuleAccessor(module_dict, module_order, pass_dict)

    # --- Clean up temporary state ---
    self._module_metadata = {}
    self._module_forward_args = {}


def _set_pass_finished(self):
    """Sets the ModelLog to "pass finished" status, indicating that the pass is done, so
    the "final" rather than "realtime debugging" mode of certain functions should be used.
    """
    for layer_label in self.layer_dict_main_keys:
        tensor = self.layer_dict_main_keys[layer_label]
        tensor._pass_finished = True
    self._pass_finished = True


def _roll_graph(self):
    """
    Converts the graph to rolled-up format for plotting purposes, such that each node now represents
    all passes of a given layer instead of having separate nodes for each pass.
    """
    for layer_label, node in self.layer_dict_main_keys.items():
        layer_label_no_pass = self[layer_label].layer_label_no_pass
        if (
            layer_label_no_pass in self.layer_dict_rolled
        ):  # If rolled-up layer has already been added, fetch it:
            rolled_node = self.layer_dict_rolled[layer_label_no_pass]
        else:  # If it hasn't been added, make it:
            rolled_node = RolledTensorLog(node)
            self.layer_dict_rolled[node.layer_label_no_pass] = rolled_node
            self.layer_list_rolled.append(rolled_node)
        rolled_node.update_data(node)
        rolled_node.add_pass_info(node)
