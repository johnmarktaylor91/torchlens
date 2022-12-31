# This module has functions for processing the computation graphs that come out of the other functions.

import random
from collections import OrderedDict
from typing import Dict, List, Union

import numpy as np
import torch

from torch_decorate import undecorate_tensor
from torchlens.helper_funcs import human_readable_size


def unmutate_tensors_and_funcs_in_history(history_dict: Dict):
    """Returns all tensors in the history dict to normal, unmutated versions.

    Args:
        history_dict: Dictionary of history.

    Returns:
        history_dict with all tensors unmutated.
    """
    tensor_log = history_dict['tensor_log']
    mutant_to_orig_funcs_dict = history_dict['mutant_to_orig_funcs_dict']
    for node in tensor_log.values():
        node['tensor_contents'] = undecorate_tensor(node['tensor_contents'])
        for p, parent_param in enumerate(node['parent_params']):
            node['parent_params'][p] = undecorate_tensor(parent_param)

        for a in range(len(node['creation_args'])):
            arg = node['creation_args'][a]
            if issubclass(type(arg), (torch.Tensor, torch.nn.Parameter)):
                new_arg = undecorate_tensor(arg)
                node['creation_args'] = node['creation_args'][:a] + tuple([new_arg]) + node['creation_args'][a + 1:]

        for key, val in node['creation_kwargs'].items():
            if issubclass(type(val), (torch.Tensor, torch.nn.Parameter)):
                node['creation_kwargs'][key] = undecorate_tensor(val)

        for f, func in enumerate(node['funcs_applied']):
            if func in mutant_to_orig_funcs_dict:
                node['funcs_applied'][f] = mutant_to_orig_funcs_dict[func]

    return history_dict


def postprocess_history_dict(history_dict: Dict) -> Dict:
    """Takes the raw history_dict after the forward pass and post-processes it, adding further useful
    annotations and trimming unnecessary information. This is the final "internal" version of the
    log that will be passed to other functions (a prettier, stripped down version is returned to the user).

    Args:
        history_dict: Dictionary of activations.

    Returns:
        Cleaned-up history dict.
    """
    # List of transforms to apply.
    # TODO: Figure out how much time this part takes, how many graph_traversals; try to only traverse graph once.

    graph_transforms = [
        unmutate_tensors_and_funcs_in_history,  # return any tensors in history dict to their original definition
        strip_irrelevant_nodes,  # remove island nodes unconnected to inputs or outputs
        mark_output_ancestors,  # mark nodes as ancestors of the output
        add_output_nodes,  # add explicit output nodes
        annotate_total_layer_passes,  # note the total passes of the param nodes
        identify_repeated_functions,  # find repeated functions between repeated param nodes
        mark_conditional_branches,  # mark branches that are involved in evaluating conditionals
        annotate_node_names,  # make the nodes names more human-readable
        map_layer_names_to_op_nums,  # get the operation numbers for any human-readable layer labels
        annotate_internal_tensor_modules,  # marks the internally generated tensors with contaning modules
        cluster_modules,  # identify the unique module instances and the right mappings
        tally_sizes_and_params  # tally total sizes of tensors and params
    ]

    for graph_transform in graph_transforms:
        history_dict = graph_transform(history_dict)

    return history_dict


def get_all_tensor_lookup_keys(node: Dict,
                               node_index: int,
                               num_tensors_to_keep: int,
                               history_dict: dict) -> List[Union[str, int]]:
    """Gets all the keys that can be used to look up a tensor in the final tensor log.

    Args:
        node: Node in question.
        node_index: Index of node in question.
        num_tensors_to_keep: Number of tensors to keep.
        history_dict: Dict of history.

    Returns:
        List of keys that can be used to look up a tensor in the tensor log.
    """
    main_pretty_barcode = rough_barcode_to_final_barcode(node['barcode'], history_dict)
    module_passes_exited = [f"{module}:{pass_num}" for module, pass_num in
                            node['module_passes_exited']]
    lookup_keys = [main_pretty_barcode, node_index, node_index - num_tensors_to_keep]
    if node['layer_type'] != 'output':
        lookup_keys += module_passes_exited[:]
    if node['param_total_passes'] == 1:  # for generality, allow indexing non-recurrent layers w/ pass
        lookup_keys.append(node['layer_label_w_pass'])

    # if just one pass for a module, allow indexing w/out pass
    for module_address in node['modules_exited']:
        if (len(history_dict['module_output_tensors'][module_address]) == 1) and (node['layer_type'] != 'output'):
            lookup_keys.append(module_address)

    # Finally, if buffer tensor, allow buffer address as lookup key.

    if node['is_buffer_tensor']:
        lookup_keys.append(node['buffer_address'])

    return lookup_keys


class TensorLogEntry:
    def __init__(self,
                 rough_barcode: str,
                 node_index: int,
                 num_tensors_to_keep: int,
                 activations_only: bool,
                 history_dict: dict,
                 model_history):
        """Log entry for a single tensor computed in the network.

        Args:
            rough_barcode: Rough barcode for the tensor.
            history_dict: Dictionary of history.
        """
        # TODO: get the mapping of module types, not just the addresses, if not done already.
        orig_tensor_log = history_dict['tensor_log']
        orig_node = orig_tensor_log[rough_barcode]

        # Initial unpacking:

        module_passes_exited = [f"{module}:{pass_num}" for module, pass_num in
                                orig_node['module_passes_exited']]

        # Tensor labeling:
        new_barcode = rough_barcode_to_final_barcode(rough_barcode, history_dict)
        self.layer_barcode = new_barcode
        self.layer_raw_barcode = rough_barcode
        self.layer_raw_tensor_num = orig_node['tensor_num']
        self.layer_label_w_pass = orig_node['layer_label_w_pass']
        self.layer_label_no_pass = orig_node['layer_label']
        self.layer_type = orig_node['layer_type']
        self.num_layer_type_seen_so_far = orig_node['layer_type_ind']
        self.num_layers_total_seen_so_far = orig_node['layer_total_ind']
        self.layer_pass_num = orig_node['pass_num']
        self.layer_passes_total = orig_node['param_total_passes']
        self.num_operations_so_far = orig_node['operation_num_exhaustive']

        # Set the possible lookup keys:

        lookup_keys = get_all_tensor_lookup_keys(orig_node,
                                                 node_index,
                                                 num_tensors_to_keep,
                                                 history_dict)

        self.lookup_keys = sorted(lookup_keys, key=str)

        # Tensor data info:
        self.tensor_shape = orig_node['tensor_shape']
        self.tensor_dtype = orig_node['tensor_dtype']
        self.tensor_fsize = orig_node['tensor_fsize']
        self.tensor_fsize_nice = human_readable_size(orig_node['tensor_fsize'])
        if activations_only:
            self.tensor_contents = orig_node['tensor_contents']
        else:
            self.tensor_contents = 'none'

        # Tensor operation info.
        self.is_input_tensor = orig_node['is_model_input']
        self.is_output_tensor = orig_node['is_model_output']
        self.is_input_descendant = orig_node['has_input_ancestor']
        self.is_output_ancestor = orig_node['is_output_ancestor']
        self.initialized_in_model = orig_node['is_internally_generated']
        self.parent_layers = [rough_barcode_to_final_barcode(barcode, history_dict)
                              for barcode in orig_node['parent_tensor_barcodes']]
        self.child_layers = [rough_barcode_to_final_barcode(barcode, history_dict)
                             for barcode in orig_node['child_tensor_barcodes']]
        self.computed_from_params = orig_node['has_params']
        self.num_param_parent_tensors = len(orig_node['parent_params_shape'])
        self.parent_params_shapes = orig_node['parent_params_shape']
        self.num_params_total = np.sum(
            [np.prod(param_shape) for param_shape in orig_node['parent_params_shape']])
        self.parent_params_fsize = orig_node['params_memory_size']
        self.parent_params_fsize_nice = human_readable_size(orig_node['params_memory_size'])
        if len(orig_node['funcs_applied']) > 0:
            self.func_applied = orig_node['funcs_applied'][0]
            self.func_applied_name = orig_node['funcs_applied_names'][0]
        else:
            self.func_applied = 'none'
            self.func_applied_name = 'none'
        self.func_time_elapsed = orig_node['func_time_elapsed']

        self.num_func_args_total = orig_node['num_args'] + orig_node['num_kwargs']
        self.func_args_non_tensor = orig_node['nontensor_args']
        self.func_kwargs_non_tensor = orig_node['nontensor_kwargs']
        if len(orig_node['gradfuncs_names']) > 0:
            self.gradfunc_name = orig_node['gradfuncs_names'][0]
        else:
            self.gradfunc_name = 'none'

        # Tensor module info:
        if len(orig_node['function_call_modules_nested']) > 0:
            self.containing_origin_module = orig_node['function_call_modules_nested'][-1][0]
        else:
            self.containing_origin_module = 'none'
        self.containing_origin_modules_nested = [mod_tuple[0] for mod_tuple in
                                                 orig_node['function_call_modules_nested']]
        self.is_computed_inside_module = (len(orig_node['function_call_modules_nested']) > 0)
        self.is_module_output = orig_node['is_module_output']
        self.modules_exited = orig_node['modules_exited']
        self.module_passes_exited = module_passes_exited
        self.is_bottom_level_module_output = orig_node['is_bottom_level_module_output']
        if self.is_bottom_level_module_output:
            self.bottom_level_module_exited = orig_node['modules_exited'][0]
        else:
            self.bottom_level_module_exited = 'none'
        self.source_model_history = model_history

    def get_child_layers(self):
        return [self.source_model_history[child_label] for child_label in self.child_layers]

    def get_parent_layers(self):
        return [self.source_model_history[parent_label] for parent_label in self.parent_layers]

    def __str__(self):
        if self.layer_passes_total > 1:
            pass_str = f" (pass {self.layer_pass_num}/{self.layer_passes_total}), "
        else:
            pass_str = ", "
        s = f"Layer {self.layer_label_no_pass}" \
            f"{pass_str}operation {self.num_operations_so_far + 1}/" \
            f"{self.source_model_history.model_num_tensors_total}:"
        s += f"\n\tOutput tensor: shape={self.tensor_shape}, dype={self.tensor_dtype}, size={self.tensor_fsize_nice}"
        if self.tensor_contents != 'none':
            if len(self.tensor_shape) == 0:
                tensor_slice = self.tensor_contents
                num_dims = 0
            elif len(self.tensor_shape) == 1:
                num_dims = min(5, self.tensor_shape[0])
                tensor_slice = self.tensor_contents[0:num_dims]
            elif len(self.tensor_shape) == 2:
                num_dims = min([5, self.tensor_shape[-2], self.tensor_shape[-1]])
                tensor_slice = self.tensor_contents[0:num_dims, 0:num_dims]
            else:
                num_dims = min([5, self.tensor_shape[-2], self.tensor_shape[-1]])
                tensor_slice = self.tensor_contents.data.clone()
                for _ in range(len(self.tensor_shape) - 2):
                    tensor_slice = tensor_slice[0]
                tensor_slice = tensor_slice[0:num_dims, 0:num_dims]
            tensor_slice = tensor_slice.detach()
            tensor_slice.requires_grad = False
            s += f"\n\t\t{str(tensor_slice)}"
            if max(self.tensor_shape) > 5:
                s += '...'
        if not self.is_input_descendant:
            s += f"\n\t(tensor was created de novo inside the model, not computed from input)"
        if not self.is_output_ancestor:
            s += f"\n\t(tensor is not an ancestor of the model output; it terminates within the model)"
        if len(self.parent_params_shapes) > 0:
            params_shapes_str = ', '.join(str(param_shape) for param_shape in self.parent_params_shapes)
            s += f"\n\tParams: Computed from params with shape {params_shapes_str}; {self.num_params_total} params total " \
                 f"({self.parent_params_fsize_nice})"
        else:
            s += f"\n\tParams: no params used"
        if len(self.parent_layers) > 0:
            parent_layers_str = ', '.join(self.parent_layers)
        else:
            parent_layers_str = "no parent layers"
        s += f"\n\tParent Layers: {parent_layers_str}"
        if len(self.child_layers) > 0:
            child_layers_str = ', '.join(self.child_layers)
        else:
            child_layers_str = "no child layers"
        s += f"\n\tChild Layers: {child_layers_str}"
        if self.containing_origin_module == 'none':
            module_str = "\n\tComputed inside module: not computed inside a module"
        else:
            module_str = f"\n\tComputed inside module: {self.containing_origin_module}"
        if not self.is_input_tensor:
            s += f"\n\tFunction: {self.func_applied_name} (gradfunc={self.gradfunc_name}) " \
                 f"{module_str}"
            s += f"\n\tTime elapsed: {self.func_time_elapsed: .3E}s"
        if len(self.modules_exited) > 0:
            modules_exited_str = ', '.join(self.modules_exited)
            s += f"\n\tOutput of modules: {modules_exited_str}"
        else:
            s += f"\n\tOutput of modules: none"
        if self.is_bottom_level_module_output:
            s += f"\n\tOutput of bottom-level module: {self.bottom_level_module_exited}"
        lookup_keys_str = ', '.join([str(key) for key in self.lookup_keys])
        s += f"\n\tLookup keys: {lookup_keys_str}"

        return s

    def __repr__(self):
        return self.__str__()


class ModelHistory:
    def __init__(self,
                 history_dict: dict,
                 activations_only: bool):
        """An object that conveniently stores all the tensor history in easily accessible format.
        This will be how saved activations, and also the full graph without activations, are encoded for the user.
        The internal barcodes are now replaced by the nicely formatted layer labels (including the pass).
        It can be indexed by the layer label, by the module address, or via the topoological sort index
        to pull out entries, each of which is an OrderedDict with the following fields:

        Args:
            history_dict: The history_dict
            activations_only: Whether to only include the nodes with saved activations, or to include all
                nodes and no activations.
        """
        # Crawl through and get the desired tensors:

        orig_tensor_log = history_dict['tensor_log']
        pretty_tensor_log = OrderedDict()
        tensor_list = []  # Ordered list of tensors.
        tensor_mapper_dict = {}  # Mapping for any user index to the appropriate tensor
        layer_labels = []  # list of layer labels without pass numbers
        layer_passes = []  # list of layer labels with pass numbers.
        layer_num_passes = {}  # for each layer, how many total passes it has
        layer_barcodes = []
        module_addresses = []  # list of module addresses without pass numbers
        module_passes = []  # list of module addresses with pass numbers

        model_is_recurrent = False
        model_max_recurrent_loops = 1
        model_is_branching = False

        node_index = 0

        # Get number of tensors to keep:
        if activations_only:
            num_tensors_to_keep = 0
            for tensor_entry in orig_tensor_log.values():
                if all([(tensor_entry['tensor_contents'] is not None),
                        (tensor_entry['tensor_num'] not in history_dict['tensor_nums_to_save_temporarily'])]):
                    num_tensors_to_keep += 1
        else:
            num_tensors_to_keep = len(orig_tensor_log)

        for rough_barcode, node in orig_tensor_log.items():
            new_pretty_node = TensorLogEntry(rough_barcode,
                                             node_index,
                                             num_tensors_to_keep,
                                             activations_only,
                                             history_dict,
                                             self)

            if new_pretty_node.layer_passes_total > model_max_recurrent_loops:
                model_max_recurrent_loops = new_pretty_node.layer_passes_total
                model_is_recurrent = True

            if len(new_pretty_node.child_layers) > 1:
                model_is_branching = True

            # Check whether to keep this entry or not.
            if activations_only and ((node['tensor_contents'] is None) or
                                     ((history_dict['tensor_nums_to_save'] != 'all') and
                                      node['tensor_num'] not in history_dict['tensor_nums_to_save'])):
                continue

            node_index += 1

            # Finally, log it
            pretty_tensor_log[new_pretty_node.layer_barcode] = new_pretty_node
            tensor_list.append(new_pretty_node)
            layer_barcodes.append(new_pretty_node.layer_barcode)
            layer_labels.append(new_pretty_node.layer_label_no_pass)
            layer_passes.append(new_pretty_node.layer_label_w_pass)
            layer_num_passes[new_pretty_node.layer_label_no_pass] = new_pretty_node.layer_passes_total
            for module in new_pretty_node.modules_exited:
                if module not in module_addresses:
                    module_addresses.append(module)
            for module_pass in new_pretty_node.module_passes_exited:
                if module_pass not in module_passes:
                    module_passes.append(module_pass)
                elif new_pretty_node.layer_type != 'output':
                    raise ValueError("There appear to be two overlapping module passes for different layers; "
                                     "check for bugs.")

            for key in new_pretty_node.lookup_keys:
                if key in tensor_mapper_dict:
                    raise ValueError("There appear to be overlapping keys in two layers; check for bugs.")
                tensor_mapper_dict[key] = new_pretty_node

        # Whole-model info.
        self.model_name = history_dict['model_name']
        self.model_is_branching = model_is_branching
        self.model_is_recurrent = model_is_recurrent
        self.model_max_recurrent_loops = model_max_recurrent_loops

        self.model_num_tensors_total = len(history_dict['tensor_log'])
        self.model_tensor_fsize_total = history_dict['total_tensor_fsize']
        self.model_tensor_fsize_total_nice = human_readable_size(history_dict['total_tensor_fsize'])
        self.pass_elapsed_time = history_dict['elapsed_time']
        self.random_seed_used = history_dict['random_seed']
        if activations_only:
            self.model_num_tensors_saved = len(tensor_list)
            self.model_tensor_fsize_saved = np.sum([t.tensor_fsize for t in tensor_list])
            self.model_tensor_fsize_saved_nice = human_readable_size(self.model_tensor_fsize_saved)
        else:
            self.model_num_tensors_saved = 0
            self.model_tensor_fsize_saved = 0
            self.model_tensor_fsize_saved_nice = human_readable_size(self.model_tensor_fsize_saved)

        self.model_total_param_tensors = history_dict['total_param_tensors']
        self.model_total_param_groups = history_dict['total_param_groups']
        self.model_total_params = history_dict['total_params']
        self.model_total_params_fsize = history_dict['total_params_fsize']
        self.model_total_params_fsize_nice = human_readable_size(history_dict['total_params_fsize'])

        # Module info.
        self.model_module_list = list(history_dict['module_dict'].keys())

        # Saved layers info.

        self.input_tensors = [rough_barcode_to_final_barcode(t, history_dict) for t in history_dict['input_tensors']]
        self.output_tensors = [rough_barcode_to_final_barcode(t, history_dict) for t in history_dict['output_tensors']]
        self.internally_generated_tensors = [rough_barcode_to_final_barcode(t, history_dict)
                                             for t in history_dict['internally_generated_tensors']]

        self.enclosing_loop_nodes = OrderedDict()
        for start_node, loop_nodes in history_dict['enclosing_loop_nodes'].items():
            self.enclosing_loop_nodes[rough_barcode_to_final_barcode(start_node, history_dict)] = \
                [rough_barcode_to_final_barcode(n, history_dict) for n in loop_nodes]

        # Finally, the logged tensor information.
        self.tensor_log = pretty_tensor_log
        self.tensor_list = tensor_list
        self.tensor_mapper_dict = tensor_mapper_dict
        self.layer_labels = layer_barcodes
        self.layer_labels_no_pass = layer_labels
        self.layer_labels_w_pass = layer_passes
        self.layer_num_passes = layer_num_passes
        self.module_addresses = module_addresses
        self.module_passes = module_passes
        self.top_level_modules = history_dict['top_level_module_clusters']
        self.module_children = history_dict['module_cluster_children_dict']

        # for each module, how many passes it has
        module_num_passes = {module: len(history_dict['module_output_tensors'][module])
                             for module in self.module_addresses}
        self.module_num_passes = module_num_passes

    def __getitem__(self, ix):
        """
        Overloaded such that entries can be fetched either by their position in the tensor log, their layer label,
        or their module address.
        #it should say so and tell them which labels are valid.
        """
        if ix in self.tensor_mapper_dict:
            return self.tensor_mapper_dict[ix]
        elif (type(ix) == int) and (ix > len(self.tensor_list)):
            raise ValueError(f"You specified the layer with index {ix}, but there are only {len(self.tensor_list)} "
                             f"layers; please specify a smaller number.")
        elif ix in self.module_addresses:
            module_num_passes = self.module_num_passes[ix]
            raise ValueError(f"You specified output of module {ix}, but it has {module_num_passes} passes; "
                             f"please specify e.g. {ix}:2 for the second pass of {ix}.")
        elif ix.split(':')[0] in self.module_addresses:
            module, pass_num = ix.split(':')
            module_num_passes = self.module_num_passes[module]
            raise ValueError(f"You specified module {module} pass {pass_num}, but {module} only has "
                             f"{module_num_passes} passes; specify a lower number.")
        elif ix in self.layer_labels_no_pass:
            layer_num_passes = self.layer_num_passes[ix]
            raise ValueError(f"You specified output of layer {ix}, but it has {layer_num_passes} passes; "
                             f"please specify e.g. {ix}:2 for the second pass of {ix}.")
        elif ix.split(':')[0] in self.layer_labels_no_pass:
            layer_label, pass_num = ix.split(':')
            layer_num_passes = self.layer_num_passes[layer_label]
            raise ValueError(f"You specified layer {layer_label} pass {pass_num}, but {layer_label} only has "
                             f"{layer_num_passes} passes. Specify a lower number.")
        else:
            raise ValueError(self._get_lookup_help_str(ix))

    def __len__(self):
        return len(self.tensor_list)

    def __iter__(self):
        """
        Returns the entries in their topological sort order.
        """
        return iter(self.tensor_list)

    def __str__(self):
        s = f"Log of {self.model_name} forward pass:"
        if self.model_is_branching:
            branch_str = "with branching"
        else:
            branch_str = 'without branching'
        if self.model_is_recurrent:
            s += f"\n\tModel structure: recurrent (at most {self.model_max_recurrent_loops} loops), {branch_str}; " \
                 f"{len(self.module_addresses)} total modules."
        else:
            s += f"\n\tModel structure: purely feedforward, {branch_str}; {len(self.module_addresses)} total modules."
        s += f"\n\t{self.model_num_tensors_total} tensors ({self.model_tensor_fsize_total_nice}) computed in forward pass; " \
             f"{self.model_num_tensors_saved} tensors ({self.model_tensor_fsize_saved_nice}) saved."
        s += f"\n\t{self.model_total_param_tensors} parameter operations ({self.model_total_params} params total; " \
             f"{self.model_total_params_fsize_nice})."
        s += f"\n\tRandom seed: {self.random_seed_used}"
        s += f"\n\tTime elapsed: {np.round(self.pass_elapsed_time, 3)}s"

        # Print the module hierarchy.
        s += f"\n\tModule Hierarchy:"
        s += self._module_hierarchy_str()

        # Now print all layers.
        s += f"\n\tLayers:"
        for l, layer_barcode in enumerate(self.layer_labels):
            pass_num = self.tensor_log[layer_barcode].layer_pass_num
            total_passes = self.tensor_log[layer_barcode].layer_passes_total
            if total_passes > 1:
                pass_str = f" ({pass_num}/{total_passes} passes)"
            else:
                pass_str = ''
            s += f"\n\t\t{l}: {layer_barcode} {pass_str}"

        return s

    def __repr__(self):
        return self.__str__()

    def _module_hierarchy_str(self):
        """
        Utility function to print the nested module hierarchy.
        """
        s = ''
        for module in self.top_level_modules:
            s += f"\n\t\t{module[0]}"
            if len(self.module_children[module]) > 0:
                s += ':'
            s += self._module_hierarchy_str_helper(module, 1)
        return s

    def _module_hierarchy_str_helper(self, module, level):
        """
        Helper function for _module_hierarchy_str.
        """
        s = ''
        any_grandchild_modules = any([len(self.module_children[sub_module]) > 0
                                      for sub_module in self.module_children[module]])
        if any_grandchild_modules or len(self.module_children[module]) == 0:
            for sub_module in self.module_children[module]:
                s += f"\n\t\t{'    ' * level}{sub_module[0]}"
                if len(self.module_children[sub_module]) == 0:
                    s += ':'
                s += self._module_hierarchy_str_helper(sub_module, level + 1)
        else:
            s += self.pretty_print_list_w_line_breaks(
                [module_child[0] for module_child in self.module_children[module]],
                line_break_every=8,
                indent_chars=f"\t\t{'    ' * level}")
        return s

    @staticmethod
    def pretty_print_list_w_line_breaks(lst, indent_chars: str, line_break_every=5):
        """
        Utility function to pretty print a list with line breaks, adding indent_chars every line.
        """
        s = f'\n{indent_chars}'
        for i, item in enumerate(lst):
            s += f"{item}"
            if i < len(lst) - 1:
                s += ', '
            if ((i + 1) % line_break_every == 0) and (i < len(lst) - 1):
                s += f'\n{indent_chars}'
        return s

    def _get_lookup_help_str(self, layer_label):
        """Generates a help string to be used in error messages when indexing fails.
        """
        sample_layer1 = random.choice(self.layer_labels_w_pass)
        sample_layer2 = random.choice(self.layer_labels_no_pass)
        if len(self.module_addresses) > 0:
            sample_module1 = random.choice(self.module_addresses)
            sample_module2 = random.choice(self.module_passes)
        else:
            sample_module1 = 'features.3'
            sample_module2 = 'features.3:2'
        module_str = f"(e.g., {sample_module1}, {sample_module2})"
        help_str = (f"Layer {layer_label} not recognized; please specify either \n\t1) an integer giving "
                    f"the ordinal position of the layer, \n\t2) the layer label (e.g., {sample_layer1}, "
                    f"{sample_layer2}), \n\t3) the module address {module_str}"
                    f"\n\t4) A substring of any desired layer labels (e.g., 'pool' will grab all maxpool2d "
                    f"or avgpool2d layers, 'maxpool' with grab all 'maxpool2d' layers, etc.)."
                    f"\n(Label meaning: conv2d_3_4:2 means the second pass of the third convolutional layer,"
                    f"and fourth layer overall in the model.)")
        return help_str

    def get_op_nums_from_user_labels(self, which_layers):
        """Given list of user layer labels, returns the original tensor numbers for those labels (i.e.,
        the numbers that were generated on the fly during the forward pass, such that they can be
        saved on a subsequent pass). Raises an error if the user's labels don't correspond to any layers.

        Args:
            which_layers: List of layers to include, using any indexing desired: either the layer label,
            the module label, or the ordinal position of the layer. If a layer has multiple passes and
            none is specified, will return all of them.

        Returns:
            Ordered, unique list of raw tensor numbers associated with the specified layers.
        """
        raw_tensor_nums_to_save = set()
        num_layers = len(self.tensor_list)
        for layer_key in which_layers:
            if type(layer_key) == int:  # if user specifies ordinal position
                if not -num_layers <= layer_key < num_layers:
                    raise ValueError(f"You specified the {layer_key}th layer, but there are only "
                                     f"{num_layers} layers in the model.")
                raw_tensor_nums_to_save.add(self[layer_key].layer_raw_tensor_num)
            elif layer_key in self.layer_labels:  # if it's a primary layer key just grab it
                raw_tensor_nums_to_save.add(self[layer_key].layer_raw_tensor_num)
            elif ':' in layer_key:  # if specific pass given, either add or complain if there aren't that many passes
                label, pass_num = layer_key.split(':')
                if (layer_key in self.layer_labels_w_pass) or (layer_key in self.module_passes):
                    raw_tensor_nums_to_save.add(self[layer_key].layer_raw_tensor_num)
                elif label in self.layer_labels_no_pass:
                    first_pass_address = f"{label}:1"
                    raise ValueError(f"You specified {label} pass #{pass_num}, but there are only "
                                     f"{self[first_pass_address].layer_passes_total} passes in {label}; "
                                     f"please specify a pass in range 1-{self[first_pass_address].layer_passes_total}.")
                elif label in self.module_addresses:
                    raise ValueError(f"You specified {label} pass #{pass_num}, but there are only "
                                     f"{self.module_num_passes[label]} passes in {label}; "
                                     f"please specify a pass in range 1-{self.module_num_passes[label]}.")
                else:
                    raise ValueError(self._get_lookup_help_str(label))
            elif layer_key in self.layer_labels_no_pass:  # if it's a layer address, add all passes of the layer
                for layer_label_w_pass in self.layer_labels_w_pass:
                    if layer_label_w_pass.startswith(f"{layer_key}:"):
                        raw_tensor_nums_to_save.add(self[layer_label_w_pass].layer_raw_tensor_num)
            elif layer_key in self.module_addresses:  # if it's a module address, add all passes
                for pass_num in range(1, self.module_num_passes[layer_key] + 1):
                    raw_tensor_nums_to_save.add(self[f"{layer_key}:{pass_num}"].layer_raw_tensor_num)
            elif type(layer_key) == str:  # as last resort check if any layer labels begin with the provided substring
                found_a_match = False
                for layer in self:
                    if layer_key in layer.layer_label_w_pass:
                        raw_tensor_nums_to_save.add(layer.layer_raw_tensor_num)
                        found_a_match = True
                if not found_a_match:
                    raise ValueError(self._get_lookup_help_str(layer_key))

        raw_tensor_nums_to_save = sorted(list(raw_tensor_nums_to_save))
        # Check for any identity functions; if so, add their parent tensor to the list, and flag parent
        # tensor not to be saved if applicable. #TODO: refactor identity stuff and get rid of this nonsense.
        raw_tensor_nums_to_save_temporarily = set()
        for node in self:
            if (node.layer_raw_tensor_num in raw_tensor_nums_to_save) and node.func_applied_name.lower() == 'identity':
                node_parent = node.get_parent_layers()[0]
                if node_parent.layer_raw_tensor_num not in raw_tensor_nums_to_save:
                    raw_tensor_nums_to_save_temporarily.add(node_parent.layer_raw_tensor_num)

        return raw_tensor_nums_to_save, raw_tensor_nums_to_save_temporarily
