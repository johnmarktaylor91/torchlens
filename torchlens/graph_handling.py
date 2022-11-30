# This module has functions for processing the computation graphs that come out of the other functions.

import random
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Tuple, Union

import graphviz
import numpy as np
import torch

from torchlens.helper_funcs import human_readable_size, make_barcode
from torchlens.tensor_tracking import safe_copy

graphviz.set_jupyter_format('png')


def annihilate_node(node_barcode: Dict, history_dict: Dict):
    """Removes a node from the graph, removes it as a parent and as a child from any connected nodes, and removes
    it from all other node lists.

    Args:
        node_barcode: Barcode of the node to annihilate..
        history_dict: The dictionary of history.

    Returns:
        The history_dict with the given node completely annihilated from all traces of the graph.
    """
    tensor_log = history_dict['tensor_log']
    node = tensor_log[node_barcode]
    for child_barcode in node['child_tensor_barcodes']:
        child_node = tensor_log[child_barcode]
        child_node['parent_tensor_barcodes'].remove(node_barcode)
    for parent_barcode in node['parent_tensor_barcodes']:
        parent_node = tensor_log[parent_barcode]
        parent_node['child_tensor_barcodes'].remove(node_barcode)
    for field_name, field in history_dict.items():
        if (type(field) in [list, tuple]) and node_barcode in field:
            field.remove(node_barcode)


def insert_sorted_child_node(parent_node, child_node, tensor_log):
    """Utility function to insert a child node into sorted order into a parent node's child list,
    ordering based on the operation order.

    Args:
        parent_node: The parent node.
        child_node: Child node to insert.
        tensor_log: The tensor log.
    """
    orig_len = len(tensor_log)
    child_barcode = child_node['barcode']
    if 'child_tensor_barcodes' not in parent_node:
        parent_node['child_tensor_barcodes'] = [child_barcode]
        return
    other_child_nodes = [tensor_log[other_child_barcode] for other_child_barcode in
                         parent_node['child_tensor_barcodes']]
    child_barcodes = parent_node['child_tensor_barcodes']
    for c, other_child_node in enumerate(other_child_nodes):
        if child_node['tensor_num'] < other_child_node['tensor_num']:
            child_barcodes = child_barcodes[0:] + [child_barcode] + child_barcodes[c:]
            parent_node['child_tensor_barcodes'] = child_barcodes[:]
            return
    parent_node['child_tensor_barcodes'] = parent_node['child_tensor_barcodes'] + [child_barcode]
    if len(tensor_log) != orig_len:
        print("STOP")
        abc = 1 + 1


def annotate_node_children(history_dict: Dict) -> Dict:
    """Annotates each node with the addresses of its child nodes, keeping them in sorted order of their
    operation number to make sure they are rendered in the right execution order.

    Args:
        history_dict: dictionary with all the data

    Returns:
        Tensor log annotated with the child node addresses.
    """
    tensor_log = history_dict['tensor_log']
    orig_len = len(tensor_log)

    for barcode, node in tensor_log.items():
        if len(node['parent_tensor_barcodes']) == 0:
            continue
        for parent_barcode in node['parent_tensor_barcodes']:
            parent_node = tensor_log[parent_barcode]
            insert_sorted_child_node(parent_node, node, tensor_log)
    for barcode, node in tensor_log.items():
        if 'child_tensor_barcodes' not in node:
            node['child_tensor_barcodes'] = []
    return history_dict


def expand_multiple_functions(history_dict: Dict) -> Dict:
    """For nodes that have had multiple functions applied to them (e.g., the identity function), expands
    them to multiple nodes for fidelity to the graph. If the functions have the same name, then only use the one node.
    #TODO: Refactor this nonsense.

    Args:
        history_dict: the history dict

    Returns:
        Tensor log with the multiple functions expanded where applicable.
    """
    tensor_log = history_dict['tensor_log']

    node_stack = list(tensor_log.values())
    new_tensor_log = OrderedDict()

    # Go through each node and expand the nodes that have multiple, non-identically-named functions.

    while len(node_stack) > 0:
        node = node_stack.pop(0)
        new_tensor_log[node['barcode']] = node
        num_funcs = len(node['funcs_applied'])
        if num_funcs < 2:
            continue

        # Go through the list of functions; if any has a new name, then make a new node for it, and add that one to the stack.

        func_names_seen = []
        for f, func_name, func in zip(range(num_funcs), node['funcs_applied_names'], node['funcs_applied']):
            if f == 0:
                func_names_seen.append(func_name)
                continue
            if func_name not in func_names_seen:
                func_applied_module = node['funcs_applied_modules'][f]
                func_applied_module_position = node['modules_exited'].index(func_applied_module)

                # if it's a new function, make a new node for it and modify fields as needed,
                # for the old node, for its original children, and for the new node inserted between the two.

                # New node:
                history_dict['tensor_counter'] += 1

                new_node = node.copy()
                new_node['barcode'] = make_barcode()
                new_node['layer_barcode'] = new_node['barcode']
                new_node['funcs_applied'] = node['funcs_applied'][f:]
                new_node['funcs_applied_names'] = node['funcs_applied_names'][f:]
                new_node['function_call_modules_nested_multfuncs'] = node['function_call_modules_nested_multfuncs'][f:]
                new_node['function_call_modules_nested'] = node['function_call_modules_nested_multfuncs'][f]
                new_node['modules_exited'] = node['modules_exited'][func_applied_module_position:]
                new_node['module_passes_exited'] = node['module_passes_exited'][func_applied_module_position:]
                new_node['child_tensor_barcodes'] = node['child_tensor_barcodes'][:]
                new_node['parent_tensor_barcodes'] = [node['barcode']]
                new_node['parent_tensor_arg_locs'] = {'args': {0: node['barcode']},
                                                      'kwargs': {}}
                new_node['args_all'] = [safe_copy(tensor_log[new_node['parent_tensor_barcodes'][0]]['tensor_contents'])]
                new_node['creation_args'] = [
                    safe_copy(tensor_log[new_node['parent_tensor_barcodes'][0]]['tensor_contents'])]
                new_node['creation_kwargs'] = {}
                new_node['tensor_num'] = history_dict['tensor_counter']
                new_node['has_params'] = False
                new_node['parent_params'] = []
                new_node['parent_params_shape'] = []
                new_node['parent_param_passes'] = {}
                new_node['params_memory_size'] = 0
                if node['is_model_output']:
                    new_node['is_model_output'] = True
                    history_dict['output_tensors'].remove(node['barcode'])
                    history_dict['output_tensors'].append(new_node['barcode'])

                # Tweak parent barcodes of child nodes to point to the new node.

                for child_barcode in new_node['child_tensor_barcodes']:
                    child_node = tensor_log[child_barcode]
                    for p, child_parent_barcode in enumerate(child_node['parent_tensor_barcodes']):
                        if child_parent_barcode == node['barcode']:
                            child_node['parent_tensor_barcodes'][p] = new_node['barcode']
                    for arg_position, arg_barcode in child_node['parent_tensor_arg_locs']['args'].items():
                        if arg_barcode == node['barcode']:
                            child_node['parent_tensor_arg_locs']['args'][arg_position] = new_node['barcode']
                    for kwarg_key, kwarg_barcode in child_node['parent_tensor_arg_locs']['kwargs'].items():
                        if kwarg_barcode == node['barcode']:
                            child_node['parent_tensor_arg_locs']['kwargs'][kwarg_key] = new_node['barcode']

                # And finally fix the original node.

                node['funcs_applied'] = node['funcs_applied'][:f]
                node['funcs_applied_names'] = node['funcs_applied_names'][:f]
                node['modules_exited'] = node['modules_exited'][:func_applied_module_position]
                node['module_passes_exited'] = node['module_passes_exited'][:func_applied_module_position]
                node['function_call_modules_nested'] = node['function_call_modules_nested_multfuncs'][f - 1]
                node['function_call_modules_nested_multfuncs'] = node['function_call_modules_nested_multfuncs'][:f]
                node['child_tensor_barcodes'] = [new_node['barcode']]
                node['is_model_output'] = False

                if ((len(node['modules_exited']) > 0)
                        and history_dict['module_dict'][node['modules_exited'][-1]].tl_is_bottom_level_module):
                    node['is_bottom_level_module_output'] = True
                else:
                    node['is_bottom_level_module_output'] = False

                node_stack = [new_node] + node_stack
                break

    history_dict['tensor_log'] = new_tensor_log
    return history_dict


def get_all_connected_nodes(tensor_log: Dict,
                            starting_node_barcodes: List[str],
                            mode: str = 'parents_and_children') -> set:
    """Returns set of all nodes connected somehow to any of the starting nodes, using a flooding algorithm.

    Args:
        tensor_log: Log of the tensors.
        starting_node_barcodes: List of barcodes of the starting nodes.
        mode: Either 'parents', 'children', or 'parents_and_children'; i.e., what kind of node
            neighbors to check for.

    Returns:
        Set of all nodes with any connection to the starting nodes.
    """
    node_stack = [tensor_log[barcode] for barcode in starting_node_barcodes]
    connected_nodes = set()

    while len(node_stack) > 0:
        node = node_stack.pop()
        connected_nodes.add(node['barcode'])
        if 'children' in mode:
            for child_barcode in node['child_tensor_barcodes']:
                if child_barcode not in connected_nodes:
                    node_stack.append(tensor_log[child_barcode])
        if 'parents' in mode:
            for parent_barcode in node['parent_tensor_barcodes']:
                if parent_barcode not in connected_nodes:
                    node_stack.append(tensor_log[parent_barcode])

    return connected_nodes


def strip_irrelevant_nodes(history_dict: Dict) -> Dict:
    """Strip irrelevant nodes, keeping nodes that are connected somehow to either the inputs or the outputs
    of the graph. Also removes any non-output, non-numeric leaf nodes since these are nearly always just
    control logic and not actual activations.

    Args:
        history_dict: input history dict

    Returns:
        history_dict with irrelevant nodes stripped
    """
    tensor_log = history_dict['tensor_log']

    input_connected_nodes = get_all_connected_nodes(tensor_log, history_dict['input_tensors'])
    output_connected_nodes = get_all_connected_nodes(tensor_log, history_dict['output_tensors'])

    relevant_tensors = input_connected_nodes.union(output_connected_nodes)
    new_tensor_log = OrderedDict()

    for barcode, node in tensor_log.items():
        if barcode in relevant_tensors:
            new_tensor_log[barcode] = node
        else:
            annihilate_node(barcode, history_dict)

    history_dict['tensor_log'] = new_tensor_log
    return history_dict


def mark_output_ancestors(history_dict: Dict):
    """Marks nodes that are ancestors of an output.

    Args:
        history_dict: Dict of the history.

    Returns:
        Nothing, but the nodes are now marked whether they're ancestors of an output.
    """
    tensor_log = history_dict['tensor_log']
    output_ancestor_nodes = get_all_connected_nodes(tensor_log, history_dict['output_tensors'], 'parents')
    for node_barcode, node in tensor_log.items():
        # Mark whether ancestor of an output.
        if node_barcode in output_ancestor_nodes:
            node['is_output_ancestor'] = True
        else:
            node['is_output_ancestor'] = False

        # Mark whether both ancestor of an output and has an input ancestor.
        if node['is_output_ancestor'] and node['has_input_ancestor']:
            node['connects_input_and_output'] = True
        else:
            node['connects_input_and_output'] = False
    return history_dict


def add_output_nodes(history_dict: Dict):
    """Adds explicit output nodes

    Args:
        history_dict: The history dict.

    Returns:
        Nothing, but new nodes are added for the output layers.
    """
    tensor_log = history_dict['tensor_log']

    # For each current output layer, unmark it as an output, clone it, and add a child.

    new_output_tensors = []

    for b, output_node_barcode in enumerate(history_dict['output_tensors']):
        history_dict['tensor_counter'] += 1
        output_node = tensor_log[output_node_barcode]
        new_output_node = output_node.copy()
        new_output_node['barcode'] = make_barcode()
        new_output_node['layer_barcode'] = make_barcode()
        new_output_node['has_params'] = False
        new_output_node['parent_params'] = new_output_node['parent_params_shape'] = []
        new_output_node['parent_params_passes'] = {}
        new_output_node['tensor_num'] = history_dict['tensor_counter']
        new_output_node['is_bottom_level_module_output'] = False

        # Change original output node:

        output_node['is_model_output'] = False
        output_node['child_tensor_barcodes'] = [new_output_node['barcode']]

        # And now tweak the new output node.

        new_output_node['parent_tensor_barcodes'] = [output_node['barcode']]
        new_output_node['is_model_output'] = True
        new_output_node['function_call_modules_nested'] = []
        new_output_node['modules_exited'] = []

        tensor_log[new_output_node['barcode']] = new_output_node
        new_output_tensors.append(new_output_node['barcode'])

    history_dict['output_tensors'] = new_output_tensors
    history_dict['tensor_log'] = tensor_log
    return history_dict


def get_internal_ancestors_for_remaining_parents(converge_nodes: List[str],
                                                 internal_ancestors_added: set,
                                                 nodes_seen: set,
                                                 tensor_log: Dict):
    """Checks all the converge nodes, and if any have all of their children computed except for internal ancestors,
    then add those internal ancestors to the stack if they haven't been added yet.

    Args:
        converge_nodes: List of convergent nodes.
        internal_ancestors_added: Set of internal ancestors seen so far
        nodes_seen: set of nodes seen so far
        tensor_log: The tensor log.

    Returns:
        List of barcodes of internal ancestors to be added.
    """
    ancestors_to_add = []

    for converge_node_barcode in converge_nodes:
        converge_node = tensor_log[converge_node_barcode]
        unseen_parents = [parent for parent in converge_node['parent_tensor_barcodes'] if parent not in nodes_seen]
        if all([parent in converge_node['parent_internal_tensor_barcodes'] for parent in unseen_parents]):
            for ancestor in converge_node['internal_ancestors']:
                if ancestor not in internal_ancestors_added:
                    ancestors_to_add.append(ancestor)
                    internal_ancestors_added.add(ancestor)

    return ancestors_to_add


def get_next_node(node_stack: List,
                  converge_nodes: List,
                  tensor_log: Dict,
                  ordered_tensor_log: Dict) -> Dict:
    """Utility function for the topological sort function; gets the next node to look at. This will
    either be the first convergent node with all its parents computed, or if not, the next node in the stack.

    Args:
        node_stack: Stack of non-converge nodes.
        converge_nodes: Stack of converge nodes.
        tensor_log: The tensor log.
        ordered_tensor_log: The new tensor log (here used just to record which nodes have been computed).

    Returns:
        The next node to look at.
    """
    found_converge = False
    for c, converge_node_barcode in enumerate(converge_nodes):
        converge_node = tensor_log[converge_node_barcode]
        if all([parent_node in ordered_tensor_log for parent_node in converge_node['parent_tensor_barcodes']]):
            node = tensor_log[converge_nodes.pop(c)]
            found_converge = True

    if not found_converge:
        node = tensor_log[node_stack.pop(0)]

    return node


def topological_sort_nodes(history_dict: Dict) -> Dict:
    """Given the tensor log, applies a topological sort to the graph, annotating the nodes with their sorted position.
    The additional constraint is applied that for parallel branches, ordering reflects how many computational
    steps have been applied since the branching began (i.e., nodes that are more computational steps along
    have a higher sorted value); the convergence point of the graph is not added until its children are.

    Args:
        history_dict: input history dict

    Returns:
        tensor_log where the nodes are annotated with their topologically sorted order.
    """
    tensor_log = history_dict['tensor_log']

    node_stack = history_dict['input_tensors'][:]
    nodes_seen = set()
    internal_ancestors_added = set()
    converge_nodes = []
    node_count = 0
    module_node_count = 0
    bottom_module_node_count = 0
    ordered_tensor_log = OrderedDict({})  # re-order the tensor log in topological order.
    while len(node_stack) > 0 or len(converge_nodes) > 0:
        # Check for any converge nodes in the stack and move them over if so.
        for node_barcode in node_stack:
            node = tensor_log[node_barcode]
            if not all([parent_node in ordered_tensor_log for parent_node in node['parent_tensor_barcodes']]):
                converge_nodes.append(node['barcode'])
                node_stack.remove(node['barcode'])

        # For any converge nodes, check if they have all parents computed except for internally generated ones;
        # if so, add these to the stack.

        ancestors_to_add = get_internal_ancestors_for_remaining_parents(converge_nodes, internal_ancestors_added,
                                                                        nodes_seen, tensor_log)
        node_stack.extend(ancestors_to_add)

        # Get the next node: either the first converge node with all parents computed, or the first node in the stack.

        node = get_next_node(node_stack, converge_nodes, tensor_log, ordered_tensor_log)

        if node['barcode'] in ordered_tensor_log:
            continue

        node['operation_num_exhaustive'] = node_count
        node_count += 1
        ordered_tensor_log[node['barcode']] = node

        if node['is_module_output']:  # if it's a module output, also annotate the order for that.
            node['operation_num_module'] = module_node_count
            module_node_count += 1
        else:
            node['operation_num_module'] = None

        if node['is_bottom_level_module_output']:  # if it's a module output, also annotate the order for that.
            bottom_module_node_count += 1
            node['operation_num_bottom_module'] = bottom_module_node_count
        else:
            node['operation_num_bottom_module'] = None

        nodes_seen.add(node['barcode'])
        node_stack.extend(node['child_tensor_barcodes'])

    # Get the final output node and tag it as the final node, this is used for plotting purposes.
    for n, node in enumerate(ordered_tensor_log.values()):
        if n == len(ordered_tensor_log) - 1:
            node['is_last_output_layer'] = True
        else:
            node['is_last_output_layer'] = False

    history_dict['tensor_log'] = ordered_tensor_log
    return history_dict


def annotate_total_layer_passes(history_dict: Dict) -> Dict:
    """Annotates tensors in the log with the total number of passes their layer goes through in the network.

    Args:
        history_dict: history_dict

    Returns:
        history_dict with tensors tagged with the total number of passes
    """
    tensor_log = history_dict['tensor_log']
    param_group_tensors = history_dict['param_group_tensors']
    module_tensors = history_dict['bottom_level_module_output_tensors']
    for node_barcode, node in tensor_log.items():  # Default to None, replace any as needed.
        node['param_total_passes'] = 1
        node['module_total_passes'] = 1
    for param_group_barcode, tensors in param_group_tensors.items():
        param_group_num_passes = len(tensors)
        for tensor_barcode in tensors:
            tensor_log[tensor_barcode]['param_total_passes'] = param_group_num_passes
    for module_barcode, tensors in module_tensors.items():
        module_num_passes = len(tensors)
        for tensor_barcode in tensors:
            tensor_log[tensor_barcode]['module_total_passes'] = module_num_passes

    return history_dict


def find_loops(history_dict) -> List[List]:
    """Utility function to find all loops in the graph; returns list of lists of all loops.
    This should never actually happen, so it's for debugging purposes.

    Args:
        history_dict: Dict with the history.

    Returns:
        List of lists of all loops, listed as the tensor barcodes.
    """

    tensor_log = history_dict['tensor_log']
    input_nodes = history_dict['input_tensors']

    node_stack = [(input_node, [input_node]) for input_node in input_nodes]  # stack annotated with any top-level loops.
    loops = []
    num_passes = 0
    while len(node_stack) > 0:
        num_passes += 1
        node_barcode, nodes_seen = node_stack.pop()
        node = tensor_log[node_barcode]
        node_children = node['child_tensor_barcodes']
        for node_child_barcode in node_children:
            if node_child_barcode in nodes_seen:
                loop = nodes_seen + [node_child_barcode]
                loops.append(loop)
                continue
            node_child = tensor_log[node_child_barcode]
            new_nodes_seen = nodes_seen + [node_child]
            node_stack.append((node_child_barcode, new_nodes_seen))
    print(num_passes)
    return loops


def find_outermost_loops(history_dict: Dict) -> Dict:
    """Utility function that tags the top-level loops in the graph. Specifically, it goes until it finds a 
    node that repeats, then gets all isntances of that repeated node, then finds the next one and does the same, etc.
    Returns a dictionary where each key is the layer barcode, and each value is the list of tensor
    barcodes for that layer (i.e., the barcode of the node corresponding to each pass).
    
    Args:
        history_dict: Input history dict 

    Returns:
        Dictionary of repeated occurrences of each layer corresponding to the outermost recurrent loops.
    """
    tensor_log = history_dict['tensor_log']
    input_nodes = history_dict['input_tensors']

    node_stack = [(input_node, None) for input_node in input_nodes]  # stack annotated with any top-level loops.
    enclosing_loop_nodes = defaultdict(lambda: [])

    while len(node_stack) > 0:
        node_tuple = node_stack.pop(0)
        node_barcode, enclosing_loop_node = node_tuple
        node = tensor_log[node_barcode]

        if all([node['has_params'], node['param_total_passes'] > 1, node['pass_num'] == 1,
                enclosing_loop_node is None]):
            #  We've hit the start of an enclosing loop, make that loop the label.
            children_enclosing_loop = node['layer_barcode']
            if node['layer_barcode'] not in enclosing_loop_nodes:
                enclosing_loop_nodes[node['layer_barcode']].append(node)
        elif all([node['layer_barcode'] == enclosing_loop_node,
                  node['pass_num'] < node['param_total_passes'],
                  node['layer_barcode'] in enclosing_loop_nodes]):
            # We've hit another occurrence of the enclosing loop.
            children_enclosing_loop = enclosing_loop_node
            if node not in enclosing_loop_nodes[node['layer_barcode']]:
                enclosing_loop_nodes[node['layer_barcode']].append(node)
        elif all([node['layer_barcode'] == enclosing_loop_node, node['pass_num'] == node['param_total_passes']]):
            # We've hit the end of an enclosing loop, remove that as the label.
            children_enclosing_loop = None
            if node not in enclosing_loop_nodes[node['layer_barcode']]:
                enclosing_loop_nodes[node['layer_barcode']].append(node)
        else:  # We're inside or outside an enclosing loop.
            children_enclosing_loop = enclosing_loop_node

        nodes_to_add = [(child_node, children_enclosing_loop) for child_node in node['child_tensor_barcodes']]
        node_stack.extend(nodes_to_add)

    return enclosing_loop_nodes


def flex_equal(a, b):
    """
    If a==b returns just one value return that, if it returns an array call "all" on it.
    """
    eq_check = a == b
    if type(eq_check) in [np.ndarray, torch.Tensor, torch.nn.Parameter]:
        return eq_check.all()
    elif eq_check in [True, False]:
        return eq_check
    else:
        return all(eq_check)


def are_args_equal(args_list1: List[Any], args_list2: List[Any]):
    """Checks whether two lists of arguments are equal. If tensors, stipulates those tensors have equal values.

    Args:
        args_list1: List of arguments.
        args_list2: List of arguments.

    Returns:
        Whether the arguments are equal or not.
    """
    for arg1, arg2 in zip(args_list1, args_list2):
        if not flex_equal(arg1, arg2):
            return False
    return True


def do_two_pass_items_match(pass1_start_barcode: str,
                            pass1_nth_item: Dict,
                            pass2_start_barcode: str,
                            pass2_nth_item: Dict,
                            diverged_pass_pairs: set) -> bool:
    """Utility function to check if corresponding items in two passes match: to match, the
    passes must not have already diverged, and the nodes must have the same functions.

    Args:
        pass1_start_barcode: Barcode for the start of the first pass being examined.
        pass1_nth_item: Nth item in the stack for the first pass being examined.
        pass2_start_barcode: Barcode for the start of the second pass being examined.
        pass2_nth_item: Nth item in the stack for the second pass being examined.
        diverged_pass_pairs: Pairs of passes that have already diverged.

    Returns:
        Whether the items for the two passes match.
    """
    if any([pass1_nth_item is None, pass2_nth_item is None]):
        return False

    # First check if they've diverged.
    pass_pair_key = tuple(sorted([pass1_start_barcode, pass2_start_barcode]))
    if pass_pair_key in diverged_pass_pairs:  # they've already diverged
        return False
    pass1_funcs = pass1_nth_item['funcs_applied']
    pass2_funcs = pass2_nth_item['funcs_applied']
    pass1_args = pass1_nth_item['nontensor_all_args']
    pass2_args = pass2_nth_item['nontensor_all_args']
    if any([(pass1_funcs != pass2_funcs), not are_args_equal(pass1_args, pass2_args),
            pass1_nth_item['parent_param_barcodes'] != pass2_nth_item[
                'parent_param_barcodes']]):  # the two passes aren't the same.
        diverged_pass_pairs.add(pass_pair_key)
        return False
    return True


def group_matching_layers(nth_item_per_stack: Dict,
                          diverged_pass_pairs: set,
                          tensor_log: Dict) -> Dict[str, List[Dict]]:
    """Utility function that takes in the nth item per stack, and the mapping of passes that have already diverged,
    and groups together corresponding layers.

    Args:
        nth_item_per_stack: Dict mapping the barcode for the start of each pass to the nth item in the stack
            for that pass.
        diverged_pass_pairs: Set of passes that have already diverged.
        tensor_log: The tensor log.

    Returns:
        Dictionary mapping the layer name to all tensor barcodes for that layer.
    """
    # these will map each node to the set of nodes it's the same layer as (indexed by the barcode of the
    # first node).
    same_layer_barcode_mapper = {}  # maps each node to the layer group it's part of
    same_layer_barcode_lists = defaultdict(list)  # list of layers for each group

    for p1, (pass1_start_barcode, pass1_nth_item_barcode) in enumerate(nth_item_per_stack.items()):
        for p2, (pass2_start_barcode, pass2_nth_item_barcode) in enumerate(nth_item_per_stack.items()):
            if any([pass1_nth_item_barcode is None, pass2_nth_item_barcode is None]):
                continue
            pass1_nth_item = tensor_log[pass1_nth_item_barcode]
            pass2_nth_item = tensor_log[pass2_nth_item_barcode]
            if p2 <= p1:
                continue

            if not do_two_pass_items_match(pass1_start_barcode,
                                           pass1_nth_item,
                                           pass2_start_barcode,
                                           pass2_nth_item,
                                           diverged_pass_pairs):
                continue

            if pass1_nth_item['barcode'] not in same_layer_barcode_mapper:
                layer_barcode = pass1_nth_item['barcode']
                same_layer_barcode_mapper[layer_barcode] = layer_barcode
                same_layer_barcode_lists[layer_barcode].append(layer_barcode)
            else:
                layer_barcode = same_layer_barcode_mapper[pass1_nth_item['barcode']]

            same_layer_barcode_mapper[pass2_nth_item['barcode']] = layer_barcode

            if pass2_nth_item['barcode'] not in same_layer_barcode_lists[layer_barcode]:
                same_layer_barcode_lists[layer_barcode].append(pass2_nth_item['barcode'])

    return same_layer_barcode_lists


def mark_corresponding_pass_layers(nth_item_per_stack: Dict, diverged_pass_pairs, tensor_log):
    """

    Args:
        nth_item_per_stack: Dictionary where each key is the starting barcode for the pass,
            and the value is the nth item of the stack for that pass.
        diverged_pass_pairs: Pairs of passes that have already diverged.
        tensor_log: Log of tensors

    Returns:
        Nothing
    """
    # Now we can go through and assign nodes as the same layer if they have the same functions, and
    # if the passes haven't already diverged.

    # Check if it's a param node, if so then skip it (because it has its own numbering)

    same_layer_barcode_lists = group_matching_layers(nth_item_per_stack, diverged_pass_pairs, tensor_log)

    # Now we have which of the nodes in each pass correspond and can mark them as such.

    for layer_barcode, node_list in same_layer_barcode_lists.items():
        num_passes = len(node_list)
        for p, node_barcode in enumerate(node_list):
            node = tensor_log[node_barcode]
            if node['has_params']:
                continue
            node['layer_barcode'] = layer_barcode
            node['pass_num'] = p + 1
            node['param_total_passes'] = num_passes


def update_pass_stack(pass_barcode: str,
                      pass_stack: List[Dict],
                      pass_stacks: Dict[str, List[Dict]],
                      new_pass_stacks: Dict,
                      nodes_seen: set,
                      first_stack: bool,
                      tensor_log: dict):
    """Utility function to update a single pass stack.

    Args:
        pass_barcode: Barcode for the start of the pass.
        pass_stack: The original stack for the pass.
        pass_stacks: All pass stacks.
        new_pass_stacks: Dictionary of updated stacks.
        nodes_seen: Set of nodes seen so far
        first_stack: Whether it's the first stack of the pass.
        tensor_log: Log of tensors

    Returns:
        Nothing, but updates the new_pass_stacks dictionary.
    """
    pass_start_nodes = list(pass_stacks.keys())
    for node_barcode in pass_stack:
        node = tensor_log[node_barcode]
        for child_barcode in node['child_tensor_barcodes']:
            if child_barcode not in nodes_seen:
                new_pass_stacks[pass_barcode]['children'].append(child_barcode)
                nodes_seen.add(child_barcode)
            elif child_barcode in pass_start_nodes:  # if we hit a node starting the stack, add that node's parents.
                nodes_seen.add(child_barcode)
                start_node = tensor_log[child_barcode]
                for parent_barcode in start_node['parent_tensor_barcodes']:
                    if parent_barcode not in nodes_seen:
                        new_pass_stacks[child_barcode]['parents'].append(parent_barcode)
                        nodes_seen.add(parent_barcode)
        for parent_barcode in node['parent_internal_tensor_barcodes']:
            if first_stack:
                continue
            if parent_barcode not in nodes_seen:
                new_pass_stacks[pass_barcode]['parents'].append(parent_barcode)
                nodes_seen.add(parent_barcode)


def update_pass_stacks(pass_stacks: Dict,
                       nodes_seen: set,
                       first_stack: bool,
                       tensor_log) -> Dict:
    """Update the stack for each pass.

    Args:
        pass_stacks: Dictionary where the key is the barcode for the beginning of each pass, and the value is the stack.
        nodes_seen: Set of barcodes of nodes seen.
        first_stack: Whether it's the first stack of the pass.
        tensor_log: log of all tensors

    Returns:
        Updated pass_stacks.
    """

    new_pass_stacks = {}
    for pass_barcode in pass_stacks:
        new_pass_stacks[pass_barcode] = {'children': [], 'parents': []}
        for node_type in ['children', 'parents']:
            pass_stack = pass_stacks[pass_barcode][node_type]
            update_pass_stack(pass_barcode,
                              pass_stack,
                              pass_stacks,
                              new_pass_stacks,
                              nodes_seen,
                              first_stack,
                              tensor_log)

    return new_pass_stacks


def get_nth_item_per_stack(pass_stacks: Dict,
                           node_type: str,
                           n: int) -> Dict:
    """Utility function to get the nth item in each stack if there are at least nth items; returns None
    otherwise.

    Args:
        pass_stacks: Dictionary where each key is the pass barcode, and each value is the stack for that pass.
        node_type: 'children' or 'parents'
        n: The index of the item to grab.

    Returns:
        Dictionary where each key is the pass barcode, and each value is the nth item.
    """
    nth_item_per_stack = {}
    for pass_start_barcode in pass_stacks:
        stack_nodes = pass_stacks[pass_start_barcode][node_type]
        stack_size = len(stack_nodes)
        if stack_size >= n + 1:
            stack_nth_item = stack_nodes[n]
        else:
            stack_nth_item = None
        nth_item_per_stack[pass_start_barcode] = stack_nth_item
    return nth_item_per_stack


def identify_repeated_functions_in_loop(repeated_node_occurrences: List[Dict],
                                        tensor_log: Dict):
    """For a single outermost loop, finds the corresponding functions in the loop.

    Args:
        repeated_node_occurrences: Barcodes for nodes corresponding to all passes of the layer
        tensor_log: Dict of all tensors

    Returns:
        Nothing, but the nodes have been tagged accordingly.
    """
    pass_stacks = {node['barcode']: {'children': [node['barcode']],
                                     'parents': []}
                   for node in repeated_node_occurrences}
    diverged_pass_pairs = set()
    nodes_seen = set(node['barcode'] for node in repeated_node_occurrences)
    first_stack = True
    while True:
        stack_lengths_total = [len(pass_stacks[barcode]['children']) + len(pass_stacks[barcode]['parents'])
                               for barcode in pass_stacks]
        if sum(stack_lengths_total[0:-1]) == 0:  # if any of the non-final stacks have run out, we're done.
            break
        for node_type in ['children', 'parents']:
            biggest_stack_size = max([len(pass_stacks[barcode][node_type]) for barcode in pass_stacks])
            for n in range(biggest_stack_size):  # iterate through the items in each stack and check for correspondence
                nth_item_per_stack = get_nth_item_per_stack(pass_stacks, node_type, n)
                mark_corresponding_pass_layers(nth_item_per_stack, diverged_pass_pairs, tensor_log)

        # Finally, set the new stacks for each pass; add any node children that haven't been seen yet,
        # and add any node parents that are internally generated and haven't been seen yet.

        pass_stacks = update_pass_stacks(pass_stacks, nodes_seen, first_stack, tensor_log)
        first_stack = False


def identify_repeated_functions(history_dict: Dict) -> Dict:
    """Goes through the graph and identifies nodes that have params, and whose params are repeated.
    This requires that the same set of ALL params be involved in each computation to count as "the same".
    Also keeps track of how many times they've been seen and tags them with the total passes at the end.
    Finally, for functions intervening between passes of the same param, tags those that are the same
    across passes as being the same.

    Args:
        history_dict: input history dict

    Returns:
        history_dict tagged with any repeated parameters
    """
    history_dict['enclosing_loop_nodes'] = {}

    # If there's no repeated parameters, there's nothing to do.
    if (len(history_dict['param_group_tensors']) == 0) or \
            (max([len(param_tensors) for param_tensors in history_dict['param_group_tensors'].values()]) < 2):
        return history_dict

    tensor_log = history_dict['tensor_log']
    enclosing_loop_nodes = find_outermost_loops(history_dict)
    for loop_nodes in enclosing_loop_nodes.values():
        history_dict['enclosing_loop_nodes'][loop_nodes[0]['barcode']] = [loop_node['barcode'] for loop_node in
                                                                          loop_nodes]

    for loop_start_barcode, repeated_node_occurrences in enclosing_loop_nodes.items():  # go through each outer loop
        identify_repeated_functions_in_loop(repeated_node_occurrences, tensor_log)

    return history_dict


def annotate_node_names(history_dict: Dict) -> Dict:
    """Given a tensor log with the topological sorting applied, annotates the log with all the nice, readable labels
    for each node.

    Args:
        history_dict: input history_dict

    Returns:
        history_dict with all the node names annotated.
    """
    tensor_log = history_dict['tensor_log']
    layer_type_counter = defaultdict(lambda: 0)
    module_type_counter = defaultdict(lambda: 0)

    layer_counter = 0  # how many unique layers encountered
    module_counter = 0  # how many unique modules encountered

    layer_type_inds = {}  # maps layers to their layer index (i.e., how many times that layer type has occurred)
    layer_total_inds = {}  # maps layers to their total index (i.e., how far along in the network they are total)
    module_type_inds = {}  # maps module layers to their index (i.e., how many times that layer type has occurred)

    for barcode, node in tensor_log.items():
        layer_barcode = node['layer_barcode']
        if node['is_model_output']:
            layer_type = 'output'
        elif node['is_model_input']:
            layer_type = 'input'
        else:
            layer_type = node['funcs_applied_names'][0].replace('_', '')

        node['layer_type'] = layer_type

        if layer_barcode not in layer_type_inds:
            layer_type_counter[layer_type] += 1
            layer_type_count = layer_type_counter[layer_type]
            layer_type_inds[layer_barcode] = layer_type_count
            layer_total_inds[layer_barcode] = layer_total_count = layer_counter
            layer_counter += 1
        else:
            layer_type_count = layer_type_inds[layer_barcode]
            layer_total_count = layer_total_inds[layer_barcode]

        node['layer_type_ind'] = layer_type_count
        node['layer_total_ind'] = layer_total_count
        pass_num = node['pass_num']

        node['layer_label'] = f'{layer_type}_{layer_type_count}_{layer_total_count}'
        node['layer_label_w_pass'] = f'{layer_type}_{layer_type_count}_{layer_total_count}:{pass_num}'

        if node['is_bottom_level_module_output']:
            node_module_address = node['bottom_module_barcode']
            module_type = node['bottom_module_type']
            node['module_type'] = module_type

            if node_module_address not in module_type_inds:
                module_type_counter[module_type] += 1
                module_counter += 1
                module_type_count = module_type_counter[module_type]
                module_type_inds[node_module_address] = module_type_count
            else:
                module_type_count = module_type_inds[node_module_address]

            node['module_type_ind'] = module_type_count
            node['module_label'] = f'{module_type}_{module_type_count}_{module_counter}:{pass_num}'

    return history_dict


def map_layer_names_to_op_nums(history_dict: Dict) -> Dict[str, List[int]]:
    """Generates a dictionary that maps from all possible layer names a user might provide (whether based
    on a function, a module, with or without specifying the pass, etc.), to the operation number
    for that layer name (i.e., how many tensors had been created at that point during the pass).
    This is to allow specific layers to be saved in subsequent passes.

    Args:
        history_dict: The history dict.

    Returns:
        history_dict with the layer-to-operation-numbers mapping.
    """
    tensor_log = history_dict['tensor_log']
    module_output_tensors = history_dict['module_output_tensors']

    label_to_op_num_dict = defaultdict(list)

    for barcode, tensor in tensor_log.items():
        label_to_op_num_dict[tensor['layer_label']].append(tensor['tensor_num'])
        label_to_op_num_dict[tensor['layer_label_w_pass']].append(tensor['tensor_num'])

    # And now fetch for all modules too.

    for module, module_tensors in module_output_tensors.items():
        label_to_op_num_dict[module] = [tensor_log[mt]['tensor_num'] for mt in module_tensors]
        for t, tensor_barcode in enumerate(module_tensors):
            module_pass_label = f"{module}:{t + 1}"
            label_to_op_num_dict[module_pass_label] = tensor_log[tensor_barcode]['tensor_num']

    history_dict['label_to_op_num_dict'] = label_to_op_num_dict
    return history_dict


def get_op_nums_from_layer_names(history_dict: Dict,
                                 layer_names: List[str]) -> List[int]:
    """Given a list of human-readable layer names and the history_dict, gets all the operation numbers
    for those layer names.

    Args:
        history_dict: Dictionary with the tensor history.
        layer_names: List of human-readable layer names.

    Returns:
        List of the operation numbers for all desired layers.
    """
    op_nums = []
    for layer_name in layer_names:
        if layer_name in history_dict['label_to_op_num_dict']:
            op_nums.extend(history_dict['label_to_op_num_dict'])
        else:
            raise ValueError("One of the layers you specified doesn't exist; try double-checking the layer names.")
    return op_nums


def tally_sizes_and_params(history_dict: Dict):
    """Adds up the total size of all tensors in the network, and the total number of params.

    Args:
        history_dict: Dictionary of history.

    Returns:
        History dict annotated with the total size of the tensors and params
    """
    tensor_log = history_dict['tensor_log']

    total_params = 0
    total_param_tensors = 0
    total_param_groups = 0
    total_params_fsize = 0
    total_tensor_size = 0

    for node_barcode, node in tensor_log.items():
        total_tensor_size += node['tensor_fsize']
        total_params_fsize += node['params_memory_size']
        for param_shape in node['parent_params_shape']:
            total_params += np.prod(param_shape)
        total_param_tensors += len(node['parent_params_shape'])
        if len(node['parent_params_shape']) > 0:
            total_param_groups += 1

    history_dict['total_tensor_fsize'] = total_tensor_size
    history_dict['total_params'] = total_params
    history_dict['total_param_tensors'] = total_param_tensors
    history_dict['total_param_groups'] = total_param_groups
    history_dict['total_params_fsize'] = total_params_fsize

    return history_dict


def find_equivalent_clusters(equiv_pairs: List[Tuple[Any, Any]]) -> Dict:
    """Utility function that takes in a list of tuples, and assigns all values in all tuples to clusters such that
    any values in the same cluster are in a pair together.

    Args:
        equiv_pairs: List of pairs of values that are to be in the same cluster.

    Returns:
        Dictionary mapping each value to the barcode of a cluster it's in.
    """
    value_to_cluster_dict = {}
    for item1, item2 in equiv_pairs:
        module = item1[0]
        if (item1 in value_to_cluster_dict) and (item2 in value_to_cluster_dict):
            continue
        elif item1 in value_to_cluster_dict:
            value_to_cluster_dict[item2] = value_to_cluster_dict[item1]
        elif item2 in value_to_cluster_dict:
            value_to_cluster_dict[item1] = value_to_cluster_dict[item2]
        else:
            new_barcode = make_barcode()
            value_to_cluster_dict[item1] = (module, new_barcode)
            value_to_cluster_dict[item2] = (module, new_barcode)

    return value_to_cluster_dict


def propagate_module_labels_for_two_nodes(parent_node: Dict,
                                          neighbor_node: Dict,
                                          nodes_seen: set,
                                          all_module_barcodes: List,
                                          module_equivalent_barcodes: Dict):
    """Utility function that given a parent and neighbor node, propagates the modules between them.
    If the neighboring node is unseen with the same module nesting, just copies from the parent;
    if it's unseen with a different module nesting, the same ones are copied, but different ones
    are given new barcode. If it's been seen, then go through and mark a

    Args:
        parent_node: The starting node
        neighbor_node: The neighbor node
        nodes_seen: Nodes seen so far
        module_equivalent_barcodes: Mapping of equivalent module barcodes.
    """
    neighbor_barcode = neighbor_node['barcode']
    parent_modules_nested = parent_node['function_call_modules_nested']
    neighbor_modules_nested = neighbor_node['function_call_modules_nested']
    if neighbor_modules_nested == parent_modules_nested:  # neighbors and in same module
        if neighbor_barcode in nodes_seen:
            for module in parent_modules_nested:  # mark them as equivalent
                node_rough_barcode = parent_node['module_instance_rough_barcodes'][module]
                neighbor_rough_barcode = neighbor_node['module_instance_rough_barcodes'][module]
                module_equivalent_barcodes[module].append((node_rough_barcode, neighbor_rough_barcode))
        else:
            neighbor_node['module_instance_rough_barcodes'] = parent_node['module_instance_rough_barcodes'].copy()
    else:
        if neighbor_barcode in nodes_seen:  # if the neighbor has the same modules, indicate them as equivalent.
            for module in neighbor_modules_nested:
                if module not in parent_modules_nested:
                    continue
                node_rough_barcode = parent_node['module_instance_rough_barcodes'][module]
                neighbor_rough_barcode = neighbor_node['module_instance_rough_barcodes'][module]
                module_equivalent_barcodes[module].append((node_rough_barcode, neighbor_rough_barcode))
        else:  # Generate the new barcodes for the neighboring node.
            neighbor_node['module_instance_rough_barcodes'] = {}
            for module in neighbor_modules_nested:
                if module in parent_modules_nested:
                    neighbor_node['module_instance_rough_barcodes'][module] = \
                        parent_node['module_instance_rough_barcodes'][module]
                else:
                    new_barcode = (module, make_barcode())
                    neighbor_node['module_instance_rough_barcodes'][module] = new_barcode
                    all_module_barcodes.append(new_barcode)


def annotate_internal_tensor_modules(history_dict: Dict):
    """Annotate any internally generated tensors with the relevant containing modules by tracing back from their children.

    Args:
        history_dict: The history dict.

    Returns:
        History dict where tensor log now has the internally generated tensors annotated with the containing modules.
    """
    tensor_log = history_dict['tensor_log']

    # Get the initial stack; any nodes that come from the input, but have at least one parent who doesn't.
    node_stack = []
    for node_barcode, node in tensor_log.items():
        if not node['has_input_ancestor']:
            continue
        for parent_node_barcode in node['parent_tensor_barcodes']:
            parent_node = tensor_log[parent_node_barcode]
            if not parent_node['has_input_ancestor']:
                node_stack.append(node_barcode)
                break

    # Now go through the stack; for each node, mark the parents with the right containing module, and add these parents
    # to the stack.

    while len(node_stack) > 0:
        node_barcode = node_stack.pop()
        node = tensor_log[node_barcode]
        for parent_node_barcode in node['parent_tensor_barcodes']:
            parent_node = tensor_log[parent_node_barcode]
            if not parent_node['has_input_ancestor']:
                # Annotate the containing module hierarchy: the rule is, start from the child node,
                # and apply in reverse any modules that were entered or exited.
                parent_node['function_call_modules_nested'] = node['function_call_modules_nested'].copy()
                thread_modules = node['containing_modules_thread']
                for enter_or_exit, module_address, entry_barcode in thread_modules[::-1]:
                    module_entry_barcode = (module_address, entry_barcode)
                    if enter_or_exit == '+':  # if it entered a module, remove that from parent nested modules.
                        parent_node['function_call_modules_nested'].remove(module_entry_barcode)
                    elif enter_or_exit == '-':  # if it exited a module, add that to the parent nested modules.
                        parent_node['function_call_modules_nested'].append(module_entry_barcode)
                node_stack.append(parent_node_barcode)

    return history_dict


def cluster_modules(history_dict: Dict):
    """Goes through the history dict and assigns 1) cluster IDs, and 2) cluster names that indicate
    the nested module structure. These must be distinct because the same module can be applied multiple times,
    and should have the same name, but can be genuinely different instances of the same module in
    the computational graph. The name of the cluster will be the module address, the ID will be the
    module address plus a barcode to make it distinct. So, each node will be tagged with this information, such
    that during the visualization step, the subgraphs can be built and populated with each node.
    This is NOT based on the programmer's structure, but on what modules are entered and left.

    Args:
        history_dict: Dictionary of the tensor history.
    """

    # TODO: populate the internally generated tensors with the right module address (inherit from children)

    tensor_log = history_dict['tensor_log']
    cluster_children_dict = defaultdict(list)
    top_level_module_clusters = []

    for barcode, node in tensor_log.items():
        module_instance_final_barcodes_dict = {}
        module_instance_final_barcodes_list = []
        for m, module_barcode in enumerate(node['function_call_modules_nested']):
            if m == 0:  # if a top level module, add to the list.
                if module_barcode not in top_level_module_clusters:
                    top_level_module_clusters.append(module_barcode)
            else:  # if a submodule and not a bottom-level module, add to the dict.
                if module_barcode not in cluster_children_dict[last_module_barcode]:
                    cluster_children_dict[last_module_barcode].append(module_barcode)

            last_module_barcode = module_barcode

    history_dict['top_level_module_clusters'] = top_level_module_clusters
    history_dict['module_cluster_children_dict'] = cluster_children_dict
    return history_dict


def subset_graph(history_dict: Dict, nodes_to_keep: List[str]) -> Dict:
    """Subsets the nodes of the graph, inheriting the parent/children of omitted nodes.

    Args:
        tensor_log: The input tensor log.
        nodes to keep: which nodes to keep
    Returns:
        output tensor_log with only the desired nodes.
    """
    raise NotImplementedError


def reduce_graph_to_modules_only(history_dict: Dict) -> Dict:
    """Reduces the graph to just tensors that are the output of lowest-level modules.

    Args:
        history_dict: The input history_dict
    Returns:
        output history_dict with only the lowest-level modules kept
    """
    raise NotImplementedError


def get_tensor_nums_for_layers(history_dict: Dict) -> Dict:
    """Fetches the tensor numbers (that is, the order in which the tensors were made for the network pass
    for all the labels the user might want to use; this is used for subsetting the graph in
    the subsequent pass that actually gets the activations.

    Args:
        history_dict: The input history_dict
    Returns:
        dict that maps layer labels (human-readable) to tensor nums.
    """
    raise NotImplementedError


def connect_node_arguments(history_dict: Dict) -> Dict:
    """Determines the mapping between the output of one node and the input to the next node.
    This is used for validating and debugging the models.

    Args:
        tensor_log: The tensor log.

    Returns:
        Tensor log annotated with the mapping between the outputs of one node and the inputs to child nodes.
    """
    raise NotImplementedError


def unmutate_tensor(t):
    """Convenience function to replace the tensor with an unmutated version of itself, keeping the same data.

    Args:
        t: tensor or parameter object

    Returns:
        Unmutated tensor.
    """
    if type(t) == torch.Tensor:
        new_t = safe_copy(t)
    elif type(t) == torch.nn.Parameter:
        new_t = torch.nn.Parameter(safe_copy(t))
    else:
        new_t = t
    return new_t


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
        node['tensor_contents'] = unmutate_tensor(node['tensor_contents'])
        for p, parent_param in enumerate(node['parent_params']):
            node['parent_params'][p] = unmutate_tensor(parent_param)

        for a in range(len(node['creation_args'])):
            arg = node['creation_args'][a]
            if issubclass(type(arg), (torch.Tensor, torch.nn.Parameter)):
                new_arg = unmutate_tensor(arg)
                node['creation_args'] = node['creation_args'][:a] + tuple([new_arg]) + node['creation_args'][a + 1:]

        for key, val in node['creation_kwargs'].items():
            if issubclass(type(val), (torch.Tensor, torch.nn.Parameter)):
                node['creation_kwargs'][key] = unmutate_tensor(val)

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
        annotate_node_children,  # note the children of each node
        strip_irrelevant_nodes,  # remove island nodes unconnected to inputs or outputs
        expand_multiple_functions,  # expand identity functions
        mark_output_ancestors,  # mark nodes as ancestors of the output
        add_output_nodes,  # add explicit output nodes
        topological_sort_nodes,  # sort nodes topologically
        annotate_total_layer_passes,  # note the total passes of the param nodes
        identify_repeated_functions,  # find repeated functions between repeated param nodes
        annotate_node_names,  # make the nodes names more human-readable
        map_layer_names_to_op_nums,  # get the operation numbers for any human-readable layer labels
        annotate_internal_tensor_modules,  # marks the internally generated tensors with contaning modules
        cluster_modules,  # identify the unique module instances and the right mappings
        tally_sizes_and_params  # tally total sizes of tensors and params
    ]

    for graph_transform in graph_transforms:
        history_dict = graph_transform(history_dict)

    return history_dict


def rough_barcode_to_final_barcode(node_barcode,
                                   history_dict):
    """Utility function that goes from the internal rough barcode to the final, human-readable barcode;
    this will be {layer_type}_{layer_type_num}_{layer_total} num with :{pass} too if there's multiple passes.

    Args:
        node_barcode: Rough barcode of node in question.
        history_dict: Dict of history.

    Returns:
        Human readable final barcode.
    """
    node = history_dict['tensor_log'][node_barcode]
    if node['param_total_passes'] > 1:
        final_barcode = node['layer_label_w_pass']
    else:
        final_barcode = node['layer_label']
    return final_barcode


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
            pass_str = f"(pass {self.layer_pass_num}/{self.layer_passes_total}), "
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
                num_dims = min(5, self.tensor_shape)
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
        module_str = f"(e.g., {sample_module1}, {sample_module2}"
        help_str = (f"Layer {layer_label} not recognized; please specify either \n\t1) an integer giving "
                    f"the ordinal position of the layer, \n\t2) the layer label (e.g., {sample_layer1}, "
                    f"{sample_layer2}), \n\t3) the module address {module_str})"
                    f"\n(conv2d_3_4:2 means the second pass of layer conv2d_3_4)")
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
            else:
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
