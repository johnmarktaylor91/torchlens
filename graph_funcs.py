# This module has functions for processing the computation graphs that come out of the other functions.

from collections import defaultdict
import graphviz
import numpy as np
import torch
import copy
from graphlib import TopologicalSorter
from IPython.display import display, Image
import networkx as nx

from collections import OrderedDict
from typing import Any, Dict, List, Tuple
from util_funcs import make_barcode, human_readable_size, in_notebook, int_list_to_compact_str

from tensor_tracking import log_tensor_metadata, log_tensor_data

graphviz.set_jupyter_format('png')

# TODO: make the nodes record so the titles pop more nicely
# TODO: Make the visualization code a bit prettier; try out some recurrent networks next.
# Visualization: put nodes in greyscale if they don't both come from the input and lead to the output.
# Add an indicator if they're outputs of a bottom-level module.
# Color inputs, outputs (green and red?), and any bottom-level module outputs
# Add another annotation for whether a node is an output ancestor.

# TODO: for cases like a + b + c, aggregate these into a single step instead of the converged binary operations.


# TODO annotate the graph log with useful metadata instead of making it denovo every time; e.g., the
# input and output nodes. Inputs, outputs, graph, dictionary of repeated nodes, counter of node types?

# Maybe tensor barcode, layer barcode?

# Get clear about when to do the barcodes vs the nodes themselves, be consistent.

# Get more consistent language for different kinds of barcodes, nodes, operations, etc.

# Hard-code the colors up here

INPUT_COLOR = "#98FB98"
OUTPUT_COLOR = "#ff9999"
PARAMS_NODE_BG_COLOR = "#E6E6E6"
DEFAULT_BG_COLOR = 'white'
CONNECTING_NODE_LINE_COLOR = 'black'
NONCONNECTING_NODE_LINE_COLOR = '#A0A0A0'
MAX_MODULE_PENWIDTH = 5
MIN_MODULE_PENWIDTH = 2
PENWIDTH_RANGE = MAX_MODULE_PENWIDTH - MIN_MODULE_PENWIDTH


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


def annotate_node_children(history_dict: Dict) -> Dict:
    """Annotates each node with the addresses of its child nodes.

    Args:
        history_dict: dictionary with all the data

    Returns:
        Tensor log annotated with the child node addresses.
    """
    tensor_log = history_dict['tensor_log']

    for barcode, node in tensor_log.items():
        if len(node['parent_tensor_barcodes']) == 0:
            continue
        for parent_barcode in node['parent_tensor_barcodes']:
            if 'child_tensor_barcodes' not in tensor_log[parent_barcode]:
                tensor_log[parent_barcode]['child_tensor_barcodes'] = []
            tensor_log[parent_barcode]['child_tensor_barcodes'].append(barcode)
    for barcode, node in tensor_log.items():
        if 'child_tensor_barcodes' not in node:
            node['child_tensor_barcodes'] = []
    return history_dict


def expand_multiple_functions(history_dict: Dict) -> Dict:
    """For nodes that have had multiple functions applied to them (e.g., the identity function), expands
    them to multiple nodes for fidelity to the graph. If the functions have the same name, then only use the one node.

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
                new_node['tensor_num'] = None
                new_node['funcs_applied'] = node['funcs_applied'][f:]
                new_node['funcs_applied_names'] = node['funcs_applied_names'][f:]
                new_node['modules_exited'] = node['modules_exited'][func_applied_module_position:]
                new_node['module_passes_exited'] = node['module_passes_exited'][func_applied_module_position:]
                new_node['child_tensor_barcodes'] = node['child_tensor_barcodes'][:]
                new_node['parent_tensor_barcodes'] = [node['barcode']]
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

                # And finally fix the original node.

                node['funcs_applied'] = node['funcs_applied'][:f]
                node['funcs_applied_names'] = node['funcs_applied_names'][:f]
                node['modules_exited'] = node['modules_exited'][:func_applied_module_position]
                node['module_passes_exited'] = node['module_passes_exited'][:func_applied_module_position]
                node['child_tensor_barcodes'] = [new_node['barcode']]
                node['is_model_output'] = False

                if ((len(node['modules_exited']) > 0)
                        and history_dict['module_dict'][node['modules_exited'][-1]].xray_is_bottom_level_module):
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

        node_count += 1
        node['operation_num_exhaustive'] = node_count
        ordered_tensor_log[node['barcode']] = node

        if node['is_module_output']:  # if it's a module output, also annotate the order for that.
            module_node_count += 1
            node['operation_num_module'] = module_node_count
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
        node['param_total_passes'] = 0
        node['module_total_passes'] = 0
    for param_group_barcode, tensors in param_group_tensors.items():
        param_group_num_passes = len(tensors)
        for tensor_barcode in tensors:
            tensor_log[tensor_barcode]['param_total_passes'] = param_group_num_passes
    for module_barcode, tensors in module_tensors.items():
        module_num_passes = len(tensors)
        for tensor_barcode in tensors:
            tensor_log[tensor_barcode]['module_total_passes'] = module_num_passes

    return history_dict


def traverse_graph(history_dict):
    """Debugging function to traverse the graph and make sure nothing weird happens.

    Args:
        history_dict: Dict with the history.

    Returns:
        List of lists of all loops, listed as the tensor barcodes.
    """

    tensor_log = history_dict['tensor_log']
    input_nodes = history_dict['input_tensors']

    pairs_seen = set()

    node_stack = input_nodes[:]
    num_passes = 0
    tensor_nums = []
    while len(node_stack) > 0:
        num_passes += 1
        node_barcode = node_stack.pop()
        node = tensor_log[node_barcode]
        tensor_nums.append(node['tensor_num'])
        node_children = node['child_tensor_barcodes']
        for node_child_barcode in node_children:
            node_stack.append(node_child_barcode)
            new_pair = (node_barcode, node_child_barcode)
            # if new_pair in pairs_seen:
            #    return new_pair
            pairs_seen.add(new_pair)
    print(num_passes)


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

    stack_lens = []
    new_child_lens = []

    while len(node_stack) > 0:
        node_tuple = node_stack.pop()
        node_barcode, enclosing_loop_node = node_tuple
        node = tensor_log[node_barcode]

        if all([node['has_params'], node['param_total_passes'] > 1, enclosing_loop_node is None]):
            #  We've hit the start of an enclosing loop, make that loop the label.
            children_enclosing_loop = node['layer_barcode']
            enclosing_loop_nodes[node['layer_barcode']].append(node)
        elif (node['layer_barcode'] == enclosing_loop_node) and (node['pass_num'] < node['param_total_passes']):
            # We've hit another occurrence of the enclosing loop.
            children_enclosing_loop = enclosing_loop_node
            enclosing_loop_nodes[node['layer_barcode']].append(node)
        elif (node['layer_barcode'] == enclosing_loop_node) and (node['pass_num'] == node['param_total_passes']):
            # We've hit the end of an enclosing loop, remove that as the label.
            enclosing_loop_nodes[node['layer_barcode']].append(node)
            children_enclosing_loop = None
        else:  # We're inside an enclosing loop.
            children_enclosing_loop = enclosing_loop_node

        nodes_to_add = [(child_node, children_enclosing_loop) for child_node in node['child_tensor_barcodes']]
        stack_lens.append(len(node_stack))
        new_child_lens.append(len(nodes_to_add))
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
    if (pass1_funcs != pass2_funcs) or not are_args_equal(pass1_args, pass2_args):  # the two passes aren't the same.
        diverged_pass_pairs.add(pass_pair_key)
        return False
    return True


def group_matching_layers(nth_item_per_stack: Dict,
                          diverged_pass_pairs: set) -> Dict[str, List[Dict]]:
    """Utility function that takes in the nth item per stack, and the mapping of passes that have already diverged,
    and groups together corresponding layers.

    Args:
        nth_item_per_stack: Dict mapping the barcode for the start of each pass to the nth item in the stack
            for that pass.
        diverged_pass_pairs: Set of passes that have already diverged.

    Returns:
        Dictionary mapping the layer name to all tensor barcodes for that layer.
    """
    # these will map each node to the set of nodes it's the same layer as (indexed by the barcode of the
    # first node).
    same_layer_barcode_mapper = {}  # maps each node to the layer group it's part of
    same_layer_barcode_lists = defaultdict(list)  # list of layers for each group

    for p1, (pass1_start_barcode, pass1_nth_item) in enumerate(nth_item_per_stack.items()):
        for p2, (pass2_start_barcode, pass2_nth_item) in enumerate(nth_item_per_stack.items()):
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

    same_layer_barcode_lists = group_matching_layers(nth_item_per_stack, diverged_pass_pairs)

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
    for node in pass_stack:
        for child_barcode in node['child_tensor_barcodes']:
            if child_barcode not in nodes_seen:
                new_pass_stacks[pass_barcode]['children'].append(tensor_log[child_barcode])
                nodes_seen.add(child_barcode)
            elif child_barcode in pass_start_nodes:  # if we hit a node starting the stack, add that node's parents.
                nodes_seen.add(child_barcode)
                start_node = tensor_log[child_barcode]
                for parent_barcode in start_node['parent_tensor_barcodes']:
                    if parent_barcode not in nodes_seen:
                        new_pass_stacks[child_barcode]['parents'].append(tensor_log[parent_barcode])
                        nodes_seen.add(parent_barcode)
        for parent_barcode in node['parent_internal_tensor_barcodes']:
            if first_stack:
                continue
            if parent_barcode not in nodes_seen:
                new_pass_stacks[pass_barcode]['parents'].append(tensor_log[parent_barcode])
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
    pass_stacks = {node['barcode']: {'children': [node],
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
    # If there's no repeated parameters, there's nothing to do.
    if (len(history_dict['param_group_tensors']) == 0) or \
            (max([len(param_tensors) for param_tensors in history_dict['param_group_tensors'].values()]) < 2):
        return history_dict

    tensor_log = history_dict['tensor_log']
    enclosing_loop_nodes = find_outermost_loops(history_dict)

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

    # TODO: have it also mark the name of child/parent layers.
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
            layer_type = node['funcs_applied_names'][0].strip('_')

        node['layer_type'] = layer_type

        if layer_barcode not in layer_type_inds:
            layer_type_counter[layer_type] += 1
            layer_counter += 1
            layer_type_count = layer_type_counter[layer_type]
            layer_type_inds[layer_barcode] = layer_type_count
            layer_total_inds[layer_barcode] = layer_total_count = layer_counter
        else:
            layer_type_count = layer_type_inds[layer_barcode]
            layer_total_count = layer_total_inds[layer_barcode]

        node['layer_type_ind'] = layer_type_count
        node['layer_total_ind'] = layer_total_count
        pass_num = node['pass_num']

        node['layer_label'] = f'{layer_type}_{layer_type_count}_{layer_counter}'
        node['layer_label_w_pass'] = f'{layer_type}_{layer_type_count}_{layer_counter}:{pass_num}'

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
    total_params_fsize = 0
    total_tensor_size = 0

    for node_barcode, node in tensor_log.items():
        total_tensor_size += node['tensor_fsize']
        total_params_fsize += node['params_memory_size']
        for param_shape in node['parent_params_shape']:
            total_params += np.prod(param_shape)

    history_dict['total_tensor_fsize'] = total_tensor_size
    history_dict['total_params'] = total_params
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
    node_stack = history_dict['input_tensors']
    for node_barcode in node_stack:
        node = tensor_log[node_barcode]
        node['module_instance_rough_barcodes'] = {}  # dict of each enclosing module and its rough barcode

    module_equivalent_barcodes = defaultdict(list)  # for each module, pairs of equivalent trial barcodes
    all_module_barcodes = []
    nodes_seen = set()
    node_pairs_seen = set()  # don't examine the same node pairs twice.

    # Dynamically generate the module equivalence classes.

    while len(node_stack) > 0:
        node_barcode = node_stack.pop()
        nodes_seen.add(node_barcode)
        node = tensor_log[node_barcode]
        adjacent_barcodes = node['child_tensor_barcodes'] + node['parent_tensor_barcodes']
        for neighbor_barcode in adjacent_barcodes:
            node_pair_label = '_'.join(sorted([node_barcode, neighbor_barcode]))
            if node_pair_label in node_pairs_seen:
                continue
            node_pairs_seen.add(node_pair_label)

            neighbor_node = tensor_log[neighbor_barcode]
            propagate_module_labels_for_two_nodes(node, neighbor_node, nodes_seen,
                                                  all_module_barcodes, module_equivalent_barcodes)
            if neighbor_barcode not in nodes_seen:
                node_stack.append(neighbor_barcode)
                nodes_seen.add(neighbor_barcode)

    # Now, for each module, greedily combine the equivalence classes, and generate their actual barcodes.

    module_rough_to_final_barcodes = {}
    for module in module_equivalent_barcodes:
        module_rough_to_final_barcodes[module] = find_equivalent_clusters(module_equivalent_barcodes[module])

    # Now go back through the network, and add these final barcodes, and also make note of all the parent-child
    # cluster relationships, and annotate history_dict with it to make the right graph clusters when
    # making the visuals.

    cluster_children_dict = defaultdict(list)
    top_level_module_clusters = []

    for barcode, node in tensor_log.items():
        module_instance_final_barcodes_dict = {}
        module_instance_final_barcodes_list = []
        for m, (module, rough_barcode) in enumerate(node['module_instance_rough_barcodes'].items()):
            if (module in module_rough_to_final_barcodes) and (
                    rough_barcode in module_rough_to_final_barcodes[module]):
                final_barcode = module_rough_to_final_barcodes[module][rough_barcode]
            else:
                final_barcode = rough_barcode
            module_instance_final_barcodes_dict[module] = final_barcode
            module_instance_final_barcodes_list.append(final_barcode)
            if m == 0:  # if a top level module, add to the list.
                if final_barcode not in top_level_module_clusters:
                    top_level_module_clusters.append(final_barcode)
            else:  # if a submodule and not a bottom-level module, add to the dict.
                if final_barcode not in cluster_children_dict[last_module_instance_id]:
                    cluster_children_dict[last_module_instance_id].append(final_barcode)

            last_module_instance_id = final_barcode

        if not node['is_model_output']:
            node['module_instance_final_barcodes_dict'] = module_instance_final_barcodes_dict
            node['module_instance_final_barcodes_list'] = module_instance_final_barcodes_list
        else:
            node['module_instance_final_barcodes_dict'] = {}
            node['module_instance_final_barcodes_list'] = []

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


def roll_graph(history_dict: Dict) -> Dict:
    """Converts the graph to rolled-up format for plotting purposes. This means that the nodes of the graph
    are now not tensors, but layers, with the layer children for each pass indicated. This is only done
    for visualization purposes; no tensor data is saved.

    Args:
        history_dict: The history_dict

    Returns:
        Rolled-up tensor log.
    """

    fields_to_copy = ['layer_barcode', 'layer_type', 'layer_type_ind', 'layer_total_ind',
                      'is_model_input', 'is_model_output', 'is_last_output_layer', 'connects_input_and_output',
                      'tensor_shape', 'tensor_fsize', 'has_params', 'param_total_passes', 'parent_params_shape',
                      'is_bottom_level_module_output', 'modules_exited',
                      'module_total_passes', 'module_instance_final_barcodes_list']

    tensor_log = history_dict['tensor_log']
    rolled_tensor_log = OrderedDict({})

    for node_barcode, node in tensor_log.items():
        # Get relevant information from each node.

        layer_barcode = node['layer_barcode']
        if layer_barcode in rolled_tensor_log:
            rolled_node = rolled_tensor_log[layer_barcode]
        else:
            rolled_node = OrderedDict({})
            for field in fields_to_copy:
                if field in node:
                    rolled_node[field] = node[field]
            rolled_node['child_layer_barcodes'] = {}  # each key is pass_num, each value list of children
            rolled_node['parent_layer_barcodes'] = {}  # each key is pass_num, each value list of parents
            rolled_tensor_log[layer_barcode] = rolled_node

        # Only difference is that now the parents and children are layer barcodes, not tensor barcodes,
        # and they are linked to each pass of the layer.

        pass_num = node['pass_num']
        child_layer_barcodes = [tensor_log[child_tensor_barcode]['layer_barcode']
                                for child_tensor_barcode in node['child_tensor_barcodes']]
        rolled_node['child_layer_barcodes'][pass_num] = child_layer_barcodes

        parent_layer_barcodes = [tensor_log[parent_tensor_barcode]['layer_barcode']
                                 for parent_tensor_barcode in node['parent_tensor_barcodes']]
        rolled_node['parent_layer_barcodes'][pass_num] = parent_layer_barcodes

    return rolled_tensor_log


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
        cluster_modules,  # identify the unique module instances and the right mappings
        tally_sizes_and_params  # tally total sizes of tensors and params
    ]

    for graph_transform in graph_transforms:
        history_dict = graph_transform(history_dict)

    # TODO: Figure out which fields to keep.

    # fields_to_keep = []  # ordered list of fields to keep; can remove irrelevant fields.
    # tensor_log = OrderedDict({k: OrderedDict({f: v[f] for f in fields_to_keep}) for k, v in tensor_log.items()})
    # history_dict['tensor_log'] = tensor_log

    return history_dict


def prettify_history_dict(history_dict: Dict) -> Dict:
    """Returns the final user-readable version of tensor_log for the user, omitting all the ugly internal stuff.

    Args:
        history_dict: Input history_dict

    Returns:
        Nicely organized/labeled final dict.
    """
    # which_fields =

    tensor_log = history_dict['tensor_log']
    pretty_tensor_log = OrderedDict()
    for tensor_barcode, tensor in tensor_log.items():
        if tensor['param_total_passes'] > 1:
            pretty_tensor_log[tensor['layer_label_w_pass']] = tensor
        else:
            pretty_tensor_log[tensor['layer_label']] = tensor
    return pretty_tensor_log


def get_lowest_containing_module_for_two_nodes(node1: Dict,
                                               node2: Dict):
    """Utility function to get the lowest-level module that contains two nodes, to know where to put the edge.

    Args:
        node1: The first node.
        node2: The second node.

    Returns:
        Barcode of lowest-level module containing both nodes.
    """
    node1_modules = node1['module_instance_final_barcodes_list']
    node2_modules = node2['module_instance_final_barcodes_list']

    if (len(node1_modules) == 0) or (len(node2_modules) == 0) or (node1_modules[0] != node2_modules[0]):
        return -1  # no submodule contains them both.

    containing_module = node1_modules[0]
    for m in range(min([len(node1_modules), len(node2_modules)])):
        if node1_modules[m] != node2_modules[m]:
            break
        containing_module = node1_modules[m]
    return containing_module


def add_rolled_edges_for_node(node: Dict,
                              graphviz_graph,
                              module_cluster_dict: Dict,
                              tensor_log: Dict):
    """Add the rolled-up edges for a node, marking for the edge which passes it happened for.

    Args:
        node: The node to add edges for.
        graphviz_graph: The graphviz graph object.
        module_cluster_dict: Dictionary mapping each cluster to the edges it contains.
        tensor_log: The tensor log.
    """

    # First determine for which passes of the node each edge happens for, then add the edges.

    child_layer_passes = defaultdict(list)  # for each layer, which passes they happen for.
    for pass_num, child_layer_barcodes in node['child_layer_barcodes'].items():
        for child_layer_barcode in child_layer_barcodes:
            child_layer_passes[child_layer_barcode].append(pass_num)

    for child_layer_barcode, pass_nums in child_layer_passes.items():
        child_node = tensor_log[child_layer_barcode]
        if node['param_total_passes'] > 1:
            edge_label = f"#{int_list_to_compact_str(pass_nums)}"
        else:
            edge_label = ''

        if child_node['connects_input_and_output'] and node['connects_input_and_output']:
            edge_color = CONNECTING_NODE_LINE_COLOR
            edge_style = 'solid'
        else:
            edge_color = CONNECTING_NODE_LINE_COLOR  # NONCONNECTING_NODE_LINE_COLOR
            edge_style = 'dashed'

        edge_dict = {'tail_name': node['layer_barcode'],
                     'head_name': child_layer_barcode,
                     'color': edge_color,
                     'style': edge_style,
                     'label': edge_label}

        containing_module = get_lowest_containing_module_for_two_nodes(node, child_node)
        if containing_module != -1:
            module_cluster_dict[containing_module].append(edge_dict)
        else:
            graphviz_graph.edge(**edge_dict)


def add_node_to_graphviz(node_barcode: Dict,
                         graphviz_graph,
                         vis_opt: str,
                         module_cluster_dict: Dict,
                         tensor_log: Dict,
                         history_dict: Dict):
    """Addes a node and its relevant edges to the graphviz figure.

    Args:
        node_barcode: Barcode of the node to add.
        graphviz_graph: The graphviz object to add the node to.
        vis_opt: Whether to roll the graph or not
        module_cluster_dict: Dictionary of the module clusters.
        tensor_log: log of tensors
        history_dict: The history_dict

        #TODO figure out the subgraph stuff. Either the context method or the new graph method, add labels,
        #TODO don't override the default node labels.

    Returns:
        Nothing, but updates the graphviz_graph
    """
    node = tensor_log[node_barcode]

    if node['is_bottom_level_module_output']:
        last_module_seen = "<br/>@" + node['modules_exited'][0]
        last_module_seen = f"{last_module_seen}"
        last_module_total_passes = node['module_total_passes']
        if (last_module_total_passes > 1) and (vis_opt == 'unrolled'):
            last_module_num_passes = node['module_passes_exited'][0][1]
            last_module_seen += f":{last_module_num_passes}"
        node_shape = 'box'
    else:
        last_module_seen = ''
        node_shape = 'oval'

    if node['is_model_input']:
        bg_color = INPUT_COLOR
    elif node['is_model_output']:
        bg_color = OUTPUT_COLOR
    elif node['has_params']:
        bg_color = PARAMS_NODE_BG_COLOR
    else:
        bg_color = DEFAULT_BG_COLOR

    tensor_shape = node['tensor_shape']
    layer_type = node['layer_type']
    layer_type_str = layer_type.replace('_', '')
    layer_type_ind = node['layer_type_ind']
    layer_total_ind = node['layer_total_ind']
    if node['connects_input_and_output']:
        node_color = CONNECTING_NODE_LINE_COLOR
        line_style = 'solid'
    else:
        node_color = CONNECTING_NODE_LINE_COLOR
        line_style = 'dashed'  # NONCONNECTING_NODE_LINE_COLOR
    if (node['param_total_passes'] > 1) and (vis_opt != 'rolled'):
        pass_num = node['pass_num']
        pass_label = f":{pass_num}"
    else:
        pass_label = ''
    if len(tensor_shape) > 1:
        tensor_shape_str = 'x'.join([str(x) for x in tensor_shape])
    elif len(tensor_shape) == 1:
        tensor_shape_str = f'x{tensor_shape[0]}'
    else:
        tensor_shape_str = 'x1'
    tensor_shape_str = f"{tensor_shape_str}"

    if node['has_params']:
        each_param_shape = []
        for param_shape in node['parent_params_shape']:
            if len(param_shape) > 1:
                each_param_shape.append('x'.join([str(s) for s in param_shape]))
            else:
                each_param_shape.append('x1')
        param_label = "<br/>params: " + ', '.join([param_shape for param_shape in each_param_shape])
    else:
        param_label = ''

    tensor_fsize = human_readable_size(node['tensor_fsize'])

    node_title = f"{layer_type_str}_{layer_type_ind}_{layer_total_ind}{pass_label}"
    node_title = f'<b>{node_title}</b>'

    node_label = (f'<{node_title}<br/>{tensor_shape_str} '
                  f'({tensor_fsize}){param_label}{last_module_seen}>')

    graphviz_graph.node(name=node_barcode, label=f"{node_label}",
                        fontcolor=node_color,
                        color=node_color,
                        style=f"filled,{line_style}",
                        fillcolor=bg_color,
                        shape=node_shape)

    if vis_opt == 'rolled':
        add_rolled_edges_for_node(node, graphviz_graph, module_cluster_dict, tensor_log)
    else:
        for child_barcode in node['child_tensor_barcodes']:
            child_node = tensor_log[child_barcode]
            if tensor_log[child_barcode]['connects_input_and_output'] and node['connects_input_and_output']:
                edge_color = CONNECTING_NODE_LINE_COLOR
                edge_style = 'solid'
            else:
                edge_color = CONNECTING_NODE_LINE_COLOR  # NONCONNECTING_NODE_LINE_COLOR
                edge_style = 'dashed'
            edge_dict = {'tail_name': node_barcode,
                         'head_name': child_barcode,
                         'color': edge_color,
                         'style': edge_style}
            containing_module = get_lowest_containing_module_for_two_nodes(node, child_node)
            if containing_module != -1:
                module_cluster_dict[containing_module].append(edge_dict)
            else:
                graphviz_graph.edge(**edge_dict)

    # Finally, if it's the final output layer, force it to be on top for visual niceness.

    if node['is_last_output_layer'] and vis_opt == 'rolled':
        with graphviz_graph.subgraph() as s:
            s.attr(rank='sink')
            s.node(node_barcode)


def setup_subgraphs_recurse(parent_graph,
                            subgraph_tuple,
                            module_cluster_dict: Dict[str, List],
                            module_cluster_children_dict: Dict,
                            nesting_depth: int,
                            max_nesting_depth: int,
                            history_dict: Dict):
    """Inner recursive function to set up the subgraphs; this is needed in order to handle arbitrarily nested
    "with" statements.

    Args:
        parent_graph: Parent of the subgraph being added.
        subgraph_tuple: Name of subgraph being processed.
        module_cluster_dict: Dict that maps each subgraph to list of edges in the subgraph.
        module_cluster_children_dict: Dict mapping each cluster to its children clusters.
        nesting_depth: How many submodules deep you are. This is used to determine the edge thickness
        max_nesting_depth: The deepest nesting depth of modules in the network.
        history_dict: Dict of the history
    """
    nesting_depth += 1
    pen_width = MIN_MODULE_PENWIDTH + ((max_nesting_depth - nesting_depth) / max_nesting_depth) * PENWIDTH_RANGE

    subgraph_module, subgraph_barcode = subgraph_tuple
    subgraph_name = '_'.join(subgraph_tuple)
    cluster_name = f"cluster_{subgraph_name}"
    module_type = str(type(history_dict['module_dict'][subgraph_module]).__name__)

    with parent_graph.subgraph(name=cluster_name) as s:
        s.attr(label=f"<<B>@{subgraph_module}</B><br align='left'/>{module_type}<br align='left'/>>",
               labelloc='b',
               penwidth=str(pen_width))
        subgraph_edges = module_cluster_dict[subgraph_tuple]
        for edge_dict in subgraph_edges:
            s.edge(**edge_dict)
        subgraph_children = module_cluster_children_dict[subgraph_tuple]
        for subgraph_child in subgraph_children:
            setup_subgraphs_recurse(s, subgraph_child,
                                    module_cluster_dict, module_cluster_children_dict,
                                    nesting_depth, max_nesting_depth, history_dict)


def get_max_nesting_depth(top_graphs,
                          module_cluster_dict,
                          module_cluster_children_dict):
    """Utility function to get the max nesting depth of the nested modules in the network; works by
    recursively crawling down the stack of modules till it hits one with no children and at least one edge.

    Args:
        top_graphs: Top-level modules
        module_cluster_dict: Edges in each module.
        module_cluster_children_dict: Mapping from each module to any children.

    Returns:
        Max nesting depth.
    """
    max_nesting_depth = 1
    module_stack = [(graph, 1) for graph in top_graphs]

    while len(module_stack) > 0:
        module, module_depth = module_stack.pop()
        module_edges = module_cluster_dict[module]
        module_children = module_cluster_children_dict[module]

        if len(module_edges) == 0:  # can ignore if no edges.
            continue
        elif len(module_children) == 0:
            max_nesting_depth = max([module_depth, max_nesting_depth])
        else:
            module_stack.extend([(module_child, module_depth + 1) for module_child in module_children])
    return max_nesting_depth


def set_up_subgraphs(graphviz_graph,
                     module_cluster_dict: Dict[str, List],
                     history_dict: Dict):
    """Given a dictionary specifying teh edges in each cluster, the graphviz graph object, and the history_dict,
    set up the nested subgraphs and the nodes that should go inside each of them. There will be some tricky
    recursive logic to set up the nested context managers.

    Args:
        graphviz_graph: Graphviz graph object.
        module_cluster_dict: Dictionary mapping each cluster name to the list of edges it contains, with each
            edge specified as a dict with all necessary arguments for creating that edge.
        history_dict: History dict.
    """
    module_cluster_children_dict = history_dict['module_cluster_children_dict']
    subgraphs = history_dict['top_level_module_clusters']

    nesting_depth = 0

    # Get the max nesting depth; it'll be the depth of the deepest module that has no edges.

    max_nesting_depth = get_max_nesting_depth(subgraphs,
                                              module_cluster_dict,
                                              module_cluster_children_dict)

    for subgraph_tuple in subgraphs:
        setup_subgraphs_recurse(graphviz_graph,
                                subgraph_tuple,
                                module_cluster_dict,
                                module_cluster_children_dict,
                                nesting_depth,
                                max_nesting_depth,
                                history_dict)


def render_graph(history_dict: Dict,
                 vis_opt: str = 'unrolled') -> None:
    """Given the history_dict, renders the computational graph.
    #TODO: add rolled up option, subsetting options.
    #TODO: Add summary info about the whole graph, such as the total size of all activations.
    #TODO: Add jupyter visualization options, make it autodetect

    Args:
        history_dict:

    """
    if vis_opt not in ['rolled', 'unrolled']:
        raise ValueError("vis_opt must be either 'rolled' or 'unrolled'")

    if vis_opt == 'unrolled':
        tensor_log = history_dict['tensor_log']
    elif vis_opt == 'rolled':
        tensor_log = roll_graph(history_dict)
    total_tensor_fsize = human_readable_size(history_dict['total_tensor_fsize'])
    total_params = history_dict['total_params']
    total_params_fsize = human_readable_size(history_dict['total_params_fsize'])
    num_tensors = len(tensor_log)
    graph_caption = (
        f"<<B>{history_dict['model_name']}</B><br align='left'/>{num_tensors} tensors total ({total_tensor_fsize})"
        f"<br align='left'/>{total_params} params total ({total_params_fsize})<br align='left'/>>")

    dot = graphviz.Digraph(name='model', comment='Computational graph for the feedforward sweep',
                           filename='model.png', format='png')
    dot.graph_attr.update({'rankdir': 'BT',
                           'label': graph_caption,
                           'labelloc': 't',
                           'labeljust': 'left'})
    dot.node_attr.update({'shape': 'box'})

    module_cluster_dict = defaultdict(list)  # list of edges for each subgraph; subgraphs will be created at the end.

    for node_barcode, node in tensor_log.items():
        add_node_to_graphviz(node_barcode,
                             dot,
                             vis_opt,
                             module_cluster_dict,
                             tensor_log,
                             history_dict)

    # Finally, set up the subgraphs.

    set_up_subgraphs(dot, module_cluster_dict, history_dict)

    if in_notebook():
        # TODO get this to work
        print("printing in notebook")
        # from IPython.display import Image
        # graphviz.Source(dot).view()
        # Image(dot)
    else:
        # dot.render(directory='doctest-output').replace('\\', '/')
        dot.render('graph.gv', view=True)
