# This module has functions for processing the computation graphs that come out of the other functions.

from collections import defaultdict
import torch
import copy
from graphlib import TopologicalSorter

from collections import OrderedDict
from typing import Dict, List
from util_funcs import make_barcode


# TODO annotate the graph log with useful metadata instead of making it denovo every time; e.g., the
# input and output nodes. Inputs, outputs, graph, dictionary of repeated nodes, counter of node types?

# Maybe tensor barcode, layer barcode?

# Get clear about when to do the barcodes vs the nodes themselves, be consistent.

# Get more consistent language for different kinds of barcodes, nodes, operations, etc.

def annotate_node_children(history_dict: Dict) -> Dict:
    """Annotates each node with the address of its parent node.

    Args:
        history_dict: dictionary with all the data

    Returns:
        Tensor log annotated with the parent node addresses.
    """
    tensor_log = history_dict['tensor_log']

    for barcode, node in tensor_log.items():
        if node['parent_barcodes'] is None:
            continue
        for parent_barcode in node['parent_tensor_barcodes']:
            if 'child_barcodes' not in tensor_log[parent_barcode]:
                tensor_log[parent_barcode]['child_tensor_barcodes'] = []
            tensor_log[parent_barcode]['child_tensor_barcodes'].append(barcode)
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

    # Go through each node and expand the nodes that have multiple, non-identically-named functions.

    while len(node_stack) > 0:
        node = node_stack.pop()
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
                # if it's a new function, make a new node for it and modify fields as needed,
                # for the old node, for its original children, and for the new node inserted between the two.

                # New node:

                new_node = copy.deepcopy(node)
                new_node['barcode'] = make_barcode()
                new_node['funcs_applied'] = node['funcs_applied'][f:]
                new_node['funcs_applied_names'] = node['funcs_applied_names'][f:]
                new_node['child_tensor_barcodes'] = node['child_tensor_barcodes']

                # Tweak parent barcodes of child nodes to point to the new node.

                for child_barcode in new_node['child_barcodes']:
                    child_node = tensor_log[child_barcode]
                    for p, child_parent_barcode in enumerate(child_node['parent_barcodes']):
                        if child_parent_barcode == node['barcode']:
                            child_node['parent_tensor_barcodes'][p] = new_node['barcode']

                # And finally fix the original node.

                node['funcs_applied'] = node['funcs_applied'][:f]
                node['funcs_applied_names'] = node['funcs_applied_names'][:f]
                node['child_tensor_barcodes'] = [new_node['barcode']]

                node_stack.append(new_node)
                break

    return history_dict


def get_all_connected_nodes(tensor_log: Dict,
                            starting_node_barcodes: List[str]) -> set:
    """Returns set of all nodes connected somehow to any of the starting nodes, using a flooding algorithm.

    Args:
        tensor_log: Log of the tensors.
        starting_node_barcodes: List of barcodes of the starting nodes.

    Returns:
        Set of all nodes with any connection to the starting nodes.
    """
    node_stack = [tensor_log['barcode'] for barcode in starting_node_barcodes]
    connected_nodes = set()

    while len(node_stack) > 0:
        node = node_stack.pop()
        connected_nodes.add(node)
        for child_barcode in tensor_log[node]['child_tensor_barcodes']:
            if child_barcode not in connected_nodes:
                node_stack.append(child_barcode)
        for parent_barcode in tensor_log[node]['parent_tensor_barcodes']:
            if parent_barcode not in connected_nodes:
                node_stack.append(parent_barcode)

    return connected_nodes


def strip_irrelevant_nodes(history_dict: Dict) -> Dict:
    """Strip irrelevant nodes, keeping nodes that are connected somehow to both the inputs and the outputs
    of the graph.

    Args:
        history_dict: input history dict

    Returns:
        history_dict with irrelevant nodes stripped
    """
    tensor_log = history_dict['tensor_log']

    input_connected_nodes = get_all_connected_nodes(tensor_log, history_dict['input_tensors'])
    output_connected_nodes = get_all_connected_nodes(tensor_log, history_dict['output_tensors'])

    non_island_tensors = input_connected_nodes.intersection(output_connected_nodes)
    new_tensor_log = OrderedDict()

    for barcode, node in tensor_log.items():
        if barcode in non_island_tensors:
            new_tensor_log[barcode] = node

    history_dict['tensor_log'] = new_tensor_log
    return history_dict


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

    node_stack = [tensor_log[barcode] for barcode in history_dict['input_tensors']]
    nodes_seen = set()
    internal_ancestors_added = set()
    converge_nodes = []
    node_count = 0
    module_node_count = 0
    bottom_module_node_count = 0
    ordered_tensor_log = OrderedDict({})  # re-order the tensor log in topological order.
    while len(node_stack) > 0 or len(converge_nodes) > 0:
        # Check if any converge nodes have all parents seen except for internally generated;
        # if so, add their internal ancestors to the stack.
        for converge_node in converge_nodes:
            if all([ancestor in internal_ancestors_added for ancestor in converge_node['internal_ancestors']]):
                continue
            unseen_parents = [parent for parent in converge_node['parent_tensor_barcodes'] if parent not in nodes_seen]
            if all([parent in node['parent_internal_tensor_barcodes'] for parent in unseen_parents]):
                for ancestor in converge_node['internal_ancestors']:
                    if ancestor not in internal_ancestors_added:
                        node_stack.append(tensor_log[ancestor])
                        internal_ancestors_added.add(ancestor)

        found_converge = False

        # Go through the list of converge nodes to check, and if none found, take the next regular node.

        for c, converge_node in enumerate(converge_nodes):
            if all([parent_node in nodes_seen for parent_node in converge_node['parent_barcodes']]):
                node = converge_nodes.pop(c)
                found_converge = True
                break

        if not found_converge:
            node = node_stack.pop()
            if not all([parent_node in nodes_seen for parent_node in node['parent_barcodes']]):
                converge_nodes.append(node)
                continue

        node_count += 1
        node['operation_number_exhaustive'] = node_count
        ordered_tensor_log[node['barcode']] = node

        if node['is_module_output']:  # if it's a module output, also annotate the order for that.
            module_node_count += 1
            node['operation_number_module'] = module_node_count
        else:
            node['operation_number_module'] = None

        if node['is_bottom_level_module_output']:  # if it's a module output, also annotate the order for that.
            bottom_module_node_count += 1
            node['operation_number_bottom_module'] = bottom_module_node_count
        else:
            node['operation_number_bottom_module'] = None

        nodes_seen.add(node['barcode'])
        node_stack.extend([tensor_log[barcode] for barcode in node['child_tensor_barcodes']])

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
    for param_group_barcode, tensors in param_group_tensors.items():
        param_group_num_passes = len(tensors)
        for tensor_barcode in tensors:
            tensor_log[tensor_barcode]['total_passes'] = param_group_num_passes
    for module_barcode, tensors in module_tensors.items():
        module_num_passes = len(tensors)
        for tensor_barcode in tensors:
            tensor_log[tensor_barcode]['total_passes'] = module_num_passes

    return history_dict


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
    input_nodes = history_dict['input_tensors']

    node_stack = [(input_node, None) for input_node in input_nodes]  # stack annotated with any top-level loops.
    enclosing_loop_nodes = defaultdict(lambda: [])

    while len(node_stack) > 0:
        node_tuple = node_stack.pop(0)
        node, enclosing_loop_node = node_tuple

        node_could_repeat = any([node['has_params'], node['is_bottom_level_module_output']])

        if all([node_could_repeat, node['total_passes'] > 1, enclosing_loop_node is None]):
            #  We've hit the start of an enclosing loop, make that loop the label.
            children_enclosing_loop = node['layer_barcode']
            enclosing_loop_nodes[node['layer_barcode']].append(node)
        elif (node['layer_barcode'] == enclosing_loop_node) and (node['pass_num'] < node['total_passes']):
            # We've hit another occurrence of the enclosing loop.
            children_enclosing_loop = enclosing_loop_node
            enclosing_loop_nodes[node['layer_barcode']].append(node)
        elif (node['layer_barcode'] == enclosing_loop_node) and (node['pass_num'] == node['total_passes']):
            # We've hit the end of an enclosing loop, remove that as the label.
            enclosing_loop_nodes[node['layer_barcode']].append(node)
            children_enclosing_loop = None
        else:  # We're inside an enclosing loop.
            children_enclosing_loop = enclosing_loop_node

        nodes_to_add = [(child_node, children_enclosing_loop) for child_node in node['child_tensor_barcodes']]
        node_stack.extend(nodes_to_add)

    return enclosing_loop_nodes


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
    # First check if they've diverged.
    pass_pair_key = tuple(sorted([pass1_start_barcode, pass2_start_barcode]))
    if pass_pair_key in diverged_pass_pairs:  # they've already diverged
        return False
    pass1_funcs = pass1_nth_item['funcs_applied']
    pass2_funcs = pass2_nth_item['funcs_applied']
    if pass1_funcs != pass2_funcs:  # the two passes aren't the same.
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

    for p1, (pass1_start_barcode, pass1_nth_item) in nth_item_per_stack:
        for p2, (pass2_start_barcode, pass2_nth_item) in nth_item_per_stack:
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
            same_layer_barcode_lists[layer_barcode].append(pass2_nth_item['barcode'])

    return same_layer_barcode_lists


def mark_corresponding_pass_layers(nth_item_per_stack: Dict, diverged_pass_pairs):
    """

    Args:
        nth_item_per_stack: Dictionary where each key is the starting barcode for the pass,
            and the value is the nth item of the stack for that pass.
        diverged_pass_pairs: Pairs of passes that have already diverged.

    Returns:
        Nothing
    """
    # Now we can go through and assign nodes as the same layer if they have the same functions, and
    # if the passes haven't already diverged.

    same_layer_barcode_lists = group_matching_layers(nth_item_per_stack, diverged_pass_pairs)

    # Now we have which of the nodes in each pass correspond and can mark them as such.

    for layer_barcode, node_list in same_layer_barcode_lists.items():
        num_passes = len(node_list)
        for p, node in enumerate(node_list):
            node['layer_barcode'] = layer_barcode
            node['pass_num'] = p + 1
            node['total_passes'] = num_passes


def update_pass_stack(pass_barcode: str,
                      pass_stack: List[Dict],
                      new_pass_stacks: Dict,
                      nodes_seen: set,
                      tensor_log: dict):
    """Utility function to update a single pass stack.

    Args:
        pass_barcode: Barcode for the start of the pass.
        pass_stack: The original stack for the pass.
        new_pass_stacks: Dictionary of updated stacks.
        nodes_seen: Set of nodes seen so far
        tensor_log: Log of tensors

    Returns:
        Nothing, but updates the new_pass_stacks dictionary.
    """
    for node in pass_stack:
        for child_barcode in node['child_tensor_barcodes']:
            if child_barcode not in nodes_seen:
                new_pass_stacks[pass_barcode]['children'].append(tensor_log[child_barcode])
                nodes_seen.add(child_barcode)
        for parent_barcode in node['parent_tensor_internal_barcodes']:
            if parent_barcode not in nodes_seen:
                new_pass_stacks[pass_barcode]['parents'].append(tensor_log[parent_barcode])
                nodes_seen.add(parent_barcode)


def update_pass_stacks(pass_stacks: Dict,
                       nodes_seen: set,
                       tensor_log) -> Dict:
    """Update the stack for each pass.

    Args:
        pass_stacks: Dictionary where the key is the barcode for the beginning of each pass, and the value is the stack.
        nodes_seen: Set of barcodes of nodes seen.
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
                              new_pass_stacks,
                              nodes_seen,
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
        stack_nodes = pass_start_barcode[pass_stacks][node_type]
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
    while True:
        stack_lengths_total = [len(pass_stacks[barcode]['children']) + len(pass_stacks[barcode]['parents'])
                               for barcode in pass_stacks]
        if sum(stack_lengths_total[0:-1]) == 0:  # if any of the non-final stacks have run out, we're done.
            break
        for node_type in ['children', 'parents']:
            biggest_stack_size = max([len(pass_stacks[barcode][node_type]) for barcode in pass_stacks])
            for n in range(biggest_stack_size):  # iterate through the items in each stack and check for correspondence
                nth_item_per_stack = get_nth_item_per_stack(pass_stacks, node_type, n)
                mark_corresponding_pass_layers(nth_item_per_stack, diverged_pass_pairs)

        # Finally, set the new stacks for each pass; add any node children that haven't been seen yet,
        # and add any node parents that are internally generated and haven't been seen yet.

        pass_stacks = update_pass_stacks(pass_stacks, nodes_seen, tensor_log)


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
    """
    tensor_log = history_dict['tensor_log']
    layer_type_counter = defaultdict(lambda: 0)
    module_type_counter = defaultdict(lambda: 0)

    layer_counter = 0  # how many unique layers encountered
    module_counter = 0  # how many unique modules encountered

    layer_type_inds = {}  # maps layers to their index (i.e., how many times that layer type has occurred)
    module_type_inds = {}  # maps module layers to their index (i.e., how many times that layer type has occurred)

    for barcode, node in tensor_log.items():
        layer_barcode = node['layer_barcode']
        layer_type = node['funcs_applied_names'][0].strip('_')
        node['layer_type'] = layer_type

        if layer_barcode not in layer_type_inds:
            layer_type_counter[layer_type] += 1
            layer_counter += 1
            layer_type_count = layer_type_counter[layer_type]
            layer_type_inds[layer_barcode] = layer_type_count
        else:
            layer_type_count = layer_type_inds[layer_barcode]

        node['layer_type_ind'] = layer_type_count
        node['layer_total_ind'] = layer_counter
        pass_num = node['pass_num']

        node['layer_label'] = f'{layer_type}_{layer_type_count}_{layer_counter}:{pass_num}'

        if node['is_bottom_level_module_output']:
            node_module = node['linked_bottom_module']
            node_module_address = node_module.xray_module_address
            module_type = type(node['linked_bottom_module']).__name__
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


def subset_graph(history_dict: Dict, nodes_to_keep: List[str]) -> Dict:
    """Subsets the nodes of the graph, inheriting the parent/children of omitted nodes.

    Args:
        tensor_log: The input tensor log.
        nodes to keep: Addresses of other modules to keep that aren't bottom-level modules.
    Returns:
        output tensor_log with only the desired nodes.
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
    for visualization purposes.

    Args:
        tensor_log: The tensor log.

    Returns:
        Rolled-up tensor log.
    """
    raise NotImplementedError


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
        topological_sort_nodes,  # sort nodes topologically
        annotate_total_layer_passes,  # note the total passes of the param nodes
        identify_repeated_functions,  # find repeated functions between repeated param nodes
        annotate_node_names  # make the nodes names more human-readable
    ]

    for graph_transform in graph_transforms:
        history_dict = graph_transform(history_dict)

    # TODO: Figure out which fields to keep.

    # fields_to_keep = []  # ordered list of fields to keep; can remove irrelevant fields.
    # tensor_log = OrderedDict({k: OrderedDict({f: v[f] for f in fields_to_keep}) for k, v in tensor_log.items()})
    # history_dict['tensor_log'] = tensor_log

    return history_dict


def prettify_history_dict(history_dict: Dict) -> Dict:
    """Returns the final user-readable version of tensor_log, omitting layers with no saved activations.

    Args:
        history_dict: Input history_dict

    Returns:
        Nicely organized/labeled final dict.
    """
