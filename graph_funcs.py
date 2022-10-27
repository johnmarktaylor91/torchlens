# This module has functions for processing the computation graphs that come out of the other functions.

from collections import defaultdict
import torch
import copy
from graphlib import TopologicalSorter

from collections import OrderedDict
from typing import Dict, List
from util_funcs import make_barcode

#TODO annotate the graph log with useful metadata instead of making it denovo every time; e.g., the
#input and output nodes. Inputs, outputs, graph, dictionary of repeated nodes, counter of node types?

#Maybe tensor barcode, layer barcode?

#Get clear about when to do the barcodes vs the nodes themselves, be consistent.

#Get more consistent language for different kinds of barcodes, nodes, operations, etc.

def annotate_node_parents(history_dict: Dict) -> Dict:
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


def strip_irrelevant_nodes(history_dict: Dict) -> Dict:
    """Remove nodes that neither have the input as an ancestor, nor have the output as a descendant (i.e.,
    they're just floating irrelevantly).

    Args:
        history_dict: input history dict

    Returns:
        history_dict with irrelevant nodes stripped
    """
    tensor_log = history_dict['tensor_log']

    # First trace back from the output and make list of barcodes for nodes that have output as descendant.


    output_ancestors = []
    node_stack = [node for node in tensor_log if tensor_log['is_model_output']]
    while len(node_stack) > 0:
        node = node_stack.pop()
        output_ancestors.append(node['barcode'])
        node_stack.extend(node['parent_tensor_barcodes'])

    # Now remove all nodes that are neither ancestors of the output, nor descendants of the input.

    new_tensor_log = {}
    for barcode, node in tensor_log.items():
        if (barcode in output_ancestors) or (node['origin'] == 'input'):
            new_tensor_log[barcode] = node

    history_dict['tensor_log'] = new_tensor_log
    return new_tensor_log





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

    node_stack = [node for node in tensor_log.values() if node['is_model_input']]
    nodes_seen = []
    converge_nodes = []
    node_count = 0
    module_node_count = 0
    ordered_tensor_log = OrderedDict({})  # re-order the tensor log in topological order.
    while len(node_stack) > 0 and len(converge_nodes) > 0:
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
        node['topological_sort_rank_exhaustive'] = node_count
        ordered_tensor_log[node['barcode']] = node

        if node['is_module_output']:  # if it's a module output, also annotate the order for that.
            module_node_count += 1
            node['topological_sort_rank_module'] = module_node_count
        else:
            node['topological_sort_rank_module'] = None

        nodes_seen.append(node['barcode'])
        node_stack.extend([tensor_log[barcode] for barcode in node['child_tensor_barcodes']])

    history_dict['tensor_log'] = ordered_tensor_log
    return history_dict


def annotate_total_param_passes(history_dict: Dict) -> Dict:
    """Annotates tensors in the log with the total number of passes their layer goes through in the network
    if they are outputs of a function with parameters.

    Args:
        history_dict: history_dict

    Returns:
        history_dict with tensors tagged with the total number of passes
    """
    tensor_log = history_dict['tensor_log']
    param_group_tensors = history_dict['param_group_tensors']
    for param_group_barcode, tensors in param_group_tensors.items():
        param_group_num_passes = len(tensors)
        for tensor_barcode in tensors:
            tensor_log[tensor_barcode]['layer_total_passes'] = param_group_num_passes
    return history_dict

def find_outermost_loops(history_dict: Dict) -> Dict: 
    """Utility function that tags the top-level loops in the graph. Specifically, it goes until it finds a 
    node that repeats, then gets all isntances of that repeated node, then finds the next one and does the same, etc.
    Returns a dictionary where each key is the layer barcode, and each value is the list of tensor
    barcodes for that layer.
    
    Args:
        history_dict: Input history dict 

    Returns:
        Dictionary of repeated occurrences of each layer corresponding to the outermost recurrent loops.
    """
    #TODO: simplify using the fact that we've saved repeated occurrences of each layer

    input_nodes = history_dict['input_tensors']

    node_stack = [(input_node, None) for input_node in input_nodes]  # stack annotated with any top-level loops.
    enclosing_loop_nodes = defaultdict(lambda: [])

    while len(node_stack) > 0:
        node_tuple = node_stack.pop(0)
        node, enclosing_loop_node = node_tuple

        if all[(node['has_params'], node['total_passes']>1, enclosing_loop_node is None]):
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
    input_nodes = history_dict['input_tensors']

    # Finally tag corresponding functions as being the same.
    # Algorithm: have a stack for each pass of the node, and for functions that are the same across stacks,
    # tag with the pass number, and the barcode of the first node where that function occurs. This is
    # done with respect to the OUTERMOST loop (I don't know how else to do it).

    # Now that we have the biggest loops, we can label corresponding functions in each pass.
    #TODO: there's some more headache logic here. For each item in the stack for each pass,
    #annotate it with which passes that thread has diverged from. Modularize this code too, it's hideous.


    for layer_barcode, repeated_node_occurrences in enclosing_loop_nodes:  # go through each outer loop
        # get the starting node for each pass of the loop
        pass_stacks = {node['barcode']: [(node, [])] for node in repeated_node_occurrences}
        while True:
            stack_lengths = [len(pass_stacks[barcode]) for barcode in pass_stacks]
            biggest_stack_size = max(stack_lengths)
            if biggest_stack_size == 0:
                break
            for n in range(biggest_stack_size):  # iterate through the items in each stack and check for correspondence
                nth_item_per_stack = {}
                for pass_start_barcode, stack_nodes in pass_stacks.items():
                    stack_size = len(stack_nodes)
                    if stack_size >= n+1:
                        stack_nth_item = stack_nodes[n]
                    else:
                        stack_nth_item = None
                    nth_item_per_stack[pass_start_barcode] = stack_nth_item

                # Now we can go through and assign nodes as the same layer if they have the same functions, and
                # if the passes haven't already diverged.

                # these will be the barcodes for the newly coined layers, and which nodes are instances of that layer.
                same_layer_barcodes = defaultdict(lambda: list())

                for p1, (pass1_start_barcode, pass1_nth_item) in nth_item_per_stack:
                    for p2, (pass2_start_barcode, pass2_nth_item) in nth_item_per_stack:
                        if p2 >= p1:
                            continue
                        # First check if they've diverged.
                        pass_pair_key = tuple(sorted([pass1_start_barcode, pass2_start_barcode]))
                        if pass_pair_key in diverged_pass_pairs:  # they've already diverged
                            continue
                        pass1_funcs = pass1_nth_item['funcs_applied']
                        pass2_funcs = pass2_nth_item['funcs_applied']
                        if pass1_funcs == pass2_funcs: # the two passes aren't the same.
                            diverged_pass_pairs.add(pass_pair_key)




    return history_dict

def annotate_node_names(history_dict: Dict) -> Dict:
    """Given a tensor log with the topological sorting applied, annotates the log with all the different labels
    for each node. The graph will be traversed backwards, so it can keep track of the highest number of passes and
    add this info, too. This is where everything should be neatly named.


    Args:
        history_dict: input history_dict

    Returns:
        tensor_log with all the node names annotated.
    """
    tensor_log = history_dict['tensor_log']

    # Keep a dict of the nodes for each parent params, and go back and annotate these nodes with the total number
    # of passes at the end.

    parent_param_nodes = {}

    for node in tensor_log:


def subset_graph(history_dict: Dict, nodes_to_keep: List[str]) -> Dict:
    """Subsets the nodes of the graph, inheriting the parent/children of omitted nodes.

    Args:
        tensor_log: The input tensor log.
        nodes to keep: Addresses of other modules to keep that aren't bottom-level modules.
    Returns:
        output tensor_log with only the desired nodes.
    """


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
    """Converts the graph to rolled-up format for plotting purposes.

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
    tensor_log = history_dict['tensor_log']

    # List of transforms to apply.
    # TODO: Figure out how much time this part takes, how many graph_traversals

    graph_transforms = [annotate_node_parents,
                        strip_irrelevant_nodes,
                        expand_multiple_functions,
                        topological_sort_nodes,
                        annotate_total_param_passes,
                        annotate_node_names]

    for graph_transform in graph_transforms:
        history_dict = graph_transform(history_dict)

    fields_to_keep = []  # ordered list of fields to keep; can remove irrelevant fields.
    tensor_log = OrderedDict({k: OrderedDict({f: v[f] for f in fields_to_keep}) for k, v in tensor_log.items()})


def prettify_history_dict(history_dict: Dict) -> Dict:
    """Returns the final user-readable version of tensor_log, omitting layers with no saved activations.

    Args:
        history_dict: Input history_dict

    Returns:
        Nicely organized/labeled final dict.
    """
