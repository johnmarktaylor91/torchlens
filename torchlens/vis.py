from collections import OrderedDict, defaultdict
from typing import Dict, List

import graphviz
from IPython.display import display

from torchlens.helper_funcs import human_readable_size, in_notebook, int_list_to_compact_str


def render_graph(history_dict: Dict,
                 vis_opt: str = 'unrolled',
                 vis_outpath: str = 'graph.gv') -> None:
    """Given the history_dict, renders the computational graph.

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
    num_tensors = len(history_dict['tensor_log'])
    graph_caption = (
        f"<<B>{history_dict['model_name']}</B><br align='left'/>{num_tensors} tensors total ({total_tensor_fsize})"
        f"<br align='left'/>{total_params} params total ({total_params_fsize})<br align='left'/>>")

    dot = graphviz.Digraph(name='model', comment='Computational graph for the feedforward sweep',
                           filename=vis_outpath, format='pdf')
    dot.graph_attr.update({'rankdir': 'BT',
                           'label': graph_caption,
                           'labelloc': 't',
                           'labeljust': 'left',
                           'ordering': 'out'})
    dot.node_attr.update({'shape': 'box', 'ordering': 'out'})

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

    # Finally, add extra invisible edges to enforce horizontal node order if there's branching.

    # align_sibling_nodes(dot, vis_opt, tensor_log)

    if in_notebook():
        display(dot)
        dot.render(vis_outpath, view=False)
    else:
        dot.render(vis_outpath, view=True)


def add_node_to_graphviz(node_barcode: str,
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
        node_address = '<br/>@' + node['modules_exited'][0]
        node_address = f"{node_address}"
        last_module_total_passes = node['module_total_passes']
        if (last_module_total_passes > 1) and (vis_opt == 'unrolled'):
            last_module_num_passes = node['module_passes_exited'][0][1]
            node_address += f":{last_module_num_passes}"
        node_shape = 'box'
        node_color = 'black'
    elif node['is_buffer_tensor']:
        node_address = '<br/>@' + node['buffer_address']
        node_shape = 'box'
        node_color = BUFFER_NODE_COLOR
    else:
        node_address = ''
        node_shape = 'oval'
        node_color = 'black'

    if node['is_model_input']:
        bg_color = INPUT_COLOR
    elif node['is_model_output']:
        bg_color = OUTPUT_COLOR
    elif node['output_is_terminal_bool']:
        bg_color = BOOL_NODE_COLOR
    elif node['has_params']:
        bg_color = PARAMS_NODE_BG_COLOR
    else:
        bg_color = DEFAULT_BG_COLOR

    tensor_shape = node['tensor_shape']
    layer_type = node['layer_type']
    layer_type_str = layer_type.replace('_', '')
    layer_type_ind = node['layer_type_ind']
    layer_total_ind = node['layer_total_ind']
    if node['has_input_ancestor']:
        line_style = 'solid'
    else:
        line_style = 'dashed'
    if (node['param_total_passes'] > 1) and (vis_opt == 'unrolled'):
        pass_num = node['pass_num']
        pass_label = f":{pass_num}"
    elif (node['param_total_passes'] > 1) and (vis_opt == 'rolled'):
        pass_label = f' (x{node["param_total_passes"]})'
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

    if node['output_is_terminal_bool']:
        label_text = str(node['output_bool_val']).upper()
        bool_label = f"<b><u>{label_text}:</u></b><br/><br/>"
    else:
        bool_label = ''

    node_label = (f'<{bool_label}{node_title}<br/>{tensor_shape_str} '
                  f'({tensor_fsize}){param_label}{node_address}>')

    graphviz_graph.node(name=node_barcode,
                        label=f"{node_label}",
                        fontcolor=node_color,
                        color=node_color,
                        style=f"filled,{line_style}",
                        fillcolor=bg_color,
                        shape=node_shape,
                        ordering='out')

    if vis_opt == 'rolled':
        add_rolled_edges_for_node(node, graphviz_graph, module_cluster_dict, tensor_log)
    else:
        for child_barcode in node['child_tensor_barcodes']:
            child_node = tensor_log[child_barcode]
            if node['has_input_ancestor']:
                edge_style = 'solid'
            else:
                edge_style = 'dashed'
            edge_dict = {'tail_name': node_barcode,
                         'head_name': child_barcode,
                         'color': node_color,
                         'style': edge_style,
                         'arrowsize': '.7'}

            if child_barcode in node['cond_branch_start_children']:  # Mark with "if" if the edge starts a cond branch
                edge_dict['label'] = '<<FONT POINT-SIZE="18"><b><u>IF</u></b></FONT>>'

            # Label the arguments to the next node if multiple inputs: TODO make this a function and allow same arg to appear multiple times
            child_node_layer_type = child_node['layer_type'].replace('_', '')
            if (len(child_node['parent_tensor_barcodes']) > 1) and (child_node_layer_type not in commute_funcs):
                found_it = False
                for arg_type in ['args', 'kwargs']:
                    for arg_loc, arg_barcode in child_node['parent_tensor_arg_locs'][arg_type].items():
                        if arg_barcode == node_barcode:
                            arg_label = arg_type[:-1] + ' ' + str(arg_loc)
                            arg_label = f"<<FONT POINT-SIZE='10'><b>{arg_label}</b></FONT>>"
                            if 'label' not in edge_dict:
                                edge_dict['label'] = arg_label
                            else:
                                edge_dict['label'] = edge_dict['label'] + '\n' + arg_label
                            found_it = True
                            break
                        if found_it:
                            break

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

def roll_graph(history_dict: Dict) -> Dict:
    """Converts the graph to rolled-up format for plotting purposes. This means that the nodes of the graph
    are now not tensors, but layers, with the layer children and parents for each pass indicated.
    The convention is that the pass numbers will be visually indicated for a layer if any one of the children
    or parent layers vary on different passes; if the same, then they won't be.
    This is only done for visualization purposes; no tensor data is saved.

    Args:
        history_dict: The history_dict

    Returns:
        Rolled-up tensor log.
    """

    fields_to_copy = ['layer_barcode', 'layer_type', 'layer_type_ind', 'layer_total_ind',
                      'is_model_input', 'is_model_output', 'is_last_output_layer',
                      'connects_input_and_output', 'has_input_ancestor', 'parent_tensor_arg_locs',
                      'cond_branch_start_children', 'output_is_terminal_bool', 'in_cond_branch', 'output_bool_val',
                      'is_buffer_tensor', 'buffer_address', 'tensor_shape', 'tensor_fsize',
                      'has_params', 'param_total_passes', 'parent_params_shape',
                      'is_bottom_level_module_output', 'function_call_modules_nested', 'modules_exited',
                      'module_total_passes', 'module_instance_final_barcodes_list', 'funcs_applied']

    tensor_log = history_dict['tensor_log']
    rolled_tensor_log = OrderedDict({})

    tensor_to_layer_dict = {}

    for node_barcode, node in tensor_log.items():
        # Get relevant information from each node.

        layer_barcode = node['layer_barcode']
        tensor_to_layer_dict[node_barcode] = layer_barcode
        if layer_barcode in rolled_tensor_log:
            rolled_node = rolled_tensor_log[layer_barcode]
        else:
            rolled_node = OrderedDict({})
            for field in fields_to_copy:
                if field in node:
                    rolled_node[field] = node[field]
            rolled_node['child_layer_barcodes'] = {}  # each key is pass_num, each value list of children
            rolled_node['parent_layer_barcodes'] = {}  # each key is pass_num, each value list of parents
            rolled_node['args_per_layer'] = len(node['parent_tensor_barcodes'])
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

    # Add an indicator of whether any of the child or parent layers vary based on the pass

    for layer_barcode, rolled_node in rolled_tensor_log.items():
        edges_vary_across_passes = False
        for pass_num in rolled_node['child_layer_barcodes']:
            if pass_num == 1:
                child_layer_barcodes = rolled_node['child_layer_barcodes'][pass_num]
                parent_layer_barcodes = rolled_node['parent_layer_barcodes'][pass_num]
                continue
            elif any([rolled_node['child_layer_barcodes'][pass_num] != child_layer_barcodes,
                      rolled_node['parent_layer_barcodes'][pass_num] != parent_layer_barcodes]):
                edges_vary_across_passes = True
                break
            child_layer_barcodes = rolled_node['child_layer_barcodes'][pass_num]
            parent_layer_barcodes = rolled_node['parent_layer_barcodes'][pass_num]
        rolled_node['edges_vary_across_passes'] = edges_vary_across_passes

    # Finally add a complementary data structure that for each child or parent edge, indicates the passes for those edges.

    for layer_barcode, rolled_node in rolled_tensor_log.items():
        child_edge_passes = defaultdict(list)
        parent_edge_passes = defaultdict(list)
        for pass_num in rolled_node['child_layer_barcodes']:
            for child_layer_barcode in rolled_node['child_layer_barcodes'][pass_num]:
                child_edge_passes[child_layer_barcode].append(pass_num)
            for parent_layer_barcode in rolled_node['parent_layer_barcodes'][pass_num]:
                parent_edge_passes[parent_layer_barcode].append(pass_num)
        rolled_node['child_layer_passes'] = child_edge_passes
        rolled_node['parent_layer_passes'] = parent_edge_passes

    # And replace the arg locations with the layer, rather than tensor barcodes.

    for layer_barcode, rolled_node in rolled_tensor_log.items():
        new_parent_arg_locs = {'args': {}, 'kwargs': {}}
        for arg_type in ['args', 'kwargs']:
            for arg_loc, arg_barcode in rolled_node['parent_tensor_arg_locs'][arg_type].items():
                new_parent_arg_locs[arg_type][arg_loc] = tensor_to_layer_dict[arg_barcode]
        rolled_node['parent_tensor_arg_locs'] = new_parent_arg_locs

    return rolled_tensor_log


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
    parent_node_barcode = node['layer_barcode']

    for child_layer_barcode, parent_pass_nums in node['child_layer_passes'].items():
        child_node = tensor_log[child_layer_barcode]
        child_pass_nums = child_node['parent_layer_passes'][parent_node_barcode]

        # Annotate passes for the child and parent nodes for the edge only if they vary across passes.

        if node['edges_vary_across_passes']:
            tail_label = f'  Out {int_list_to_compact_str(parent_pass_nums)}  '
        else:
            tail_label = ''

        # Mark the head label with the argument if need be:

        if child_node['edges_vary_across_passes'] and not child_node['is_model_output']:
            head_label = f'  In {int_list_to_compact_str(child_pass_nums)}  '
        else:
            head_label = ''

        if node['has_input_ancestor']:
            edge_style = 'solid'
        else:
            edge_style = 'dashed'

        if node['is_buffer_tensor']:
            node_color = BUFFER_NODE_COLOR
        else:
            node_color = 'black'

        edge_dict = {'tail_name': node['layer_barcode'],
                     'head_name': child_layer_barcode,
                     'color': node_color,
                     'fontcolor': node_color,
                     'style': edge_style,
                     'headlabel': head_label,
                     'taillabel': tail_label,
                     'arrowsize': '.7',
                     'labelfontsize': '8'}

        if child_layer_barcode in node['cond_branch_start_children']:  # Mark with "if" if the edge starts a cond branch
            edge_dict['label'] = '<<FONT POINT-SIZE="18"><b><u>IF</u></b></FONT>>'

        # Label the arguments to the next node if multiple inputs: TODO make this a function
        child_node_layer_type = child_node['layer_type'].replace('_', '')
        if (child_node['args_per_layer'] > 1) and (child_node_layer_type not in commute_funcs):
            found_it = False
            for arg_type in ['args', 'kwargs']:
                for arg_loc, arg_barcode in child_node['parent_tensor_arg_locs'][arg_type].items():
                    if arg_barcode == parent_node_barcode:
                        arg_label = arg_type[:-1] + ' ' + str(arg_loc)
                        arg_label = f"<<FONT POINT-SIZE='10'><b>{arg_label}</b></FONT>>"
                        if 'label' not in edge_dict:
                            edge_dict['label'] = arg_label
                        else:
                            edge_dict['label'] = edge_dict['label'] + '\n' + arg_label
                        found_it = True
                        break
                    if found_it:
                        break

        containing_module = get_lowest_containing_module_for_two_nodes(node, child_node)
        if containing_module != -1:
            module_cluster_dict[containing_module].append(edge_dict)
        else:
            graphviz_graph.edge(**edge_dict)


def set_up_subgraphs(graphviz_graph,
                     module_cluster_dict: Dict[str, List],
                     history_dict: Dict):
    """Given a dictionary specifying the edges in each cluster, the graphviz graph object, and the history_dict,
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

    # Get the max nesting depth; it'll be the depth of the deepest module that has no edges.

    max_nesting_depth = get_max_nesting_depth(subgraphs,
                                              module_cluster_dict,
                                              module_cluster_children_dict)

    subgraph_stack = [[subgraph_tuple] for subgraph_tuple in subgraphs]
    nesting_depth = 0
    while len(subgraph_stack) > 0:
        parent_graph_list = subgraph_stack.pop(0)
        setup_subgraphs_recurse(graphviz_graph,
                                parent_graph_list,
                                module_cluster_dict,
                                module_cluster_children_dict,
                                subgraph_stack,
                                nesting_depth,
                                max_nesting_depth,
                                history_dict)


def setup_subgraphs_recurse(starting_subgraph,
                            parent_graph_list: List,
                            module_cluster_dict,
                            module_cluster_children_dict,
                            subgraph_stack,
                            nesting_depth,
                            max_nesting_depth,
                            history_dict):
    """Utility function to crawl down several layers deep into nested subgraphs.

    Args:
        starting_subgraph: The subgraph we're starting from.
        parent_graph_list: List of parent graphs.
        module_cluster_dict: Dict mapping each cluster to its edges.
        module_cluster_children_dict: Dict mapping each cluster to its subclusters.
        subgraph_stack: Stack of subgraphs to look at.
        nesting_depth: Nesting depth so far.
        max_nesting_depth: The total depth of the subgraphs.
        history_dict: The history dict
    """
    subgraph_tuple = parent_graph_list[nesting_depth]
    subgraph_module, subgraph_barcode = subgraph_tuple
    subgraph_name = '_'.join(subgraph_tuple)
    cluster_name = f"cluster_{subgraph_name}"
    module_type = str(type(history_dict['module_dict'][subgraph_module]).__name__)

    if nesting_depth < len(parent_graph_list) - 1:  # we haven't gotten to the bottom yet, keep going.
        with starting_subgraph.subgraph(name=cluster_name) as s:
            setup_subgraphs_recurse(s, parent_graph_list,
                                    module_cluster_dict, module_cluster_children_dict, subgraph_stack,
                                    nesting_depth + 1, max_nesting_depth, history_dict)

    else:  # we made it, make the subgraph and add all edges.
        with starting_subgraph.subgraph(name=cluster_name) as s:
            pen_width = MIN_MODULE_PENWIDTH + ((max_nesting_depth - nesting_depth) / max_nesting_depth) * PENWIDTH_RANGE
            s.attr(
                label=f"<<B>@{subgraph_module}</B><br align='left'/>({module_type})<br align='left'/>>",
                labelloc='b',
                penwidth=str(pen_width))
            subgraph_edges = module_cluster_dict[subgraph_tuple]
            for edge_dict in subgraph_edges:
                s.edge(**edge_dict)
            subgraph_children = module_cluster_children_dict[subgraph_tuple]
            for subgraph_child in subgraph_children:  # it's weird but have to go in reverse order.
                subgraph_stack.append(parent_graph_list[:] + [subgraph_child])


def get_lowest_containing_module_for_two_nodes(node1: Dict,
                                               node2: Dict):
    """Utility function to get the lowest-level module that contains two nodes, to know where to put the edge.

    Args:
        node1: The first node.
        node2: The second node.

    Returns:
        Barcode of lowest-level module containing both nodes.
    """
    node1_modules = node1['function_call_modules_nested']
    node2_modules = node2['function_call_modules_nested']

    if (len(node1_modules) == 0) or (len(node2_modules) == 0) or (node1_modules[0] != node2_modules[0]):
        return -1  # no submodule contains them both.

    containing_module = node1_modules[0]
    for m in range(min([len(node1_modules), len(node2_modules)])):
        if node1_modules[m] != node2_modules[m]:
            break
        containing_module = node1_modules[m]
    return containing_module


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

        if (len(module_edges) == 0) and (len(module_children) == 0):  # can ignore if no edges and no children.
            continue
        elif (len(module_edges) > 0) and (len(module_children) == 0):
            max_nesting_depth = max([module_depth, max_nesting_depth])
        elif (len(module_edges) == 0) and (len(module_children) > 0):
            module_stack.extend([(module_child, module_depth + 1) for module_child in module_children])
        else:
            max_nesting_depth = max([module_depth, max_nesting_depth])
            module_stack.extend([(module_child, module_depth + 1) for module_child in module_children])
    return max_nesting_depth
