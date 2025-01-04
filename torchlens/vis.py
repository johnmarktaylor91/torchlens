from collections import defaultdict
from typing import Dict, List, Set, TYPE_CHECKING, Tuple, Union

import graphviz
from IPython.display import display

from .helper_funcs import in_notebook, int_list_to_compact_str
from .postprocess import _roll_graph
from .tensor_log import RolledTensorLogEntry, TensorLogEntry

if TYPE_CHECKING:
    from .model_history import ModelHistory

INPUT_COLOR = "#98FB98"
OUTPUT_COLOR = "#ff9999"
PARAMS_NODE_BG_COLOR = "#E6E6E6"
BUFFER_NODE_COLOR = "#888888"
GRADIENT_ARROW_COLOR = "#9197F6"
DEFAULT_BG_COLOR = "white"
BOOL_NODE_COLOR = "#F7D460"
MAX_MODULE_PENWIDTH = 5
MIN_MODULE_PENWIDTH = 2
PENWIDTH_RANGE = MAX_MODULE_PENWIDTH - MIN_MODULE_PENWIDTH
COMMUTE_FUNCS = ["add", "mul", "cat", "eq", "ne"]


def render_graph(
        self: "ModelHistory",
        vis_opt: str = "unrolled",
        vis_nesting_depth: int = 1000,
        vis_outpath: str = "modelgraph",
        vis_graph_overrides: Dict = None,
        vis_node_overrides: Dict = None,
        vis_nested_node_overrides: Dict = None,
        vis_edge_overrides: Dict = None,
        vis_gradient_edge_overrides: Dict = None,
        vis_module_overrides: Dict = None,
        save_only: bool = False,
        vis_fileformat: str = "pdf",
        show_buffer_layers: bool = False,
        direction: str = "bottomup",
) -> None:
    """Renders the computational graph for the model.

    Args:
        vis_opt: either 'rolled' or 'unrolled'
        vis_nesting_depth: How many levels of nested modules to show; 1 for only top-level modules, 2 for two
            levels, etc.
        vis_outpath: where to store the rendered graph
        save_only: whether to only save the graph without immediately showing it
        vis_fileformat: file format to use for the rendered graph
        show_buffer_layers: whether to show the buffer layers
        direction: which way the graph should go: either 'bottomup', 'topdown', or 'leftright'

    """
    if vis_graph_overrides is None:
        vis_graph_overrides = {}
    if vis_node_overrides is None:
        vis_node_overrides = {}
    if vis_nested_node_overrides is None:
        vis_nested_node_overrides = {}
    if vis_edge_overrides is None:
        vis_edge_overrides = {}
    if vis_gradient_edge_overrides is None:
        vis_gradient_edge_overrides = {}
    if vis_module_overrides is None:
        vis_module_overrides = {}

    if not self._all_layers_logged:
        raise ValueError(
            "Must have all layers logged in order to render the graph; either save all layers,"
            "set keep_unsaved_layers to True, or use show_model_graph."
        )

    # Fix the filename if need be, to remove the extension:
    split_outpath = vis_outpath.split(".")
    if split_outpath[-1] in [
        "pdf",
        "png",
        "jpg",
        "svg",
        "jpg",
        "jpeg",
        "bmp",
        "pic",
        "tif",
        "tiff",
    ]:
        vis_outpath = ".".join(split_outpath[:-1])

    if vis_opt == "unrolled":
        entries_to_plot = self.layer_dict_main_keys
    elif vis_opt == "rolled":
        _roll_graph(self)
        entries_to_plot = self.layer_dict_rolled
    else:
        raise ValueError("vis_opt must be either 'rolled' or 'unrolled'")

    if direction == "bottomup":
        rankdir = "BT"
    elif direction == "leftright":
        rankdir = "LR"
    elif direction == "topdown":
        rankdir = "TB"
    else:
        raise ValueError(
            "direction must be either 'bottomup', 'topdown', or 'leftright'."
        )

    graph_caption = (
        f"<<B>{self.model_name}</B><br align='left'/>{self.num_tensors_total} "
        f"tensors total ({self.tensor_fsize_total_nice})"
        f"<br align='left'/>{self.total_params} params total ({self.total_params_fsize_nice})<br align='left'/>>"
    )

    dot = graphviz.Digraph(
        name=self.model_name,
        comment="Computational graph for the feedforward sweep",
        format=vis_fileformat,
    )

    graph_args = {'rankdir': rankdir,
                  'label': graph_caption,
                  'labelloc': 't',
                  'labeljust': 'left',
                  'ordering': 'out'}

    for arg_name, arg_val in vis_graph_overrides.items():
        if callable(arg_val):
            graph_args[arg_name] = str(arg_val(self))
        else:
            graph_args[arg_name] = str(arg_val)

    dot.graph_attr.update(graph_args)
    dot.node_attr.update({"ordering": "out"})

    # list of edges for each subgraph; subgraphs will be created at the end.
    module_cluster_dict = defaultdict(
        lambda: {"edges": [], "has_input_ancestor": False}
    )
    collapsed_modules = set()
    edges_used = set()

    for node_barcode, node in entries_to_plot.items():
        if node.is_buffer_layer and not show_buffer_layers:
            continue
        _add_node_to_graphviz(
            self,
            node,
            dot,
            module_cluster_dict,
            edges_used,
            vis_opt,
            collapsed_modules,
            vis_nesting_depth,
            show_buffer_layers,
            vis_node_overrides,
            vis_nested_node_overrides,
            vis_edge_overrides,
            vis_gradient_edge_overrides
        )

    # Finally, set up the subgraphs.
    _set_up_subgraphs(self, dot, vis_opt, module_cluster_dict, vis_module_overrides)

    if in_notebook() and not save_only:
        display(dot)

    dot.render(vis_outpath, view=(not save_only))


def _add_node_to_graphviz(
        self: "ModelHistory",
        node: Union["TensorLogEntry", "RolledTensorLogEntry"],
        graphviz_graph,
        module_edge_dict: Dict,
        edges_used: Set,
        vis_opt: str,
        collapsed_modules: Set,
        vis_nesting_depth: int = 1000,
        show_buffer_layers: bool = False,
        vis_node_overrides: Dict = None,
        vis_collapsed_node_overrides: Dict = None,
        vis_edge_overrides: Dict = None,
        vis_gradient_edge_overrides: Dict = None
):
    """Addes a node and its relevant edges to the graphviz figure.

    Args:
        node: node to add
        graphviz_graph: The graphviz object to add the node to.
        module_edge_dict: Dictionary of the module clusters.
        vis_opt: Whether to roll the graph or not
        vis_nesting_depth: How many levels of nested modules to show
        collapsed_modules: Labels of collapsed module nodes that have been made so far.
        show_buffer_layers: Whether to show the buffer layers
    """
    is_collapsed_module = _check_if_collapsed_module(node, vis_nesting_depth)

    if is_collapsed_module:
        _construct_collapsed_module_node(
            self, node, graphviz_graph, collapsed_modules, vis_opt, vis_nesting_depth, vis_collapsed_node_overrides
        )
        node_color = "black"
    else:
        node_color = _construct_layer_node(
            self, node, graphviz_graph, show_buffer_layers, vis_opt, vis_node_overrides
        )

    _add_edges_for_node(
        self,
        node,
        is_collapsed_module,
        vis_nesting_depth,
        node_color,
        module_edge_dict,
        edges_used,
        graphviz_graph,
        vis_opt,
        show_buffer_layers,
        vis_edge_overrides,
        vis_gradient_edge_overrides
    )


def _check_if_collapsed_module(node, vis_nesting_depth):
    node_nesting_depth = len(node.containing_modules_origin_nested)
    if node.is_bottom_level_submodule_output:
        node_nesting_depth -= 1

    if node_nesting_depth >= vis_nesting_depth:
        return True
    else:
        return False


def _construct_layer_node(self: "ModelHistory",
                          node,
                          graphviz_graph,
                          show_buffer_layers,
                          vis_opt,
                          vis_node_overrides):
    # Get the address, shape, color, and line style:

    node_address, node_shape, node_color = _get_node_address_shape_color(
        self, node, show_buffer_layers
    )
    node_bg_color = _get_node_bg_color(self, node)

    if node.has_input_ancestor:
        line_style = "solid"
    else:
        line_style = "dashed"

    # Get the text for the node label:

    node_label = _make_node_label(node, node_address, vis_opt)

    node_args = {'name': node.layer_label.replace(":", "pass"),
                 'label': node_label,
                 'fontcolor': node_color,
                 'color': node_color,
                 'style': f'filled,{line_style}',
                 'fillcolor': node_bg_color,
                 'shape': node_shape,
                 'ordering': 'out'
                 }
    for arg_name, arg_val in vis_node_overrides.items():
        if callable(arg_val):
            node_args[arg_name] = str(arg_val(self, node))
        else:
            node_args[arg_name] = str(arg_val)

    graphviz_graph.node(**node_args)

    if node.is_last_output_layer:
        with graphviz_graph.subgraph() as s:
            s.attr(rank="sink")
            s.node(node.layer_label.replace(":", "pass"))

    return node_color


def _construct_collapsed_module_node(
        self: "ModelHistory",
        node,
        graphviz_graph,
        collapsed_modules,
        vis_opt,
        vis_nesting_depth,
        vis_collapsed_node_overrides
):
    module_address_w_pass = node.containing_modules_origin_nested[
        vis_nesting_depth - 1
        ]
    module_tuple = module_address_w_pass.split(":")
    module_output_layer = self[module_address_w_pass]
    module_output_shape = module_output_layer.tensor_shape
    module_output_fsize = module_output_layer.tensor_fsize_nice
    module_address, pass_num = module_tuple
    module_type = self.module_types[module_address]
    module_num_passes = self.module_num_passes[module_address]
    module_nparams = self.module_nparams[module_address]

    if vis_opt == "unrolled":
        node_name = "pass".join(module_tuple)
        module_num_tensors = self.module_pass_num_tensors[module_address_w_pass]
        module_has_input_ancestor = any(
            [
                self[layer].has_input_ancestor
                for layer in self.module_pass_layers[module_address_w_pass]
            ]
        )
    else:
        node_name = module_tuple[0]
        module_num_tensors = self.module_num_tensors[module_address]
        module_has_input_ancestor = any(
            [
                self[layer].has_input_ancestor
                for layer in self.module_layers[module_address]
            ]
        )

    if node_name in collapsed_modules:
        return  # collapsed node already added

    if module_num_passes == 1:
        node_title = f"<b>@{module_address}</b>"
    elif vis_opt == "unrolled" and (module_num_passes > 1):
        node_title = f"<b>@{module_address}:{pass_num}</b>"
    else:
        node_title = f"<b>@{module_address} (x{module_num_passes})</b>"

    if len(module_output_shape) > 1:
        tensor_shape_str = "x".join([str(x) for x in module_output_shape])
    elif len(node.tensor_shape) == 1:
        tensor_shape_str = f"x{module_output_shape[0]}"
    else:
        tensor_shape_str = "x1"

    if module_nparams > 0:
        bg_color = PARAMS_NODE_BG_COLOR
    else:
        bg_color = DEFAULT_BG_COLOR

    if module_has_input_ancestor:
        line_style = "solid"
    else:
        line_style = "dashed"

    node_label = (
        f"<{node_title}<br/>"
        f"{module_type}<br/>"
        f"{tensor_shape_str} ({module_output_fsize})<br/>"
        f"{module_num_tensors} layers total<br/>"
        f"{module_nparams} parameters>"
    )

    node_args = {'name': node_name,
                 'label': node_label,
                 'fontcolor': 'black',
                 'color': 'black',
                 'style': f'filled,{line_style}',
                 'fillcolor': bg_color,
                 'shape': 'box3d',
                 'ordering': 'out'
                 }

    for arg_name, arg_val in vis_collapsed_node_overrides.items():
        if callable(arg_val):
            node_args[arg_name] = str(arg_val(self, node))
        else:
            node_args[arg_name] = str(arg_val)

    graphviz_graph.node(**node_args)


def _get_node_address_shape_color(
        self: "ModelHistory",
        node: Union["TensorLogEntry", "RolledTensorLogEntry"],
        show_buffer_layers: bool,
) -> Tuple[str, str, str]:
    """Gets the node shape, address, and color for the graphviz figure.

    Args:
        node: node to add

    Returns:
        node_address: address of the node
        node_shape: shape of the node
        node_color: color of the node
    """
    if not show_buffer_layers:
        only_non_buffer_layer = _check_if_only_non_buffer_in_module(self, node)
    else:
        only_non_buffer_layer = False

    if (node.is_bottom_level_submodule_output or only_non_buffer_layer) and (
            len(node.containing_modules_origin_nested) > 0
    ):
        if type(node) == TensorLogEntry:
            module_pass_exited = node.containing_modules_origin_nested[-1]
            module, _ = module_pass_exited.split(":")
            if self.module_num_passes[module] == 1:
                node_address = module
            else:
                node_address = module_pass_exited
        else:
            sample_module_pass = node.containing_modules_origin_nested[-1]
            module = sample_module_pass.split(":")[0]
            node_address = module

        node_address = "<br/>@" + node_address
        node_shape = "box"
        node_color = "black"
    elif node.is_buffer_layer:
        if ((self.buffer_num_passes[node.buffer_address] == 1) or
                (isinstance(node, RolledTensorLogEntry) and node.layer_passes_total > 1)):
            buffer_address = node.buffer_address
        else:
            buffer_address = f"{node.buffer_address}:{node.buffer_pass}"
        node_address = "<br/>@" + buffer_address
        node_shape = "box"
        node_color = BUFFER_NODE_COLOR
    elif node.is_output_layer or node.is_input_layer:
        node_address = "<br/>@" + node.input_output_address
        node_shape = "oval"
        node_color = "black"
    else:
        node_address = ""
        node_shape = "oval"
        node_color = "black"

    return node_address, node_shape, node_color


def _check_if_only_non_buffer_in_module(
        self: "ModelHistory",
        node: Union["TensorLogEntry", "RolledTensorLogEntry"]
):
    """Utility function to check if a layer is the only non-buffer layer in the module"""
    # Check whether it leaves its module:
    if not (
            (len(node.modules_exited) > 0)
            and (len(node.containing_modules_origin_nested) > 0)
            and (
                    node.containing_modules_origin_nested[-1].split(":")[0]
                    in node.modules_exited
            )
    ):
        return False

    # Now check whether all of its parents are either buffers, or are outside the module.
    # If any aren't, return False.

    for parent_layer_label in node.parent_layers:
        if type(node) == TensorLogEntry:
            parent_layer = self[parent_layer_label]
        else:
            parent_layer = self.layer_dict_rolled[parent_layer_label]
        if (not parent_layer.is_buffer_layer) and (
                (len(parent_layer.containing_modules_origin_nested) > 0)
                and parent_layer.containing_modules_origin_nested[-1]
                == node.containing_modules_origin_nested[-1]
        ):
            return False

    return True


def _get_node_bg_color(
        self: "ModelHistory",
        node: Union["TensorLogEntry", "RolledTensorLogEntry"]
) -> str:
    """Gets the node background color for the graphviz figure.

    Args:
        node: node to add

    Returns:
        node_bg_color: background color of the node
    """
    if node.is_input_layer:
        bg_color = INPUT_COLOR
    elif node.is_output_layer:
        bg_color = OUTPUT_COLOR
    elif node.is_terminal_bool_layer:
        bg_color = BOOL_NODE_COLOR
    elif node.computed_with_params:
        bg_color = PARAMS_NODE_BG_COLOR
    else:
        bg_color = DEFAULT_BG_COLOR
    return bg_color


def _make_node_label(
        node: Union["TensorLogEntry", "RolledTensorLogEntry"],
        node_address: str,
        vis_opt: str,
) -> str:
    """Gets the text for the graphviz node."""
    # Pass info:

    if (node.layer_passes_total > 1) and (vis_opt == "unrolled"):
        pass_label = f":{node.pass_num}"
    elif (node.layer_passes_total > 1) and (vis_opt == "rolled"):
        pass_label = f" (x{node.layer_passes_total})"
    else:
        pass_label = ""

    # Tensor shape info:

    if len(node.tensor_shape) > 1:
        tensor_shape_str = "x".join([str(x) for x in node.tensor_shape])
    elif len(node.tensor_shape) == 1:
        tensor_shape_str = f"x{node.tensor_shape[0]}"
    else:
        tensor_shape_str = "x1"

    # Layer param info:

    param_label = _make_param_label(node)

    tensor_fsize = node.tensor_fsize_nice
    if node.layer_type in ["input", "output", "buffer"]:
        node_title = f"<b>{node.layer_type}_{node.layer_type_num}{pass_label}</b>"
    else:
        node_title = f"<b>{node.layer_type}_{node.layer_type_num}_{node.layer_total_num}{pass_label}</b>"

    if node.is_terminal_bool_layer:
        label_text = str(node.atomic_bool_val).upper()
        bool_label = f"<b><u>{label_text}:</u></b><br/><br/>"
    else:
        bool_label = ""

    node_label = (
        f"<{bool_label}{node_title}<br/>{tensor_shape_str} "
        f"({tensor_fsize}){param_label}{node_address}>"
    )

    return node_label


def _make_param_label(node: Union["TensorLogEntry", "RolledTensorLogEntry"]) -> str:
    """Makes the label for parameters of a node."""
    if node.num_param_tensors == 0:
        return ""

    each_param_shape = []
    for param_shape in node.parent_param_shapes:
        if len(param_shape) > 1:
            each_param_shape.append("x".join([str(s) for s in param_shape]))
        elif len(param_shape) == 1:
            each_param_shape.append(f"x{param_shape[0]}")
        else:
            each_param_shape.append("x1")

    param_label = "<br/>params: " + ", ".join(
        [param_shape for param_shape in each_param_shape]
    )
    return param_label


def _add_edges_for_node(
        self: "ModelHistory",
        parent_node: Union["TensorLogEntry", "RolledTensorLogEntry"],
        parent_is_collapsed_module: bool,
        vis_nesting_depth: int,
        node_color: str,
        module_edge_dict: Dict,
        edges_used: Set,
        graphviz_graph,
        vis_opt: str = "unrolled",
        show_buffer_layers: bool = False,
        vis_edge_overrides: Dict = None,
        vis_gradient_edge_overrides: Dict = None
):
    """Add the rolled-up edges for a node, marking for the edge which passes it happened for.

    Args:
        parent_node: The node to add edges for.
        parent_is_collapsed_module: Whether the node is a collapsed module node.
        vis_nesting_depth: How many levels of module nesting to show.
        node_color: Color of the node
        graphviz_graph: The graphviz graph object.
        module_edge_dict: Dictionary mapping each cluster to the edges it contains.
        edges_used: Edges used so far.
        vis_opt: Either 'unrolled' or 'rolled'
        show_buffer_layers: whether to show the buffer layers
    """
    for child_layer_label in parent_node.child_layers:
        if vis_opt == "unrolled":
            child_node = self.layer_dict_main_keys[child_layer_label]
        elif vis_opt == "rolled":
            child_node = self.layer_dict_rolled[child_layer_label]
        else:
            raise ValueError(
                f"vis_opt must be 'unrolled' or 'rolled', not {vis_opt}"
            )

        if child_node.is_buffer_layer and not show_buffer_layers:
            continue

        if parent_node.has_input_ancestor:
            edge_style = "solid"
        else:
            edge_style = "dashed"

        if parent_is_collapsed_module:
            module_name_w_pass = parent_node.containing_modules_origin_nested[
                vis_nesting_depth - 1
                ]
            module_tuple = module_name_w_pass.split(":")
            if vis_opt == "unrolled":
                tail_name = "pass".join(module_tuple)
            else:
                tail_name = module_tuple[0]
        else:
            tail_name = parent_node.layer_label.replace(":", "pass")

        child_is_collapsed_module = _check_if_collapsed_module(
            child_node, vis_nesting_depth
        )

        if child_is_collapsed_module:
            module_name_w_pass = child_node.containing_modules_origin_nested[
                vis_nesting_depth - 1
                ]
            module_tuple = module_name_w_pass.split(":")
            if vis_opt == "unrolled":
                head_name = "pass".join(module_tuple)
            else:
                head_name = module_tuple[0]
        else:
            head_name = child_node.layer_label.replace(":", "pass")

        both_nodes_collapsed_modules = (
                parent_is_collapsed_module and child_is_collapsed_module
        )

        # If both child and parent are in a collapsed module of the same pass, skip the edge:
        if both_nodes_collapsed_modules:
            child_containing_modules = child_node.containing_modules_origin_nested[
                                       :
                                       ]
            parent_containing_modules = (
                parent_node.containing_modules_origin_nested[:]
            )
            if child_node.is_bottom_level_submodule_output:
                child_containing_modules = child_containing_modules[:-1]
            if parent_node.is_bottom_level_submodule_output:
                parent_containing_modules = parent_containing_modules[:-1]
            if (
                    child_containing_modules[:vis_nesting_depth]
                    == parent_containing_modules[:vis_nesting_depth]
            ):
                continue

        # Skip repeated edges:
        if (tail_name, head_name) in edges_used:
            continue
        edges_used.add((tail_name, head_name))

        edge_dict = {
            "tail_name": tail_name,
            "head_name": head_name,
            "color": node_color,
            "fontcolor": node_color,
            "style": edge_style,
            "arrowsize": ".7",
            "labelfontsize": "8",
        }

        # Mark with "if" in the case edge starts a cond branch
        if (child_layer_label in parent_node.cond_branch_start_children) and (not child_is_collapsed_module):
            edge_dict["label"] = '<<FONT POINT-SIZE="18"><b><u>IF</u></b></FONT>>'

        # Label the arguments to the next node if multiple inputs
        if not child_is_collapsed_module:
            _label_node_arguments_if_needed(
                self, parent_node, child_node, edge_dict, show_buffer_layers
            )

        # Annotate passes for rolled node edge if it varies across passes
        if vis_opt == "rolled":
            _label_rolled_pass_nums(child_node, parent_node, edge_dict)

        for arg_name, arg_val in vis_edge_overrides.items():
            if callable(arg_val):
                edge_dict[arg_name] = str(arg_val(self, parent_node, child_node))
            else:
                edge_dict[arg_name] = str(arg_val)

        # Add it to the appropriate module cluster (most nested one containing both nodes)
        containing_module = _get_lowest_containing_module_for_two_nodes(
            parent_node, child_node, both_nodes_collapsed_modules, vis_nesting_depth
        )
        if containing_module != -1:
            module_edge_dict[containing_module]["edges"].append(edge_dict)
            if parent_node.has_input_ancestor or child_node.has_input_ancestor:
                module_edge_dict[containing_module]["has_input_ancestor"] = True
                for module in parent_node.containing_modules_origin_nested:
                    module_edge_dict[module]["has_input_ancestor"] = True
                    if module == containing_module:
                        break
                for module in child_node.containing_modules_origin_nested:
                    module_edge_dict[module]["has_input_ancestor"] = True
                    if module == containing_module:
                        break
        else:
            graphviz_graph.edge(**edge_dict)

        # Finally, add a backwards edge if both tensors have stored gradients.
        if vis_opt == "unrolled":
            _add_gradient_edge(
                self,
                parent_node,
                child_node,
                edge_style,
                containing_module,
                module_edge_dict,
                graphviz_graph,
                vis_gradient_edge_overrides
            )


def _label_node_arguments_if_needed(
        self: "ModelHistory",
        parent_node: Union["TensorLogEntry", "RolledTensorLogEntry"],
        child_node: Union["TensorLogEntry", "RolledTensorLogEntry"],
        edge_dict: Dict,
        show_buffer_layers: bool = False,
):
    """Checks if a node has multiple non-commutative arguments, and if so, adds labels in edge_dict

    Args:
        parent_node: parent node
        child_node: child node
        edge_dict: dict of information about the edge
        show_buffer_layers: whether to show the buffer layers
    """
    if not _check_whether_to_mark_arguments_on_edge(
            self, child_node, show_buffer_layers
    ):
        return

    arg_labels = []
    for arg_type in ["args", "kwargs"]:
        for arg_loc, arg_label in child_node.parent_layer_arg_locs[
            arg_type
        ].items():
            if (parent_node.layer_label == arg_label) or (
                    parent_node.layer_label in arg_label
            ):
                arg_labels.append(f"{arg_type[:-1]} {str(arg_loc)}")

    arg_labels = "<br/>".join(arg_labels)
    arg_label = f"<<FONT POINT-SIZE='10'><b>{arg_labels}</b></FONT>>"
    if "label" not in edge_dict:
        edge_dict["label"] = arg_label
    else:
        edge_dict["label"] = edge_dict["label"][:-1] + "<br/>" + arg_label[1:]


def _check_whether_to_mark_arguments_on_edge(
        self: "ModelHistory",
        child_node: Union["TensorLogEntry", "RolledTensorLogEntry"],
        show_buffer_layers: bool = False,
):
    if child_node.layer_type in COMMUTE_FUNCS:
        return False

    if isinstance(child_node, TensorLogEntry):
        return _check_whether_to_mark_arguments_on_unrolled_edge(
            self, child_node, show_buffer_layers
        )
    elif isinstance(child_node, RolledTensorLogEntry):
        return _check_whether_to_mark_arguments_on_rolled_edge(self, child_node)


def _check_whether_to_mark_arguments_on_unrolled_edge(
        self, child_node: "TensorLogEntry", show_buffer_layers: bool = False
):
    num_parents_shown = len(child_node.parent_layers)

    if not show_buffer_layers:
        num_parents_shown -= sum(
            [
                int(self[parent].is_buffer_layer)
                for parent in child_node.parent_layers
            ]
        )

    if num_parents_shown > 1:
        return True
    else:
        return False


def _check_whether_to_mark_arguments_on_rolled_edge(
        self: "ModelHistory",
        child_node: "RolledTensorLogEntry",
        show_buffer_layers: bool = False
):
    for pass_num, pass_parents in child_node.parent_layers_per_pass.items():
        num_parents_shown = len(pass_parents)
        if not show_buffer_layers:
            num_parents_shown -= sum(
                [
                    int(self.layer_dict_rolled[parent].is_buffer_layer)
                    for parent in pass_parents
                ]
            )
        if num_parents_shown > 1:
            return True

    return False


def _label_rolled_pass_nums(
        child_node: "RolledTensorLogEntry",
        parent_node: "RolledTensorLogEntry",
        edge_dict: Dict,
):
    """Adds labels for the pass numbers to the edge dict for rolled nodes.

    Args:
        child_node: child node
        parent_node: parent node
        edge_dict: dictionary of edge information
    """
    parent_pass_nums = parent_node.child_passes_per_layer[child_node.layer_label]
    child_pass_nums = child_node.parent_passes_per_layer[parent_node.layer_label]
    if parent_node.edges_vary_across_passes:
        edge_dict[
            "taillabel"
        ] = f"  Out {int_list_to_compact_str(parent_pass_nums)}  "

    # Mark the head label with the argument if need be:
    if child_node.edges_vary_across_passes:
        edge_dict[
            "headlabel"
        ] = f"  In {int_list_to_compact_str(child_pass_nums)}  "


def _get_lowest_containing_module_for_two_nodes(
        node1: Union["TensorLogEntry", "RolledTensorLogEntry"],
        node2: Union["TensorLogEntry", "RolledTensorLogEntry"],
        both_nodes_collapsed_modules: bool,
        vis_nesting_depth: int,
):
    """Utility function to get the lowest-level module that contains two nodes, to know where to put the edge.

    Args:
        node1: The first node.
        node2: The second node.
        vis_nesting_depth: How many levels deep to visualize.

    Returns:
        Lowest-level module pass containing two nodes.
    """
    node1_modules = node1.containing_modules_origin_nested[:]
    node2_modules = node2.containing_modules_origin_nested[:]

    if isinstance(node1, RolledTensorLogEntry):
        node1_modules = [module.split(":")[0] for module in node1_modules]
        node2_modules = [module.split(":")[0] for module in node2_modules]

    if node1.is_bottom_level_submodule_output:
        node1_nestmodules = node1_modules[:-1]
    else:
        node1_nestmodules = node1_modules[:]

    if (
            (len(node1_modules) == 0)
            or (len(node2_modules) == 0)
            or (node1_modules[0] != node2_modules[0])
    ):
        return -1  # no submodule contains them both.

    if node1 == node2:
        if node1.is_bottom_level_submodule_output and (len(node1_modules) == 1):
            return -1
        elif node1.is_bottom_level_submodule_output and (len(node1_modules) > 1):
            containing_module = node1_modules[-2]
        else:
            containing_module = node1_modules[-1]
        return containing_module

    if both_nodes_collapsed_modules:
        if (vis_nesting_depth == 1) or (len(node1_nestmodules) == 1):
            return -1
        if node1_modules[vis_nesting_depth - 1] == node2_modules[vis_nesting_depth - 1]:
            containing_module = node1_modules[vis_nesting_depth - 2]
            return containing_module

    containing_module = node1_modules[0]
    for m in range(min([len(node1_modules), len(node2_modules)])):
        if node1_modules[m] != node2_modules[m]:
            break
        containing_module = node1_modules[m]

    return containing_module


def _add_gradient_edge(
        self: "ModelHistory",
        parent_layer,
        child_layer,
        edge_style,
        containing_module,
        module_edge_dict,
        graphviz_graph,
        vis_gradient_edge_overrides
):
    """Adds a backwards edge if both layers have saved gradients, showing the backward pass."""
    if parent_layer.has_saved_grad and child_layer.has_saved_grad:
        edge_dict = {
            "tail_name": child_layer.layer_label.replace(":", "pass"),
            "head_name": parent_layer.layer_label.replace(":", "pass"),
            "color": self.GRADIENT_ARROW_COLOR,
            "fontcolor": self.GRADIENT_ARROW_COLOR,
            "style": edge_style,
            "arrowsize": ".7",
            "labelfontsize": "8",
        }
        for arg_name, arg_val in vis_gradient_edge_overrides.items():
            if callable(arg_val):
                edge_dict[arg_name] = str(arg_val(self, parent_layer, child_layer))
            else:
                edge_dict[arg_name] = str(arg_val)

        if containing_module != -1:
            module_edge_dict[containing_module]["edges"].append(edge_dict)
        else:
            graphviz_graph.edge(**edge_dict)


def _set_up_subgraphs(
        self: "ModelHistory",
        graphviz_graph,
        vis_opt: str,
        module_edge_dict: Dict,
        vis_module_overrides: Dict = None
):
    """Given a dictionary specifying the edges in each cluster and the graphviz graph object,
    set up the nested subgraphs and the nodes that should go inside each of them. There will be some tricky
    recursive logic to set up the nested context managers.

    Args:
        graphviz_graph: Graphviz graph object.
        vis_opt: 'rolled' or 'unrolled'
        module_edge_dict: Dictionary mapping each cluster to the list of edges it contains, with each
            edge specified as a dict with all necessary arguments for creating that edge.
    """
    if vis_opt == "unrolled":
        module_submodule_dict = self.module_pass_children.copy()
        subgraphs = self.top_level_module_passes[:]
    else:
        module_submodule_dict = self.module_children.copy()
        subgraphs = self.top_level_modules[:]

    # Get the max module nesting depth:

    max_nesting_depth = _get_max_nesting_depth(
        subgraphs, module_edge_dict, module_submodule_dict
    )

    subgraph_stack = [[subgraph] for subgraph in subgraphs]
    nesting_depth = 0
    while len(subgraph_stack) > 0:
        parent_graph_list = subgraph_stack.pop(0)
        _setup_subgraphs_recurse(
            self,
            graphviz_graph,
            parent_graph_list,
            module_edge_dict,
            module_submodule_dict,
            subgraph_stack,
            nesting_depth,
            max_nesting_depth,
            vis_opt,
            vis_module_overrides
        )


def _setup_subgraphs_recurse(
        self: "ModelHistory",
        starting_subgraph,
        parent_graph_list: List,
        module_edge_dict,
        module_submodule_dict,
        subgraph_stack,
        nesting_depth,
        max_nesting_depth,
        vis_opt,
        vis_module_overrides
):
    """Utility function to crawl down several layers deep into nested subgraphs.

    Args:
        starting_subgraph: The subgraph we're starting from.
        parent_graph_list: List of parent graphs.
        module_edge_dict: Dict mapping each cluster to its edges.
        module_submodule_dict: Dict mapping each cluster to its subclusters.
        subgraph_stack: Stack of subgraphs to look at.
        nesting_depth: Nesting depth so far.
        max_nesting_depth: The total depth of the subgraphs.
        vis_opt: 'rolled' or 'unrolled'
    """
    subgraph_name_w_pass = parent_graph_list[nesting_depth]
    subgraph_module = subgraph_name_w_pass.split(":")[0]
    if vis_opt == "unrolled":
        cluster_name = f"cluster_{subgraph_name_w_pass.replace(':', '_pass')}"
        subgraph_name = subgraph_name_w_pass
    elif vis_opt == "rolled":
        cluster_name = f"cluster_{subgraph_module}"
        subgraph_name = subgraph_module
    else:
        raise ValueError("vis_opt must be 'rolled' or 'unrolled'")
    module_type = self.module_types[subgraph_module]
    if (self.module_num_passes[subgraph_module] > 1) and (vis_opt == "unrolled"):
        subgraph_title = subgraph_name_w_pass
    elif (self.module_num_passes[subgraph_module] > 1) and (vis_opt == "rolled"):
        subgraph_title = (
            f"{subgraph_module} (x{self.module_num_passes[subgraph_module]})"
        )
    else:
        subgraph_title = subgraph_module

    if (
            nesting_depth < len(parent_graph_list) - 1
    ):  # we haven't gotten to the bottom yet, keep going.
        with starting_subgraph.subgraph(name=cluster_name) as s:
            _setup_subgraphs_recurse(
                self,
                s,
                parent_graph_list,
                module_edge_dict,
                module_submodule_dict,
                subgraph_stack,
                nesting_depth + 1,
                max_nesting_depth,
                vis_opt,
                vis_module_overrides
            )

    else:  # we made it, make the subgraph and add all edges.
        with starting_subgraph.subgraph(name=cluster_name) as s:
            nesting_fraction = (
                                       max_nesting_depth - nesting_depth
                               ) / max_nesting_depth
            pen_width = (
                    MIN_MODULE_PENWIDTH + nesting_fraction * PENWIDTH_RANGE
            )
            if module_edge_dict[subgraph_name]["has_input_ancestor"]:
                line_style = "solid"
            else:
                line_style = "dashed"

            module_args = {
                'label': f"<<B>@{subgraph_title}</B><br align='left'/>({module_type})<br align='left'/>>",
                'labelloc': 'b',
                'style': f'filled,{line_style}',
                'fillcolor': 'white',
                'penwidth': str(pen_width)}

            for arg_name, arg_val in vis_module_overrides.items():
                if callable(arg_val):
                    module_args[arg_name] = str(arg_val(self, subgraph_name))
                else:
                    module_args[arg_name] = str(arg_val)
            s.attr(**module_args)
            subgraph_edges = module_edge_dict[subgraph_name]["edges"]
            for edge_dict in subgraph_edges:
                s.edge(**edge_dict)
            subgraph_children = module_submodule_dict[subgraph_name_w_pass]
            for (
                    subgraph_child
            ) in subgraph_children:  # it's weird but have to go in reverse order.
                subgraph_stack.append(parent_graph_list[:] + [subgraph_child])


def _get_max_nesting_depth(top_modules, module_edge_dict, module_submodule_dict):
    """Utility function to get the max nesting depth of the nested modules in the network; works by
    recursively crawling down the stack of modules till it hits one with no children and at least one edge.

    Args:
        top_modules: modules at highest level of nesting
        module_edge_dict: Edges in each module.
        module_submodule_dict: Mapping from each module to any children.

    Returns:
        Max nesting depth.
    """
    max_nesting_depth = 1
    module_stack = [(graph, 1) for graph in top_modules]

    while len(module_stack) > 0:
        module, module_depth = module_stack.pop()
        module_edges = module_edge_dict[module]["edges"]
        module_submodules = module_submodule_dict[module]

        if (len(module_edges) == 0) and (
                len(module_submodules) == 0
        ):  # can ignore if no edges and no children.
            continue
        elif (len(module_edges) > 0) and (len(module_submodules) == 0):
            max_nesting_depth = max([module_depth, max_nesting_depth])
        elif (len(module_edges) == 0) and (len(module_submodules) > 0):
            module_stack.extend(
                [
                    (module_child, module_depth + 1)
                    for module_child in module_submodules
                ]
            )
        else:
            max_nesting_depth = max([module_depth, max_nesting_depth])
            module_stack.extend(
                [
                    (module_child, module_depth + 1)
                    for module_child in module_submodules
                ]
            )
    return max_nesting_depth
