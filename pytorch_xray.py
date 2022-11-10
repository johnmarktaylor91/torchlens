import copy
from collections import OrderedDict
import multiprocessing as mp
from typing import List, Union, Optional

import torch
from torch import nn

from model_funcs import cleanup_model, prepare_model, run_model_and_save_specified_activations
from util_funcs import remove_list_duplicates
from graph_funcs import get_op_nums_from_layer_names, render_graph, postprocess_history_dict, prettify_history_dict


# TODO: Figure out best way to go from full graph functionality to just modules (for graph, list, activations);
# TODO: And along with that figure out how to handle the rolled-up functionality for both the modules, and for the
# function view (the latter requires checking the parameters). Have it count the number of passes.
# Actually maybe save the rolled-up for later since it stands alone, it can be a final thing.
# But at least individuate the functions that see the input multiple times so they can be labeled (this comes from
# counting the parameter passes. Maybe fields like: module_num_passes, function_num_passes? Increment the
# function_num_passes and module_num_passes as input passes through.


def xray_model(model: nn.Module,
               x: torch.Tensor,
               which_layers: Union[str, List] = 'all',
               vis_opt: str = 'unrolled',
               random_seed: Optional[int] = None) -> OrderedDict[str, OrderedDict]:
    """Run a forward pass through a model, and return activations of desired hidden layers.
    Specify mode as 'modules_only' to do so only for proper modules, or as 'exhaustive' to
    also return activations from non-module functions. If only a subset of layers
    is desired, specify the list of layer names (e.g., 'conv1_5') in which_layers; if you wish to
    further specify that only certain passes through a layer should be saved
    (i.e., in a recurrent network, only save the third pass through a layer), then
    add :{pass_number} to the layer name (e.g., 'conv1_5:3'). Additionally, the graph
    can be visualized if desire to see the architecture and easily reference the names.

    Args:
        model: PyTorch model
        x: desired Tensor input.
        which_layers: List of layers to include. If 'all', then include all layers.
        vis_opt: Whether, and how, to visualize the network; 'none' for
            no visualization, 'rolled' to show the graph in rolled-up format (i.e.,
            one node per layer if a recurrent network), or 'unrolled' to show the graph
            in unrolled format (i.e., one node per pass through a layer if a recurrent)
        random_seed: Which random seed to use if desired (e.g., for networks with randomness)

    Returns:
        activations: Dict of dicts with the activations from each layer.
    """
    if mp.current_process().name != 'MainProcess':
        print("WARNING: It looks like you are using parallel execution; it is strongly advised"
              "to only run pytorch-xray in the main process, since certain operations "
              "depend on execution order.")

    x = copy.deepcopy(x)

    if vis_opt not in ['none', 'rolled', 'unrolled']:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    # If not saving all layers, do a probe pass.

    if which_layers != 'all':
        history_dict = run_model_and_save_specified_activations(model, x, mode, None, random_seed)
        history_dict = postprocess_history_dict(history_dict)
        tensor_nums_to_save = get_op_nums_from_layer_names(history_dict, which_layers)
    else:
        tensor_nums_to_save = 'all'

    # And now save the activations for real.

    history_dict = run_model_and_save_specified_activations(model, x, 'exhaustive', tensor_nums_to_save, random_seed)

    # Visualize if desired.
    if vis_opt != 'none':
        render_graph(history_dict, vis_opt)  # change after adding options

    history_pretty = prettify_history_dict(history_dict)  # for user readability
    return history_pretty


def show_model_graph(model: nn.Module,
                     x: torch.Tensor,
                     mode: str = 'modules_only',
                     visualize_opt: str = 'rolled',
                     random_seed: Optional[int] = None) -> None:
    """Visualize the model graph without saving any activations.

    Args:
        model: PyTorch model.
        x: Input for which you want to visualize the graph (this is needed in case the graph varies based on input)
        mode: 'modules_only' to only view modules, 'exhaustive' to
            view all tensor operations.
        visualize_opt: 'rolled' to show the graph in rolled-up format (one node
            per layer, even if multiple passes), or 'unrolled' to view with
            one node per operation.
        random_seed: random seed in case model is stochastic

    Returns:
        Nothing.
    """
    if mode not in ['modules_only', 'exhaustive']:
        raise ValueError("Mode must be either 'modules_only' or 'exhaustive'.")
    if visualize_opt not in ['none', 'rolled', 'unrolled']:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    # Simply call xray_model without saving any layers.
    history_dict = run_model_and_save_specified_activations(model, x, None, random_seed)
    render_model_graph(history_dict, mode, visualize_opt)


def list_model(model: nn.Module,
               x: torch.Tensor,
               mode: str = 'modules_only',
               unrolled_opt: str = 'rolled',
               random_seed: Optional[int] = None) -> List[str]:
    """List the layers in a model.

    Args:
        model: PyTorch model.
        x: Input for which you want to visualize the graph (this is needed in case the graph varies based on input)
        mode: 'modules_only' to only view modules, 'exhaustive' to
            view all tensor operations.
        unrolled_opt: Put 'rolled' to list each layer once, or 'unrolled' to list each computational step.
        random_seed: random seed in case model is stochastic

    Returns:
        List of layer names.
    """
    if mode not in ['modules_only', 'exhaustive']:
        raise ValueError("Mode must be either 'modules_only' or 'exhaustive'.")
    if unrolled_opt not in ['none', 'rolled', 'unrolled']:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    # Simply call xray_model without saving any layers.
    history_dict = run_model_and_save_specified_activations(model, x, mode, None, random_seed)
    # TODO: maybe just wrap this into the post-processing function?
    layer_list = get_layer_list_from_history(history_dict, mode, unrolled_opt)
    return layer_list
