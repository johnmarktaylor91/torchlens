import copy
from collections import OrderedDict
import multiprocessing as mp
from typing import List, Union

import torch
from torch import nn

from model_handling import cleanup_model, prepare_model
from xray_utils import remove_list_duplicates


def xray_model(model: nn.Module,
               x: torch.Tensor,
               mode: str = 'modules_only',
               which_layers: Union[str, List] = 'all',
               visualize_opt: str = 'none') -> OrderedDict[str, OrderedDict]:
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
        mode: 'modules_only' to return activations only for module objects, or
            'exhaustive' to do it for ALL tensor operations.
        which_layers: List of layers to include. If 'all', then include all layers.
        visualize_opt: Whether, and how, to visualize the network; 'none' for
            no visualization, 'rolled' to show the graph in rolled-up format (i.e.,
            one node per layer if a recurrent network), or 'unrolled' to show the graph
            in unrolled format (i.e., one node per pass through a layer if a recurrent)

    Returns:
        activations: Dict of dicts with the activations from each layer.
    """
    if mp.current_process().name != 'MainProcess':
        print("WARNING: It looks like you are using parallel execution; it is strongly advised"
              "to only run pytorch-xray in the main process, since certain operations "
              "depend on execution order.")

    x = copy.deepcopy(x)

    if mode not in ['modules_only', 'exhaustive']:
        raise ValueError("Mode must be either 'modules_only' or 'exhaustive'.")
    if visualize_opt not in ['none', 'rolled', 'unrolled']:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    hook_handles = []
    HistoryTensor.clear_history()
    x_history = HistoryTensor(x, mode, which_layers)

    # Wrap everything in a try-except block to guarantee the model remains unchanged in case of error.
    try:
        hook_handles = prepare_model(model, hook_handles)
        output = model(x_history)
        module_output_dict = model.xray_all_modules_dict
        module_output_dict = postprocess_module_output_dict(module_output_dict)
        HistoryTensor.postprocess_tensor_history()
        tensor_history = HistoryTensor.copy_tidy_tensor_history()
        if visualize_opt != 'none':
            model_graph = make_model_graph(module_output_dict,
                                           tensor_history,
                                           mode,
                                           visualize_opt)
            render_model_graph(model_graph)
        cleanup_model(model, hook_handles)
        HistoryTensor.clear_history()
        if mode == 'modules_only':
            return module_output_dict
        elif mode == 'exhaustive':
            return tensor_history
    finally:  # if anything fails, clean up the model and re-raise the error.
        print("Execution failed somewhere, returning model to original state...")
        cleanup_model(model, hook_handles)
        HistoryTensor.clear_history()
        raise e


def show_model_graph(model: nn.Module,
                     x: torch.Tensor,
                     mode: str = 'modules_only',
                     visualize_opt: str = 'rolled') -> None:
    """Visualize the model graph without saving any activations.

    Args:
        model: PyTorch model.
        x: Input for which you want to visualize the graph (this is needed in case the graph varies based on input)
        mode: 'modules_only' to only view modules, 'exhaustive' to
            view all tensor operations.
        visualize_opt: 'rolled' to show the graph in rolled-up format (one node
            per layer, even if multiple passes), or 'unrolled' to view with
            one node per operation.

    Returns:
        Nothing.
    """
    if mode not in ['modules_only', 'exhaustive']:
        raise ValueError("Mode must be either 'modules_only' or 'exhaustive'.")
    if visualize_opt not in ['none', 'rolled', 'unrolled']:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    # Simply call xray_model without saving any layers.
    _ = xray_model(model, x, mode, which_layers=[], visualize_opt=visualize_opt)


def list_model(model: nn.Module,
               x: torch.Tensor,
               mode: str = 'modules_only',
               unrolled: bool = False) -> List[str]:
    """List the layers in a model.

    Args:
        model: PyTorch model.
        x: Input for which you want to visualize the graph (this is needed in case the graph varies based on input)
        mode: 'modules_only' to only view modules, 'exhaustive' to
            view all tensor operations.
        unrolled: Whether to list each layer once, or list each computational step (if recurrent).

    Returns:
        List of layer names.
    """
    if mode not in ['modules_only', 'exhaustive']:
        raise ValueError("Mode must be either 'modules_only' or 'exhaustive'.")

    # Simply call xray_model without saving any layers.
    layer_dict = xray_model(model, x, mode, which_layers=[], visualize_opt='none')
    layer_list = list(layer_dict.keys())

    if not unrolled:
        layer_list = [layer.split(':')[0] for layer in layer_list]
        layer_list = remove_list_duplicates(layer_list)

    return layer_list
