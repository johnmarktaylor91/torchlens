from typing import List, Optional, Union

import torch
from torch import nn

from torchlens.graph_handling import ModelHistory, get_op_nums_from_layer_names, postprocess_history_dict
from torchlens.helper_funcs import warn_parallel
from torchlens.model_funcs import run_model_and_save_specified_activations
from torchlens.validate import validate_model_history
from torchlens.vis import render_graph


def get_model_activations(model: nn.Module,
                          x: torch.Tensor,
                          which_layers: Union[str, List] = 'all',
                          vis_opt: str = 'none',
                          random_seed: Optional[int] = None) -> ModelHistory:
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
    warn_parallel()

    if vis_opt not in ['none', 'rolled', 'unrolled']:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    # If not saving all layers, do a probe pass.

    if which_layers == 'all':
        tensor_nums_to_save = 'all'
        tensor_nums_to_save_temporarily = []
        activations_only = True
    elif which_layers in ['none', None, []]:
        tensor_nums_to_save = None
        tensor_nums_to_save_temporarily = []
        activations_only = False
    else:
        history_dict = run_model_and_save_specified_activations(model, x, None, random_seed)
        history_pretty = ModelHistory(history_dict, activations_only=False)
        tensor_nums_to_save, tensor_nums_to_save_temporarily = history_pretty.get_op_nums_from_user_labels(which_layers)
        activations_only = True

    # And now save the activations for real.

    history_dict = run_model_and_save_specified_activations(model,
                                                            x,
                                                            tensor_nums_to_save,
                                                            tensor_nums_to_save_temporarily,
                                                            random_seed)

    # Visualize if desired.
    if vis_opt != 'none':
        render_graph(history_dict, vis_opt)  # change after adding options

    history_pretty = ModelHistory(history_dict, activations_only=activations_only)
    return history_pretty


def get_model_structure(model: nn.Module,
                        x: torch.Tensor,
                        random_seed: Optional[int] = None):
    """
    Get the metadata for the model graph without saving any activations.

    Args:
        model: PyTorch model.
        x: Input for which you want to visualize the graph (this is needed in case the graph varies based on input)
        random_seed: random seed in case model is stochastic

    Returns:
        history_dict: Dict of dicts with the activations from each layer.
    """
    warn_parallel()
    history_dict = run_model_and_save_specified_activations(model, x, None, random_seed)
    history_pretty = ModelHistory(history_dict, activations_only=False)
    return history_pretty


def show_model_graph(model: nn.Module,
                     x: torch.Tensor,
                     visualize_opt: str = 'rolled',
                     random_seed: Optional[int] = None) -> None:
    """Visualize the model graph without saving any activations.

    Args:
        model: PyTorch model.
        x: Input for which you want to visualize the graph (this is needed in case the graph varies based on input)
        visualize_opt: 'rolled' to show the graph in rolled-up format (one node
            per layer, even if multiple passes), or 'unrolled' to view with
            one node per operation.
        random_seed: random seed in case model is stochastic

    Returns:
        Nothing.
    """
    if visualize_opt not in ['none', 'rolled', 'unrolled']:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    history_dict = run_model_and_save_specified_activations(model, x, None, random_seed)
    render_graph(history_dict, visualize_opt)


def validate_saved_activations(model: nn.Module,
                               x: torch.Tensor,
                               random_seed: Union[int, None] = None,
                               min_proportion_consequential_layers=.9,
                               verbose: bool = False) -> bool:
    """Validate that the saved model activations correctly reproduce the ground truth output.

    Args:
        model: PyTorch model.
        x: Input for which to validate the saved activations.
        random_seed: random seed in case model is stochastic
        min_proportion_consequential_layers: The percentage of layers for which perturbing them must change the output
            in order for the model to count as validated; if 0, doesn't check.
        verbose: Whether to have messages during the validation process.

    Returns:
        True if the saved activations correctly reproduce the ground truth output, false otherwise.
    """
    warn_parallel()
    history_dict = run_model_and_save_specified_activations(model, x, 'all', random_seed)
    history_pretty = ModelHistory(history_dict, activations_only=True)
    activations_are_valid = validate_model_history(history_dict,
                                                   history_pretty,
                                                   min_proportion_consequential_layers,
                                                   verbose)
    for node in history_pretty:
        del node.tensor_contents
        del node
    del history_pretty
    for node in history_dict['tensor_log'].values():
        node.clear()
    history_dict.clear()
    return activations_are_valid
