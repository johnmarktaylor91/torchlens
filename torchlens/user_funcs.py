from typing import List, Optional, Union

import torch
from torch import nn

from torchlens.helper_funcs import warn_parallel, get_vars_of_type_from_obj
from torchlens.model_funcs import ModelHistory, run_model_and_save_specified_activations


def get_model_activations(model: nn.Module,
                          x: torch.Tensor,
                          which_layers: Union[str, List] = 'all',
                          vis_opt: str = 'none',
                          vis_outpath: str = 'graph.gv',
                          vis_fileformat: str = 'pdf',
                          vis_buffer_layers: bool = False,
                          vis_direction: str = 'vertical',
                          random_seed: Optional[int] = None) -> ModelHistory:
    """Runs a forward pass through a model given input x, and returns a ModelHistory object containing a log
    (layer activations and accompanying layer metadata) of the forward pass for all layers specified in which_layers,
    and optionally visualizes the model graph if vis_opt is set to 'rolled' or 'unrolled'.

    In which_layers, can specify 'all', for all layers (default), or a list containing any combination of:
    1) desired layer names (e.g., 'conv2d_1_1'; if a layer has multiple passes, this includes all passes),
    2) a layer pass (e.g., conv2d_1_1:2 for just the second pass), 3) a module name to fetch the output of a particular
    module, 4) the ordinal index of a layer in the model (e.g. 3 for the third layer, -2 for the second to last, etc.),
    or 5) a desired substring with which to filter desired layers (e.g., 'conv2d' for all conv2d layers).

    Args:
        model: PyTorch model
        x: model input
        which_layers: list of layers to include (described above), or 'all' to include all layers.
        vis_opt: whether, and how, to visualize the network; 'none' for
            no visualization, 'rolled' to show the graph in rolled-up format (i.e.,
            one node per layer if a recurrent network), or 'unrolled' to show the graph
            in unrolled format (i.e., one node per pass through a layer if a recurrent)
        vis_outpath: file path to save the graph visualization
        vis_fileformat: the format of the visualization (e.g,. 'pdf', 'jpg', etc.)
        vis_buffer_layers: whether to visualize the buffer layers
        vis_direction: either 'vertical' or 'horizontal'
        random_seed: which random seed to use in case model involves randomness

    Returns:
        ModelHistory object with layer activations and metadata
    """
    warn_parallel()

    if vis_opt not in ['none', 'rolled', 'unrolled']:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    # If not saving all layers, do a probe pass.

    if which_layers == 'all':
        tensor_nums_to_save = 'all'
    elif which_layers in ['none', None, []]:
        tensor_nums_to_save = None
    else:
        model_history = run_model_and_save_specified_activations(model, x, None, random_seed)
        tensor_nums_to_save, tensor_nums_to_save_temporarily = model_history.get_op_nums_from_user_labels(which_layers)

    # And now save the activations for real.

    model_history = run_model_and_save_specified_activations(model,
                                                             x,
                                                             tensor_nums_to_save,
                                                             random_seed)

    # Visualize if desired.
    if vis_opt != 'none':
        model_history.render_graph(vis_opt,
                                   vis_outpath,
                                   vis_fileformat,
                                   vis_buffer_layers,
                                   vis_direction)  # change after adding options

    return model_history


def get_model_structure(model: nn.Module,
                        x: torch.Tensor,
                        random_seed: Optional[int] = None):
    """
    Equivalent to get_model_activations, but only fetches layer metadata without saving activations.

    Args:
        model: PyTorch model.
        x: model input
        random_seed: which random seed to use in case model involves randomness

    Returns:
        history_dict: Dict of dicts with the activations from each layer.
    """
    warn_parallel()
    model_history = run_model_and_save_specified_activations(model, x, None, random_seed)
    return model_history


def show_model_graph(model: nn.Module,
                     x: torch.Tensor,
                     vis_opt: str = 'none',
                     vis_outpath: str = 'graph.gv',
                     vis_fileformat: str = 'pdf',
                     vis_buffer_layers: bool = False,
                     vis_direction: str = 'vertical',
                     random_seed: Optional[int] = None) -> None:
    """Visualize the model graph without saving any activations.

    Args:
        model: PyTorch model
        x: model input
        vis_opt: whether, and how, to visualize the network; 'none' for
            no visualization, 'rolled' to show the graph in rolled-up format (i.e.,
            one node per layer if a recurrent network), or 'unrolled' to show the graph
            in unrolled format (i.e., one node per pass through a layer if a recurrent)
        vis_outpath: file path to save the graph visualization
        vis_fileformat: the format of the visualization (e.g,. 'pdf', 'jpg', etc.)
        vis_buffer_layers: whether to visualize the buffer layers
        vis_direction: either 'vertical' or 'horizontal'
        random_seed: which random seed to use in case model involves randomness

    Returns:
        Nothing.
    """
    if vis_opt not in ['none', 'rolled', 'unrolled']:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    model_history = run_model_and_save_specified_activations(model, x, None, random_seed)
    model_history.render_graph(vis_opt, vis_outpath, vis_fileformat,
                               vis_buffer_layers, vis_direction)


def validate_saved_activations(model: nn.Module,
                               x: torch.Tensor,
                               random_seed: Union[int, None] = None) -> bool:
    """Validate that the saved model activations correctly reproduce the ground truth output. This function works by
    running a forward pass through the model, saving all activations, re-running the forward pass starting from
    the saved activations in each layer, and checking that the resulting output matches the original output.
    Additionally, it substitutes in random activations and checks whether the output changes accordingly, for
    at least min_proportion_consequential_layers of the layers (in case some layers do not change the output for some
    reason). Returns True if a model passes these tests for the given input, and False otherwise.

    Args:
        model: PyTorch model.
        x: Input for which to validate the saved activations.
        random_seed: random seed in case model is stochastic
    Returns:
        True if the saved activations correctly reproduce the ground truth output, false otherwise.
    """
    warn_parallel()
    ground_truth_output_tensors = get_vars_of_type_from_obj(model(x), torch.Tensor)
    model_history = run_model_and_save_specified_activations(model, x, 'all', random_seed)
    activations_are_valid = model_history.validate_saved_activations(ground_truth_output_tensors)

    model_history.cleanup()
    del model_history
    return activations_are_valid
