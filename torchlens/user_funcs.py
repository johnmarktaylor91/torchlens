import copy
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import nn
import random

from tqdm import tqdm

from torchlens.helper_funcs import (
    warn_parallel,
    get_vars_of_type_from_obj,
    set_random_seed,
)
from torchlens.model_history import (
    ModelHistory,
    run_model_and_save_specified_activations,
)


def log_forward_pass(
        model: nn.Module,
        input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
        input_kwargs: Dict[Any, Any] = None,
        layers_to_save: Optional[Union[str, List]] = "all",
        keep_unsaved_layers: bool = True,
        output_device: str = "same",
        activation_postfunc: Optional[Callable] = None,
        mark_input_output_distances: bool = False,
        detach_saved_tensors: bool = False,
        save_function_args: bool = False,
        save_gradients: bool = False,
        vis_opt: str = "none",
        vis_nesting_depth: int = 1000,
        vis_outpath: str = "graph.gv",
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        vis_buffer_layers: bool = False,
        vis_direction: str = "bottomup",
        vis_graph_overrides: Dict = None,
        vis_node_overrides: Dict = None,
        vis_nested_node_overrides: Dict = None,
        vis_edge_overrides: Dict = None,
        vis_gradient_edge_overrides: Dict = None,
        vis_module_overrides: Dict = None,
        random_seed: Optional[int] = None,
) -> ModelHistory:
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
        input_args: input arguments for model forward pass; as a list if multiple, else as a single tensor.
        input_kwargs: keyword arguments for model forward pass
        layers_to_save: list of layers to include (described above), or 'all' to include all layers.
        keep_unsaved_layers: whether to keep layers without saved activations in the log (i.e., with just metadata)
        activation_postfunc: Function to apply to tensors before saving them (e.g., channelwise averaging).
        mark_input_output_distances: whether to mark the distance of each layer from the input or output;
            False by default since this is computationally expensive.
        output_device: device where saved tensors are to be stored. Either 'same' to keep on the same device,
            or 'cpu' or 'cuda' to move them to cpu or cuda when saved.
        detach_saved_tensors: whether to detach the saved tensors, so they remain attached to the computational graph
        save_function_args: whether to save the arguments to each function involved in computing each tensor
        save_gradients: whether to save gradients from any subsequent backward pass
        vis_opt: whether, and how, to visualize the network; 'none' for
            no visualization, 'rolled' to show the graph in rolled-up format (i.e.,
            one node per layer if a recurrent network), or 'unrolled' to show the graph
            in unrolled format (i.e., one node per pass through a layer if a recurrent)
        vis_nesting_depth: How many levels of nested modules to show; 1 for only top-level modules, 2 for two
            levels, etc.
        vis_outpath: file path to save the graph visualization
        vis_save_only: whether to only save the graph visual without immediately showing it
        vis_fileformat: the format of the visualization (e.g,. 'pdf', 'jpg', etc.)
        vis_buffer_layers: whether to visualize the buffer layers
        vis_direction: either 'bottomup', 'topdown', or 'leftright'
        random_seed: which random seed to use in case model involves randomness

    Returns:
        ModelHistory object with layer activations and metadata
    """
    warn_parallel()

    if vis_opt not in ["none", "rolled", "unrolled"]:
        raise ValueError(
            "Visualization option must be either 'none', 'rolled', or 'unrolled'."
        )

    if output_device not in ["same", "cpu", "cuda"]:
        raise ValueError("output_device must be either 'same', 'cpu', or 'cuda'.")

    if type(layers_to_save) is str:
        layers_to_save = layers_to_save.lower()

    if layers_to_save in ["all", "none", None, []]:
        model_history = run_model_and_save_specified_activations(
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            keep_unsaved_layers=keep_unsaved_layers,
            output_device=output_device,
            activation_postfunc=activation_postfunc,
            mark_input_output_distances=mark_input_output_distances,
            detach_saved_tensors=detach_saved_tensors,
            save_function_args=save_function_args,
            save_gradients=save_gradients,
            random_seed=random_seed,
        )
    else:
        model_history = run_model_and_save_specified_activations(
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=None,
            keep_unsaved_layers=True,
            output_device=output_device,
            activation_postfunc=activation_postfunc,
            mark_input_output_distances=mark_input_output_distances,
            detach_saved_tensors=detach_saved_tensors,
            save_function_args=save_function_args,
            save_gradients=save_gradients,
            random_seed=random_seed,
        )
        model_history.keep_unsaved_layers = keep_unsaved_layers
        model_history.save_new_activations(
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            random_seed=random_seed,
        )

    # Visualize if desired.
    if vis_opt != "none":
        model_history.render_graph(
            vis_opt,
            vis_nesting_depth,
            vis_outpath,
            vis_graph_overrides,
            vis_node_overrides,
            vis_nested_node_overrides,
            vis_edge_overrides,
            vis_gradient_edge_overrides,
            vis_module_overrides,
            vis_save_only,
            vis_fileformat,
            vis_buffer_layers,
            vis_direction,
        )

    return model_history


def get_model_metadata(
        model: nn.Module,
        input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
        input_kwargs: Dict[Any, Any] = None,
) -> ModelHistory:
    """Logs all metadata for a given model and inputs without saving any activations. NOTE: this function
    will be removed in a future version of TorchLens, since calling it is identical to calling
    log_forward_pass without saving any layers.

    Args:
        model: model to inspect
        input_args: list of input positional arguments, or a single tensor
        input_kwargs: dict of keyword arguments
    Returns:
        ModelHistory object with metadata about the model.
    """
    model_history = log_forward_pass(
        model,
        input_args,
        input_kwargs,
        layers_to_save=None,
        mark_input_output_distances=True,
    )
    return model_history


def show_model_graph(
        model: nn.Module,
        input_args: Union[torch.Tensor, List, Tuple],
        input_kwargs: Dict[Any, Any] = None,
        vis_opt: str = "unrolled",
        vis_nesting_depth: int = 1000,
        vis_outpath: str = "graph.gv",
        vis_graph_overrides: Dict = None,
        vis_node_overrides: Dict = None,
        vis_nested_node_overrides: Dict = None,
        vis_edge_overrides: Dict = None,
        vis_gradient_edge_overrides: Dict = None,
        vis_module_overrides: Dict = None,
        save_only: bool = False,
        vis_fileformat: str = "pdf",
        vis_buffer_layers: bool = False,
        vis_direction: str = "bottomup",
        random_seed: Optional[int] = None,
) -> None:
    """Visualize the model graph without saving any activations.

    Args:
        model: PyTorch model
        input_args: Arguments for model forward pass
        input_kwargs: Keyword arguments for model forward pass
        vis_opt: whether, and how, to visualize the network; 'none' for
            no visualization, 'rolled' to show the graph in rolled-up format (i.e.,
            one node per layer if a recurrent network), or 'unrolled' to show the graph
            in unrolled format (i.e., one node per pass through a layer if a recurrent)
        vis_nesting_depth: How many levels of nested modules to show; 1 for only top-level modules, 2 for two
            levels, etc.
        vis_outpath: file path to save the graph visualization
        save_only: whether to only save the graph visual without immediately showing it
        vis_fileformat: the format of the visualization (e.g,. 'pdf', 'jpg', etc.)
        vis_buffer_layers: whether to visualize the buffer layers
        vis_direction: either 'bottomup', 'topdown', or 'leftright'
        random_seed: which random seed to use in case model involves randomness

    Returns:
        Nothing.
    """
    if not input_kwargs:
        input_kwargs = {}

    if vis_opt not in ["none", "rolled", "unrolled"]:
        raise ValueError(
            "Visualization option must be either 'none', 'rolled', or 'unrolled'."
        )

    model_history = run_model_and_save_specified_activations(
        model=model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        layers_to_save=None,
        activation_postfunc=None,
        mark_input_output_distances=False,
        detach_saved_tensors=False,
        save_gradients=False,
        random_seed=random_seed,
    )
    model_history.render_graph(
        vis_opt,
        vis_nesting_depth,
        vis_outpath,
        vis_graph_overrides,
        vis_node_overrides,
        vis_nested_node_overrides,
        vis_edge_overrides,
        vis_gradient_edge_overrides,
        vis_module_overrides,
        save_only,
        vis_fileformat,
        vis_buffer_layers,
        vis_direction,
    )


def validate_saved_activations(
        model: nn.Module,
        input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
        input_kwargs: Dict[Any, Any] = None,
        random_seed: Union[int, None] = None,
        verbose: bool = False,
) -> bool:
    """Validate that the saved model activations correctly reproduce the ground truth output. This function works by
    running a forward pass through the model, saving all activations, re-running the forward pass starting from
    the saved activations in each layer, and checking that the resulting output matches the original output.
    Additionally, it substitutes in random activations and checks whether the output changes accordingly, for
    at least min_proportion_consequential_layers of the layers (in case some layers do not change the output for some
    reason). Returns True if a model passes these tests for the given input, and False otherwise.

    Args:
        model: PyTorch model.
        input_args: Input for which to validate the saved activations.
        input_kwargs: Keyword arguments for model forward pass
        random_seed: random seed in case model is stochastic
        verbose: whether to show verbose error messages
    Returns:
        True if the saved activations correctly reproduce the ground truth output, false otherwise.
    """
    warn_parallel()
    if random_seed is None:  # set random seed
        random_seed = random.randint(1, 4294967294)
    set_random_seed(random_seed)
    if type(input_args) is tuple:
        input_args = list(input_args)
    elif (type(input_args) not in [list, tuple]) and (input_args is not None):
        input_args = [input_args]
    if not input_args:
        input_args = []
    if not input_kwargs:
        input_kwargs = {}
    input_args_copy = [copy.deepcopy(arg) for arg in input_args]
    input_kwargs_copy = {key: copy.deepcopy(val) for key, val in input_kwargs.items()}
    state_dict = model.state_dict()
    ground_truth_output_tensors = get_vars_of_type_from_obj(
        model(*input_args_copy, **input_kwargs_copy),
        torch.Tensor,
        search_depth=5,
        allow_repeats=True,
    )
    model.load_state_dict(state_dict)
    model_history = run_model_and_save_specified_activations(
        model=model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        layers_to_save="all",
        keep_unsaved_layers=True,
        activation_postfunc=None,
        mark_input_output_distances=False,
        detach_saved_tensors=False,
        save_gradients=False,
        save_function_args=True,
        random_seed=random_seed,
    )
    activations_are_valid = model_history.validate_saved_activations(
        ground_truth_output_tensors, verbose
    )

    model_history.cleanup()
    del model_history
    return activations_are_valid


def validate_batch_of_models_and_inputs(
        models_and_inputs_dict: Dict[str, Dict[str, Union[str, Callable, Dict]]],
        out_path: str,
        redo_model_if_already_run: bool = True,
) -> pd.DataFrame:
    """Given multiple models and several inputs for each, validates the saved activations for all of them
    and returns a Pandas dataframe summarizing the validation results.

    Args:
        models_and_inputs_dict: Dict mapping each model name to a dict of model info:
                model_category: category of model (e.g., torchvision; just for directory bookkeeping)
                model_loading_func: function to load the model
                model_sample_inputs: dict of example inputs {name: input}
        out_path: Path to save the validation results to
        redo_model_if_already_run: If True, will re-run the validation for a model even if already run

    Returns:
        Pandas dataframe with validation information for each model and input
    """
    if os.path.exists(out_path):
        current_csv = pd.read_csv(out_path)
    else:
        current_csv = pd.DataFrame.from_dict(
            {
                "model_category": [],
                "model_name": [],
                "input_name": [],
                "validation_success": [],
            }
        )
    models_already_run = current_csv["model_name"].unique()
    for model_name, model_info in tqdm(
            models_and_inputs_dict.items(), desc="Validating models"
    ):
        print(f"Validating model {model_name}")
        if model_name in models_already_run and not redo_model_if_already_run:
            continue
        model_category = model_info["model_category"]
        model_loading_func = model_info["model_loading_func"]
        model = model_loading_func()
        model_sample_inputs = model_info["model_sample_inputs"]
        for input_name, x in model_sample_inputs.items():
            validation_success = validate_saved_activations(model, x)
            current_csv = current_csv.append(
                {
                    "model_category": model_category,
                    "model_name": model_name,
                    "input_name": input_name,
                    "validation_success": validation_success,
                },
                ignore_index=True,
            )
        current_csv.to_csv(out_path, index=False)
        del model
    return current_csv
