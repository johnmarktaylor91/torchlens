"""Public API entry points for TorchLens.

This module contains every user-facing function:
  - ``log_forward_pass``  — the main entry point (runs model, returns ModelLog)
  - ``validate_forward_pass`` — replay-based correctness check
  - ``show_model_graph`` — visualization convenience wrapper
  - ``get_model_metadata`` — metadata-only convenience wrapper (deprecated path)
  - ``validate_batch_of_models_and_inputs`` — bulk validation harness

**Two-pass strategy** (``log_forward_pass`` with selective layers):
When the user requests specific layers (not "all" or "none"), TorchLens must
first run an exhaustive pass to discover the full graph structure — only then can
it resolve user-friendly layer names/indices to internal layer numbers.  A second
fast pass replays the model, saving only the requested activations.  This is why
``log_forward_pass`` has two branches: the simple path (save all/none) and the
two-pass path (save specific layers).
"""

import collections.abc
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from .utils.introspection import get_vars_of_type_from_obj
from .utils.rng import set_random_seed
from .utils.display import warn_parallel, _vprint
from .utils.arg_handling import safe_copy_args, safe_copy_kwargs, normalize_input_args
from .data_classes.model_log import (
    ModelLog,
)


def _unwrap_data_parallel(model: nn.Module) -> nn.Module:
    """Unwrap nn.DataParallel to get the underlying module."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _move_tensors_to_device(obj, device):
    """Recursively move tensors in a nested structure (lists, tuples, dicts) to *device*.

    Handles common dict-like types (OrderedDict, HuggingFace BatchEncoding, etc.)
    by attempting to reconstruct the original container type after moving values.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        moved = [_move_tensors_to_device(item, device) for item in obj]
        return type(obj)(moved) if not isinstance(obj, tuple) else tuple(moved)
    elif isinstance(obj, collections.abc.MutableMapping):
        # Handles dict, UserDict, BatchEncoding, OrderedDict, etc.
        moved = {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
        if type(obj) is dict:
            return moved
        try:
            return type(obj)(moved)
        except Exception:
            return moved
    return obj


def _run_model_and_save_specified_activations(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any]],
    input_kwargs: Dict[Any, Any],
    layers_to_save: Optional[Union[str, List[Union[int, str]]]] = "all",
    keep_unsaved_layers: bool = True,
    output_device: str = "same",
    activation_postfunc: Optional[Callable] = None,
    mark_input_output_distances: bool = False,
    detach_saved_tensors: bool = False,
    save_function_args: bool = False,
    save_gradients: bool = False,
    random_seed: Optional[int] = None,
    num_context_lines: int = 7,
    optimizer=None,
    save_source_context: bool = False,
    save_rng_states: bool = False,
    detect_loops: bool = True,
    verbose: bool = False,
) -> ModelLog:
    """Run a forward pass with logging enabled, returning a populated ModelLog.

    This is the single internal entry point that creates a ModelLog, configures it,
    and delegates to ``ModelLog._run_and_log_inputs_through_model`` which handles
    model preparation, the exhaustive (and optionally fast) forward pass, and all
    postprocessing.

    Args:
        model: PyTorch model.
        input_args: Positional arguments to model.forward(); a single tensor or list.
        input_kwargs: Keyword arguments to model.forward().
        layers_to_save: Which layers to save activations for ('all', 'none'/None, or a list).
        keep_unsaved_layers: If False, layers without saved activations are pruned from the log.
        output_device: Device for saved tensors: 'same' (default), 'cpu', or 'cuda'.
        activation_postfunc: Optional transform applied to each activation before storage
            (e.g., channel-wise averaging to reduce memory).
        mark_input_output_distances: Compute BFS distances from input/output layers.
            Expensive for large graphs — off by default.
        detach_saved_tensors: If True, saved tensors are detached from the autograd graph.
        save_function_args: If True, store the non-tensor arguments to each function call.
            Required for validation replay (``validate_saved_activations``).
        save_gradients: If True, register backward hooks to capture gradients.
        random_seed: Fixed RNG seed for reproducibility (important for stochastic models).
        num_context_lines: Number of source-code context lines stored per function call.
        optimizer: Optional optimizer — used to tag which parameters have optimizers attached.
        detect_loops: If True (default), run full isomorphic subgraph expansion to
            detect repeated patterns (loops). If False, only group operations that
            share the same parameters — much faster for very large graphs.
        verbose: If True, print timed progress messages at each major pipeline stage.

    Returns:
        Fully-populated ModelLog.
    """
    # Auto-detect model device from its first parameter and move inputs to match.
    # This prevents silent device-mismatch errors when the model is on CUDA but
    # the user passes CPU tensors (a common mistake).
    model_device = next((p.device for p in model.parameters()), None)
    if model_device is not None:
        input_args = _move_tensors_to_device(input_args, model_device)
        if input_kwargs is not None:
            input_kwargs = _move_tensors_to_device(input_kwargs, model_device)

    model_name = str(type(model).__name__)
    model_log = ModelLog(
        model_name,
        output_device,
        activation_postfunc,
        keep_unsaved_layers,
        save_function_args,
        save_gradients,
        detach_saved_tensors,
        mark_input_output_distances,
        num_context_lines,
        optimizer,
        save_source_context,
        save_rng_states,
        detect_loops,
        verbose,
    )
    model_log._run_and_log_inputs_through_model(
        model, input_args, input_kwargs, layers_to_save, random_seed
    )
    return model_log


def log_forward_pass(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    layers_to_save: Optional[Union[str, List]] = "all",
    keep_unsaved_layers: bool = True,
    output_device: str = "same",
    activation_postfunc: Optional[Callable] = None,
    mark_input_output_distances: bool = False,
    detach_saved_tensors: bool = False,
    save_function_args: bool = False,
    save_gradients: bool = False,
    save_source_context: bool = False,
    save_rng_states: bool = False,
    vis_mode: str = "none",
    vis_nesting_depth: int = 1000,
    vis_outpath: str = "graph.gv",
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
    vis_buffer_layers: bool = False,
    vis_direction: str = "bottomup",
    vis_graph_overrides: Optional[Dict] = None,
    vis_node_overrides: Optional[Dict] = None,
    vis_nested_node_overrides: Optional[Dict] = None,
    vis_edge_overrides: Optional[Dict] = None,
    vis_gradient_edge_overrides: Optional[Dict] = None,
    vis_module_overrides: Optional[Dict] = None,
    vis_node_placement: str = "auto",
    vis_renderer: str = "graphviz",
    vis_theme: str = "torchlens",
    random_seed: Optional[int] = None,
    num_context_lines: int = 7,
    optimizer=None,
    detect_loops: bool = True,
    verbose: bool = False,
) -> ModelLog:
    """Run a forward pass through *model*, log every operation, and return a ModelLog.

    This is the primary user-facing entry point for TorchLens.  It intercepts every
    tensor-producing operation during ``model.forward()``, records metadata and
    (optionally) saves activations, then returns a ``ModelLog`` that provides
    dict-like access to every layer's data.

    **Layer selection** (``layers_to_save``):

    - ``'all'`` (default) — save activations for every layer.
    - ``'none'`` / ``None`` / ``[]`` — save no activations (metadata only).
    - A list containing any mix of:
      1. Layer name, e.g. ``'conv2d_1_1'`` (all passes).
      2. Pass-qualified label, e.g. ``'conv2d_1_1:2'`` (second pass only).
      3. Module address, e.g. ``'features.0'`` (output of that module).
      4. Integer index (ordinal position; negative indices work).
      5. Substring filter, e.g. ``'conv2d'`` (all matching layers).

    When specific layers are requested, a **two-pass strategy** is used: first an
    exhaustive pass discovers the full graph structure (needed to resolve names),
    then ``save_new_activations`` replays the model in fast mode to save only the
    requested layers.  For ``'all'`` or ``'none'``, a single pass suffices.

    Args:
        model: PyTorch model.
        input_args: Positional args for ``model.forward()``; a single tensor or list.
        input_kwargs: Keyword args for ``model.forward()``.
        layers_to_save: Which layers to save activations for (see above).
        keep_unsaved_layers: If False, layers without saved activations are removed from
            the returned ModelLog (they still exist during processing).
        output_device: Device for stored tensors: ``'same'``, ``'cpu'``, or ``'cuda'``.
        activation_postfunc: Optional function applied to each activation before saving.
        mark_input_output_distances: Compute BFS distances from inputs/outputs (expensive).
        detach_saved_tensors: If True, detach saved tensors from the autograd graph.
        save_function_args: Store non-tensor args for each function call (needed for
            ``validate_saved_activations``).
        save_gradients: Capture gradients during a subsequent backward pass.
        save_source_context: If True, record the Python call stack for each
            tensor operation and capture module source code (file, line, signatures).
            Default False for speed; enable for debugging and code inspection.
        save_rng_states: If True, capture RNG states before each operation (needed for
            validation replay of stochastic ops like dropout). Auto-enabled when
            ``validate_forward_pass`` is used. Default False for speed.
        vis_mode: ``'none'`` (default), ``'rolled'``, or ``'unrolled'`` visualization.
        vis_nesting_depth: Max module nesting depth shown in visualization.
        vis_outpath: Output file path for the graph visualization.
        vis_save_only: If True, save the visualization file without displaying it.
        vis_fileformat: Image format (``'pdf'``, ``'png'``, ``'jpg'``, etc.).
        vis_buffer_layers: Include buffer layers in the visualization.
        vis_direction: Layout direction: ``'bottomup'``, ``'topdown'``, or ``'leftright'``.
        vis_graph_overrides: Graphviz graph-level attribute overrides.
        vis_node_overrides: Graphviz node attribute overrides.
        vis_nested_node_overrides: Graphviz attribute overrides for nested (module) nodes.
        vis_edge_overrides: Graphviz edge attribute overrides.
        vis_gradient_edge_overrides: Graphviz attribute overrides for gradient edges.
        vis_module_overrides: Graphviz subgraph (module cluster) attribute overrides.
        vis_node_placement: Layout engine: ``'auto'`` (default), ``'dot'``, ``'elk'``,
            or ``'sfdp'``.  ``'auto'`` uses dot for small graphs and ELK/sfdp for large.
        vis_renderer: Visualization backend: ``'graphviz'`` (default) or ``'dagua'``.
        vis_theme: Named Dagua theme when ``vis_renderer='dagua'``.
        random_seed: Fixed RNG seed for reproducibility with stochastic models.
        num_context_lines: Lines of source context to capture per function call.
        optimizer: Optional optimizer to annotate which params are being optimized.
        verbose: If True, print timed progress messages at each major pipeline stage.

    Returns:
        A ``ModelLog`` containing layer activations (if requested) and full metadata.
    """
    # DataParallel is not supported — unwrap and warn if present.
    warn_parallel()
    model = _unwrap_data_parallel(model)

    if vis_mode not in ["none", "rolled", "unrolled"]:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    if output_device not in ["same", "cpu", "cuda"]:
        raise ValueError("output_device must be either 'same', 'cpu', or 'cuda'.")

    if type(layers_to_save) is str:
        layers_to_save = layers_to_save.lower()

    if layers_to_save in ["all", "none", None, []]:
        # --- SINGLE-PASS path ---
        # "all" or "none": no name resolution needed, so one pass suffices.
        model_log = _run_model_and_save_specified_activations(
            model=model,
            input_args=input_args,  # type: ignore[arg-type]
            input_kwargs=input_kwargs,  # type: ignore[arg-type]
            layers_to_save=layers_to_save,
            keep_unsaved_layers=keep_unsaved_layers,
            output_device=output_device,
            activation_postfunc=activation_postfunc,
            mark_input_output_distances=mark_input_output_distances,
            detach_saved_tensors=detach_saved_tensors,
            save_function_args=save_function_args,
            save_gradients=save_gradients,
            random_seed=random_seed,
            num_context_lines=num_context_lines,
            optimizer=optimizer,
            save_source_context=save_source_context,
            save_rng_states=save_rng_states,
            detect_loops=detect_loops,
            verbose=verbose,
        )
    else:
        # --- TWO-PASS path ---
        # Pass 1 (exhaustive): Run with layers_to_save=None and keep_unsaved_layers=True
        # so the full graph is discovered and all layer labels are assigned.  No
        # activations are saved yet — this pass is purely for metadata/structure.
        if verbose:
            print("[torchlens] Two-pass mode: Pass 1 (exhaustive, metadata only)")
        model_log = _run_model_and_save_specified_activations(
            model=model,
            input_args=input_args,  # type: ignore[arg-type]
            input_kwargs=input_kwargs,  # type: ignore[arg-type]
            layers_to_save=None,
            keep_unsaved_layers=True,
            output_device=output_device,
            activation_postfunc=activation_postfunc,
            mark_input_output_distances=mark_input_output_distances,
            detach_saved_tensors=detach_saved_tensors,
            save_function_args=save_function_args,
            save_gradients=save_gradients,
            random_seed=random_seed,
            num_context_lines=num_context_lines,
            optimizer=optimizer,
            save_source_context=save_source_context,
            save_rng_states=save_rng_states,
            detect_loops=detect_loops,
            verbose=verbose,
        )
        # Pass 2 (fast): Now that layer labels exist, resolve the user's requested
        # layers and replay the model, saving only the matching activations.
        _vprint(model_log, "Two-pass mode: Pass 2 (fast, saving requested layers)")
        model_log.keep_unsaved_layers = keep_unsaved_layers
        model_log.save_new_activations(
            model=model,
            input_args=input_args,  # type: ignore[arg-type]
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,  # type: ignore[arg-type]
            random_seed=random_seed,
        )

    # Print final summary.
    _vprint(
        model_log,
        f"Done: {len(model_log.layer_logs)} layers, "
        f"{model_log.num_tensors_saved} saved, "
        f"{model_log.total_activation_memory_str}",
    )

    # Visualize if desired.
    if vis_mode != "none":
        model_log.render_graph(
            vis_mode,
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
            vis_node_placement=vis_node_placement,
            vis_renderer=vis_renderer,
            vis_theme=vis_theme,
        )

    return model_log


def get_model_metadata(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
) -> ModelLog:
    """Return model metadata without saving any activations.

    Equivalent to ``log_forward_pass(model, input_args, input_kwargs, layers_to_save=None,
    mark_input_output_distances=True)``.  Prefer using ``log_forward_pass`` directly —
    this wrapper exists for backward compatibility and may be removed in a future release.

    Args:
        model: PyTorch model to inspect.
        input_args: Positional args for ``model.forward()``.
        input_kwargs: Keyword args for ``model.forward()``.

    Returns:
        ModelLog with full metadata but no saved activations.
    """
    model_log = log_forward_pass(
        model,
        input_args,
        input_kwargs,
        layers_to_save=None,
        mark_input_output_distances=True,
    )
    return model_log


def show_model_graph(
    model: nn.Module,
    input_args: Union[torch.Tensor, List, Tuple],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    vis_mode: str = "unrolled",
    vis_nesting_depth: int = 1000,
    vis_outpath: str = "graph.gv",
    vis_graph_overrides: Optional[Dict] = None,
    vis_node_overrides: Optional[Dict] = None,
    vis_nested_node_overrides: Optional[Dict] = None,
    vis_edge_overrides: Optional[Dict] = None,
    vis_gradient_edge_overrides: Optional[Dict] = None,
    vis_module_overrides: Optional[Dict] = None,
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
    vis_buffer_layers: bool = False,
    vis_direction: str = "bottomup",
    vis_node_placement: str = "auto",
    vis_renderer: str = "graphviz",
    vis_theme: str = "torchlens",
    random_seed: Optional[int] = None,
    detect_loops: bool = True,
    verbose: bool = False,
) -> None:
    """Convenience wrapper: visualize the computational graph without saving activations.

    Runs an exhaustive forward pass (no activations saved) to discover the graph
    structure, renders the visualization, then cleans up the ModelLog.  For more
    control, use ``log_forward_pass`` with ``vis_mode`` set and access the ModelLog
    directly.

    Args:
        model: PyTorch model.
        input_args: Positional args for ``model.forward()``.
        input_kwargs: Keyword args for ``model.forward()``.
        vis_mode: ``'rolled'`` or ``'unrolled'`` (``'none'`` is accepted but a no-op).
        vis_nesting_depth: Max module nesting depth shown (default 1000 = all).
        vis_outpath: Output file path for the visualization.
        vis_save_only: If True, save without displaying.
        vis_fileformat: Image format (``'pdf'``, ``'png'``, ``'jpg'``, etc.).
        vis_buffer_layers: Include buffer layers in the visualization.
        vis_direction: ``'bottomup'``, ``'topdown'``, or ``'leftright'``.
        vis_renderer: Visualization backend: ``'graphviz'`` (default) or ``'dagua'``.
        vis_theme: Named Dagua theme when ``vis_renderer='dagua'``.
        random_seed: Fixed RNG seed for stochastic models.

    Returns:
        None.
    """
    model = _unwrap_data_parallel(model)
    if not input_kwargs:
        input_kwargs = {}

    if vis_mode not in ["none", "rolled", "unrolled"]:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    model_log = _run_model_and_save_specified_activations(
        model=model,
        input_args=input_args,  # type: ignore[arg-type]
        input_kwargs=input_kwargs,
        layers_to_save=None,
        activation_postfunc=None,
        mark_input_output_distances=False,
        detach_saved_tensors=False,
        save_gradients=False,
        random_seed=random_seed,
        detect_loops=detect_loops,
        verbose=verbose,
    )
    # Render in a try/finally so temporary tl_ attributes on the model are
    # always cleaned up, even if Graphviz rendering raises.
    try:
        model_log.render_graph(
            vis_mode,
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
            vis_node_placement=vis_node_placement,
            vis_renderer=vis_renderer,
            vis_theme=vis_theme,
        )
    finally:
        model_log.cleanup()


def validate_forward_pass(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    random_seed: Union[int, None] = None,
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool:
    """Validate that saved activations faithfully reproduce the model's output.

    **How it works:**

    1. Run model.forward() *without* TorchLens to get ground-truth output tensors.
    2. Run ``log_forward_pass`` with ``save_function_args=True`` and ``layers_to_save='all'``
       to capture every activation and its creating function's arguments.
    3. Call ``ModelLog.validate_saved_activations`` which replays the forward pass
       layer-by-layer from saved activations, checking that the output matches
       ground truth.  It also injects random activations and verifies the output
       changes (proving the saved activations are actually used, not just ignored).
    4. If ``validate_metadata=True``, run comprehensive invariant checks on all
       metadata cross-references (graph edges, module containment, labels, etc.).

    **Why save_function_args=True is required:**  The validation replay re-executes
    each function using its saved non-tensor arguments (e.g., stride, padding for
    conv2d).  Without them, replay cannot reconstruct the correct computation.

    Args:
        model: PyTorch model.
        input_args: Input for which to validate the saved activations.
        input_kwargs: Keyword arguments for model forward pass.
        random_seed: Fixed RNG seed for reproducibility (auto-generated if None).
        verbose: If True, print detailed error messages on validation failure.
        validate_metadata: If True (default), also run metadata invariant checks.

    Returns:
        True if all validation checks pass, False otherwise.
    """
    warn_parallel()
    model = _unwrap_data_parallel(model)
    # Fix a random seed so both the ground-truth run and the logged run see
    # identical randomness (critical for models with dropout, etc.).
    if random_seed is None:
        random_seed = random.randint(1, 4294967294)
    set_random_seed(random_seed)
    input_args = normalize_input_args(input_args, model)
    if not input_kwargs:
        input_kwargs = {}
    # Deep-copy inputs so the ground-truth forward pass doesn't mutate the
    # originals (some models modify inputs in-place).
    input_args_copy = safe_copy_args(input_args)
    input_kwargs_copy = safe_copy_kwargs(input_kwargs)

    model_device = next((p.device for p in model.parameters()), None)
    if model_device is not None:
        input_args_copy = _move_tensors_to_device(input_args_copy, model_device)
        input_kwargs_copy = _move_tensors_to_device(input_kwargs_copy, model_device)

    # Step 1: Get ground-truth outputs by running the model *outside* TorchLens.
    # Save state_dict first because requires_grad forcing during logging can
    # alter parameter metadata; we restore it afterward.
    state_dict = model.state_dict()
    ground_truth_output_all = get_vars_of_type_from_obj(
        model(*input_args_copy, **input_kwargs_copy),
        torch.Tensor,
        search_depth=5,
        return_addresses=True,
        allow_repeats=True,
    )
    # Deduplicate by structural address to match how capture/trace.py extracts
    # outputs (same tensor returned in multiple positions is counted once).
    addresses_used = []
    ground_truth_output_tensors = []
    for entry in ground_truth_output_all:
        if entry[1] in addresses_used:
            continue
        ground_truth_output_tensors.append(entry[0])
        addresses_used.append(entry[1])
    model.load_state_dict(state_dict)

    # Step 2: Run the model *through* TorchLens, saving all activations.
    # save_function_args=True is essential — the replay needs each function's
    # non-tensor arguments to re-execute the computation from saved activations.
    model_log = _run_model_and_save_specified_activations(
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
        save_rng_states=True,
    )
    # Step 3: Validate by replaying the forward pass from saved activations.
    try:
        activations_are_valid = model_log.validate_saved_activations(
            ground_truth_output_tensors, verbose, validate_metadata=validate_metadata
        )
    finally:
        model_log.cleanup()
        del model_log
    return activations_are_valid


def validate_saved_activations(
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any], Tuple[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    random_seed: Union[int, None] = None,
    verbose: bool = False,
) -> bool:
    """Deprecated: use ``validate_forward_pass`` instead."""
    import warnings

    warnings.warn(
        "validate_saved_activations is deprecated, use validate_forward_pass instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return validate_forward_pass(
        model, input_args, input_kwargs, random_seed=random_seed, verbose=verbose
    )


def validate_batch_of_models_and_inputs(
    models_and_inputs_dict: Dict[str, Dict[str, Union[str, Callable, Dict]]],
    out_path: str,
    redo_model_if_already_run: bool = True,
) -> pd.DataFrame:
    """Batch-validate multiple models, writing incremental results to a CSV.

    For each model/input pair, calls ``validate_forward_pass`` and appends the
    result to a running CSV at *out_path*.  If the CSV already exists, previously
    validated models can be skipped (controlled by *redo_model_if_already_run*).

    Args:
        models_and_inputs_dict: Mapping of model_name to a dict with keys:
            - ``model_category`` (str): grouping label (e.g. 'torchvision').
            - ``model_loading_func`` (callable): zero-arg function returning an nn.Module.
            - ``model_sample_inputs`` (dict[str, input]): named sample inputs.
        out_path: File path for the results CSV (created if absent, appended otherwise).
        redo_model_if_already_run: Re-validate models already present in the CSV.

    Returns:
        DataFrame with columns: model_category, model_name, input_name, validation_success.
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
    for model_name, model_info in tqdm(models_and_inputs_dict.items(), desc="Validating models"):
        print(f"Validating model {model_name}")
        if model_name in models_already_run and not redo_model_if_already_run:
            continue
        model_category = model_info["model_category"]
        model_loading_func = model_info["model_loading_func"]
        model = model_loading_func()
        model_sample_inputs = model_info["model_sample_inputs"]
        for input_name, input_data in model_sample_inputs.items():
            validation_success = validate_forward_pass(model, input_data)
            current_csv = pd.concat(
                [
                    current_csv,
                    pd.DataFrame(
                        [
                            {
                                "model_category": model_category,
                                "model_name": model_name,
                                "input_name": input_name,
                                "validation_success": validation_success,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        current_csv.to_csv(out_path, index=False)
        del model
    return current_csv
