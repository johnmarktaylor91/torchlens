"""Focused utility modules: RNG, tensor ops, argument handling, introspection, collections, hashing, display."""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

import torch
from torch import nn

from .rng import (
    set_random_seed,
    log_current_rng_states,
    set_rng_from_saved_states,
    log_current_autocast_state,
    AutocastRestore,
    _AUTOCAST_DEVICES,
)
from .tensor_utils import (
    MAX_FLOATING_POINT_TOLERANCE,
    _cuda_available,
    _is_cuda_available,
    tensor_all_nan,
    tensor_nanequal,
    safe_to,
    get_tensor_memory_amount,
    safe_copy,
    print_override,
)
from .arg_handling import (
    _safe_copy_arg,
    safe_copy_args,
    safe_copy_kwargs,
    _model_expects_single_arg,
    normalize_input_args,
)
from .introspection import (
    _ATTR_SKIP_SET,
    get_vars_of_type_from_obj,
    get_attr_values_from_tensor_list,
    nested_getattr,
    nested_assign,
    iter_accessible_attributes,
    remove_attributes_with_prefix,
    _get_func_call_stack,
)
from .collections import (
    is_iterable,
    ensure_iterable,
    index_nested,
    remove_entry_from_list,
    assign_to_sequence_or_dict,
)
from .hashing import (
    make_random_barcode,
    make_short_barcode_from_input,
)
from .display import (
    identity,
    int_list_to_compact_str,
    human_readable_size,
    in_notebook,
    warn_parallel,
)


def list_modules(model: nn.Module) -> list[tuple[str, type[nn.Module]]]:
    """List every module registered on a model.

    Parameters
    ----------
    model:
        PyTorch module to inspect.

    Returns
    -------
    list[tuple[str, type[nn.Module]]]
        Module addresses paired with concrete module classes. The root module is
        reported as ``"self"``.
    """

    return [(address or "self", type(module)) for address, module in model.named_modules()]


def _ops_from_log(model_log: Any) -> list[tuple[str, int]]:
    """Summarize operator counts from a ModelLog.

    Parameters
    ----------
    model_log:
        TorchLens log produced by ``log_forward_pass``.

    Returns
    -------
    list[tuple[str, int]]
        Operator names and counts sorted by first observed name.
    """

    counts: Counter[str] = Counter()
    for layer in model_log.layer_list:
        op_name = getattr(layer, "func_name", None) or getattr(layer, "layer_type", "unknown")
        counts[str(op_name)] += 1
    return sorted(counts.items())


def _log_ops_for_mode(
    model: nn.Module,
    x: Any,
    mode: Literal["current", "eval", "train"],
) -> list[tuple[str, int]]:
    """Run a metadata-only capture and return operator counts.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Input passed to ``log_forward_pass``.
    mode:
        Module training mode to use for this one capture.

    Returns
    -------
    list[tuple[str, int]]
        Operator names and counts.
    """

    from torchlens import log_forward_pass
    from torchlens.options import CaptureOptions

    original_mode = model.training
    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    try:
        model_log = log_forward_pass(
            model,
            x,
            capture=CaptureOptions(layers_to_save=None, keep_unsaved_layers=True),
        )
    finally:
        model.train(original_mode)
    return _ops_from_log(model_log)


def list_ops(
    model: nn.Module,
    x: Any,
    mode: Literal["current", "eval", "train", "both"] = "current",
) -> list[tuple[str, int]] | dict[str, list[tuple[str, int]]]:
    """List operators that run in a model forward pass.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Input passed to ``log_forward_pass``.
    mode:
        ``"current"``, ``"eval"``, ``"train"``, or ``"both"``. ``"both"``
        returns separate eval/train summaries.

    Returns
    -------
    list[tuple[str, int]] | dict[str, list[tuple[str, int]]]
        Operator count table, or eval/train tables for ``mode="both"``.
    """

    if mode == "both":
        return {
            "eval": _log_ops_for_mode(model, x, "eval"),
            "train": _log_ops_for_mode(model, x, "train"),
        }
    if mode not in {"current", "eval", "train"}:
        raise ValueError("mode must be 'current', 'eval', 'train', or 'both'.")
    return _log_ops_for_mode(model, x, mode)


def flop_count(model: nn.Module, x: Any) -> int:
    """Return a lightweight FLOP count from TorchLens per-layer metadata.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Input passed to ``log_forward_pass``.

    Returns
    -------
    int
        Sum of available forward FLOP estimates. Operators without a built-in
        estimate contribute zero.
    """

    from torchlens import log_forward_pass
    from torchlens.options import CaptureOptions

    model_log = log_forward_pass(
        model,
        x,
        capture=CaptureOptions(layers_to_save=None, keep_unsaved_layers=True),
    )
    return int(sum(getattr(layer, "flops_forward", None) or 0 for layer in model_log.layer_list))


def peek_graph(
    model: nn.Module,
    x: Any,
    view: Literal["unrolled", "rolled", "none"] = "unrolled",
    output_path: str | Path = "modelgraph",
    file_format: str = "pdf",
) -> None:
    """Capture and render a model graph with quickstart defaults.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Input passed to ``log_forward_pass``.
    view:
        Visualization view vocabulary passed as ``vis_mode``.
    output_path:
        Output path stem for the renderer.
    file_format:
        Renderer output format.

    Returns
    -------
    None
        The graph renderer writes its output as a side effect.
    """

    from torchlens import log_forward_pass
    from torchlens.options import CaptureOptions

    model_log = log_forward_pass(
        model,
        x,
        capture=CaptureOptions(layers_to_save=None, keep_unsaved_layers=True),
    )
    model_log.show(
        vis_mode=view,
        vis_outpath=str(output_path),
        vis_fileformat=file_format,
        vis_save_only=True,
    )
    return None


def _shape_from_annotation(annotation: Any) -> tuple[int, ...] | None:
    """Extract a tensor shape from a forward-parameter annotation.

    Parameters
    ----------
    annotation:
        Raw annotation from ``inspect.signature``.

    Returns
    -------
    tuple[int, ...] | None
        Shape tuple when inferable.
    """

    if annotation is inspect.Signature.empty:
        return None
    if isinstance(annotation, (tuple, list)) and all(isinstance(dim, int) for dim in annotation):
        return tuple(annotation)
    if get_origin(annotation) is Annotated:
        for item in get_args(annotation)[1:]:
            if isinstance(item, (tuple, list)) and all(isinstance(dim, int) for dim in item):
                return tuple(item)
    return None


def _synthetic_arg_for_parameter(parameter: inspect.Parameter) -> torch.Tensor:
    """Build one synthetic tensor for a forward parameter.

    Parameters
    ----------
    parameter:
        Parameter from ``model.forward``.

    Returns
    -------
    torch.Tensor
        Zero tensor matching the inferred shape.

    Raises
    ------
    ValueError
        If no shape can be inferred.
    """

    if isinstance(parameter.default, torch.Tensor):
        return torch.zeros_like(parameter.default)
    if isinstance(parameter.default, (tuple, list)) and all(
        isinstance(dim, int) for dim in parameter.default
    ):
        return torch.zeros(tuple(parameter.default))
    shape = _shape_from_annotation(parameter.annotation)
    if shape is None:
        raise ValueError(
            "Cannot infer synthetic input shape for forward parameter "
            f"{parameter.name!r}. Annotate it as Annotated[torch.Tensor, (..shape..)] "
            "or provide a tensor/shape default."
        )
    return torch.zeros(shape)


def synthetic_input(model: nn.Module) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Generate a dummy input from a model's ``forward`` signature.

    Parameters
    ----------
    model:
        PyTorch model whose ``forward`` signature should be inspected.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, ...]
        One tensor for single-input models, otherwise a tuple of positional tensors.

    Raises
    ------
    ValueError
        If any required input shape cannot be inferred from the signature alone.
    """

    signature = inspect.signature(model.forward)
    annotations = get_type_hints(model.forward, include_extras=True)
    args: list[torch.Tensor] = []
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.default is not inspect.Signature.empty and not isinstance(
            parameter.default, (torch.Tensor, tuple, list)
        ):
            continue
        if parameter.name in annotations:
            parameter = parameter.replace(annotation=annotations[parameter.name])
        args.append(_synthetic_arg_for_parameter(parameter))
    if not args:
        raise ValueError("Cannot infer any tensor inputs from model.forward signature.")
    if len(args) == 1:
        return args[0]
    return tuple(args)


def _memory_budget_to_bytes(memory_budget: int | str) -> int:
    """Convert an integer or simple memory string to bytes.

    Parameters
    ----------
    memory_budget:
        Byte count or string ending in KB, MB, or GB.

    Returns
    -------
    int
        Memory budget in bytes.
    """

    if isinstance(memory_budget, int):
        return memory_budget
    text = memory_budget.strip().lower()
    multipliers = {"kb": 1024, "mb": 1024**2, "gb": 1024**3}
    for suffix, multiplier in multipliers.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * multiplier)
    return int(text)


def find_executable_save_set(
    model_log: Any,
    layers: Iterable[str],
    memory_budget: int | str,
) -> list[str]:
    """Choose the largest heuristic subset of layers that fits a memory budget.

    Parameters
    ----------
    model_log:
        ModelLog containing layer memory metadata.
    layers:
        Candidate layer labels or lookup strings.
    memory_budget:
        Byte budget, or a simple string such as ``"64 MB"``.

    Returns
    -------
    list[str]
        Selected layer labels. The heuristic sorts candidates by activation
        memory ascending to maximize count.
    """

    budget = _memory_budget_to_bytes(memory_budget)
    candidates: list[tuple[int, str]] = []
    for layer in layers:
        entry = model_log[layer]
        label = str(getattr(entry, "layer_label_no_pass", getattr(entry, "layer_label", layer)))
        memory = int(
            getattr(entry, "transformed_activation_memory", None)
            or getattr(entry, "tensor_memory", None)
            or 0
        )
        candidates.append((memory, label))

    selected: list[str] = []
    total = 0
    for memory, label in sorted(candidates):
        if total + memory <= budget:
            selected.append(label)
            total += memory
    return selected


__all__ = [
    "AutocastRestore",
    "MAX_FLOATING_POINT_TOLERANCE",
    "_ATTR_SKIP_SET",
    "_AUTOCAST_DEVICES",
    "_cuda_available",
    "_get_func_call_stack",
    "_is_cuda_available",
    "_model_expects_single_arg",
    "_safe_copy_arg",
    "assign_to_sequence_or_dict",
    "ensure_iterable",
    "get_attr_values_from_tensor_list",
    "get_tensor_memory_amount",
    "get_vars_of_type_from_obj",
    "human_readable_size",
    "identity",
    "in_notebook",
    "index_nested",
    "int_list_to_compact_str",
    "is_iterable",
    "iter_accessible_attributes",
    "find_executable_save_set",
    "flop_count",
    "list_modules",
    "list_ops",
    "log_current_autocast_state",
    "log_current_rng_states",
    "make_random_barcode",
    "make_short_barcode_from_input",
    "nested_assign",
    "nested_getattr",
    "normalize_input_args",
    "print_override",
    "peek_graph",
    "remove_attributes_with_prefix",
    "remove_entry_from_list",
    "safe_copy",
    "safe_copy_args",
    "safe_copy_kwargs",
    "safe_to",
    "set_random_seed",
    "synthetic_input",
    "set_rng_from_saved_states",
    "tensor_all_nan",
    "tensor_nanequal",
    "warn_parallel",
]
