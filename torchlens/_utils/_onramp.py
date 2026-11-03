"""Convenience helpers documented at ``torchlens._utils``.

``list_modules``, ``list_ops``, ``flop_count``, ``peek_graph``,
``synthetic_input``, ``find_executable_save_set`` and ``trace_streaming`` are
quick-look wrappers over a normal capture. The README and the refusal contract
name them at ``torchlens._utils``, which re-exports them from here.
"""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated, Any, Literal, cast, get_args, get_origin, get_type_hints

import torch
from torch import nn

from .._record_type import record_type

# FACADE role (Rule F): the quick-look wrappers the README documents under
# ``torchlens._utils`` call ``torchlens.trace`` and the report layer lazily.
__tl_layer__ = "FACADE"


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


def _ops_from_log(trace: Any) -> list[tuple[str, int]]:
    """Summarize operator counts from a Trace.

    Parameters
    ----------
    trace:
        TorchLens log produced by ``trace``.

    Returns
    -------
    list[tuple[str, int]]
        Operator names and counts sorted by first observed name.
    """

    counts: Counter[str] = Counter()
    for layer in trace._layer_list:
        op_name = getattr(layer, "func_name", None) or record_type(layer, "unknown")
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
        Input passed to ``trace``.
    mode:
        Module training mode to use for this one capture.

    Returns
    -------
    list[tuple[str, int]]
        Operator names and counts.
    """

    from torchlens import trace as trace_fn

    # Snapshot every submodule's training flag, not just the root's. A recursive
    # ``model.train(root_mode)`` restore would clobber mixed child states (e.g. a
    # frozen ``bn.eval()`` under a training root). For ``mode="current"`` no mode
    # change is applied at all, so the model is left byte-for-byte as found.
    original_modes = {submodule: submodule.training for submodule in model.modules()}
    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    try:
        trace = cast(Callable[..., Any], trace_fn)(
            model,
            x,
            save=None,
        )
    finally:
        for submodule, was_training in original_modes.items():
            submodule.training = was_training
    return _ops_from_log(trace)


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
        Input passed to ``trace``.
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


def _refuse_flop_count_options(flop_convention: object, unsupported: dict[str, Any]) -> None:
    """Refuse a removed or unknown ``flop_count`` option before any capture runs."""

    from torchlens._errors import InvalidArgumentError

    if "count_fma_as_two" in unsupported:
        problem = (
            "flop_count() no longer accepts count_fma_as_two= (removed); use "
            "flop_convention='fma2' (was True) or flop_convention='fma1' (was False)."
        )
    elif unsupported:
        problem = (
            f"flop_count() got unknown option(s) {sorted(unsupported)}; its only "
            "option is flop_convention ('fma2' or 'fma1')."
        )
    elif flop_convention not in ("fma2", "fma1"):
        problem = (
            f"flop_count() got invalid flop_convention={flop_convention!r}; "
            "valid choices: 'fma2', 'fma1'."
        )
    else:
        return
    raise InvalidArgumentError(
        problem,
        code="flop_count_option_invalid",
        remedy="pass flop_convention='fma2' or flop_convention='fma1'",
    )


def flop_count(
    model: nn.Module,
    x: Any,
    *,
    flop_convention: Literal["fma2", "fma1"] = "fma2",
    **unsupported: Any,
) -> int:
    """Return a lightweight forward FLOP count from TorchLens metadata.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Input passed to ``trace``.
    flop_convention:
        ``"fma2"`` (default; one multiply-accumulate = 2 FLOPs, the stored
        convention) or ``"fma1"`` (recount from each op's two-term compute
        record). Ops with no derivable MAC split make the fma=1 conversion
        impossible and refuse typed (``flop_convention_unavailable``) -- the
        request is NEVER accepted-and-ignored. Same vocabulary as
        ``summary(flop_convention=...)``.
    **unsupported:
        Refused typed (``flop_count_option_invalid``). The removed
        ``count_fma_as_two`` spelling names ``flop_convention`` as its
        replacement.

    Raises
    ------
    InvalidArgumentError
        ``flop_count_option_invalid`` for an unknown or removed option name
        or a ``flop_convention`` outside ``"fma2"``/``"fma1"``.

    Returns
    -------
    int
        Sum of available forward FLOP estimates under the requested
        convention. Operators without a built-in estimate contribute zero
        (disclosed by ``Trace.summary``'s unknown-op count).
    """

    from torchlens import trace as trace_fn
    from torchlens.report._compute_truth import forward_flops_total

    _refuse_flop_count_options(flop_convention, unsupported)
    trace = cast(Callable[..., Any], trace_fn)(
        model,
        x,
        save=None,
    )
    fma = 1 if flop_convention == "fma1" else 2
    return int(forward_flops_total(trace, fma=fma))


def peek_graph(
    model: nn.Module,
    x: Any,
    view: Literal["unrolled", "rolled", "none"] = "unrolled",
    container_path: str | Path = "modelgraph",
    file_format: str = "pdf",
) -> None:
    """Capture and render a model graph with quickstart defaults.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Input passed to ``trace``.
    view:
        Which graph to draw: ``"unrolled"`` (every pass), ``"rolled"`` (each
        repeated module once) or ``"none"`` (capture only, no picture).
    container_path:
        Output path stem for the renderer.
    file_format:
        Renderer output format.

    Returns
    -------
    None
        The graph renderer writes its output as a side effect.
    """

    from torchlens import trace as trace_fn

    trace = cast(Callable[..., Any], trace_fn)(
        model,
        x,
        save=None,
    )
    if view == "none":
        return None
    trace.draw(
        path=str(container_path),
        rolled=view == "rolled",
        file_format=file_format,
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
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            # The public return is positional-only, so a keyword-only argument
            # can never be delivered through it. An optional keyword-only param
            # is safely omitted (forward uses its default); a *required* one
            # cannot be represented and must fail loudly here rather than emit a
            # positional tuple that raises a confusing TypeError at forward call.
            if parameter.default is not inspect.Signature.empty:
                continue
            raise ValueError(
                "Cannot build a positional synthetic input for required "
                f"keyword-only forward parameter {parameter.name!r}. "
                "synthetic_input only returns positional tensors; pass this "
                "input explicitly."
            )
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
    trace: Any,
    layers: Iterable[str],
    memory_budget: int | str,
) -> list[str]:
    """Choose the largest heuristic subset of layers that fits a memory budget.

    Parameters
    ----------
    trace:
        Trace containing layer memory metadata.
    layers:
        Candidate layer labels or lookup strings.
    memory_budget:
        Byte budget, or a simple string such as ``"64 MB"``.

    Returns
    -------
    list[str]
        Selected layer labels. The heuristic sorts candidates by out
        memory ascending to maximize count.
    """

    budget = _memory_budget_to_bytes(memory_budget)
    candidates: list[tuple[int, str]] = []
    for layer in layers:
        entry = trace[layer]
        label = str(getattr(entry, "layer_label", layer))
        memory = int(
            getattr(entry, "transformed_activation_memory", None)
            or getattr(entry, "activation_memory", None)
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


def trace_streaming(model: nn.Module, inputs_iter: Iterable[Any], **kwargs: Any) -> Any:
    """Capture an iterable of inputs as a stacked multi-pass log.

    Parameters
    ----------
    model:
        Model to capture.
    inputs_iter:
        Iterable producing model inputs.
    **kwargs:
        Keyword arguments forwarded to ``torchlens.trace``.

    Returns
    -------
    tuple[Any, ...]
        Captured traces, one per input item.
    """

    import torchlens

    trace_fn = cast(Callable[..., Any], torchlens.trace)
    logs = [trace_fn(model, inputs, **kwargs) for inputs in inputs_iter]
    if not logs:
        raise ValueError("inputs_iter must yield at least one input.")
    return tuple(logs)
