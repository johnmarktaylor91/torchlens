"""Input-shape inference helpers for TorchLens debug utilities."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from torchlens._errors import ShapeInferenceError

if TYPE_CHECKING:
    import pandas as pd

    from torchlens.data_classes.op import Op
    from torchlens.data_classes.trace import Trace

FailureReason = Literal[
    "non_shape_blocker",
    "exact_size_unreachable",
    "multi_input_unsupported",
    "budget_exhausted",
    "unknown_entry",
]


@dataclass(frozen=True)
class InferInputShapeResult:
    """Result returned by :func:`infer_input_shape`.

    Parameters
    ----------
    found:
        Whether a verified input was found.
    shape:
        Single tensor input shape, when applicable.
    shapes:
        Multi-input shapes, when applicable.
    dtype:
        Input dtype for the primary tensor.
    value_range:
        Synthetic value recipe: ``("uniform", 0, 1)`` or ``("randint", 0, vocab)``.
    flexible_dims:
        Dimensions that are expected to tolerate other sizes.
    constraining_module:
        Module qualname that constrained the inferred shape, when known.
    constraining_op:
        Operation name that constrained the inferred shape, when known.
    source_line:
        Source line for the constraining op, when known.
    example_input:
        Ready-to-use input object.
    strategy:
        Strategy that produced the winning input.
    reason:
        Failure reason when ``found`` is false.
    attempts:
        Probe diary as ``(shape, outcome)`` tuples.
    trace:
        Final verification trace when ``return_trace=True``.
    message:
        Actionable human-readable summary.
    """

    found: bool
    shape: tuple[int, ...] | None
    shapes: tuple[tuple[int, ...], ...] | None
    dtype: torch.dtype | None
    value_range: tuple[str, float, float] | None
    flexible_dims: tuple[int, ...]
    constraining_module: str | None
    constraining_op: str | None
    source_line: str | None
    example_input: Any | None
    strategy: str
    reason: str | None
    attempts: tuple[tuple[tuple[int, ...] | None, str], ...]
    trace: Trace | None
    message: str


@dataclass(frozen=True)
class _InputPrior:
    """Static facts inferred without running the model."""

    kind: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    value_range: tuple[str, float, float]
    flexible_dims: tuple[int, ...]
    constraining_module: str | None
    constraining_op: str | None
    strategy: str
    device: torch.device
    spatial_rank: int | None = None
    channels: int | None = None
    min_side: int = 1


@dataclass(frozen=True)
class _ProbeResult:
    """Outcome of one synthetic forward probe."""

    ok: bool
    outcome: str
    exception: Exception | None
    diary: tuple[tuple[str, tuple[tuple[int, ...], ...], tuple[str, ...]], ...]
    got_features: int | None
    target_features: int | None
    constraining_module: str | None


@dataclass(frozen=True)
class _ExecutedConstraint:
    """Constraint read from an executed TorchLens op."""

    kind: str
    label: str | None
    module: str | None
    source_line: str | None
    input_shape: tuple[int, ...] | None
    dtype: torch.dtype | None
    in_features: int | None = None
    in_channels: int | None = None
    spatial_rank: int | None = None
    kernel_size: tuple[int, ...] = ()
    flexible: bool = False


# Brittleness: these patterns parse PyTorch human error text. They are
# intentionally centralized so the infer_input_shape redesign can remove them.
_EXCEPTION_PARSE_PATTERNS: dict[str, re.Pattern[str]] = {
    "linear": re.compile(
        r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)"
    ),
    "channel": re.compile(
        r"expected input\[[^\]]+\] to have (\d+) channels, but got (\d+) channels"
    ),
    "bool": re.compile(r"mask|bool|boolean", re.IGNORECASE),
    "complex": re.compile(r"complex|ComplexFloat|ComplexDouble", re.IGNORECASE),
    "integer": re.compile(r"Long|Int|integer|indices", re.IGNORECASE),
    "kernel": re.compile(r"kernel size|calculated padded input size", re.IGNORECASE),
    "size": re.compile(r"size of tensor|shape|shapes|Expected|expected|dimension|mat1 and mat2"),
    "position": re.compile(
        r"(pos_embed|position_embeddings|positional|position|grid)", re.IGNORECASE
    ),
}
_LINEAR_RE = _EXCEPTION_PARSE_PATTERNS["linear"]
_CHANNEL_RE = _EXCEPTION_PARSE_PATTERNS["channel"]
_BOOL_RE = _EXCEPTION_PARSE_PATTERNS["bool"]
_COMPLEX_RE = _EXCEPTION_PARSE_PATTERNS["complex"]
_INTEGER_RE = _EXCEPTION_PARSE_PATTERNS["integer"]
_KERNEL_RE = _EXCEPTION_PARSE_PATTERNS["kernel"]
_SIZE_RE = _EXCEPTION_PARSE_PATTERNS["size"]
_POSITION_RE = _EXCEPTION_PARSE_PATTERNS["position"]


def _shape_tuple(shape: Any) -> tuple[int, ...] | None:
    """Convert a Torch/TorchLens shape-like object to an integer tuple.

    Parameters
    ----------
    shape:
        Shape-like object.

    Returns
    -------
    tuple[int, ...] | None
        Converted shape, or ``None`` when conversion is not possible.
    """

    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def _dtype_from_message(message: str) -> torch.dtype | None:
    """Infer a replacement dtype from a probe error message.

    Parameters
    ----------
    message:
        Exception message from a failed probe.

    Returns
    -------
    torch.dtype | None
        Suggested dtype, or ``None`` when the message is not dtype-specific.
    """

    if _BOOL_RE.search(message):
        return torch.bool
    if _COMPLEX_RE.search(message):
        return torch.complex64
    if _INTEGER_RE.search(message):
        return torch.long
    return None


def _is_skippable_shape_error(message: str) -> bool:
    """Return whether a failed probe should still allow other size probes.

    Parameters
    ----------
    message:
        Exception message.

    Returns
    -------
    bool
        Whether the error looks like a shape/rank/size miss.
    """

    return bool(
        _KERNEL_RE.search(message) or _SIZE_RE.search(message) or _CHANNEL_RE.search(message)
    )


def _rank_default(rank: int) -> int:
    """Return a conservative default side for a spatial rank.

    Parameters
    ----------
    rank:
        Spatial rank.

    Returns
    -------
    int
        Default side length.
    """

    return 16 if rank == 3 else 32


def _module_index(model: nn.Module) -> dict[nn.Module, int]:
    """Return definition-order indices for modules.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    dict[nn.Module, int]
        Module object to definition-order index.
    """

    return {module: index for index, module in enumerate(model.modules())}


def _module_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype | None]:
    """Return the first parameter or buffer device and dtype.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    tuple[torch.device, torch.dtype | None]
        Device and dtype, defaulting to CPU with no dtype when the model has no state.
    """

    for tensor in list(model.parameters(recurse=True)) + list(model.buffers(recurse=True)):
        return tensor.device, tensor.dtype
    return torch.device("cpu"), None


def _float_dtype(dtype: torch.dtype | None) -> torch.dtype:
    """Normalize a model dtype to a synthetic floating input dtype.

    Parameters
    ----------
    dtype:
        Model parameter dtype.

    Returns
    -------
    torch.dtype
        Floating dtype for generated inputs.
    """

    if dtype in {torch.float16, torch.bfloat16, torch.float64, torch.float32}:
        return dtype
    return torch.float32


def _first_module(
    model: nn.Module,
    classes: tuple[type[nn.Module], ...],
) -> tuple[str, nn.Module] | None:
    """Return the first named module matching any requested class.

    Parameters
    ----------
    model:
        Model to inspect.
    classes:
        Module classes to match.

    Returns
    -------
    tuple[str, nn.Module] | None
        Qualname and module, or ``None``.
    """

    for name, module in model.named_modules():
        if isinstance(module, classes):
            return name, module
    return None


def _has_adaptive_pool(model: nn.Module) -> bool:
    """Return whether the model contains an adaptive pooling module.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    bool
        Whether an adaptive pooling module is present.
    """

    return any(
        isinstance(
            module,
            (
                nn.AdaptiveAvgPool1d,
                nn.AdaptiveAvgPool2d,
                nn.AdaptiveAvgPool3d,
                nn.AdaptiveMaxPool1d,
                nn.AdaptiveMaxPool2d,
                nn.AdaptiveMaxPool3d,
            ),
        )
        for module in model.modules()
    )


def _shape_has_executed_adaptive_pool(trace: Trace) -> bool:
    """Return whether the executed trace used adaptive pooling.

    Parameters
    ----------
    trace:
        Successful TorchLens trace.

    Returns
    -------
    bool
        Whether an adaptive pooling op executed.
    """

    return any(
        "adaptive" in str(getattr(op, "func_name", "")).lower()
        and "pool" in str(getattr(op, "func_name", "")).lower()
        for op in trace.layers
    )


def _positional_side(model: nn.Module, stride: int) -> int | None:
    """Infer a square ViT image side from positional embeddings.

    Parameters
    ----------
    model:
        Model to inspect.
    stride:
        Patch stride.

    Returns
    -------
    int | None
        Inferred side length, or ``None``.
    """

    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.ndim < 2 or not _POSITION_RE.search(name):
            continue
        length = int(tensor.shape[1] if tensor.ndim >= 3 else tensor.shape[0])
        for offset in (2, 1, 0):
            patches = length - offset
            root = math.isqrt(patches)
            if root * root == patches and root > 0:
                return root * stride
    return None


def _positional_side_from_any_table(model: nn.Module, stride: int) -> int | None:
    """Infer a square ViT side from any plausible position table.

    Parameters
    ----------
    model:
        Model to inspect.
    stride:
        Patch stride.

    Returns
    -------
    int | None
        Inferred side length, or ``None``.
    """

    named_tensors = list(model.named_parameters()) + list(model.named_buffers())
    preferred = [(name, tensor) for name, tensor in named_tensors if _POSITION_RE.search(name)]
    fallback = [
        (name, tensor)
        for name, tensor in named_tensors
        if tensor.ndim == 3 and int(tensor.shape[0]) == 1 and int(tensor.shape[-1]) > 1
    ]
    for _name, tensor in [*preferred, *fallback]:
        length = int(tensor.shape[1] if tensor.ndim >= 3 else tensor.shape[0])
        for offset in (2, 1, 0):
            patches = length - offset
            root = math.isqrt(patches)
            if root * root == patches and root > 0:
                return root * stride
    return None


def _make_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    value_range: tuple[str, float, float],
) -> torch.Tensor:
    """Create a synthetic input tensor.

    Parameters
    ----------
    shape:
        Tensor shape.
    dtype:
        Tensor dtype.
    device:
        Tensor device.
    value_range:
        Synthetic value recipe.

    Returns
    -------
    torch.Tensor
        Generated tensor.
    """

    if dtype == torch.bool:
        return torch.rand(shape, device=device) > 0.5
    if dtype.is_complex:
        real = torch.randn(shape, device=device, dtype=torch.float32)
        imag = torch.randn(shape, device=device, dtype=torch.float32)
        return torch.complex(real, imag).to(dtype)
    if value_range[0] == "randint" or dtype in {
        torch.long,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
    }:
        return torch.randint(
            int(value_range[1]), int(value_range[2]), shape, device=device, dtype=dtype
        )
    return torch.rand(shape, device=device, dtype=dtype)


def _training_states(model: nn.Module) -> dict[nn.Module, bool]:
    """Capture training flags for all modules.

    Parameters
    ----------
    model:
        Model whose module states are captured.

    Returns
    -------
    dict[nn.Module, bool]
        Training flags keyed by module object.
    """

    return {module: module.training for module in model.modules()}


def _restore_training_states(states: dict[nn.Module, bool]) -> None:
    """Restore module training flags.

    Parameters
    ----------
    states:
        Training flags from :func:`_training_states`.
    """

    for module, training in states.items():
        module.train(training)


def _probe(model: nn.Module, example_input: Any, seed: int) -> _ProbeResult:
    """Run a no-grad forward probe with pre-hook shape diary.

    Parameters
    ----------
    model:
        Model to run.
    example_input:
        Input object passed to ``model``.
    seed:
        RNG seed.

    Returns
    -------
    _ProbeResult
        Probe outcome and diary.
    """

    diary: list[tuple[str, tuple[tuple[int, ...], ...], tuple[str, ...]]] = []
    handles: list[Any] = []

    def hook(name: str) -> Any:
        """Build a pre-hook that records incoming tensor metadata."""

        def record(_module: nn.Module, args: tuple[Any, ...]) -> None:
            """Record tensor shapes and dtypes from positional module inputs."""

            tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
            diary.append(
                (
                    name,
                    tuple(tuple(int(dim) for dim in tensor.shape) for tensor in tensors),
                    tuple(str(tensor.dtype) for tensor in tensors),
                )
            )

        return record

    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_pre_hook(hook(name)))

    states = _training_states(model)
    got_features: int | None = None
    target_features: int | None = None
    constraining_module: str | None = None
    try:
        torch.manual_seed(seed)
        model.eval()
        with torch.no_grad():
            if isinstance(example_input, tuple):
                model(*example_input)
            elif isinstance(example_input, dict):
                model(**example_input)
            else:
                model(example_input)
    except Exception as exc:  # noqa: BLE001 - debug inference classifies arbitrary model failures.
        message = str(exc)
        match = _LINEAR_RE.search(message)
        if match is not None:
            got_features = int(match.group(2))
            target_features = int(match.group(3))
        for name, shapes, _dtypes in reversed(diary):
            module = dict(model.named_modules()).get(name)
            if isinstance(module, nn.Linear):
                constraining_module = name
                if shapes:
                    got_features = int(shapes[0][-1])
                target_features = int(module.in_features)
                break
        return _ProbeResult(
            ok=False,
            outcome=message,
            exception=exc,
            diary=tuple(diary),
            got_features=got_features,
            target_features=target_features,
            constraining_module=constraining_module,
        )
    finally:
        for handle in handles:
            handle.remove()
        _restore_training_states(states)

    return _ProbeResult(
        ok=True,
        outcome="ok",
        exception=None,
        diary=tuple(diary),
        got_features=None,
        target_features=None,
        constraining_module=None,
    )


def _shape_from_prior(
    model: nn.Module,
    batch_size: int,
    input_dtype: torch.dtype | None,
    channels: int | None,
    spatial_rank: int | Literal["auto"],
    seq_len: int | None,
    min_size: int,
    preferred_sizes: Sequence[int],
    device: torch.device,
) -> _InputPrior | None:
    """Build the best static single-input prior.

    Parameters
    ----------
    model:
        Model to inspect.
    batch_size:
        Batch dimension.
    input_dtype:
        Optional dtype override.
    channels:
        Optional channel override.
    spatial_rank:
        Optional spatial rank override.
    seq_len:
        Optional sequence length override.
    min_size:
        Minimum free dimension.
    preferred_sizes:
        Preferred side lengths.
    device:
        Target device.

    Returns
    -------
    _InputPrior | None
        Static prior, or ``None`` when no known entry is found.
    """

    _model_device, model_dtype = _module_device_dtype(model)
    emb = _first_module(model, (nn.Embedding,))
    if emb is not None:
        name, module = emb
        assert isinstance(module, nn.Embedding)
        cap = _sequence_cap(model)
        size = seq_len or min(16, cap or 16)
        return _InputPrior(
            kind="embedding",
            shape=(batch_size, size),
            dtype=input_dtype or torch.long,
            value_range=("randint", 0.0, float(module.num_embeddings)),
            flexible_dims=(1,),
            constraining_module=name,
            constraining_op="Embedding",
            strategy="introspection",
            device=device,
        )

    rnn = _first_module(model, (nn.RNN, nn.LSTM, nn.GRU))
    conv = _first_module(model, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
    linear = _first_module(model, (nn.Linear,))
    if rnn is not None and (
        conv is None or list(model.modules()).index(rnn[1]) < list(model.modules()).index(conv[1])
    ):
        name, module = rnn
        assert isinstance(module, (nn.RNN, nn.LSTM, nn.GRU))
        size = seq_len or 16
        shape = (batch_size, size, int(module.input_size))
        if not bool(module.batch_first):
            shape = (size, batch_size, int(module.input_size))
        return _InputPrior(
            kind="rnn",
            shape=shape,
            dtype=input_dtype or _float_dtype(model_dtype),
            value_range=("uniform", 0.0, 1.0),
            flexible_dims=(1 if bool(module.batch_first) else 0,),
            constraining_module=name,
            constraining_op=module.__class__.__name__,
            strategy="introspection",
            device=device,
        )

    if conv is not None:
        name, module = conv
        assert isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
        rank = (
            int(spatial_rank)
            if spatial_rank != "auto"
            else {nn.Conv1d: 1, nn.Conv2d: 2, nn.Conv3d: 3}[type(module)]
        )
        channel_count = channels or int(module.in_channels)
        first_preferred = _rank_default(rank) if rank == 3 else next(iter(preferred_sizes), 32)
        side = max(min_size, first_preferred)
        stride = module.stride[0] if isinstance(module.stride, tuple) else int(module.stride)
        side = _positional_side(model, int(stride)) or side
        strategy = "fixed_size" if _positional_side(model, int(stride)) else "adaptive_default"
        flexible = tuple(range(2, 2 + rank)) if _has_adaptive_pool(model) else ()
        return _InputPrior(
            kind="conv",
            shape=(batch_size, channel_count, *([side] * rank)),
            dtype=input_dtype or _float_dtype(model_dtype),
            value_range=("uniform", 0.0, 1.0),
            flexible_dims=flexible,
            constraining_module=name,
            constraining_op=module.__class__.__name__,
            strategy=strategy,
            device=device,
            spatial_rank=rank,
            channels=channel_count,
        )

    if linear is not None:
        name, module = linear
        assert isinstance(module, nn.Linear)
        return _InputPrior(
            kind="linear",
            shape=(batch_size, int(module.in_features)),
            dtype=input_dtype or _float_dtype(model_dtype),
            value_range=("uniform", 0.0, 1.0),
            flexible_dims=(),
            constraining_module=name,
            constraining_op="Linear",
            strategy="introspection",
            device=device,
        )
    return None


def _parameter_priors(
    model: nn.Module,
    batch_size: int,
    input_dtype: torch.dtype | None,
    seq_len: int | None,
    min_size: int,
    preferred_sizes: Sequence[int],
    device: torch.device,
) -> list[_InputPrior]:
    """Build functional-op priors from raw parameter shapes.

    Parameters
    ----------
    model:
        Model to inspect.
    batch_size:
        Batch dimension.
    input_dtype:
        Optional dtype override.
    seq_len:
        Optional sequence length override.
    min_size:
        Minimum spatial side.
    preferred_sizes:
        Preferred spatial sizes.
    device:
        Target device.

    Returns
    -------
    list[_InputPrior]
        Candidate priors for functional models.
    """

    priors: list[_InputPrior] = []
    _model_device, model_dtype = _module_device_dtype(model)
    cap = _sequence_cap(model)
    for name, param in model.named_parameters():
        if param.ndim in {3, 4, 5}:
            rank = int(param.ndim - 2)
            channels = int(param.shape[1])
            kernel = tuple(int(dim) for dim in param.shape[2:])
            default_side = _rank_default(rank)
            first_preferred = next(iter(preferred_sizes), default_side)
            side = max(min_size, max(max(kernel), default_side if rank == 3 else first_preferred))
            if rank == 2:
                side = _positional_side_from_any_table(model, max(kernel)) or side
            priors.append(
                _InputPrior(
                    kind="conv",
                    shape=(batch_size, channels, *([side] * rank)),
                    dtype=input_dtype or _float_dtype(model_dtype),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=(),
                    constraining_module=None,
                    constraining_op=f"parameter:{name}",
                    strategy="op_seed",
                    device=device,
                    spatial_rank=rank,
                    channels=channels,
                    min_side=max(kernel),
                )
            )
        elif param.ndim == 2:
            rows, cols = int(param.shape[0]), int(param.shape[1])
            if rows > cols and ("embed" in name.lower() or "token" in name.lower()):
                size = seq_len or min(16, cap or 16)
                priors.append(
                    _InputPrior(
                        kind="embedding",
                        shape=(batch_size, size),
                        dtype=input_dtype or torch.long,
                        value_range=("randint", 0.0, float(rows)),
                        flexible_dims=(1,),
                        constraining_module=None,
                        constraining_op=f"parameter:{name}",
                        strategy="op_seed",
                        device=device,
                    )
                )
            priors.append(
                _InputPrior(
                    kind="linear",
                    shape=(batch_size, cols),
                    dtype=input_dtype
                    or _float_dtype(
                        param.dtype
                        if param.dtype.is_floating_point or param.dtype.is_complex
                        else model_dtype
                    ),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=(),
                    constraining_module=None,
                    constraining_op=f"parameter:{name}",
                    strategy="op_seed",
                    device=device,
                )
            )
    return priors


def _input_priors(
    model: nn.Module,
    batch_size: int,
    input_dtype: torch.dtype | None,
    channels: int | None,
    spatial_rank: int | Literal["auto"],
    seq_len: int | None,
    min_size: int,
    preferred_sizes: Sequence[int],
    device: torch.device,
) -> list[_InputPrior]:
    """Build ordered candidate input priors.

    Parameters
    ----------
    model:
        Model to inspect.
    batch_size:
        Batch dimension.
    input_dtype:
        Optional dtype override.
    channels:
        Optional channel override.
    spatial_rank:
        Optional spatial rank override.
    seq_len:
        Optional sequence length override.
    min_size:
        Minimum spatial side.
    preferred_sizes:
        Preferred spatial sizes.
    device:
        Target device.

    Returns
    -------
    list[_InputPrior]
        Candidate priors, de-duplicated by shape and dtype.
    """

    priors: list[_InputPrior] = []
    static = _shape_from_prior(
        model,
        batch_size,
        input_dtype,
        channels,
        spatial_rank,
        seq_len,
        min_size,
        preferred_sizes,
        device,
    )
    if static is not None:
        priors.append(static)

    _model_device, model_dtype = _module_device_dtype(model)
    index = _module_index(model)
    modules = [(name, module) for name, module in model.named_modules() if name]
    for name, module in sorted(modules, key=lambda item: index[item[1]]):
        if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            size = seq_len or 16
            shape = (batch_size, size, int(module.input_size))
            if not bool(module.batch_first):
                shape = (size, batch_size, int(module.input_size))
            priors.append(
                _InputPrior(
                    kind="rnn",
                    shape=shape,
                    dtype=input_dtype or _float_dtype(model_dtype),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=(1 if bool(module.batch_first) else 0,),
                    constraining_module=name,
                    constraining_op=module.__class__.__name__,
                    strategy="introspection",
                    device=device,
                )
            )
        elif isinstance(module, nn.Embedding):
            cap = _sequence_cap(model)
            size = seq_len or min(16, cap or 16)
            priors.append(
                _InputPrior(
                    kind="embedding",
                    shape=(batch_size, size),
                    dtype=input_dtype or torch.long,
                    value_range=("randint", 0.0, float(module.num_embeddings)),
                    flexible_dims=(1,),
                    constraining_module=name,
                    constraining_op="Embedding",
                    strategy="introspection",
                    device=device,
                )
            )
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            rank = (
                int(spatial_rank)
                if spatial_rank != "auto"
                else {nn.Conv1d: 1, nn.Conv2d: 2, nn.Conv3d: 3}[type(module)]
            )
            kernel = (
                module.kernel_size
                if isinstance(module.kernel_size, tuple)
                else (int(module.kernel_size),)
            )
            stride = module.stride[0] if isinstance(module.stride, tuple) else int(module.stride)
            side = _positional_side_from_any_table(model, int(stride)) or max(
                max(kernel),
                _rank_default(rank) if rank == 3 else next(iter(preferred_sizes), 32),
                min_size,
            )
            priors.append(
                _InputPrior(
                    kind="conv",
                    shape=(batch_size, channels or int(module.in_channels), *([side] * rank)),
                    dtype=input_dtype or _float_dtype(model_dtype),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=tuple(range(2, 2 + rank)) if _has_adaptive_pool(model) else (),
                    constraining_module=name,
                    constraining_op=module.__class__.__name__,
                    strategy="adaptive_default" if _has_adaptive_pool(model) else "introspection",
                    device=device,
                    spatial_rank=rank,
                    channels=channels or int(module.in_channels),
                    min_side=max(kernel),
                )
            )
        elif isinstance(module, nn.TransformerEncoderLayer):
            size = seq_len or 16
            priors.append(
                _InputPrior(
                    kind="transformer",
                    shape=(batch_size, size, int(module.self_attn.embed_dim)),
                    dtype=input_dtype or _float_dtype(model_dtype),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=(1,),
                    constraining_module=name,
                    constraining_op=module.__class__.__name__,
                    strategy="introspection",
                    device=device,
                )
            )
        elif isinstance(module, nn.GroupNorm):
            priors.append(
                _InputPrior(
                    kind="norm",
                    shape=(batch_size, int(module.num_channels), 32, 32),
                    dtype=input_dtype or _float_dtype(model_dtype),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=(2, 3),
                    constraining_module=name,
                    constraining_op="GroupNorm",
                    strategy="introspection",
                    device=device,
                )
            )
        elif isinstance(module, nn.Linear):
            priors.append(
                _InputPrior(
                    kind="linear",
                    shape=(batch_size, int(module.in_features)),
                    dtype=input_dtype or _float_dtype(module.weight.dtype),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=(),
                    constraining_module=name,
                    constraining_op="Linear",
                    strategy="introspection",
                    device=device,
                )
            )
    priors.extend(
        _parameter_priors(
            model, batch_size, input_dtype, seq_len, min_size, preferred_sizes, device
        )
    )

    unique: list[_InputPrior] = []
    seen: set[tuple[tuple[int, ...], torch.dtype, str]] = set()
    for prior in priors:
        key = (prior.shape, prior.dtype, prior.kind)
        if key not in seen:
            seen.add(key)
            unique.append(prior)
    return unique


def _sequence_cap(model: nn.Module) -> int | None:
    """Infer a maximum sequence length from positional tables.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    int | None
        Sequence cap, or ``None``.
    """

    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.ndim >= 2 and _POSITION_RE.search(name):
            return int(tensor.shape[1] if tensor.ndim >= 3 else tensor.shape[0])
    return None


def _candidate_sides(
    start: int,
    min_size: int,
    max_size: int,
    preferred_sizes: Sequence[int],
) -> list[int]:
    """Return ordered spatial side candidates.

    Parameters
    ----------
    start:
        Initial side from static prior.
    min_size:
        Minimum side.
    max_size:
        Maximum side.
    preferred_sizes:
        Preferred side lengths.

    Returns
    -------
    list[int]
        Unique side candidates in probe order.
    """

    values = [start, *preferred_sizes, 32, 28, 16, 8, min_size, max_size]
    return [
        side
        for index, side in enumerate(values)
        if min_size <= side <= max_size and side not in values[:index]
    ]


def _trace_model(model: nn.Module, example_input: Any) -> Trace:
    """Run a final TorchLens inference-only trace.

    Parameters
    ----------
    model:
        Model to trace.
    example_input:
        Input object.

    Returns
    -------
    Trace
        Completed TorchLens trace.
    """

    from torchlens.user_funcs import trace

    return trace(model, example_input, inference_only=True)


def _shape_op_label(op: Any) -> str | None:
    """Return an op label from a TorchLens layer/op object.

    Parameters
    ----------
    op:
        TorchLens op-like object.

    Returns
    -------
    str | None
        Label when present.
    """

    label = getattr(op, "layer_label", None) or getattr(op, "op_label", None)
    return str(label) if label is not None else None


def _shape_op_module(op: Any) -> str | None:
    """Return a module address from a TorchLens op-like object.

    Parameters
    ----------
    op:
        TorchLens op-like object.

    Returns
    -------
    str | None
        Module address when present.
    """

    module = getattr(op, "module_address", None) or getattr(op, "module", None)
    return str(module) if module is not None else None


def _shape_op_source_line(op: Any) -> str | None:
    """Return an op source-line string when TorchLens exposes one.

    Parameters
    ----------
    op:
        TorchLens op-like object.

    Returns
    -------
    str | None
        Source-line string, or ``None``.
    """

    source_line = getattr(op, "source_line", None)
    return str(source_line) if source_line is not None else None


def _shape_op_config_int(op: Any, key: str) -> int | None:
    """Read an integer from an op's ``func_config``.

    Parameters
    ----------
    op:
        TorchLens op-like object.
    key:
        Config key.

    Returns
    -------
    int | None
        Integer value when present.
    """

    config = getattr(op, "func_config", {}) or {}
    value = config.get(key)
    return int(value) if value is not None else None


def _executed_constraint(trace: Trace) -> _ExecutedConstraint | None:
    """Read the first executed input-consuming constraint op from a trace.

    Parameters
    ----------
    trace:
        Successful TorchLens trace.

    Returns
    -------
    _ExecutedConstraint | None
        Executed constraint metadata, or ``None``.
    """

    for op in trace.layers:
        func_name = str(getattr(op, "func_name", "")).lower()
        input_shapes = tuple(_shape_tuple(shape) for shape in getattr(op, "input_shapes", ()) or ())
        input_dtypes = tuple(getattr(op, "input_dtypes", ()) or ())
        first_shape = next((shape for shape in input_shapes if shape is not None), None)
        first_dtype = input_dtypes[0] if input_dtypes else None
        param_shapes = tuple(_shape_tuple(shape) for shape in getattr(op, "param_shapes", []) or [])
        if func_name in {"conv1d", "conv2d", "conv3d"}:
            weight_shape = next((shape for shape in param_shapes if shape is not None), None)
            kernel = weight_shape[2:] if weight_shape is not None and len(weight_shape) >= 3 else ()
            return _ExecutedConstraint(
                kind="conv",
                label=_shape_op_label(op),
                module=_shape_op_module(op),
                source_line=_shape_op_source_line(op),
                input_shape=first_shape,
                dtype=first_dtype,
                in_channels=_shape_op_config_int(op, "in_channels")
                or (
                    weight_shape[1] if weight_shape is not None and len(weight_shape) > 1 else None
                ),
                spatial_rank=len(kernel)
                or (len(first_shape) - 2 if first_shape is not None else None),
                kernel_size=kernel,
                flexible=_shape_has_executed_adaptive_pool(trace),
            )
        if func_name in {"linear", "addmm"}:
            weight_shape = next((shape for shape in param_shapes if shape is not None), None)
            return _ExecutedConstraint(
                kind="linear",
                label=_shape_op_label(op),
                module=_shape_op_module(op),
                source_line=_shape_op_source_line(op),
                input_shape=first_shape,
                dtype=first_dtype,
                in_features=_shape_op_config_int(op, "in_features")
                or (
                    weight_shape[1] if weight_shape is not None and len(weight_shape) > 1 else None
                ),
            )
        if func_name in {"matmul", "mm", "bmm"}:
            needed = None
            if len(param_shapes) >= 1 and param_shapes[0] is not None:
                needed = param_shapes[0][0]
            if first_shape is not None and len(first_shape) >= 1:
                needed = first_shape[-1]
            return _ExecutedConstraint(
                kind="matmul",
                label=_shape_op_label(op),
                module=_shape_op_module(op),
                source_line=_shape_op_source_line(op),
                input_shape=first_shape,
                dtype=first_dtype,
                in_features=needed,
            )
        if func_name == "embedding":
            vocab = None
            weight_shape = next((shape for shape in param_shapes if shape is not None), None)
            if weight_shape is not None:
                vocab = weight_shape[0]
            return _ExecutedConstraint(
                kind="embedding",
                label=_shape_op_label(op),
                module=_shape_op_module(op),
                source_line=_shape_op_source_line(op),
                input_shape=first_shape,
                dtype=first_dtype,
                in_features=vocab,
            )
    return None


def _verified_result(
    prior: _InputPrior,
    shape: tuple[int, ...],
    example: torch.Tensor,
    attempts: list[tuple[tuple[int, ...] | None, str]],
    trace_obj: Trace,
    return_trace: bool,
    strategy: str,
) -> InferInputShapeResult:
    """Build a successful result from a verified trace.

    Parameters
    ----------
    prior:
        Candidate prior that produced the input.
    shape:
        Verified input shape.
    example:
        Verified example tensor.
    attempts:
        Probe attempts.
    trace_obj:
        Verification trace.
    return_trace:
        Whether to retain the trace in the public result.
    strategy:
        Winning strategy label.

    Returns
    -------
    InferInputShapeResult
        Successful result.
    """

    constraint = _executed_constraint(trace_obj)
    flexible: tuple[int, ...] = ()
    if constraint is not None and constraint.kind == "conv" and constraint.flexible:
        rank = constraint.spatial_rank or prior.spatial_rank or max(0, len(shape) - 2)
        flexible = tuple(range(2, 2 + rank))
    elif constraint is not None and constraint.kind in {"embedding", "matmul", "linear"}:
        flexible = prior.flexible_dims if prior.kind in {"embedding", "rnn", "transformer"} else ()
    elif constraint is None:
        flexible = prior.flexible_dims
    return InferInputShapeResult(
        found=True,
        shape=shape,
        shapes=None,
        dtype=example.dtype,
        value_range=prior.value_range,
        flexible_dims=flexible,
        constraining_module=constraint.module
        if constraint is not None
        else prior.constraining_module,
        constraining_op=constraint.label if constraint is not None else prior.constraining_op,
        source_line=constraint.source_line if constraint is not None else None,
        example_input=example,
        strategy=strategy,
        reason=None,
        attempts=tuple(attempts),
        trace=trace_obj if return_trace else None,
        message=f"Found valid input shape {shape} using executed op metadata.",
    )


def _maybe_normalize_success(
    model: nn.Module,
    prior: _InputPrior,
    example: torch.Tensor,
    trace_obj: Trace,
    attempts: list[tuple[tuple[int, ...] | None, str]],
    seed: int,
) -> tuple[torch.Tensor, Trace, str]:
    """Prefer the minimal executed linear input over a decoy-shaped success.

    Parameters
    ----------
    model:
        Model being inferred.
    prior:
        Prior that produced the success.
    example:
        Successful example.
    trace_obj:
        Successful trace for ``example``.
    attempts:
        Probe attempt list to append to.
    seed:
        RNG seed.

    Returns
    -------
    tuple[torch.Tensor, Trace, str]
        Possibly normalized example, trace, and strategy.
    """

    constraint = _executed_constraint(trace_obj)
    if (
        constraint is None
        or constraint.kind not in {"linear", "matmul"}
        or constraint.input_shape is None
        or len(constraint.input_shape) <= 2
        or constraint.in_features is None
    ):
        return example, trace_obj, prior.strategy

    shape = (constraint.input_shape[0], constraint.in_features)
    normalized = _make_tensor(shape, example.dtype, example.device, prior.value_range)
    probe = _probe(model, normalized, seed)
    attempts.append((shape, probe.outcome))
    if not probe.ok:
        return example, trace_obj, prior.strategy
    normalized_trace = _trace_model(model, normalized)
    return normalized, normalized_trace, "executed_op_normalize"


def _failure_result(
    reason: FailureReason,
    attempts: list[tuple[tuple[int, ...] | None, str]],
    message: str,
) -> InferInputShapeResult:
    """Build a standardized failed inference result.

    Parameters
    ----------
    reason:
        Failure reason.
    attempts:
        Probe attempts.
    message:
        User-facing message.

    Returns
    -------
    InferInputShapeResult
        Failed result.
    """

    return InferInputShapeResult(
        found=False,
        shape=None,
        shapes=None,
        dtype=None,
        value_range=None,
        flexible_dims=(),
        constraining_module=None,
        constraining_op=None,
        source_line=None,
        example_input=None,
        strategy="probe_success",
        reason=reason,
        attempts=tuple(attempts),
        trace=None,
        message=message,
    )


def _maybe_raise(result: InferInputShapeResult, on_failure: Literal["return", "raise"]) -> None:
    """Raise for a failed result when requested.

    Parameters
    ----------
    result:
        Inference result.
    on_failure:
        Failure behavior.
    """

    if not result.found and on_failure == "raise":
        raise ShapeInferenceError(result.message)


def infer_input_shape(
    model: nn.Module,
    *,
    batch_size: int = 1,
    input_dtype: torch.dtype | None = None,
    channels: int | None = None,
    spatial_rank: int | Literal["auto"] = "auto",
    seq_len: int | None = None,
    square: bool = True,
    min_size: int = 1,
    max_size: int = 512,
    preferred_sizes: Sequence[int] = (224, 256, 384, 299, 128, 96, 64, 32, 28),
    max_probes: int = 64,
    device: torch.device | str | None = None,
    seed: int = 0,
    return_trace: bool = False,
    on_failure: Literal["return", "raise"] = "return",
    input_specs: Any = None,
) -> InferInputShapeResult:
    """Infer a verified synthetic input shape for a PyTorch module.

    Parameters
    ----------
    model:
        PyTorch module to probe.
    batch_size:
        Batch dimension for synthesized inputs.
    input_dtype:
        Optional primary input dtype override.
    channels:
        Optional channel count override for convolutional inputs.
    spatial_rank:
        Spatial rank override, or ``"auto"`` from the first convolution.
    seq_len:
        Optional sequence length override for token or recurrent inputs.
    square:
        Whether spatial search should use equal side lengths. Non-square exact inference is
        intentionally not attempted without a future aspect-ratio hint.
    min_size:
        Minimum spatial side considered.
    max_size:
        Maximum spatial side considered.
    preferred_sizes:
        Spatial candidates to try before measured fallback search.
    max_probes:
        Maximum forward probes.
    device:
        Optional device override; defaults to the model's first parameter or buffer device.
    seed:
        RNG seed used for deterministic probes.
    return_trace:
        Whether to include the final verification ``Trace``.
    on_failure:
        ``"return"`` for notebook-friendly diagnostics or ``"raise"`` for
        ``ShapeInferenceError``.
    input_specs:
        Reserved for explicit multi-input specs. Coupled multi-input inference is currently
        unsupported unless the caller supplies a ready-made tensor/container in future work.

    Returns
    -------
    InferInputShapeResult
        Verified input, diagnostic failure, and probe attempts.

    Notes
    -----
    The implementation combines static module introspection, forward pre-hook probe diaries,
    successful ``tl.trace(..., inference_only=True)`` verification, and torch exception parsing.
    It measures candidate shapes instead of deriving convolution and pooling formulas.

    Limitations
    -----------
    Inferring valid inputs for arbitrary Python ``forward`` code is undecidable in general. This
    helper targets common MLP, CNN, adaptive-pool, ViT patch-embedding, transformer-LM, RNN, and
    1D/3D convolution cases. It cannot reliably infer coupled multi-tensor or dict inputs,
    value-dependent/data-dependent-control-flow shapes, non-tensor kwargs such as masks, labels,
    or ``past_key_values``, tokenizer/image-processor preprocessing, stateful or buffer-mutating
    forwards, opaque C++/CUDA errors, non-square exact spatial sizes without an aspect hint, or
    ``torch.compile``/scripted/exported artifacts.
    """

    if not isinstance(model, nn.Module):
        raise ShapeInferenceError("infer_input_shape expects a torch.nn.Module.")
    if input_specs is not None:
        result = _failure_result(
            "multi_input_unsupported",
            [],
            "input_specs multi-input inference is not implemented yet; pass a single-input module.",
        )
        _maybe_raise(result, on_failure)
        return result
    if not square:
        result = _failure_result(
            "exact_size_unreachable",
            [],
            "Non-square spatial inference requires an explicit aspect-ratio hint, which is not supported.",
        )
        _maybe_raise(result, on_failure)
        return result

    base_device, _base_dtype = _module_device_dtype(model)
    resolved_device = torch.device(device) if device is not None else base_device
    priors = _input_priors(
        model,
        batch_size,
        input_dtype,
        channels,
        spatial_rank,
        seq_len,
        min_size,
        preferred_sizes,
        resolved_device,
    )
    attempts: list[tuple[tuple[int, ...] | None, str]] = []
    if not priors:
        result = _failure_result(
            "unknown_entry",
            attempts,
            "No supported executed-op seed was found; identity-like models are not inferred.",
        )
        _maybe_raise(result, on_failure)
        return result

    probes = 0
    delayed_blockers: list[str] = []
    for prior in priors:
        if probes >= max_probes:
            break
        example = _make_tensor(prior.shape, prior.dtype, prior.device, prior.value_range)
        probe = _probe(model, example, seed)
        probes += 1
        attempts.append((prior.shape, probe.outcome))
        if probe.ok:
            trace_obj = _trace_model(model, example)
            final_example, final_trace, strategy = _maybe_normalize_success(
                model, prior, example, trace_obj, attempts, seed
            )
            shape = tuple(int(dim) for dim in final_example.shape)
            return _verified_result(
                prior, shape, final_example, attempts, final_trace, return_trace, strategy
            )

        lower_outcome = probe.outcome.lower()
        suggested_dtype = _dtype_from_message(probe.outcome)
        if suggested_dtype is not None and suggested_dtype != prior.dtype and probes < max_probes:
            value_range = prior.value_range
            if suggested_dtype == torch.long and value_range[0] != "randint":
                value_range = ("randint", 0.0, 2.0)
            fixed = _make_tensor(prior.shape, suggested_dtype, prior.device, value_range)
            second = _probe(model, fixed, seed)
            probes += 1
            attempts.append((prior.shape, second.outcome))
            if second.ok:
                trace_obj = _trace_model(model, fixed)
                dtype_prior = _InputPrior(
                    kind=prior.kind,
                    shape=prior.shape,
                    dtype=suggested_dtype,
                    value_range=value_range,
                    flexible_dims=prior.flexible_dims,
                    constraining_module=prior.constraining_module,
                    constraining_op=prior.constraining_op,
                    strategy="dtype_corrected",
                    device=prior.device,
                    spatial_rank=prior.spatial_rank,
                    channels=prior.channels,
                    min_side=prior.min_side,
                )
                return _verified_result(
                    dtype_prior,
                    prior.shape,
                    fixed,
                    attempts,
                    trace_obj,
                    return_trace,
                    "dtype_corrected",
                )

        if prior.kind == "linear" and probe.target_features is not None and probes < max_probes:
            shape = (batch_size, probe.target_features)
            fixed = _make_tensor(shape, prior.dtype, prior.device, prior.value_range)
            second = _probe(model, fixed, seed)
            probes += 1
            attempts.append((shape, second.outcome))
            if second.ok:
                trace_obj = _trace_model(model, fixed)
                corrected = _InputPrior(
                    kind=prior.kind,
                    shape=shape,
                    dtype=prior.dtype,
                    value_range=prior.value_range,
                    flexible_dims=prior.flexible_dims,
                    constraining_module=second.constraining_module or prior.constraining_module,
                    constraining_op=prior.constraining_op,
                    strategy="executed_op_linear",
                    device=prior.device,
                )
                return _verified_result(
                    corrected, shape, fixed, attempts, trace_obj, return_trace, "executed_op_linear"
                )

        if prior.kind != "conv":
            if not _is_skippable_shape_error(probe.outcome):
                delayed_blockers.append(probe.outcome)
            continue

        rank = prior.spatial_rank or max(1, len(prior.shape) - 2)
        lower_bound = max(min_size, prior.min_side, 1)
        measured: list[tuple[int, int]] = []
        target = probe.target_features
        if probe.got_features is not None and probe.target_features is not None:
            measured.append((prior.shape[-1], probe.got_features))
        sides = _candidate_sides(prior.shape[-1], lower_bound, max_size, preferred_sizes)
        for side in sides:
            if probes >= max_probes:
                break
            if side == prior.shape[-1]:
                continue
            shape = (batch_size, prior.channels or prior.shape[1], *([side] * rank))
            example = _make_tensor(shape, prior.dtype, prior.device, prior.value_range)
            side_probe = _probe(model, example, seed)
            probes += 1
            attempts.append((shape, side_probe.outcome))
            if side_probe.ok:
                trace_obj = _trace_model(model, example)
                final_example, final_trace, strategy = _maybe_normalize_success(
                    model, prior, example, trace_obj, attempts, seed
                )
                final_shape = tuple(int(dim) for dim in final_example.shape)
                return _verified_result(
                    prior,
                    final_shape,
                    final_example,
                    attempts,
                    final_trace,
                    return_trace,
                    strategy if strategy != prior.strategy else "probe_success",
                )
            if side_probe.got_features is not None and side_probe.target_features is not None:
                measured.append((side, side_probe.got_features))
                target = side_probe.target_features
            elif not _is_skippable_shape_error(side_probe.outcome):
                delayed_blockers.append(side_probe.outcome)
                break

        if measured and target is not None and probes < max_probes:
            low = lower_bound
            high = max_size
            while low <= high and probes < max_probes:
                side = (low + high) // 2
                shape = (batch_size, prior.channels or prior.shape[1], *([side] * rank))
                example = _make_tensor(shape, prior.dtype, prior.device, prior.value_range)
                search_probe = _probe(model, example, seed)
                probes += 1
                attempts.append((shape, search_probe.outcome))
                if search_probe.ok:
                    trace_obj = _trace_model(model, example)
                    final_example, final_trace, strategy = _maybe_normalize_success(
                        model, prior, example, trace_obj, attempts, seed
                    )
                    final_shape = tuple(int(dim) for dim in final_example.shape)
                    return _verified_result(
                        prior,
                        final_shape,
                        final_example,
                        attempts,
                        final_trace,
                        return_trace,
                        strategy if strategy != prior.strategy else "binary_search",
                    )
                if search_probe.got_features is None:
                    if _is_skippable_shape_error(search_probe.outcome):
                        low = side + 1
                        continue
                    delayed_blockers.append(search_probe.outcome)
                    break
                if search_probe.got_features < target:
                    low = side + 1
                else:
                    high = side - 1
            for side in range(max(lower_bound, low - 4), min(max_size, low + 4) + 1):
                if probes >= max_probes:
                    break
                shape = (batch_size, prior.channels or prior.shape[1], *([side] * rank))
                example = _make_tensor(shape, prior.dtype, prior.device, prior.value_range)
                near_probe = _probe(model, example, seed)
                probes += 1
                attempts.append((shape, near_probe.outcome))
                if near_probe.ok:
                    trace_obj = _trace_model(model, example)
                    final_example, final_trace, strategy = _maybe_normalize_success(
                        model, prior, example, trace_obj, attempts, seed
                    )
                    final_shape = tuple(int(dim) for dim in final_example.shape)
                    return _verified_result(
                        prior,
                        final_shape,
                        final_example,
                        attempts,
                        final_trace,
                        return_trace,
                        strategy if strategy != prior.strategy else "binary_search",
                    )

        if lower_outcome and not _is_skippable_shape_error(probe.outcome):
            delayed_blockers.append(probe.outcome)

    if probes >= max_probes:
        result = _failure_result(
            "budget_exhausted", attempts, f"No valid shape was found within {max_probes} probes."
        )
        _maybe_raise(result, on_failure)
        return result
    if delayed_blockers:
        result = _failure_result(
            "non_shape_blocker",
            attempts,
            f"Shape inference was blocked by an unsupported forward error: {delayed_blockers[-1]}",
        )
        _maybe_raise(result, on_failure)
        return result
    result = _failure_result(
        "exact_size_unreachable",
        attempts,
        "No square valid input was found; pass an explicit size/aspect hint for rectangular cases.",
    )
    _maybe_raise(result, on_failure)
    return result
