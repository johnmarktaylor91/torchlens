"""Input-shape inference helpers for TorchLens debug utilities."""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from torchlens._errors import ShapeInferenceError

if TYPE_CHECKING:
    from torchlens.data_classes.trace import Trace

FailureReason = Literal[
    "non_shape_blocker",
    "exact_size_unreachable",
    "multi_input_unsupported",
    "budget_exhausted",
    "unknown_entry",
    "lazy_uninitialized",
    "verification_failed",
    "device_mismatch",
    "rank_undetermined",
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
    "device": re.compile(
        r"(expected device|same device|two devices|is not on the expected device"
        r"|Input type \([^)]*\) and weight type \([^)]*\))",
        re.IGNORECASE,
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
_DEVICE_RE = _EXCEPTION_PARSE_PATTERNS["device"]


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

    if _DEVICE_RE.search(message):
        return False
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


def _conv_spatial_rank(module: nn.Module) -> int | None:
    """Resolve the spatial rank of a conv-like module without exact-type lookups.

    Parameters
    ----------
    module:
        Convolution module, possibly a subclass such as ``LazyConv2d`` or a
        third-party ``Conv2d`` variant.

    Returns
    -------
    int | None
        Spatial rank, or ``None`` when the rank cannot be resolved.
    """

    kernel = getattr(module, "kernel_size", None)
    if isinstance(kernel, (tuple, list)) and len(kernel) > 0:
        return int(len(kernel))
    if isinstance(kernel, int):
        return 1
    for conv_class, rank in ((nn.Conv3d, 3), (nn.Conv2d, 2), (nn.Conv1d, 1)):
        if isinstance(module, conv_class):
            return rank
    return None


def _has_uninitialized_lazy_state(model: nn.Module) -> bool:
    """Return whether the model still carries un-materialized lazy state.

    Probing a lazy module would permanently materialize it inside the caller's
    model (often at a degenerate width such as ``in_features=0``), so inference
    must refuse before building any prior.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    bool
        Whether any module has uninitialized lazy parameters or buffers.
    """

    for module in model.modules():
        if isinstance(module, nn.modules.lazy.LazyModuleMixin):
            try:
                if module.has_uninitialized_params():
                    return True
            except Exception:  # noqa: BLE001 - treat unreadable lazy state as uninitialized.
                return True
    uninitialized = (nn.parameter.UninitializedParameter, nn.parameter.UninitializedBuffer)
    return any(
        isinstance(tensor, uninitialized)
        for tensor in list(model.parameters(recurse=True)) + list(model.buffers(recurse=True))
    )


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


def _transformer_prior(
    name: str,
    module: nn.TransformerEncoderLayer,
    batch_size: int,
    seq_len: int | None,
    dtype: torch.dtype,
    device: torch.device,
) -> _InputPrior:
    """Build a layout-honest prior for a transformer encoder layer.

    The layer's attention module decides whether the batch dimension leads
    (``batch_first=True``) or the sequence dimension leads; ignoring that
    silently transposes batch and sequence in the reported shape.

    Parameters
    ----------
    name:
        Module qualname.
    module:
        Transformer encoder layer.
    batch_size:
        Requested batch size.
    seq_len:
        Optional sequence length override.
    dtype:
        Input dtype.
    device:
        Target device.

    Returns
    -------
    _InputPrior
        Transformer input prior with the correct batch/sequence layout.
    """

    size = seq_len or 16
    batch_first = bool(getattr(module.self_attn, "batch_first", False))
    embed_dim = int(module.self_attn.embed_dim)
    shape = (batch_size, size, embed_dim) if batch_first else (size, batch_size, embed_dim)
    return _InputPrior(
        kind="transformer",
        shape=shape,
        dtype=dtype,
        value_range=("uniform", 0.0, 1.0),
        flexible_dims=(1,) if batch_first else (0,),
        constraining_module=name,
        constraining_op=module.__class__.__name__,
        strategy="introspection",
        device=device,
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
        resolved_rank = int(spatial_rank) if spatial_rank != "auto" else _conv_spatial_rank(module)
        if resolved_rank is not None:
            rank = resolved_rank
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

    transformer = _first_module(model, (nn.TransformerEncoderLayer,))
    if transformer is not None:
        name, module = transformer
        assert isinstance(module, nn.TransformerEncoderLayer)
        return _transformer_prior(
            name,
            module,
            batch_size,
            seq_len,
            input_dtype or _float_dtype(model_dtype),
            device,
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
            resolved_rank = (
                int(spatial_rank) if spatial_rank != "auto" else _conv_spatial_rank(module)
            )
            if resolved_rank is None:
                continue
            rank = resolved_rank
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
            priors.append(
                _transformer_prior(
                    name,
                    module,
                    batch_size,
                    seq_len,
                    input_dtype or _float_dtype(model_dtype),
                    device,
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
            if tensor.ndim >= 3:
                return int(tensor.shape[1])
            if int(tensor.shape[0]) == 1:
                # HF-style ``position_ids`` buffers are laid out as (1, max_len).
                return int(tensor.shape[1])
            return int(tensor.shape[0])
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
    """Run a final TorchLens inference-only trace without mutating model state.

    The verification trace runs under ``eval()`` with the training flags saved
    and restored so it uses the same regime as :func:`_probe`; a train-mode
    trace would silently update BatchNorm running statistics inside the
    caller's model.

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

    from torchlens.options import CaptureOptions
    from torchlens.user_funcs import trace

    states = _training_states(model)
    try:
        model.eval()
        return trace(model, example_input, capture=CaptureOptions(inference_only=True))
    finally:
        _restore_training_states(states)


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


def _is_rank_inflated(constraint: _ExecutedConstraint | None, example: torch.Tensor) -> bool:
    """Return whether a success only ran by broadcasting the input to a higher rank.

    A probe that "works" because the model broadcast a low-rank input into a
    fabricated higher-rank view (for example a 2D tensor repeated into a 1xN
    "image") verifies a different computation than intended. The tell is that
    the first executed input-consuming constraint saw a HIGHER-rank tensor with
    MORE non-singleton dimensions than the probe input itself; rank-preserving
    reshapes such as ``unsqueeze`` only add singleton dimensions and stay legal.

    Parameters
    ----------
    constraint:
        First executed constraint from the verification trace.
    example:
        Probe input that ran successfully.

    Returns
    -------
    bool
        Whether the success is a broadcast/rank artifact.
    """

    if constraint is None or constraint.input_shape is None:
        return False
    if constraint.kind not in {"conv", "linear", "matmul"}:
        return False
    if len(constraint.input_shape) <= int(example.ndim):
        return False
    example_wide = sum(1 for dim in example.shape if int(dim) > 1)
    constraint_wide = sum(1 for dim in constraint.input_shape if int(dim) > 1)
    return constraint_wide > example_wide


def _resolved_flexible_dims(
    prior: _InputPrior,
    constraint: _ExecutedConstraint | None,
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Resolve the claimed flexible dimensions for a verified success.

    Parameters
    ----------
    prior:
        Winning prior.
    constraint:
        First executed constraint from the verification trace.
    shape:
        Final verified shape.

    Returns
    -------
    tuple[int, ...]
        Claimed flexible dimensions, bounded to the final shape's rank.
    """

    flexible: tuple[int, ...] = ()
    if constraint is not None and constraint.kind == "conv" and constraint.flexible:
        rank = constraint.spatial_rank or prior.spatial_rank or max(0, len(shape) - 2)
        flexible = tuple(range(2, 2 + rank))
    elif constraint is not None and constraint.kind in {"embedding", "matmul", "linear"}:
        flexible = prior.flexible_dims if prior.kind in {"embedding", "rnn", "transformer"} else ()
    elif constraint is None:
        flexible = prior.flexible_dims
    return tuple(dim for dim in flexible if 0 <= dim < len(shape))


def _resolved_value_range(
    prior: _InputPrior,
    constraint: _ExecutedConstraint | None,
) -> tuple[str, float, float]:
    """Resolve the reported value recipe from executed-op evidence.

    Token inputs corrected from a generic prior carry a placeholder
    ``("randint", 0, 2)`` recipe; when the executed embedding op exposes the
    real vocabulary size, report that instead so callers regenerating inputs
    exercise the whole table.

    Parameters
    ----------
    prior:
        Winning prior.
    constraint:
        First executed constraint from the verification trace.

    Returns
    -------
    tuple[str, float, float]
        Honest synthetic value recipe.
    """

    if (
        constraint is not None
        and constraint.kind == "embedding"
        and constraint.in_features is not None
        and int(constraint.in_features) >= 1
        and prior.value_range[0] == "randint"
    ):
        return ("randint", 0.0, float(constraint.in_features))
    return prior.value_range


def _verified_result(
    prior: _InputPrior,
    constraint: _ExecutedConstraint | None,
    shape: tuple[int, ...],
    example: torch.Tensor,
    attempts: list[tuple[tuple[int, ...] | None, str]],
    trace_obj: Trace,
    return_trace: bool,
    strategy: str,
    flexible: tuple[int, ...],
    value_range: tuple[str, float, float],
    message_note: str = "",
) -> InferInputShapeResult:
    """Build a successful result from a verified trace.

    Parameters
    ----------
    prior:
        Candidate prior that produced the input.
    constraint:
        First executed constraint from the verification trace.
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
    flexible:
        Probe-verified flexible dimensions.
    value_range:
        Honest synthetic value recipe.
    message_note:
        Optional honesty caveat appended to the success message.

    Returns
    -------
    InferInputShapeResult
        Successful result.
    """

    return InferInputShapeResult(
        found=True,
        shape=shape,
        shapes=None,
        dtype=example.dtype,
        value_range=value_range,
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
        message=f"Found valid input shape {shape} (strategy: {strategy}).{message_note}",
    )


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
        strategy="none",
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


def _validate_search_arguments(
    *,
    batch_size: int,
    channels: int | None,
    spatial_rank: int | Literal["auto"],
    seq_len: int | None,
    min_size: int,
    max_size: int,
    max_probes: int,
) -> None:
    """Validate caller-supplied search arguments with typed errors.

    Invalid arguments are caller bugs, so they raise :class:`ShapeInferenceError`
    regardless of ``on_failure`` instead of producing degenerate probes such as
    a ``(0, N)`` "success" or a raw negative-dimension ``RuntimeError``.

    Parameters
    ----------
    batch_size:
        Requested batch size.
    channels:
        Optional channel override.
    spatial_rank:
        Spatial rank override or ``"auto"``.
    seq_len:
        Optional sequence length override.
    min_size:
        Minimum spatial side.
    max_size:
        Maximum spatial side.
    max_probes:
        Probe budget.
    """

    if batch_size < 1:
        raise ShapeInferenceError(f"batch_size must be >= 1, got {batch_size}.")
    if channels is not None and channels < 1:
        raise ShapeInferenceError(f"channels must be >= 1 when provided, got {channels}.")
    if spatial_rank != "auto" and int(spatial_rank) < 1:
        raise ShapeInferenceError(f"spatial_rank must be 'auto' or >= 1, got {spatial_rank!r}.")
    if seq_len is not None and seq_len < 1:
        raise ShapeInferenceError(f"seq_len must be >= 1 when provided, got {seq_len}.")
    if min_size < 1:
        raise ShapeInferenceError(f"min_size must be >= 1, got {min_size}.")
    if max_size < min_size:
        raise ShapeInferenceError(f"max_size ({max_size}) must be >= min_size ({min_size}).")
    if max_probes < 1:
        raise ShapeInferenceError(f"max_probes must be >= 1, got {max_probes}.")


def _run_search(
    model: nn.Module,
    attempts: list[tuple[tuple[int, ...] | None, str]],
    *,
    batch_size: int,
    input_dtype: torch.dtype | None,
    channels: int | None,
    spatial_rank: int | Literal["auto"],
    seq_len: int | None,
    min_size: int,
    max_size: int,
    preferred_sizes: Sequence[int],
    max_probes: int,
    device: torch.device | str | None,
    seed: int,
    return_trace: bool,
) -> InferInputShapeResult:
    """Run the probe/verify search loop and always return a structured result.

    This function never raises intentionally; the caller wraps it in the
    ``on_failure`` safety net. Every successful probe is verified by a trace
    through one shared finalizer that (a) converts trace errors into typed
    ``verification_failed`` failures, (b) rejects broadcast/rank-inflated
    "successes", (c) gates decoy normalization away from sequence-structural
    priors, (d) probe-verifies claimed flexible dimensions, and (e) reports the
    executed embedding vocabulary in ``value_range``.

    Parameters
    ----------
    model:
        Model to probe.
    attempts:
        Shared probe diary, mutated in place so partial attempts survive
        internal errors.
    batch_size, input_dtype, channels, spatial_rank, seq_len, min_size, max_size, preferred_sizes, max_probes, device, seed, return_trace:
        See :func:`infer_input_shape`.

    Returns
    -------
    InferInputShapeResult
        Success or structured failure.
    """

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
    if not priors:
        return _failure_result(
            "unknown_entry",
            attempts,
            "No supported executed-op seed was found; identity-like models are not inferred.",
        )

    probes = 0
    delayed_blockers: list[str] = []
    device_blockers: list[str] = []
    rank_rejections: list[str] = []

    def record_blocker(message: str) -> None:
        """Classify a non-shape blocker as device-specific or generic."""

        if _DEVICE_RE.search(message):
            device_blockers.append(message)
        else:
            delayed_blockers.append(message)

    def finalize(
        prior: _InputPrior,
        example: torch.Tensor,
        fallback_strategy: str | None,
        allow_normalize: bool,
    ) -> InferInputShapeResult | None:
        """Verify one successful probe with a trace and build an honest result.

        Returns ``None`` when the success is rejected as a broadcast/rank
        artifact so the caller continues the search. Returns a terminal result
        otherwise, including a typed ``verification_failed`` failure when
        ``tl.trace`` rejects an input whose plain forward ran.
        """

        nonlocal probes
        shape = tuple(int(dim) for dim in example.shape)
        try:
            trace_obj = _trace_model(model, example)
        except Exception as exc:  # noqa: BLE001 - honor the on_failure contract for any trace error.
            return _failure_result(
                "verification_failed",
                attempts,
                f"Forward probes succeeded with shape {shape}, but the TorchLens "
                f"verification trace failed: {exc}",
            )
        constraint = _executed_constraint(trace_obj)
        if _is_rank_inflated(constraint, example):
            assert constraint is not None and constraint.input_shape is not None
            attempts.append(
                (
                    shape,
                    "ok_but_rank_inflated: ran only after in-model broadcasting to rank "
                    f"{len(constraint.input_shape)}",
                )
            )
            rank_rejections.append(str(shape))
            return None
        strategy = fallback_strategy or prior.strategy
        if (
            allow_normalize
            and prior.kind not in {"rnn", "transformer", "embedding"}
            and constraint is not None
            and constraint.kind in {"linear", "matmul"}
            and constraint.input_shape is not None
            and len(constraint.input_shape) > 2
            and constraint.in_features is not None
            and probes < max_probes
        ):
            normalized_shape = (int(constraint.input_shape[0]), int(constraint.in_features))
            normalized = _make_tensor(
                normalized_shape, example.dtype, example.device, prior.value_range
            )
            norm_probe = _probe(model, normalized, seed)
            probes += 1
            attempts.append((normalized_shape, norm_probe.outcome))
            if norm_probe.ok:
                normalized_trace: Trace | None
                try:
                    normalized_trace = _trace_model(model, normalized)
                except Exception:  # noqa: BLE001 - keep the already-verified original input.
                    normalized_trace = None
                if normalized_trace is not None:
                    normalized_constraint = _executed_constraint(normalized_trace)
                    if not _is_rank_inflated(normalized_constraint, normalized):
                        example = normalized
                        trace_obj = normalized_trace
                        constraint = normalized_constraint
                        shape = normalized_shape
                        strategy = "executed_op_normalize"
        value_range = _resolved_value_range(prior, constraint)
        flexible = _resolved_flexible_dims(prior, constraint, shape)
        if flexible and probes < max_probes:
            flexible_set = set(flexible)
            grown_shape = tuple(
                dim + 4 if index in flexible_set else dim for index, dim in enumerate(shape)
            )
            grown = _make_tensor(grown_shape, example.dtype, example.device, value_range)
            grown_probe = _probe(model, grown, seed)
            probes += 1
            attempts.append((grown_shape, grown_probe.outcome))
            if not grown_probe.ok:
                flexible = ()
        message_note = ""
        if (
            len(shape) == 2
            and example.dtype.is_floating_point
            and (constraint is None or constraint.kind in {"linear", "matmul"})
        ):
            message_note = (
                " Note: only the trailing feature dimension is pinned by the constraining"
                " op, so the input rank may be under-determined; higher-rank"
                f" (batch, ..., {shape[-1]}) inputs may also be valid."
            )
        return _verified_result(
            prior,
            constraint,
            shape,
            example,
            attempts,
            trace_obj,
            return_trace,
            strategy,
            flexible,
            value_range,
            message_note,
        )

    for prior in priors:
        if probes >= max_probes:
            break
        example = _make_tensor(prior.shape, prior.dtype, prior.device, prior.value_range)
        probe = _probe(model, example, seed)
        probes += 1
        attempts.append((prior.shape, probe.outcome))
        if probe.ok:
            outcome = finalize(prior, example, None, allow_normalize=True)
            if outcome is not None:
                return outcome
            continue

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
                dtype_prior = replace(
                    prior,
                    dtype=suggested_dtype,
                    value_range=value_range,
                    strategy="dtype_corrected",
                )
                outcome = finalize(dtype_prior, fixed, "dtype_corrected", allow_normalize=False)
                if outcome is not None:
                    return outcome
                continue

        if prior.kind == "linear" and probe.target_features is not None and probes < max_probes:
            shape: tuple[int, ...] = (batch_size, probe.target_features)
            fixed = _make_tensor(shape, prior.dtype, prior.device, prior.value_range)
            second = _probe(model, fixed, seed)
            probes += 1
            attempts.append((shape, second.outcome))
            if second.ok:
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
                outcome = finalize(corrected, fixed, "executed_op_linear", allow_normalize=False)
                if outcome is not None:
                    return outcome
                continue

        if prior.kind != "conv":
            if not _is_skippable_shape_error(probe.outcome):
                record_blocker(probe.outcome)
            continue

        if channels is None and probes < max_probes:
            channel_match = _CHANNEL_RE.search(probe.outcome)
            if channel_match is not None:
                expected_channels = int(channel_match.group(1))
                got_channels = int(channel_match.group(2))
                base_channels = int(prior.channels or prior.shape[1])
                if got_channels > 0 and (base_channels * expected_channels) % got_channels == 0:
                    corrected_channels = (base_channels * expected_channels) // got_channels
                    if corrected_channels >= 1 and corrected_channels != base_channels:
                        prior = replace(
                            prior,
                            shape=(prior.shape[0], corrected_channels, *prior.shape[2:]),
                            channels=corrected_channels,
                            strategy="channel_corrected",
                        )
                        example = _make_tensor(
                            prior.shape, prior.dtype, prior.device, prior.value_range
                        )
                        probe = _probe(model, example, seed)
                        probes += 1
                        attempts.append((prior.shape, probe.outcome))
                        if probe.ok:
                            outcome = finalize(
                                prior, example, "channel_corrected", allow_normalize=True
                            )
                            if outcome is not None:
                                return outcome
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
                outcome = finalize(prior, example, "probe_success", allow_normalize=True)
                if outcome is not None:
                    return outcome
                continue
            if side_probe.got_features is not None and side_probe.target_features is not None:
                measured.append((side, side_probe.got_features))
                target = side_probe.target_features
            elif not _is_skippable_shape_error(side_probe.outcome):
                record_blocker(side_probe.outcome)
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
                    outcome = finalize(prior, example, "binary_search", allow_normalize=True)
                    if outcome is not None:
                        return outcome
                    break
                if search_probe.got_features is None:
                    if _is_skippable_shape_error(search_probe.outcome):
                        low = side + 1
                        continue
                    record_blocker(search_probe.outcome)
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
                    outcome = finalize(prior, example, "binary_search", allow_normalize=True)
                    if outcome is not None:
                        return outcome
                    continue

        if lower_outcome and not _is_skippable_shape_error(probe.outcome):
            record_blocker(probe.outcome)

    if probes >= max_probes:
        return _failure_result(
            "budget_exhausted", attempts, f"No valid shape was found within {max_probes} probes."
        )
    if device_blockers:
        return _failure_result(
            "device_mismatch",
            attempts,
            "Shape inference was blocked by a device mismatch, not an input-shape problem: "
            f"{device_blockers[-1]} Move the model to a single device or pass a matching "
            "device= argument.",
        )
    if rank_rejections:
        return _failure_result(
            "rank_undetermined",
            attempts,
            "Probes ran only after in-model broadcasting of a lower-rank input (rejected: "
            f"{', '.join(rank_rejections)}); the true input rank is under-determined. Pass "
            "seq_len= or explicit channels/spatial_rank hints.",
        )
    if delayed_blockers:
        return _failure_result(
            "non_shape_blocker",
            attempts,
            f"Shape inference was blocked by an unsupported forward error: {delayed_blockers[-1]}",
        )
    return _failure_result(
        "exact_size_unreachable",
        attempts,
        "No square valid input was found; the model may need a rectangular (non-square) input, "
        "whose exact aspect ratio is not searched, or sides outside min_size/max_size.",
    )


def _infer_input_shape_impl(
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
        ``ShapeInferenceError``. With ``"return"``, inference failures of any kind
        (including trace-verification and internal errors) come back as a structured
        result instead of raising; only invalid arguments raise regardless.
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

    The helper is read-only with respect to the caller's model: probes and verification
    traces run under ``eval()`` with training flags saved and restored, models with
    un-materialized lazy modules are refused (``lazy_uninitialized``) instead of being
    silently materialized, and successes that only ran through in-model broadcasting of a
    lower-rank input are rejected (``rank_undetermined``) rather than reported as verified.

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
    _validate_search_arguments(
        batch_size=batch_size,
        channels=channels,
        spatial_rank=spatial_rank,
        seq_len=seq_len,
        min_size=min_size,
        max_size=max_size,
        max_probes=max_probes,
    )
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

    if _has_uninitialized_lazy_state(model):
        result = _failure_result(
            "lazy_uninitialized",
            [],
            "The model contains un-materialized lazy modules (for example LazyLinear or "
            "LazyConv2d) whose input widths are undefined until a real forward pass runs. "
            "Run one real forward pass to materialize them, then retry; the model was left "
            "untouched.",
        )
        _maybe_raise(result, on_failure)
        return result

    attempts: list[tuple[tuple[int, ...] | None, str]] = []
    training_states = _training_states(model)
    try:
        result = _run_search(
            model,
            attempts,
            batch_size=batch_size,
            input_dtype=input_dtype,
            channels=channels,
            spatial_rank=spatial_rank,
            seq_len=seq_len,
            min_size=min_size,
            max_size=max_size,
            preferred_sizes=preferred_sizes,
            max_probes=max_probes,
            device=device,
            seed=seed,
            return_trace=return_trace,
        )
    except Exception as exc:  # noqa: BLE001 - the on_failure contract admits no raw escape.
        message = f"Shape inference aborted on an unexpected internal error: {exc!r}"
        if on_failure == "raise":
            raise ShapeInferenceError(message) from exc
        return _failure_result("non_shape_blocker", attempts, message)
    finally:
        _restore_training_states(training_states)
    _maybe_raise(result, on_failure)
    return result


@functools.wraps(_infer_input_shape_impl)
def infer_input_shape(model: nn.Module, **kwargs: Any) -> InferInputShapeResult:
    """Infer a synthetic input shape while leaving the caller's global RNG untouched.

    ``infer_input_shape`` synthesizes probe tensors (``torch.rand``/``randn``/
    ``randint``) and seeds the probe with ``torch.manual_seed(seed)``. Both mutate
    the process-global torch RNG, so a bare call permanently advanced the caller's
    RNG stream. Snapshot the CPU (and CUDA) RNG state on entry and restore it on
    exit so this diagnostic is RNG-neutral. Model state is likewise left untouched:
    training flags are saved and restored around the whole call, probes and traces
    run in ``eval()`` mode, and lazy modules are never materialized. All
    keyword-only arguments and the full signature/docstring of
    :func:`_infer_input_shape_impl` are preserved via :func:`functools.wraps`.
    """

    cpu_rng_state = torch.get_rng_state()
    # Initialized-CUDA-only, latch-guarded (R36-4): the bare is_available()
    # gate allocated a CUDA context per visible device on CPU-only runs.
    from ..utils.rng import _snapshot_cuda_rng_states

    cuda_rng_state = _snapshot_cuda_rng_states() or None
    try:
        return _infer_input_shape_impl(model, **kwargs)
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


# Preserve the honest public identity that ``functools.wraps`` copies from the impl.
infer_input_shape.__name__ = "infer_input_shape"
infer_input_shape.__qualname__ = "infer_input_shape"
