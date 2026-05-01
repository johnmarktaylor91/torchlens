"""One-shot public fastlog recording API."""

from __future__ import annotations

from typing import Any, cast

from torch import nn

from .._deprecations import MISSING, MissingType, warn_deprecated_alias
from .._training_validation import reject_compiled_model
from ..options import StreamingOptions
from ..types import ActivationPostfunc
from ._recorder import Recorder
from ._validation import validate_postprocess
from .options import PredicateErrorMode, PredicateFn
from .types import CaptureSpec, Recording


def record(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None = None,
    *,
    keep_op: PredicateFn | None = None,
    keep_module: PredicateFn | None = None,
    default_op: bool | CaptureSpec | MissingType = MISSING,
    default_module: bool | CaptureSpec | MissingType = MISSING,
    history_size: int = 8,
    include_source_events: bool = False,
    max_predicate_failures: int = 32,
    on_predicate_error: PredicateErrorMode = "auto",
    streaming: StreamingOptions | None = None,
    return_output: bool = False,
    postprocess: str = "none",
    random_seed: int | None = None,
    activation_transform: ActivationPostfunc | None = None,
    save_raw_activation: bool = True,
    train_mode: bool = False,
    activation_postfunc: ActivationPostfunc | None | MissingType = MISSING,
) -> Recording | tuple[Any, Recording]:
    """Record one model forward pass with fastlog predicates.

    Parameters
    ----------
    model:
        PyTorch module to execute.
    input_args:
        Tensor, list, or tuple of positional model inputs.
    input_kwargs:
        Optional keyword arguments for the model call.
    keep_op, keep_module, default_op, default_module, history_size,
    include_source_events, max_predicate_failures, on_predicate_error, streaming,
    random_seed:
        Fastlog recording options.
    activation_transform:
        Optional callable applied to each retained activation copy after
        dtype/device transforms. Errors propagate as
        :class:`torchlens.TorchLensPostfuncError`.
    save_raw_activation:
        When ``False`` and ``activation_transform`` is set, only transformed
        payloads are retained. Defaults to ``True`` to mirror the slow path.
    train_mode:
        If True, omitted defaults are promoted to keep-grad capture specs.
    return_output:
        Whether to return ``(model_output, recording)``.
    postprocess:
        Optional postprocess enrichment preset.

    Returns
    -------
    Recording | tuple[Any, Recording]
        Fastlog recording, optionally with the model output.
    """

    reject_compiled_model(model, api_name="torchlens.fastlog.record")
    validate_postprocess(postprocess)
    if activation_postfunc is not MISSING:
        if activation_transform is not None:
            raise TypeError(
                "kwarg activation_postfunc deprecated, use activation_transform; do not pass both"
            )
        warn_deprecated_alias("activation_postfunc", "activation_transform")
        activation_transform = cast(ActivationPostfunc | None, activation_postfunc)
    with Recorder(
        model,
        keep_op=keep_op,
        keep_module=keep_module,
        default_op=default_op,
        default_module=default_module,
        history_size=history_size,
        include_source_events=include_source_events,
        max_predicate_failures=max_predicate_failures,
        on_predicate_error=on_predicate_error,
        streaming=streaming,
        random_seed=random_seed,
        activation_transform=activation_transform,
        save_raw_activation=save_raw_activation,
        train_mode=train_mode,
    ) as recorder:
        output = recorder.log(input_args, input_kwargs)
    recording = recorder.recording
    if postprocess != "none":
        recording = recording.enrich(postprocess)
    if return_output:
        return output, recording
    return recording
