"""One-shot public fastlog recording API."""

from __future__ import annotations

from typing import Any, cast
import warnings

from torch import nn

from .._deprecations import MISSING, MissingType
from .._input_coerce import _coerce_input_args
from .._training_validation import reject_compiled_model
from ..backends import BackendName, BackendUnsupportedError
from ..intervention.predicates import InterventionPredicate
from ..options import StreamingOptions
from ..types import ActivationPostfunc, GradientPostfunc
from ._recorder import Recorder
from ._validation import validate_postprocess
from .options import (
    GradPredicateFn,
    HaltPredicateFn,
    LookbackPayloadPolicy,
    PredicateErrorMode,
    PredicateFn,
)
from .types import CaptureSpec, Recording


def _resolve_save_alias(
    *,
    save: PredicateFn | None | MissingType,
    keep_op: PredicateFn | None | MissingType,
) -> PredicateFn | None:
    """Resolve ``record(save=...)`` and deprecated ``keep_op=...`` spelling."""

    save_supplied = not isinstance(save, MissingType)
    keep_op_supplied = not isinstance(keep_op, MissingType)
    if save_supplied and keep_op_supplied:
        raise ValueError("record() received both save= and deprecated keep_op=.")
    if keep_op_supplied:
        warnings.warn(
            "record(keep_op=...) is deprecated; use record(save=...) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return cast("PredicateFn | None", keep_op)
    if not save_supplied:
        return None
    return cast("PredicateFn | None", save)


def record(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None = None,
    *,
    save: PredicateFn | None | MissingType = MISSING,
    keep_op: PredicateFn | None | MissingType = MISSING,
    keep_module: PredicateFn | None | MissingType = MISSING,
    default_op: bool | CaptureSpec | MissingType = MISSING,
    default_module: bool | CaptureSpec | MissingType = MISSING,
    history_size: int = 8,
    lookback: int = 0,
    lookback_payload_policy: LookbackPayloadPolicy = "metadata_only",
    include_source_events: bool = False,
    intervene: InterventionPredicate | None = None,
    halt: HaltPredicateFn | None = None,
    max_predicate_failures: int = 32,
    on_predicate_error: PredicateErrorMode = "auto",
    storage: StreamingOptions | None = None,
    streaming: StreamingOptions | None = None,
    return_output: bool = False,
    postprocess: str = "none",
    random_seed: int | None = None,
    activation_transform: ActivationPostfunc | None = None,
    save_raw_activations: bool = True,
    save_grads: GradPredicateFn | bool | CaptureSpec | None = None,
    default_grad: bool | CaptureSpec | MissingType = MISSING,
    grad_transform: GradientPostfunc | None = None,
    save_raw_gradients: bool = True,
    backward_ready: bool = False,
    backend: BackendName | None = None,
) -> Recording | tuple[Any, Recording]:
    """Record one model forward pass with capture predicates.

    Migration note
    --------------
    ``record(save=...)`` is the canonical predicate spelling and matches
    ``trace(save=...)``. ``keep_op=`` and ``keep_module=`` are deprecated
    compatibility aliases; ``tl.fastlog.record`` remains a shim to this API.

    Parameters
    ----------
    model:
        PyTorch module to execute.
    input_args:
        Tensor, list, or tuple of positional model inputs.
    input_kwargs:
        Optional keyword arguments for the model call.
    save, keep_op, keep_module, default_op, default_module, history_size,
    lookback, lookback_payload_policy, include_source_events, max_predicate_failures,
    on_predicate_error, storage, streaming, random_seed:
        Fastlog recording options.
    intervene:
        Optional predicate-time intervention slot evaluated on operation contexts.
    halt:
        Optional predicate evaluated after each event's save decision. Returning
        ``True`` stops the active forward pass and marks the recording halted.
    activation_transform:
        Optional callable applied to each retained out copy after
        dtype/device transforms. Errors propagate as
        :class:`torchlens.TorchLensPostfuncError`.
    save_raw_activations:
        When ``False`` and ``activation_transform`` is set, only transformed
        payloads are retained. Defaults to ``True`` to mirror the slow path.
    backward_ready:
        If True, omitted defaults are promoted to keep-grad capture specs.
    backend:
        Optional backend selector. ``tl.record`` is torch-only in backend v1;
        non-torch backends raise a canonical unsupported error.
    return_output:
        Whether to return ``(model_output, recording)``.
    postprocess:
        Optional postprocess enrichment preset.

    Returns
    -------
    Recording | tuple[Any, Recording]
        Fastlog recording, optionally with the model output.
    """

    if backend not in (None, "torch"):
        raise BackendUnsupportedError(
            "tl.record() is torch-only in backend v1. Use tl.trace(..., backend='jax') "
            "for the JAX full-save preview."
        )
    reject_compiled_model(model, api_name="torchlens.fastlog.record")
    if storage is not None and streaming is not None:
        raise TypeError("Do not pass both `storage` and `streaming`.")
    validate_postprocess(postprocess)
    resolved_keep_op = _resolve_save_alias(save=save, keep_op=keep_op)
    if keep_module is not MISSING:
        warnings.warn(
            "record(keep_module=...) is deprecated; use record(save=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    input_args = _coerce_input_args(model, input_args)
    with Recorder(
        model,
        save=resolved_keep_op,
        keep_op=MISSING,
        keep_module=keep_module,
        default_op=default_op,
        default_module=default_module,
        history_size=history_size,
        lookback=lookback,
        lookback_payload_policy=lookback_payload_policy,
        include_source_events=include_source_events,
        intervene=intervene,
        halt=halt,
        max_predicate_failures=max_predicate_failures,
        on_predicate_error=on_predicate_error,
        storage=storage,
        streaming=streaming,
        random_seed=random_seed,
        activation_transform=activation_transform,
        save_raw_activations=save_raw_activations,
        save_grads=save_grads,
        default_grad=default_grad,
        grad_transform=grad_transform,
        save_raw_gradients=save_raw_gradients,
        backward_ready=backward_ready,
    ) as recorder:
        output = recorder.log(input_args, input_kwargs)
    recording = recorder.recording
    if postprocess != "none":
        recording = recording.enrich(postprocess)
    if return_output:
        return output, recording
    return recording
