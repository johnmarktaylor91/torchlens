"""One-shot public fastlog recording API."""

from __future__ import annotations

from typing import Any

from torch import nn

from .._capture_state_helpers import unwrap_compiled_model
from .._deprecations import MISSING, MissingType
from .._errors import KeywordConflictError
from .._input_coerce import _coerce_input_args
from .._robustness import check_model_and_input_variants
from ..backends import (
    TORCH_BACKEND_NAME,
    BackendName,
    BackendUnsupportedError,
    get_backend_spec,
    require_capability_implementation,
)
from ..intervention.predicates import InterventionPredicate
from ..options import StreamingOptions
from ..types import ActivationPostfunc, GradientPostfunc
from ._recorder import Recorder
from ._validation import validate_postprocess
from .options import (
    ForwardErrorMode,
    GradPredicateFn,
    HaltPredicateFn,
    LookbackPayloadPolicy,
    PredicateErrorMode,
    PredicateFn,
)
from .types import CaptureSpec, Recording


def record(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None = None,
    *,
    save: PredicateFn | None = None,
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
    on_forward_error: ForwardErrorMode = "raise",
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
    echo: Any | None = None,
    backend: BackendName | None = None,
) -> Recording | tuple[Any, Recording]:
    """Record one model forward pass with capture predicates.

    ``record(save=...)`` is the one predicate spelling and matches
    ``trace(save=...)``; ``tl.fastlog.record`` remains a shim to this API.

    Parameters
    ----------
    model:
        PyTorch module to execute.
    input_args:
        Tensor, list, or tuple of positional model inputs.
    input_kwargs:
        Optional keyword arguments for the model call.
    save, default_op, default_module, history_size, lookback,
    lookback_payload_policy, include_source_events, max_predicate_failures,
    on_predicate_error, storage, streaming, random_seed:
        Fastlog recording options.
    on_forward_error:
        Controls failed-forward handling. ``"raise"`` preserves the historical
        behavior, ``"attach_partial"`` attaches ``exc.partial_recording`` and
        re-raises, and ``"return_partial"`` returns a failed partial Recording.
        With ``return_output=True``, the returned output is ``None`` because no
        valid model output exists on failure.
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
    echo:
        Live narration scope: ``True``, ``"modules"``, a live selector, or a
        grouped ``tl.options.EchoOptions``. Narration and retention are fully
        independent ("narrate everything, keep nothing" is legal).
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

    # The capability table gates this surface for EVERY resolution, including
    # the default torch path: flipping torch's fastlog flag False must refuse
    # instead of silently running the Recorder anyway.
    backend_spec = get_backend_spec(str(backend) if backend is not None else "torch")
    if not backend_spec.capabilities.fastlog:
        if str(backend_spec.name) == TORCH_BACKEND_NAME:
            raise BackendUnsupportedError(
                "tl.record() refuses: the resolved 'torch' backend declares "
                "capabilities.fastlog=False, and the registered capability "
                "table is the load-bearing gate for this surface."
            )
        raise BackendUnsupportedError(
            "tl.record() is torch-only in backend v1. Use tl.trace(..., backend='jax') "
            "for the JAX full-save preview."
        )
    # The flag alone never opens the gate: the spec must bind the fastlog
    # implementing surface, and this entry only runs the torch Recorder —
    # a foreign implementation cannot be silently substituted with it.
    fastlog_implementation = require_capability_implementation(backend_spec, "fastlog")
    if fastlog_implementation is not Recorder:
        raise BackendUnsupportedError(
            f"tl.record() cannot dispatch backend {backend_spec.name!r}: its "
            "registered fastlog implementation is not the torch one-shot "
            "Recorder, and backend v1 record() has no non-torch dispatch path."
        )
    model = unwrap_compiled_model(model)
    if storage is not None and streaming is not None:
        raise KeywordConflictError(
            "Do not pass both `storage` and `streaming`",
            code="storage_argument_conflict",
            remedy="prefer storage=, or remove one of the two arguments",
        )
    validate_postprocess(postprocess)
    input_args = _coerce_input_args(model, input_args)
    # Fail fast on tensor variants the logging pipeline cannot handle (meta tensors have no
    # storage, sparse layouts break copy/print/FLOPs paths, symbolic shapes break metadata).
    # record() shares trace()'s decorated hot path, so it must enforce the SAME up-front guard
    # instead of crashing deep in capture with an opaque torch error.
    check_model_and_input_variants(model, input_args, input_kwargs)
    with Recorder(
        model,
        save=save,
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
        on_forward_error=on_forward_error,
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
        echo=echo,
    ) as recorder:
        output = recorder.log(input_args, input_kwargs)
    recording = recorder.recording
    if postprocess != "none":
        recording = recording.enrich(postprocess)
    if return_output:
        return output, recording
    return recording
