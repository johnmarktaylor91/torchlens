"""One-shot public fastlog recording API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from torchlens.backends.registry import BackendUnsupportedError

from .._capture_state_helpers import unwrap_compiled_model
from .._deprecations import MISSING, MissingType
from .._errors import ArgumentTypeError
from .._input_coerce import _coerce_input_args
from .._output_slot import returned as output_returned
from .._robustness import check_model_and_input_variants
from ..backends.registry import (
    TORCH_BACKEND_NAME,
    BackendName,
    get_backend_spec,
    require_capability_implementation,
)
from ..intervention.predicates import InterventionPredicate
from ..options import CaptureOptions, StorageOptions
from ._generation import GenerationRecording, enrich_generation, record_generation
from ._recorder import Recorder
from ._validation import validate_postprocess
from .options import ForwardErrorMode, HaltPredicateFn, PredicateErrorMode, PredicateFn
from .types import CaptureSpec, Recording

_CAPTURE_DOOR_FIELDS: tuple[str, ...] = (
    "lookback",
    "lookback_payload_policy",
    "seed",
    "activation_transform",
    "save_raw_activations",
    "save_grads",
    "grad_transform",
    "save_raw_grads",
    "detach_saved_activations",
)


#: The one-shot door's historical defaults for the knobs it used to take flat.
_RECORD_FLAT_DEFAULTS: dict[str, Any] = {
    "lookback": 0,
    "lookback_payload_policy": "metadata_only",
    "save_raw_activations": True,
    "save_raw_grads": True,
    "seed": None,
    "activation_transform": None,
    "grad_transform": None,
    "save_grads": None,
}


def _unfold_capture_options(capture: CaptureOptions | None) -> dict[str, Any]:
    """Read the former ``record`` door keywords off ``capture=``.

    An explicitly set field is forwarded; an unset one takes the door's former
    flat default (``MISSING`` where the Recorder resolves it). ``backward_ready``
    is a plain bool on the Recorder and reads the capture value directly.
    """

    flat: dict[str, Any] = {}
    if capture is not None:
        unread = sorted(
            name
            for name in capture._specified_fields  # noqa: SLF001
            if name not in _CAPTURE_DOOR_FIELDS and name != "backward_ready"
        )
        if unread:
            raise ArgumentTypeError(
                f"tl.record does not read CaptureOptions field(s) {', '.join(unread)}: the "
                "recorder has no door for them, so they would be a silent no-op",
                code="record_capture_option_unsupported",
                remedy=(
                    "drop the field(s) from capture= for tl.record, or use tl.trace, which "
                    "reads every CaptureOptions field"
                ),
                argument="capture",
            )
    for name in _CAPTURE_DOOR_FIELDS:
        explicit = isinstance(capture, CaptureOptions) and capture._is_field_explicit(name)  # noqa: SLF001
        flat[name] = getattr(capture, name) if explicit else MISSING
    flat["backward_ready"] = bool(capture.backward_ready) if capture is not None else False
    # The door's former flat defaults, resolved here so the generation path
    # (which reads the bundle directly) sees the same values the Recorder would.
    for name, default in _RECORD_FLAT_DEFAULTS.items():
        if flat[name] is MISSING:
            flat[name] = default
    return flat


def _require_torch_fastlog_backend(backend: Any) -> None:
    """Refuse unless the resolved backend binds the torch one-shot Recorder as fastlog."""

    # The capability table gates this surface for EVERY resolution, including
    # the default torch path: flipping torch's fastlog flag False must refuse
    # instead of silently running the Recorder anyway.
    backend_spec = get_backend_spec(str(backend) if backend is not None else "torch")
    if not backend_spec.capabilities.fastlog:
        from ..backends._backend_protocol import refusal_code_for_capability

        if str(backend_spec.name) == TORCH_BACKEND_NAME:
            raise BackendUnsupportedError(
                "tl.record() refuses: the resolved 'torch' backend declares "
                "capabilities.fastlog=False, and the registered capability "
                "table is the load-bearing gate for this surface.",
                code=refusal_code_for_capability("fastlog"),
                backend=TORCH_BACKEND_NAME,
            )
        raise BackendUnsupportedError(
            "tl.record() is torch-only in backend v1. Use tl.trace(..., backend='jax') "
            "for the JAX full-save preview.",
            code=refusal_code_for_capability("fastlog"),
            backend=str(backend_spec.name),
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


def record(
    model: nn.Module,
    input_args: Any,
    *,
    input_kwargs: dict[str, Any] | None = None,
    save: PredicateFn | None = None,
    intervene: InterventionPredicate | None = None,
    stop_after: HaltPredicateFn | None = None,
    storage: StorageOptions | None = None,
    capture: CaptureOptions | None = None,
    default_op: bool | CaptureSpec | MissingType = MISSING,
    default_module: bool | CaptureSpec | MissingType = MISSING,
    history_size: int = 8,
    include_source_events: bool = False,
    max_predicate_failures: int = 32,
    on_predicate_error: PredicateErrorMode = "auto",
    on_forward_error: ForwardErrorMode = "raise",
    postprocess: str = "none",
    default_grad: bool | CaptureSpec | MissingType = MISSING,
    narrate: Any | None = None,
    backend: BackendName | None = None,
    generate_kwargs: Mapping[str, Any] | None = None,
) -> Recording | GenerationRecording:
    """Record one model forward pass with capture predicates.

    ``record(save=...)`` is the one predicate spelling and matches
    ``trace(save=...)``; ``tl.record`` remains a shim to this API.

    Parameters
    ----------
    model:
        PyTorch module to execute.
    input_args:
        Tensor, list, or tuple of positional model inputs.
    input_kwargs:
        Optional keyword arguments for the model call.
    save:
        The one predicate spelling (matches ``trace(save=...)``).
    storage:
        Grouped storage options (``tl.options.StorageOptions``).
    capture:
        ``tl.options.CaptureOptions``. The fields ``record`` reads: ``lookback``,
        ``lookback_payload_policy``, ``seed``, ``activation_transform``,
        ``save_raw_activations``, ``save_grads``, ``grad_transform``,
        ``save_raw_grads``, ``backward_ready`` and ``detach_saved_activations``
        (meanings below).
    default_op, default_module, history_size, include_source_events,
    max_predicate_failures, on_predicate_error:
        Fastlog recording options.
    on_forward_error:
        Controls failed-forward handling: ``"raise"`` raises the capture's
        ``TraceCaptureError``, ``"attach_partial"`` also attaches the failed partial
        Recording to it as ``partial_recording``, ``"return_partial"`` returns it,
        and its ``output`` refuses with ``output_forward_failed``.
        With ``return_output=True``, the returned output is ``None`` because no
        valid model output exists on failure.
    intervene:
        Optional predicate-time intervention slot evaluated on operation contexts.
    stop_after:
        Optional predicate evaluated after each event's save decision. Returning
        ``True`` stops the active forward pass and marks the recording stopped.
    CaptureOptions fields read by ``record``:
        ``activation_transform`` is applied to each retained out copy after
        dtype/device transforms (errors propagate as
        :class:`torchlens.UserTransformError`); ``save_raw_activations=False``
        with a transform keeps only transformed payloads. ``backward_ready=True``
        promotes omitted defaults to keep-grad capture specs.
        ``detach_saved_activations`` (DOCUMENTED-UNSTABLE; ``record`` default
        ``True``) detaches retained activations; ``False`` keeps graph-connected
        copies wherever the recorded tensor carries autograd history (a spec with
        ``save_grad=True`` stays attached either way; disk payloads are always
        detached; ``backward_ready=True`` refuses an explicit ``True``).
        ``save_grads``, ``default_grad`` (a ``record`` keyword),
        ``grad_transform`` and ``save_raw_grads`` configure a later
        ``Recording.backward()``: gradient hooks are installed only when
        ``backward_ready=True``, ``save_grads=`` or ``default_grad=`` is given;
        without one, ``log_backward()`` refuses with code
        ``recording_backward_not_requested``, and a record made under
        ``torch.no_grad()`` or ``torch.inference_mode()`` implies
        ``inference_only=True`` (``recording_backward_inference_only``).
    postprocess:
        Optional postprocess enrichment preset (``"none"`` by default).
    narrate:
        Live narration scope: ``True``, ``"modules"``, a live selector, or a
        grouped ``tl.options.NarrationOptions``. Narration and retention are fully
        independent ("narrate everything, keep nothing" is legal).
    backend:
        Optional backend selector. ``tl.record`` is torch-only in backend v1;
        non-torch backends raise a canonical unsupported error.
    generate_kwargs:
        DOCUMENTED-UNSTABLE. When given (``{}`` for the model's defaults), run
        the model's own ``generate(*input_args, **input_kwargs,
        **generate_kwargs)`` once instead of one forward, and record the
        selected sites at every forward it runs (the prefill, then each decode
        step) while the KV cache is on, so each decode step records only its
        new position. Returns a :class:`GenerationRecording` (step and absolute
        position of every forward, the token sequences, the settings that
        decide the tokens). Beam search, more than one return sequence, batched
        prompts, assisted decoding, encoder-decoder models, ``use_cache=False``,
        ``intervene=``, ``stop_after=`` and partial forward-error modes refuse with
        code ``record_generation_unsupported``.

    Returns
    -------
    Recording | GenerationRecording
        The recording (a ``GenerationRecording`` with ``generate_kwargs``). Its
        ``output`` is the model's return value (``generate()``'s own return value
        with ``generate_kwargs``), held in memory only.

    Examples
    --------
    >>> recording = tl.record(model, x, save=tl.func("relu"))
    >>> recording.status, len(recording.records)
    ('complete', 1)
    >>> tuple(recording.records[0].ram_payload.shape)
    (1, 4, 8, 8)

    """

    _require_torch_fastlog_backend(backend)
    model = unwrap_compiled_model(model)
    flat = _unfold_capture_options(capture)
    lookback = flat["lookback"]
    lookback_payload_policy = flat["lookback_payload_policy"]
    halt = stop_after
    seed = flat["seed"]
    activation_transform = flat["activation_transform"]
    save_raw_activations = flat["save_raw_activations"]
    save_grads = flat["save_grads"]
    grad_transform = flat["grad_transform"]
    save_raw_grads = flat["save_raw_grads"]
    backward_ready = flat["backward_ready"]
    detach_saved_activations = flat["detach_saved_activations"]
    validate_postprocess(postprocess)
    input_args = _coerce_input_args(model, input_args)
    # Fail fast on tensor variants the logging pipeline cannot handle (meta tensors have no
    # storage, sparse layouts break copy/print/FLOPs paths, symbolic shapes break metadata).
    # record() shares trace()'s decorated hot path, so it must enforce the SAME up-front guard
    # instead of crashing deep in capture with an opaque torch error.
    check_model_and_input_variants(model, input_args, input_kwargs)
    if generate_kwargs is not None:
        output, generation = record_generation(
            model,
            input_args,
            input_kwargs,
            generate_kwargs,
            {
                "save": save,
                "default_op": default_op,
                "default_module": default_module,
                "history_size": history_size,
                "lookback": lookback,
                "lookback_payload_policy": lookback_payload_policy,
                "include_source_events": include_source_events,
                "intervene": intervene,
                "halt": halt,
                "max_predicate_failures": max_predicate_failures,
                "on_predicate_error": on_predicate_error,
                "on_forward_error": on_forward_error,
                "storage": storage,
                "seed": seed,
                "activation_transform": activation_transform,
                "save_raw_activations": save_raw_activations,
                "save_grads": save_grads,
                "default_grad": default_grad,
                "grad_transform": grad_transform,
                "save_raw_grads": save_raw_grads,
                "backward_ready": backward_ready,
                "narrate": narrate,
                "detach_saved_activations": detach_saved_activations,
            },
        )
        object.__setattr__(generation.recording, "_output", output_returned(output))
        return enrich_generation(generation, postprocess)
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
        seed=seed,
        activation_transform=activation_transform,
        save_raw_activations=save_raw_activations,
        save_grads=save_grads,
        default_grad=default_grad,
        grad_transform=grad_transform,
        save_raw_grads=save_raw_grads,
        backward_ready=backward_ready,
        narrate=narrate,
        detach_saved_activations=detach_saved_activations,
    ) as recorder:
        recorder(input_args, input_kwargs)
    result = recorder.recording
    if postprocess != "none":
        result = result.enrich(postprocess)
    return result
