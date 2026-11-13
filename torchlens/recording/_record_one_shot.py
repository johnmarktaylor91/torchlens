"""One-shot public fastlog recording API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch import nn

from torchlens.backends.registry import BackendUnsupportedError

from .._capture_state_helpers import unwrap_compiled_model
from .._deprecations import MISSING
from .._errors import ArgumentTypeError, InvalidArgumentError
from .._input_coerce import _coerce_inputs
from .._output_slot import returned as output_returned
from .._robustness import check_model_and_input_variants
from ..backends.registry import (
    BackendName,
    get_backend_spec,
    require_capability_implementation,
)
from ..intervention._predicates import InterventionPredicate
from ..options import CaptureOptions, StorageOptions, _merge_grouped_options
from ._backend_events import is_backend_record_producer
from ._backend_options import refuse_record_option
from ._backend_recorder import backend_recorder_factory
from ._generation import GenerationRecording, enrich_generation, record_generation
from ._generation_options import GenerationOptions
from ._options import HaltPredicateFn, PredicateFn, RecordingOptions
from ._recorder import Recorder
from ._types import Recording
from ._validation import refuse_episode_on_record, validate_postprocess


def _refuse_unread_fields(group: str, unread: list[str], why: str, remedy: str) -> None:
    """Refuse grouped-option fields ``tl.record`` does not read (a silent no-op)."""

    if unread:
        raise ArgumentTypeError(
            f"tl.record does not read {group}= field(s) {', '.join(unread)}: {why}, so they "
            "would be a silent no-op",
            code="record_capture_option_unsupported",
            remedy=remedy,
            argument=group,
        )


#: The ``RecordingOptions`` fields ``tl.record`` reads off ``recording=``. An
#: unset field without a door default here stays ``MISSING`` so the Recorder
#: resolves it (``default_op``, ``default_module``, ``default_grad``).
_RECORDING_DOOR_DEFAULTS: dict[str, Any] = {
    "default_op": MISSING,
    "default_module": MISSING,
    "history_size": 8,
    "include_source_events": False,
    "max_predicate_failures": 32,
    "on_predicate_error": "auto",
    "on_forward_error": "raise",
    "default_grad": MISSING,
    "postprocess": "none",
    "lean": False,
}


def _unfold_recording_options(recording: RecordingOptions | None, lean: Any) -> dict[str, Any]:
    """Read the recorder knobs off ``recording=``.

    ``keep_op`` and ``intervene`` are the door's own ``save=`` and
    ``intervene=``; setting them on the bundle refuses instead of being a
    silent no-op. ``lean`` arrives as the door keyword or on ``recording=``;
    either one asks for it (a non-``False`` door value wins and is validated
    by the Recorder).
    """

    knobs = _read_recording_options(recording)
    if lean is not False:
        knobs["lean"] = lean
    return knobs


def _read_recording_options(recording: RecordingOptions | None) -> dict[str, Any]:
    """Return the ``RecordingOptions`` fields ``tl.record`` reads, door defaults filled in."""

    if recording is None:
        return dict(_RECORDING_DOOR_DEFAULTS)
    recording = _merge_grouped_options(
        option=recording,
        option_factory=RecordingOptions,
        flat_to_group={},
        flat_values={},
        group_name="recording",
        conflict_message="",
    )
    unread = sorted(set(recording._specified_fields) - set(_RECORDING_DOOR_DEFAULTS))  # noqa: SLF001
    _refuse_unread_fields(
        "recording",
        unread,
        "the door takes them as save= and intervene=",
        "pass the predicate as tl.record(save=...) and the intervention as intervene=",
    )
    return {
        name: getattr(recording, name) if recording._is_field_explicit(name) else default  # noqa: SLF001
        for name, default in _RECORDING_DOOR_DEFAULTS.items()
    }


def _require_generate_kwargs(
    generation: GenerationOptions | None, generate_kwargs: Mapping[str, Any] | None
) -> None:
    """Refuse ``generation=`` without ``generate_kwargs=``."""

    if generation is not None and generate_kwargs is None:
        raise InvalidArgumentError(
            "tl.record(generation=GenerationOptions(...)) needs generate_kwargs=: the per-token "
            "options describe a recorded generation",
            code="record_generation_options_invalid",
            remedy="pass generate_kwargs={} (the model's defaults) or drop generation=",
        )


def _fastlog_recorder_factory(
    backend: Any, generate_kwargs: Mapping[str, Any] | None, recording: RecordingOptions | None
) -> Callable[..., Any]:
    """Return the recorder ``tl.record`` runs for the resolved backend.

    The capability table gates this surface for every resolution, including the
    default torch path, and the flag alone never opens it: the spec must bind
    the fastlog implementing surface. Torch binds the ``Recorder`` class; any
    other backend binds a :class:`BackendRecordProducer`, which runs through
    ``BackendRecorder``. Any other binding refuses typed.
    """

    from ..backends._backend_protocol import refusal_code_for_capability

    backend_spec = get_backend_spec(str(backend) if backend is not None else "torch")
    name = str(backend_spec.name)
    if backend_spec.capabilities.fastlog:
        implementation = require_capability_implementation(backend_spec, "fastlog")
        if implementation is Recorder:
            return Recorder
        if is_backend_record_producer(implementation):
            _refuse_torch_only_record_doors(name, generate_kwargs, recording)
            return backend_recorder_factory(implementation)
        problem = (
            f"tl.record() cannot dispatch backend {name!r}: its registered fastlog "
            "implementation is neither the torch Recorder nor a backend record producer, "
            "so record() has no non-torch dispatch path for it."
        )
    else:
        problem = (
            f"tl.record() refuses: the resolved {name!r} backend declares "
            "capabilities.fastlog=False, and the registered capability table is the "
            "load-bearing gate for this surface (tl.record is torch-only on a backend "
            "that binds no record producer). Use tl.trace(..., backend=...) instead."
        )
    raise BackendUnsupportedError(
        problem, code=refusal_code_for_capability("fastlog"), backend=name
    )


def _refuse_torch_only_record_doors(
    name: str, generate_kwargs: Mapping[str, Any] | None, recording: RecordingOptions | None
) -> None:
    """Refuse the one-shot doors that run only on the torch path."""

    if generate_kwargs is not None:
        refuse_record_option(
            name, "generate_kwargs (a torch generate() loop)", code="record_generation_unsupported"
        )
    if (
        recording is not None
        and recording._is_field_explicit("postprocess")  # noqa: SLF001
        and recording.postprocess != "none"
    ):
        refuse_record_option(name, "recording=RecordingOptions(postprocess=...)")


def _checked_inputs(
    recorder_factory: Any, model: nn.Module, inputs: Any, input_kwargs: Any
) -> tuple[Any, Any]:
    """Coerce ``inputs`` and ``input_kwargs`` and run trace()'s up-front tensor-variant guard on them.

    Fail fast on tensor variants the logging pipeline cannot handle (meta tensors have no
    storage, sparse layouts break copy/print/FLOPs paths, symbolic shapes break metadata).
    record() shares trace()'s decorated hot path, so it must enforce the SAME up-front guard
    instead of crashing deep in capture with an opaque torch error. A backend
    producer takes its inputs as given and checks them itself.
    """

    if recorder_factory is not Recorder:
        return inputs, input_kwargs

    coerced, coerced_kwargs = _coerce_inputs(model, inputs, input_kwargs)
    check_model_and_input_variants(model, coerced, coerced_kwargs)
    return coerced, coerced_kwargs


def record(
    model: nn.Module,
    inputs: Any,
    *,
    input_kwargs: dict[str, Any] | None = None,
    save: PredicateFn | None = None,
    intervene: InterventionPredicate | None = None,
    stop_after: HaltPredicateFn | None = None,
    storage: StorageOptions | None = None,
    capture: CaptureOptions | None = None,
    recording: RecordingOptions | None = None,
    narrate: Any | None = None,
    backend: BackendName | None = None,
    generate_kwargs: Mapping[str, Any] | None = None,
    episode: Any | None = None,
    lean: bool = False,
    generation: GenerationOptions | None = None,
) -> Recording | GenerationRecording:
    """Record one model forward pass with capture predicates.

    ``record(save=...)`` is the one predicate spelling and matches
    ``trace(save=...)``; ``tl.record`` remains a shim to this API.

    Parameters
    ----------
    model:
        PyTorch module to execute.
    inputs:
        Tensor, list, or tuple of positional model inputs; for a Hugging Face model also a string, PIL image or modality dict, resolved as ``tl.trace`` resolves it.
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
    recording:
        Grouped recorder options (``tl.options.RecordingOptions``). The fields
        ``record`` reads: ``default_op``, ``default_module``, ``default_grad``,
        ``history_size``, ``include_source_events``, ``max_predicate_failures``,
        ``on_predicate_error``, ``on_forward_error``, ``postprocess`` and ``lean``
        (see ``RecordingOptions``); ``keep_op`` and ``intervene`` refuse with code
        ``record_capture_option_unsupported`` (pass ``save=`` and ``intervene=``).
        ``on_forward_error`` controls failed-forward handling: ``"raise"`` raises
        the capture's ``TraceCaptureError``, ``"attach_partial"`` also attaches the
        failed partial Recording to it as ``partial_recording``,
        ``"return_partial"`` returns it, and its ``output`` refuses with
        ``output_forward_failed``. ``postprocess`` is an optional enrichment preset
        (``"none"`` by default).
    intervene:
        Optional predicate-time intervention slot evaluated on operation contexts.
    stop_after:
        Optional predicate evaluated after each event's save decision. Returning
        ``True`` stops the active forward pass and marks the recording stopped.
    CaptureOptions fields read by ``record``:
        ``activation_transform`` is applied to each retained out copy after dtype/device transforms
        (errors propagate as :class:`torchlens.UserTransformError`); ``save_raw_activations=False``
        with a transform keeps only transformed payloads. ``backward_ready=True`` promotes omitted
        defaults to keep-grad capture specs. ``detach_saved_activations`` (DOCUMENTED-UNSTABLE;
        ``record`` default ``True``) detaches retained activations; ``False`` keeps graph-connected
        copies wherever the recorded tensor carries autograd history (a spec with ``save_grad=True``
        stays attached either way; disk payloads are always detached; ``backward_ready=True``
        refuses an explicit ``True``). ``save_grads``, ``default_grad`` (a ``RecordingOptions``
        field), ``grad_transform`` and ``save_raw_grads`` configure a later
        ``Recording.backward()``: gradient hooks are installed only when ``backward_ready=True``,
        ``save_grads=`` or ``default_grad=`` is given; without one, ``backward()`` refuses with code
        ``recording_backward_not_requested``, and a record made under ``torch.no_grad()`` or
        ``torch.inference_mode()`` implies ``inference_only=True``
        (``recording_backward_inference_only``).
    narrate:
        Live narration scope: ``True``, ``"modules"``, a live selector, or a
        grouped ``tl.options.NarrationOptions``. Narration and retention are fully
        independent ("narrate everything, keep nothing" is legal).
    backend:
        Optional backend selector. A backend that binds a record producer records
        through it; any other non-torch backend refuses with ``fastlog_unsupported``.
    generate_kwargs:
        DOCUMENTED-UNSTABLE. When given (``{}`` for the model's defaults), run the model's own
        ``generate(*inputs, **input_kwargs, **generate_kwargs)`` once instead of one forward, and
        record the selected sites at every forward it runs (the prefill, then each decode step)
        while the KV cache is on, so each decode step records only its new position. Returns a
        :class:`GenerationRecording` (step and absolute position of every forward, the token
        sequences, the settings that decide the tokens). Beam search, more than one return sequence,
        batched prompts, assisted decoding, encoder-decoder models, ``use_cache=False``,
        ``intervene=``, ``stop_after=`` and partial forward-error modes refuse with code
        ``record_generation_unsupported``.
    episode:
        Refused typed (``record_episode_unsupported``); episode capture runs on ``tl.trace``.
    lean:
        PROVISIONAL SPELLING. ``True`` journals only the events the selection keeps;
        ``Recording.lean_omissions`` names what it omitted and the reads that refuse
        (``recording_lean_omitted``). Default ``False`` is unchanged.
    generation:
        DOCUMENTED-UNSTABLE. :class:`~torchlens.recording.GenerationOptions` widening the generation
        door: per-token ``reducers=``, ``kv_sites=``, a ``keep_last=`` ring buffer with ``spill=``,
        and the ``max_saved_bytes=`` guard. Read the per-token values with
        ``GenerationRecording.series(name)``. Needs ``generate_kwargs=``.

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

    recorder_factory = _fastlog_recorder_factory(backend, generate_kwargs, recording)
    model = unwrap_compiled_model(model)
    refuse_episode_on_record(episode)
    knobs = _unfold_recording_options(recording, lean)
    postprocess = knobs.pop("postprocess")
    validate_postprocess(postprocess)
    inputs, input_kwargs = _checked_inputs(recorder_factory, model, inputs, input_kwargs)
    _require_generate_kwargs(generation, generate_kwargs)
    recorder_kwargs: dict[str, Any] = {
        "save": save,
        "intervene": intervene,
        "stop_after": stop_after,
        "storage": storage,
        "narrate": narrate,
        "capture": capture,
        "recording": RecordingOptions(**knobs),
    }
    if generate_kwargs is not None:
        output, generated = record_generation(
            model, (inputs, input_kwargs), generate_kwargs, recorder_kwargs, generation
        )
        object.__setattr__(generated.recording, "_output", output_returned(output))
        return enrich_generation(generated, postprocess)
    with recorder_factory(model, **recorder_kwargs) as recorder:
        recorder(inputs, input_kwargs)
    result = recorder.recording
    if postprocess != "none":
        result = result.enrich(postprocess)
    return result
