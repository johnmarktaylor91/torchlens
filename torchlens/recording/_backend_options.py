"""Option resolution for ``tl.record`` on a backend that binds a record producer.

The backend recorder reads the same doors as the torch ``Recorder``
(``capture=``, ``recording=``, ``save=``, ``stop_after=``, ``storage=``) and
resolves the same ``_RecordingConfig``. Options whose meaning lives in torch's
capture path (gradients, transforms, graph-connected copies, live edits,
narration, module-boundary records) refuse typed unless the producer lists them
in ``supported_options``; nothing is accepted and then ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from .._deprecations import MISSING
from ..backends._backend_protocol import refusal_code_for_capability
from ..backends.registry import BackendUnsupportedError
from ._capture_door import _unfold_capture_options
from ._options import _RecordingConfig, _resolve_recording_config
from ._recording_door import _unfold_recording_options
from ._validation import refuse_episode_on_record, validate_recording_options

if TYPE_CHECKING:
    from ..options import CaptureOptions
    from ._backend_events import BackendRecordProducer
    from ._options import RecordingOptions

_API_NAME = "tl.record"


@dataclass(frozen=True, slots=True)
class RecordDoors:
    """The recorder's keyword doors, as ``tl.record`` passes them (``MISSING`` = not given)."""

    save: Any = MISSING
    intervene: Any = MISSING
    stop_after: Any = MISSING
    storage: Any = MISSING
    narrate: Any = MISSING
    capture: CaptureOptions | None = None
    recording: RecordingOptions | None = None
    episode: Any = None
    lean: Any = MISSING


def resolve_backend_record_config(
    producer: BackendRecordProducer, doors: RecordDoors
) -> _RecordingConfig:
    """Resolve and validate the recorder config for one backend producer.

    Parameters
    ----------
    producer:
        The backend's record producer; ``producer.supported_options`` names the
        torch-path options it honours itself.
    doors:
        The recorder's keyword doors (``MISSING`` where the caller gave none).

    Returns
    -------
    _RecordingConfig
        The resolved configuration, validated as the torch recorder validates it.

    Raises
    ------
    BackendUnsupportedError
        For an option the producer does not honour.
    """

    flat = _unfold_capture_options(doors.capture, api_name=_API_NAME)
    knobs = _unfold_recording_options(doors.recording, api_name=_API_NAME, lean=doors.lean)
    refuse_episode_on_record(doors.episode)
    backend = str(producer.backend)
    supported = frozenset(getattr(producer, "supported_options", frozenset()))
    for option, code in _requested_torch_path_options(flat, knobs, doors):
        if option not in supported:
            refuse_record_option(backend, option, code=code)
    options = _resolve_recording_config(
        keep_op=doors.save,
        halt=doors.stop_after,
        intervene=doors.intervene,
        streaming=doors.storage,
        lookback=flat["lookback"],
        lookback_payload_policy=flat["lookback_payload_policy"],
        seed=flat["seed"],
        save_raw_activations=flat["save_raw_activations"],
        **{name: value for name, value in knobs.items() if name != "default_grad"},
    )
    validate_recording_options(options, backward_requested=False)
    return options


def _requested_torch_path_options(
    flat: dict[str, Any], knobs: dict[str, Any], doors: RecordDoors
) -> list[tuple[str, str]]:
    """Return ``(option, refusal code)`` for every torch-path option the caller set.

    One independent check per option, read top to bottom (radon C, 11): the
    list is the census of torch-path options, so it stays in one place.
    """

    trace_code = "trace_option_unsupported"
    requested: list[tuple[str, str]] = []
    if doors.intervene not in (MISSING, None):
        requested.append(("intervene", refusal_code_for_capability("can_intervene")))
    if doors.narrate not in (MISSING, None, False):
        requested.append(("narrate", trace_code))
    grads = flat["backward_ready"] or flat["save_grads"] not in (None, False, MISSING)
    if grads or knobs["default_grad"] not in (MISSING, False):
        requested.append(("save_grads", refusal_code_for_capability("backward_capture")))
    if flat["activation_transform"] is not None:
        requested.append(("activation_transform", trace_code))
    if flat["detach_saved_activations"] is False:
        requested.append(("detach_saved_activations", trace_code))
    if flat["seed"] is not None:
        requested.append(("seed", refusal_code_for_capability("rng_replay")))
    if flat["lookback_payload_policy"] != "metadata_only":
        requested.append(("lookback_payload_policy", trace_code))
    if knobs["default_module"] not in (MISSING, False):
        requested.append(("default_module", trace_code))
    return requested


def refuse_record_option(
    backend: str, option: str, *, code: str = "trace_option_unsupported"
) -> NoReturn:
    """Raise the typed refusal for a record option this backend's producer does not honour.

    Parameters
    ----------
    backend:
        Backend name.
    option:
        The option or capture-spec field the producer has no step for.
    code:
        An existing contracted refusal code.

    Raises
    ------
    BackendUnsupportedError
        Always, naming the backend and the option.
    """

    raise BackendUnsupportedError(
        f"tl.record(backend={backend!r}) does not support {option}: the backend's record "
        "producer has no step for it, so it would be ignored. Remove it, or record this "
        "model on the torch backend.",
        code=code,
        backend=backend,
    )
