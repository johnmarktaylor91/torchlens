"""The ``recording=`` door of ``torchlens.recording.Recorder``.

``Recorder(recording=RecordingOptions(...))`` carries the recorder's own knobs
(``default_op``, ``default_module``, ``history_size``, ``include_source_events``,
``max_predicate_failures``, ``on_predicate_error``, ``on_forward_error``,
``default_grad``, ``lean``), the same shape ``tl.record`` takes; ``lean``
also arrives as the constructor's door keyword, which wins when given. ``keep_op`` and
``intervene`` are the constructor's own ``save=`` and ``intervene=`` and
``postprocess`` is a one-shot enrichment the recorder never runs: setting one
of them on the bundle refuses instead of becoming a silent no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._deprecations import MISSING
from .._errors import ArgumentTypeError

if TYPE_CHECKING:
    from .options import RecordingOptions

#: The ``RecordingOptions`` fields the recorder reads; an unset field stays
#: ``MISSING`` so the recorder config resolves it.
_RECORDING_DOOR_FIELDS: tuple[str, ...] = (
    "default_op",
    "default_module",
    "history_size",
    "include_source_events",
    "max_predicate_failures",
    "on_predicate_error",
    "on_forward_error",
    "default_grad",
    "lean",
)


def _unfold_recording_options(
    recording: RecordingOptions | None, *, api_name: str, lean: Any = MISSING
) -> dict[str, Any]:
    """Read the recorder knobs off ``recording=``.

    Parameters
    ----------
    recording:
        The grouped recording options, or ``None`` for every default.
    api_name:
        The door named in a refusal.
    lean:
        The constructor's ``lean=`` door keyword; when given it wins over
        ``RecordingOptions.lean`` and is validated by the recorder config.

    Returns
    -------
    dict[str, Any]
        One entry per door field: the explicit value, else ``MISSING``.

    Raises
    ------
    ArgumentTypeError
        If a field the recorder has no door for was set explicitly
        (``record_capture_option_unsupported``).
    """

    if recording is None:
        knobs = dict.fromkeys(_RECORDING_DOOR_FIELDS, MISSING)
    else:
        unread = sorted(set(recording._specified_fields) - set(_RECORDING_DOOR_FIELDS))  # noqa: SLF001
        if unread:
            raise ArgumentTypeError(
                f"{api_name} does not read recording= field(s) {', '.join(unread)}: "
                "keep_op and intervene are the constructor's save= and intervene=, and "
                "postprocess is a tl.record enrichment, so they would be a silent no-op",
                code="record_capture_option_unsupported",
                remedy=f"pass the predicate as {api_name}(save=...) and the intervention as "
                "intervene=; run postprocess through tl.record or Recording.enrich",
                argument="recording",
            )
        knobs = {
            name: getattr(recording, name) if recording._is_field_explicit(name) else MISSING  # noqa: SLF001
            for name in _RECORDING_DOOR_FIELDS
        }
    if lean is not MISSING:
        knobs["lean"] = lean
    return knobs
