"""The capture door shared by ``tl.record`` and ``Recorder``.

Both read the capture-time knobs (lookback, seed, save transforms, gradient
retention) off a ``tl.options.CaptureOptions`` instead of flat keywords; the
helpers here unfold that object into the values the recorder config resolves.
"""

from __future__ import annotations

from typing import Any

from .._deprecations import MISSING
from .._errors import ArgumentTypeError
from ..options import CaptureOptions

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


def _unfold_capture_options(
    capture: CaptureOptions | None, *, api_name: str = "tl.record"
) -> dict[str, Any]:
    """Read the capture-door knobs off ``capture=``.

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
                f"{api_name} does not read CaptureOptions field(s) {', '.join(unread)}: the "
                "recorder has no door for them, so they would be a silent no-op",
                code="record_capture_option_unsupported",
                remedy=(
                    f"drop the field(s) from capture= for {api_name}, or use tl.trace, which "
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


def _capture_without_seed(capture: CaptureOptions | None) -> CaptureOptions | None:
    """Return ``capture`` with its ``seed`` unset (the generation path seeds once itself)."""

    if capture is None or not capture._is_field_explicit("seed"):  # noqa: SLF001
        return capture
    values = capture.as_dict()
    values["seed"] = None
    return CaptureOptions._from_values(  # noqa: SLF001
        values,
        capture._specified_fields - {"seed"},  # noqa: SLF001
        _layers_to_save=capture._layers_to_save,  # noqa: SLF001
        _layers_to_save_explicit=capture._layers_to_save_explicit,  # noqa: SLF001
    )
