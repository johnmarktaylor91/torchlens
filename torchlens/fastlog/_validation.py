"""Validation helpers for public fastlog APIs."""

from __future__ import annotations

from typing import Final

from .exceptions import RecordingConfigError
from .options import RecordingOptions
from .types import CaptureSpec

_POSTPROCESS_PRESETS: Final[set[str]] = {
    "none",
    "module_path_strings",
    "param_addresses",
    "all-feasible",
}


def validate_recording_options(options: RecordingOptions) -> None:
    """Validate fastlog options that depend on combined public settings.

    Parameters
    ----------
    options:
        Fully merged fastlog recording options.

    Raises
    ------
    RecordingConfigError
        If no event can be selected or disk-only storage conflicts with a
        static ``keep_grad=True`` default.
    """

    _validate_non_empty_capture(options)
    _validate_disk_only_keep_grad_defaults(options)


def validate_postprocess(postprocess: str) -> None:
    """Validate a public postprocess preset name.

    Parameters
    ----------
    postprocess:
        Requested postprocess preset.

    Raises
    ------
    RecordingConfigError
        If the preset is not part of the public contract.
    """

    if postprocess not in _POSTPROCESS_PRESETS:
        raise RecordingConfigError(
            "postprocess must be 'none', 'module_path_strings', "
            "'param_addresses', or 'all-feasible'"
        )


def _validate_non_empty_capture(options: RecordingOptions) -> None:
    """Reject statically empty fastlog configurations.

    An armed ``echo=`` narrator is a legal capture on its own ("narrate
    everything, keep nothing" is the narrate-only fast path, snoop D1): the
    echo term keeps ``tl.record(model, x, echo=...)`` from being refused as
    capturing nothing.
    """

    if (
        options.keep_op is None
        and options.halt is None
        and options.default_op is False
        and options.default_module is False
        and options.echo is None
    ):
        # User-vocabulary refusal (R65): the caller typed tl.record(), not
        # "fastlog", and the remedy is one token.
        raise RecordingConfigError(
            "tl.record() would capture nothing: no save= predicate was given and "
            "both default_op and default_module are False. Remedy: pass a "
            "predicate such as save=tl.func('relu'), or set default_op=True."
        )


def _validate_disk_only_keep_grad_defaults(options: RecordingOptions) -> None:
    """Reject static keep_grad defaults for disk-only storage."""

    if (
        options.streaming is None
        or options.streaming.bundle_path is None
        or options.streaming.retain_in_memory
    ):
        return
    for name, default in (
        ("default_op", options.default_op),
        ("default_module", options.default_module),
    ):
        if isinstance(default, CaptureSpec) and default.keep_grad:
            raise RecordingConfigError(
                f"{name} cannot use keep_grad=True with disk-only fastlog storage"
            )
