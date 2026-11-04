"""Validation helpers for public fastlog APIs."""

from __future__ import annotations

from typing import Final

from ._lean import validate_lean_config
from .exceptions import RecordingConfigError
from .options import _RecordingConfig
from .types import CaptureSpec

_POSTPROCESS_PRESETS: Final[set[str]] = {
    "none",
    "module_path_strings",
    "all-feasible",
}


def validate_recording_options(
    options: _RecordingConfig, *, backward_requested: bool = False
) -> None:
    """Validate fastlog options that depend on combined public settings.

    Parameters
    ----------
    options:
        Fully merged fastlog recording options.
    backward_requested:
        Whether gradient hooks were asked for; the lean journal refuses them.

    Raises
    ------
    RecordingConfigError
        If no event can be selected, disk-only storage conflicts with a static
        ``save_grad=True`` default, or ``lean=True`` meets an option that reads
        the events it drops (``record_lean_unsupported``).
    """

    _validate_non_empty_capture(options)
    _validate_disk_only_keep_grad_defaults(options)
    validate_lean_config(options, backward_requested=backward_requested)


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
            "postprocess must be 'none', 'module_path_strings', or 'all-feasible'"
        )


def _validate_non_empty_capture(options: _RecordingConfig) -> None:
    """Reject statically empty fastlog configurations.

    An armed ``narrate=`` narrator is a legal capture on its own ("narrate
    everything, keep nothing" is the narrate-only fast path, snoop D1): the
    echo term keeps ``tl.record(model, x, narrate=...)`` from being refused as
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


def _validate_disk_only_keep_grad_defaults(options: _RecordingConfig) -> None:
    """Reject static save_grad defaults for disk-only storage."""

    if (
        options.streaming is None
        or options.streaming.path is None
        or options.streaming.retain_in_memory
    ):
        return
    for name, default in (
        ("default_op", options.default_op),
        ("default_module", options.default_module),
    ):
        if isinstance(default, CaptureSpec) and default.save_grad:
            raise RecordingConfigError(
                f"{name} cannot use save_grad=True with disk-only fastlog storage"
            )


def refuse_episode_on_record(episode: object) -> None:
    """Refuse ``episode=`` on the record door with the typed teaching refusal.

    Episode capture (the per-step ledger, the step-qualified selectors and
    the coupled-replay rules) lives on ``tl.trace``; the sparse recorder has
    no step ledger to attach it to, so the keyword is accepted only to
    refuse it typed instead of with a bare ``TypeError``.

    Parameters
    ----------
    episode:
        The ``episode=`` value as passed; ``None`` means not given.

    Raises
    ------
    RecordingConfigError
        ``record_episode_unsupported`` when an episode declaration was given.
    """

    if episode is None:
        return
    raise RecordingConfigError(
        "tl.record() does not take episode=: episode capture (the step ledger, "
        "at_step selectors and coupled replay) runs on the exhaustive door. "
        "Remedy: tl.trace(model, x, episode=...) with the same declaration, "
        "and save= to keep the capture sparse.",
        code="record_episode_unsupported",
        remedy="use tl.trace(model, x, episode=..., save=...)",
    )
