"""Validation and entry helpers for public fastlog APIs."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Final, cast

from torch import nn

from .._capture.predicates import _save_selector_matches
from .._deprecations import MissingType
from .._errors import CaptureNotActiveError
from .._utils._torch_compat import get_fsdp_wrapper_type
from ..options import StorageOptions
from ._recorder_streaming import _rank_prefixed_streaming_options
from .exceptions import RecordingConfigError
from .options import _RecordingConfig
from .types import CaptureSpec

if TYPE_CHECKING:
    from .._capture.projections import RecordingState

_POSTPROCESS_PRESETS: Final[set[str]] = {
    "none",
    "module_path_strings",
    "all-feasible",
}


def validate_recording_options(options: _RecordingConfig) -> None:
    """Validate fastlog options that depend on combined public settings.

    Parameters
    ----------
    options:
        Fully merged fastlog recording options.

    Raises
    ------
    RecordingConfigError
        If no event can be selected or disk-only storage conflicts with a
        static ``save_grad=True`` default.
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


def _warn_zero_match_capture_selectors(state: RecordingState) -> None:
    """Warn when sparse capture selectors matched no sites.

    Parameters
    ----------
    state:
        Completed recording state carrying all retained records and fire counts.

    Returns
    -------
    None
        Emits at most one warning for each configured selector slot.
    """

    from .._selection.selectors import BaseSelector
    from ..intervention.hooks import site_universe

    save_selector = state.options.keep_op
    if isinstance(save_selector, BaseSelector):
        save_matched = any(
            _save_selector_matches(save_selector, record.ctx) for record in state.recording.records
        )
        if not save_matched:
            warnings.warn(
                f"Capture-time save selector {save_selector!r} matched zero sites; "
                "no activations were selected by it.",
                UserWarning,
                stacklevel=3,
            )
    intervene_selector = getattr(state.options.intervene, "selector", None)
    if (
        isinstance(intervene_selector, BaseSelector)
        and site_universe(intervene_selector) == "activation"
        and state.intervene_selector_fire_count == 0
    ):
        warnings.warn(
            f"Capture-time intervention selector {intervene_selector!r} matched zero sites; "
            "no intervention fired.",
            UserWarning,
            stacklevel=3,
        )


def _unwrap_ddp_for_fastlog(
    model: nn.Module,
    streaming: StorageOptions | None | MissingType,
) -> tuple[nn.Module, StorageOptions | None | MissingType]:
    """Unwrap DDP/DataParallel wrappers for rank-local fastlog capture.

    Parameters
    ----------
    model:
        Candidate model supplied to fastlog.
    streaming:
        Caller-supplied streaming options.

    Returns
    -------
    tuple[nn.Module, StorageOptions | None | MissingType]
        The model to execute and possibly rewritten streaming options.
    """

    # The lazy probe never imports torch.distributed.fsdp on plain captures.
    fsdp_wrapper_type = get_fsdp_wrapper_type()
    if fsdp_wrapper_type is not None and isinstance(model, fsdp_wrapper_type):
        raise CaptureNotActiveError(
            "torchlens.recording does not support FullyShardedDataParallel (FSDP): "
            "parameters are sharded across ranks and there is no unsharded module to log",
            code="fsdp_capture_unsupported",
            remedy="record the unsharded module before FSDP wrapping",
        )

    try:
        from torch.nn.parallel import DistributedDataParallel
    except ImportError:
        distributed_data_parallel: type[nn.Module] | None = None
    else:
        distributed_data_parallel = DistributedDataParallel

    if distributed_data_parallel is not None and isinstance(model, distributed_data_parallel):
        return cast(nn.Module, model.module), _rank_prefixed_streaming_options(streaming)
    if isinstance(model, nn.DataParallel):
        return cast(nn.Module, model.module), _rank_prefixed_streaming_options(streaming)
    return model, streaming
