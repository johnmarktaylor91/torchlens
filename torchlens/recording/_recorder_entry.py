"""Forward-entry helpers for the Recorder (split out of ``_recorder``)."""

from __future__ import annotations

import warnings
from typing import cast

from torch import nn

from .._capture.predicates import _save_selector_matches
from .._capture.projections import RecordingState
from .._deprecations import MissingType
from .._errors import CaptureNotActiveError
from .._utils._torch_compat import get_fsdp_wrapper_type
from ..options import StorageOptions
from ._recorder_streaming import _rank_prefixed_streaming_options


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
