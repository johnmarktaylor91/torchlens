"""Module-boundary adoption records and the pre-forward tensor ownership snapshot.

A module boundary that meets an untagged tensor adopts it as an internal source.
:func:`record_module_boundary_adoption` decides whether that adoption is an escape
(postprocess raises the provenance warning and the rescue signal) or a
pre-forward closure/forward-global tensor (disclosed and persisted as a
``source_provenance`` gap, no escape signal), from the snapshot
:func:`collect_pre_forward_tensor_ids` takes before the forward runs.
"""

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from ...data_classes.trace import Trace


def collect_pre_forward_tensor_ids(
    model: nn.Module,
) -> tuple[dict[int, torch.Tensor], frozenset[int]]:
    """Return the pinned pre-forward snapshot and its forward-callable-only subset.

    Module ``__dict__`` object graphs are walked first, every forward callable's
    defaults, closure cells and referenced globals second, through ONE shared
    ``seen`` set, so the second subset holds exactly the tensors reachable ONLY
    from a forward callable (a closure or module-global tensor). Those are not
    model-held sources: the held-tensor scan never stamps them and the per-op
    ``unattributed_tensor_args`` witness flags them, so the module-boundary
    adoption record must still persist a ``source_provenance`` gap for them.

    Parameters
    ----------
    model
        The prepared root model.

    Returns
    -------
    tuple[dict[int, torch.Tensor], frozenset[int]]
        The pinned ``id -> tensor`` snapshot (see
        :func:`model_prep._collect_model_owned_tensor_ids`) and the ids reachable only from
        forward callables.
    """

    from .model_prep import (
        _clear_callable_session_tensor_metadata,
        _clear_session_tensor_metadata,
    )

    owned: dict[int, torch.Tensor] = {}
    callable_only: set[int] = set()
    seen: set[int] = set()

    def _note(tensor: torch.Tensor) -> None:
        """Record and pin one reachable tensor under its object id."""

        owned[id(tensor)] = tensor

    def _note_callable(tensor: torch.Tensor) -> None:
        """Record one tensor first reached through a forward callable."""

        if id(tensor) not in owned:
            callable_only.add(id(tensor))
        owned[id(tensor)] = tensor

    submodules = list(model.modules())
    for submodule in submodules:
        for attr_val in submodule.__dict__.values():
            _clear_session_tensor_metadata(attr_val, seen, visit=_note)
    for submodule in submodules:
        _clear_callable_session_tensor_metadata(
            getattr(submodule, "forward", None), seen, visit=_note_callable
        )
    return owned, frozenset(callable_only)


def record_module_boundary_adoption(
    trace: "Trace", t: torch.Tensor, label: str | None, boundary: str, module_address: str
) -> None:
    """Record an untagged tensor adopted as an internal source at a module boundary.

    R16: adoption must not LAUNDER an escape. A stale pre-wrap torch reference
    leaves an untagged output; when a module CONSUMES it (``entry``) or RETURNS
    it (``exit``; transformers' ``GELUActivation`` holds ``F.gelu`` and returns
    ``self.act(x)``), no wrapped op ever sees an unattributed argument, so the
    op vanished with no warning and no rescue. The record makes postprocess
    raise the same provenance warning and escape signal the function path
    raises. Disclosed transform/dynamo regions legitimately produce untagged
    tensors, and a tensor in the PRE-FORWARD ownership snapshot existed before
    the forward (a nested cache or forward-global whose stale labels the
    previous session cleared); neither is an escape.

    Not being an escape is not provenance, though: a snapshot tensor reachable
    ONLY from a forward callable (a closure cell or module-global tensor a
    module consumes or returns directly) is no model-held source, exactly as
    the per-op ``unattributed_tensor_args`` witness treats it. It is recorded on
    ``_module_boundary_outside_sources``, which postprocess discloses and
    persists as a ``source_provenance`` gap without raising the escape signal.
    """

    if (
        label is None
        or getattr(trace, "_raw_transform_escape_detected", False)
        or getattr(trace, "_raw_dynamo_region_detected", False)
    ):
        return
    build_data = trace._module_capture_ws.module_build_data
    record = (str(label), boundary, str(module_address))
    if id(t) not in (build_data.get("model_owned_tensor_ids_at_entry") or ()):
        trace.__dict__.setdefault("_module_boundary_adoptions", []).append(record)
    elif id(t) in (build_data.get("forward_outside_tensor_ids_at_entry") or ()):
        trace.__dict__.setdefault("_module_boundary_outside_sources", []).append(record)
