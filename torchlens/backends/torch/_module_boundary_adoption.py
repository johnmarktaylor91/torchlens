"""Module-boundary adoption records and the pre-forward tensor ownership snapshot.

A module boundary that meets an untagged tensor adopts it as an internal source.
:func:`record_module_boundary_adoption` decides whether that adoption is an escape
(postprocess raises the provenance warning and the rescue signal) or a
pre-forward closure/forward-global tensor (disclosed and persisted as a
``source_provenance`` gap, no escape signal), from the snapshot
:func:`collect_pre_forward_tensor_ids` takes before the forward runs.
"""

import weakref
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ..._utils._model_walk import model_modules
from ..._utils._tensor_label import (
    get_buffer_address,
    get_module_meta,
    get_tensor_label,
    promote_label_to_buffer_source_and_clear_label,
)
from ._held_late_index import ROOT_MODEL_KEY
from .buffer_writes import register_session_buffer_stamp

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


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
        :func:`_collect_model_owned_tensor_ids`) and the ids reachable only from
        forward callables.
    """

    from .model_prep import (
        _clear_module_attribute_tensor_metadata,
        _clear_module_callables_tensor_metadata,
        method_index,
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

    submodules = list(model_modules(model))
    for submodule in submodules:
        _clear_module_attribute_tensor_metadata(submodule, seen, visit=_note)
    methods = method_index(type(submodule) for submodule in submodules)
    swept: dict[int, Any] = {}
    for submodule in submodules:
        _clear_module_callables_tensor_metadata(
            submodule, seen, visit=_note_callable, methods=methods, swept=swept
        )
    return owned, frozenset(callable_only)


def pre_forward_snapshot_build_data(model: nn.Module) -> dict[str, Any]:
    """Return the ``module_build_data`` entries of the pre-forward ownership snapshot.

    Parameters
    ----------
    model
        The prepared root model.

    Returns
    -------
    dict[str, Any]
        The pinned ``id -> tensor`` snapshot, the forward-callable-only id subset,
        and a weak reference to ``model`` from which
        ``_held_late_index`` derives held addresses for snapshot tensors the
        preparation scan did not stamp.
    """

    owned, callable_only = collect_pre_forward_tensor_ids(model)
    return {
        "model_owned_tensor_ids_at_entry": owned,
        "forward_outside_tensor_ids_at_entry": callable_only,
        ROOT_MODEL_KEY: weakref.ref(model),
    }


def is_forward_closure_constant(trace: "Trace", t: torch.Tensor) -> bool:
    """Return whether ``t`` is a pre-forward tensor reachable only from a forward callable.

    True only for the exact object the pre-forward snapshot pinned (identity,
    not a recycled ``id``) whose id is in the forward-callable-only subset
    (closure cells, defaults, referenced globals; never model-held state), and
    only while it carries no TorchLens label. Disclosed transform and dynamo
    regions are excluded: their untagged tensors are not proven pre-forward.
    """

    if (
        trace._raw_transform_escape_detected
        or trace._raw_dynamo_region_detected
        or isinstance(t, nn.Parameter)
        or get_tensor_label(t) is not None
    ):
        return False
    build_data = trace._module_capture_ws.module_build_data
    if id(t) not in (build_data.get("forward_outside_tensor_ids_at_entry") or ()):
        return False
    return (build_data.get("model_owned_tensor_ids_at_entry") or {}).get(id(t)) is t


def _collect_model_owned_tensor_ids(model: nn.Module) -> dict[int, torch.Tensor]:
    """Snapshot tensors reachable from the model's PRE-FORWARD state, PINNED.

    Walks exactly the surfaces the session-end clear walks -- submodule
    ``__dict__`` object graphs plus each forward callable's defaults, keyword
    defaults, closure cells, and referenced globals -- via the shared ``visit``
    traversal, so a tensor the previous session's cleanup could reach (a
    nested model-owned cache, a forward-global mask) is recognized as
    model-owned by the next capture. The R16 module-entry adoption disclosure
    consults this snapshot: a pre-forward model-owned tensor is a KNOWN
    internal source (its stale labels were legitimately cleared between
    sessions), while a tensor first appearing MID-forward is absent from the
    snapshot and keeps the escape disclosure.

    The mapping VALUES are strong references, deliberately: a bare ``set[int]``
    of recyclable ids had no liveness pinning, so a model that dropped a
    snapshotted cache tensor mid-forward freed the object and a later
    stale-pre-wrap escape product could reuse the exact id -- classified
    "model-owned known source", silently suppressing the adoption disclosure (a
    laundering path the strong references close). Pinning every snapshot member
    for the session makes id reuse impossible; the workspace drop at the
    transient-state cleanup seam releases the pins.

    Parameters
    ----------
    model
        The prepared root model.

    Returns
    -------
    dict[int, torch.Tensor]
        ``id() -> tensor`` for every reachable non-Parameter tensor. Consumers
        test membership (``id(t) in snapshot``), identical to the historical
        set semantics; the values exist only to pin the ids.
    """

    owned, _ = collect_pre_forward_tensor_ids(model)
    return owned


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


def _tag_untagged_buffers(trace: "Trace", module: nn.Module) -> bool:
    """Tag any buffers that lack ``_tl.address`` metadata.

    Called during ``_record_module_entry_metadata`` to catch buffers that were created
    dynamically (e.g. in ``forward()``) after the initial ``prepare_buffer_tensors``
    scan. If a buffer already has ``_tl.label_raw`` from being logged as
    an intermediate tensor, that label is moved to ``_tl.buffer_source`` and cleared
    so the buffer gets a fresh source-tensor entry on next use.

    Dynamically stamped buffers join ``trace._session_buffer_inventory`` so the
    inventory-driven cleanup clears them even if they are popped from
    ``_buffers`` later in the same forward. The stamp routes through
    ``register_session_buffer_stamp`` so it also joins the session identity
    registry consulted by the buffer-rung storage-identity belt.

    Returns whether any buffer was stamped. A module with no submodules and no
    buffers has nothing to walk.
    """

    if not module._modules and not module._buffers:
        return False
    stamped = False
    for buffer_name, buffer_tensor in module.named_buffers():
        if get_buffer_address(buffer_tensor) is not None:
            continue
        meta = get_module_meta(module)
        module_addr = "" if meta is None or meta.address is None else meta.address
        if module_addr == "":
            address = buffer_name
        else:
            address = f"{module_addr}.{buffer_name}"
        register_session_buffer_stamp(trace, buffer_tensor, address)
        # If this buffer was already logged as an intermediate tensor, save the
        # previous label as parent and reset so it gets a proper buffer source entry.
        promote_label_to_buffer_source_and_clear_label(buffer_tensor)
        stamped = True
    return stamped
