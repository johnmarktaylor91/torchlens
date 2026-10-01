"""Shared detection and enumeration of un-materialized lazy module state.

Hoisted from ``torchlens.debug._infer_input_shape`` (quickstart memo wave 1a):
the same scan now serves shape inference AND the capture entry gate, and the
enumeration returns the pending set by ``id()`` -- for parameters, buffers,
AND modules -- which is exactly the structure the future lazy completion unit
consumes. Detection never probes a lazy module (probing would permanently
materialize it inside the caller's model, often at a degenerate width), and
enumeration reads only isinstance facts, never ``.shape``/``.size()`` -- a
pending parameter raises on shape access and silently answers
``torch.Size([0])`` to ``.size()``, which is why tolerating pending state by
exception class is forbidden (isinstance before the access, always).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from torch import nn

__all__ = [
    "PendingLazyState",
    "has_uninitialized_lazy_state",
    "pending_lazy_state",
    "raise_if_pending_state_slots",
]

_UNINITIALIZED_TENSOR_TYPES = (
    nn.parameter.UninitializedParameter,
    nn.parameter.UninitializedBuffer,
)


@dataclass(frozen=True)
class PendingLazyState:
    """Enumeration of a model's un-materialized lazy state, keyed by ``id()``.

    Materialization swaps ``__class__`` in place on the same object (identity
    preserved), so ``id()`` is the one key that survives the transition --
    the completion unit's required work-list key.
    """

    #: ``(module_address, module_type_name, id(module))`` per pending module.
    modules: tuple[tuple[str, str, int], ...] = field(default_factory=tuple)
    #: ``(qualified_name, id(parameter))`` per pending parameter.
    parameters: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    #: ``(qualified_name, id(buffer))`` per pending buffer.
    buffers: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        """Return whether any pending lazy state was found."""

        return bool(self.modules or self.parameters or self.buffers)


def has_uninitialized_lazy_state(model: nn.Module) -> bool:
    """Return whether the model still carries un-materialized lazy state.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    bool
        Whether any module has uninitialized lazy parameters or buffers.
    """

    for module in model.modules():
        if isinstance(module, nn.modules.lazy.LazyModuleMixin):
            try:
                if module.has_uninitialized_params():
                    return True
            except Exception:  # noqa: BLE001 - treat unreadable lazy state as uninitialized.
                return True
    return any(
        isinstance(tensor, _UNINITIALIZED_TENSOR_TYPES)
        for tensor in list(model.parameters(recurse=True)) + list(model.buffers(recurse=True))
    )


def pending_lazy_state(model: nn.Module) -> PendingLazyState:
    """Enumerate the model's pending lazy modules, parameters, and buffers.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    PendingLazyState
        The pending set by name and ``id()``; falsy when nothing is pending.
    """

    pending_modules: list[tuple[str, str, int]] = []
    for address, module in model.named_modules():
        if isinstance(module, nn.modules.lazy.LazyModuleMixin):
            try:
                # torch types the method against its private _LazyProtocol
                # self, which mypy rejects for the mixin-narrowed type.
                is_pending = bool(module.has_uninitialized_params())  # type: ignore[misc]
            except Exception:  # noqa: BLE001 - treat unreadable lazy state as uninitialized.
                is_pending = True
            if is_pending:
                pending_modules.append((address, type(module).__name__, id(module)))
    pending_parameters = [
        (name, id(parameter))
        for name, parameter in model.named_parameters(recurse=True)
        if isinstance(parameter, _UNINITIALIZED_TENSOR_TYPES)
    ]
    pending_buffers = [
        (name, id(buffer))
        for name, buffer in model.named_buffers(recurse=True)
        if isinstance(buffer, _UNINITIALIZED_TENSOR_TYPES)
    ]
    return PendingLazyState(
        modules=tuple(pending_modules),
        parameters=tuple(pending_parameters),
        buffers=tuple(pending_buffers),
    )


def raise_if_pending_state_slots(state: Mapping[str, object]) -> None:
    """Refuse typed when a capture-boundary state mapping holds pending slots.

    Quickstart memo 4.5: a pending parameter has no bytes at the capture
    boundary, so no replayable byte baseline can exist for it -- and an
    ``UninitializedParameter`` IS a ``torch.Tensor``, so it sails past
    tensor-only mapping checks and historically exploded inside the baseline
    clone. The one forbidden outcome is an armed capture whose byte witness
    silently cannot see the materialized state; this fires before any
    mutation.

    Parameters
    ----------
    state:
        The ``state_dict()`` mapping about to be cloned as the baseline.

    Raises
    ------
    torchlens._errors.CaptureContextError
        Typed ``state_baseline_unavailable`` naming the pending slots on
        ``exc.fields["pending_slots"]``.
    """

    pending_slots = [
        name for name, value in state.items() if isinstance(value, _UNINITIALIZED_TENSOR_TYPES)
    ]
    if not pending_slots:
        return
    from .._errors import CaptureContextError

    raise CaptureContextError(
        "The capture-boundary state baseline cannot be taken: "
        f"{len(pending_slots)} state slot(s) are un-materialized lazy "
        f"parameters/buffers with no bytes yet (first: {pending_slots[0]!r}; "
        "full list on exc.fields['pending_slots'])",
        code="state_baseline_unavailable",
        remedy=(
            "materialize the lazy modules with one real forward pass outside "
            "capture -- `with torch.no_grad(): model(x)` -- then retry"
        ),
        pending_slots=tuple(pending_slots),
    )
