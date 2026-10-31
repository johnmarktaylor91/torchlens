"""Parameter arguments that forward validation's own capture keeps by reference.

Forward validation captures with ``save_arg_values=True``, so every tensor
argument of every op used to be copied, parameters included: the first
validation trace held a full extra copy of the weights on the model's device
until validation ended. Validation runs that capture on a deep copy of the
model, so its parameters are validation's own. While that one capture runs,
a plain parameter argument (or a tensor on the same storage as a parameter) is
saved as a detached alias of that storage, sharing its version counter, instead
of a copy, and the version counter of each such tensor and of its parameter is
recorded.

A by-reference value is the parameter's value at call time only if nothing
wrote the parameter afterwards. An in-place write moves the version counter of
the tensor written (a write through ``param.data`` moves only that alias's
counter). So the counters are checked over every referenced tensor at the end
of the capture, and again, together with the parameters' bytes against the
replay copy's pre-capture snapshot, before that copy is restored (the restore
rewrites every parameter whenever any state changed) and before the replay
reads anything. On any change the whole validation reruns with every argument
copied, exactly as before, and the rerun's trace carries a diagnostic saying
so. A user's own ``save_arg_values=True`` trace never sees any of this: outside
the validation capture the copy path is unchanged.

When validation runs with an ``output_device`` other than ``"same"``, the same
capture also keeps its retained copy of every other eligible tensor argument on
that device (one direct copy per argument) and tags the copy with the device the
op consumed it on, so the accelerator holds no saved-argument copies beyond the
parameter aliases. Replay moves each tagged copy back to that device, as an
exact copy, for the one layer being checked (see
:mod:`torchlens.validation._replay_device`). This also applies on the clone-path
rerun; with ``output_device="same"`` nothing is moved or tagged.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import gc
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import Any, cast

import torch
from torch import nn

FALLBACK_CHECK = "validation_parameter_reference_fallback"
CONSUMED_DEVICE_ATTR = "_torchlens_saved_arg_consumed_device"
"""Attribute naming the device an op consumed an offloaded saved-argument copy on."""

_StorageKey = tuple[torch.device, int]


@dataclasses.dataclass(frozen=True)
class _ReferenceSession:
    """One validation call; ``fallback_reason`` is set on its clone-path rerun."""

    fallback_reason: str | None = None


class _ParameterArgumentChanged(Exception):
    """A by-reference parameter argument changed; validation reruns on the clone path."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_SESSION: ContextVar[_ReferenceSession | None] = ContextVar(
    "torchlens_validation_reference_session", default=None
)
_ACTIVE: ContextVar[ParameterArgumentReferences | None] = ContextVar(
    "torchlens_validation_argument_references", default=None
)


def _storage_key(tensor: torch.Tensor) -> _StorageKey | None:
    """Return the storage identity of a plain dense tensor, or ``None`` if not eligible."""

    if type(tensor) not in (torch.Tensor, nn.Parameter) or tensor.layout is not torch.strided:
        return None
    # Inference tensors keep no version counter, so no guard could cover them.
    if tensor.device.type == "meta" or tensor.is_quantized or tensor.is_inference():
        return None
    data_ptr = tensor.untyped_storage().data_ptr()
    return None if data_ptr == 0 else (tensor.device, data_ptr)


class ParameterArgumentReferences:
    """Version guard over the parameter arguments one validation capture kept by reference.

    Parameters
    ----------
    owners:
        Parameters of the model being captured, by storage identity, with their
        names. Empty when references are off (the copy path runs).
    fallback_reason:
        On the clone-path rerun, why the first attempt was abandoned.
    offload_device:
        Where retained copies of non-parameter tensor arguments are kept
        (validation's ``output_device``), or ``None`` to keep them in place.
    """

    def __init__(
        self,
        owners: dict[_StorageKey, list[tuple[str, nn.Parameter]]],
        fallback_reason: str | None = None,
        offload_device: str | None = None,
    ) -> None:
        self._owners = owners
        self._fallback_reason = fallback_reason
        self._offload_device = None if offload_device is None else torch.device(offload_device)
        self._versions: dict[int, tuple[torch.Tensor, int, str]] = {}
        self._owners_seen: dict[int, tuple[str, nn.Parameter]] = {}
        self._first_change: str | None = None

    def keep(self, tensor: torch.Tensor) -> bool:
        """Record ``tensor`` and return ``True`` when it is kept by reference."""

        key = _storage_key(tensor)
        owners = self._owners.get(key) if key is not None else None
        if not owners:
            return False
        name = owners[0][0]
        self._note(tensor, name)
        for owner_name, owner in owners:
            self._note(owner, owner_name)
            self._owners_seen[id(owner)] = (owner_name, owner)
        return True

    def alias(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """Return a by-reference alias of ``tensor``, or ``None`` when it must be copied.

        The alias is a plain detached tensor on the parameter's storage, sharing
        its version counter, with the argument's ``requires_grad``: the flag
        replay reads from a saved argument. Like the copy it replaces, it is a
        leaf. It is never the ``nn.Parameter`` itself, so capture cleanup that
        strips TorchLens metadata from saved arguments cannot reach the
        parameter or its registry entry.
        """

        if not self.keep(tensor):
            return None
        alias = tensor.detach()
        if tensor.requires_grad:
            alias.requires_grad_(True)
        return alias

    @property
    def offloads(self) -> bool:
        """Whether retained copies of non-parameter tensor arguments move off their device."""

        return self._offload_device is not None

    def retention_device(self, tensor: torch.Tensor) -> torch.device | None:
        """Return where the retained copy of ``tensor`` is kept, or ``None`` to keep it in place.

        Only a plain dense tensor that is not on a parameter's storage moves,
        and only to a device of another type (or another explicit index).
        """

        target = self._offload_device
        if target is None or type(tensor) is not torch.Tensor:
            return None
        if tensor.layout is not torch.strided or tensor.device.type == "meta":
            return None
        if tensor.is_quantized or tensor.device == target:
            return None
        if tensor.device.type == target.type and target.index is None:
            return None
        key = _storage_key(tensor)
        return None if key is not None and key in self._owners else target

    def shares_parameter_storage(self, tensor: torch.Tensor) -> bool:
        """Whether ``tensor`` shares storage with a live parameter of validation's copy.

        Read now, from the parameter's current storage, not assumed from how
        the tensor was made. Validation's capture charges such an alias
        nothing: it adds no storage beyond the copy's own parameters.
        """

        key = _storage_key(tensor)
        if key is None:
            return False
        return any(
            owner.device == key[0] and owner.untyped_storage().data_ptr() == key[1]
            for _, owner in self._owners.get(key, ())
        )

    def offload(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """Return the retained copy of ``tensor`` on the offload device, or ``None``.

        The copy is one direct transfer with the argument's autograd semantics
        (as the in-place copy it replaces), tagged with the device the op
        consumed the argument on so replay can move it back exactly.
        """

        target = self.retention_device(tensor)
        if target is None:
            return None
        from .utils.tensor_utils import _clone_tensor_payload

        moved = cast(
            torch.Tensor,
            _clone_tensor_payload(
                tensor, detach_tensor=False, save_mode="copy", target_device=str(target)
            ),
        )
        if moved.device == tensor.device:
            return moved
        setattr(moved, CONSUMED_DEVICE_ATTR, tensor.device)
        return moved

    def _note(self, tensor: torch.Tensor, name: str) -> None:
        """Record a tensor's version the first time; flag a later different version."""

        seen = self._versions.get(id(tensor))
        if seen is None:
            self._versions[id(tensor)] = (tensor, tensor._version, name)
        elif seen[1] != tensor._version and self._first_change is None:
            self._first_change = _describe_change(name, seen[1], tensor._version)

    def first_change(self) -> str | None:
        """Describe the first referenced tensor whose version moved, or ``None``."""

        if self._first_change is not None:
            return self._first_change
        for tensor, version, name in self._versions.values():
            if tensor._version != version:
                return _describe_change(name, version, tensor._version)
        return None

    def first_value_change(self, snapshot: Any) -> str | None:
        """Name the first referenced parameter whose bytes differ from ``snapshot``.

        A write through ``param.data`` moves a counter of its own, not the
        parameter's, so the parameters are also compared with the replay
        copy's pre-capture snapshot, one entry at a time on its own device.
        """

        from ._validation_state_snapshot import _on_source_device
        from .validation._live_model_state import _tensor_bits_equal

        for name, parameter in self._owners_seen.values():
            if name not in snapshot.tensors:
                return f"parameter {name!r} has no pre-capture snapshot to compare"
            if not _tensor_bits_equal(parameter.detach(), _on_source_device(snapshot, name)):
                return f"parameter {name!r} changed value during the forward"
        return None

    def raise_if_changed(self, trace: Any = None, snapshot: Any = None) -> None:
        """Abandon this attempt if a referenced value changed; on the rerun, record why.

        Parameters
        ----------
        trace:
            The validation trace about to be replayed. On the clone-path rerun
            the fallback diagnostic is recorded on it.
        snapshot:
            The replay copy's pre-capture state snapshot. When given, the
            referenced parameters' bytes are compared with it too. Call this
            before the replay copy is restored: the restore rewrites (and so
            moves the version of) every parameter whenever any state changed.
        """

        if self._fallback_reason is not None:
            if trace is not None:
                _record_fallback(trace, self._fallback_reason)
            return
        change = self.first_change()
        if change is None and snapshot is not None:
            change = self.first_value_change(snapshot)
        if change is not None:
            raise _ParameterArgumentChanged(change)


def _describe_change(name: str, before: int, after: int) -> str:
    """Name the parameter whose version counter moved."""

    return f"parameter {name!r} was written in place (version {before} to {after})"


def _record_fallback(trace: Any, reason: str) -> None:
    """Record on the rerun's trace that validation used copied parameter arguments."""

    from .validation.diagnostics import ValidationDiagnostic, record_validation_diagnostic

    record_validation_diagnostic(
        trace,
        ValidationDiagnostic(
            check_name=FALLBACK_CHECK,
            message=(
                "TorchLens validation reran with every argument copied because a "
                f"parameter changed during the forward: {reason}. The verdict "
                "comes from the copied-argument run."
            ),
            extra={"reason": reason},
        ),
    )


def _parameter_owners(model: nn.Module) -> dict[_StorageKey, list[tuple[str, nn.Parameter]]]:
    """Map each eligible parameter storage of ``model`` to its parameters."""

    owners: dict[_StorageKey, list[tuple[str, nn.Parameter]]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        key = _storage_key(parameter)
        if key is None:
            continue
        entries = owners.setdefault(key, [])
        if all(existing is not parameter for _, existing in entries):
            entries.append((name, parameter))
    return owners


def active_argument_references() -> ParameterArgumentReferences | None:
    """Return the guard of the validation capture running in this context, if any."""

    return _ACTIVE.get()


@contextlib.contextmanager
def parameter_argument_references(
    model: Any, model_is_copy: bool, output_device: str = "same"
) -> Iterator[ParameterArgumentReferences]:
    """Keep ``model``'s parameter arguments by reference for the capture in this block.

    References are on only inside a validation call wrapped by
    :func:`with_parameter_reference_fallback`, on its first attempt, and only
    when ``model`` is validation's deep copy. Inside such a call, an
    ``output_device`` other than ``"same"`` also keeps the retained copies of
    other tensor arguments on that device, on every attempt. The context is
    cleared on every exit; on a normal exit the guard runs over every
    referenced tensor.

    Parameters
    ----------
    model:
        The model the capture runs.
    model_is_copy:
        Whether ``model`` is validation's own deep copy (never the live model).
    output_device:
        Validation's ``output_device``.

    Yields
    ------
    ParameterArgumentReferences
        The guard (an empty one when references are off).
    """

    session = _SESSION.get()
    offload = None if session is None or output_device == "same" else output_device
    if session is None or session.fallback_reason is not None or not model_is_copy:
        reason = None if session is None else session.fallback_reason
        if offload is None:
            yield ParameterArgumentReferences({}, reason)
            return
        references = ParameterArgumentReferences({}, reason, offload)
    else:
        references = ParameterArgumentReferences(_parameter_owners(model), None, offload)
    token = _ACTIVE.set(references)
    try:
        yield references
    finally:
        _ACTIVE.reset(token)
    references.raise_if_changed()


def with_parameter_reference_fallback(validate: Callable[..., bool]) -> Callable[..., bool]:
    """Run ``validate`` with parameter references, rerunning on the clone path if one changed.

    Parameters
    ----------
    validate:
        The forward-validation function. It raises the private change signal
        only through :class:`ParameterArgumentReferences`.

    Returns
    -------
    Callable[..., bool]
        The wrapped function, same arguments and verdict.
    """

    @functools.wraps(validate)
    def run(*args: Any, **kwargs: Any) -> bool:
        """Validate with references on; on a change, rerun with every argument copied."""

        token = _SESSION.set(_ReferenceSession())
        try:
            return validate(*args, **kwargs)
        except _ParameterArgumentChanged as changed:
            reason = changed.reason
        finally:
            _SESSION.reset(token)
        # The abandoned attempt's replay copy and trace sit in reference cycles;
        # free them before the rerun makes its own copy.
        gc.collect()
        token = _SESSION.set(_ReferenceSession(fallback_reason=reason))
        try:
            return validate(*args, **kwargs)
        finally:
            _SESSION.reset(token)

    return run
