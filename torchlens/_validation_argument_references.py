"""Parameter arguments that forward validation's own capture keeps by reference.

Forward validation captures with ``save_arg_values=True``, so every tensor
argument of every op used to be copied, parameters included: the first
validation trace held a full extra copy of the weights on the model's device
until validation ended. Validation runs that capture on a deep copy of the
model, so its parameters are validation's own. While that one capture runs,
plain parameter arguments (and tensors on the same storage as a parameter) are
kept by reference instead of copied, and the version counter of each such
tensor and of its parameter is recorded.

A by-reference value is the parameter's value at call time only if nothing
wrote the parameter afterwards. Every in-place write moves a version counter,
and validation's restore of the replay copy writes (and so moves the counter
of) any parameter whose value changed. The counters are checked over every
referenced tensor at the end of the capture and again before the replay reads
anything; if any moved, the whole validation reruns with every argument copied,
exactly as before, and the rerun's trace carries a diagnostic saying so. A
user's own ``save_arg_values=True`` trace never sees any of this: outside the
validation capture the copy path is unchanged.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import Any

import torch
from torch import nn

FALLBACK_CHECK = "validation_parameter_reference_fallback"

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
    if tensor.device.type == "meta" or tensor.is_quantized:
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
    """

    def __init__(
        self,
        owners: dict[_StorageKey, list[tuple[str, nn.Parameter]]],
        fallback_reason: str | None = None,
    ) -> None:
        self._owners = owners
        self._fallback_reason = fallback_reason
        self._versions: dict[int, tuple[torch.Tensor, int, str]] = {}
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
        return True

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

    def raise_if_changed(self, trace: Any = None) -> None:
        """Abandon this attempt if a referenced value changed; on the rerun, record why.

        Parameters
        ----------
        trace:
            The validation trace about to be replayed. On the clone-path rerun
            the fallback diagnostic is recorded on it.
        """

        if self._fallback_reason is not None:
            if trace is not None:
                _record_fallback(trace, self._fallback_reason)
            return
        change = self.first_change()
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
            check=FALLBACK_CHECK,
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
    model: Any, model_is_copy: bool
) -> Iterator[ParameterArgumentReferences]:
    """Keep ``model``'s parameter arguments by reference for the capture in this block.

    References are on only inside a validation call wrapped by
    :func:`with_parameter_reference_fallback`, on its first attempt, and only
    when ``model`` is validation's deep copy. The context is cleared on every
    exit; on a normal exit the guard runs over every referenced tensor.

    Parameters
    ----------
    model:
        The model the capture runs.
    model_is_copy:
        Whether ``model`` is validation's own deep copy (never the live model).

    Yields
    ------
    ParameterArgumentReferences
        The guard (an empty one when references are off).
    """

    session = _SESSION.get()
    if session is None or session.fallback_reason is not None or not model_is_copy:
        reason = None if session is None else session.fallback_reason
        yield ParameterArgumentReferences({}, reason)
        return
    references = ParameterArgumentReferences(_parameter_owners(model))
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
        token = _SESSION.set(_ReferenceSession())
        try:
            return validate(*args, **kwargs)
        except _ParameterArgumentChanged as changed:
            reason = changed.reason
        finally:
            _SESSION.reset(token)
        token = _SESSION.set(_ReferenceSession(fallback_reason=reason))
        try:
            return validate(*args, **kwargs)
        finally:
            _SESSION.reset(token)

    return run
