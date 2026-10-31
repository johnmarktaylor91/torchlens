"""Public value objects for module forward-pre-hook input provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from .._io import FieldPolicy
from .module import HookInfo

TypedInputPath = tuple[Any, ...]
PreHookOutcome = Literal["none", "returned", "invalid_return", "raised"]


@dataclass(frozen=True)
class TensorInputObservation:
    """Metadata and optional payload for one tensor in a module input snapshot.

    Parameters
    ----------
    path:
        Typed path beginning with ``"args"`` or ``"kwargs"``.
    source_op:
        Final producer operation label when one was available.
    object_id:
        In-process object identity. This diagnostic is deliberately non-portable.
    version:
        Captured PyTorch version counter. This diagnostic is deliberately non-portable.
    shape, dtype, device, layout, requires_grad:
        Cheap tensor metadata captured without synchronizing device contents.
    payload:
        Detached snapshot payload when retained.
    payload_status:
        ``"saved"`` or an explicit reason that no payload is available.
    alias_group:
        Snapshot-local identity shared by aliased input paths.
    payload_origin:
        Description of how the retained payload was obtained.
    """

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "path": FieldPolicy.KEEP,
        "source_op": FieldPolicy.KEEP,
        "object_id": FieldPolicy.DROP,
        "version": FieldPolicy.DROP,
        "shape": FieldPolicy.KEEP,
        "dtype": FieldPolicy.KEEP,
        "device": FieldPolicy.KEEP,
        "layout": FieldPolicy.KEEP,
        "requires_grad": FieldPolicy.KEEP,
        "payload": FieldPolicy.BLOB,
        "payload_status": FieldPolicy.KEEP,
        "alias_group": FieldPolicy.KEEP,
        "payload_origin": FieldPolicy.KEEP,
    }

    path: TypedInputPath
    source_op: str | None
    object_id: int | None
    version: int | None
    shape: tuple[int, ...]
    dtype: str
    device: str
    layout: str
    requires_grad: bool
    payload: Any | None
    payload_status: str
    alias_group: int
    payload_origin: str


@dataclass(frozen=True)
class ModuleInputSnapshot:
    """Immutable snapshot of positional and keyword module inputs.

    The public name is provisional pending the TorchLens human naming session.
    ``args`` and ``kwargs`` reconstruct the atomic logical input state, while
    ``tensor_observations`` exposes ordered metadata and payload provenance.

    Parameters
    ----------
    _args, _kwargs:
        Reconstructed detached input values.
    structure:
        Immutable compact description of the input container topology.
    tensor_observations:
        Ordered tensor observations keyed by typed input paths.
    capture_complete:
        Whether all supported mutation evidence was available.
    incomplete_reasons:
        Stable reasons explaining incomplete evidence.
    summary:
        Bounded summary that does not materialize lazy payloads.
    """

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "_args": FieldPolicy.BLOB_RECURSIVE,
        "_kwargs": FieldPolicy.BLOB_RECURSIVE,
        "structure": FieldPolicy.KEEP,
        "tensor_observations": FieldPolicy.KEEP,
        "capture_complete": FieldPolicy.KEEP,
        "incomplete_reasons": FieldPolicy.KEEP,
        "summary": FieldPolicy.KEEP,
    }

    _args: tuple[Any, ...]
    _kwargs: dict[str, Any]
    structure: tuple[Any, ...]
    tensor_observations: tuple[TensorInputObservation, ...]
    capture_complete: bool
    incomplete_reasons: tuple[str, ...]
    summary: str

    @property
    def args(self) -> tuple[Any, ...]:
        """Return reconstructed positional inputs."""

        return self._args

    @property
    def kwargs(self) -> dict[str, Any]:
        """Return a shallow copy of reconstructed keyword inputs."""

        return dict(self._kwargs)


@dataclass(frozen=True)
class PreHookEffect:
    """Immutable record of one executed user forward pre-hook transition.

    The public name is provisional pending the TorchLens human naming session.

    Parameters
    ----------
    ordinal:
        One-based execution ordinal shared by global and module-local hooks.
    scope:
        ``"global"`` or ``"module"``.
    registration_id:
        Diagnostic PyTorch registry identifier.
    hook:
        Portable descriptive hook metadata.
    with_kwargs:
        Whether PyTorch invoked the hook with keyword arguments.
    outcome:
        Normalized execution outcome.
    changed:
        ``True``/``False`` when mutation evidence is complete, otherwise ``None``.
    change_kinds, changed_paths:
        Compact attribution for the observed transition.
    before_fingerprint, after_fingerprint:
        Metadata-only fingerprints that never materialize payload blobs.
    incomplete_reasons:
        Stable reasons for a tri-state or attribution downgrade.
    exception_type, exception_message:
        String-only failure metadata for interrupted captures.
    """

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "ordinal": FieldPolicy.KEEP,
        "scope": FieldPolicy.KEEP,
        "registration_id": FieldPolicy.KEEP,
        "hook": FieldPolicy.KEEP,
        "with_kwargs": FieldPolicy.KEEP,
        "outcome": FieldPolicy.KEEP,
        "changed": FieldPolicy.KEEP,
        "change_kinds": FieldPolicy.KEEP,
        "changed_paths": FieldPolicy.KEEP,
        "before_fingerprint": FieldPolicy.KEEP,
        "after_fingerprint": FieldPolicy.KEEP,
        "incomplete_reasons": FieldPolicy.KEEP,
        "exception_type": FieldPolicy.KEEP,
        "exception_message": FieldPolicy.KEEP,
    }

    ordinal: int
    scope: Literal["global", "module"]
    registration_id: int
    hook: HookInfo
    with_kwargs: bool
    outcome: PreHookOutcome
    changed: bool | None
    change_kinds: tuple[str, ...]
    changed_paths: tuple[TypedInputPath, ...]
    before_fingerprint: tuple[Any, ...]
    after_fingerprint: tuple[Any, ...]
    incomplete_reasons: tuple[str, ...] = ()
    exception_type: str | None = None
    exception_message: str | None = None
