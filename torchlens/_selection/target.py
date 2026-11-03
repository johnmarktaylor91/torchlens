"""Portable target specs: the saved form of a selector plus its slice.

``TargetSpec`` and ``FrozenTargetSpec`` are the persisted spelling of a site
selector; they live below :mod:`torchlens.intervention` so selectors can
build them without importing upward. Their public home stays
``torchlens.intervention.types`` (re-exported there), and the pickled module
path is pinned to that spelling below so saved artifacts and the unpickle
allowlist are unchanged by the source move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["FrozenTargetSpec", "TargetSpec", "TensorSliceSpec"]


@dataclass(frozen=True)
class TensorSliceSpec:
    """Portable tensor slicing metadata for an intervention target."""

    positions: Any | None = None
    heads: Any | None = None
    batch: Any | None = None
    output_index: int | None = None
    position_axis: int | None = None
    head_axis: int | None = None
    query_axis: int | None = None
    key_axis: int | None = None
    feature_axis: int | None = None


def _freeze_value(value: Any) -> Any:
    """Recursively freeze built-in mutable containers.

    Parameters
    ----------
    value:
        Value to freeze.

    Returns
    -------
    Any
        Immutable equivalent for built-in list, dict, and set containers, or
        the original value for opaque objects.
    """

    if isinstance(value, dict):
        return tuple(
            (key, _freeze_value(item))
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_stable(value: Any) -> bool:
    """Return whether ``_freeze_value(value)`` is invariant while ``value`` is held.

    Parameters
    ----------
    value:
        Candidate selector value.

    Returns
    -------
    bool
        False when the value reaches a built-in dict/list/set whose contents
        ``_freeze_value`` snapshots (in-place mutation would stale a cached
        freeze), True for pass-through and tuple-of-stable values.
    """

    if isinstance(value, dict | list | set | frozenset):
        return False
    if isinstance(value, tuple):
        return all(_freeze_stable(item) for item in value)
    return True


@dataclass
class TargetSpec:
    """Mutable internal selector target specification."""

    selector_kind: str
    selector_value: Any | None = None
    strict: bool = False
    slice_spec: TensorSliceSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def freeze(self) -> FrozenTargetSpec:
        """Return an immutable view of this target spec.

        Returns
        -------
        FrozenTargetSpec
            Frozen target spec with shallow-frozen metadata.
        """

        # Dedup scans over spec.targets refreeze both sides per pair, which is
        # quadratic in unique targets. The cache is only stored for specs whose
        # freeze output cannot drift under in-place mutation (empty metadata,
        # snapshot-free selector_value) and is only returned while every field
        # still holds the exact cached value, so a hit is byte-identical to a
        # fresh freeze.
        cached = self.__dict__.get("_tl_frozen_cache")
        if (
            cached is not None
            and not self.metadata
            and cached[0] is self.selector_value
            and cached[1] == self.selector_kind
            and cached[2] is self.strict
            and cached[3] is self.slice_spec
        ):
            return cached[4]
        frozen = FrozenTargetSpec(
            selector_kind=self.selector_kind,
            selector_value=_freeze_value(self.selector_value),
            strict=self.strict,
            slice_spec=self.slice_spec,
            metadata=tuple(
                (key, _freeze_value(value))
                for key, value in sorted(self.metadata.items(), key=lambda pair: repr(pair[0]))
            ),
        )
        if not self.metadata and _freeze_stable(self.selector_value):
            self.__dict__["_tl_frozen_cache"] = (
                self.selector_value,
                self.selector_kind,
                self.strict,
                self.slice_spec,
                frozen,
            )
        return frozen

    def __getstate__(self) -> dict[str, Any]:
        """Return picklable state without the transient freeze cache."""

        state = dict(self.__dict__)
        state.pop("_tl_frozen_cache", None)
        return state


@dataclass(frozen=True)
class FrozenTargetSpec:
    """Immutable public selector target specification."""

    selector_kind: str
    selector_value: Any | None = None
    strict: bool = False
    slice_spec: TensorSliceSpec | None = None
    metadata: tuple[tuple[str, Any], ...] = ()


# Persisted identity: metadata.pkl names these classes by module path and the
# unpickle allowlist admits exactly ("torchlens.intervention.types", <name>).
for _cls in (TensorSliceSpec, TargetSpec, FrozenTargetSpec):
    _cls.__module__ = "torchlens.intervention.types"
del _cls
