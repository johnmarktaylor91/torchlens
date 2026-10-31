"""Typed errors for cross-rank merged traces.

Every raise carries ``fields["code"]`` (a :class:`MergedErrorCode` value) and,
where findings exist, ``fields["findings"]`` -- callers branch on structure,
never message text (design-merge-ranks-c v5, P4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors._base import CompatibilityError, TorchLensError
from ._enums import MergedErrorCode

__all__ = [
    "MergeConflictError",
    "MergeInputError",
    "MergedArtifactError",
    "MergedFinding",
    "MergedSurfaceUnsupportedError",
    "MergedTraceError",
]


@dataclass(frozen=True)
class MergedFinding:
    """One structured merge finding.

    Parameters
    ----------
    kind:
        A value from ``MERGE_FINDING_KINDS`` (contract-doc frozen).
    detail:
        Human-readable explanation; diagnostics may sharpen it but are never
        authority.
    membership_digest:
        The membership the finding concerns, when membership-scoped.
    key:
        The merge-side correlation key ``(membership_digest,
        lifetime_ordinal, channel, seq_delta)`` the finding concerns, when
        key-scoped.
    ranks:
        Global ranks the finding cites.
    """

    kind: str
    detail: str
    membership_digest: str | None = None
    key: tuple[str, int, str, int] | None = None
    ranks: tuple[int, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical JSON-serializable projection."""

        return {
            "kind": self.kind,
            "detail": self.detail,
            "membership_digest": self.membership_digest,
            "key": None if self.key is None else list(self.key),
            "ranks": list(self.ranks),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> MergedFinding:
        """Rebuild a finding from :meth:`to_payload` output."""

        raw_key = payload.get("key")
        key = None
        if raw_key is not None:
            key = (str(raw_key[0]), int(raw_key[1]), str(raw_key[2]), int(raw_key[3]))
        return cls(
            kind=str(payload["kind"]),
            detail=str(payload["detail"]),
            membership_digest=payload.get("membership_digest"),
            key=key,
            ranks=tuple(int(rank) for rank in payload.get("ranks", ())),
        )


class MergedTraceError(TorchLensError):
    """Base class for merged-trace errors; ``fields["code"]`` is frozen."""

    def __init__(
        self,
        message: str,
        *,
        code: MergedErrorCode,
        findings: tuple[MergedFinding, ...] = (),
        **payload: Any,
    ) -> None:
        super().__init__(
            message,
            code=code.value,
            findings=findings,
            **payload,
        )


class MergeInputError(MergedTraceError, ValueError):
    """A merge input is unusable: not a rank capture, duplicate rank,
    missing distributed evidence, or an out-of-scope (C1) boundary."""


class MergeConflictError(MergedTraceError, RuntimeError):
    """The presented rank cores structurally contradict each other.

    Raised whenever any structural finding exists -- pre-join lineage audit
    conflicts, relation violations, order contradictions, or correlation
    cross-check disagreements. A conflicted merge never constructs; the
    graph-free diagnostic escape hatch is ``tl.merge_report(...)``.
    """


class MergedArtifactError(MergedTraceError, CompatibilityError):
    """Merged artifact save/load failure: schema, integrity, or tamper."""


class MergedSurfaceUnsupportedError(MergedTraceError, RuntimeError):
    """A refused surface on a merged trace (run/export/validate/selectors)."""
