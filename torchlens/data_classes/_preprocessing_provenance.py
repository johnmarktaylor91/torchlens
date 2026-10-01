"""Preprocessing provenance derivations that live WITH the record (tvscope).

:class:`~torchlens.data_classes.trace.ResolvedPreprocessing` (the persisted
authority-provenance record) lives in ``torchlens.data_classes.trace``; the
closed authority-standing vocabulary and its derivation are pure reads over
that record's own fields, so they live at the record's layer. The L5 package
``torchlens.preprocessing`` re-exports them as the public spellings
(``torchlens.preprocessing.status_of`` and the ``STATUS_*`` constants). The
B1 capture-entry stamp rides along: it is the ONE non-bridge site that mints
a record, and minting belongs beside the mint's vocabulary.

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from .trace import ResolvedPreprocessing, Trace

#: Closed authority-standing vocabulary (tvscope memo D4).
STATUS_AUTHORITATIVE = "authoritative"
STATUS_UNVERIFIED_FALLBACK = "unverified_fallback"
STATUS_UNKNOWN = "unknown"

#: Resolver sources that are TorchLens-authored recipes, never authorities
#: (tvscope memo D9: the tier-4 ImageNet fallback is demoted -- never the
#: reference for a verdict, never ``verified``).
_FALLBACK_SOURCES = frozenset({"imagenet_default"})


def status_of(record: ResolvedPreprocessing) -> str:
    """Derive the closed authority-standing status from a provenance record.

    The mapping is total over every record TorchLens mints (tvscope memo D4):

    - ``verified=True`` (metadata read off the user's own model object) or an
      explicitly supplied authority (``resolution_method="explicit"`` in the
      config) is ``"authoritative"``.
    - The TorchLens-authored ImageNet default is ``"unverified_fallback"``
      (memo D9: demoted -- never an authority, never a verdict reference).
    - Everything else -- nothing resolved, opaque user transforms, raw user
      tensors -- is ``"unknown"``.

    Parameters
    ----------
    record:
        Any ``ResolvedPreprocessing`` record.

    Returns
    -------
    str
        One of :data:`STATUS_AUTHORITATIVE`, :data:`STATUS_UNVERIFIED_FALLBACK`,
        :data:`STATUS_UNKNOWN`.
    """

    if record.source in _FALLBACK_SOURCES:
        return STATUS_UNVERIFIED_FALLBACK
    if record.verified:
        return STATUS_AUTHORITATIVE
    config = record.config if isinstance(record.config, dict) else {}
    if config.get("resolution_method") == "explicit":
        return STATUS_AUTHORITATIVE
    return STATUS_UNKNOWN


def stamp_user_transform_provenance(trace: Trace, input_transform: Callable[..., Any]) -> None:
    """Populate ``trace.input_preprocessor`` for a user-supplied transform (B1).

    A capture that APPLIED an input transform carries provenance on every
    path, not only the HF bridges (which stamp a richer record over this one
    after ``tl.trace`` returns -- an already-stamped record is never
    overwritten). A raw callable is honest-unknown: ``verified`` stays False
    and the derived status is ``"unknown"`` -- never a guess about what the
    callable did.
    """

    if getattr(trace, "input_preprocessor", None) is not None:
        return
    from .trace import ResolvedPreprocessing, _scrubbed_transform_repr

    transform_repr = _scrubbed_transform_repr(input_transform) or "<transform>"
    trace.input_preprocessor = ResolvedPreprocessing(
        source="user_transform",
        identifier=transform_repr,
        verified=False,
        config={"resolution_method": "user_transform", "transform_repr": transform_repr},
        description=f"user-supplied input transform: {transform_repr}",
    )
