"""Merged-artifact tamper refusals and the member-core checks that raise them.

Split out of :mod:`torchlens._merged._artifact` (file cap): the typed tamper builder,
the member-load failure classifier, and the foreign-pickle guard ``load_merged``
runs on every integrity-verified rank core.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from .._io import _json
from .._io.format3.contract import FORMAT_NAME, MANIFEST_FILE
from ._enums import MergedErrorCode
from ._errors import MergedArtifactError

_TAMPER_REMEDY = (
    "The artifact's integrity records do not match its contents; it was edited "
    "or corrupted in transit. Re-generate it from the original rank cores with "
    "tl.merge_ranks([...]).save(path)."
)

# Bundle-load failures that are INTEGRITY signals (tamper), not environmental
# degradation. A guarded-unpickler denylist refusal / corrupt pickle stream is
# raised as ``pickle.UnpicklingError``/``EOFError`` and, at the ``bundle.load``
# boundary, wrapped into a ``ArtifactError`` tagged with this code (A-R58-1).
_BUNDLE_INTEGRITY_CODE = "bundle_metadata_integrity_refused"

#: The 2.x metadata carrier. The 3.0 writer never produces it, so inside a 3.0 core it is
#: foreign content.
_LEGACY_PICKLE = "metadata.pkl"


def _is_bundle_integrity_refusal(exc: BaseException) -> bool:
    """Return True if a member bundle-load failure is a tamper signal, not drift.

    A guarded-unpickler denylist refusal or a corrupt/truncated pickle stream is
    a bundle-INTEGRITY failure: it means the member core's bytes are inconsistent
    with an honest capture, which is tamper. It is distinguished from genuine
    environmental degradation (torch/codec import drift, a missing optional
    dependency) both by the stable ``code`` the ``bundle.load`` boundary stamps
    on it AND by walking the ``__cause__``/``__context__`` chain for the raw
    ``pickle.UnpicklingError``/``EOFError`` (belt-and-braces: a refusal that
    reaches here through any other wrapper is still caught).
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (pickle.UnpicklingError, EOFError)):
            return True
        fields = getattr(current, "fields", None)
        if isinstance(fields, dict) and fields.get("code") == _BUNDLE_INTEGRITY_CODE:
            return True
        current = current.__cause__ or current.__context__
    return False


def _tamper(detail: str, *, remedy: str = _TAMPER_REMEDY, **payload: Any) -> MergedArtifactError:
    """Build the typed tamper refusal (integrity failure, never a presence gap).

    Every merged refusal now carries a ``fields["remedy"]`` (R65: the tamper
    refusals shipped with none, even where the correct sentence existed verbatim
    elsewhere in this module).
    """

    return MergedArtifactError(
        f"Merged artifact integrity failure: {detail}",
        code=MergedErrorCode.MERGED_DESCRIPTOR_TAMPER,
        remedy=remedy,
        **payload,
    )


def _refuse_foreign_pickle(rank: int, member_path: Path) -> Path:
    """Return ``member_path``, refusing a 3.0 rank core that holds a root ``metadata.pkl``.

    The pickle is never opened. ``tl.load`` refuses any root holding it as a 2.x artifact
    before reading it, which ``load_merged`` would otherwise record as a runtime
    degradation (capping the merge at partial). A core whose manifest does not name the
    3.0 format is left to the loader: a genuine 2.x core still degrades.

    Raises
    ------
    MergedArtifactError
        ``merged_descriptor_tamper`` when the core's manifest names the 3.0 format.
    """

    pickle_path = member_path / _LEGACY_PICKLE
    manifest_path = member_path / MANIFEST_FILE
    if not (pickle_path.exists() or pickle_path.is_symlink()):
        return member_path
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return member_path
    try:
        manifest = _json.read_bounded(manifest_path)
    except (OSError, json.JSONDecodeError):
        return member_path
    if isinstance(manifest, dict) and manifest.get("format") == FORMAT_NAME:
        raise _tamper(
            f"rank {rank} core {str(member_path)!r} is a 3.0 core but holds a root "
            f"{_LEGACY_PICKLE}; the 3.0 format writes no pickle, so the file is foreign "
            "content (refused unopened, never unpickled)"
        )
    return member_path
