"""Merged-artifact member-load classification: which member failures are tamper, not drift.

Split out of :mod:`torchlens._merged._artifact` (file cap). ``load_merged`` asks
:func:`_is_bundle_integrity_refusal` about every rank core that fails to load; a True answer
refuses the whole artifact as ``merged_descriptor_tamper`` instead of recording a runtime
degradation that caps the merge at partial.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from .._io import _json
from .._io.format3.contract import FORMAT_NAME, MANIFEST_FILE

# Bundle-load failures that are INTEGRITY signals (tamper), not environmental
# degradation. A guarded-unpickler denylist refusal / corrupt pickle stream is
# raised as ``pickle.UnpicklingError``/``EOFError`` and, at the ``bundle.load``
# boundary, wrapped into a ``ArtifactError`` tagged with this code (A-R58-1).
_BUNDLE_INTEGRITY_CODE = "bundle_metadata_integrity_refused"

#: The 2.x metadata carrier. The 3.0 writer never produces it, so inside a 3.0 core it is
#: foreign content.
_LEGACY_PICKLE = "metadata.pkl"


def _is_bundle_integrity_refusal(exc: BaseException, member_path: Path | None = None) -> bool:
    """Return True if a member bundle-load failure is a tamper signal, not drift.

    A guarded-unpickler denylist refusal or a corrupt/truncated pickle stream is
    a bundle-INTEGRITY failure: it means the member core's bytes are inconsistent
    with an honest capture, which is tamper. It is distinguished from genuine
    environmental degradation (torch/codec import drift, a missing optional
    dependency) both by the stable ``code`` the ``bundle.load`` boundary stamps
    on it AND by walking the ``__cause__``/``__context__`` chain for the raw
    ``pickle.UnpicklingError``/``EOFError`` (belt-and-braces: a refusal that
    reaches here through any other wrapper is still caught). A 3.0 core at
    ``member_path`` that holds a root ``metadata.pkl`` is tamper too, whatever the
    loader raised (see :func:`_holds_foreign_pickle`).
    """

    if member_path is not None and _holds_foreign_pickle(member_path):
        return True
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


def _holds_foreign_pickle(member_path: Path) -> bool:
    """Return whether a rank core names the 3.0 format yet holds a root ``metadata.pkl``.

    The pickle is never opened: ``tl.load`` refuses any root holding it as a 2.x artifact
    before reading it, so without this check ``load_merged`` recorded the planted file as a
    runtime degradation. A core whose manifest does not name the 3.0 format (or cannot be
    read) answers False and stays with the loader's verdict: a genuine 2.x core still degrades.
    """

    pickle_path = member_path / _LEGACY_PICKLE
    manifest_path = member_path / MANIFEST_FILE
    if not (pickle_path.exists() or pickle_path.is_symlink()):
        return False
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return False
    try:
        manifest = _json.read_bounded(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(manifest, dict) and manifest.get("format") == FORMAT_NAME
