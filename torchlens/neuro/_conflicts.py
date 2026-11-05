"""Known extras conflicts of the neuro desk, diagnosed from package metadata.

``torchlens[neuro,brainscore]`` resolves, but rsatoolbox then cannot import:
rsatoolbox 0.3.x imports ``sklearn.utils.validation.validate_data``
(scikit-learn >= 1.6) without declaring that floor, while brainscore-vision
2.3 pins ``scikit-learn<1.6``, so the resolver installs scikit-learn 1.5.x and
``import rsatoolbox`` fails (measured on Python 3.11 with rsatoolbox 0.3.2 and
scikit-learn 1.5.2; docs/reference/neuro.md).

The facade gate must answer ``hasattr``/``dir()`` without executing foreign
code (the cold-import gate), so the diagnosis reads installed distribution
METADATA only (``importlib.metadata``); it never imports rsatoolbox or
scikit-learn. :func:`require_rsatoolbox` (defined here, re-exported by ``_handoff``) attempts
the real import and consults the same diagnosis only when that import fails.

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import metadata
from importlib.util import find_spec
from typing import Any, NoReturn

from .._errors import MissingDependencyError

__all__ = [
    "RsatoolboxConflict",
    "check_rsatoolbox_conflict",
    "refuse_rsatoolbox_conflict",
    "require_rsatoolbox",
    "rsatoolbox_conflict",
    "rsatoolbox_version",
]

#: Stable refusal code (DOCUMENTED-UNSTABLE spelling) for an installed neuro
#: dependency that cannot import because another installed package pins one
#: of its own dependencies below what it needs.
CONFLICT_CODE = "neuro_dependency_conflict"

#: First rsatoolbox release whose import needs scikit-learn >= 1.6 (0.3.x
#: imports ``sklearn.utils.validation.validate_data``).
_RSATOOLBOX_NEEDS_NEW_SKLEARN = (0, 3)

#: First scikit-learn release that ships ``validate_data``.
_SKLEARN_FLOOR = (1, 6)

#: Distribution that pins ``scikit-learn<1.6`` (the brainscore extra).
_PINNING_DISTRIBUTION = "brainscore-vision"

_RELEASE_PATTERN = re.compile(r"^\s*(\d+)(?:\.(\d+))?")

_SEPARATE_ENVIRONMENTS = (
    "install torchlens[neuro] and torchlens[brainscore] in separate environments "
    '(`pip install "torchlens[neuro]"` in one, `pip install "torchlens[brainscore]"` '
    "in the other)"
)


@dataclass(frozen=True)
class RsatoolboxConflict:
    """One diagnosed rsatoolbox / scikit-learn version conflict.

    Parameters
    ----------
    rsatoolbox:
        Installed rsatoolbox version string.
    scikit_learn:
        Installed scikit-learn version string (below the 1.6 floor).
    pinned_by:
        ``"<distribution> <version>"`` of the installed package that pins
        scikit-learn below 1.6, or ``None`` when no known pinning package is
        installed.
    """

    rsatoolbox: str
    scikit_learn: str
    pinned_by: str | None


def _installed(distribution: str) -> str | None:
    """Return a distribution's installed version from metadata, or ``None``.

    Parameters
    ----------
    distribution:
        Distribution name as published (``"scikit-learn"``).

    Returns
    -------
    str | None
        The installed version, or ``None`` when it is not installed.
    """

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _release(version: str) -> tuple[int, int] | None:
    """Parse the leading ``major.minor`` release of a version string.

    Parameters
    ----------
    version:
        A version string such as ``"1.5.2"`` or ``"0.3.2.dev1"``.

    Returns
    -------
    tuple[int, int] | None
        ``(major, minor)`` (minor 0 when absent), or ``None`` when the
        string does not start with a release number.
    """

    match = _RELEASE_PATTERN.match(version)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def rsatoolbox_conflict() -> RsatoolboxConflict | None:
    """Diagnose the rsatoolbox / scikit-learn conflict from metadata only.

    Returns
    -------
    RsatoolboxConflict | None
        The conflict when rsatoolbox >= 0.3 and scikit-learn < 1.6 are both
        installed; ``None`` otherwise (including when either version is not
        a parseable release, so an unknown layout is never refused here).
    """

    rsatoolbox_version = _installed("rsatoolbox")
    sklearn_version = _installed("scikit-learn")
    if rsatoolbox_version is None or sklearn_version is None:
        return None
    rsatoolbox_release = _release(rsatoolbox_version)
    sklearn_release = _release(sklearn_version)
    if rsatoolbox_release is None or sklearn_release is None:
        return None
    if rsatoolbox_release < _RSATOOLBOX_NEEDS_NEW_SKLEARN or sklearn_release >= _SKLEARN_FLOOR:
        return None
    pinning_version = _installed(_PINNING_DISTRIBUTION)
    pinned_by = None if pinning_version is None else f"{_PINNING_DISTRIBUTION} {pinning_version}"
    return RsatoolboxConflict(rsatoolbox_version, sklearn_version, pinned_by)


def refuse_rsatoolbox_conflict(
    owner: str, attribute: str, conflict: RsatoolboxConflict
) -> NoReturn:
    """Raise the typed refusal for a diagnosed rsatoolbox conflict.

    Parameters
    ----------
    owner:
        Fully qualified owning module (``"torchlens.neuro"``).
    attribute:
        The feature the user touched (``"rdms"``).
    conflict:
        The diagnosed conflict.

    Raises
    ------
    MissingDependencyError
        Always, with code ``neuro_dependency_conflict``; ``fields`` carry the
        dependency, the conflicting package and both installed versions.
    """

    cause = (
        f"{conflict.pinned_by} pins scikit-learn<1.6"
        if conflict.pinned_by is not None
        else "an installed package holds scikit-learn below 1.6"
    )
    remedy = (
        _SEPARATE_ENVIRONMENTS
        if conflict.pinned_by is not None
        else 'upgrade it with `pip install "scikit-learn>=1.6"`'
    )
    raise MissingDependencyError(
        f"{owner}.{attribute} needs rsatoolbox, which is installed but cannot import: "
        f"rsatoolbox {conflict.rsatoolbox} needs scikit-learn>=1.6 and scikit-learn "
        f"{conflict.scikit_learn} is installed ({cause})",
        code=CONFLICT_CODE,
        remedy=remedy,
        owner=owner,
        attribute=attribute,
        dependency="rsatoolbox",
        dependency_version=conflict.rsatoolbox,
        conflicting="scikit-learn",
        conflicting_version=conflict.scikit_learn,
        pinned_by=conflict.pinned_by,
    )


def check_rsatoolbox_conflict(owner: str, attribute: str) -> None:
    """Facade pre-resolution check: refuse a gated name under the conflict.

    Parameters
    ----------
    owner:
        Fully qualified owning module.
    attribute:
        The gated name being resolved.

    Raises
    ------
    MissingDependencyError
        When :func:`rsatoolbox_conflict` diagnoses the conflict.
    """

    conflict = rsatoolbox_conflict()
    if conflict is not None:
        refuse_rsatoolbox_conflict(owner, attribute, conflict)


def require_rsatoolbox(owner: str = "torchlens.neuro", attribute: str = "rdms") -> Any:
    """Import and return rsatoolbox, teaching the right fix when it fails.

    Parameters
    ----------
    owner:
        Fully qualified module of the feature being used.
    attribute:
        The feature being used (``"rdms"``, ``"datasets"``), named with
        ``owner`` in the conflict refusal.

    Returns
    -------
    Any
        The imported ``rsatoolbox`` package.

    Raises
    ------
    MissingDependencyError
        Code ``neuro_dependency_conflict`` when rsatoolbox is installed but
        fails to import and the metadata diagnosis names the scikit-learn
        conflict (the [neuro,brainscore] pin).
    ImportError
        When rsatoolbox is not installed (the message names the exact extra
        spelling), or is installed but fails to import for an undiagnosed
        reason (the message says so; the cause is chained).
    """

    try:
        import rsatoolbox
    except ImportError as exc:
        if find_spec("rsatoolbox") is None:
            raise ImportError(
                "torchlens.neuro requires rsatoolbox: install it with "
                '`pip install "torchlens[neuro]"`.'
            ) from exc
        conflict = rsatoolbox_conflict()
        if conflict is not None:
            # Raised inside the handler, so the failed import rides __context__.
            refuse_rsatoolbox_conflict(owner, attribute, conflict)
        raise ImportError(
            f"torchlens.neuro needs rsatoolbox, which is installed but failed to import "
            f"({exc}); see the chained exception for the root cause."
        ) from exc
    return rsatoolbox


def rsatoolbox_version() -> str:
    """Return the installed rsatoolbox version via package metadata.

    Neither validated rsatoolbox version (0.1.5 floor, 0.3.x current)
    exposes ``__version__`` (measured), so the recorded version comes from
    ``importlib.metadata``.

    Returns
    -------
    str
        The installed distribution version, or ``"unknown"`` when metadata
        is unavailable (e.g. a path-injected source tree).
    """

    import importlib.metadata

    try:
        return importlib.metadata.version("rsatoolbox")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
