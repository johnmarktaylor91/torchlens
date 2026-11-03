"""Runtime identity policy shared by every artifact reader.

A saved manifest stamps the torch, TorchLens and Python versions that wrote it.
Both readers (the 2.x bundle path in :mod:`torchlens._io.manifest` and the 3.0
reader in :mod:`torchlens._io.format3`) apply the same policy to those three
strings, after their own format floor and ceiling checks have run:

* torch: a different major version refuses (``bundle_torch_incompatible``);
  a different minor version warns; an unparsable pair that differs refuses.
* TorchLens: an unparsable writer version refuses
  (``bundle_producer_unverifiable``); a newer writer warns; an older one logs.
* Python: a different major version warns; an unparsable, different string warns.
"""

from __future__ import annotations

import logging
import sys
import warnings

import torch
from packaging.version import InvalidVersion, Version

from .._vocab._package_version import __version__ as TORCHLENS_VERSION
from ..errors._base import TorchLensWarning
from .format_errors import ArtifactError, ArtifactRuntimeIncompatibleError

__all__ = ["check_runtime_identity", "parse_pep440", "runtime_python_version"]

LOGGER = logging.getLogger(__name__)

_TORCH_REMEDY = "load the bundle under a torch runtime with the recorded major version"


def parse_pep440(
    version_text: str,
    *,
    label: str,
    warn_on_failure: bool = True,
) -> Version | None:
    """Parse a version string under PEP 440; ``None`` (optionally warning) when it fails."""

    try:
        return Version(version_text)
    except InvalidVersion:
        if warn_on_failure:
            warnings.warn(
                f"Could not parse {label} version {version_text!r} under PEP 440; "
                "falling back to string comparison.",
                TorchLensWarning,
                stacklevel=4,
            )
        return None


def runtime_python_version() -> str:
    """Return the active interpreter's ``major.minor.micro`` string."""

    info = sys.version_info
    return f"{info.major}.{info.minor}.{info.micro}"


def check_runtime_identity(
    *,
    torch_version: str,
    torchlens_version: str,
    python_version: str,
    subject: str = "Bundle",
) -> None:
    """Apply the torch, TorchLens and Python identity policy to a manifest's stamps.

    Callers run their format floor and ceiling checks first; ``subject`` opens every
    message (``"Bundle"`` for the 2.x path, ``"Artifact"`` for a 3.0 artifact).
    """

    _check_torch(torch_version, subject)
    _check_torchlens(torchlens_version, subject)
    _check_python(python_version, subject)


def _check_torch(recorded: str, subject: str) -> None:
    runtime = parse_pep440(torch.__version__, label="runtime torch")
    saved = parse_pep440(recorded, label="manifest torch")
    if runtime is not None and saved is not None:
        if runtime.major != saved.major:
            raise ArtifactRuntimeIncompatibleError(
                f"{subject} torch_version={recorded} is incompatible with runtime "
                f"torch_version={torch.__version__} (major version mismatch). "
                f"Remedy: {_TORCH_REMEDY}.",
                code="bundle_torch_incompatible",
                remedy=_TORCH_REMEDY,
            )
        if runtime.minor != saved.minor:
            warnings.warn(
                f"{subject} torch_version={recorded} differs from runtime "
                f"torch_version={torch.__version__} (minor version mismatch).",
                TorchLensWarning,
                stacklevel=4,
            )
    elif recorded != torch.__version__:
        raise ArtifactRuntimeIncompatibleError(
            f"{subject} torch_version={recorded} could not be parsed compatibly with runtime "
            f"torch_version={torch.__version__}; refusing load. Remedy: {_TORCH_REMEDY}.",
            code="bundle_torch_incompatible",
            remedy=_TORCH_REMEDY,
        )


def _check_torchlens(recorded: str, subject: str) -> None:
    runtime = parse_pep440(TORCHLENS_VERSION, label="runtime torchlens")
    saved = parse_pep440(recorded, label="manifest torchlens", warn_on_failure=False)
    if saved is None:
        raise ArtifactError(
            f"{subject} torchlens_version={recorded!r} could not be parsed under PEP 440; "
            "refusing a current-schema artifact with unverifiable producer provenance. "
            "Remedy: re-save the artifact with a released torchlens.",
            code="bundle_producer_unverifiable",
            remedy="re-save the artifact with a released torchlens",
        )
    if runtime is None:
        return
    if saved > runtime:
        warnings.warn(
            f"{subject} torchlens_version={recorded} is newer than runtime "
            f"torchlens_version={TORCHLENS_VERSION}.",
            TorchLensWarning,
            stacklevel=4,
        )
    elif saved < runtime:
        LOGGER.info(
            "%s torchlens_version=%s is older than runtime torchlens_version=%s.",
            subject,
            recorded,
            TORCHLENS_VERSION,
        )


def _check_python(recorded: str, subject: str) -> None:
    current = runtime_python_version()
    runtime = parse_pep440(current, label="runtime python")
    saved = parse_pep440(recorded, label="manifest python")
    if runtime is not None and saved is not None:
        if runtime.major != saved.major:
            warnings.warn(
                f"{subject} python_version={recorded} differs from runtime "
                f"python_version={current} (major version mismatch).",
                TorchLensWarning,
                stacklevel=4,
            )
    elif recorded != current:
        warnings.warn(
            f"{subject} python_version={recorded} differs from runtime python_version="
            f"{current} and could not be parsed under PEP 440.",
            TorchLensWarning,
            stacklevel=4,
        )
