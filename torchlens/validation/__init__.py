"""Validation subpackage for saved activations, backward capture, and invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import _entry_refusals as _refusals
from .._io import _json
from .._io.format_contract import _MODELLOG_216_NOTE, below_floor_error
from .._user_public_impls import (
    validate_backward_pass,
    validate_forward_pass,
)
from .._utils.facade import resolve_facade_attr
from ._descriptor_versions import validate_format3
from ._diagnostics import (
    get_validation_diagnostics,
    get_validation_failure,
    last_validation_failure,
    last_validation_peak_memory,
)
from ._invariants import check_metadata_invariants
from ._status import ValidationReplayState as ValidationReplayState


def validate_tlspec(
    path: str | Path,
    *,
    allow_unsupported_runnable_versions: bool = False,
) -> None:
    """Validate a saved format 3 artifact; refuse saves written by torchlens 2.37 and earlier.

    Parameters
    ----------
    path:
        ``.tlspec`` directory path.
    allow_unsupported_runnable_versions:
        Preserve structural validation while deferring runnable descriptor
        version ceilings to the structured readiness report. Public callers
        should retain the default tripwire behavior.

    Raises
    ------
    ValueError
        If a unified manifest violates the v1 schema.
    FileNotFoundError
        If a unified manifest file is missing.
    torchlens.errors.ArtifactTooOldError
        If the artifact carries a recognized TorchLens 2.x marker.

    Examples
    --------
    >>> import os, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = os.path.join(tmp, "run.tlspec")
    ...     _ = tl.trace(model, x).save(path)
    ...     print(tl.validation.validate_tlspec(path))
    None
    """

    from ..io import _pre3_observed, detect_tlspec_format  # deferred: cycle io

    tlspec_path = Path(path)
    if not tlspec_path.exists():
        raise FileNotFoundError(f"TorchLens .tlspec path does not exist: {tlspec_path}.")
    _refusals.refuse_if_not_directory(tlspec_path)

    manifest_path = tlspec_path / "manifest.json"
    if manifest_path.exists():
        # Same untrusted artifact as ``tl.load(path)``, different door: route the
        # manifest read through the ONE bounded reader (byte ceiling + depth
        # prescan) so a hostile depth-900 manifest refuses typed here too instead
        # of escaping as an uncaught RecursionError from stdlib ``json.load``.
        try:
            manifest_value = _json.read_bounded(manifest_path)
        except json.JSONDecodeError as exc:
            _refusals.refuse_manifest_unparseable(manifest_path, exc)
        _refusals.refuse_if_manifest_not_object(manifest_path, manifest_value)

    tlspec_format = detect_tlspec_format(tlspec_path)
    if tlspec_format == "pre_3":
        observed = _pre3_observed(tlspec_path) or "2.x artifact markers"
        raise below_floor_error(
            observed=observed,
            path=str(tlspec_path),
            remedy_note=(
                _MODELLOG_216_NOTE
                if observed.startswith("io_format_version=2, n_activation_blobs=")
                else ""
            ),
            code="artifact_version_below_floor",
        )
    _refusals.refuse_if_unknown_tlspec_format(tlspec_path, tlspec_format)
    validate_format3(tlspec_path, allow_unsupported_runnable_versions)


#: Removed public names: each raises the typed ``facade_redirect`` error naming
#: its replacement (clean break, no alias).
_REDIRECTS: dict[str, str] = {
    "validate_saved_outs": (
        "use torchlens.validate(model, x, scope='saved'), or "
        "trace.validate_forward_pass(ground_truth_outputs) -- validate_saved_outs was removed"
    ),
    "validate_trace_saved_outs": (
        "use torchlens.validate(model, x, scope='saved'), or "
        "trace.validate_forward_pass(ground_truth_outputs) -- validate_trace_saved_outs "
        "was removed"
    ),
}


# Runtime only: under TYPE_CHECKING a module ``__getattr__`` would make every
# name on this module type-check as Any.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        """Raise the typed teaching error for a removed torchlens.validation name."""

        return resolve_facade_attr(
            owner=__name__, name=name, module_globals=globals(), redirects=_REDIRECTS
        )


__all__ = [
    "get_validation_failure",
    "get_validation_diagnostics",
    "last_validation_failure",
    "last_validation_peak_memory",
    "validate_backward_pass",
    "validate_forward_pass",
    "check_metadata_invariants",
    "validate_tlspec",
]
