"""Run the metadata-invariant tripwire from a backend's validate path."""

from __future__ import annotations

from typing import Any

from .. import _events

__all__ = ["run_metadata_invariants"]


def run_metadata_invariants(trace: Any) -> bool:
    """Run the registered tripwire on ``trace``, loading it first when needed.

    ``torchlens.validation`` registers ``check_metadata_invariants`` at import;
    a backend validating before anything imported that package loads it here so
    the check runs rather than being skipped.
    """
    if _events.registered_metadata_invariants() is None:
        import torchlens.validation  # noqa: F401  # deferred: cycle validation (registers the hook)
    return _events.run_metadata_invariants(trace)
