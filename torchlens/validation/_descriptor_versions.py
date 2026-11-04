"""Runnable descriptor version tripwire shared by the sparse-descriptor and artifact validators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._io import _json


def validate_format3(tlspec_path: Path, allow_unsupported_runnable_versions: bool) -> None:
    """Validate a directory artifact, then apply the runnable descriptor version tripwire.

    The artifact check covers the manifest, the version window and the digests.
    ``tl.load`` defers an unsupported descriptor version to the structured readiness
    report; public validation keeps the tripwire, as it does for the sparse descriptor.
    An artifact without a runnable object has nothing to check.
    """

    from .._io.format3.runnable import DESCRIPTOR_PATH
    from ..io import _validate_format3_artifact

    _validate_format3_artifact(tlspec_path)
    if allow_unsupported_runnable_versions:
        return
    manifest = _json.read_bounded(tlspec_path / "manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("runnable"), dict):
        return
    descriptor = _json.read_bounded(tlspec_path / DESCRIPTOR_PATH)
    if not isinstance(descriptor, dict):
        raise ValueError(
            f"Runnable descriptor at {tlspec_path / DESCRIPTOR_PATH} must be an object."
        )
    validate_descriptor_versions(descriptor)


def validate_descriptor_versions(value: dict[str, Any]) -> None:
    """Require the exact descriptor component versions this runtime writes.

    Raises
    ------
    ValueError
        If a descriptor version field is missing or differs from this runtime's.
    """

    from .._runnable import (
        RUNNABLE_CALL_RECIPE_VERSION,
        RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
        RUNNABLE_INITIALIZER_POLICY_VERSION,
        RUNNABLE_TLSPEC_SCHEMA_VERSION,
    )

    expected_versions = {
        "capability": RUNNABLE_TLSPEC_SCHEMA_VERSION,
        "call_recipe": RUNNABLE_CALL_RECIPE_VERSION,
        "callable_ref_schema": RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
        "initializer_policy_version": RUNNABLE_INITIALIZER_POLICY_VERSION,
        "state_binding": "module_path_role_v1",
        "input_binding": "model_site_io_role_v1",
        "control_witness": "scalar_bool_and_arm_entry_v1",
    }
    for field_name, expected in expected_versions.items():
        if value.get(field_name) != expected:
            raise ValueError(
                f"Runnable descriptor {field_name} must be {expected!r}; "
                f"got {value.get(field_name)!r}."
            )
