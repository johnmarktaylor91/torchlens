"""Audit-only parsing and characterization for execution-read manifest v1.

This quarantine preserves immutable-history inspection.  Live driver, policy,
supervisor, subprocess, reducer, and writer paths must never import this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.policy import static_source_check


_EXECUTION_CODE_SUFFIXES_BY_KIND = {
    "native-library": frozenset({".a", ".dylib", ".pyd", ".so"}),
    "native-source": frozenset({".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}),
    "python-bytecode": frozenset({".pyc"}),
    "python-source": frozenset({".py", ".pyi", ".pyx"}),
}
_RUNTIME_SUPPORT_KINDS = frozenset({"runtime-file", "runtime-root"})
_MODEL_DATA_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".h5",
        ".hdf5",
        ".joblib",
        ".mar",
        ".msgpack",
        ".npy",
        ".npz",
        ".onnx",
        ".params",
        ".pb",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".safetensors",
        ".tflite",
        ".weights",
    }
)


@dataclass(frozen=True)
class ExecutionReadManifest:
    """Immutable audit representation of a legacy v1 read manifest."""

    manifest_id: str
    stable_id: str
    work_id: str
    execution_identity: str
    code_manifest_identity: str
    code_members: tuple[tuple[Path, str, str], ...]
    standard_input_asset: Optional[tuple[Path, str, str]]
    runtime_support: tuple[tuple[Path, str], ...]


def _validated_digest(value: str, *, field: str) -> str:
    """Return one canonical SHA-256 identity or reject legacy corruption.

    Parameters
    ----------
    value:
        Candidate identity.
    field:
        Diagnostic field name.

    Returns
    -------
    str
        The validated identity.
    """

    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a canonical sha256 identity")
    return value


def _regular_unaliased_file(path: Path) -> bool:
    """Return whether an audit member is one absolute regular non-symlink file.

    Parameters
    ----------
    path:
        Candidate legacy member.

    Returns
    -------
    bool
        True only for the legacy compiler's accepted file shape.
    """

    if not path.is_absolute() or path.is_symlink():
        return False
    try:
        status = path.stat()
    except OSError:
        return False
    return path.is_file() and status.st_nlink == 1 and path.resolve() == path


def _runtime_model_data_path(path: Path) -> bool:
    """Return whether a legacy member is shaped like forbidden model data.

    Parameters
    ----------
    path:
        Candidate legacy member.

    Returns
    -------
    bool
        True for checkpoint, weight, or serialized-model suffixes.
    """

    suffix = path.suffix.lower()
    if suffix in _MODEL_DATA_SUFFIXES:
        return True
    try:
        size = path.stat().st_size
    except OSError:
        return True
    if not suffix:
        return True
    return size >= 1024**2


def _manifest_identity_payload(
    *,
    stable_id: str,
    work_id: str,
    execution_identity: str,
    code_manifest_identity: str,
    code_members: Sequence[tuple[Path, str, str]],
    standard_input_asset: Optional[tuple[Path, str, str]],
    runtime_support: Sequence[tuple[Path, str]],
) -> dict[str, Any]:
    """Build the immutable v1 identity payload for historical comparison.

    Parameters
    ----------
    stable_id, work_id, execution_identity, code_manifest_identity:
        Historical request associations.
    code_members:
        Exact legacy implementation members.
    standard_input_asset:
        Optional exact standard input.
    runtime_support:
        Legacy runtime files and root grants.

    Returns
    -------
    dict[str, Any]
        Canonical JSON-compatible historical payload.
    """

    return {
        "version": "menagerie.crawler.execution-read-manifest.v1",
        "stable_id": stable_id,
        "work_id": work_id,
        "execution_identity": execution_identity,
        "code_manifest_identity": code_manifest_identity,
        "code_members": [
            {"path": str(path), "sha256": digest, "kind": kind}
            for path, digest, kind in code_members
        ],
        "standard_input_asset": (
            None
            if standard_input_asset is None
            else {
                "path": str(standard_input_asset[0]),
                "sha256": standard_input_asset[1],
                "asset_id": standard_input_asset[2],
            }
        ),
        "runtime_support": [{"path": str(path), "kind": kind} for path, kind in runtime_support],
    }


def compile_execution_read_manifest(
    *,
    stable_id: str,
    work_id: str,
    execution_identity: str,
    code_manifest_identity: str,
    code_members: Sequence[tuple[Path, str, str]],
    standard_input_asset: Optional[tuple[Path, str, str]] = None,
    runtime_support: Sequence[tuple[Path, str]] = (),
) -> ExecutionReadManifest:
    """Reconstruct a legacy v1 manifest for audit-only historical validation.

    Parameters
    ----------
    stable_id, work_id, execution_identity, code_manifest_identity:
        Historical request associations.
    code_members:
        Exact legacy implementation members.
    standard_input_asset:
        Optional exact standard input.
    runtime_support:
        Legacy runtime support declarations, including historical root grants.

    Returns
    -------
    ExecutionReadManifest
        Frozen audit representation.  It is not accepted by any live spawn API.
    """

    if not stable_id or not work_id:
        raise ValueError("execution manifest identities must be non-empty")
    _validated_digest(execution_identity, field="execution_identity")
    _validated_digest(code_manifest_identity, field="code_manifest_identity")
    normalized_code: list[tuple[Path, str, str]] = []
    seen_paths: set[Path] = set()
    for raw_path, digest, kind in code_members:
        path = raw_path.absolute()
        suffixes = _EXECUTION_CODE_SUFFIXES_BY_KIND.get(kind)
        if suffixes is None or path.suffix.lower() not in suffixes:
            raise ValueError(f"execution code member has forbidden kind/suffix: {path}")
        if _runtime_model_data_path(path) or not _regular_unaliased_file(path):
            raise ValueError(f"execution code member is aliased or model-data-shaped: {path}")
        _validated_digest(digest, field="code member digest")
        if hash_bytes(path.read_bytes()) != digest:
            raise ValueError(f"execution code member digest mismatch: {path}")
        if kind == "python-source":
            static_source_check(path)
        if path in seen_paths:
            raise ValueError(f"duplicate execution code member: {path}")
        seen_paths.add(path)
        normalized_code.append((path, digest, kind))
    normalized_code.sort(key=lambda item: str(item[0]))

    normalized_asset: Optional[tuple[Path, str, str]] = None
    if standard_input_asset is not None:
        raw_path, digest, asset_id = standard_input_asset
        path = raw_path.absolute()
        if not asset_id or not _regular_unaliased_file(path):
            raise ValueError("standard input asset must be an unaliased regular file")
        _validated_digest(digest, field="standard input asset digest")
        if hash_bytes(path.read_bytes()) != digest:
            raise ValueError("standard input asset digest mismatch")
        normalized_asset = (path, digest, asset_id)

    normalized_runtime: list[tuple[Path, str]] = []
    for raw_path, kind in runtime_support:
        path = raw_path.absolute()
        if kind not in _RUNTIME_SUPPORT_KINDS:
            raise ValueError(f"unknown runtime support kind: {kind}")
        if path == Path("/") or path.is_symlink() or not path.exists() or path.resolve() != path:
            raise ValueError(f"unsafe runtime support path: {path}")
        if kind == "runtime-file" and not path.is_file():
            raise ValueError(f"runtime-file support is not a file: {path}")
        if kind == "runtime-root" and not path.is_dir():
            raise ValueError(f"runtime-root support is not a directory: {path}")
        normalized_runtime.append((path, kind))
    normalized_runtime = sorted(set(normalized_runtime), key=lambda item: (str(item[0]), item[1]))
    payload = _manifest_identity_payload(
        stable_id=stable_id,
        work_id=work_id,
        execution_identity=execution_identity,
        code_manifest_identity=code_manifest_identity,
        code_members=normalized_code,
        standard_input_asset=normalized_asset,
        runtime_support=normalized_runtime,
    )
    return ExecutionReadManifest(
        manifest_id=stable_hash(payload),
        stable_id=stable_id,
        work_id=work_id,
        execution_identity=execution_identity,
        code_manifest_identity=code_manifest_identity,
        code_members=tuple(normalized_code),
        standard_input_asset=normalized_asset,
        runtime_support=tuple(normalized_runtime),
    )


def verify_execution_read_manifest(manifest: ExecutionReadManifest) -> None:
    """Recompile and compare one historical v1 manifest without executing it.

    Parameters
    ----------
    manifest:
        Frozen legacy manifest under audit.
    """

    rebuilt = compile_execution_read_manifest(
        stable_id=manifest.stable_id,
        work_id=manifest.work_id,
        execution_identity=manifest.execution_identity,
        code_manifest_identity=manifest.code_manifest_identity,
        code_members=manifest.code_members,
        standard_input_asset=manifest.standard_input_asset,
        runtime_support=manifest.runtime_support,
    )
    if rebuilt != manifest:
        raise ValueError("execution read manifest identity mismatch")


def audit_runtime_root_grants(manifest: ExecutionReadManifest) -> tuple[Path, ...]:
    """Return historical root grants for reporting, never live enforcement.

    Parameters
    ----------
    manifest:
        Validated v1 manifest under audit.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Exact roots formerly granted by v1 semantics.
    """

    verify_execution_read_manifest(manifest)
    return tuple(path for path, kind in manifest.runtime_support if kind == "runtime-root")
