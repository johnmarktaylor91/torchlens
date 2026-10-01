"""Artifact session: manifest-first facts, content identity, bounded loads.

The agent surface's ONE loading seam (agent memo 3.1-3.2). Manifest facts are
TORCH-FREE: a validated ``manifest.json`` read answers kind, identity,
fingerprint, declared payload bytes, and site counts in ~60 ms without
unpickling anything -- the only safe first look at an artifact the caller did
not produce. Content identity is a sha256 digest; paths are inputs, never
identities. Payload tensors are NEVER retained by the structural cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._errors import InvalidArgumentError

#: Manifest-declared payload bytes above which loads go LAZY (blobs on disk).
EAGER_LOAD_MAX_BYTES = 256 * 1024 * 1024

#: Declared payload-entry count above which a served load REFUSES.
LOAD_MAX_PAYLOAD_ENTRIES = 100_000

#: Structural cache of loaded artifacts, keyed by manifest digest PLUS the
#: metadata.pkl content witness (:func:`_metadata_witness`).
_TRACE_CACHE: dict[str, tuple[Any, dict[str, Any]]] = {}
_TRACE_CACHE_MAX = 4

#: (path, dev, ino, size, mtime_ns) -> digest. A VERIFIED cache hint only.
_DIGEST_HINTS: dict[tuple[str, int, int, int, int], str] = {}
_DIGEST_HINTS_MAX = 16

#: metadata.pkl stat identity -> sha256 witness (the cache-key join, 3.11c).
_METADATA_HINTS: dict[tuple[str, int, int, int, int], str] = {}


def _loads_bounded(text: str) -> Any:
    """Parse JSON through the ONE bounded reader, without the torch-heavy parent.

    ``torchlens._io._json`` is stdlib-only, but importing it normally executes
    the ``_io`` package ``__init__`` (which imports torch) -- and the manifest
    preflight is a TIER-0 path that must stay torch-free. When ``_io`` is
    already loaded we use its module; otherwise the same FILE loads under a
    private name (shared code, never a duplicated guard).

    Parameters
    ----------
    text:
        Raw JSON text.

    Returns
    -------
    Any
        Parsed value (the bounded reader's refusals propagate).
    """

    import importlib.util
    import sys

    module = sys.modules.get("torchlens._io._json")
    if module is None:
        name = "torchlens.agent._json_bounded"
        module = sys.modules.get(name)
        if module is None:
            path = Path(__file__).resolve().parent.parent / "_io" / "_json.py"
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:  # pragma: no cover - packaging corruption
                raise ImportError(f"cannot load the bounded JSON reader from {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    return module.loads_bounded(text)


def artifact_digest(path: Path) -> tuple[str, bytes | None]:
    """Return the artifact's content digest plus raw manifest bytes.

    Parameters
    ----------
    path:
        Artifact path (a ``.tlspec`` directory or a single file).

    Returns
    -------
    tuple[str, bytes | None]
        Hex digest identifying the artifact content, and the manifest bytes
        when the artifact has a readable ``manifest.json`` (``None`` for
        single-file artifacts, which are stream-hashed).
    """

    from hashlib import sha256

    manifest_path = path / "manifest.json" if path.is_dir() else None
    if manifest_path is not None and manifest_path.is_file():
        manifest_bytes = manifest_path.read_bytes()
        return sha256(manifest_bytes).hexdigest(), manifest_bytes
    return _sha256_of_file(path), None


def _sha256_of_file(path: Path) -> str:
    """Stream-hash one file with the stdlib only.

    A LOCAL copy of ``torchlens._io.manifest.sha256_of_file``: importing that
    module executes the ``_io`` package ``__init__`` (which imports torch),
    and this runs on the TIER-0 ``info`` path for single-file / non-directory
    inputs (AUD-CODE 3.11e). Parity with the ``_io`` authority is pinned by
    tests/test_w051_agent_artifacts.py.

    Parameters
    ----------
    path:
        File to hash (a directory hashes as empty: nothing is read).

    Returns
    -------
    str
        Hex digest.
    """

    from hashlib import sha256

    digest = sha256()
    if not path.is_file():
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_witness(path: Path) -> str:
    """Return the content witness of ``metadata.pkl`` for the structural cache.

    The manifest digest is the artifact's IDENTITY, but the cache serves the
    UNPICKLED trace, and two directories can share one manifest while their
    ``metadata.pkl`` bytes differ (a copy whose metadata was overwritten or
    corrupted). Keying the cache on the manifest alone served the healthy
    trace for the corrupted copy (AUD-CODE 3.11c); the witness joins the
    key so any metadata change is a cache MISS and reloads through the
    integrity-checked ``tl.load`` door. Hashed once per stat identity
    (the same verified-hint pattern as :data:`_DIGEST_HINTS`); a single-
    file artifact is already content-hashed and reads ``""``.

    Parameters
    ----------
    path:
        Artifact path.

    Returns
    -------
    str
        Hex digest of ``metadata.pkl`` (``""`` when the artifact has none).
    """

    metadata_path = path / "metadata.pkl" if path.is_dir() else None
    if metadata_path is None or not metadata_path.is_file():
        return ""
    identity = _stat_identity(metadata_path)
    if identity is not None:
        hinted = _METADATA_HINTS.get(identity)
        if hinted is not None:
            return hinted
    witness = _sha256_of_file(metadata_path)
    if identity is not None:
        if len(_METADATA_HINTS) >= _DIGEST_HINTS_MAX:
            _METADATA_HINTS.pop(next(iter(_METADATA_HINTS)))
        _METADATA_HINTS[identity] = witness
    return witness


def _stat_identity(path: Path) -> tuple[str, int, int, int, int] | None:
    """Return the (path, dev, ino, size, mtime_ns) identity for the hint map."""

    probe = path / "manifest.json" if path.is_dir() else path
    try:
        stat_result = probe.stat()
    except OSError:
        return None
    return (
        str(path),
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def resolve_digest(path: Path) -> tuple[str, bytes | None]:
    """Resolve the content digest through the verified stat-identity hint map.

    Parameters
    ----------
    path:
        Artifact path.

    Returns
    -------
    tuple[str, bytes | None]
        Content digest, and the manifest bytes when hashing read them.
    """

    identity = _stat_identity(path)
    if identity is not None:
        hinted = _DIGEST_HINTS.get(identity)
        if hinted is not None:
            return hinted, None
    digest, manifest_bytes = artifact_digest(path)
    if identity is not None:
        if len(_DIGEST_HINTS) >= _DIGEST_HINTS_MAX:
            _DIGEST_HINTS.pop(next(iter(_DIGEST_HINTS)))
        _DIGEST_HINTS[identity] = digest
    return digest, manifest_bytes


def resolve_artifact_path(path_arg: str) -> Path:
    """Validate and expand one artifact path argument.

    Parameters
    ----------
    path_arg:
        Filesystem path string from a tool request.

    Returns
    -------
    Path
        Existing artifact path.

    Raises
    ------
    InvalidArgumentError
        ``agent_artifact_unreadable`` when nothing exists at the path.
    """

    path = Path(path_arg).expanduser()
    if not path.exists():
        raise InvalidArgumentError(
            f"No artifact at {str(path)!r}",
            code="agent_artifact_unreadable",
            remedy="pass the path of an artifact saved with tl.save(trace, path)",
            path=str(path),
        )
    return path


def read_manifest(path: Path) -> tuple[dict[str, Any] | None, bytes | None]:
    """Read and parse the artifact manifest, torch-free.

    Parameters
    ----------
    path:
        Artifact path.

    Returns
    -------
    tuple[dict | None, bytes | None]
        Parsed manifest mapping (``None`` when absent or unparseable) and the
        raw bytes when read.
    """

    manifest_path = path / "manifest.json" if path.is_dir() else None
    if manifest_path is None or not manifest_path.is_file():
        return None, None
    raw = manifest_path.read_bytes()
    try:
        parsed = _loads_bounded(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, raw
    return (parsed if isinstance(parsed, dict) else None), raw


#: Transport dtype byte widths for ``body_index`` rows. A LOCAL copy of the
#: ``torchlens._io.payload_reader`` table: importing that module drags the
#: ``_io`` package (and torch) into tier-0 verbs, and a budget gate must stay
#: torch-free. Parity with the ``_io`` authority is pinned by
#: tests/test_agent_surface_cli.py::test_declared_bytes_matches_the_io_authority.
_DTYPE_NBYTES: dict[str, int] = {
    "float64": 8,
    "double": 8,
    "float32": 4,
    "float": 4,
    "float16": 2,
    "half": 2,
    "bfloat16": 2,
    "complex64": 8,
    "complex128": 16,
    "int64": 8,
    "long": 8,
    "int32": 4,
    "int": 4,
    "int16": 2,
    "short": 2,
    "int8": 1,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "bool": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float8_e4m3fnuz": 1,
    "float8_e5m2fnuz": 1,
    "float8_e8m0fnu": 1,
}


def _entry_bytes(entry: Any) -> int:
    """Return one ``body_index`` entry's declared bytes (overestimate unknowns)."""

    if not isinstance(entry, dict):
        return 0
    elements = entry.get("num_elements")
    if not isinstance(elements, int):
        elements = 1
        for dim in entry.get("shape") or []:
            elements *= int(dim)
    dtype = str(entry.get("dtype", "")).removeprefix("torch.")
    return elements * _DTYPE_NBYTES.get(dtype, 8)


def declared_bytes_and_count(manifest: dict[str, Any] | None) -> tuple[int, int]:
    """Return manifest-DECLARED payload bytes and blob count, torch-free.

    Parameters
    ----------
    manifest:
        Parsed manifest mapping, or ``None``.

    Returns
    -------
    tuple[int, int]
        Declared payload bytes and declared payload entry count.
    """

    if manifest is None:
        return 0, 0
    body_index = manifest.get("body_index")
    if not isinstance(body_index, list):
        return 0, 0
    return sum(_entry_bytes(entry) for entry in body_index), len(body_index)


def build_load_plan(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Choose eager/lazy/refuse from manifest-DECLARED numbers, torch-free.

    Parameters
    ----------
    manifest:
        Parsed manifest mapping, or ``None``.

    Returns
    -------
    dict[str, Any]
        The disclosed ``load_plan``: mode, declared bytes/count, thresholds,
        and the reason.
    """

    declared_bytes, declared_count = declared_bytes_and_count(manifest)
    if declared_count > LOAD_MAX_PAYLOAD_ENTRIES:
        mode = "refuse"
        reason = (
            f"declared payload entries ({declared_count:,}) exceed the "
            f"single-tool-call bound ({LOAD_MAX_PAYLOAD_ENTRIES:,})"
        )
    elif declared_bytes > EAGER_LOAD_MAX_BYTES:
        mode = "lazy"
        reason = (
            f"declared payload bytes ({declared_bytes:,}) exceed the eager "
            f"threshold ({EAGER_LOAD_MAX_BYTES:,}); payloads stay on disk"
        )
    else:
        mode = "eager"
        reason = "declared payload bytes fit the eager threshold"
    return {
        "mode": mode,
        "declared_payload_bytes": declared_bytes,
        "declared_payload_count": declared_count,
        "eager_threshold_bytes": EAGER_LOAD_MAX_BYTES,
        "max_payload_entries": LOAD_MAX_PAYLOAD_ENTRIES,
        "reason": reason,
    }


def artifact_block(path: Path, digest: str, manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Build the envelope's artifact identity block (agent memo 3.9).

    ``basename`` is a human courtesy only -- no absolute or cwd-derived path
    ever appears in a machine payload.

    Parameters
    ----------
    path:
        Artifact path (basename source only).
    digest:
        Content digest hex string.
    manifest:
        Parsed manifest mapping, or ``None``.

    Returns
    -------
    dict[str, Any]
        Artifact identity block.
    """

    manifest = manifest or {}
    fingerprint = manifest.get("model_fingerprint")
    fingerprint_hash = (
        fingerprint.get("parameter_meta_hash") if isinstance(fingerprint, dict) else None
    )
    return {
        "kind": manifest.get("kind", "unknown"),
        "id": f"sha256:{digest}",
        "tlspec_version": manifest.get("tlspec_version"),
        "model_fingerprint": fingerprint_hash,
        "model_signature": manifest.get("model_signature"),
        "torchlens_version_at_save": manifest.get("torchlens_version"),
        "created_at": manifest.get("created_at"),
        "basename": path.name,
    }


def load_trace(path_arg: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load a saved Trace for read-only inspection, manifest-first and cached.

    Parameters
    ----------
    path_arg:
        Filesystem path to a ``.tlspec`` artifact.

    Returns
    -------
    tuple[Any, dict[str, Any], dict[str, Any]]
        Loaded ``Trace``, the disclosed ``load_plan``, and the envelope
        artifact block.

    Raises
    ------
    InvalidArgumentError
        ``agent_artifact_unreadable`` for a missing path;
        ``agent_artifact_load_refused`` when declared entries exceed the
        single-call bound; ``agent_artifact_kind_unsupported`` when the
        artifact is not a single Trace.
    """

    path = resolve_artifact_path(path_arg)
    digest, manifest_bytes = resolve_digest(path)
    manifest, _ = read_manifest(path) if manifest_bytes is None else _parse(manifest_bytes)
    block = artifact_block(path, digest, manifest)
    cache_key = f"{digest}:{_metadata_witness(path)}"
    cached = _TRACE_CACHE.get(cache_key)
    if cached is not None:
        return cached[0], cached[1], block
    plan = build_load_plan(manifest)
    if plan["mode"] == "refuse":
        raise InvalidArgumentError(
            f"Refusing to load {path.name!r}: {plan['reason']}",
            code="agent_artifact_load_refused",
            remedy=(
                "load it in Python via tl.load(path, lazy=True) where you "
                "control the process budget"
            ),
            declared_payload_count=plan["declared_payload_count"],
        )
    import torchlens as tl

    loaded = tl.load(str(path), lazy=plan["mode"] == "lazy")
    if not isinstance(loaded, tl.Trace):
        raise InvalidArgumentError(
            f"{path.name!r} loaded as {type(loaded).__name__}, not a Trace",
            code="agent_artifact_kind_unsupported",
            remedy=(
                "the agent tools cover single-Trace artifacts; load bundles "
                "or intervention specs in Python via tl.load(path)"
            ),
            loaded_kind=type(loaded).__name__,
        )
    # Settle the health basis ONCE at the cache seam through the explicit
    # scan door (health_facts default). Renders/dumps stay no-implicit-scan
    # (D4); without this, the basis a cached trace serves would depend on
    # WHICH tool touched payloads first, and repeated identical tool calls
    # would not be byte-identical. The cost is bounded by the load plan's
    # declared-entry refusal above.
    from ..report._health import health_facts

    health_facts(loaded)
    if len(_TRACE_CACHE) >= _TRACE_CACHE_MAX:
        _TRACE_CACHE.pop(next(iter(_TRACE_CACHE)))
    _TRACE_CACHE[cache_key] = (loaded, plan)
    return loaded, plan, block


def _parse(manifest_bytes: bytes) -> tuple[dict[str, Any] | None, bytes]:
    """Parse raw manifest bytes, tolerating malformed JSON.

    Parameters
    ----------
    manifest_bytes:
        Raw ``manifest.json`` bytes.

    Returns
    -------
    tuple[dict | None, bytes]
        Parsed mapping (``None`` when unparseable) and the bytes unchanged.
    """

    try:
        parsed = _loads_bounded(manifest_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, manifest_bytes
    return (parsed if isinstance(parsed, dict) else None), manifest_bytes
