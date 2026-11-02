"""Capture-cache lookup and store for the PREPARE stage.

Moved verbatim from ``torchlens._user_funcs`` by the capture seam; the door's
cache-key resolution moved verbatim from ``prepare.door``.
"""

import os
import pickle
import stat
import tempfile
import time
import warnings
from collections.abc import Mapping
from dataclasses import fields as dataclass_fields, replace
from pathlib import Path
from typing import Any, cast

from torchlens import _state

from ..._autoroute._builtin_output import semantic_output_cache_key
from ..._capture_state_helpers import (
    _capture_cache_dir,
    _capture_cache_key,
    _facet_recipe_cache_key,
    _prepare_log_for_capture_cache,
)
from ..._data_classes.trace import (
    Trace,
)
from ..._trace_selector_helpers import (
    _predicate_cache_key,
    _stable_cache_fragment,
)
from .chunked import normalize_chunk_paths
from .request import _DoorState

# --------------------------------------------------------------------------- #
# Capture-cache integrity (``trace(..., cache=True)``)                         #
# --------------------------------------------------------------------------- #
#
# The capture cache stores a whole pickled ``Trace``. That object graph is far
# outside what ``_io._safe_unpickle.SafeBundleUnpickler`` admits (it deliberately
# refuses tensor/storage CONSTRUCTION and every non-allowlisted torchlens type), and
# widening that allowlist to fit a full ``Trace`` would disarm the ``.tlspec`` front
# door -- so the cache CANNOT route through it. The cache boundary is instead closed
# on the AUTHENTICITY axis, which is the axis the threat actually lives on:
#
# 1. The cache directories torchlens itself creates are made PRIVATE to this user.
#    ``TORCHLENS_CACHE_DIR`` pointed at a shared path (CI cache mount, container
#    volume, group-writable NFS home, a world-writable tmpdir) turned the next cache
#    HIT into arbitrary code execution with no error and no warning. Group/other write
#    bits are stripped from the two directories torchlens owns; a root owned by another
#    user, or one whose permissions cannot be tightened, refuses typed. Ancestors ABOVE
#    the configured cache directory are the caller's to secure and are not inspected.
# 2. Every entry is ONE self-authenticating record: a fixed-size header (magic +
#    HMAC-SHA256 tag keyed by a 0600 secret inside that directory) followed by the
#    pickled payload, committed by a single ``os.replace``. Bytes we cannot
#    authenticate are NEVER handed to ``pickle`` -- they are a cache MISS with a
#    warning, so a planted, stale, or corrupt entry is inert while a legitimate
#    pre-upgrade cache simply refills. The historical payload + ``.hmac`` sidecar
#    PAIR was retired because its two-step commit had a torn-generation window
#    (crash or concurrent reader between the payload swap and the tag write saw a
#    payload against the wrong generation's tag); the single record makes a torn
#    payload/tag state unrepresentable.
#
# (2) is the load-bearing guard and (1) is defense in depth: a mode check alone cannot
# speak to a file planted while the mode was briefly permissive, nor to bytes copied in
# from an untrusted archive, while the tag makes any such entry inert.

_CAPTURE_CACHE_SECRET_FILE = ".capture_cache_secret"
# Legacy pair-format sidecar suffix: recognized only for cleanup/eviction debris.
_CAPTURE_CACHE_TAG_SUFFIX = ".hmac"
_CAPTURE_CACHE_SECRET_BYTES = 32
_CAPTURE_CACHE_MAX_ENTRIES = 64
_CAPTURE_CACHE_MAX_BYTES = 2 * 1024**3
_CAPTURE_CACHE_MAGIC = b"TLCCv3\n"
_CAPTURE_CACHE_TAG_HEX_CHARS = 64  # HMAC-SHA256 hexdigest length
_CAPTURE_CACHE_HEADER_BYTES = len(_CAPTURE_CACHE_MAGIC) + _CAPTURE_CACHE_TAG_HEX_CHARS + 1


def _capture_cache_io_error(message: str) -> Exception:
    """Build the typed artifact-boundary error for a refused capture cache."""

    from ..._io import ArtifactError

    return ArtifactError(message)


def _harden_capture_cache_dir(directory: Path) -> None:
    """Make one torchlens-owned cache directory private to the current user.

    The default cache path is created under the caller's umask, which on a great many
    Linux installs (umask 002) yields a group-writable ``0775`` directory -- so a hard
    refusal would break the DEFAULT cache. Since torchlens owns these directories, the
    write bits are stripped instead. A directory owned by a different user, or one whose
    permissions cannot be tightened, is a genuine misconfiguration and refuses typed.

    Parameters
    ----------
    directory
        A cache directory torchlens created (the configured root, or its ``capture``
        subdirectory). Ancestors above the configured root are the caller's to secure
        and are deliberately not inspected.

    Returns
    -------
    None
        Returns normally once the directory is private.

    Raises
    ------
    torchlens.errors.ArtifactError
        When the directory is owned by another user, or is group/other-writable and
        cannot be tightened. Either condition means a second principal can substitute
        the pickle this process will load.
    """

    if os.name != "posix":  # pragma: no cover - POSIX mode bits are the checked signal
        return
    euid = os.geteuid()
    try:
        info = directory.stat()
    except OSError as exc:  # pragma: no cover - mkdir ran immediately before
        raise _capture_cache_io_error(
            f"TorchLens capture cache directory {directory} cannot be inspected ({exc})."
        ) from exc
    if info.st_uid != euid:
        raise _capture_cache_io_error(
            f"TorchLens capture cache directory {directory} is owned by uid "
            f"{info.st_uid}, not this process (uid {euid}). The cache stores pickled "
            "traces, so loading one written by another user would execute their code. "
            "Point TORCHLENS_CACHE_DIR at a directory you own, or drop cache=True."
        )
    permissive = info.st_mode & 0o022
    if not permissive:
        return
    try:
        directory.chmod(stat.S_IMODE(info.st_mode) & ~0o022)
    except OSError as exc:
        raise _capture_cache_io_error(
            f"TorchLens capture cache directory {directory} is group- or "
            f"world-writable (mode {stat.S_IMODE(info.st_mode):04o}) and could not be "
            f"tightened ({exc}). The cache stores pickled traces, so any principal who "
            "can write there could execute arbitrary code in this process on the next "
            f"cache hit. Run 'chmod go-w {directory}', point TORCHLENS_CACHE_DIR at a "
            "private directory, or drop cache=True."
        ) from exc
    if directory.stat().st_mode & 0o022:  # pragma: no cover - chmod silently ignored
        raise _capture_cache_io_error(
            f"TorchLens capture cache directory {directory} stayed group- or "
            "world-writable after chmod (a filesystem that ignores mode bits). Point "
            "TORCHLENS_CACHE_DIR at a private directory, or drop cache=True."
        )


def _prepare_capture_cache_dir(cache_dir_value: str | Path | None) -> tuple[Path, bytes]:
    """Create, harden, and key the capture-cache directory.

    Parameters
    ----------
    cache_dir_value
        Explicit ``cache_dir`` argument, or ``None`` to use ``TORCHLENS_CACHE_DIR`` /
        the ``~/.cache/torchlens`` default.

    Returns
    -------
    tuple[pathlib.Path, bytes]
        The per-capture entry directory and the secret authenticating its entries.
    """

    cache_dir = _capture_cache_dir(cache_dir_value)
    # ``mkdir(parents=True, mode=...)`` applies the mode to the LEAF only, so the
    # configured directory itself would keep the ambient umask permissions. Harden it
    # only when THIS call created it: a caller-supplied ``cache_dir`` that already exists
    # may be shared with other purposes, and silently tightening it would be a surprising
    # side effect on a path torchlens does not own. ``capture/`` is unambiguously ours and
    # is always hardened -- and the authentication tag, not the mode, is what makes a
    # permissive ancestor harmless (a planted entry carries no valid tag, and an
    # attacker-supplied secret fails the ownership check).
    created_cache_dir = not cache_dir.exists()
    cache_root = cache_dir / "capture"
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if created_cache_dir:
        _harden_capture_cache_dir(cache_dir)
    _harden_capture_cache_dir(cache_root)
    return cache_root, _capture_cache_secret(cache_root)


def _capture_cache_secret(cache_root: Path) -> bytes:
    """Return the per-root HMAC secret, creating it 0600 on first use.

    Parameters
    ----------
    cache_root
        Private capture-cache directory (already validated).

    Returns
    -------
    bytes
        The secret keying every entry's authentication tag.

    Raises
    ------
    torchlens.errors.ArtifactError
        When an existing secret file is not a private regular file, so a tag
        computed with it would prove nothing.
    """

    secret_path = cache_root / _CAPTURE_CACHE_SECRET_FILE
    if secret_path.exists():
        if secret_path.is_symlink() or not secret_path.is_file():
            raise _capture_cache_io_error(
                f"TorchLens capture cache secret {secret_path} is not a regular file; "
                "remove it (the cache refills automatically) or drop cache=True."
            )
        info = secret_path.stat()
        if os.name == "posix" and info.st_uid != os.geteuid():
            raise _capture_cache_io_error(
                f"TorchLens capture cache secret {secret_path} is owned by uid "
                f"{info.st_uid}, not this process; a tag keyed by it would prove "
                "nothing. Remove it (the cache refills automatically) or drop "
                "cache=True."
            )
        if os.name == "posix" and stat.S_IMODE(info.st_mode) & 0o077:
            raise _capture_cache_io_error(
                f"TorchLens capture cache secret {secret_path} is readable or writable "
                f"by other users (mode {stat.S_IMODE(info.st_mode):04o}); run "
                f"'chmod 600 {secret_path}' or remove it to have it regenerated."
            )
        try:
            return secret_path.read_bytes()
        except OSError as exc:
            raise _capture_cache_io_error(
                f"TorchLens capture cache secret {secret_path} cannot be read ({exc}); "
                "remove it (the cache refills automatically) or drop cache=True."
            ) from exc
    secret = os.urandom(_CAPTURE_CACHE_SECRET_BYTES)
    # O_EXCL so two concurrent first-captures cannot each believe they own the
    # secret; the loser re-reads the winner's bytes.
    try:
        descriptor = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # pragma: no cover - concurrent first capture
        return secret_path.read_bytes()
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(secret)
    return secret


class _CaptureCacheEntryOverCeilingError(Exception):
    """Raised mid-stream when a cache entry would exceed the byte ceiling."""


class _TaggingWriter:
    """File wrapper that HMACs every byte ``pickle.dump`` streams through it.

    Also enforces the write-side byte ceiling: an entry the load path would
    refuse at ``_CAPTURE_CACHE_MAX_BYTES`` must never be committed (it can
    never hit, it is rewritten on every capture, and its exempt-from-eviction
    bytes used to force the eviction pass to delete every OTHER valid entry).
    Raising mid-stream aborts the pickle at the ceiling instead of paying the
    full multi-GiB write first.
    """

    def __init__(self, handle: Any, mac: Any, byte_ceiling: int | None = None) -> None:
        self._handle = handle
        self._mac = mac
        self._byte_ceiling = byte_ceiling
        self._bytes_written = 0

    def write(self, data: Any) -> int:
        """Tag and forward one write, returning the bytes written."""

        self._bytes_written += len(data)
        if self._byte_ceiling is not None and self._bytes_written > self._byte_ceiling:
            raise _CaptureCacheEntryOverCeilingError
        self._mac.update(data)
        return cast(int, self._handle.write(data))


def _load_authenticated_capture_cache(cache_path: Path, secret: bytes) -> Any:
    """Unpickle a cache entry ONLY after its HMAC tag verifies.

    Parameters
    ----------
    cache_path
        Path of the cached pickle.
    secret
        Secret keying the entry's tag.

    Returns
    -------
    Any
        The cached ``Trace``, or ``None`` when the entry cannot be authenticated
        (treated as a cache miss; the caller recaptures and rewrites it).
    """

    import hashlib
    import hmac

    if cache_path.is_symlink():
        reason = "the entry is a symlink"
    else:
        # SINGLE read: the bytes that are authenticated MUST be the exact bytes that
        # are unpickled. Streaming the HMAC from one ``open`` and then unpickling from
        # a SECOND, independent ``open`` of the same path was a time-of-check/
        # time-of-use gap -- a second principal with write access to the
        # (attacker-writable-by-hypothesis) cache directory could swap the payload
        # after the tag verified over the benign bytes and before the load, turning a
        # cache hit into arbitrary code execution. Reading once binds authentication to
        # the exact bytes consumed (the embedded tag rides in the same read). The cost
        # is one transient buffer of the serialized trace, which ``pickle`` would
        # materialize as a live object graph regardless.
        data: bytes | None = None
        try:
            size = cache_path.stat().st_size
            if size > _CAPTURE_CACHE_MAX_BYTES + _CAPTURE_CACHE_HEADER_BYTES:
                reason = (
                    f"it is {size} bytes, above the {_CAPTURE_CACHE_MAX_BYTES}-byte "
                    "cache-entry ceiling"
                )
            else:
                data = cache_path.read_bytes()
        except OSError as exc:
            reason = f"it cannot be read ({exc})"
        if data is not None:
            header_end = _CAPTURE_CACHE_HEADER_BYTES
            if (
                not data.startswith(_CAPTURE_CACHE_MAGIC)
                or len(data) < header_end
                or data[header_end - 1 : header_end] != b"\n"
            ):
                reason = (
                    "it is not a single-record authenticated cache entry "
                    "(pre-upgrade pair format, or foreign bytes)"
                )
            else:
                recorded = data[len(_CAPTURE_CACHE_MAGIC) : header_end - 1].decode(
                    "ascii", errors="replace"
                )
                payload = data[header_end:]
                observed = hmac.new(secret, payload, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(recorded, observed):
                    reason = "its embedded authentication tag does not match its bytes"
                else:
                    return pickle.loads(payload)
    warnings.warn(
        f"Ignoring TorchLens capture cache entry {cache_path} because {reason}. The "
        "entry is NOT unpickled (unauthenticated pickles are never loaded); the "
        "capture runs normally and the entry is rewritten. "
        "torchlens.clear_cache() empties the cache.",
        UserWarning,
        stacklevel=2,
    )
    return None


def _store_authenticated_capture_cache(trace: Trace, cache_path: Path, secret: bytes) -> bool:
    """Commit a self-authenticating cache entry in ONE atomic step.

    The record is ``magic + hex HMAC tag + newline + pickled payload``. The
    payload is streamed through the tagging writer after a placeholder header,
    the real tag is seeked back into the header, and the finished record is
    installed by a single ``os.replace`` -- so no observer (crash recovery or
    concurrent reader) can ever see a payload paired with another generation's
    tag, which the historical payload + ``.hmac`` sidecar two-step commit
    allowed.

    Parameters
    ----------
    trace
        Trace to cache.
    cache_path
        Destination entry path.
    secret
        Secret keying the entry's embedded tag.

    Returns
    -------
    bool
        ``True`` when the entry was committed; ``False`` when the store was
        refused because the payload exceeds ``_CAPTURE_CACHE_MAX_BYTES`` (the
        load ceiling -- committing such an entry poisons the cache: it can
        never be loaded, and its bytes force eviction of every valid entry).
    """

    import hashlib
    import hmac

    mac = hmac.new(secret, digestmod=hashlib.sha256)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=cache_path.parent,
        prefix=f".{cache_path.name}.tmp.",
    )
    temporary_path = Path(temporary_name)
    try:
        # Streamed, so caching a multi-GiB trace does not additionally materialize
        # the whole pickle in memory just to tag it; the header placeholder is
        # overwritten in place once the streaming MAC settles.
        with os.fdopen(descriptor, "wb+") as file:
            file.write(_CAPTURE_CACHE_MAGIC + b"0" * _CAPTURE_CACHE_TAG_HEX_CHARS + b"\n")
            pickle.dump(trace, _TaggingWriter(file, mac, byte_ceiling=_CAPTURE_CACHE_MAX_BYTES))
            file.flush()
            file.seek(len(_CAPTURE_CACHE_MAGIC))
            file.write(mac.hexdigest().encode("ascii"))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, cache_path)
    except _CaptureCacheEntryOverCeilingError:
        temporary_path.unlink(missing_ok=True)
        warnings.warn(
            f"Not caching this capture: its serialized size is above the "
            f"{_CAPTURE_CACHE_MAX_BYTES}-byte cache-entry ceiling, so the entry could "
            "never be loaded back. The capture itself is unaffected; it simply will "
            "not hit the cache. Existing valid entries are left in place "
            "(torchlens.clear_cache() empties the cache).",
            UserWarning,
            stacklevel=2,
        )
        return False
    except (pickle.PickleError, TypeError, AttributeError, OSError) as exc:
        # A performance cache must degrade to "not cached", never annihilate a
        # capture that already succeeded (b6 R25, 3rd round): an unpicklable
        # Op.func -- e.g. a python-level Tensor method captured from a stock
        # nn.MultiheadAttention -- raised the bare PicklingError out of
        # tl.trace(..., cache=True) itself. Disk failures (ENOSPC) degrade
        # the same way.
        temporary_path.unlink(missing_ok=True)
        warnings.warn(
            f"Not caching this capture: serializing or writing the cache entry "
            f"failed ({type(exc).__name__}: {exc}). The capture itself is "
            "unaffected; it simply will not hit the cache. Existing valid "
            "entries are left in place (torchlens.clear_cache() "
            "empties the cache).",
            UserWarning,
            stacklevel=2,
        )
        return False
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return True


#: Age (seconds since last mtime) after which an orphaned ``.tmp.`` staging
#: file counts as hard-crash debris. An IN-FLIGHT writer's temp file keeps a
#: fresh mtime while ``pickle.dump`` streams into it, so an hour of mtime
#: silence cannot be a live store.
_CAPTURE_CACHE_TEMP_DEBRIS_AGE_SECONDS = 3600.0


def _sweep_stale_capture_cache_temp_files(cache_root: Path) -> None:
    """Remove orphaned ``mkstemp`` staging debris left by hard crashes.

    ``_store_authenticated_capture_cache`` unlinks its temp file on every
    EXCEPTION path, but a hard crash (SIGKILL, power loss) between
    ``mkstemp`` and the atomic ``os.replace`` strands ``.<name>.tmp.<rand>``
    files that no glob over ``*.pkl`` ever sees: they were invisible to both
    eviction accounting and ``clear_cache``, accumulating without
    bound. The sweep is age-gated so a concurrent in-flight store is never
    raced.
    """

    now = time.time()
    for temp_path in cache_root.glob(".*.tmp.*"):
        if temp_path.is_symlink():
            continue
        try:
            temp_stat = temp_path.stat()
        except OSError:
            continue
        if now - temp_stat.st_mtime > _CAPTURE_CACHE_TEMP_DEBRIS_AGE_SECONDS:
            temp_path.unlink(missing_ok=True)


def _evict_capture_cache(cache_root: Path, *, keep: Path) -> None:
    """Enforce capture-cache entry and byte limits using mtime LRU order.

    Parameters
    ----------
    cache_root:
        Directory containing authenticated cache entries.
    keep:
        Just-written entry, which is never evicted in the same operation.
    """

    _sweep_stale_capture_cache_temp_files(cache_root)
    entries: list[tuple[int, int, Path, Path]] = []
    for payload in cache_root.glob("*.pkl"):
        # Legacy pair-format ``.hmac`` sidecars are unreadable debris under the
        # single-record format; they ride along with their payload's eviction.
        tag = payload.with_name(payload.name + _CAPTURE_CACHE_TAG_SUFFIX)
        if payload.is_symlink() or tag.is_symlink():
            continue
        try:
            payload_stat = payload.stat()
        except OSError:
            continue
        mtime_ns = payload_stat.st_mtime_ns
        total_size = payload_stat.st_size
        if tag.is_file():
            try:
                tag_stat = tag.stat()
            except OSError:
                tag_stat = None
            if tag_stat is not None:
                mtime_ns = max(mtime_ns, tag_stat.st_mtime_ns)
                total_size += tag_stat.st_size
        entries.append((mtime_ns, total_size, payload, tag))
    entries.sort(reverse=True)
    total_bytes = sum(entry[1] for entry in entries)
    retained = len(entries)
    for _mtime_ns, size, payload, tag in reversed(entries):
        if retained <= _CAPTURE_CACHE_MAX_ENTRIES and total_bytes <= _CAPTURE_CACHE_MAX_BYTES:
            break
        if payload == keep:
            continue
        payload.unlink(missing_ok=True)
        tag.unlink(missing_ok=True)
        total_bytes -= size
        retained -= 1


# Capture-cache key INVERSION (M(oracles) item 6; listA row 26). The key was a
# hand-enumerated include-list, so semantic knobs added later (raise_on_nan,
# track_nonfinite, save_budget, ...) silently fell outside it and a warm cache
# served captures that never armed them -- fail-open on safety options. The
# key now covers EVERY CaptureOptions field by default: fields below in
# CAPTURE_CACHE_KEY_CURATED are represented by hand-built entries in the
# config dict (richer fragments than a bare value); fields in
# CAPTURE_CACHE_KEY_NEUTRAL are declared session-neutral WITH A REASON; every
# other field -- including any field added in the future -- is auto-included
# by the sweep at key-assembly time. A new field defaults INTO the key: a
# spurious miss is the safe direction, a false hit is the failure mode this
# inversion exists to kill.
CAPTURE_CACHE_KEY_CURATED: frozenset[str] = frozenset(
    {
        "layers_to_save",
        "save_raw_input",
        "output_transform",
        "save_transformed_output",
        "layer_visualizers",
        "save_visualizations",
        "keep_orphans",
        "output_device",
        "save_args",
        "save_grads",
        "capture_tensor_grad_hooks",
        "save_source_context",
        "save_rng",
        "seed",
        "source_context_lines",
        "optimizer",
        "compute_input_output_distances",
        "detach_saved_activations",
        "recurrence_detection",
        "replayable",
        "capture_container_structure",
        "track_device_memory",
        "measure_host_peak_memory",
        "aten_ops",
        "hooks",
        "backward_ready",
        "inference_only",
        "module_filter",
        "stop_after",
        "jax_control_flow",
        "jax_max_control_flow_unroll",
        "module_identity_mode",
        "payload_policy",
        "save_preview",
        "structure_only",
    }
)

CAPTURE_CACHE_KEY_NEUTRAL: dict[str, str] = {
    "batch_render": "presentation-only; re-stamped on every cache hit",
    "cache": "the cache machinery itself, not capture content",
    "cache_dir": "cache location, not capture content",
    "unwrap_when_done": "post-capture global wrapper lifecycle; trace content identical",
    "verbose": "console progress printing only",
}


def _sweep_option_fields_into_cache_config(
    cache_config: dict[str, Any],
    options_obj: Any,
    *,
    prefix: str,
    curated: frozenset[str] = frozenset(),
    neutral: Mapping[str, str] | None = None,
) -> None:
    """Auto-include every undeclared dataclass option field into the cache key.

    Parameters
    ----------
    cache_config:
        Mutable cache-key config being assembled.
    options_obj:
        Options dataclass instance to sweep.
    prefix:
        Namespace prefix for swept entries.
    curated:
        Field names already represented by hand-built config entries.
    neutral:
        Field-name -> reason ledger of declared session-neutral fields.
    """

    neutral_names = set(neutral or ())
    for option_field in dataclass_fields(options_obj):
        field_name = option_field.name
        if field_name.startswith("_") or field_name in curated or field_name in neutral_names:
            continue
        field_value = getattr(options_obj, field_name)
        if isinstance(field_value, Path):
            field_value = str(field_value)
        cache_config[f"{prefix}:{field_name}"] = _stable_cache_fragment(field_value)


def _store_capture_cache_entry(trace: Trace, path: Path, secret: Any) -> None:
    """Write the authenticated capture-cache entry and evict its stale siblings."""
    _prepare_log_for_capture_cache(trace)
    if _store_authenticated_capture_cache(trace, path, secret):
        _evict_capture_cache(path.parent, keep=path)


def _resolve_cache_key(state: _DoorState) -> _DoorState:
    """Name the trace and, when caching, derive the cache key, path and secret."""
    capture_options = state.capture_options
    name = capture_options.name
    log_name = name if name is not None else _state._auto_name(state.model)
    state = replace(state, log_name=log_name)
    if not capture_options.cache:
        return state
    config = _curated_cache_config(state)
    # The inversion sweep: every CaptureOptions field not hand-curated
    # above and not declared session-neutral enters the key here --
    # raise_on_nan, track_nonfinite, save_budget, distributed_witness,
    # measure_python_peak_memory, emit_nvtx, transform, name, and any
    # future field. Streaming options change what the returned trace
    # retains (and where), so they sweep in full as well.
    _sweep_option_fields_into_cache_config(
        config,
        capture_options,
        prefix="capture_option",
        curated=CAPTURE_CACHE_KEY_CURATED,
        neutral=CAPTURE_CACHE_KEY_NEUTRAL,
    )
    _sweep_option_fields_into_cache_config(
        config,
        state.streaming_options,
        prefix="streaming_option",
    )
    cache_key = _capture_cache_key(state.model, state.input_args, state.input_kwargs, config)
    cache_root, cache_secret = _prepare_capture_cache_dir(capture_options.cache_dir)
    return replace(
        state,
        cache_key=cache_key,
        cache_path=cache_root / f"{cache_key}.pkl",
        cache_secret=cache_secret,
    )


# Cache-key curation notes (the rows below, in order):
# * stop_after compiles INTO the halt slot, but its never-fired policy
#   differs by provenance, so the two spellings must not share a key:
#   a completed halt= capture could otherwise satisfy a stop_after=
#   request whose cold run would have refused stop_after_never_fired.
# * Capability / payload-policy options that change WHAT is captured or
#   stored in the returned trace. Omitting any of these let a second
#   trace() with a different capability silently return an earlier cached
#   trace built with the wrong capability (e.g. asked replayable=True,
#   got a cached False). Include them so a capability change misses the cache;
#   callables are repr-keyed (conservative -- a distinct object misses rather
#   than risking a false hit), matching how output_transform is keyed.
# * A cached capture without the measure_host_peak_memory opt-in holds no CPU peak.
def _curated_cache_config(state: _DoorState) -> dict[str, Any]:
    """Hand-curated cache-key rows; ``_resolve_cache_key`` sweeps the remaining fields in."""
    opts = state.capture_options
    args = state.args
    cache_config = {
        "layers_to_save": state.requested_layers_to_save,
        "keep_orphans": opts.keep_orphans,
        "output_device": opts.output_device,
        "save_args": opts.save_args,
        "capture_tensor_grad_hooks": opts.capture_tensor_grad_hooks,
        "save_grads": repr(state.save_grads_policy),
        "save_source_context": opts.save_source_context,
        "save_rng": opts.save_rng,
        "source_context_lines": opts.source_context_lines,
        "compute_input_output_distances": opts.compute_input_output_distances,
        "detach_saved_activations": opts.detach_saved_activations,
        "save_mode": state.save_mode_value,
        "recurrence_detection": opts.recurrence_detection,
        "backward_ready": state.train_mode_value,
        "inference_only": state.inference_only_value,
        "chunk_size": state.normalized_chunk_size,
        "chunk_paths": normalize_chunk_paths(state.chunk_paths_value),
        "capture_container_structure": opts.capture_container_structure,
        "output_transform": _stable_cache_fragment(opts.output_transform),
        "semantic_output_cache_key": semantic_output_cache_key(state.model),
        "facets": _facet_recipe_cache_key(state.facet_recipes),
        "save_predicate": _predicate_cache_key(state.save_predicate),
        "intervene": _predicate_cache_key(args.intervene),
        "halt": _predicate_cache_key(state.halt),
        "stop_after": _stable_cache_fragment(state.stop_after_site),
        "replayable": opts.replayable,
        "structure_only": opts.structure_only,
        "track_device_memory": opts.track_device_memory,
        "measure_host_peak_memory": opts.measure_host_peak_memory,
        "aten_ops": opts.aten_ops,
        "log_injections": opts.log_injections,
        "save_raw_input": repr(state.save_raw_input_policy),
        "save_transformed_output": repr(state.save_raw_output_policy),
        "save_raw_activations": state.save_options.save_raw_activations,
        "save_raw_gradients": state.save_options.save_raw_gradients,
        "activation_transform": _stable_cache_fragment(state.save_options.activation_transform),
        "grad_transform": _stable_cache_fragment(state.save_options.grad_transform),
        "seed": opts.seed,
        "module_filter": _stable_cache_fragment(opts.module_filter),
        "layer_visualizers": _stable_cache_fragment(opts.layer_visualizers),
        "save_visualizations": _stable_cache_fragment(opts.save_visualizations),
        "optimizer": repr(opts.optimizer),
        "hooks": _stable_cache_fragment(opts.hooks),
        "lookback": args.lookback,
        "lookback_payload_policy": args.lookback_payload_policy,
        "jax_control_flow": opts.jax_control_flow,
        "jax_max_control_flow_unroll": opts.jax_max_control_flow_unroll,
        "module_identity_mode": opts.module_identity_mode,
        "payload_policy": opts.payload_policy,
        "save_preview": opts.save_preview,
        "profile": state.profile_enabled,
    }
    return cache_config
