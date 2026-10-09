"""Public API entry points for TorchLens.

This module contains every user-facing function:
  - ``trace``  - the main entry point (runs model, returns Trace)
  - ``validate_forward_pass`` - replay-based correctness check
  - ``show_model_graph`` - visualization convenience wrapper
  - ``draw_backward`` - backward grad_fn_handle visualization wrapper
  - ``log_model_metadata`` - metadata-only convenience wrapper
  - ``validate_batch_of_models_and_inputs`` - bulk validation harness

**Selective save strategy**:
Predicate ``save=`` and most string/substring ``layers_to_save`` requests are
resolved during the primary forward pass. TorchLens falls back to the two-pass
discovery/replay path only for selectors that require finalized labels or
gradient-specific resolution.
"""

import collections.abc
import copy
import functools
import os
import pickle
import re
import stat
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import fields as dataclass_fields, replace
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn

from . import _state
from ._capture_intervention import (
    _backward_intervention_spec_from_predicate,
    _intervention_spec_from_hook_plan,
    _merge_intervention_spec_hooks,
    _record_capture_intervention_event,
)
from ._capture_state_helpers import (
    _capture_cache_dir,
    _capture_cache_key,
    _capture_output_metadata_from_model_config,
    _clone_state_dict_with_metadata as _clone_state_dict_with_metadata,
    _facet_recipe_cache_key,
    _fingerprint_model_weights,
    _hash_input_signatures,
    _input_id_for_relationship_evidence,
    _move_tensors_to_device,
    _prepare_log_for_capture_cache,
    _qualname_for_model,
    _reject_opaque_wrappers,
    _unwrap_data_parallel,
    decide_recording_of_batch as decide_recording_of_batch,
    unwrap_compiled_model,
)
from ._chunked_capture_helpers import (
    _append_chunk_trace_state,
    _should_store_auto_coerced_raw_input,
    _validate_chunked_forward_capture,
)
from ._chunking import iter_chunked_inputs, normalize_chunk_paths, normalize_chunk_size, plan_chunks
from ._deprecations import MISSING, MissingType
from ._errors import (
    ArgumentConflictError,
    ArgumentTypeError,
    CaptureContextError,
    InvalidArgumentError,
    KeywordConflictError,
    TorchLensPostfuncError,
)
from ._input_coerce import (
    _coerce_input_args,
    _reject_extra_positional_input,
    _reject_unrouted_forward_kwargs,
)
from ._io import TorchLensIOError
from ._io.streaming import BundleStreamWriter
from ._literals import (
    OutputDeviceLiteral,
)
from ._robustness import check_model_and_input_variants
from ._save_budget import SaveBudgetExceededError, SaveBudgetOption
from ._trace_selector_helpers import (
    _TRACE_OPTION_FILTERED_NAMES,
    _combine_save_predicates,
    _is_selective_label_save,
    _layers_to_save_has_negative_index,
    _layers_to_save_live_subset,
    _layers_to_save_mentions_identity,
    _layers_to_save_mentions_output,
    _layers_to_save_needs_final_resolution,
    _make_layers_to_save_predicate,
    _predicate_cache_key,
    _split_save_options_and_predicate,
    _stable_cache_fragment,
)
from ._trace_state import TraceState
from ._training_validation import TrainingModeConfigError, validate_training_compatibility
from .autoroute._builtin_output import semantic_output_cache_key
from .backends import (
    BackendName,
    BackendSpec,
    BackendUnsupportedError,
    get_backend_spec,
    resolve_backend_spec,
)
from .backends._options import (
    MLX_EXTRA_KWARG_POLICY,
    TRACE_OPTION_CAPABILITY_GATES,
    reject_extra_trace_kwargs,
)
from .backends.torch._tl import get_tensor_label
from .backends.torch.bound_root import (
    TLBoundMethodRoot,
    _reject_non_module_ladder_root,
    is_bound_method_of_module,
)
from .bridge import hf as _hf_bridge
from .capture._episode_failed import attach_failed_episode_ledger
from .capture._episode_ledger import (
    attach_episode_header,
    resolve_episode_declaration,
    write_episode_ledger,
)
from .capture._structure_only_entry import (
    _enforce_structure_only_entry_contract,
    _StructureOnlyEntryFacts,
)
from .capture.stop import StopDirective
from .data_classes._preprocessing_provenance import stamp_user_transform_provenance
from .data_classes.trace import (
    Trace,
)
from .fastlog.exceptions import PredicateError
from .fastlog.options import HaltPredicateFn, PredicateFn, RecordingOptions
from .fastlog.types import CaptureSpec
from .intervention import injection as _injection, model_door as _model_door
from .intervention._module_alias_guard import refuse_model_alias_spellings
from .intervention.errors import ChunkedForwardConfigError
from .intervention.hooks import normalize_hook_plan
from .intervention.predicates import InterventionPredicate
from .intervention.resolver import _selector_resolution_direction, resolve_sites
from .intervention.selectors import BaseSelector
from .intervention.types import InterventionDecision
from .ir import ParentEdge
from .ir.op_record import amend_graph_edge_insertion
from .ir.selector_eval import selector_contains_kind
from .options import (
    CaptureOptions,
    EpisodeSpec,
    ReplayOptions,
    SaveOptions,
    StreamingOptions,
    VisualizationOptions,
    merge_capture_options,
    merge_save_options,
    merge_streaming_options,
)
from .postprocess._selective_save import (
    apply_static_label_save_policy,
    reject_selector_outside_kinds,
)
from .types import ActivationPostfunc, GradientPostfunc
from .utils._torch_compat import is_dynamo_compiled_callable
from .utils.display import _vprint, ensure_trace_visualizer_dir, warn_parallel
from .utils.env_flags import closed_bool_env
from .utils.introspection import _get_code_context
from .utils.tensor_utils import SaveMode
from .visualization.code_panel import (
    capture_model_source_code,
    make_weak_model_ref,
)

_can_resolve_hf_processor = _hf_bridge._can_resolve_hf_processor
_can_resolve_hf_tokenizer = _hf_bridge._can_resolve_hf_tokenizer
_has_attached_image_processor = _hf_bridge._has_attached_image_processor
_is_hf_image_input = _hf_bridge._is_hf_image_input
_is_hf_multimodal_input = _hf_bridge._is_hf_multimodal_input
_is_hf_text_input = _hf_bridge._is_hf_text_input
_MLX_STATIC_LABEL_SAVE_SELECTOR_KINDS = frozenset(
    {"label", "func", "module", "contains", "in_module", "and", "or", "not"}
)

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

    from ._io import TorchLensIOError

    return TorchLensIOError(message)


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
    torchlens.errors.TorchLensIOError
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
    torchlens.errors.TorchLensIOError
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
        "torchlens.clear_capture_cache() empties the cache.",
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
            "(torchlens.clear_capture_cache() empties the cache).",
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
            "entries are left in place (torchlens.clear_capture_cache() "
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
    eviction accounting and ``clear_capture_cache``, accumulating without
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


def clear_capture_cache(cache_dir: str | Path | None = None) -> int:
    """Delete authenticated capture-cache entries while preserving the secret.

    Parameters
    ----------
    cache_dir:
        Cache root accepted by ``trace(cache_dir=...)``. ``None`` uses the
        configured default.

    Returns
    -------
    int
        Number of payload entries removed.
    """

    cache_root = _capture_cache_dir(cache_dir) / "capture"
    if not cache_root.exists():
        return 0
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise _capture_cache_io_error(f"Refusing non-directory capture cache {cache_root}.")
    _sweep_stale_capture_cache_temp_files(cache_root)
    removed = 0
    for payload in cache_root.glob("*.pkl"):
        if payload.is_symlink():
            continue
        tag = payload.with_name(payload.name + _CAPTURE_CACHE_TAG_SUFFIX)
        payload.unlink(missing_ok=True)
        if not tag.is_symlink():
            tag.unlink(missing_ok=True)
        removed += 1
    return removed


def list_logs() -> tuple[Trace, ...]:
    """Return a snapshot of currently live ``Trace`` objects.

    Returns
    -------
    tuple[Trace, ...]
        Immutable snapshot from TorchLens' process-wide weak registry.
    """

    return _state.list_logs()


def reset_naming_counter(class_name: str | None = None) -> None:
    """Reset automatic ``Trace`` naming counters.

    Parameters
    ----------
    class_name:
        Lowercase short class name to reset, or ``None`` to reset all counters.

    Returns
    -------
    None
        The process-global counter dictionary is updated.
    """

    _state.reset_naming_counter(class_name)


def _trace_mlx_model(
    model: object,
    input_args: object,
    input_kwargs: dict[Any, Any] | None,
    *,
    layers_to_save: str | list[Any] | None | MissingType,
    transform: Callable[[Any], Any] | None | MissingType,
    save_raw_input: str | bool | MissingType,
    batch_render: str | MissingType,
    output_transform: Callable[[Any], Any] | None | MissingType,
    output_style: str | None | MissingType,
    output_head: str | None | MissingType,
    save_raw_output: str | bool | MissingType,
    layer_visualizers: dict[Any, Callable[..., Any]] | None | MissingType,
    save_visualizations: bool | MissingType,
    keep_orphans: bool | MissingType,
    output_device: OutputDeviceLiteral | MissingType,
    activation_transform: ActivationPostfunc | None | MissingType,
    grad_transform: GradientPostfunc | None | MissingType,
    save_raw_activations: bool | MissingType,
    save_raw_gradients: bool | MissingType,
    capture_tensor_grad_hooks: bool | MissingType,
    save_arg_values: bool | MissingType,
    save_grads: bool | str | list[Any] | PredicateFn | BaseSelector | None | MissingType,
    save_code_context: bool | MissingType,
    save_rng_states: bool | MissingType,
    random_seed: int | None | MissingType,
    num_context_lines: int | MissingType,
    compute_input_output_distances: bool | MissingType,
    recurrence_detection: bool | MissingType,
    intervention_ready: bool | MissingType,
    hooks: Any | None | MissingType,
    capture: CaptureOptions | None,
    save: SaveOptions | None,
    save_predicate: PredicateFn | BaseSelector | None,
    visualization: VisualizationOptions | None,
    backward_ready: bool | MissingType,
    name: str | None | MissingType,
    module_filter: Callable[[Any], bool] | None | MissingType,
    module_identity_mode: str | None | MissingType,
    grad_options: Any | None | MissingType,
    verbose: bool | MissingType,
    intervene: Any | None = None,
    halt: Any | None = None,
) -> Trace:
    """Dispatch an MLX module capture through the optional MLX backend.

    Parameters
    ----------
    model, input_args, input_kwargs:
        MLX model and forward inputs.

    Returns
    -------
    Trace
        Captured technical-preview MLX trace.
    """

    if activation_transform is MISSING:
        resolved_activation_transform = None
    else:
        resolved_activation_transform = activation_transform
    capture_options = merge_capture_options(
        capture=capture,
        layers_to_save=layers_to_save,
        transform=transform,
        save_raw_input=save_raw_input,
        batch_render=batch_render,
        output_transform=output_transform,
        save_raw_output=save_raw_output,
        layer_visualizers=layer_visualizers,
        save_visualizations=save_visualizations,
        keep_orphans=keep_orphans,
        output_device=output_device,
        capture_tensor_grad_hooks=capture_tensor_grad_hooks,
        save_arg_values=save_arg_values,
        save_grads=save_grads,
        save_code_context=save_code_context,
        save_rng_states=save_rng_states,
        random_seed=random_seed,
        source_context_lines=MISSING,
        num_context_lines=num_context_lines,
        compute_input_output_distances=compute_input_output_distances,
        mark_layer_depths=MISSING,
        detach_saved_activations=MISSING,
        recurrence_detection=recurrence_detection,
        intervention_ready=intervention_ready,
        hooks=hooks,
        unwrap_when_done=MISSING,
        verbose=verbose,
        backward_ready=backward_ready,
        inference_only=MISSING,
        name=name,
        cache=MISSING,
        cache_dir=MISSING,
        module_filter=module_filter,
        module_identity_mode=module_identity_mode,
        stop_after=MISSING,
        raise_on_nan=MISSING,
    )
    save_options = merge_save_options(
        save=save,
        activation_transform=resolved_activation_transform,
        grad_transform=grad_transform,
        save_raw_activations=save_raw_activations,
        save_raw_gradients=save_raw_gradients,
    )
    if capture_options.intervention_ready:
        raise BackendUnsupportedError(
            "MLX backend does not support intervention_ready=True: post-capture "
            "rerun readiness requires PyTorch autograd integration not present in "
            "MLX. Live static-label trace(intervene=tl.when(...)) IS supported; "
            "omit intervention_ready or set False."
        )
    if capture_options.hooks:
        raise BackendUnsupportedError(
            "MLX backend does not support pre-attached hooks. "
            "Omit hooks or use the PyTorch backend."
        )
    if visualization is not None and visualization.view not in ["none", "rolled", "unrolled"]:
        raise InvalidArgumentError(
            f"MLX visualization mode={visualization.view!r} is not supported",
            code="visualization_mode_invalid",
            remedy="set visualization.view to 'none', 'rolled', or 'unrolled'",
            argument="visualization.view",
        )
    if capture_options.save_grads:
        raise BackendUnsupportedError("backward capture is not supported on the mlx backend")
    raw_input = None
    model_input_args = input_args
    model_input_kwargs = input_kwargs
    if capture_options.transform is not None:
        raw_input = input_args
        transformed_input = capture_options.transform(input_args)
        if isinstance(transformed_input, collections.abc.Mapping):
            model_input_args = []
            model_input_kwargs = dict(transformed_input)
        else:
            model_input_args = transformed_input
            model_input_kwargs = None
    from .backends.mlx import MLXBackend

    backend = MLXBackend()
    trace = backend.capture_trace(
        model,
        model_input_args,
        model_input_kwargs,
        layers_to_save=capture_options.layers_to_save,
        keep_orphans=capture_options.keep_orphans,
        output_device=capture_options.output_device,
        activation_transform=save_options.activation_transform,
        save_raw_activations=save_options.save_raw_activations,
        detach_saved_activations=capture_options.detach_saved_activations,
        save_grads=capture_options.save_grads,
        random_seed=capture_options.random_seed,
        num_context_lines=capture_options.source_context_lines,
        save_arg_values=capture_options.save_arg_values,
        save_code_context=capture_options.save_code_context,
        save_rng_states=capture_options.save_rng_states,
        recurrence_detection=capture_options.recurrence_detection,
        compute_input_output_distances=capture_options.compute_input_output_distances,
        verbose=capture_options.verbose,
        backward_ready=capture_options.backward_ready,
        name=capture_options.name,
        module_filter=capture_options.module_filter,
        transform=None,
        raw_input=raw_input,
        save_raw_input=capture_options.save_raw_input,
        batch_render=capture_options.batch_render,
        output_transform=capture_options.output_transform,
        save_raw_output=capture_options.save_raw_output,
        layer_visualizers=cast(
            "dict[Any, Callable[..., Any]] | None", capture_options.layer_visualizers
        ),
        save_visualizations=capture_options.save_visualizations,
        module_identity_mode=capture_options.module_identity_mode,
        grad_options=cast("Any", None if grad_options is MISSING else grad_options),
        intervene=intervene,
        halt=halt,
    )
    apply_static_label_save_policy(trace, save_predicate, backend_name="MLX")
    return trace


def _trace_mlx_model_from_public_kwargs(**kwargs: Any) -> Trace:
    """Dispatch MLX capture from the public ``trace`` keyword bundle.

    Parameters
    ----------
    **kwargs:
        Public ``trace`` keyword bundle captured before torch-specific normalization.

    Returns
    -------
    Trace
        Captured MLX trace.
    """

    # Idempotent when the registry entry already resolved it; load-bearing for
    # direct/autoroute callers so the internal flat keys are honored, not dropped.
    # intervene=/halt= are DISPATCHED options on MLX (static-label live
    # interventions); recipes= is refused typed by the capture path below.
    # None of the three may flow through the capability-gated reject helper:
    # with interventions=True that would raise the never-dispatches
    # conformance error instead of running or refusing them properly.
    # ``capture`` itself is NOT checked here (N5 fix): it is a grouped
    # CaptureOptions object, almost never strictly None/MISSING in practice,
    # and MLX supports many of its fields (layers_to_save, output_device,
    # ...). Blanket-rejecting the whole object made every
    # ``capture=CaptureOptions(...)`` call fail with "does not support:
    # capture" regardless of content; the real per-field support/rejection
    # already happens downstream in ``_trace_mlx_model`` (e.g.
    # ``capture_options.intervention_ready``, ``capture_options.hooks``).
    reject_extra_trace_kwargs(
        {
            "lookback": kwargs["lookback"],
            "lookback_payload_policy": kwargs["lookback_payload_policy"],
            "storage": kwargs["storage"],
            "streaming": kwargs["streaming"],
            "inference_only": kwargs.get("inference_only", MISSING),
            "cache": kwargs.get("cache", MISSING),
            "stop_after": kwargs.get("stop_after", MISSING),
            "raise_on_nan": kwargs.get("raise_on_nan", MISSING),
            "profile": kwargs.get("profile", MISSING),
            "payload_policy": kwargs.get("payload_policy", MISSING),
            "save_preview": kwargs.get("save_preview", MISSING),
            "chunk_size": kwargs.get("chunk_size", MISSING),
            "chunk_paths": kwargs.get("chunk_paths", MISSING),
            "save_outs_to": kwargs.get("save_outs_to", MISSING),
            "keep_outs_in_memory": kwargs.get("keep_outs_in_memory", MISSING),
            "out_sink": kwargs.get("out_sink", MISSING),
            "cache_dir": kwargs.get("cache_dir", MISSING),
            "save_mode": kwargs.get("save_mode", MISSING),
            "capture_tensor_grad_hooks": kwargs.get("capture_tensor_grad_hooks", MISSING),
            "save_raw_gradients": kwargs.get("save_raw_gradients", MISSING),
            "source_context_lines": kwargs.get("source_context_lines", MISSING),
            "unwrap_when_done": kwargs.get("unwrap_when_done", MISSING),
            "reconstruction_ready": kwargs.get("reconstruction_ready", MISSING),
        },
        MLX_EXTRA_KWARG_POLICY,
        spec=get_backend_spec("mlx"),
    )
    save_options, save_predicate = _split_save_options_and_predicate(kwargs["save"])
    if save_predicate is not None:
        reject_selector_outside_kinds(
            save_predicate,
            allowed=_MLX_STATIC_LABEL_SAVE_SELECTOR_KINDS,
            backend_name="MLX",
        )
    recipes_value = kwargs.get("recipes", MISSING)
    if recipes_value is not MISSING and recipes_value is not None:
        raise BackendUnsupportedError(
            "MLX backend supports interventions only through static-label "
            "trace(intervene=tl.when(...)) predicates; recipe specs (recipes=) "
            "are not supported. Use the PyTorch backend for intervention recipes."
        )
    # N5 fix: every name below except the genuine top-level ``trace()``
    # parameters (model, input_args, input_kwargs, capture, save,
    # grad_transform, grad_options, intervene, halt) moved into the grouped
    # ``capture=CaptureOptions(...)`` object and no longer reaches this
    # function as a flat key -- ``kwargs["activation_transform"]`` and
    # friends raised ``KeyError`` on every call once the sprint removed the
    # flat spelling from ``trace()``'s own signature. ``.get(..., MISSING)``
    # reports "not flatly specified" so ``_trace_mlx_model``'s own
    # ``merge_capture_options``/``merge_save_options`` calls fall through to
    # the grouped object's value, exactly like direct/internal callers who
    # still pass the flat spelling.
    activation_transform = kwargs.get("activation_transform", MISSING)
    return _trace_mlx_model(
        kwargs["model"],
        kwargs["input_args"],
        kwargs["input_kwargs"],
        layers_to_save=kwargs.get("layers_to_save", MISSING),
        transform=kwargs.get("transform", MISSING),
        save_raw_input=kwargs.get("save_raw_input", MISSING),
        batch_render=kwargs.get("batch_render", MISSING),
        output_transform=kwargs.get("output_transform", MISSING),
        output_style=kwargs.get("output_style", MISSING),
        output_head=kwargs.get("output_head", MISSING),
        save_raw_output=kwargs.get("save_raw_output", MISSING),
        layer_visualizers=MISSING,
        save_visualizations=MISSING,
        keep_orphans=kwargs.get("keep_orphans", MISSING),
        output_device=kwargs.get("output_device", MISSING),
        activation_transform=activation_transform,
        grad_transform=kwargs["grad_transform"],
        save_raw_activations=kwargs.get("save_raw_activations", MISSING),
        save_raw_gradients=kwargs.get("save_raw_gradients", MISSING),
        capture_tensor_grad_hooks=kwargs.get("capture_tensor_grad_hooks", MISSING),
        save_arg_values=kwargs.get("save_arg_values", MISSING),
        save_grads=kwargs.get("save_grads", MISSING),
        save_code_context=kwargs.get("save_code_context", MISSING),
        save_rng_states=kwargs.get("save_rng_states", MISSING),
        random_seed=kwargs.get("random_seed", MISSING),
        num_context_lines=kwargs.get("num_context_lines", MISSING),
        compute_input_output_distances=kwargs.get("compute_input_output_distances", MISSING),
        recurrence_detection=kwargs.get("recurrence_detection", MISSING),
        intervention_ready=kwargs.get("intervention_ready", MISSING),
        hooks=kwargs.get("hooks", MISSING),
        capture=kwargs["capture"],
        save=save_options,
        save_predicate=save_predicate,
        visualization=None,
        backward_ready=kwargs.get("backward_ready", MISSING),
        name=kwargs.get("name", MISSING),
        module_filter=kwargs.get("module_filter", MISSING),
        module_identity_mode=kwargs.get("module_identity_mode", MISSING),
        grad_options=kwargs["grad_options"],
        verbose=kwargs.get("verbose", MISSING),
        intervene=kwargs["intervene"],
        halt=kwargs["halt"],
    )


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
        "output_style",
        "output_head",
        "save_raw_output",
        "layer_visualizers",
        "save_visualizations",
        "keep_orphans",
        "output_device",
        "save_arg_values",
        "save_grads",
        "capture_tensor_grad_hooks",
        "save_code_context",
        "save_rng_states",
        "random_seed",
        "source_context_lines",
        "optimizer",
        "compute_input_output_distances",
        "detach_saved_activations",
        "recurrence_detection",
        "intervention_ready",
        "capture_container_structure",
        "track_device_memory",
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


# list-A row 30 latch: once-per-process disclosure that train-mode tracing
# advances norm running statistics. Process-global by design ("warn once");
# tests that assert the warning must save and restore this flag.
_BATCHNORM_TRAIN_STATS_WARNED = False


def _warn_once_train_mode_running_stats(model: nn.Module) -> None:
    """Disclose (once per process) that this capture mutates running stats.

    Parameters
    ----------
    model:
        Model about to run its real forward under capture.

    Returns
    -------
    None
        Warns with code ``batchnorm_train_stats_mutated`` and latches the
        process-global flag when a train-mode norm layer tracking running
        statistics is found; otherwise a no-op that leaves the flag unset
        (a later capture of a mutating model still discloses).
    """

    global _BATCHNORM_TRAIN_STATS_WARNED
    from torch.nn.modules.batchnorm import _BatchNorm
    from torch.nn.modules.instancenorm import _InstanceNorm

    mutating = [
        module
        for module in model.modules()
        if isinstance(module, (_BatchNorm, _InstanceNorm))
        and module.training
        and getattr(module, "track_running_stats", False)
        and getattr(module, "running_mean", None) is not None
    ]
    if not mutating:
        return
    _BATCHNORM_TRAIN_STATS_WARNED = True
    from .errors import TorchLensWarning as _TorchLensWarning

    warnings.warn(
        _TorchLensWarning(
            f"tracing runs the model's REAL forward: {len(mutating)} norm "
            "layer(s) in train mode with track_running_stats=True updated "
            "their running statistics in place during this capture "
            "(running_mean/running_var advance under momentum, even inside "
            "torch.no_grad or inference_only=True). This disclosure fires "
            "once per process. "
            "Remedy: call model.eval() before tracing for observational "
            "captures (model.train() restores the mode), or snapshot "
            "model.state_dict() beforehand to roll the statistics back; "
            "training-through-trace users can ignore this",
            code="batchnorm_train_stats_mutated",
        ),
        stacklevel=2,
    )


def _release_preparation_after_failed_capture(model: nn.Module) -> None:
    """Strip TorchLens preparation from a model whose capture failed.

    A FAILED capture must leave the model and TorchLens's preparation
    bookkeeping "as if never traced", whatever ended it: the session cleanup
    already put every submodule ``forward`` back, and this also evicts the
    module metadata and prepared-model registry entries so the next capture
    prepares from scratch. Every ``BaseException`` (``KeyboardInterrupt``,
    ``SystemExit``) takes this path, exactly like an ordinary ``Exception``.
    State the partial forward already mutated (e.g. norm running statistics)
    is NOT rolled back; that boundary is documented at the failure warning.

    Parameters
    ----------
    model:
        Model whose capture attempt raised. Releasing an unprepared model is
        a no-op, so chunked fan-outs that already released on an inner
        failure are safe.

    Returns
    -------
    None
        The model is released in place; a secondary release failure warns
        coded (``failed_capture_release_incomplete``) and never masks the
        capture exception.
    """

    from .backends.torch.model_prep import release_model

    try:
        release_model(model)
    except CaptureContextError as refusal:
        if refusal.fields.get("code") == "release_during_active_capture":
            # A NESTED capture failed while an outer capture still owns the
            # model's instrumentation (e.g. tl.trace called from a forward
            # hook): releasing here would strip the module metadata the live
            # capture is reading. The refusal is the guard working; the outer
            # capture's own failure/teardown path owns the release, so
            # nothing leaks by skipping.
            return
        _warn_failed_capture_release_incomplete(refusal)
    except Exception as release_exc:
        _warn_failed_capture_release_incomplete(release_exc)


def _warn_failed_capture_release_incomplete(release_exc: BaseException) -> None:
    """Disclose a secondary failure while releasing after a failed capture.

    Parameters
    ----------
    release_exc:
        The exception the release raised.

    Returns
    -------
    None
        Warns coded; never raises (the capture exception must propagate).
    """

    from .errors import TorchLensWarning as _TorchLensWarning

    warnings.warn(
        _TorchLensWarning(
            "TorchLens could not fully release its preparation of "
            f"the model after the failed capture "
            f"({type(release_exc).__name__}: {release_exc}); its module "
            "metadata and prepared-model bookkeeping may stay registered and "
            "plain attributes holding torch functions may stay unnormalized. "
            "Remedy: call tl.release_model(model) once the underlying "
            "condition is resolved",
            code="failed_capture_release_incomplete",
        ),
        stacklevel=4,
    )


def _warn_zero_match_capture_selectors(
    trace: Trace,
    *,
    save_selector: Any,
    intervene_selector: Any,
    intervene_direction: str | None,
    halt_selector: Any = None,
    layers_to_save_request: Any = None,
) -> None:
    """Warn when a capture-time save/intervention/halt selector matched no sites.

    Parameters
    ----------
    trace:
        Completed trace carrying selector fire counts.
    save_selector:
        Capture-time save selector, if configured.
    intervene_selector:
        Capture-time intervention selector, if configured.
    intervene_direction:
        Intervention direction whose live phase determines warning timing.
    halt_selector:
        Capture-time halt predicate, if configured. ``halt=`` was the one
        selector slot outside the zero-match disclosure family: a typo'd
        label selector silently ran the FULL forward (spending the memory/
        latency the halt was meant to avoid) and handed back the model's
        real outputs where the caller expected a frontier. Only ``BaseSelector``
        halts are judged -- an arbitrary value-dependent callable legitimately
        never firing is data, not a typo.
    layers_to_save_request:
        The ORIGINAL selective label-based ``layers_to_save`` request, when
        one was made. ``layers_to_save`` resolves through its own predicate/
        deferred machinery (never a ``BaseSelector``), so it sat outside this
        disclosure family: a typo'd layer name retained only the
        always-retained output tail and disclosed NOTHING, live and in the
        artifact -- the highest-traffic instance of the cc2cabbb class
        (grind-r6 b3 R15, opus MED). The request is re-resolved against the
        FINAL label space here; zero matches record and warn like the
        sibling slots.

    Returns
    -------
    None
        Emits at most one warning for each configured selector slot, and
        appends a matching string-only record to the PERSISTED
        ``trace.annotations["unmatched_capture_selectors"]`` ledger
        (B3R4-R15-1): the warning is the most losable disclosure kind, and
        without a durable record a zero-match ablation sweep read as "this
        layer does not matter" on the returned and saved Trace alike.
    """

    def _record_unmatched(slot: str, selector: Any, direction: str | None = None) -> None:
        """Append one zero-match fact to the persisted trace annotations."""

        annotations = getattr(trace, "annotations", None)
        if not isinstance(annotations, dict):
            return
        from ._io._source_privacy import _relativize_path_literals

        # R62: this ledger persists at every save level, so a selector repr
        # embedding an absolute host path (e.g. a file-payload predicate) must
        # go through the same path-relativization belt as every other
        # persisted repr.
        entry: dict[str, str] = {
            "slot": slot,
            "selector": _relativize_path_literals(repr(selector)),
        }
        if direction is not None:
            entry["direction"] = str(direction)
        annotations.setdefault("unmatched_capture_selectors", []).append(entry)

    defer_backward_intervention = False
    try:
        if (
            isinstance(save_selector, BaseSelector)
            and int(getattr(trace, "_tl_save_selector_fire_count", 0)) == 0
        ):
            _record_unmatched("save", save_selector)
            warnings.warn(
                f"Capture-time save selector {save_selector!r} matched zero sites; "
                "no activations were selected by it.",
                UserWarning,
                stacklevel=3,
            )
        if isinstance(intervene_selector, BaseSelector):
            selector_direction = _selector_resolution_direction(intervene_selector)
            defer_backward_intervention = (
                selector_direction == "backward" and intervene_direction in {"backward", "both"}
            )
        if (
            isinstance(intervene_selector, BaseSelector)
            and not defer_backward_intervention
            and intervene_direction in {"forward", "both"}
            and int(getattr(trace, "_tl_intervene_selector_fire_count", 0)) == 0
        ):
            _record_unmatched("intervene", intervene_selector, intervene_direction)
            warnings.warn(
                f"Capture-time intervention selector {intervene_selector!r} matched zero sites; "
                "no intervention fired.",
                UserWarning,
                stacklevel=3,
            )
        if isinstance(halt_selector, BaseSelector) and not bool(getattr(trace, "halted", False)):
            _record_unmatched("halt", halt_selector)
            warnings.warn(
                f"Capture-time halt selector {halt_selector!r} matched zero sites; "
                "the capture ran the full forward and completed without halting, so "
                "the outputs are the model's real outputs, not the intended frontier.",
                UserWarning,
                stacklevel=3,
            )
        if layers_to_save_request is not None:
            from .capture.trace import _get_op_nums_from_user_labels

            try:
                matched: Any = _get_op_nums_from_user_labels(
                    trace,
                    layers_to_save_request,
                )
            except InvalidArgumentError:
                # The post-capture lookup is loud for unknown keys; here the
                # capture already completed, so an unknown name IS the
                # zero-match fact to disclose, not grounds to destroy the
                # finished trace at return time.
                matched = []
            if not matched:
                _record_unmatched("layers_to_save", layers_to_save_request)
                warnings.warn(
                    f"layers_to_save={layers_to_save_request!r} matched zero layers; "
                    "only the always-retained output tail was saved. Check the "
                    "layer names against trace.layer_labels.",
                    UserWarning,
                    stacklevel=3,
                )
    finally:
        trace.__dict__.pop("_tl_save_selector_fire_count", None)
        if not defer_backward_intervention:
            trace.__dict__.pop("_tl_intervene_selector_fire_count", None)


def record_kpi_in_graph(name: str, value: Any) -> None:
    """Record a user KPI on the active capture graph.

    Parameters
    ----------
    name:
        KPI name.
    value:
        JSON-like value to attach to the current ``Trace``.

    Raises
    ------
    RuntimeError
        If no forward pass is being captured.
    """

    trace = _state._active_trace
    if trace is None:
        raise CaptureContextError(
            "record_kpi_in_graph() was called without an active trace",
            code="capture_context_required",
            remedy="call record_kpi_in_graph() from model.forward() while tl.trace() is running",
            operation="record_kpi_in_graph",
        )
    trace.annotations[str(name)] = value


def register_tensor_connection(parent: torch.Tensor, child: torch.Tensor) -> None:
    """Register a manual parent-child tensor edge during capture.

    Parameters
    ----------
    parent:
        Parent tensor already tagged by TorchLens.
    child:
        Child tensor already tagged by TorchLens.

    Raises
    ------
    RuntimeError
        If no forward pass is being captured.
    ValueError
        If either tensor has not been tagged by TorchLens.
    """

    trace = _state._active_trace
    if trace is None:
        raise CaptureContextError(
            "register_tensor_connection() was called without an active trace",
            code="capture_context_required",
            remedy=(
                "call register_tensor_connection() from model.forward() while tl.trace() is running"
            ),
            operation="register_tensor_connection",
        )
    parent_label = get_tensor_label(parent)
    child_label = get_tensor_label(child)
    if parent_label is None or child_label is None:
        raise InvalidArgumentError(
            "register_tensor_connection() received a tensor without a TorchLens capture label",
            code="tensor_connection_labels_missing",
            remedy="pass tensors produced by already-captured operations in the active trace",
            argument="parent/child",
        )
    trace.manual_tensor_connections.append((parent_label, child_label))
    _register_live_tensor_connection(trace, parent_label, child_label)


def _register_live_tensor_connection(
    trace: Trace,
    parent_label: str,
    child_label: str,
) -> None:
    """Register a parent-child edge on live capture records.

    Parameters
    ----------
    trace:
        Active trace receiving the manual edge.
    parent_label:
        Raw label for the parent tensor operation.
    child_label:
        Raw label for the child tensor operation.

    Returns
    -------
    None
        Mutates the live parent and child field mappings.
    """

    event = trace.capture_events.live_index.require_event(child_label)
    if parent_label in {edge.parent_label_raw for edge in event.parents}:
        return
    parent_arg_positions = copy.deepcopy(event.parent_arg_positions)
    parent_arg_positions.setdefault("args", {})[len(parent_arg_positions.get("args", {}))] = (
        parent_label
    )
    trace.capture_events.append_amendment(
        amend_graph_edge_insertion(
            event.seq,
            child_label,
            parents=(
                *event.parents,
                ParentEdge(parent_label_raw=parent_label, arg_position=None, edge_use="output"),
            ),
            parent_arg_positions=parent_arg_positions,
        )
    )


def _run_model_and_save_specified_outs(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None,
    layers_to_save: str | list[int | str] | None = "all",
    keep_orphans: bool = False,
    output_device: OutputDeviceLiteral = "same",
    activation_transform: ActivationPostfunc | None = None,
    grad_transform: GradientPostfunc | None = None,
    save_raw_activations: bool = True,
    save_raw_gradients: bool = True,
    save_mode: SaveMode = "copy",
    capture_tensor_grad_hooks: bool = True,
    mark_layer_depths: bool = False,
    detach_saved_activations: bool = False,
    save_arg_values: bool = False,
    save_grads: bool = False,
    grads_to_save: str | list[int | str] | None = "all",
    random_seed: int | None = None,
    num_context_lines: int = 7,
    optimizer: Any = None,
    save_code_context: bool = False,
    save_rng_states: bool = False,
    recurrence_detection: bool = True,
    save_outs_to: str | Path | None = None,
    keep_outs_in_memory: bool = True,
    stream_custom_attributes: bool = True,
    stream_buffer_values: bool = True,
    stream_async_writes: bool | None = None,
    stream_max_pending_bytes: int | None = None,
    grad_storage_path: str | Path | None = None,
    retain_grads_in_memory: bool = True,
    out_sink: Callable[[str, torch.Tensor], None] | None = None,
    intervention_ready: bool = False,
    capture_container_structure: bool = False,
    hooks: Any | None = None,
    intervention_spec: Any | None = None,
    normalized_hook_plan: Any | None = None,
    verbose: bool = False,
    backward_ready: bool = False,
    inference_only: bool = False,
    name: str | None = None,
    module_filter: Callable[[Any], bool] | None = None,
    emit_nvtx: bool = False,
    measure_python_peak_memory: bool = False,
    distributed_witness: str = "none",
    save_budget: SaveBudgetOption = "auto",
    raise_on_nan: bool = False,
    track_nonfinite: bool = False,
    track_device_memory: bool = False,
    structure_only: bool = False,
    log_injections: bool = False,
    transform: Callable[[Any], Any] | None = None,
    raw_input: Any | None = None,
    save_raw_input: str | bool = "small",
    batch_render: str = "auto",
    output_transform: Callable[[Any], Any] | None = None,
    output_style: str | None = None,
    output_head: str | None = None,
    save_raw_output: str | bool = "small",
    layer_visualizers: dict[Any, Callable[..., Any]] | None = None,
    save_visualizations: bool = False,
    recipes: list[Callable[[Any], dict[str, Any]]]
    | tuple[Callable[[Any], dict[str, Any]], ...]
    | None = None,
    save_predicate: PredicateFn | None = None,
    intervene_predicate: InterventionPredicate | None = None,
    halt_predicate: HaltPredicateFn | None = None,
    _halt_from_stop_after: bool = False,
    lookback: int = 0,
    lookback_payload_policy: str = "metadata_only",
    retain_output_parents_for_layers_to_save: bool = False,
    episode_resolved: Any | None = None,
    echo_options: Any | None = None,
    _selective_layers_to_save_request: object | None = None,
    _resolved_layer_nums_to_save: tuple[int, ...] | None = None,
    _resolved_grad_layer_nums_to_save: tuple[int, ...] | str | None = None,
    _deferred_retention_selector: Any = None,
    _deferred_gradient_selector: Any = None,
    _refresh_projection_capture: bool = False,
) -> Trace:
    """Run a forward pass with logging enabled, returning a populated Trace.

    This is the single internal entry point that creates a Trace, configures it,
    and delegates to ``Trace._run_and_log_inputs_through_model`` which handles
    model preparation, the exhaustive (and optionally fast) forward pass, and all
    postprocessing.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Positional arguments to model.forward(); a single tensor or list.
    input_kwargs:
        Keyword arguments to model.forward().
    layers_to_save:
        Which layers to save outs for ('all', 'none'/None, or a list).
    keep_orphans:
        If True, island ops are retained in raw metadata and exposed via
        ``trace.orphans`` while remaining hidden from the main graph.
    output_device:
        Device for saved tensors: 'same' (default), 'cpu', or 'cuda'.
        activation_transform: Optional transform applied to each out before storage
            (e.g., channel-wise averaging to reduce memory).
        grad_transform: Optional transform applied to each grad before storage.
        save_raw_activations: Whether raw outs are retained when ``activation_transform``
            is set. Metadata always describes the raw out.
        save_raw_gradients: Whether raw grads are retained when ``grad_transform`` is set.
            Metadata always describes the raw grad.
        save_mode: Tensor retention mode for saved activation and gradient payloads.
            ``"copy"`` is the safe cloning default; ``"reference"`` preserves the
            captured value through in-place handling; ``"view"`` is a live alias that
            downstream in-place operations can mutate; and ``"cpu_async"`` clones to CPU.
        capture_tensor_grad_hooks: Whether forward tensors receive tensor-level
            backward hooks for implicit backward events and per-op gradient payloads.
        mark_layer_depths: Compute BFS distances from input/output layers.
            Expensive for large graphs - off by default.
        detach_saved_activations: If True, saved tensors are detached from the autograd graph.
        save_arg_values: If True, store the non-tensor arguments to each function call.
            Required for validation replay (``validate_saved_outs``).
        save_grads: If True, register backward hooks to capture grads.
        grads_to_save: Which layer grads to save.
        random_seed: Fixed RNG seed for reproducibility (important for stochastic models).
            The capture snapshots the global RNG states (random/NumPy/torch CPU, plus
            CUDA when live), reseeds them for its forward, and restores the snapshot on
            every exit path, so code after the capture continues its own streams. When
            None the seed is drawn from a private entropy-seeded stream, so an outer
            ``torch.manual_seed`` does not make an unseeded capture reproducible. The
            seed used is recorded on ``trace.random_seed``.
        num_context_lines: Number of source-code context lines stored per function call.
        optimizer: Optional optimizer - used to tag which parameters have optimizers attached.
        recurrence_detection: If True (default), run full isomorphic subgraph expansion to
            detect repeated patterns (loops). Set this to False when the forward pass has
            more than about 1M operations and postprocessing speed matters; the False path
            skips the expensive expansion step and only groups operations that share the
            same parameters.
        save_outs_to: Optional portable bundle directory for streaming out save.
        stream_custom_attributes: Whether harvested module attributes are
            persisted in the streamed bundle (streaming counterpart of
            tl.save's include_custom_attributes).
        stream_buffer_values: Whether captured pre-forward buffer values are
            persisted in the streamed bundle (streaming counterpart of
            tl.save's include_buffer_values).
        stream_async_writes: Tri-state async-write routing for the streaming
            writer (StreamingOptions.async_writes): None/True arm the bounded
            async pipeline, False keeps synchronous per-blob writes.
        stream_max_pending_bytes: Pending snapshot byte budget for the async
            pipeline (None uses the writer default).
        keep_outs_in_memory: Whether streamed outs should remain in memory
            after finalization.
        grad_storage_path: Optional portable bundle directory for streaming grad save.
        retain_grads_in_memory: Whether streamed grads should remain in memory after
            backward finalization.
        out_sink: Optional callback invoked with ``(label, tensor)`` for each
            saved out.
        intervention_ready: If True, capture replay-template metadata and mark the
            returned log as eligible for intervention mutators, replay, rerun, and
            intervention spec persistence.
        capture_container_structure: If True, persist input and output container
            structure without enabling intervention replay metadata.
        hooks: Optional live forward post-hook plan. Accepts the same shapes as
            ``Trace.attach_hooks`` and executes during this capture when supplied.
        intervention_spec: Active intervention spec to expose in runtime context.
        normalized_hook_plan: Optional pre-normalized hook entries for internal engines.
            When ``intervention_spec`` is also supplied they must be its normalization:
            they drive live dispatch and are never merged back into that spec.
        verbose: If True, print timed progress messages at each major pipeline stage.
        backward_ready: If True, keep saved outs attached to autograd for training.
        inference_only: If True, wrap the user forward in ``torch.no_grad()``.
        name: User-facing log name. If omitted, generated by the public wrapper.
        emit_nvtx: If True, emit NVTX ranges around decorated torch operations.
            This is a profiling aid for CUDA/Nsight workflows and does not
            change graph construction or saved payloads.
        measure_python_peak_memory: If True, fold a ``tracemalloc``
            Python-allocation peak into the CPU/MPS ``forward_peak_memory``
            measurement. Off by default because the allocator hook taxes every
            traced operation.
        distributed_witness: Session-time witness level for collective boundary
            records captured under the distributed opt-in. ``"none"`` (default)
            records structure and correlation only; ``"digest"`` adds byte-exact
            SHA-256 contribution/destination digests (redundant merge evidence
            that can only demote, never rescue). ``"payload"`` is reserved.
        save_budget: Per-device ceiling on retained activation bytes. ``"auto"``
            (default) allows half of each device's available memory, a float sets
            another fraction, an int an absolute byte cap, and ``None`` disables
            the guard. Crossing it raises ``SaveBudgetExceededError`` naming the
            projected or committed footprint. The primary retained copy is admitted
            before allocation; this is not a general OOM-prevention guarantee.
        raise_on_nan: If True, stop capture at the first NaN or Inf tensor and raise
            ``CaptureError`` with the offending operation metadata.
        track_nonfinite: If True, record a per-op finiteness verdict for every
            committed op output, served by ``Trace.nonfinite_ops`` (session-time
            knob; never changes control flow).
        log_injections: Record intervention-hook torch calls as injected-op records.
        transform: Optional callable used to produce model-ready inputs from raw user input.
        raw_input: Original user input before ``transform`` was applied.
        save_raw_input: Portable save policy for the original raw input.
        batch_render: Raw-input batch rendering policy for visualization.
        output_transform: Optional callable used to produce human-readable
            output metadata from model output.
        output_style: Optional semantic output decode style.
        output_head: Optional live-output head to decode.
        save_raw_output: Portable save policy for the transformed raw output.
        layer_visualizers: Optional mapping from selectors to thumbnail visualizer callables.
        save_visualizations: Whether rendered thumbnails should persist in portable bundles.
        recipes: Per-trace additive facet recipes captured into the trace-owned
            immutable registry snapshot.
        save_predicate: Optional in-flight predicate controlling saved activation
            payloads during the exhaustive pass.
        intervene_predicate: Optional in-flight predicate controlling current-op
            interventions during the exhaustive pass.
        halt_predicate: Optional in-flight predicate that finalizes a partial trace
            at the matching source, operation, or module boundary.
        lookback: Number of recent events available to predicate-window queries.
        lookback_payload_policy: Candidate payload retention policy for retroactive
            ``followed_by`` saves. Memory cost is bounded by ``lookback`` times
            the candidate payload size.
        retain_output_parents_for_layers_to_save: Whether this predicate capture
            originated from selective ``layers_to_save`` and must preserve the
            legacy output-parent payload rule.
        _resolved_layer_nums_to_save: Internal refresh-only raw operation numbers
            already resolved against the original Trace.
        _resolved_grad_layer_nums_to_save: Internal refresh-only gradient operation
            numbers already resolved against the original Trace.
        _refresh_projection_capture: Whether this run supplies a RefreshProjector.

    Returns

    -------
        Fully-populated Trace.
    """
    # A shared module's alias spelling cannot select one call site: refuse before the forward.
    alias_sources = (save_predicate, intervene_predicate, halt_predicate, hooks, intervention_spec)
    deferred = (normalized_hook_plan, _deferred_retention_selector, _deferred_gradient_selector)
    refuse_model_alias_spellings(model, *alias_sources, *deferred)
    # Auto-detect model device from its first parameter and move inputs to match.
    # This prevents silent device-mismatch errors when the model is on CUDA but
    # the user ops CPU tensors (a common mistake). A META first parameter is
    # never a destination: offload-hooked models (accelerate device_map /
    # cpu/disk offload, lane F37) hold meta params between forwards and their
    # hooks place inputs on the real execution device themselves — moving
    # inputs to meta would poison the forward ("Cannot copy out of meta
    # tensor") that runs fine unlogged.
    model_device = next((p.device for p in model.parameters()), None)
    if model_device is not None and model_device.type != "meta":
        input_args = _move_tensors_to_device(input_args, model_device)
        if input_kwargs is not None:
            input_kwargs = _move_tensors_to_device(input_kwargs, model_device)

    if isinstance(model, TLBoundMethodRoot):
        # F41 bound-method root: the synthetic root's identity reads the
        # OWNER (type(owner).__name__ / its qualified class name), never the
        # TL-authored wrapper class or the string "method"; the entry-point
        # fact below discloses the bound_method invocation kind.
        model_class_name = model.tl_owner_class_name
        model_class_qualname = model.tl_owner_class_qualname
        root_entry_point = model.tl_root_entry_point
    else:
        model_class_name = str(type(model).__name__)
        model_class_qualname = _qualname_for_model(model)
        root_entry_point = f"module_call:{model_class_qualname}.forward"
    model_object_id = id(model)
    weight_fingerprint = _fingerprint_model_weights(model)
    input_object_id = _input_id_for_relationship_evidence(input_args)
    input_signature_hash = _hash_input_signatures(input_args, input_kwargs)
    # A rerun replays the REQUEST on its new graph: a staged edit's inserted op
    # shifts every later raw index, so the resolved save set does not transfer.
    rerun_save_request = {
        "layers_to_save": copy.copy(layers_to_save),
        "save_predicate": save_predicate,
        "lookback": lookback,
        "lookback_payload_policy": lookback_payload_policy,
        "retain_output_parents_for_layers_to_save": retain_output_parents_for_layers_to_save,
        "_deferred_retention_selector": _deferred_retention_selector,
    }
    module_save_selector = (
        save_predicate
        if isinstance(save_predicate, BaseSelector)
        and selector_contains_kind(save_predicate, "module")
        else None
    )
    if module_save_selector is not None:
        if _deferred_retention_selector is None:
            _deferred_retention_selector = module_save_selector
        elif isinstance(_deferred_retention_selector, list):
            _deferred_retention_selector = [
                *_deferred_retention_selector,
                module_save_selector,
            ]
        else:
            _deferred_retention_selector = [
                _deferred_retention_selector,
                module_save_selector,
            ]
        if backward_ready and _deferred_gradient_selector is None:
            _deferred_gradient_selector = "all"
        save_predicate = None
        layers_to_save = "none"

    module_intervene_selector = None
    lowered_intervene_spec = None
    module_intervene_entries: list[Any] = []
    candidate_intervene_selector = getattr(intervene_predicate, "selector", None)
    candidate_intervene_decision = getattr(intervene_predicate, "decision", None)
    if (
        isinstance(candidate_intervene_selector, BaseSelector)
        and selector_contains_kind(candidate_intervene_selector, "module")
        and isinstance(candidate_intervene_decision, InterventionDecision)
        and candidate_intervene_decision.hook is not None
        and candidate_intervene_decision.direction in {"forward", "both"}
    ):
        module_intervene_selector = candidate_intervene_selector
        # .selector is non-None only for single-rule specs, so rules[0] is
        # exact; the rule digest is unrecoverable downstream of this
        # lowering (W051-BIND 2.3b).
        spec_rules: Any = getattr(intervene_predicate, "rules", ())
        lowered_rule_id = spec_rules[0].rule_id
        module_intervene_entries = [
            replace(
                entry,
                metadata={
                    **dict(entry.metadata),
                    "created_by": "intervene_predicate",
                    "zero_match_ledger": "intervene_selector",
                    "rule_id": lowered_rule_id,
                },
            )
            for entry in normalize_hook_plan(
                candidate_intervene_selector,
                candidate_intervene_decision.hook,
                direction=candidate_intervene_decision.direction,
            )
        ]
        # Keep the original spec: the capture-door event must record the rules
        # payload the loader's injected-op anchor (W051-BIND 2.3c) vouches by.
        lowered_intervene_spec = intervene_predicate
        intervene_predicate = None
    hook_plan = list(normalized_hook_plan) if normalized_hook_plan is not None else []
    # A plan an internal engine pre-normalized FROM the caller's spec (rerun)
    # is already staged on that spec. Folding it back in re-appended the
    # spec's own hooks to the live spec object on every rerun, so the staged
    # plan doubled per call (1, 2, 4, 8) and each rerun fired it that often.
    spec_derived_count = len(hook_plan) if intervention_spec is not None else 0
    if intervention_spec is None:
        intervention_spec = _backward_intervention_spec_from_predicate(intervene_predicate)
    if hook_plan == [] and hooks:
        hook_plan = normalize_hook_plan(hooks)
    hook_plan.extend(module_intervene_entries)
    hook_plan_spec = _intervention_spec_from_hook_plan(hook_plan[spec_derived_count:])
    if intervention_spec is None:
        intervention_spec = hook_plan_spec
    elif hook_plan_spec is not None:
        intervention_spec = _merge_intervention_spec_hooks(intervention_spec, hook_plan_spec)
    # r8 R54 (sol 2-thread repro): the process-global runtime-context reset +
    # configure below ran BEFORE any admission check, so a concurrent capture
    # destined for the typed refusal first RESET the admitted winner's live
    # context (clearing its replay-template flag and intervention plan
    # mid-capture). Claim the capture slot FIRST; the inner claim in
    # ``run_and_log_inputs_through_model`` passes through on the token. The
    # slot is released exactly once on every settlement path.
    _capture_slot = _state.capture_reservation()
    _reservation_token = _capture_slot.__enter__()
    _reservation_live = [True]

    def _release_capture_slot() -> None:
        """Release the early admission claim exactly once."""

        if _reservation_live[0]:
            _reservation_live[0] = False
            _capture_slot.__exit__(None, None, None)

    try:
        _state.reset_capture_runtime_context()
        _state.configure_capture_runtime_context(
            hook_plan=hook_plan,
            intervention_spec=intervention_spec,
            capture_replay_templates=intervention_ready,
            model_object_id=model_object_id,
            model_class_qualname=model_class_qualname,
            weight_fingerprint=weight_fingerprint,
            input_object_id=input_object_id,
            input_signature_hash=input_signature_hash,
        )
    except BaseException:
        _release_capture_slot()
        raise
    from .snoop._entry import echo_forward_failure, finish_echo, open_echo_session

    echo_session: Any = None
    try:
        from .semantic import facets as facets_mod

        trace = Trace(
            model_class_name=model_class_name,
            output_device=output_device,
            activation_transform=activation_transform,
            grad_transform=grad_transform,
            save_raw_activations=save_raw_activations,
            save_raw_gradients=save_raw_gradients,
            save_mode=save_mode,
            capture_tensor_grad_hooks=capture_tensor_grad_hooks,
            keep_orphans=keep_orphans,
            save_arg_values=save_arg_values,
            save_grads=grads_to_save if save_grads else None,
            detach_saved_activations=detach_saved_activations,
            mark_layer_depths=mark_layer_depths,
            num_context_lines=num_context_lines,
            optimizer=optimizer,
            save_code_context=save_code_context,
            save_rng_states=save_rng_states,
            recurrence_detection=recurrence_detection,
            verbose=verbose,
            backward_ready=backward_ready,
            inference_only=inference_only,
            module_filter=module_filter,
            emit_nvtx=emit_nvtx,
            measure_python_peak_memory=measure_python_peak_memory,
            distributed_witness=distributed_witness,
            save_budget=save_budget,
            transform=transform,
            raw_input=raw_input,
            save_raw_input=save_raw_input,
            batch_render=batch_render,
            output_transform=output_transform,
            save_raw_output=save_raw_output,
            layer_visualizers=layer_visualizers,
            save_visualizations=save_visualizations,
            facet_registry_snapshot=facets_mod.snapshot(recipes),
        )
        # One session field carries the capture's save request: the torch
        # backend reads its output-parent retention flag during the forward,
        # and a legacy rerun replays it unless this is a refresh capture,
        # whose resolved raw-index save set the projector rebinds instead.
        trace._rerun_save_request = {
            **rerun_save_request,
            "replayable": _resolved_layer_nums_to_save is None,
        }
        if _resolved_layer_nums_to_save is not None:
            trace._refresh_resolved_layer_nums_to_save = list(_resolved_layer_nums_to_save)
        if _resolved_grad_layer_nums_to_save is not None:
            trace._refresh_resolved_grad_layer_nums_to_save = _resolved_grad_layer_nums_to_save
        if _deferred_retention_selector is not None:
            trace._deferred_retention_selector = _deferred_retention_selector
        if _deferred_gradient_selector is not None:
            trace._deferred_gradient_selector = _deferred_gradient_selector
        if _refresh_projection_capture:
            trace._refresh_projection_capture = True
        _capture_output_metadata_from_model_config(trace, model)
        trace._output_style = output_style
        trace._output_head = output_head
        trace._output_tokenizer = getattr(model, "_torchlens_output_tokenizer", None)
        trace._semantic_output_metadata = semantic_output_cache_key(
            model,
            output_style=output_style,
            output_head=output_head,
        )
        if intervention_spec is not None:
            trace._intervention_spec = intervention_spec
        trace.trace_label = name
        trace.code_context = _get_code_context(
            num_context_lines,
            source_loading_enabled=save_code_context,
        )
        forward_code = getattr(model.forward, "__code__", None)
        trace.forward_source_line = getattr(forward_code, "co_firstlineno", None)
        trace.intervention_ready = intervention_ready
        trace._capture_container_structure = capture_container_structure
        if hook_plan:
            trace.state = TraceState.LIVE_CAPTURED
        trace.model_object_id = model_object_id
        trace.model_class_qualname = model_class_qualname
        trace.param_hash_quick = weight_fingerprint
        trace.param_hash_full = weight_fingerprint
        # C07X (iv)/D10: unconditional write; F41's fail-closed gate reads it.
        # module_call for plain module roots, bound_method for the F41
        # TL-authored wrapper root (derived beside the identity fields above).
        trace.root_entry_point = root_entry_point
        trace.input_object_id = input_object_id
        trace.input_signature_hash = input_signature_hash
        trace._source_code_blob = capture_model_source_code(model)
        trace._source_model_ref = make_weak_model_ref(model)
        trace._out_sink = out_sink
        trace._keep_outs_in_memory = keep_outs_in_memory
        trace._grad_stream_retain_in_memory = retain_grads_in_memory
        trace._defer_streaming_bundle_finalization = grad_storage_path is not None
        trace._wrapper_runtime_ws.in_exhaustive_pass = True
        trace.raise_on_nan = raise_on_nan
        trace.track_nonfinite = track_nonfinite
        trace.track_device_memory = track_device_memory
        _injection.arm_injection_logging(trace, armed=log_injections)
        # L7a mode-marker prep (S2 SEAM, labeled): the flag DECLARES the mode
        # (memo sec 1.5) and stamps the mirror field here at entry. At S2
        # ratification the settlement-side stamp moves to the
        # capture/outcome.py witness machinery in the S2 author's PR; this
        # entry write stays as the declared-mode source of truth.
        trace.structure_only = structure_only
        trace._stop_directive = StopDirective(
            halt_options=getattr(trace, "_predicate_save_options", None),
            raise_on_nan=raise_on_nan,
            forward_error_mode="raise",
            inference_only=inference_only,
        )
        if (
            save_predicate is not None
            or intervene_predicate is not None
            or halt_predicate is not None
        ):
            predicate_history_size = lookback if lookback > 0 else 8
            default_save = save_predicate is None and layers_to_save == "all"
            default_op: bool | CaptureSpec = default_save
            if backward_ready:
                default_op = CaptureSpec(
                    save_out=default_save,
                    save_metadata=default_save,
                    keep_grad=True,
                )
            trace._predicate_save_options = RecordingOptions(
                keep_op=save_predicate,
                intervene=intervene_predicate,
                halt=halt_predicate,
                default_op=default_op,
                streaming=StreamingOptions(
                    bundle_path=save_outs_to,
                    retain_in_memory=keep_outs_in_memory,
                    include_custom_attributes=stream_custom_attributes,
                    include_buffer_values=stream_buffer_values,
                    async_writes=stream_async_writes,
                    max_pending_bytes=stream_max_pending_bytes,
                )
                if save_outs_to is not None
                else None,
                history_size=predicate_history_size,
                lookback=lookback,
                lookback_payload_policy=lookback_payload_policy,  # type: ignore[arg-type]
                on_predicate_error="fail-fast",
            )
            trace._halt_returns_partial_trace = halt_predicate is not None
            trace._stop_directive = StopDirective(
                halt_options=trace._predicate_save_options,
                raise_on_nan=raise_on_nan,
                forward_error_mode=trace._predicate_save_options.on_forward_error,
                inference_only=inference_only,
            )
            trace._predicate_history_size = predicate_history_size
            trace._predicate_lookback = lookback
            trace._predicate_lookback_payload_policy = lookback_payload_policy
        bundle_path = grad_storage_path if grad_storage_path is not None else save_outs_to
        if bundle_path is not None:
            trace._out_writer = BundleStreamWriter(
                bundle_path,
                include_custom_attributes=stream_custom_attributes,
                include_buffer_values=stream_buffer_values,
            )
            # None = consumer default: trace captures overlap blob writes
            # with the forward; False is the explicit synchronous opt-out.
            if stream_async_writes is not False:
                trace._out_writer.arm_async_writes(stream_max_pending_bytes)
        if episode_resolved is not None:
            # Pre-capture episode declaration marker: rides the partial
            # product on failure and is visible to postprocess consumers; the
            # settlement-time writer replaces it with the finalized ledger.
            attach_episode_header(trace, episode_resolved)
        # Echo narrator (snoop D1): one read-only observer per capture
        # (runtime-only state). Opened INSIDE the guarded region: the echo
        # sink opens its path eagerly, and a sink-open failure (missing
        # directory, unwritable path) used to escape between the slot claim
        # and the forward's own guard, leaking the capture reservation for as
        # long as the exception object lived (AUD-CODE 2.15) -- the next
        # tl.trace then refused reentrant_trace on an idle process.
        echo_session = open_echo_session(trace, echo_options)
    except BaseException:
        # A pre-forward setup failure (the Trace ctor, the echo sink open, or
        # any later pre-forward step) must not leak the capture-global runtime
        # context configured just above: the forward's own try/finally only
        # guards the window starting below. Reset here so the protected region
        # begins no later than configure_capture_runtime_context().
        _state.reset_capture_runtime_context()
        _release_capture_slot()
        raise
    try:
        # C03 live site-key minting (surgery Build 0c): when a configured
        # predicate addresses by structural site (tl.site), arm one streaming
        # minter for exactly this capture -- captures are single-threaded, so
        # the ContextVar scope is exact and unarmed surfaces keep their typed
        # capability refusal.
        from contextlib import nullcontext

        from .intervention.site_keys import (
            armed_live_minter,
            predicate_needs_live_site_keys,
        )

        live_minter_scope = (
            armed_live_minter()
            if predicate_needs_live_site_keys(save_predicate, intervene_predicate, halt_predicate)
            else nullcontext()
        )
        with live_minter_scope:
            trace._run_and_log_inputs_through_model(
                model,
                cast(torch.Tensor | list[Any], input_args),
                input_kwargs,
                layers_to_save,
                grads_to_save,
                random_seed,
                reservation_resume=_reservation_token,
            )
    except BaseException as exc:
        # Crash tail (snoop D5 tail 2): synchronous flush; note rides PEP 678.
        echo_forward_failure(echo_session, exc)
        # F5: postprocess pops ``_out_writer`` at its transient-state seam, so
        # a post-seam failure (teardown, streaming tail) reaches this handler
        # on a trace WITHOUT the attribute; the unguarded read used to mask
        # the real exception with AttributeError.
        out_writer = trace.__dict__.get("_out_writer")
        if out_writer is not None:
            if not getattr(out_writer, "_closed", False):
                out_writer.abort(str(exc))
            if isinstance(exc, Exception) and not isinstance(
                exc,
                (PredicateError, SaveBudgetExceededError, TorchLensIOError, TorchLensPostfuncError),
            ):
                raise TorchLensIOError("Streaming out save failed during forward pass.") from exc
        raise
    finally:
        _state.reset_capture_runtime_context()
        _release_capture_slot()
        if hasattr(trace, "_capture_container_structure"):
            delattr(trace, "_capture_container_structure")
    if isinstance(model, TLBoundMethodRoot):
        # F41 TL-authored-root disclosure (tlspec v9 entry-dark slot, C07X
        # item (iv)): the marker lands on the root op record -- the
        # output-boundary op records, the terminal records of the wrapper's
        # DAG -- coherent only beside the bound_method entry-point fact
        # (fail-closed load validation in _io/_forgery_identity_facts.py).
        for output_label in trace.output_layers:
            for output_op in trace[output_label].ops:
                output_op.tl_authored_root = True
    finish_echo(echo_session, trace)
    warning_intervene_decision = (
        candidate_intervene_decision
        if isinstance(candidate_intervene_decision, InterventionDecision)
        else getattr(intervene_predicate, "decision", None)
    )
    _warn_zero_match_capture_selectors(
        trace,
        save_selector=module_save_selector or save_predicate,
        intervene_selector=(
            module_intervene_selector or getattr(intervene_predicate, "selector", None)
        ),
        intervene_direction=getattr(warning_intervene_decision, "direction", None),
        # stop_after-compiled halts carry their OWN provenance-split never-fired
        # policy at the trace entry (typed refusal / coded warning); the generic
        # halt zero-match warning would double-disclose ahead of that refusal.
        halt_selector=None if _halt_from_stop_after else halt_predicate,
        layers_to_save_request=_selective_layers_to_save_request,
    )
    _record_capture_intervention_event(
        trace,
        intervene_predicate
        if intervene_predicate is not None
        else (
            lowered_intervene_spec
            if lowered_intervene_spec is not None
            else module_intervene_selector
        ),
    )
    return trace


def _render_layer_visualizers(
    trace: Trace,
    layer_visualizers: dict[Any, Callable[..., Any]],
) -> None:
    """Render configured per-layer visualizer thumbnails after capture.

    Parameters
    ----------
    trace:
        Completed trace whose layer outs may be rendered.
    layer_visualizers:
        Mapping from TorchLens site selectors to visualizer callables.
    """

    output_dir = ensure_trace_visualizer_dir(trace)
    visualizer_dir = output_dir / "visualizers"
    visualizer_dir.mkdir(parents=True, exist_ok=True)
    max_fanout = max(1, len(trace.layer_list))

    for selector, visualizer in layer_visualizers.items():
        try:
            selected_ops = tuple(resolve_sites(trace, selector, max_fanout=max_fanout))
        except Exception as exc:
            warnings.warn(
                f"Skipping layer visualizer for selector {selector!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        for op in selected_ops:
            _render_one_layer_visualizer(visualizer_dir, op, visualizer)


def _render_one_layer_visualizer(
    visualizer_dir: Path,
    op: Any,
    visualizer: Callable[..., Any],
) -> None:
    """Render one op visualizer and store its output path on the op.

    Parameters
    ----------
    visualizer_dir:
        Directory that receives rendered files.
    op:
        Layer operation to render.
    visualizer:
        Callable accepting ``(tensor, *, layer_label=None)``.
    """

    if not bool(getattr(op, "has_saved_activation", False)) or getattr(op, "out", None) is None:
        return
    try:
        rendered = visualizer(op.out, layer_label=getattr(op, "layer_label", None))
        if rendered is None:
            return
        safe_label = _safe_visualizer_filename(str(op.layer_label))
        if isinstance(rendered, str):
            output_path = visualizer_dir / f"{safe_label}.html"
            output_path.write_text(rendered, encoding="utf-8")
        else:
            output_path = visualizer_dir / f"{safe_label}.png"
            rendered.save(output_path)
        op.visualizer_path = str(output_path)
    except Exception as exc:
        warnings.warn(
            f"Layer visualizer failed for {getattr(op, 'layer_label', '<unknown>')}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _safe_visualizer_filename(label: str) -> str:
    """Return a filesystem-safe visualizer basename for a layer label.

    Parameters
    ----------
    label:
        Layer label to encode.

    Returns
    -------
    str
        Safe filename stem.
    """

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label.replace(":", "pass")).strip("_") or "layer"


def _trace_option_explicit(option_name: str, public_trace_kwargs: dict[str, Any]) -> bool:
    """Return whether a public trace option was explicitly supplied.

    Parameters
    ----------
    option_name:
        Public trace option name.
    public_trace_kwargs:
        Mutable public trace keyword bundle.

    Returns
    -------
    bool
        ``True`` when the flat option or grouped ``CaptureOptions`` field was
        explicitly supplied by the caller.
    """

    flat_value = public_trace_kwargs.get(option_name, MISSING)
    if flat_value is not MISSING:
        return True
    capture_options = public_trace_kwargs.get("capture")
    if isinstance(capture_options, CaptureOptions) and hasattr(capture_options, option_name):
        return capture_options.is_field_explicit(option_name)
    return False


def _unsupported_trace_option_message(option_name: str, backend_name: str) -> str:
    """Return an actionable unsupported-option message.

    Parameters
    ----------
    option_name:
        Public trace option name.
    backend_name:
        Resolved backend name.

    Returns
    -------
    str
        Error message for unsupported explicit option use.
    """

    if option_name == "jax_static_argnums":
        return "jax_static_argnums is only supported with backend='jax'."
    if option_name == "grad_options":
        return (
            "grad_options is only supported with backend='jax', backend='mlx', "
            "backend='tinygrad', backend='paddle', or backend='tf'."
        )
    if option_name in {"jax_control_flow", "jax_max_control_flow_unroll"}:
        return (
            f"backend={backend_name!r} does not yet support {option_name}. "
            "JAX control-flow unrolling is declared but not implemented in this backend phase; "
            "omit the option or use backend='torch'."
        )
    if option_name == "module_identity_mode":
        return (
            f"backend={backend_name!r} does not yet support module_identity_mode selection. "
            "Module-mode selection is declared but not implemented for this backend phase; "
            "omit the option or use backend='torch'."
        )
    if option_name == "payload_policy":
        return (
            f"backend={backend_name!r} does not yet support payload_policy. "
            "Non-torch payload codec policy is declared but not implemented in this backend "
            "phase; omit the option or use backend='torch'."
        )
    if option_name == "save_preview":
        return (
            f"backend={backend_name!r} does not yet support save_preview. "
            "Preview save semantics are declared but not implemented for this backend phase; "
            "omit the option or use backend='torch'."
        )
    return (
        f"backend={backend_name!r} does not support trace option {option_name!r}; "
        "omit the option or choose a backend that declares it."
    )


def _filter_trace_kwargs_for_backend(
    public_trace_kwargs: dict[str, Any],
    resolved_spec: BackendSpec,
) -> None:
    """Strip unsupported omitted options and reject unsupported explicit ones.

    Parameters
    ----------
    public_trace_kwargs:
        Mutable public trace keyword bundle passed to the backend entry.
    resolved_spec:
        Backend selected for this trace call.

    Returns
    -------
    None
        ``public_trace_kwargs`` is updated in place.
    """

    supported_trace_options = set(resolved_spec.capabilities.trace_options)
    backend_name = str(resolved_spec.name)
    for option_name in _TRACE_OPTION_FILTERED_NAMES:
        if option_name in supported_trace_options:
            continue
        if _trace_option_explicit(option_name, public_trace_kwargs):
            raise BackendUnsupportedError(
                _unsupported_trace_option_message(option_name, backend_name)
            )
        public_trace_kwargs.pop(option_name, None)


def _capability_option_requested(value: Any) -> bool:
    """Return whether a gated public option value actually requests behavior.

    ``MISSING``, ``None``, and explicit ``False`` are all "off"; identity
    checks avoid calling ``bool()`` on predicate/selector values.
    """

    return not (value is MISSING or value is None or value is False)


def _enforce_capability_option_gates(
    public_trace_kwargs: dict[str, Any],
    resolved_spec: BackendSpec,
) -> None:
    """Refuse gated public options whose owning capability flag is ``False``.

    The registered capability table is the load-bearing authority in BOTH
    directions for EVERY backend, torch included: a flag flipped to ``False``
    must refuse the corresponding public surface typed instead of silently
    running (or silently ignoring) it. Options settable through
    ``capture=CaptureOptions(...)`` are gated on their explicit fields too.

    Parameters
    ----------
    public_trace_kwargs:
        Public trace keyword bundle about to be dispatched.
    resolved_spec:
        Backend spec selected for this trace call.

    Returns
    -------
    None
        Returns when every requested gated option's flag is ``True``.
    """

    capture_value = public_trace_kwargs.get("capture")
    for option_name, flag in TRACE_OPTION_CAPABILITY_GATES.items():
        if getattr(resolved_spec.capabilities, flag):
            continue
        requested = _capability_option_requested(public_trace_kwargs.get(option_name, MISSING))
        if (
            not requested
            and isinstance(capture_value, CaptureOptions)
            and hasattr(capture_value, option_name)
            and capture_value.is_field_explicit(option_name)
        ):
            requested = _capability_option_requested(getattr(capture_value, option_name))
        if requested:
            raise BackendUnsupportedError(
                f"backend {str(resolved_spec.name)!r} declares capability "
                f"{flag}=False, so explicit {option_name} is refused instead of "
                "silently ignored or run unsupported. Use a backend whose "
                f"registered capability table declares {flag}=True (e.g. "
                "backend='torch')."
            )


def _reject_unsupported_torch_trace_option_values(capture_options: CaptureOptions) -> None:
    """Reject explicit torch trace-option values that torch does not implement.

    Parameters
    ----------
    capture_options:
        Normalized capture options for a torch trace call.

    Returns
    -------
    None
        Returns when all explicit values are supported or default-equivalent.
    """

    if capture_options.is_field_explicit("module_identity_mode") and (
        capture_options.module_identity_mode not in {None, "torch_module"}
    ):
        raise BackendUnsupportedError(
            "backend='torch' supports module_identity_mode=None or 'torch_module' only."
        )
    if capture_options.is_field_explicit("payload_policy") and (
        capture_options.payload_policy not in {None, "full"}
    ):
        raise BackendUnsupportedError(
            "backend='torch' supports payload_policy=None or 'full' only."
        )
    if capture_options.is_field_explicit("save_preview") and capture_options.save_preview:
        raise BackendUnsupportedError("backend='torch' does not support save_preview=True.")
    for option_name in ("jax_control_flow", "jax_max_control_flow_unroll"):
        if capture_options.is_field_explicit(option_name):
            raise BackendUnsupportedError(
                f"backend='torch' does not support explicit {option_name}; "
                "JAX control-flow options are only meaningful with backend='jax'."
            )


def render(*args: Any, **kwargs: Any) -> Any:
    """One-call model picture over the quickstart input ladder (F17 B15).

    ONE metadata-only eval/no-grad capture (training flags, RNG, and norm
    buffers restored), rendered with the facade default ``collapse="auto"``
    (``Trace.draw`` keeps ``"none"``). Detached
    :class:`torchlens.quickstart.RenderResult`; never auto-opens a viewer.
    See :func:`torchlens.quickstart.render` for the parameters.
    """

    from .quickstart._render import render as _render_impl

    return _render_impl(*args, **kwargs)


def trace(
    model: nn.Module | Callable[..., Any],
    input_args: str | torch.Tensor | list[Any] | tuple[Any, ...] | None = None,
    input_kwargs: dict[Any, Any] | None = None,
    grad_transform: GradientPostfunc | None | MissingType = MISSING,
    save_mode: SaveMode | MissingType = MISSING,
    reconstruction_ready: bool | MissingType = MISSING,
    capture: CaptureOptions | None = None,
    save: SaveOptions | PredicateFn | BaseSelector | None = None,
    intervene: InterventionPredicate | None = None,
    halt: HaltPredicateFn | None = None,
    lookback: int = 0,
    lookback_payload_policy: str = "metadata_only",
    storage: StreamingOptions | None = None,
    streaming: StreamingOptions | None = None,
    profile: bool | MissingType = MISSING,
    recipes: (
        list[Callable[[Any], dict[str, Any]]]
        | tuple[Callable[[Any], dict[str, Any]], ...]
        | None
        | MissingType
    ) = MISSING,
    *,
    grouping: str | MissingType = MISSING,
    echo: Any | None = None,
    jax_static_argnums: int | Sequence[int] | MissingType = MISSING,
    grad_options: Any | None | MissingType = MISSING,
    episode: EpisodeSpec | None = None,
    chunk_size: int | None | MissingType = MISSING,
    chunk_paths: Iterable[Any] | None | MissingType = MISSING,
    backend: BackendName | None = None,
    input_size: Any | None = None,
    **forward_kwargs: Any,
) -> Trace:
    """Run a forward pass through *model*, log every operation, and return a Trace.

    This is the primary user-facing entry point for TorchLens.  It intercepts every
    tensor-producing operation during ``model.forward()``, records metadata and
    (optionally) saves outs, then returns a ``Trace`` that provides
    dict-like access to every layer's data.

    Torch functions are automatically wrapped on the first call and stay wrapped
    afterward.  Pass ``capture=CaptureOptions(unwrap_when_done=True)`` to restore
    the original torch callables after logging completes.

    **Layer selection** (``save=``, the canonical spelling):

    - ``'all'`` (default) - save outs for every layer.
    - ``'none'`` / ``None`` / ``[]`` - save no outs (metadata only).
    - A predicate selector, e.g. ``tl.func("relu")``, ``tl.in_module("encoder")``,
      or combinators such as ``tl.func("conv2d") & tl.followed_by(tl.func("relu"))``.
    - A list containing any mix of:
      1. Layer name, e.g. ``'conv2d_1_1'`` (all ops).
      2. Pass-qualified label, e.g. ``'conv2d_1_1:2'`` (second pass only).
      3. Module address, e.g. ``'features.0'`` (output of that module).
      4. Integer index (ordinal position; negative indices work).
      5. Substring filter, e.g. ``'conv2d'`` (all matching layers).

    Most string and substring layer selections are absorbed into a single-pass
    predicate save. TorchLens falls back to the two-pass discovery/replay path
    only for selectors that require finalized labels, such as negative indexes,
    identity/output labels, or gradient-specific selection.

    Parameters
    ----------
    model:
        PyTorch model, or a bound method of one (the ruled root contract,
        F41): ``tl.trace(model.generate, ids, ...)`` resolves the owner via
        ``method.__self__``, wraps it in a TL-authored synthetic root, and
        calls the method exactly once.
    input_args:
        Positional args for ``model.forward()``; a single tensor or list.
        The input ladder (memo D2): a real input is the gold rung; omit
        it and pass ``input_size=``, or omit both to infer -- synthesized
        values, disclosed, or a teach; mixing refuses ``input_rung_conflict``.
    input_kwargs:
        Keyword args for ``model.forward()``.
    input_size:
        Declared input shape(s) (torch backend only): one flat positive-int
        tuple, a sequence of shape tuples, or a mapping of forward keyword
        names to shapes (batch included exactly as written). Dtype/value
        recipes are fail-closed static facts (override:
        ``torchlens.quickstart.InputSpec``); local seed-0 synthesis,
        disclosed in the trace's persistent provenance record.
    save:
        Which layers to save outs for; the canonical selection kwarg (see
        **Layer selection** above). Accepts ``'all'``, ``'none'``/``None``/``[]``,
        label/module/index/substring lists, predicate selectors
        (``tl.func``, ``tl.in_module``, ``tl.followed_by``, ``tl.when``
        combinators), or a grouped ``SaveOptions``.
    grad_transform:
        Optional function applied to each grad before saving. The raw
        grad remains in ``layer.grad`` by default, and the transform result is stored
        in ``layer.transformed_grad``.
    save_mode:
        Tensor retention mode for saved activation and gradient payloads.
        ``"copy"`` is the safe cloning default; ``"reference"`` preserves the
        captured value through in-place handling; ``"view"`` is a live alias that
        downstream in-place operations can mutate; and ``"cpu_async"`` clones to CPU.
    reconstruction_ready:
        If True, auto-enable the argument and RNG capture
        prerequisites needed by read-only reconstructed facets such as fused
        SDPA ``scores``, ``pattern``, and ``z``.
    capture:
        Grouped capture options (``CaptureOptions``). The one spelling for
        every capture knob (``layers_to_save``, ``random_seed``,
        ``intervention_ready``, ``verbose``, ``structure_only``, ...); the
        former flat kwargs are removed.
    intervene:
        Optional predicate returning an intervention decision for
        current-op live mutation.
    halt:
        Optional predicate returning ``True`` to stop after the matching
        source, operation, or module-boundary event and return the partial trace.
    lookback:
        Number of recent capture events queryable by predicate-window helpers.
    lookback_payload_policy:
        Retention policy for retroactive ``followed_by`` saves.
        ``"metadata_only"`` keeps the default metadata-only window and cannot
        retroactively save payloads. Non-default policies retain up to ``lookback``
        candidate payloads, for a memory cost of roughly ``lookback`` times the
        candidate payload size.
    storage:
        Shared storage routing option. ``storage=tl.to_disk(path)``
        streams predicate-selected saves to a disk bundle during the
        forward pass. ``None`` preserves the existing in-RAM behavior.
    streaming:
        Grouped streaming-save options (``StreamingOptions``).
    chunk_size:
        If supplied, split a positional tensor input into forward
        chunks of this size along dimension 0 and append them into one
        in-memory ``Trace``. Forward-only and torch-only.
    chunk_paths:
        Optional explicit tensor leaf paths to split when multiple
        batched tensor leaves are present.
    profile:
        If True, explicitly marks the returned trace as profiled. Phase timings are
        always populated on ``trace._phase_timings``.
    recipes:
        Per-trace additive facet recipes captured into the immutable
        registry snapshot for the returned trace.
    grouping:
        UNSTABLE (no deprecation shim owed). Closed-vocabulary grouping
        policy: ``"structural"`` (the default -- today's recurrence
        grouping), ``"strict_shapes"`` (reserved; refuses typed until its
        own reviewed design lands), ``"fold_sites"`` (the D1 folding axis;
        refuses typed on plain captures until an affirmative D1 ruling).
        The requested value is recorded on ``trace.grouping`` and the
        policy that actually ran on ``trace.grouping_policy``. Distinct
        from the display-only ``fold_repeats`` viz knob, which folds
        repeated module runs at RENDER time and never changes grouping.
    echo:
        Live narration (DOCUMENTED-UNSTABLE, lane F28): ``True`` narrates
        every completed tensor-output event plus module structure lines;
        ``"modules"`` narrates structure only; a live selector scopes the
        stream; ``tl.options.EchoOptions`` groups the full surface (stats
        rungs, sink, crash tail). Narration is a display of capture
        events -- independent of ``save=`` retention. Torch-only.
    jax_static_argnums:
        JAX-only positional argument indexes passed to
        ``jax.make_jaxpr(..., static_argnums=...)`` when
        ``backend="jax"``. Non-default values require the explicit JAX
        backend.
    grad_options:
        Backend-specific derived-gradient options for the
        leaf-level preview. Supported by explicit ``backend="jax"`` and
        ``backend="tinygrad"`` only.
    episode:
        Torch-only episode capture declaration (``EpisodeSpec``).
    backend:
        Explicit backend name. ``None`` preserves legacy auto-resolution.
    **forward_kwargs:
        Never accepted -- unknown keywords refuse typed with
        ``trace_forward_kwargs_unrouted`` naming the ``input_kwargs=`` routing.

    Postfunc behavior:
        ``save.activation_transform`` and ``grad_transform`` both take a tensor, should return a
        tensor for portable-save and streaming compatibility, run under ``pause_logging()``, and
        raise ``TorchLensPostfuncError`` with layer/function/tensor context if they fail.

        Activation transforms run during forward capture. Their result is stored alongside the
        raw out by default, and ``capture.backward_ready=True`` requires the transformed out to
        stay graph-connected and differentiable when the raw out requires grads.

        Gradient transforms run from the backward hook output, so they follow the grad tensor's
        shorter lifetime rather than forward out retention. When the raw grad itself
        requires grads under ``capture.backward_ready=True``, the same checks apply.

    Returns
    -------
    Trace
        A ``Trace`` containing layer outs (if requested) and full metadata.
    """
    _reject_unrouted_forward_kwargs(forward_kwargs)
    del forward_kwargs  # must never ride the ladder's locals() snapshot below
    model, intervene = _model_door.resolve_trace_operands(model, intervene)
    if input_size is not None or input_args is None:
        model = _reject_non_module_ladder_root(model)
        # Quickstart input ladder (F17, memo D2/D4): declared/inferred rungs
        # resolve BEFORE any capture machinery, then re-enter trace() as gold.
        ladder_kwargs = locals().copy()
        for consumed in ("model", "input_args", "input_kwargs", "input_size"):
            ladder_kwargs.pop(consumed)
        from .quickstart._ladder import _trace_via_ladder

        return _trace_via_ladder(model, input_args, input_kwargs, input_size, ladder_kwargs)
    if not isinstance(model, nn.Module) and is_dynamo_compiled_callable(model):
        raise InvalidArgumentError(
            "TorchLens cannot capture a torch.compile-produced plain callable because it is "
            "not an nn.Module",
            code="compiled_callable_unsupported",
            remedy="pass the original eager nn.Module instead of the compiled callable",
            argument="model",
        )
    _reject_extra_positional_input(grad_transform)
    public_trace_kwargs = locals().copy()
    public_trace_kwargs.pop("backend")
    public_trace_kwargs.pop("input_size")  # ladder-consumed; always None here
    # grouping= (UNSTABLE, keyword-only; L1 wave 0): closed-vocabulary knob.
    # Only "structural" (today's grouping, the default) is entry-legal;
    # "strict_shapes" waits on its own reviewed design and "fold_sites" on
    # an affirmative D1 ruling (the flip PR) / the episode capture kind.
    public_trace_kwargs.pop("grouping")
    if grouping is not MISSING:
        from .postprocess._grouping_stamp import validate_grouping_knob

        validate_grouping_knob(grouping)
    if echo is not None and echo is not False:
        # Entry-time echo refusals (bad spellings, finalized-label selectors)
        # fire BEFORE any capture work on every backend path.
        from .snoop import normalize_echo

        normalize_echo(echo)
    if chunk_paths is not MISSING and chunk_paths is not None and chunk_size in (MISSING, None):
        raise ChunkedForwardConfigError("chunk_paths requires chunk_size.")
    if backend is None and (jax_static_argnums is not MISSING or grad_options is not MISSING):
        raise BackendUnsupportedError(
            "jax_static_argnums is only supported with backend='jax'; grad_options is "
            "only supported with backend='jax', backend='mlx', backend='tinygrad', "
            "backend='paddle', or backend='tf'."
        )
    explicit_backend_spec = None
    if backend is not None:
        explicit_backend_spec = get_backend_spec(str(backend))
        _filter_trace_kwargs_for_backend(public_trace_kwargs, explicit_backend_spec)
        explicit_backend_spec = resolve_backend_spec(backend, model, input_args, input_kwargs)
    if (
        backend is None
        and chunk_size in (MISSING, None)
        and (capture is None or not capture.is_field_explicit("transform"))
    ):
        from . import autoroute

        # Only the surviving public kwargs are forwarded: capture knobs travel
        # inside the grouped ``capture=`` object, so re-entrant trace() calls
        # inside detectors receive exactly what the user could have passed.
        autoroute_kwargs = {
            "input_kwargs": input_kwargs,
            "grad_transform": grad_transform,
            "save_mode": save_mode,
            "reconstruction_ready": reconstruction_ready,
            "capture": capture,
            "save": save,
            "intervene": intervene,
            "halt": halt,
            "lookback": lookback,
            "lookback_payload_policy": lookback_payload_policy,
            "storage": storage,
            "streaming": streaming,
            "profile": profile,
            "recipes": recipes,
            "episode": episode,
            "grouping": grouping,
            "echo": echo,
        }
        for detector in autoroute.input.iter_by_priority():
            result = detector(model, input_args, **autoroute_kwargs)
            if result is not None:
                return cast("Trace", result)
    if closed_bool_env("TORCHLENS_AUTO"):
        raise CaptureContextError(
            "TORCHLENS_AUTO requested an unsupported implicit capture mode",
            code="auto_environment_unsupported",
            remedy="unset TORCHLENS_AUTO and call auto_capture() explicitly",
            environment_variable="TORCHLENS_AUTO",
        )
    resolved_spec = explicit_backend_spec or resolve_backend_spec(
        backend, model, input_args, input_kwargs
    )
    from .snoop._entry import refuse_echo_non_torch

    # Never a silent no-op: echo= is torch-only in wave 1 (typed refusal).
    refuse_echo_non_torch(echo, resolved_spec)
    _filter_trace_kwargs_for_backend(public_trace_kwargs, resolved_spec)
    _enforce_capability_option_gates(public_trace_kwargs, resolved_spec)
    _refuse_non_torch_episode(public_trace_kwargs, resolved_spec)
    return cast("Trace", resolved_spec.capture_trace(**public_trace_kwargs))


def _refuse_non_torch_episode(public_trace_kwargs: dict[str, Any], resolved_spec: Any) -> None:
    """episode= capture (capture_kind=episode) is torch-only in this release."""

    if str(resolved_spec.name) != "torch":
        episode_value = public_trace_kwargs.pop("episode", None)
        if episode_value is not None:
            raise BackendUnsupportedError(
                "episode= capture (capture_kind=episode) is torch-only in this "
                f"release; backend {str(resolved_spec.name)!r} does not support "
                "episode declarations."
            )


def _trace_torch_model(
    model: nn.Module | Callable[..., Any],
    input_args: str | torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    layers_to_save: str | list[Any] | None | MissingType = MISSING,
    transform: Callable[[Any], Any] | None | MissingType = MISSING,
    save_raw_input: str | bool | MissingType = MISSING,
    batch_render: str | MissingType = MISSING,
    output_transform: Callable[[Any], Any] | None | MissingType = MISSING,
    output_style: str | None | MissingType = MISSING,
    output_head: str | None | MissingType = MISSING,
    save_raw_output: str | bool | MissingType = MISSING,
    keep_orphans: bool | MissingType = MISSING,
    output_device: OutputDeviceLiteral | MissingType = MISSING,
    activation_transform: ActivationPostfunc | None | MissingType = MISSING,
    grad_transform: GradientPostfunc | None | MissingType = MISSING,
    save_raw_activations: bool | MissingType = MISSING,
    save_raw_gradients: bool | MissingType = MISSING,
    save_mode: SaveMode | MissingType = MISSING,
    capture_tensor_grad_hooks: bool | MissingType = MISSING,
    mark_layer_depths: bool | MissingType = MISSING,
    detach_saved_activations: bool | MissingType = MISSING,
    save_arg_values: bool | MissingType = MISSING,
    save_grads: bool | str | list[Any] | PredicateFn | BaseSelector | None | MissingType = MISSING,
    save_code_context: bool | MissingType = MISSING,
    save_rng_states: bool | MissingType = MISSING,
    reconstruction_ready: bool | MissingType = MISSING,
    random_seed: int | None | MissingType = MISSING,
    num_context_lines: int | MissingType = MISSING,
    optimizer: Any | MissingType = MISSING,
    save_outs_to: str | Path | None | MissingType = MISSING,
    keep_outs_in_memory: bool | MissingType = MISSING,
    out_sink: Callable[[str, torch.Tensor], None] | None | MissingType = MISSING,
    intervention_ready: bool | MissingType = MISSING,
    capture_container_structure: bool | MissingType = MISSING,
    hooks: Any | None | MissingType = MISSING,
    unwrap_when_done: bool | MissingType = MISSING,
    verbose: bool | MissingType = MISSING,
    source_context_lines: int | MissingType = MISSING,
    compute_input_output_distances: bool | MissingType = MISSING,
    recurrence_detection: bool | MissingType = MISSING,
    capture: CaptureOptions | None = None,
    save: SaveOptions | PredicateFn | BaseSelector | None = None,
    intervene: InterventionPredicate | None = None,
    halt: HaltPredicateFn | None = None,
    lookback: int = 0,
    lookback_payload_policy: str = "metadata_only",
    storage: StreamingOptions | None = None,
    streaming: StreamingOptions | None = None,
    backward_ready: bool | MissingType = MISSING,
    inference_only: bool | MissingType = MISSING,
    name: str | None | MissingType = MISSING,
    cache: bool | MissingType = MISSING,
    cache_dir: str | Path | None | MissingType = MISSING,
    module_filter: Callable[[Any], bool] | None | MissingType = MISSING,
    stop_after: Any | None | MissingType = MISSING,
    raise_on_nan: bool | MissingType = MISSING,
    structure_only: bool | MissingType = MISSING,
    profile: bool | MissingType = MISSING,
    jax_control_flow: Literal["reject", "unroll", "region"] | MissingType = MISSING,
    jax_max_control_flow_unroll: int | MissingType = MISSING,
    module_identity_mode: str | None | MissingType = MISSING,
    payload_policy: str | None | MissingType = MISSING,
    save_preview: bool | MissingType = MISSING,
    recipes: (
        list[Callable[[Any], dict[str, Any]]]
        | tuple[Callable[[Any], dict[str, Any]], ...]
        | None
        | MissingType
    ) = MISSING,
    episode: EpisodeSpec | None = None,
    capture_output_structure: bool | MissingType = MISSING,
    chunk_size: int | None | MissingType = MISSING,
    chunk_paths: Iterable[Any] | None | MissingType = MISSING,
    echo: Any | None = None,
    retain_output_parents_for_layers_to_save: bool = False,
    _selective_layers_to_save_request: object | None = None,
) -> Trace:
    """Run the registry-owned torch trace implementation.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Positional args for ``model.forward()``.
    input_kwargs:
        Keyword args for ``model.forward()``.
    **capture_options:
        The remaining parameters match ``trace`` exactly, excluding ``backend``.

    Returns
    -------
    Trace
        Captured torch trace.
    """
    # DataParallel is not supported - unwrap and warn if present.
    warn_parallel()
    # Reject non-Module input first: `_reject_opaque_wrappers` calls
    # `model.named_modules()` (FSDP/ScriptModule checks), which only makes sense on a real
    # nn.Module. A non-Module input previously leaked an AttributeError from that call
    # instead of the documented "Unsupported model type" ValueError.
    # The ruled root contract (foldA MEMO s5 item 9, a direct maintainer ruling) is
    # nn.Module OR a bound method of one (owner resolved via ``__self__``):
    # a bound method wraps into the TL-authored synthetic root (owner
    # registered as a submodule; method called exactly once); anything else
    # refuses teaching that spelling at the point of failure.
    if not isinstance(model, nn.Module):
        if is_bound_method_of_module(model):
            model = TLBoundMethodRoot(model)
        else:
            raise InvalidArgumentError(
                f"Unsupported model type for capture: received {type(model).__name__}, "
                "not a torch.nn.Module or a bound method of one",
                code="model_type_unsupported",
                remedy=(
                    "pass a torch.nn.Module, or a bound method of an nn.Module "
                    "(owner resolved via method.__self__, e.g. "
                    "tl.trace(model.generate, input_ids)), or select the backend "
                    "that owns the supplied model"
                ),
                argument="model",
                received_type=type(model).__name__,
            )
    _reject_opaque_wrappers(model)
    # grind-r5 b7 R55 (fable MED, survived from round 1 -- it misdirected two
    # hostile review lanes): the natural multi-input spelling
    # ``tl.trace(model, q, k, v)`` lands tensor ``k`` in ``input_kwargs`` and
    # ``v`` in the next positional slot, then crashes DEEP with an
    # ambient-import-dependent error that never names the mistake. Refuse
    # typed at entry, naming the tuple spelling.
    if input_kwargs is not None and not isinstance(input_kwargs, collections.abc.Mapping):
        raise ArgumentTypeError(
            "input_kwargs must be a Mapping of keyword arguments for "
            f"model.forward(), got {type(input_kwargs).__name__}. Passing "
            "multiple positional inputs as separate arguments is not "
            "supported: bundle them as one tuple, e.g. "
            "tl.trace(model, (input_a, input_b, input_c)).",
            code="input_kwargs_type_invalid",
            remedy="pass keyword args as a dict, or bundle positional inputs into one tuple",
        )
    model = unwrap_compiled_model(model)
    model = _unwrap_data_parallel(model)
    if reconstruction_ready is not MISSING and reconstruction_ready:
        save_arg_values = True
        save_rng_states = True

    capture_options = merge_capture_options(
        capture=capture,
        layers_to_save=layers_to_save,
        transform=transform,
        save_raw_input=save_raw_input,
        batch_render=batch_render,
        output_transform=output_transform,
        output_style=output_style,
        output_head=output_head,
        save_raw_output=save_raw_output,
        layer_visualizers=MISSING,
        save_visualizations=MISSING,
        keep_orphans=keep_orphans,
        output_device=output_device,
        save_arg_values=save_arg_values,
        save_grads=save_grads,
        capture_tensor_grad_hooks=capture_tensor_grad_hooks,
        save_code_context=save_code_context,
        save_rng_states=save_rng_states,
        random_seed=random_seed,
        source_context_lines=source_context_lines,
        num_context_lines=num_context_lines,
        optimizer=optimizer,
        compute_input_output_distances=compute_input_output_distances,
        mark_layer_depths=mark_layer_depths,
        detach_saved_activations=detach_saved_activations,
        recurrence_detection=recurrence_detection,
        intervention_ready=intervention_ready,
        capture_container_structure=capture_container_structure,
        capture_output_structure=capture_output_structure,
        hooks=hooks,
        unwrap_when_done=unwrap_when_done,
        verbose=verbose,
        backward_ready=backward_ready,
        inference_only=inference_only,
        name=name,
        cache=cache,
        cache_dir=cache_dir,
        module_filter=module_filter,
        stop_after=stop_after,
        jax_control_flow=jax_control_flow,
        jax_max_control_flow_unroll=jax_max_control_flow_unroll,
        module_identity_mode=module_identity_mode,
        payload_policy=payload_policy,
        save_preview=save_preview,
        raise_on_nan=raise_on_nan,
        structure_only=structure_only,
    )
    if isinstance(capture_options.layers_to_save, torch.Tensor):
        raise ArgumentTypeError(
            "capture.layers_to_save received a torch.Tensor -- this is almost "
            "always a model input routed to the wrong slot. Bundle model inputs "
            "as one tuple, e.g. tl.trace(model, (input_a, input_b, input_c)), "
            "and use save= for selective capture.",
            code="layers_to_save_type_invalid",
            remedy="bundle positional inputs into one tuple; use save= for selection",
        )
    _reject_unsupported_torch_trace_option_values(capture_options)
    # Echo teaching refusals fire whether or not echo= is armed: EchoOptions
    # routed into save=/hooks= refuses with the measured receipts, and
    # echo x cache/chunked/structure_only composition refuses typed (snoop D1).
    from .snoop._entry import resolve_echo_capture_options

    echo_options = resolve_echo_capture_options(
        echo,
        save=save,
        capture_options=capture_options,
        chunked=chunk_size is not MISSING and chunk_size is not None,
    )
    profile_enabled = False if isinstance(profile, MissingType) else bool(profile)
    raw_input = None
    input_transform = capture_options.transform
    if input_transform is not None:
        raw_input = input_args
        transformed_input = input_transform(input_args)
        if isinstance(transformed_input, collections.abc.Mapping):
            input_args = []
            input_kwargs = dict(transformed_input)
        else:
            input_args = transformed_input
            input_kwargs = None
    else:
        original_input_args = input_args
        input_args = _coerce_input_args(model, input_args)
        if _should_store_auto_coerced_raw_input(original_input_args, input_args):
            raw_input = original_input_args

    # W2 (weightsfree memo D2): admit_meta derives ONCE from the resolved
    # structure-only option state at this CAPTURE entry; the returned
    # admission record (None on ordinary/real captures) is armed around the
    # capture driver below and consumed by the forward scope + settlement.
    meta_admission = check_model_and_input_variants(
        model,
        input_args,
        input_kwargs,
        admit_meta=bool(capture_options.structure_only),
    )
    grouped_save_options, save_predicate = _split_save_options_and_predicate(save)
    if intervene is not None and not callable(intervene):
        raise ArgumentTypeError(
            f"intervene received non-callable type {type(intervene).__name__}",
            code="intervention_predicate_type_invalid",
            remedy="pass a predicate callable such as tl.when(...) or None",
            argument="intervene",
            received_type=type(intervene).__name__,
        )
    if halt is not None and not callable(halt):
        raise ArgumentTypeError(
            f"halt received non-callable type {type(halt).__name__}",
            code="halt_predicate_type_invalid",
            remedy="pass a predicate callable or None",
            argument="halt",
            received_type=type(halt).__name__,
        )
    if not isinstance(lookback, int) or not 0 <= lookback <= 1024:
        raise InvalidArgumentError(
            f"lookback={lookback!r} is not an integer in the supported range [0, 1024]",
            code="lookback_invalid",
            remedy="set lookback to an integer from 0 through 1024 inclusive",
            argument="lookback",
        )
    if lookback_payload_policy not in {
        "metadata_only",
        "detached_raw",
        "transformed",
        "grad_connected",
        "disk_spilled",
    }:
        raise InvalidArgumentError(
            f"lookback_payload_policy={lookback_payload_policy!r} is not supported",
            code="lookback_payload_policy_invalid",
            remedy=(
                "set lookback_payload_policy to 'metadata_only', 'detached_raw', "
                "'transformed', 'grad_connected', or 'disk_spilled'"
            ),
            argument="lookback_payload_policy",
        )
    episode_resolved = None
    if episode is not None:
        from .errors.episode import EpisodeDeclarationError

        if chunk_size not in (MISSING, None) or chunk_paths not in (MISSING, None):
            raise EpisodeDeclarationError(
                "episode= cannot combine with chunked forwards (chunk_size/"
                "chunk_paths): an episode is ONE wrapped session capture, and "
                "the chunk fan-out produces several.",
                code="episode_declaration_invalid",
            )
        if cache is not MISSING and cache:
            raise EpisodeDeclarationError(
                "episode= cannot combine with cache=True in this release: the "
                "capture cache replays a stored product, and episode ledgers "
                "are settled per capture.",
                code="episode_declaration_invalid",
            )
        if getattr(capture_options, "structure_only", False):
            # S2 marker-combination table: episode x structure_only is TYPED
            # REFUSE this sprint (no value semantics to fold). The refusal
            # code is L7a's capability-table constant — ONE vocabulary for
            # the combination across the entry check and the post-capture
            # capability gate (capture/structure_only.py owns the row).
            from .capture.structure_only import STRUCTURE_ONLY_EPISODE_UNSUPPORTED

            raise EpisodeDeclarationError(
                "structure-only episodes are out of scope this sprint: "
                "capture_kind=episode with the structure-only marker refuses "
                "typed per the ratified marker-combination table.",
                code=STRUCTURE_ONLY_EPISODE_UNSUPPORTED,
            )
        if getattr(episode, "step_output_kind", "tokens") != "none" and (
            capture_options.layers_to_save in (None, "none", "None", "NONE")
            or (
                isinstance(capture_options.layers_to_save, list)
                and not capture_options.layers_to_save
            )
        ):
            # Kind-conditional save rule (foldA D8, lane F40b): tokens/digest
            # evidence derives from the retained root output, so a value-free
            # save policy refuses; a declared status-only episode
            # (step_output_kind='none') derives no evidence and is admitted.
            raise EpisodeDeclarationError(
                "episode= with step_output_kind='tokens'/'digest' derives its "
                "per-step evidence column from the retained root output; "
                "save='none' retains no output payload. Remedy: keep the "
                "default save policy, include the output in the save= "
                "selection, or declare step_output_kind='none' for a "
                "status-only ledger.",
                code="episode_declaration_invalid",
            )
        episode_resolved = resolve_episode_declaration(episode, model)
        if intervene is not None:
            # ATTESTED COUPLING (lane F42, the foldA D5 flip): the session
            # armed here feeds the settlement writers (fire counts,
            # intervention digest, perturbed fidelity) -- the D5 evidence bar.
            from dataclasses import replace as _dataclass_replace

            from .capture._episode_coupling import CouplingSession, rule_identity_of

            episode_resolved = _dataclass_replace(
                episode_resolved,
                coupling_session=CouplingSession(rule_identity=rule_identity_of(intervene)),
            )
    save_options = merge_save_options(
        save=grouped_save_options,
        activation_transform=activation_transform,
        grad_transform=grad_transform,
        save_raw_activations=save_raw_activations,
        save_raw_gradients=save_raw_gradients,
    )
    if storage is not None and streaming is not None:
        raise KeywordConflictError(
            "Both storage and streaming options were supplied",
            code="storage_argument_conflict",
            remedy="remove streaming and pass only storage, or remove storage",
            arguments=("storage", "streaming"),
        )
    streaming_options = merge_streaming_options(
        streaming=storage if storage is not None else streaming,
    )
    chunk_size_value = None if isinstance(chunk_size, MissingType) else chunk_size
    chunk_paths_value = None if isinstance(chunk_paths, MissingType) else chunk_paths
    normalized_chunk_size = normalize_chunk_size(chunk_size_value)
    if chunk_paths_value is not None and normalized_chunk_size is None:
        raise ChunkedForwardConfigError("chunk_paths requires chunk_size.")
    layers_to_save = capture_options.layers_to_save
    save_raw_input_policy = capture_options.save_raw_input
    batch_render_policy = capture_options.batch_render
    output_transform_value = capture_options.output_transform
    output_style_value = capture_options.output_style
    output_head_value = capture_options.output_head
    save_raw_output_policy = capture_options.save_raw_output
    layer_visualizers_value = cast(
        "dict[Any, Callable[..., Any]] | None", capture_options.layer_visualizers
    )
    save_visualizations_value = capture_options.save_visualizations
    keep_orphans = capture_options.keep_orphans
    output_device = capture_options.output_device
    activation_transform = save_options.activation_transform
    grad_transform = save_options.grad_transform
    save_raw_activations = save_options.save_raw_activations
    save_raw_gradients = save_options.save_raw_gradients
    save_mode_value = "copy" if save_mode is MISSING else save_mode
    if save_mode_value not in {"copy", "reference", "view", "cpu_async"}:
        raise InvalidArgumentError(
            f"save_mode={save_mode_value!r} is not supported",
            code="save_mode_invalid",
            remedy="set save_mode to 'copy', 'reference', 'view', or 'cpu_async'",
            argument="save_mode",
        )
    save_arg_values = capture_options.save_arg_values
    capture_tensor_grad_hooks = capture_options.capture_tensor_grad_hooks
    save_code_context = capture_options.save_code_context
    save_rng_states = capture_options.save_rng_states
    random_seed = capture_options.random_seed
    source_context_lines = capture_options.source_context_lines
    optimizer = capture_options.optimizer
    compute_input_output_distances = capture_options.compute_input_output_distances
    detach_saved_activations = capture_options.detach_saved_activations
    recurrence_detection = capture_options.recurrence_detection
    intervention_ready = capture_options.intervention_ready
    capture_container_structure = capture_options.capture_container_structure
    hooks = capture_options.hooks
    unwrap_when_done = capture_options.unwrap_when_done
    verbose = capture_options.verbose
    name = capture_options.name
    cache_enabled = capture_options.cache
    facet_recipes = None if isinstance(recipes, MissingType) else recipes
    if capture_options.stop_after is not None and halt is not None:
        raise ArgumentConflictError(
            "Both stop_after= and halt= were configured; they compile into the same "
            "stop-directive slot and cannot combine",
            code="stop_after_halt_conflict",
            remedy=(
                "pass one stop directive: keep halt= (compose predicates with & |) "
                "or keep stop_after="
            ),
            arguments=("stop_after", "halt"),
        )
    stop_after_site = capture_options.stop_after
    stop_after_ambient = False
    if stop_after_site is None and halt is None:
        from .experimental import _active_stop_after_site

        stop_after_site = _active_stop_after_site()
        stop_after_ambient = stop_after_site is not None
    stop_after_selector_shaped = False
    if stop_after_site is not None:
        if normalized_chunk_size is not None:
            raise ChunkedForwardConfigError(
                "chunk_size cannot be combined with stop_after: the chunk fan-out runs "
                "several captures and a single stop frontier is ambiguous across them. "
                "Remedy: drop chunk_size or drop stop_after.",
                code="stop_after_chunked_conflict",
            )
        if isinstance(stop_after_site, BaseSelector):
            halt = stop_after_site
            stop_after_selector_shaped = True
        elif isinstance(stop_after_site, str):
            from .ir.selector_eval import looks_like_finalized_label

            if looks_like_finalized_label(stop_after_site):
                raise InvalidArgumentError(
                    f"stop_after={stop_after_site!r} looks like a finalized postprocess "
                    "label, but stop_after runs live during capture, before finalized "
                    "labels exist",
                    code="stop_after_site_not_live",
                    remedy=(
                        "pass a module address string (e.g. 'encoder.layer.4'), a live "
                        "selector such as tl.func('relu') or tl.module('encoder.layer.4'), "
                        "or a callable predicate"
                    ),
                    argument="stop_after",
                )
            from .intervention.selectors import FuncSelector, ModuleSelector

            # A bare string is the pluck-vocabulary spelling: halt at the first
            # emission whose module address OR function name matches, whichever
            # fires first ("encoder.layer.4" never names a func; "relu" never
            # names a module address in practice, so collisions are benign).
            halt = ModuleSelector(stop_after_site) | FuncSelector(stop_after_site)
            stop_after_selector_shaped = True
        elif callable(stop_after_site):
            halt = stop_after_site
        else:
            raise ArgumentTypeError(
                f"stop_after received unsupported type {type(stop_after_site).__name__}",
                code="stop_after_type_invalid",
                remedy=("pass a module address string, a tl.* selector, or a callable predicate"),
                argument="stop_after",
                received_type=type(stop_after_site).__name__,
            )
    save_grads_policy = capture_options.save_grads
    should_save_grads = save_grads_policy not in (None, False)
    if save_grads_policy is True:
        grads_to_save_resolved: Any = "all"
    elif save_grads_policy in (None, False):
        grads_to_save_resolved = None
    elif callable(save_grads_policy):
        # Selector/callable retention predicates are HONORED, never collapsed
        # to "all": they ride the deferred-gradient path and are resolved
        # against the finalized ops post-postprocess, so only matching ops
        # get gradient hooks (listA row 10 first clause: the collapse saved
        # ALL grads and silently discarded the predicate).
        grads_to_save_resolved = save_grads_policy
    else:
        grads_to_save_resolved = cast("str | list[Any] | None", save_grads_policy)
    grad_storage_path_value = streaming_options.bundle_path if should_save_grads else None
    retain_grads_in_memory_value = streaming_options.retain_in_memory

    from ._options_validation import _validate_output_device

    _validate_output_device(output_device)
    if streaming_options.bundle_path is not None and streaming_options.out_callback is not None:
        raise ArgumentConflictError(
            "Both disk-backed output storage and an output callback were configured",
            code="output_sink_conflict",
            remedy=(
                "choose either disk storage (storage=tl.to_disk(...) / "
                "streaming.bundle_path) or streaming.out_callback"
            ),
            arguments=("bundle_path", "out_callback"),
        )
    if capture_options.structure_only:
        layers_to_save = _enforce_structure_only_entry_contract(
            capture_options,
            _StructureOnlyEntryFacts(
                layers_to_save=layers_to_save,
                save_predicate=save_predicate,
                halt=halt,
                streaming_options=streaming_options,
                lookback_payload_policy=lookback_payload_policy,
                raise_on_nan_value=capture_options.raise_on_nan,
                track_nonfinite_value=capture_options.track_nonfinite,
                intervention_ready=intervention_ready,
                should_save_grads=should_save_grads,
            ),
        )
        save_raw_input_policy = False
        save_raw_output_policy = False
    train_mode_explicit = capture_options.is_field_explicit("backward_ready")
    train_mode_value = capture_options.backward_ready
    inference_only_conflicts: list[str] = []
    if capture_options.is_field_explicit("backward_ready") and train_mode_value is True:
        inference_only_conflicts.append("backward_ready")
    if capture_options.is_field_explicit("save_grads") and should_save_grads:
        inference_only_conflicts.append("save_grads")
    if capture_options.is_field_explicit("intervention_ready") and intervention_ready is True:
        inference_only_conflicts.append("intervention_ready")
    backward_opted_in = (
        capture_options.is_field_explicit("save_grads")
        and should_save_grads
        and save_grads_policy is not True
    )
    grad_streaming_requested = grad_storage_path_value is not None
    if grad_streaming_requested:
        should_save_grads = True
    if backward_opted_in:
        if train_mode_explicit and train_mode_value is False:
            raise InvalidArgumentError(
                "save_grads requests backward capture and requires backward_ready=True, but "
                "backward_ready=False was supplied",
                code="backward_capture_conflict",
                remedy="omit backward_ready or set backward_ready=True",
                arguments=("save_grads", "backward_ready"),
            )
        train_mode_value = True
        should_save_grads = True
    if train_mode_value and grad_storage_path_value is not None:
        raise TrainingModeConfigError(
            "backward_ready=True is not compatible with disk-backed gradient storage. "
            "Remedy: drop the gradient storage path or drop backward_ready=True.",
            code="backward_ready_conflict",
        )

    validate_training_compatibility(
        backward_ready=train_mode_value,
        streaming=streaming_options,
        detach_saved_activations=detach_saved_activations,
        inference_mode_active=torch.is_inference_mode_enabled(),
        inference_only=capture_options.inference_only,
        inference_only_conflicts=tuple(inference_only_conflicts),
    )
    chunk_plan = None
    if normalized_chunk_size is not None:
        _validate_chunked_forward_capture(
            input_kwargs=input_kwargs,
            backward_ready=train_mode_value,
            save_grads=should_save_grads,
            hooks=hooks,
            intervene=intervene,
            halt=halt,
            streaming=streaming_options,
        )
        chunk_plan = plan_chunks(
            input_args,
            chunk_size=normalized_chunk_size,
            chunk_paths=chunk_paths_value,
        )

    if type(layers_to_save) is str:
        layers_to_save = layers_to_save.lower()
    if type(grads_to_save_resolved) is str:
        grads_to_save_resolved = grads_to_save_resolved.lower()
    requested_layers_to_save = layers_to_save
    uses_deferred_gradients = grads_to_save_resolved not in ["all", "none", None, []]
    uses_deferred_activation = _is_selective_label_save(layers_to_save) and (
        _layers_to_save_mentions_output(layers_to_save)
        or _layers_to_save_has_negative_index(layers_to_save)
        or _layers_to_save_mentions_identity(layers_to_save)
        # Integer ordinals and final-label-shaped strings are defined against
        # FINAL layer numbering, which orphan removal renumbers after capture;
        # they must resolve post-postprocess, never against raw indexes.
        or _layers_to_save_needs_final_resolution(layers_to_save)
    )
    live_layers_to_save = (
        _layers_to_save_live_subset(layers_to_save) if uses_deferred_activation else layers_to_save
    )
    needs_live_output_projection = (
        _is_selective_label_save(layers_to_save)
        and should_save_grads
        and not uses_deferred_activation
    )
    uses_selective_layers_to_save = _is_selective_label_save(layers_to_save)
    if uses_selective_layers_to_save and not uses_deferred_activation:
        layers_predicate = _make_layers_to_save_predicate(layers_to_save)
        save_predicate = _combine_save_predicates(save_predicate, layers_predicate)
        layers_to_save = "all"
    elif uses_deferred_activation and live_layers_to_save is not None:
        layers_predicate = _make_layers_to_save_predicate(live_layers_to_save)
        save_predicate = _combine_save_predicates(save_predicate, layers_predicate)
    if save_predicate is not None or intervene is not None or halt is not None:
        from .capture.predicates import validate_followed_by_capability

    if save_predicate is not None:
        validate_followed_by_capability(
            save_predicate,
            api_name="trace(save=...)",
            supports_retroactive=True,
        )
    if intervene is not None:
        validate_followed_by_capability(
            intervene,
            api_name="trace(intervene=...)",
            supports_retroactive=False,
        )
    if halt is not None:
        validate_followed_by_capability(
            halt,
            api_name="trace(halt=...)",
            supports_retroactive=False,
        )
    log_name = name if name is not None else _state._auto_name(model)
    cache_path: Path | None = None
    cache_key: str | None = None
    cache_secret: bytes | None = None
    if cache_enabled:
        cache_config = {
            "layers_to_save": requested_layers_to_save,
            "keep_orphans": keep_orphans,
            "output_device": output_device,
            "save_arg_values": save_arg_values,
            "capture_tensor_grad_hooks": capture_tensor_grad_hooks,
            "save_grads": repr(save_grads_policy),
            "save_code_context": save_code_context,
            "save_rng_states": save_rng_states,
            "source_context_lines": source_context_lines,
            "compute_input_output_distances": compute_input_output_distances,
            "detach_saved_activations": detach_saved_activations,
            "save_mode": save_mode_value,
            "recurrence_detection": recurrence_detection,
            "backward_ready": train_mode_value,
            "inference_only": capture_options.inference_only,
            "chunk_size": normalized_chunk_size,
            "chunk_paths": normalize_chunk_paths(chunk_paths_value),
            "capture_container_structure": capture_container_structure,
            "output_transform": _stable_cache_fragment(output_transform_value),
            "output_style": output_style_value,
            "output_head": output_head_value,
            "semantic_output_cache_key": semantic_output_cache_key(
                model,
                output_style=output_style_value,
                output_head=output_head_value,
            ),
            "facet_recipes": _facet_recipe_cache_key(facet_recipes),
            "save_predicate": _predicate_cache_key(save_predicate),
            "intervene": _predicate_cache_key(intervene),
            "halt": _predicate_cache_key(halt),
            # stop_after compiles INTO the halt slot, but its never-fired policy
            # differs by provenance, so the two spellings must not share a key:
            # a completed halt= capture could otherwise satisfy a stop_after=
            # request whose cold run would have refused stop_after_never_fired.
            "stop_after": _stable_cache_fragment(stop_after_site),
            # Capability / payload-policy options that change WHAT is captured or
            # stored in the returned trace. Omitting any of these let a second
            # trace() with a different capability silently return an earlier cached
            # trace built with the wrong capability (e.g. asked intervention_ready=True,
            # got a cached False). Include them so a capability change misses the cache;
            # callables are repr-keyed (conservative -- a distinct object misses rather
            # than risking a false hit), matching how output_transform is keyed above.
            "intervention_ready": intervention_ready,
            "structure_only": capture_options.structure_only,
            "track_device_memory": capture_options.track_device_memory,
            "log_injections": capture_options.log_injections,
            "save_raw_input": repr(save_raw_input_policy),
            "save_raw_output": repr(save_raw_output_policy),
            "save_raw_activations": save_raw_activations,
            "save_raw_gradients": save_raw_gradients,
            "activation_transform": _stable_cache_fragment(activation_transform),
            "grad_transform": _stable_cache_fragment(grad_transform),
            "random_seed": random_seed,
            "module_filter": _stable_cache_fragment(capture_options.module_filter),
            "layer_visualizers": _stable_cache_fragment(layer_visualizers_value),
            "save_visualizations": _stable_cache_fragment(save_visualizations_value),
            "optimizer": repr(optimizer),
            "hooks": _stable_cache_fragment(hooks),
            "lookback": lookback,
            "lookback_payload_policy": lookback_payload_policy,
            "jax_control_flow": capture_options.jax_control_flow,
            "jax_max_control_flow_unroll": capture_options.jax_max_control_flow_unroll,
            "module_identity_mode": capture_options.module_identity_mode,
            "payload_policy": capture_options.payload_policy,
            "save_preview": capture_options.save_preview,
            "profile": profile_enabled,
        }
        # The inversion sweep: every CaptureOptions field not hand-curated
        # above and not declared session-neutral enters the key here --
        # raise_on_nan, track_nonfinite, save_budget, distributed_witness,
        # measure_python_peak_memory, emit_nvtx, transform, name, and any
        # future field. Streaming options change what the returned trace
        # retains (and where), so they sweep in full as well.
        _sweep_option_fields_into_cache_config(
            cache_config,
            capture_options,
            prefix="capture_option",
            curated=CAPTURE_CACHE_KEY_CURATED,
            neutral=CAPTURE_CACHE_KEY_NEUTRAL,
        )
        _sweep_option_fields_into_cache_config(
            cache_config,
            streaming_options,
            prefix="streaming_option",
        )
        cache_key = _capture_cache_key(model, input_args, input_kwargs, cache_config)
        cache_root, cache_secret = _prepare_capture_cache_dir(capture_options.cache_dir)
        cache_path = cache_root / f"{cache_key}.pkl"
        if cache_path.exists():
            cached_log = cast(
                "Trace | None", _load_authenticated_capture_cache(cache_path, cache_secret)
            )
            if cached_log is not None:
                try:
                    os.utime(cache_path, None)
                except OSError:
                    pass
                cached_log.capture_cache_hit = True
                cached_log.capture_cache_key = cache_key
                cached_log.capture_cache_path = str(cache_path)
                cached_log.batch_render = batch_render_policy
                return cached_log
    # list-A row 30: tracing runs the model's REAL forward, so train-mode norm
    # layers with running statistics advance them in place (momentum applies
    # even inside torch.no_grad / inference_only). Disclose once per process at
    # the point the mutation is about to happen. Cache hits return above (no
    # forward, no mutation) and structure-only captures retain no values, so
    # neither path reaches this warn. The chunked fan-out recurses back into
    # this function, but the outer call warns first and latches the flag.
    if not _BATCHNORM_TRAIN_STATS_WARNED and not capture_options.structure_only:
        _warn_once_train_mode_running_stats(model)
    if (
        chunk_plan is not None
        and normalized_chunk_size is not None
        and normalized_chunk_size < chunk_plan.total_size
    ):
        chunks = iter_chunked_inputs(input_args, chunk_plan)
        recursive_jax_control_flow = (
            capture_options.jax_control_flow
            if capture_options.is_field_explicit("jax_control_flow")
            else MISSING
        )
        recursive_jax_max_control_flow_unroll = (
            capture_options.jax_max_control_flow_unroll
            if capture_options.is_field_explicit("jax_max_control_flow_unroll")
            else MISSING
        )
        recursive_capture_options = CaptureOptions(
            layers_to_save=layers_to_save,
            transform=None,
            save_raw_input=save_raw_input_policy,
            batch_render=batch_render_policy,
            output_transform=output_transform_value,
            output_style=output_style_value,
            output_head=output_head_value,
            save_raw_output=save_raw_output_policy,
            layer_visualizers=layer_visualizers_value,
            save_visualizations=save_visualizations_value,
            keep_orphans=keep_orphans,
            output_device=output_device,
            save_arg_values=save_arg_values,
            save_grads=save_grads_policy,
            capture_tensor_grad_hooks=capture_tensor_grad_hooks,
            save_code_context=save_code_context,
            save_rng_states=save_rng_states,
            random_seed=random_seed,
            source_context_lines=source_context_lines,
            optimizer=optimizer,
            compute_input_output_distances=compute_input_output_distances,
            detach_saved_activations=detach_saved_activations,
            recurrence_detection=recurrence_detection,
            intervention_ready=intervention_ready,
            capture_container_structure=capture_container_structure,
            hooks=MISSING,
            unwrap_when_done=False,
            verbose=verbose,
            backward_ready=train_mode_value,
            inference_only=capture_options.inference_only,
            name=log_name,
            cache=False,
            cache_dir=capture_options.cache_dir,
            module_filter=capture_options.module_filter,
            stop_after=MISSING,
            jax_control_flow=recursive_jax_control_flow,
            jax_max_control_flow_unroll=recursive_jax_max_control_flow_unroll,
            module_identity_mode=capture_options.module_identity_mode,
            payload_policy=capture_options.payload_policy,
            save_preview=capture_options.save_preview,
            # Session knobs the chunk path historically DROPPED (list-A row 26,
            # second clause): the recursive options object was rebuilt from a
            # hand-enumerated field list, so a chunked capture silently reset
            # save_budget (and the other session-time knobs) to defaults. The
            # recursive constructor must cover EVERY CaptureOptions field;
            # tests/test_capopts_truth_cache_key.py pins the full-coverage
            # set difference so a future field cannot silently drop here.
            emit_nvtx=capture_options.emit_nvtx,
            measure_python_peak_memory=capture_options.measure_python_peak_memory,
            distributed_witness=capture_options.distributed_witness,
            save_budget=capture_options.save_budget,
            raise_on_nan=capture_options.raise_on_nan,
            track_nonfinite=capture_options.track_nonfinite,
            track_device_memory=capture_options.track_device_memory,
            structure_only=capture_options.structure_only,
            log_injections=capture_options.log_injections,
        )
        recursive_save_options = SaveOptions(
            activation_transform=activation_transform,
            grad_transform=grad_transform,
            save_raw_activations=save_raw_activations,
            save_raw_gradients=save_raw_gradients,
        )
        recursive_save_value: SaveOptions | PredicateFn | BaseSelector | None = (
            save_predicate if save_predicate is not None else recursive_save_options
        )
        recursive_activation_transform: ActivationPostfunc | None | MissingType = (
            activation_transform
            if save_predicate is not None and activation_transform is not None
            else MISSING
        )
        recursive_grad_transform: GradientPostfunc | None | MissingType = (
            grad_transform if save_predicate is not None and grad_transform is not None else MISSING
        )
        recursive_save_raw_activations: bool | MissingType = (
            save_raw_activations
            if save_predicate is not None and save_raw_activations is not True
            else MISSING
        )
        recursive_save_raw_gradients: bool | MissingType = (
            save_raw_gradients
            if save_predicate is not None and save_raw_gradients is not True
            else MISSING
        )
        trace = _trace_torch_model(
            model=model,
            input_args=cast(torch.Tensor | list[Any] | tuple[Any, ...], chunks[0]),
            input_kwargs=None,
            layers_to_save=MISSING,
            transform=MISSING,
            save_raw_input=MISSING,
            batch_render=MISSING,
            output_transform=MISSING,
            output_style=MISSING,
            output_head=MISSING,
            save_raw_output=MISSING,
            keep_orphans=MISSING,
            output_device=MISSING,
            activation_transform=recursive_activation_transform,
            grad_transform=recursive_grad_transform,
            save_raw_activations=recursive_save_raw_activations,
            save_raw_gradients=recursive_save_raw_gradients,
            save_mode=cast(SaveMode, save_mode_value),
            capture_tensor_grad_hooks=MISSING,
            mark_layer_depths=MISSING,
            detach_saved_activations=MISSING,
            save_arg_values=MISSING,
            save_grads=MISSING,
            save_code_context=MISSING,
            save_rng_states=MISSING,
            reconstruction_ready=MISSING,
            random_seed=MISSING,
            num_context_lines=MISSING,
            optimizer=MISSING,
            save_outs_to=MISSING,
            keep_outs_in_memory=MISSING,
            out_sink=MISSING,
            intervention_ready=MISSING,
            capture_container_structure=MISSING,
            hooks=MISSING,
            unwrap_when_done=MISSING,
            verbose=MISSING,
            source_context_lines=MISSING,
            compute_input_output_distances=MISSING,
            recurrence_detection=MISSING,
            capture=recursive_capture_options,
            save=recursive_save_value,
            intervene=None,
            halt=halt,
            lookback=lookback,
            lookback_payload_policy=lookback_payload_policy,
            storage=None,
            streaming=None,
            backward_ready=MISSING,
            inference_only=MISSING,
            name=MISSING,
            cache=MISSING,
            cache_dir=MISSING,
            module_filter=MISSING,
            stop_after=MISSING,
            raise_on_nan=MISSING,
            profile=profile_enabled,
            jax_control_flow=MISSING,
            jax_max_control_flow_unroll=MISSING,
            module_identity_mode=MISSING,
            payload_policy=MISSING,
            save_preview=MISSING,
            recipes=facet_recipes,
            capture_output_structure=MISSING,
            chunk_size=None,
            chunk_paths=None,
            retain_output_parents_for_layers_to_save=uses_selective_layers_to_save,
            _selective_layers_to_save_request=(
                requested_layers_to_save if uses_selective_layers_to_save else None
            ),
        )
        initial_chunk_size = min(normalized_chunk_size, chunk_plan.total_size)
        initial_record = {
            "engine": "trace",
            "append": False,
            "chunk_size": initial_chunk_size,
            "total_batch_size": chunk_plan.total_size,
            "append_sequence_id": 0,
            "chunk_paths": normalize_chunk_paths(chunk_paths_value),
        }
        for chunk_index, chunk in enumerate(chunks[1:], start=1):
            if save_predicate is None:
                trace.run(model, chunk, replay=ReplayOptions(append=True), transform=False)
                continue
            new_trace = _trace_torch_model(
                model=model,
                input_args=cast(torch.Tensor | list[Any] | tuple[Any, ...], chunk),
                input_kwargs=None,
                layers_to_save=MISSING,
                transform=MISSING,
                save_raw_input=MISSING,
                batch_render=MISSING,
                output_transform=MISSING,
                output_style=MISSING,
                output_head=MISSING,
                save_raw_output=MISSING,
                keep_orphans=MISSING,
                output_device=MISSING,
                activation_transform=recursive_activation_transform,
                grad_transform=recursive_grad_transform,
                save_raw_activations=recursive_save_raw_activations,
                save_raw_gradients=recursive_save_raw_gradients,
                save_mode=cast(SaveMode, save_mode_value),
                capture_tensor_grad_hooks=MISSING,
                mark_layer_depths=MISSING,
                detach_saved_activations=MISSING,
                save_arg_values=MISSING,
                save_grads=MISSING,
                save_code_context=MISSING,
                save_rng_states=MISSING,
                reconstruction_ready=MISSING,
                random_seed=MISSING,
                num_context_lines=MISSING,
                optimizer=MISSING,
                save_outs_to=MISSING,
                keep_outs_in_memory=MISSING,
                out_sink=MISSING,
                intervention_ready=MISSING,
                capture_container_structure=MISSING,
                hooks=MISSING,
                unwrap_when_done=MISSING,
                verbose=MISSING,
                source_context_lines=MISSING,
                compute_input_output_distances=MISSING,
                recurrence_detection=MISSING,
                capture=recursive_capture_options,
                save=recursive_save_value,
                intervene=None,
                halt=halt,
                lookback=lookback,
                lookback_payload_policy=lookback_payload_policy,
                storage=None,
                streaming=None,
                backward_ready=MISSING,
                inference_only=MISSING,
                name=MISSING,
                cache=MISSING,
                cache_dir=MISSING,
                module_filter=MISSING,
                stop_after=MISSING,
                raise_on_nan=MISSING,
                profile=profile_enabled,
                jax_control_flow=MISSING,
                jax_max_control_flow_unroll=MISSING,
                module_identity_mode=MISSING,
                payload_policy=MISSING,
                save_preview=MISSING,
                recipes=facet_recipes,
                capture_output_structure=MISSING,
                chunk_size=None,
                chunk_paths=None,
                retain_output_parents_for_layers_to_save=uses_selective_layers_to_save,
                _selective_layers_to_save_request=(
                    requested_layers_to_save if uses_selective_layers_to_save else None
                ),
            )
            appended_chunk_size = min(
                normalized_chunk_size,
                chunk_plan.total_size - (chunk_index * normalized_chunk_size),
            )
            _append_chunk_trace_state(
                trace,
                new_trace,
                chunk_size=appended_chunk_size,
                total_batch_size=chunk_plan.total_size,
                append_sequence_id=chunk_index,
                chunk_paths=normalize_chunk_paths(chunk_paths_value),
            )
        trace.append_history = [initial_record, *trace.append_history]
        trace.chunked_forward = True
        trace.last_run = dict(trace.last_run or {})
        trace.last_run["chunk_size"] = normalized_chunk_size
        trace.last_run["chunk_paths"] = normalize_chunk_paths(chunk_paths_value)
        trace.profile_enabled = profile_enabled
        if uses_selective_layers_to_save:
            trace._layer_nums_to_save = [
                op.raw_index
                for op in trace.layer_list
                if op.has_saved_activation and op.layer_type not in {"input", "output"}
            ]
            if not save_arg_values:
                trace._replay_arg_version_data_complete = False
        if layer_visualizers_value:
            _render_layer_visualizers(trace, layer_visualizers_value)
        if unwrap_when_done:
            from .backends.torch.wrappers import unwrap_torch

            unwrap_torch()
        if cache_path is not None and cache_key is not None and cache_secret is not None:
            trace.capture_cache_hit = False
            trace.capture_cache_key = cache_key
            trace.capture_cache_path = str(cache_path)
            _prepare_log_for_capture_cache(trace)
            if _store_authenticated_capture_cache(trace, cache_path, cache_secret):
                _evict_capture_cache(cache_path.parent, keep=cache_path)
        return trace

    run_capture = functools.partial(
        _run_model_and_save_specified_outs,
        model=model,
        input_args=cast(torch.Tensor | list[Any] | tuple[Any, ...], input_args),
        input_kwargs=input_kwargs,
        layers_to_save=layers_to_save,
        keep_orphans=keep_orphans,
        output_device=output_device,
        activation_transform=activation_transform,
        grad_transform=grad_transform,
        save_raw_activations=save_raw_activations,
        save_raw_gradients=save_raw_gradients,
        save_mode=cast(SaveMode, save_mode_value),
        capture_tensor_grad_hooks=capture_tensor_grad_hooks,
        mark_layer_depths=compute_input_output_distances,
        detach_saved_activations=detach_saved_activations,
        save_arg_values=save_arg_values,
        save_grads=should_save_grads,
        grads_to_save=grads_to_save_resolved,
        random_seed=random_seed,
        num_context_lines=source_context_lines,
        optimizer=optimizer,
        echo_options=echo_options,
        save_code_context=save_code_context,
        save_rng_states=save_rng_states,
        recurrence_detection=recurrence_detection,
        save_outs_to=streaming_options.bundle_path,
        keep_outs_in_memory=streaming_options.retain_in_memory,
        stream_custom_attributes=streaming_options.include_custom_attributes,
        stream_buffer_values=streaming_options.include_buffer_values,
        stream_async_writes=streaming_options.async_writes,
        stream_max_pending_bytes=streaming_options.max_pending_bytes,
        grad_storage_path=grad_storage_path_value,
        retain_grads_in_memory=retain_grads_in_memory_value,
        out_sink=streaming_options.out_callback,
        intervention_ready=intervention_ready,
        capture_container_structure=capture_container_structure,
        hooks=hooks,
        intervention_spec=None,
        normalized_hook_plan=None,
        verbose=verbose,
        backward_ready=train_mode_value,
        inference_only=capture_options.inference_only,
        name=log_name,
        module_filter=capture_options.module_filter,
        emit_nvtx=capture_options.emit_nvtx,
        measure_python_peak_memory=capture_options.measure_python_peak_memory,
        distributed_witness=capture_options.distributed_witness,
        save_budget=capture_options.save_budget,
        raise_on_nan=capture_options.raise_on_nan,
        track_nonfinite=capture_options.track_nonfinite,
        track_device_memory=capture_options.track_device_memory,
        structure_only=capture_options.structure_only,
        log_injections=capture_options.log_injections,
        transform=input_transform,
        raw_input=raw_input,
        save_raw_input=save_raw_input_policy,
        batch_render=batch_render_policy,
        output_transform=output_transform_value,
        output_style=output_style_value,
        output_head=output_head_value,
        save_raw_output=save_raw_output_policy,
        layer_visualizers=layer_visualizers_value,
        save_visualizations=save_visualizations_value,
        recipes=facet_recipes,
        save_predicate=save_predicate,
        intervene_predicate=intervene,
        halt_predicate=halt,
        _halt_from_stop_after=stop_after_site is not None,
        lookback=lookback,
        lookback_payload_policy=lookback_payload_policy,
        retain_output_parents_for_layers_to_save=(
            retain_output_parents_for_layers_to_save
            or uses_selective_layers_to_save
            or episode_resolved is not None
        ),
        episode_resolved=episode_resolved,
        _selective_layers_to_save_request=(
            _selective_layers_to_save_request
            if _selective_layers_to_save_request is not None
            else (requested_layers_to_save if uses_selective_layers_to_save else None)
        ),
        _deferred_retention_selector=(
            requested_layers_to_save
            if uses_deferred_activation
            else (["output"] if needs_live_output_projection else None)
        ),
        _deferred_gradient_selector=(
            grads_to_save_resolved
            if uses_deferred_gradients
            else ("all" if uses_deferred_activation and train_mode_value else None)
        ),
    )
    from .backends.torch.rescue import capture_with_rescue

    # Streaming saves, sinks, and halt-predicate partials are not re-runnable;
    # they report an escape as before instead of attempting a rescue re-run.
    # EVERY user-callable channel the re-run would invoke a SECOND time is
    # refused (fail closed): side effects (counters, file writes,
    # externally-held state) would double-apply with only a session-time
    # disclosure. That covers intervention transforms, pre-attached hooks,
    # AND the in-capture transform callables -- ``activation_transform``,
    # ``grad_transform``, and ``output_transform`` run inside ``run_capture``
    # and measurably fired twice on a recovered rescue (b6-opus-R16-1 reopen).
    # Channels applied OUTSIDE the rescue boundary stay eligible: ``save=``
    # selector predicates (pure by contract), the input ``transform=``
    # (applied before the re-run boundary; fires once), and
    # ``layer_visualizers`` (rendered once on the returned trace, after the
    # driver).
    rescue_eligible = (
        streaming_options.bundle_path is None
        and streaming_options.out_callback is None
        and grad_storage_path_value is None
        and halt is None
        and intervene is None
        and not hooks
        and activation_transform is None
        and grad_transform is None
        and output_transform_value is None
    )
    from .capture._episode_join import armed_capture
    from .capture._weightsfree_admission import pending_admission

    try:
        with (
            armed_capture(episode_resolved, run_capture) as capture_callable,
            pending_admission(meta_admission),
        ):
            trace = capture_with_rescue(capture_callable, eligible=rescue_eligible, model=model)
    except BaseException as capture_exc:
        # One settlement path for every escape: a KeyboardInterrupt or
        # SystemExit mid-capture is released exactly like a RuntimeError.
        if episode_resolved is not None:
            # Best-effort: the FAILED partial product carries the episode
            # declaration; attach the derived ledger disclosure without ever
            # masking the user's exception.
            attach_failed_episode_ledger(capture_exc, episode_resolved)
        # M(oracles) item 8 (FORK-A 3/3-agreed cell): every FAILED call is
        # PURE of TorchLens instrumentation. The capture slot was released as
        # the exception unwound the inner finally, so the public release door
        # is legal here; exc.partial_log keeps its already-materialized
        # records and stays recoverable after the release.
        _release_preparation_after_failed_capture(model)
        raise
    if stop_after_site is not None and not bool(getattr(trace, "halted", False)):
        # Never-fired policy split by provenance (brainpipe D-16): a
        # selector-shaped site that never fired is a typo-shaped wrong result
        # (the full forward ran; the caller asked for a frontier) and refuses
        # typed; an exploratory callable or ambient context-manager site
        # legitimately may never fire and warns with a durable ledger entry.
        site_repr = repr(stop_after_site)
        if stop_after_selector_shaped and not stop_after_ambient:
            raise InvalidArgumentError(
                f"stop_after={site_repr} never fired: the capture ran the full "
                "forward and completed without halting, so the result is not the "
                "requested frontier",
                code="stop_after_never_fired",
                remedy=(
                    "check the site against the model's module addresses "
                    "(model.named_modules()) or use tl.func(...)/tl.module(...); "
                    "an exploratory predicate that may legitimately never fire "
                    "belongs in a callable"
                ),
                argument="stop_after",
            )
        stop_annotations = getattr(trace, "annotations", None)
        if isinstance(stop_annotations, dict):
            stop_annotations.setdefault("unmatched_capture_selectors", []).append(
                {"slot": "stop_after", "selector": site_repr}
            )
        from .errors import TorchLensWarning

        warnings.warn(
            TorchLensWarning(
                f"stop_after={site_repr} never fired; the capture ran the full "
                "forward and the outputs are the model's real outputs, not a "
                "frontier. Remedy: check the stop_after site against the executed "
                "model, or drop stop_after",
                code="stop_after_never_fired_callable",
            ),
            stacklevel=2,
        )
    module_filter_suppressed = int(trace.__dict__.pop("_tl_module_filter_suppressed", 0))
    if (
        capture_options.module_filter is not None
        and module_filter_suppressed > 0
        and int(getattr(trace, "num_saved_ops", 0) or 0) == 0
    ):
        from .errors import TorchLensWarning as _TorchLensWarning

        warnings.warn(
            _TorchLensWarning(
                f"module_filter suppressed every selected payload "
                f"({module_filter_suppressed} ops): the returned trace has full "
                "metadata but ZERO saved activations. module_filter is a third "
                "save gate composed with save=/layers_to_save, and its argument "
                "is an op-record namespace (fields such as func_name, "
                "layer_label, modules), NEVER an nn.Module. "
                "Remedy: write the filter against op-record fields (e.g. lambda "
                "op: op.func_name == 'linear'), or drop module_filter",
                code="module_filter_zero_saved",
            ),
            stacklevel=2,
        )
    if input_transform is not None:
        stamp_user_transform_provenance(trace, input_transform)
    trace.profile_enabled = profile_enabled
    trace.save_grads = save_grads_policy
    if uses_selective_layers_to_save:
        trace._layer_nums_to_save = [
            op.raw_index
            for op in trace.layer_list
            if op.has_saved_activation and op.layer_type not in {"input", "output"}
        ]
        if not save_arg_values:
            trace._replay_arg_version_data_complete = False

    # Print final summary.
    _vprint(
        trace,
        f"Done: {len(trace.layer_logs)} layers, "
        f"{trace.num_saved_ops} saved, "
        f"{trace.total_activation_memory}",
    )

    if layer_visualizers_value:
        _render_layer_visualizers(trace, layer_visualizers_value)

    if unwrap_when_done:
        from .backends.torch.wrappers import unwrap_torch

        unwrap_torch()

    if cache_path is not None and cache_key is not None and cache_secret is not None:
        trace.capture_cache_hit = False
        trace.capture_cache_key = cache_key
        trace.capture_cache_path = str(cache_path)
        _prepare_log_for_capture_cache(trace)
        if _store_authenticated_capture_cache(trace, cache_path, cache_secret):
            _evict_capture_cache(cache_path.parent, keep=cache_path)

    if episode_resolved is not None:
        # Settlement-time episode ledger (S7): written ONCE, after postprocess
        # and outcome settlement, from the settled record + module-call truth.
        write_episode_ledger(trace, episode_resolved)

    # Weightsfree settlement (F33): wrap-generation stamp on every capture
    # (W1-ORD preflight input), D22 invariant net + persisted evidence
    # envelope on structure-only captures.
    from .capture._structure_evidence import settle_weightsfree_capture

    settle_weightsfree_capture(trace, meta_admission)

    return trace


def log_model_metadata(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
) -> Trace:
    """Return model metadata without saving any outs.

    Parameters
    ----------
    model:
        Model whose metadata should be captured.
    input_args:
        Positional input arguments for ``model.forward``.
    input_kwargs:
        Keyword input arguments for ``model.forward``.

    Returns
    -------
    Trace
        Metadata-only trace with input/output distance metadata enabled.
    """

    return trace(
        model,
        input_args,
        input_kwargs,
        capture=CaptureOptions(
            layers_to_save=None,
            compute_input_output_distances=True,
        ),
    )


def _public_impls_module() -> Any:
    """Return private public-command implementations.

    The implementations call ``trace`` and ``_run_model_and_save_specified_outs``
    through this module at call time; nothing is copied into theirs, so a patch
    of either name here is seen while it is in place and gone once it is undone.
    """

    from . import _user_public_impls

    _sync_public_impl_wrapper_metadata(_user_public_impls)
    return _user_public_impls


def release_model(model: nn.Module) -> None:
    """Release a traced PyTorch model from persistent TorchLens preparation.

    Parameters
    ----------
    model:
        Model whose full module tree should be restored. The operation is safe
        for never-traced and already-released models.

    Returns
    -------
    None
        The model is restored in place and may be pickled or traced again.

    Notes
    -----
    TorchLens installs persistent, toggle-gated wrappers on non-root module
    ``forward`` methods. Call ``release_model`` after the final trace when the
    complete model object must be serialized with :func:`torch.save` or
    :mod:`pickle`. Saving ``model.state_dict()`` is unaffected by preparation
    and does not require release.
    """
    model = _model_door.resolve_released_model(model)
    _public_impls_module().release_model(model)


def summary(*args: Any, **kwargs: Any) -> str:
    """Run a one-call capture and return the rendered summary string.

    ``tl.summary(model, x)`` is the first-class one-call front door: it runs a
    metadata-only capture and RETURNS the rendered summary text (it never
    auto-prints). ``trace.summary()`` reports an existing capture instead.
    """

    return cast(str, _public_impls_module().summary(*args, **kwargs))


def show_model_graph(*args: Any, **kwargs: Any) -> None:
    """Forward to the model-graph rendering implementation."""

    return cast(None, _public_impls_module().show_model_graph(*args, **kwargs))


def draw_backward(*args: Any, **kwargs: Any) -> str:
    """Forward to the backward graph rendering implementation."""

    return cast(str, _public_impls_module().draw_backward(*args, **kwargs))


def draw_combined(*args: Any, **kwargs: Any) -> str:
    """Forward to the combined graph rendering implementation."""

    return cast(str, _public_impls_module().draw_combined(*args, **kwargs))


def show_bundle_graph(*args: Any, **kwargs: Any) -> str | None:
    """Forward to the bundle graph rendering implementation."""

    return cast(str | None, _public_impls_module().show_bundle_graph(*args, **kwargs))


def validate_forward_pass(*args: Any, **kwargs: Any) -> bool:
    """Forward to the backend-dispatched forward validation implementation."""

    return cast(bool, _public_impls_module().validate_forward_pass(*args, **kwargs))


def _validate_forward_pass_torch(*args: Any, **kwargs: Any) -> bool:
    """Forward to the torch forward validation implementation."""

    return cast(bool, _public_impls_module()._validate_forward_pass_torch(*args, **kwargs))


def validate_backward_pass(*args: Any, **kwargs: Any) -> bool:
    """Forward to the backward validation implementation."""

    return cast(bool, _public_impls_module().validate_backward_pass(*args, **kwargs))


def validate_batch_of_models_and_inputs(*args: Any, **kwargs: Any) -> Any:
    """Forward to the batch validation implementation."""

    return _public_impls_module().validate_batch_of_models_and_inputs(*args, **kwargs)


_PUBLIC_IMPL_WRAPPER_NAMES = (
    "summary",
    "show_model_graph",
    "draw_backward",
    "draw_combined",
    "show_bundle_graph",
    "validate_forward_pass",
    "validate_backward_pass",
    "validate_batch_of_models_and_inputs",
)

_public_impl_metadata_synced = False


def _sync_public_impl_wrapper_metadata(implementations: Any = None) -> None:
    """Expose canonical signatures on lazily delegated public wrappers.

    ``torchlens._user_public_impls`` imports this module at its top, so when IT is
    the module imported first (``import torchlens._user_public_impls``) the sync
    below observes a partially initialized implementation module. Skipping the
    not-yet-defined names -- instead of raising ``AttributeError`` and breaking a
    standalone import -- keeps the cycle inert; the next
    :func:`_public_impls_module` call completes the sync, and the flag makes the
    completed sync a one-time cost.

    Parameters
    ----------
    implementations:
        Already-resolved implementation module, when the caller holds one.

    Returns
    -------
    None
        Updates wrapper metadata in place once the implementation module is fully
        loaded.
    """

    global _public_impl_metadata_synced
    if _public_impl_metadata_synced:
        return
    if implementations is None:
        from . import _user_public_impls as implementations
    pending = False
    for name in _PUBLIC_IMPL_WRAPPER_NAMES:
        implementation = getattr(implementations, name, None)
        if implementation is None:
            pending = True
            continue
        functools.update_wrapper(globals()[name], implementation)
    _public_impl_metadata_synced = not pending


_sync_public_impl_wrapper_metadata()
