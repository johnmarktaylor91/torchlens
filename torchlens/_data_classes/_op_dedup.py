"""Saved-activation dedup helpers for retained payload copies.

Split out of ``op.py`` (F20 FIX-2 size-ratchet split, the ``_op_transforms``
precedent): the identity/content dedup probes that decide whether a payload
about to be retained can reuse an already-saved copy, plus the content hash
they share. ``op.py`` re-exposes the two dedup entry points unchanged (the
torch backend imports through it), so the public import site is unmoved.
``IdentityCacheEntry`` is the per-trace ``_out_identity_cache`` value shape:
the source slot is a weakref (F20 W1a) so dedup bookkeeping never pins a
live forward intermediate.
"""

import hashlib
import weakref
from typing import TYPE_CHECKING, Any, cast

import torch

from .._utils._torch_compat import tensor_version_or_none
from .._utils.tensor_utils import is_functorch_wrapped_tensor

# _state / _transport sit above this module's layer (arch-spine layer lint):
# their names defer inside the one function each serves rather than
# importing eagerly here.

if TYPE_CHECKING:
    from .trace import Trace

#: Value shape of ``Trace._out_identity_cache``: weak source ref, raw layer
#: label, saved payload, source ``_version``, and the dense dedup ordinal.
IdentityCacheEntry = tuple[weakref.ref[torch.Tensor], str, torch.Tensor, int | None, int]


class _IdentityDedupCache(dict[int, IdentityCacheEntry]):
    """Per-pass identity dedup cache that also issues the dense source ordinals.

    The cache is keyed by ``id()``, and a new tensor at a dead source's
    address replaces that slot without growing the cache. Ordinals therefore
    come from ``issued``, the count handed out since the cache was created or
    last cleared, never from the cache size: each source gets a fresh ordinal
    in capture order, whatever the heap reuses.
    """

    def __init__(self, entries: dict[int, IdentityCacheEntry] | None = None) -> None:
        """Adopt ``entries``, issuing after the highest ordinal they hold."""
        super().__init__(entries or {})
        self.issued = max((entry[4] for entry in self.values()), default=0)

    def clear(self) -> None:
        """Drop every slot and restart the ordinals (a new pass)."""
        super().clear()
        self.issued = 0

    def next_ordinal(self) -> int:
        """Return the next unused dense ordinal."""
        self.issued += 1
        return self.issued


def _tensor_content_hash(value: torch.Tensor) -> str:
    """Return a CPU content hash for a tensor.

    Parameters
    ----------
    value:
        Tensor to hash.

    Returns
    -------
    str
        SHA-256 digest.

    Notes
    -----
    The digest frames the LOGICAL dtype so a bfloat16 tensor can never
    collide with the float32 tensor of the same values (content-mode dedup
    aliasing across dtypes). The payload is hashed through the buffer
    protocol (no whole-payload ``tobytes`` copy), the transport is the
    shared zero-copy-when-possible ``to_cpu_contiguous`` (r7 b5 R35-1: the
    old ``safe_copy(...).cpu().contiguous()`` paid one unconditional full
    clone for an already-contiguous CPU tensor and materialized twice for a
    CUDA/permuted source), and bf16 hashes its OWN bytes -- the uint8
    reinterpret view needs no numpy-transport upcast (R35 fable: the
    bf16->f32 copy was pointless once the logical dtype was framed; this
    digest is process-local, so the byte change is invisible).
    """

    if is_functorch_wrapped_tensor(value):
        return f"functorch_wrapped_tensor:{id(value)}"

    from .._state import pause_logging
    from .._transport import digest_byte_view

    with pause_logging():
        # r8 R35: conj/neg resolution + the uint8 reinterpret live in the
        # ONE transport authority (``_transport.digest_byte_view``); the
        # local resolve guard this site pioneered moved there so the other
        # digest sites cannot drift from it.
        logical_dtype = str(value.dtype)
        shape = tuple(value.shape)
        payload = digest_byte_view(value)
        hasher = hashlib.sha256()
        hasher.update(repr((shape, logical_dtype)).encode("utf-8"))
        hasher.update(payload)
    return hasher.hexdigest()


def _dedup_cached_identity_out(
    trace: "Trace | None",
    source_tensor: torch.Tensor,
    annotations: dict[str, Any],
    save_arg_values: bool,
) -> torch.Tensor | None:
    """Return the already-saved payload for this live source, or ``None``.

    Parameters
    ----------
    trace:
        Trace that owns the per-pass dedup caches.
    source_tensor:
        Live output tensor about to be copied for retention.
    annotations:
        Mutable annotation dictionary for the saved output.
    save_arg_values:
        Whether argument values are being saved (disables activation dedup).

    Notes
    -----
    Pre-copy identity probe: the historical order CLONED the payload first
    and only consulted the identity cache afterwards, discarding the fresh
    clone on every hit — a full wasted payload copy per repeated-source save
    (dedup-after-copy ordering). Hit semantics, annotations, and the miss
    path (which still inserts post-copy via
    :func:`_dedup_saved_activation_out`) are unchanged.
    """

    if trace is None or save_arg_values or source_tensor.is_meta:
        return None
    try:
        dedup_mode = trace._out_dedup_mode
    except AttributeError:
        dedup_mode = "identity"
    if dedup_mode != "identity":
        return None
    try:
        identity_cache = trace._out_identity_cache
    except AttributeError:
        identity_cache = None
    if identity_cache is None:
        return None
    source_key = id(source_tensor)
    from ..backends.torch._internal_read import internal_scalar_read

    with internal_scalar_read():
        source_version = tensor_version_or_none(source_tensor)
    cached = identity_cache.get(source_key)
    if cached is None:
        return None
    cached_source_ref, cached_label, cached_out, cached_version, cached_ordinal = cached
    # F20 W1a: the source slot is a weakref so bookkeeping never pins a live
    # forward intermediate; a dead referent is an ordinary miss (the id may
    # have been reused by a new tensor).
    cached_source = cached_source_ref()
    if cached_source is source_tensor and cached_version == source_version:
        # B3R4-R21-1: the annotation carries the trace-local dense dedup
        # ordinal, never the raw ``id()`` bookkeeping key -- a memory address
        # in a persisted field made same-program artifacts byte-differ per
        # process and exposed a meaningless public value.
        annotations["dedup_source_id"] = cached_ordinal
        annotations["dedup_source_version"] = source_version
        annotations["dedup_reference_label"] = cached_label
        return cast(torch.Tensor, cached_out)
    return None


def _ordinal_identity_cache(trace: "Trace") -> _IdentityDedupCache:
    """Return ``trace``'s identity dedup cache, adopting a plain dict on first use.

    Parameters
    ----------
    trace:
        Trace that owns the per-pass dedup caches.

    Returns
    -------
    _IdentityDedupCache
        The trace's cache, now able to issue ordinals.
    """

    try:
        identity_cache = trace._out_identity_cache
    except AttributeError:
        identity_cache = None
    if not isinstance(identity_cache, _IdentityDedupCache):
        identity_cache = _IdentityDedupCache(identity_cache)
        trace._out_identity_cache = identity_cache
    return identity_cache


def _dedup_saved_activation_out(
    trace: "Trace | None",
    source_tensor: torch.Tensor,
    raw_out: torch.Tensor,
    label: str,
    annotations: dict[str, Any],
    save_arg_values: bool,
) -> torch.Tensor:
    """Return a deduplicated saved activation payload when configured.

    Parameters
    ----------
    trace:
        Trace that owns the per-pass dedup caches.
    source_tensor:
        Live output tensor before ``safe_copy`` created ``raw_out``.
    raw_out:
        Copied activation payload.
    label:
        Raw layer label for the saved output.
    annotations:
        Mutable annotation dictionary for the saved output.
    save_arg_values:
        Whether argument values are being saved. Argument snapshots consume
        independent payloads, so activation dedup is disabled in that mode.

    Returns
    -------
    torch.Tensor
        Either ``raw_out`` or a previously saved payload for the same live
        source tensor.
    """

    if trace is None or save_arg_values or raw_out.is_meta:
        return raw_out

    try:
        mode = trace._out_dedup_mode
    except AttributeError:
        mode = "identity"
    if mode == "none":
        return raw_out

    if mode == "content":
        try:
            hash_cache = trace._out_hash_cache
        except AttributeError:
            hash_cache = None
        if hash_cache is None:
            hash_cache = {}
            trace._out_hash_cache = hash_cache
        # R36: the content digest is a host-side byte read; a cpu_async
        # payload may still be an in-flight pinned buffer. No-op unless
        # async fence events are pending.
        from .._utils.tensor_utils import synchronize_pending_cpu_async_copies

        synchronize_pending_cpu_async_copies()
        out_hash = _tensor_content_hash(raw_out)
        if out_hash in hash_cache:
            annotations["dedup_out_hash"] = out_hash
            annotations["dedup_reference_label"] = hash_cache[out_hash][0]
            return hash_cache[out_hash][1]
        hash_cache[out_hash] = (label, raw_out)
        return raw_out

    identity_cache = _ordinal_identity_cache(trace)
    source_key = id(source_tensor)
    # r65: TorchLens's OWN dedup-bookkeeping ``_version`` read runs under the explicit
    # internal-read marker so the r65 state-metadata property observer never mistakes it
    # for a user ``._version`` read on a registered buffer/param receiver (unmarked it
    # fires for every saved state source and would spuriously refuse any model whose
    # consumed buffer was ever mutated in place before capture). Imported lazily:
    # ``data_classes`` sits below the torch backend in the layering.
    from ..backends.torch._internal_read import internal_scalar_read

    with internal_scalar_read():
        source_version = tensor_version_or_none(source_tensor)
    cached = identity_cache.get(source_key)
    if cached is not None and cached[0]() is not source_tensor:
        # F20 W1a: weak source slot (see _dedup_cached_identity_out). A dead
        # referent means a new tensor reused the address: a new source.
        cached = None
    if cached is not None:
        _, cached_label, cached_out, cached_version, cached_ordinal = cached
        if cached_version == source_version:
            # B3R4-R21-1: dense trace-local ordinal, never the raw ``id()``.
            annotations["dedup_source_id"] = cached_ordinal
            annotations["dedup_source_version"] = source_version
            annotations["dedup_reference_label"] = cached_label
            return cached_out

    # Dense ordinal in first-save order from the cache's own counter, so it is
    # the same in any process and heap state; the same live source at a new
    # ``_version`` keeps its ordinal (the version tells the saves apart).
    ordinal = cached[4] if cached is not None else identity_cache.next_ordinal()
    identity_cache[source_key] = (
        weakref.ref(source_tensor),
        label,
        raw_out,
        source_version,
        ordinal,
    )
    return raw_out
