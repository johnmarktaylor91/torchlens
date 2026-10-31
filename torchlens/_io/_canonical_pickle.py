"""Hash-seed-independent metadata pickling for portable bundles (B3R4-R21-2).

Split out of ``scrub.py`` under the R43 file-size ratchet: this is the
byte-determinism seam for ``metadata.pkl``, consumed by the bundle and
streaming writers, independent of the scrub walk itself.
"""

import pickle
import struct
from typing import Any

import torch

from ._inline_tensor import reduce_inline_tensor

__all__ = ["dump_canonical_metadata"]


def _canonical_member_key(member: Any) -> tuple[int, str, str]:
    """Total order that stays deterministic for NaN-carrying float members."""

    if isinstance(member, float):
        return (0, struct.pack(">d", member).hex(), "")
    return (1, type(member).__qualname__, repr(member))


class _CanonicalMetadataPickler(pickle._Pickler):
    """Pickler that emits exact set/frozenset members in sorted order (B3R4-R21-2).

    A ``frozenset`` pickles in its hash-table iteration order, which for str
    elements is salted by ``PYTHONHASHSEED`` -- so two processes capturing the
    identical program emitted byte-different ``metadata.pkl`` for identical
    logical content (the M6 relation views are exact frozensets of labels).
    Rewriting the reduction as ``cls(sorted_members)`` makes the persisted
    bytes hash-seed independent; ``builtins.set``/``builtins.frozenset`` are
    already on the safe-unpickler's explicit-globals allowlist, so loads
    admit the REDUCE spelling. Exact types only: subclasses keep their own
    reduce protocol.

    Deliberately the pure-Python ``pickle._Pickler``: the C pickler hardcodes
    exact set/frozenset saves and consults neither ``reducer_override`` nor
    ``dispatch_table`` for them (probed on 3.10). Metadata is the small
    sidecar of a bundle (tensor payloads ride safetensors blobs), so the
    slower Python walk is a save-time-only, determinism-buying cost.
    """

    def reducer_override(self, obj: Any) -> Any:
        """Return the canonical reduction for exact set/frozenset and plain tensor values.

        A plain dense ``torch.Tensor`` is emitted through the TorchLens-owned
        inline reduction (``_inline_tensor``: raw bytes, dtype, shape,
        ``requires_grad``, device) instead of torch's default reduction, whose
        storage leg (``torch.storage._load_from_bytes``) the restricted loader
        refuses. Every other tensor kind keeps torch's own reduction, which the
        loader judges.
        """

        cls = type(obj)
        if cls is torch.Tensor:
            inline = reduce_inline_tensor(obj)
            if inline is not None:
                return inline
            return NotImplemented
        if cls is frozenset or cls is set:
            members_list = list(obj)
            if any(isinstance(member, float) and member != member for member in members_list):
                # NaN members defeat ``sorted``'s comparison-based order (every
                # comparison is False), silently leaving hash/iteration order in
                # the bytes -- the exact instability this pickler exists to
                # kill. Order them by their IEEE bit pattern instead (b3-sol
                # hardening note: closed by construction, not by probe).
                members = sorted(members_list, key=_canonical_member_key)
            else:
                try:
                    members = sorted(members_list)
                except TypeError:
                    # Heterogeneous members: any deterministic total order works
                    # for byte stability; type-name-then-repr is stable for the
                    # pure-data values the scrub admits.
                    members = sorted(
                        members_list,
                        key=lambda member: (type(member).__qualname__, repr(member)),
                    )
            return (cls, (members,))
        return NotImplemented


def dump_canonical_metadata(state: Any, handle: Any) -> None:
    """Pickle scrubbed trace state with hash-seed-independent container bytes."""

    _CanonicalMetadataPickler(handle, protocol=pickle.HIGHEST_PROTOCOL).dump(state)
