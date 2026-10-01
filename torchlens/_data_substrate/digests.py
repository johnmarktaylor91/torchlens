"""Digest primitives, built once, used everywhere (extract memo item 1).

Two primitives with PINNED algorithm ids + versions (they are manifest fields
any third party can recompute from):

* :func:`merkle_digest` — the threaded Merkle CRYPTO digest (extract D6):
  per-entry leaf digests hashed in a thread pool, folded in canonical input
  order — deterministic across thread counts by construction. blake2b
  default, sha256 selectable. Measured 5.4x faster than the serial fold on a
  real 3.55 GB model (827 ms vs 4478 ms).
* :func:`value_reduction` — the order-sensitive VALUE reduction (extract D7):
  int64 Horner over the flattened byte view with a single BLOCK-length
  position-weight vector built once and SLICED (a per-size cache measured
  50x slower on many-small-tensor vision models), 0-dim-safe flatten before
  the byte view, per-tensor name/shape/dtype fold. Runs ON the tensor's
  device (computable before the host copy) and is re-verifiable after load.
  It is NEVER called a file checksum or an identity — it covers tensor
  VALUES, and a byte flipped in container metadata is invisible to it
  (CRC-32 over file bytes covers that disjoint class).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

from .._errors import InvalidArgumentError

__tl_layer__ = "L3"

__all__ = [
    "MERKLE_ALGORITHM_ID",
    "MERKLE_ALGORITHM_VERSION",
    "VALUE_REDUCTION_ALGORITHM_ID",
    "VALUE_REDUCTION_ALGORITHM_VERSION",
    "MerkleDigest",
    "merkle_digest",
    "value_reduction",
]

#: Pinned algorithm id of the threaded Merkle state digest.
MERKLE_ALGORITHM_ID = "tl_state_merkle"

#: Pinned algorithm version; any byte-level change to leaf or fold encoding bumps it.
MERKLE_ALGORITHM_VERSION = 1

#: Pinned algorithm id of the order-sensitive value reduction.
VALUE_REDUCTION_ALGORITHM_ID = "tl_value_reduction"

#: Pinned algorithm version; any change to base, block length, or fold bumps it.
VALUE_REDUCTION_ALGORITHM_VERSION = 1

#: Horner base (the 64-bit FNV prime; odd, well-mixed under wrapping int64).
_HORNER_BASE = 1099511628211

#: Position-weight vector length; the vector is built once and SLICED.
_HORNER_BLOCK = 65536

_MASK64 = (1 << 64) - 1

#: Supported crypto hash constructors (closed; recorded in the manifest).
_HASHES: dict[str, Callable[[], Any]] = {
    "blake2b": hashlib.blake2b,
    "sha256": hashlib.sha256,
}


def _tensor_payload_bytes(tensor: torch.Tensor) -> bytes:
    """Return a tensor's raw little-endian payload bytes (0-dim safe).

    Parameters
    ----------
    tensor:
        Any dense tensor (bf16/fp8 included; the byte view never routes
        through a NumPy dtype for the element type itself).

    Returns
    -------
    bytes
        The flattened contiguous payload bytes.
    """

    flat = tensor.detach().reshape(-1).cpu().contiguous()
    if flat.numel() == 0:
        return b""
    return flat.view(torch.uint8).numpy().tobytes()


def _leaf_header(name: str, tensor: torch.Tensor) -> bytes:
    """Encode one leaf's name/shape/dtype/layout header.

    Parameters
    ----------
    name:
        Canonical entry name (e.g. the state_dict key).
    tensor:
        The entry tensor.

    Returns
    -------
    bytes
        NUL-separated header bytes folded into the leaf digest.
    """

    return "\x00".join(
        [name, repr(tuple(tensor.shape)), str(tensor.dtype), str(tensor.layout)]
    ).encode("utf-8")


@dataclass(frozen=True)
class MerkleDigest:
    """The recomputable record of one threaded-Merkle fold.

    Attributes
    ----------
    digest:
        ``"<hash>:<hex>"`` of the canonical-order fold.
    algorithm_id:
        :data:`MERKLE_ALGORITHM_ID` (pinned).
    algorithm_version:
        :data:`MERKLE_ALGORITHM_VERSION` (pinned).
    hash_name:
        Crypto hash used for leaves and fold (``"blake2b"`` or ``"sha256"``).
    n_leaves:
        Number of entries folded.
    """

    digest: str
    algorithm_id: str
    algorithm_version: int
    hash_name: str
    n_leaves: int

    def record(self) -> dict[str, str | int]:
        """Return the JSON-portable manifest record.

        Returns
        -------
        dict[str, str | int]
            Algorithm id + version + hash + digest + leaf count.
        """

        return {
            "digest": self.digest,
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "hash": self.hash_name,
            "n_leaves": self.n_leaves,
        }


def merkle_digest(
    entries: Iterable[tuple[str, torch.Tensor]],
    *,
    hash_name: str = "blake2b",
    max_workers: int | None = None,
) -> MerkleDigest:
    """Fold named tensors into one thread-count-invariant crypto digest.

    Per-entry leaf digests (``H(header || payload)``) are computed in a
    thread pool; the fold hashes leaf digests in CANONICAL INPUT ORDER, so
    the result is deterministic across thread counts and reproducible by a
    third party from one library call.

    Parameters
    ----------
    entries:
        Ordered ``(name, tensor)`` pairs — the caller's order IS the
        canonical order (e.g. ``state_dict()`` order).
    hash_name:
        ``"blake2b"`` (default) or ``"sha256"``.
    max_workers:
        Optional thread-pool width; ``None`` lets the pool size itself.

    Returns
    -------
    MerkleDigest
        The digest record (algorithm id + version pinned).

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``digest_hash_invalid`` on an unknown ``hash_name`` (the vocabulary
        is closed because the name is a recomputation contract in the
        manifest).
    """

    if hash_name not in _HASHES:
        raise InvalidArgumentError(
            f"merkle_digest hash {hash_name!r} is not in the closed vocabulary "
            f"{sorted(_HASHES)}; the hash name is recorded in the manifest as "
            "a recomputation contract.",
            code="digest_hash_invalid",
            remedy="pass hash_name='blake2b' (default) or 'sha256'",
            hash_name=hash_name,
        )
    ordered: Sequence[tuple[str, torch.Tensor]] = list(entries)
    hasher_ctor = _HASHES[hash_name]

    def _leaf(pair: tuple[str, torch.Tensor]) -> bytes:
        """Hash one ``(name, tensor)`` entry into its Merkle leaf digest."""

        name, tensor = pair
        hasher = hasher_ctor()
        hasher.update(_leaf_header(name, tensor))
        hasher.update(b"\x00")
        hasher.update(_tensor_payload_bytes(tensor))
        return hasher.digest()

    if ordered:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            leaves = list(pool.map(_leaf, ordered))
    else:
        leaves = []
    fold = hasher_ctor()
    fold.update(f"{MERKLE_ALGORITHM_ID}_v{MERKLE_ALGORITHM_VERSION}".encode())
    for leaf in leaves:
        fold.update(leaf)
    return MerkleDigest(
        digest=f"{hash_name}:{fold.hexdigest()}",
        algorithm_id=MERKLE_ALGORITHM_ID,
        algorithm_version=MERKLE_ALGORITHM_VERSION,
        hash_name=hash_name,
        n_leaves=len(ordered),
    )


def _position_weights(device: torch.device) -> torch.Tensor:
    """Return the BLOCK-length descending position-weight vector for a device.

    Built once per device and SLICED for shorter final blocks (never a
    per-size cache — measured 50x slower on many-small-tensor models).

    Parameters
    ----------
    device:
        Device the reduction runs on.

    Returns
    -------
    torch.Tensor
        ``int64[_HORNER_BLOCK]`` with ``W[i] = base^(BLOCK-1-i) mod 2^64``.
    """

    cached = _WEIGHTS_BY_DEVICE.get(str(device))
    if cached is not None:
        return cached
    factors = torch.full((_HORNER_BLOCK,), _HORNER_BASE, dtype=torch.int64, device=device)
    factors[0] = 1
    weights = torch.flip(torch.cumprod(factors, dim=0), dims=(0,))
    _WEIGHTS_BY_DEVICE[str(device)] = weights
    return weights


_WEIGHTS_BY_DEVICE: dict[str, torch.Tensor] = {}


def _to_unsigned(value: int) -> int:
    """Map a signed int64 into the unsigned 64-bit ring.

    Parameters
    ----------
    value:
        Possibly-negative int64 value.

    Returns
    -------
    int
        ``value mod 2^64``.
    """

    return value & _MASK64


def value_reduction(name: str, tensor: torch.Tensor) -> str:
    """Compute the order-sensitive int64 Horner value reduction (extract D7).

    Runs with torch ops ON the tensor's device (computable before any host
    copy) and re-verifiable after load. Covers tensor VALUES in order plus a
    per-tensor name/shape/dtype fold; it is not a file checksum and not an
    identity.

    Parameters
    ----------
    name:
        Canonical entry name folded into the seed.
    tensor:
        Any dense tensor; 0-dim flattens safely.

    Returns
    -------
    str
        ``"0x..."`` — the 64-bit reduction in hex (JSON-portable).
    """

    header = hashlib.blake2b(
        f"{name}\x00{tuple(tensor.shape)!r}\x00{tensor.dtype}".encode(), digest_size=8
    ).digest()
    acc = int.from_bytes(header, "little")
    flat = tensor.detach().reshape(-1)
    if flat.numel() > 0:
        flat = flat.contiguous()
        byte_view = flat.view(torch.uint8)
        weights = _position_weights(byte_view.device)
        n = byte_view.numel()
        for start in range(0, n, _HORNER_BLOCK):
            block = byte_view[start : start + _HORNER_BLOCK].to(torch.int64)
            length = block.numel()
            block_sum = _to_unsigned(int((block * weights[_HORNER_BLOCK - length :]).sum().item()))
            acc = (acc * pow(_HORNER_BASE, length, 1 << 64) + block_sum) & _MASK64
    return f"0x{acc:016x}"
