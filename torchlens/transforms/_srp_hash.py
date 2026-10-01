"""Deterministic counter-hash machinery for SRP matrix generation (memo B6).

TorchLens owns its SRP construction: a splitmix64-class integer hash computed
in torch int64 ops with LIMB-DECOMPOSED wrapping arithmetic (no intermediate
ever exceeds 2^40 before a final well-masked compose), so generation is a
pure function of ``(effective_seed, column, draw counter)`` — block-independent
by construction: generating one column block or sixteen chunks yields
bit-identical matrices (O13), which is what makes the O(chunk)-memory
``dense_chunked`` multiply path universally available.

Every byte-level operation here (limb scheme, 63-bit modulo reduction, the
threshold/sign bit split, the rejection tie rule) is part of
``ALGORITHM_VERSION``, not implementation folklore: a change to any of them
is a construction change and bumps the version (T-C13).

Cross-device bit-exactness is REASONED, measured on CPU only, and
gate-verified before any on-device claim ships (memo 16.4; the O3 test stays
executable precisely because "holds by construction" is the claim that must
be tested).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

from ._errors import TransformContractError

__tl_layer__ = "L4"

__all__ = ["ALGORITHM_VERSION", "MatrixHeader"]

#: Numerics-visible version of the owned construction (T-C13): covers the
#: splitmix64 limb scheme, the 63-bit modulo position draw, the
#: first-occurrence-wins rejection rule, the position-keyed sign bit, and the
#: canonical serialization below.
ALGORITHM_VERSION = 1

# splitmix64 round constants (Steele, Lea, Flood 2014).
_SM64_GAMMA = 0x9E3779B97F4A7C15
_SM64_MUL1 = 0xBF58476D1CE4E5B9
_SM64_MUL2 = 0x94D049BB133111EB

# Stream-separation constants for the two-argument position hash.
_STREAM_COLUMN = 0xA24BAED4963EE407
_STREAM_DRAW = 0x9FB21C651E98DF25

_MASK16 = 0xFFFF
_MASK32 = 0xFFFFFFFF
_MASK63 = 0x7FFFFFFFFFFFFFFF


def _lshr(value: torch.Tensor, bits: int) -> torch.Tensor:
    """Logical (zero-fill) right shift on int64 bit patterns.

    Parameters
    ----------
    value:
        int64 tensor holding u64 bit patterns.
    bits:
        Shift amount, ``1 <= bits <= 63``.

    Returns
    -------
    torch.Tensor
        ``value >> bits`` with zero fill.
    """

    return (value >> bits) & ((1 << (64 - bits)) - 1)


def _mul_u64_const(value: torch.Tensor, constant: int) -> torch.Tensor:
    """Wrapping u64 multiply of a tensor by a Python constant, limb-safe.

    Schoolbook multiplication over 16-bit limbs: every partial product is at
    most ``2^32`` and every carry accumulator below ``2^37``, so no int64
    intermediate ever overflows before the final masked compose.

    Parameters
    ----------
    value:
        int64 tensor holding u64 bit patterns.
    constant:
        Non-negative Python int below ``2^64``.

    Returns
    -------
    torch.Tensor
        ``value * constant mod 2^64`` as int64 bit patterns.
    """

    v0 = value & _MASK16
    v1 = _lshr(value, 16) & _MASK16
    v2 = _lshr(value, 32) & _MASK16
    v3 = _lshr(value, 48) & _MASK16
    c0 = constant & _MASK16
    c1 = (constant >> 16) & _MASK16
    c2 = (constant >> 32) & _MASK16
    c3 = (constant >> 48) & _MASK16
    r0 = v0 * c0
    r1 = v1 * c0 + v0 * c1
    r2 = v2 * c0 + v1 * c1 + v0 * c2
    r3 = v3 * c0 + v2 * c1 + v1 * c2 + v0 * c3
    limb0 = r0 & _MASK16
    carry = _lshr(r0, 16)
    acc = r1 + carry
    limb1 = acc & _MASK16
    carry = _lshr(acc, 16)
    acc = r2 + carry
    limb2 = acc & _MASK16
    carry = _lshr(acc, 16)
    limb3 = (r3 + carry) & _MASK16
    return limb0 | (limb1 << 16) | (limb2 << 32) | (limb3 << 48)


def _add_u64_const(value: torch.Tensor, constant: int) -> torch.Tensor:
    """Wrapping u64 add of a Python constant, limb-safe.

    Parameters
    ----------
    value:
        int64 tensor holding u64 bit patterns.
    constant:
        Non-negative Python int below ``2^64``.

    Returns
    -------
    torch.Tensor
        ``value + constant mod 2^64`` as int64 bit patterns.
    """

    lo = (value & _MASK32) + (constant & _MASK32)
    hi = (_lshr(value, 32) + ((constant >> 32) & _MASK32) + _lshr(lo, 32)) & _MASK32
    return (lo & _MASK32) | (hi << 32)


def _splitmix64(value: torch.Tensor) -> torch.Tensor:
    """One splitmix64 finalizer round over u64 bit patterns.

    Parameters
    ----------
    value:
        int64 tensor holding u64 bit patterns.

    Returns
    -------
    torch.Tensor
        Mixed u64 bit patterns.
    """

    z = _add_u64_const(value, _SM64_GAMMA)
    z = z ^ _lshr(z, 30)
    z = _mul_u64_const(z, _SM64_MUL1)
    z = z ^ _lshr(z, 27)
    z = _mul_u64_const(z, _SM64_MUL2)
    return z ^ _lshr(z, 31)


def _splitmix64_int(value: int) -> int:
    """Pure-Python reference of one splitmix64 round (the exactness oracle).

    Parameters
    ----------
    value:
        Non-negative Python int below ``2^64``.

    Returns
    -------
    int
        Mixed u64 value.
    """

    z = (value + _SM64_GAMMA) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 30
    z = (z * _SM64_MUL1) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 27
    z = (z * _SM64_MUL2) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def _hash_position(seed: int, column: torch.Tensor, draw: torch.Tensor) -> torch.Tensor:
    """Position-keyed two-argument counter hash.

    A pure function of ``(seed, column, draw)`` with no sequential state, so
    any column chunking generates bit-identical values (O13).

    Parameters
    ----------
    seed:
        Effective seed (non-negative, below ``2^63``).
    column:
        int64 tensor of output-column ids.
    draw:
        int64 tensor of per-column draw counters (broadcastable to
        ``column``).

    Returns
    -------
    torch.Tensor
        u64 bit patterns, one per (column, draw) pair.
    """

    x = _mul_u64_const(column, _STREAM_COLUMN)
    x = _splitmix64(_add_u64_const(x, seed & _MASK63))
    y = _mul_u64_const(draw, _STREAM_DRAW)
    return _splitmix64(x ^ y)


def draw_positions(
    seed: int, column: torch.Tensor, draw: torch.Tensor, extent: int
) -> torch.Tensor:
    """Draw input positions in ``[0, extent)`` from the counter hash.

    The reduction takes the LOW 63 BITS then a plain modulo — the modulo
    bias is below ``extent / 2^63`` and the top-bit drop below ``2^-63``,
    both documented as part of :data:`ALGORITHM_VERSION`.

    Parameters
    ----------
    seed:
        Effective seed.
    column:
        int64 tensor of output-column ids.
    draw:
        int64 tensor of draw counters.
    extent:
        Input extent ``D``.

    Returns
    -------
    torch.Tensor
        int64 positions in ``[0, extent)``.
    """

    return (_hash_position(seed, column, draw) & _MASK63) % extent


def position_signs(seed: int, column: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Derive the +/-1 sign for each (column, position) pair.

    Signs are keyed to the POSITION, never the draw slot, so a rejection
    redraw can never move a sign (part of the tie/probe rule).

    Parameters
    ----------
    seed:
        Effective seed.
    column:
        int64 tensor of output-column ids.
    position:
        int64 tensor of input positions.

    Returns
    -------
    torch.Tensor
        int8 tensor of ``+1`` / ``-1`` signs.
    """

    bits = _hash_position(seed ^ 0x5851F42D4C957F2D, column, position)
    return ((bits & 1) * 2 - 1).to(torch.int8)


#: Safety cap on rejection rounds (probabilistically unreachable under
#: ``m <= D``; the loop's else-branch is the construction-bug tripwire).
_MAX_REJECTION_ROUNDS = 100_000


def generate_fixed_columns(
    seed: int,
    extent: int,
    columns_block: range,
    nonzeros_per_column: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``very_sparse_fixed`` columns by deterministic rejection.

    Each output column holds exactly ``nonzeros_per_column`` DISTINCT input
    positions: draws come from consecutive counters; a slot whose position
    was already produced by an EARLIER slot (first occurrence wins; earlier =
    lower draw counter, with the value-sorted stable order breaking ties)
    redraws from the column's next unused counter, in slot order, until all
    positions are distinct. Realized nnz therefore EQUALS
    ``len(columns_block) * nonzeros_per_column`` exactly.

    Parameters
    ----------
    seed:
        Effective seed.
    extent:
        Input extent ``D``; must satisfy ``nonzeros_per_column <= extent``.
    columns_block:
        GLOBAL ids of the output columns to generate (a ``range``, so any
        chunking of the full ``range(k)`` is bit-identical by construction).
    nonzeros_per_column:
        Exact per-column nonzero count ``m``.
    device:
        Torch device for generation (default CPU).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(positions, signs)``: positions int64 ``(len(columns_block), m)``
        sorted ascending per column; signs int8 aligned to the sorted
        positions.
    """

    if nonzeros_per_column > extent:
        raise TransformContractError(
            f"SRP cannot place {nonzeros_per_column} DISTINCT nonzeros in an "
            f"input extent of {extent}; the realized density exceeds 1.",
            code="transform_params_invalid",
            remedy="lower density (or n_components) so density * extent <= extent",
            extent=extent,
            nonzeros_per_column=nonzeros_per_column,
        )
    dev = device if device is not None else torch.device("cpu")
    m = nonzeros_per_column
    n_columns = len(columns_block)
    columns = torch.arange(
        columns_block.start, columns_block.stop, dtype=torch.int64, device=dev
    ).unsqueeze(1)
    draws = torch.arange(m, dtype=torch.int64, device=dev).unsqueeze(0).expand(n_columns, m)
    positions = draw_positions(seed, columns.expand(n_columns, m), draws, extent)
    next_counter = torch.full((n_columns, 1), m, dtype=torch.int64, device=dev)
    for _ in range(_MAX_REJECTION_ROUNDS):
        sorted_vals, order = torch.sort(positions, dim=1, stable=True)
        dup_sorted = torch.zeros_like(sorted_vals, dtype=torch.bool)
        dup_sorted[:, 1:] = sorted_vals[:, 1:] == sorted_vals[:, :-1]
        duplicate = torch.zeros_like(dup_sorted)
        duplicate.scatter_(1, order, dup_sorted)
        if not bool(duplicate.any().item()):
            break
        # Redraw duplicate slots in slot order from each column's next
        # unused counters (the deterministic tie/probe rule).
        redraw_rank = torch.cumsum(duplicate.to(torch.int64), dim=1) - 1
        redraw_counter = next_counter + redraw_rank
        fresh = draw_positions(seed, columns.expand(n_columns, m), redraw_counter, extent)
        positions = torch.where(duplicate, fresh, positions)
        next_counter = next_counter + duplicate.to(torch.int64).sum(dim=1, keepdim=True)
    else:
        # Probabilistically unreachable under m <= D; kept as a hard tripwire.
        raise TransformContractError(
            "SRP rejection sampling failed to converge; this indicates a "
            "construction bug, not user error.",
            code="transform_matrix_verification_failed",
            remedy="report this with the (seed, extent, n_components, density) tuple",
            seed=seed,
            extent=extent,
        )
    positions, _ = torch.sort(positions, dim=1)
    signs = position_signs(seed, columns.expand(n_columns, m), positions)
    return positions, signs


def generate_bernoulli_columns(
    seed: int,
    extent: int,
    columns_block: range,
    density: float,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate ``iid_bernoulli`` columns (every entry independently nonzero).

    Entry ``(j, i)`` is nonzero iff the high 62 hash bits of the
    position-keyed counter hash fall below ``floor(density * 2^62)``; the
    sign comes from bit 0 of the same word. O(extent * len(columns_block))
    work — the planner discloses this cost so a multi-second generation
    cannot surprise a harvest (memo section 6).

    Parameters
    ----------
    seed:
        Effective seed.
    extent:
        Input extent ``D``.
    columns_block:
        GLOBAL ids of the output columns to generate (a ``range``; any
        chunking of the full ``range(k)`` is bit-identical).
    density:
        Bernoulli nonzero probability in ``(0, 1]``.
    device:
        Torch device for generation (default CPU).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(column_ids, positions, signs)`` flat COO triplets, positions
        ascending within each column; column ids are GLOBAL.
    """

    dev = device if device is not None else torch.device("cpu")
    n_columns = len(columns_block)
    cut = int(density * (1 << 62))
    columns = torch.arange(
        columns_block.start, columns_block.stop, dtype=torch.int64, device=dev
    ).unsqueeze(1)
    rows = torch.arange(extent, dtype=torch.int64, device=dev).unsqueeze(0)
    words = _hash_position(seed, columns.expand(n_columns, extent), rows.expand(n_columns, extent))
    threshold_bits = _lshr(words, 1) & ((1 << 62) - 1)
    keep = threshold_bits < cut
    col_ids, positions = torch.nonzero(keep, as_tuple=True)
    signs = ((words[keep] & 1) * 2 - 1).to(torch.int8)
    return col_ids + columns_block.start, positions, signs


@dataclass(frozen=True)
class MatrixHeader:
    """The canonical digest header: the construction's declared byte-level facts.

    Attributes
    ----------
    construction:
        Construction name recorded in the header.
    extent:
        Input extent ``D``.
    n_components:
        Output extent ``k``.
    nonzeros_per_column:
        Exact per-column nonzero count ``m`` (``-1`` for ``iid_bernoulli``,
        whose realized counts are Binomial).
    scale:
        Nonzero magnitude, serialized as its exact hex float.
    """

    construction: str
    extent: int
    n_components: int
    nonzeros_per_column: int
    scale: float


def digest_fixed_matrix(positions: torch.Tensor, signs: torch.Tensor, header: MatrixHeader) -> str:
    """Digest the canonical serialized matrix BEFORE any layout conversion.

    Canonical form: a canonical-JSON header followed by, per output column
    in order, the ascending int64 little-endian positions then the aligned
    int8 signs. The digest promise is exact; floating OUTPUT parity across
    kernels/devices is tolerance-based and rides ``multiply_path`` instead.

    Parameters
    ----------
    positions:
        int64 ``(k, m)`` ascending per-column positions.
    signs:
        int8 ``(k, m)`` aligned signs.
    header:
        The construction facts serialized into the canonical header.

    Returns
    -------
    str
        ``sha256:<hex>`` digest of the canonical bytes.
    """

    header_bytes = json.dumps(
        {
            "algorithm_version": ALGORITHM_VERSION,
            "construction": header.construction,
            "extent": header.extent,
            "n_components": header.n_components,
            "nonzeros_per_column": header.nonzeros_per_column,
            "scale_hex": float(header.scale).hex(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    hasher = hashlib.sha256()
    hasher.update(header_bytes)
    hasher.update(positions.to("cpu", torch.int64).contiguous().numpy().tobytes())
    hasher.update(signs.to("cpu", torch.int8).contiguous().numpy().tobytes())
    return "sha256:" + hasher.hexdigest()
