"""THE one absolute-byte three-valued alias/overlap engine (r37 INV-2).

Every disjointness / overlap / identity / containment proof over tensor memory in the
runnable witness/execution surface routes through these helpers. Local pointer-equality
shortcuts are FORBIDDEN (hon1_1: ``torch.from_numpy(arr[:6])`` vs ``arr[2:8]`` own
DISTINCT torch storages with distinct base pointers over genuinely overlapping host
memory, so ``data_ptr() != data_ptr()`` is never a disjointness proof). All coordinates
are ABSOLUTE, device-scoped byte addresses; the relation vocabulary is exactly
``overlap | disjoint | unknown`` and anything unproven is ``unknown`` (fail closed).

Split out of ``tensor_utils.py`` under the R43 file-size ratchet (fixwave-5 settle);
the engine is self-contained and its consumers import it directly.
"""

from typing import Literal

import torch

AliasRelation = Literal["overlap", "disjoint", "unknown"]
"""Three-valued alias-proof vocabulary (INV-2). ``unknown`` is a first-class verdict."""

ALIAS_ENUMERATION_ELEMENT_CAP = 65536
"""Exact-enumeration bound (inclusive, per view) for the alias proof engine."""


class TensorByteFootprint:
    """Absolute, device-scoped byte footprint of one strided tensor view.

    ``start_byte``/``end_byte`` bound the touched span on ABSOLUTE addresses
    (``storage.data_ptr()`` + offset + min/max stride contributions; negative and
    zero strides sound). ``origin_byte`` is the absolute address of the
    ``storage_offset`` element (the grid origin for residue/enumeration proofs).
    """

    __slots__ = (
        "device_key",
        "start_byte",
        "end_byte",
        "origin_byte",
        "element_size",
        "shape",
        "strides",
        "numel",
    )

    def __init__(
        self,
        device_key: tuple[str, int | None],
        start_byte: int,
        end_byte: int,
        origin_byte: int,
        element_size: int,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        numel: int,
    ) -> None:
        self.device_key = device_key
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.origin_byte = origin_byte
        self.element_size = element_size
        self.shape = shape
        self.strides = strides
        self.numel = numel


def tensor_byte_footprint(value: torch.Tensor) -> TensorByteFootprint | None:
    """Compute a tensor's absolute byte footprint, or ``None`` when unprovable.

    ``None`` (the caller must treat the relation as ``unknown``) covers exotic
    layouts that refuse geometry reads AND any tensor whose storage base pointer is
    ``0`` with nonzero elements -- every meta tensor reports ``data_ptr() == 0``, so
    absolute-address math on it would collide unrelated tensors (pre-closed r38
    adjacent: meta ``data_ptr==0``).
    """

    try:
        storage_ptr = int(value.untyped_storage().data_ptr())
        element_size = int(value.element_size())
        numel = int(value.numel())
        if storage_ptr == 0 and numel > 0:
            return None
        device = value.device
        device_key = (str(device.type), device.index)
        origin = storage_ptr + int(value.storage_offset()) * element_size
        shape = tuple(int(dim) for dim in value.shape)
        strides = tuple(int(stride) for stride in value.stride())
        if numel == 0:
            return TensorByteFootprint(
                device_key, origin, origin, origin, element_size, shape, strides, 0
            )
        low = 0
        high = 0
        # shape and strides both come from the same tensor: equal by rank.
        for size, stride in zip(shape, strides, strict=True):
            contribution = (size - 1) * stride
            if contribution < 0:
                low += contribution
            else:
                high += contribution
        return TensorByteFootprint(
            device_key,
            origin + low * element_size,
            origin + high * element_size + element_size,
            origin,
            element_size,
            shape,
            strides,
            numel,
        )
    except (RuntimeError, AttributeError, TypeError, ValueError, NotImplementedError):
        return None


def _footprint_is_dense_interval(footprint: TensorByteFootprint) -> bool:
    """Return whether a footprint's element starts cover ONE canonical dense byte interval (r39).

    Pure-integer proof (corr2_6): the touched element addresses form a contiguous no-hole,
    no-overlap grid -- so the WHOLE ``[start_byte, end_byte)`` byte span is fully covered -- iff,
    after dropping singleton dims and sorting the rest by absolute element stride, the smallest
    absolute stride is ``1`` and each next equals the running product of the preceding dimension
    sizes (then multiply by that size). This is the canonical row-major recurrence up to a
    dimension permutation and independent per-dim sign, so it proves contiguous, transposed/
    permuted-dense, and mathematically-valid negative-stride layouts alike -- and NOTHING else.

    It is deliberately NOT ``numel * element_size == end_byte - start_byte``: duplicate element
    addresses plus holes can satisfy that count/span equality without dense coverage. A zero
    stride on any non-singleton dim (an expanded view) repeats addresses -> not dense -> ``False``.
    Numel-independent: no enumeration, sound above the enumeration cap.
    """

    if footprint.numel == 0:
        return False
    dims = [
        (abs(stride), size)
        for size, stride in zip(footprint.shape, footprint.strides, strict=True)
        if size > 1
    ]
    if not dims:
        # All dims singleton: the footprint touches exactly one element -> a trivially dense
        # (single-element) interval of ``element_size`` bytes.
        return True
    if any(abs_stride == 0 for abs_stride, _size in dims):
        # An expanded (zero-stride) non-singleton dim repeats addresses -> not dense.
        return False
    dims.sort(key=lambda item: item[0])
    expected = 1
    for abs_stride, size in dims:
        if abs_stride != expected:
            return False
        expected *= size
    return True


def _footprint_stride_gcd(footprint: TensorByteFootprint) -> int:
    """gcd of nonzero element strides over nonsingleton dims (``0`` == one element)."""

    from math import gcd

    result = 0
    for size, stride in zip(footprint.shape, footprint.strides, strict=True):
        if size > 1 and stride != 0:
            result = gcd(result, abs(stride))
    return result


def footprint_touched_element_addresses(footprint: TensorByteFootprint) -> set[int]:
    """Enumerate the ABSOLUTE byte address of every element start a view touches.

    Pure Python integer arithmetic ONLY (r37 corr2-2): no torch factory may appear
    here, so the proof is identical under an implicit CPU default, a process-global
    meta default device, and nested ``torch.device(...)`` modes. Bounded by
    :data:`ALIAS_ENUMERATION_ELEMENT_CAP` at the call site.
    """

    if footprint.numel == 0:
        return set()
    esize = footprint.element_size
    addresses = {footprint.origin_byte}
    for size, stride in zip(footprint.shape, footprint.strides, strict=True):
        if size <= 1:
            continue
        step = stride * esize
        addresses = {address + index * step for address in addresses for index in range(size)}
    return addresses


def touched_bytes_relation(left: torch.Tensor, right: torch.Tensor) -> AliasRelation:
    """Three-valued exact touched-byte relation on absolute, device-scoped addresses.

    Proof layers, in order (INV-2): repeated object identity proves ``overlap``;
    unprovable footprints are ``unknown``; empty views, distinct device address
    spaces, and disjoint absolute byte intervals prove ``disjoint``; identical
    absolute geometry proves ``overlap``; an element-grid residue/GCD argument on
    absolute coordinates (equal element sizes, byte starts congruent on the shared
    element grid) proves ONLY disjointness; bounded pure-integer enumeration of
    absolute touched addresses proves either; everything else is ``unknown``. No
    bounding-interval overlap alone is an overlap proof, no complexity cap is a
    disjointness proof, and storage-pointer (in)equality NEVER decides anything --
    distinct storage objects can overlay one host allocation (hon1_1).
    """

    left_footprint = tensor_byte_footprint(left)
    right_footprint = left_footprint if left is right else tensor_byte_footprint(right)
    if left_footprint is None or right_footprint is None:
        return "unknown"
    if left_footprint.numel == 0 or right_footprint.numel == 0:
        return "disjoint"
    if left is right:
        return "overlap"
    if left_footprint.device_key != right_footprint.device_key:
        # Distinct device address spaces cannot share bytes. Same device TYPE with
        # one concrete and one None index is conservatively comparable only when
        # equal; treat a None-vs-concrete mismatch as unknown (unprovable).
        left_type, left_index = left_footprint.device_key
        right_type, right_index = right_footprint.device_key
        if left_type != right_type:
            return "disjoint"
        if left_index is None or right_index is None:
            return "unknown"
        return "disjoint"
    if (
        left_footprint.end_byte <= right_footprint.start_byte
        or right_footprint.end_byte <= left_footprint.start_byte
    ):
        return "disjoint"
    if (
        left_footprint.element_size == right_footprint.element_size
        and left_footprint.origin_byte == right_footprint.origin_byte
        and left_footprint.shape == right_footprint.shape
        and left_footprint.strides == right_footprint.strides
    ):
        return "overlap"
    if left_footprint.element_size == right_footprint.element_size:
        esize = left_footprint.element_size
        delta_bytes = left_footprint.origin_byte - right_footprint.origin_byte
        if delta_bytes % esize == 0:
            # Shared element grid: every touched element address of a view is
            # congruent to its origin modulo gcd(strides)*esize, so an origin-residue
            # disagreement modulo the combined gcd proves disjointness. A congruence
            # NEVER proves overlap.
            from math import gcd

            combined = gcd(
                _footprint_stride_gcd(left_footprint), _footprint_stride_gcd(right_footprint)
            )
            if combined == 0:
                # Both views touch exactly one element inside overlapping bounds.
                return (
                    "overlap"
                    if left_footprint.origin_byte == right_footprint.origin_byte
                    else "disjoint"
                )
            if (delta_bytes // esize) % combined != 0:
                return "disjoint"
    # r39 corr2_6 (the sole relaxation, sequenced after all fail-closed work): when BOTH
    # footprints are proven canonical dense byte intervals, each fully covers its own
    # ``[start_byte, end_byte)`` span, so their device-scoped byte intervals already passed the
    # disjointness check above => the overlapping region is touched by both => ``overlap``,
    # exactly and numel-independently (no enumeration, sound above the cap). This never converts
    # an ``unknown`` into a false ``overlap``: it fires ONLY on the provable dense geometry
    # (contiguous, permuted/transposed, signed-stride), keeping genuinely sparse/expanded
    # over-cap layouts ``unknown``. Element sizes need not match -- both byte intervals are
    # individually proved full.
    if _footprint_is_dense_interval(left_footprint) and _footprint_is_dense_interval(
        right_footprint
    ):
        return "overlap"
    if (
        left_footprint.numel <= ALIAS_ENUMERATION_ELEMENT_CAP
        and right_footprint.numel <= ALIAS_ENUMERATION_ELEMENT_CAP
    ):
        left_addresses = footprint_touched_element_addresses(left_footprint)
        right_addresses = footprint_touched_element_addresses(right_footprint)
        if (
            left_footprint.element_size == right_footprint.element_size
            and (left_footprint.origin_byte - right_footprint.origin_byte)
            % left_footprint.element_size
            == 0
        ):
            return "overlap" if left_addresses & right_addresses else "disjoint"
        left_bytes = {
            address + byte
            for address in left_addresses
            for byte in range(left_footprint.element_size)
        }
        right_bytes = {
            address + byte
            for address in right_addresses
            for byte in range(right_footprint.element_size)
        }
        return "overlap" if left_bytes & right_bytes else "disjoint"
    return "unknown"
