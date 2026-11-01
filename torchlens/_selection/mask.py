"""Canonical internal mask algebra for :mod:`torchlens._selection.selection`.

One site's selected-element set is a :class:`_Mask` over the site's output
index space, in one of four forms (whole, empty, slices, dense). The
``_mask_*`` constructors and set operators normalize to the cheapest form and
are exact as sets. Masks are internal: any user-facing tensor access returns
a fresh materialization. The names callers use are re-exported from
``torchlens._selection.selection``, the home its callers import from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Canonical internal mask representation. NEVER handed out: any user-facing
# tensor access returns a FRESH materialization (mask immutability pin).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Mask:
    """Internal canonical mask over one site's output index space."""

    shape: tuple[int, ...]
    form: str  # "whole" | "empty" | "slices" | "dense"
    slice_bounds: tuple[tuple[int, int, int], ...] | None = None
    dense: torch.Tensor | None = field(default=None, compare=False)

    def numel(self) -> int:
        """Return the site's index-space size."""

        total = 1
        for extent in self.shape:
            total *= extent
        return total

    def to_dense(self) -> torch.Tensor:
        """Return a FRESH dense bool tensor (safe to hand out)."""

        if self.form == "whole":
            return torch.ones(self.shape, dtype=torch.bool)
        if self.form == "empty":
            return torch.zeros(self.shape, dtype=torch.bool)
        if self.form == "slices":
            if self.slice_bounds is None:
                raise RuntimeError("slices-form mask lost its slice_bounds")
            mask = torch.zeros(self.shape, dtype=torch.bool)
            mask[tuple(slice(*bounds) for bounds in self.slice_bounds)] = True
            return mask
        if self.dense is None:
            raise RuntimeError("dense-form mask lost its dense tensor")
        return self.dense.clone()

    def _dense_ro(self) -> torch.Tensor:
        """Return the dense form for internal composition (never handed out)."""

        if self.form == "dense":
            if self.dense is None:
                raise RuntimeError("dense-form mask lost its dense tensor")
            return self.dense
        return self.to_dense()

    def count(self) -> int:
        """Return the number of selected elements."""

        if self.form == "whole":
            return self.numel()
        if self.form == "empty":
            return 0
        if self.form == "slices":
            if self.slice_bounds is None:
                raise RuntimeError("slices-form mask lost its slice_bounds")
            total = 1
            for start, stop, step in self.slice_bounds:
                total *= max(0, (stop - start + step - 1) // step)
            return total
        if self.dense is None:
            raise RuntimeError("dense-form mask lost its dense tensor")
        return int(self.dense.sum().item())

    def canonical_bytes(self) -> bytes:
        """Return canonical identity bytes (dense packing, form-independent)."""

        return self._dense_ro().to(torch.uint8).numpy().tobytes()


def _mask_whole(shape: tuple[int, ...]) -> _Mask:
    """Build the whole-form mask (every element of ``shape`` selected)."""

    return _Mask(shape=shape, form="whole")


def _mask_empty(shape: tuple[int, ...]) -> _Mask:
    """Build the empty-form mask (no element of ``shape`` selected)."""

    return _Mask(shape=shape, form="empty")


def _mask_from_dense(shape: tuple[int, ...], dense: torch.Tensor) -> _Mask:
    """Build a canonical mask from a dense bool tensor (normalizing extremes)."""

    selected = int(dense.sum().item())
    if selected == 0:
        return _mask_empty(shape)
    if selected == dense.numel():
        return _mask_whole(shape)
    return _Mask(shape=shape, form="dense", dense=dense.clone())


def _mask_from_slices(shape: tuple[int, ...], slices: tuple[slice, ...]) -> _Mask:
    """Build a canonical mask from a slice tuple over ``shape``."""

    bounds: list[tuple[int, int, int]] = []
    for one_slice, extent in zip(slices, shape, strict=True):
        start, stop, step = one_slice.indices(extent)
        if stop <= start:
            return _mask_empty(shape)
        bounds.append((start, stop, step))
    if all(
        start == 0 and stop == extent and step == 1
        for (start, stop, step), extent in zip(bounds, shape, strict=True)
    ):
        return _mask_whole(shape)
    return _Mask(shape=shape, form="slices", slice_bounds=tuple(bounds))


def _mask_union(a: _Mask, b: _Mask) -> _Mask:
    """Union two same-shape masks (whole/empty fast paths, else dense OR)."""

    if a.form == "whole" or b.form == "empty":
        return a if a.form == "whole" else (a if b.form == "empty" else b)
    if b.form == "whole" or a.form == "empty":
        return b if b.form == "whole" else (b if a.form == "empty" else a)
    return _mask_from_dense(a.shape, a._dense_ro() | b._dense_ro())


def _mask_intersect(a: _Mask, b: _Mask) -> _Mask:
    """Intersect two same-shape masks (whole/empty fast paths, else dense AND)."""

    if a.form == "empty" or b.form == "whole":
        return a
    if b.form == "empty" or a.form == "whole":
        return b
    return _mask_from_dense(a.shape, a._dense_ro() & b._dense_ro())


def _mask_difference(a: _Mask, b: _Mask) -> _Mask:
    """Subtract ``b`` from ``a`` (set difference on same-shape masks)."""

    if a.form == "empty" or b.form == "empty":
        return a
    if b.form == "whole":
        return _mask_empty(a.shape)
    return _mask_from_dense(a.shape, a._dense_ro() & ~b._dense_ro())


def _mask_complement(a: _Mask) -> _Mask:
    """Complement a mask within its own index space."""

    if a.form == "whole":
        return _mask_empty(a.shape)
    if a.form == "empty":
        return _mask_whole(a.shape)
    return _mask_from_dense(a.shape, ~a._dense_ro())


def _mask_equal(a: _Mask, b: _Mask) -> bool:
    """Compare two masks as element sets (form-independent equality)."""

    if a.shape != b.shape:
        return False
    if a.form == b.form == "whole" or a.form == b.form == "empty":
        return True
    return bool(torch.equal(a._dense_ro(), b._dense_ro()))
