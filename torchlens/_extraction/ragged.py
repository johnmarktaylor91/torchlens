"""D-RAGGED: refuse by default; trimmed is the only ragged carrier (D4).

The entry gate is ``ragged="refuse"`` (an enum, not a bool). On first width
drift the run refuses BEFORE committing the offending shard, in one typed
message naming the remedies (``pool=``, fixed-length collation — both
signature-changing and disclosed as such — or the one-word
``ragged="trim"`` opt-in). Honest prose, measured: on real text the first
drift arrives at batch index 1 at every batch size, so the preserved prefix
is in practice ONE shard — the refusal must not oversell it.

Trimmed representation: packed values + per-row offsets + row shapes (the
Arrow-style triplet). Offsets are the STORED authority; ``(start, extent)``
is the write-time gather fact — left padding is why extent alone
mis-slices (measured starts ``[4, 0, 2]`` on real masks). Whenever the
collator supplies a mask, per-row ``(start, extent)`` plus the disclosed
``padding_side`` are ledgered REGARDLESS of regime, so a dense artifact
written from fixed-``max_length`` collation can be trimmed at read exactly.
A non-contiguous mask stores the full boolean mask and refuses trimming.

Readers expose exact ragged rows and an explicit
:func:`to_padded` returning values + mask — documented as VALUE-equivalent
within the cross-batch tolerance, NOT byte-identical to a historical padded
pipeline. They never silently pad, never emit object arrays, never fall
through to ``torch.cat``.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from .._errors import InvalidArgumentError

__tl_layer__ = "L5"

__all__ = [
    "RAGGED_MODES",
    "RaggedBatch",
    "mask_row_geometry",
    "raise_ragged_refusal",
    "to_padded",
    "trim_batch",
]

#: Closed ragged-mode vocabulary (D4). ``as_captured`` is the never-default
#: byte-exact reproduction mode (item 19, sequenced LAST; the enum makes it
#: additive).
RAGGED_MODES: tuple[str, ...] = ("refuse", "trim", "as_captured")


@dataclasses.dataclass(frozen=True)
class RaggedBatch:
    """One shard's trimmed carrier for one output key (the Arrow triplet).

    Attributes
    ----------
    values:
        Packed per-row slices concatenated along axis 0:
        ``[sum(extents), *feature_shape]``.
    offsets:
        ``int64`` tensor of length ``rows + 1``; row ``i`` occupies
        ``values[offsets[i]:offsets[i+1]]``. The STORED authority.
    row_shapes:
        Per-row logical shapes (each row's true ``[extent, *feature]``).
    """

    values: torch.Tensor
    offsets: torch.Tensor
    row_shapes: tuple[tuple[int, ...], ...]

    @property
    def row_count(self) -> int:
        """Number of logical rows in the carrier.

        Returns
        -------
        int
            ``len(offsets) - 1``.
        """

        return int(self.offsets.shape[0]) - 1

    def row(self, index: int) -> torch.Tensor:
        """Return one exact ragged row.

        Parameters
        ----------
        index:
            Row index within this carrier.

        Returns
        -------
        torch.Tensor
            The row at its true extent.
        """

        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return self.values[start:stop]


def validate_ragged_mode(ragged: Any) -> str:
    """Validate the ``ragged=`` kwarg against the closed vocabulary.

    Parameters
    ----------
    ragged:
        Requested mode.

    Returns
    -------
    str
        The validated mode.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_ragged_invalid`` outside the closed vocabulary (a bool
        gets a teaching message — the gate is an enum by design).
    """

    if ragged in RAGGED_MODES:
        return str(ragged)
    raise InvalidArgumentError(
        f"ragged= value {ragged!r} is not in the closed vocabulary "
        f"{RAGGED_MODES} (an enum, not a bool: 'refuse' is the default, "
        "'trim' stores true per-stimulus extents, 'as_captured' keeps "
        "batch-padded bytes for byte-exact reproduction and is never the "
        "default).",
        code="extraction_ragged_invalid",
        remedy=f"pass one of {RAGGED_MODES}",
        value=repr(ragged),
    )


def mask_row_geometry(mask: torch.Tensor) -> dict[str, Any]:
    """Derive per-row gather facts from a two-dimensional attention mask.

    Parameters
    ----------
    mask:
        ``[rows, width]`` attention mask (nonzero = valid).

    Returns
    -------
    dict[str, Any]
        ``starts`` (per-row first valid index), ``extents`` (per-row valid
        count), ``contiguous`` (whether EVERY row's valid run is one
        contiguous span), and ``padding_side`` (``"right"`` / ``"left"`` /
        ``"mixed"`` / ``"none"`` — the disclosed pad geometry).
    """

    valid = mask != 0
    rows, width = valid.shape
    extents = valid.sum(dim=1).tolist()
    first = torch.argmax(valid.long(), dim=1)
    starts = [int(first[i]) if extents[i] else 0 for i in range(rows)]
    contiguous = True
    for i in range(rows):
        if (
            extents[i]
            and int(valid[i, starts[i] : starts[i] + int(extents[i])].sum()) != extents[i]
        ):
            contiguous = False
            break
    any_left = any(starts[i] > 0 and extents[i] for i in range(rows))
    any_right = any(starts[i] + extents[i] < width for i in range(rows))
    if not any_left and not any_right:
        padding_side = "none"
    elif any_left and not any_right:
        padding_side = "left"
    elif any_right and not any_left:
        padding_side = "right"
    else:
        padding_side = "mixed"
    return {
        "starts": [int(s) for s in starts],
        "extents": [int(e) for e in extents],
        "contiguous": contiguous,
        "padding_side": padding_side,
    }


def raise_ragged_refusal(
    key: str,
    batch_index: int,
    planned_shape: list[Any],
    observed_shape: list[int],
    *,
    late_mask_shaped: bool = False,
) -> None:
    """Raise the D4 entry-gate refusal on first width drift.

    Fired BEFORE the offending shard commits. Honest about the preserved
    prefix: on real text the first drift arrives at batch index 1, so the
    prefix is in practice the already-committed shards only.

    Parameters
    ----------
    key:
        Output key whose width drifted.
    batch_index:
        Zero-based index of the offending batch.
    planned_shape:
        Frozen per-stimulus shape from batch zero.
    observed_shape:
        This batch's observed per-stimulus shape.
    late_mask_shaped:
        Whether the key was frozen DENSE at batch zero (no mask, or an
        all-equal-width batch) under ``ragged="trim"`` and only now proves
        ragged. A key's layout is frozen once for the whole artifact, so a
        late trimmed admission would mix dense and trimmed shards under a
        manifest-dense key and readers would silently drop rows; the
        refusal teaches the batch-zero requirement instead.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_ragged_refused``, always.
    """

    if late_mask_shaped:
        problem = (
            f"Output key {key!r} changed per-stimulus shape at batch "
            f"{batch_index}: batch zero froze {planned_shape} as a DENSE "
            f"layout (no attention mask reached the model, or every row had "
            f"the same width), this batch produced {observed_shape} with a "
            "mask. ragged='trim' stores a key TRIMMED only when batch zero "
            "is mask-shaped: the artifact's per-key layout is frozen once, "
            "and admitting a trimmed shard under a manifest-dense key would "
            "make every reader drop rows silently. Refused BEFORE the "
            "offending shard commits; only already-committed shards are "
            "preserved."
        )
        remedy = (
            "supply the attention mask on EVERY batch (including all-equal-"
            "width ones) so batch zero is mask-shaped, or pad/collate every "
            "batch to one fixed width, or pool the site (pool=)"
        )
    else:
        problem = (
            f"Output key {key!r} changed per-stimulus shape at batch "
            f"{batch_index}: batch zero froze {planned_shape}, this batch "
            f"produced {observed_shape}. Variable-width outputs are refused "
            "by default (ragged='refuse') BEFORE the offending shard commits; "
            "only already-committed shards are preserved — on real text the "
            "first drift typically arrives at batch 1, so expect the preserved "
            "prefix to be a single shard."
        )
        remedy = (
            "pool the site (pool=), collate to a fixed length (both change "
            "the signature and are disclosed as such), or opt into "
            "ragged='trim' to store true per-stimulus extents"
        )
    raise InvalidArgumentError(
        problem,
        code="extraction_ragged_refused",
        remedy=remedy,
        key=key,
        batch_index=batch_index,
        planned_shape=planned_shape,
        observed_shape=observed_shape,
        late_mask_shaped=late_mask_shaped,
    )


def trim_batch(
    key: str,
    tensor: torch.Tensor,
    geometry: dict[str, Any],
) -> RaggedBatch:
    """Trim one batch tensor to its true per-row extents (the D4 carrier).

    Slicing uses ``(start, extent)`` — never extent alone: left padding
    shifts each row's first valid index (measured starts ``[4, 0, 2]`` on
    real BERT/GPT-2 masks), so extent-only slicing reads pad values.

    Parameters
    ----------
    key:
        Output key (refusal text).
    tensor:
        ``[rows, width, *feature]`` batch tensor.
    geometry:
        The batch's :func:`mask_row_geometry` facts.

    Returns
    -------
    RaggedBatch
        Packed values + offsets + per-row shapes.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_ragged_mask_noncontiguous`` when any row's valid run
        is not one contiguous span (the full boolean mask is the only
        honest carrier there, and trimming refuses);
        ``extraction_ragged_geometry_mismatch`` when the tensor's width
        axis cannot carry the mask geometry.
    """

    if not geometry["contiguous"]:
        raise InvalidArgumentError(
            f"Output key {key!r} cannot be trimmed: at least one mask row's "
            "valid positions are not one contiguous span, so (start, "
            "extent) slicing would mis-read interior gaps.",
            code="extraction_ragged_mask_noncontiguous",
            remedy=(
                "store dense with the full mask (ragged='as_captured' once "
                "available, or pool= the site), or fix the collator's mask"
            ),
            key=key,
        )
    starts = geometry["starts"]
    extents = geometry["extents"]
    if (
        tensor.ndim < 2
        or tensor.shape[0] != len(starts)
        or tensor.shape[1] < max((starts[i] + extents[i] for i in range(len(starts))), default=0)
    ):
        raise InvalidArgumentError(
            f"Output key {key!r} with shape {tuple(tensor.shape)} cannot "
            f"carry the batch mask geometry (rows={len(starts)}, max end="
            f"{max((starts[i] + extents[i] for i in range(len(starts))), default=0)}); "
            "the site's second axis is not the masked token axis.",
            code="extraction_ragged_geometry_mismatch",
            remedy=(
                "trim only token-axis sites (pool or transform the others), "
                "or extract this key densely with pool=/fixed-length collation"
            ),
            key=key,
            shape=list(tensor.shape),
        )
    pieces = [tensor[i, starts[i] : starts[i] + extents[i]] for i in range(len(starts))]
    values = (
        torch.cat(pieces, dim=0) if pieces else tensor.new_zeros((0,) + tuple(tensor.shape[2:]))
    )
    offsets = torch.zeros(len(pieces) + 1, dtype=torch.int64)
    total = 0
    for i, piece in enumerate(pieces):
        total += int(piece.shape[0])
        offsets[i + 1] = total
    row_shapes = tuple(tuple(piece.shape) for piece in pieces)
    return RaggedBatch(values=values, offsets=offsets, row_shapes=row_shapes)


def to_padded(
    batches: list[RaggedBatch],
    *,
    pad_value: float = 0.0,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize trimmed rows as one padded tensor plus its mask (D4).

    Documented as VALUE-equivalent within the cross-batch tolerance, NOT
    byte-identical to a historical padded pipeline (padded positions here
    are ``pad_value``, not the model's pad-position activations).

    Parameters
    ----------
    batches:
        Trimmed carriers in row order.
    pad_value:
        Fill value for padded positions.
    max_len:
        Optional fixed width; rows longer than it refuse (truncation would
        silently drop values).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(values, mask)``: right-padded ``[rows, width, *feature]`` values
        and the ``[rows, width]`` boolean validity mask.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_ragged_geometry_mismatch`` when ``max_len`` is shorter
        than a stored row.
    """

    rows: list[torch.Tensor] = []
    for batch in batches:
        rows.extend(batch.row(i) for i in range(batch.row_count))
    widths = [int(row.shape[0]) for row in rows]
    width = max(widths, default=0)
    if max_len is not None:
        if widths and max(widths) > max_len:
            raise InvalidArgumentError(
                f"to_padded(max_len={max_len}) is shorter than the longest "
                f"stored row ({max(widths)}); truncation would silently "
                "drop values.",
                code="extraction_ragged_geometry_mismatch",
                remedy="raise max_len to at least the longest row, or omit it",
                max_len=max_len,
                longest_row=max(widths),
            )
        width = max_len
    if not rows:
        return torch.zeros(0, width), torch.zeros(0, width, dtype=torch.bool)
    feature = tuple(rows[0].shape[1:])
    values = rows[0].new_full((len(rows), width) + feature, pad_value)
    mask = torch.zeros(len(rows), width, dtype=torch.bool)
    for i, row in enumerate(rows):
        values[i, : row.shape[0]] = row
        mask[i, : row.shape[0]] = True
    return values, mask
