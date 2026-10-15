"""Quantile and top-k accumulators, pooled or per unit (DOCUMENTED-UNSTABLE).

``Quantile`` and ``TopK`` pool every value of a batch by default. With
``unit_axis=`` they keep one reservoir, or one top-k list, per unit (a
neuron, a channel, an SAE latent), and per-unit ``TopK`` records which
example, and which position, each retained value came from. Neurons and SAE
latents share this one top-k-over-a-dataset tool: ``encoder=`` maps each
batch (for example through an SAE's encoder) before it is folded.

Batch layout: axis 0 is the example axis, ``unit_axis`` names the unit axis,
and every other axis is a position axis (a token position, a spatial
location), kept in order.

Per-unit state is plain tensors with a fixed layout:

- per-unit ``Quantile``: ``_unit_reservoir`` ``float64 [units,
  reservoir_size]`` (the first ``min(_seen, reservoir_size)`` columns are
  filled), ``_seen`` (values folded per unit) and ``_generator`` (a private
  ``torch.Generator``; ``get_state()`` is a ``uint8`` tensor);
- per-unit ``TopK``: ``_values`` ``float64 [units, k]`` (descending per unit,
  NaN in empty slots), ``_example_ids`` ``int64 [units, k]`` and
  ``_positions`` ``int64 [units, k, position_axes]`` (``-1`` in empty slots),
  and ``_examples_seen`` (the next default example id).
"""

from __future__ import annotations

import dataclasses
import heapq
import math
import random
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import torch

from .._errors import ArgumentTypeError, InvalidArgumentError
from ..errors import TorchLensWarning
from ._streaming import _as_float_tensor

__tl_layer__ = "L5"

#: Per-unit state above this many bytes warns once (the per-unit size guard).
STATE_WARN_BYTES = 256 * 1024**2

#: Element budget for one transient per-unit fold buffer (units are chunked).
_FOLD_CHUNK_ELEMENTS = 2**22

#: ``torch.quantile`` refuses inputs above this many elements.
_QUANTILE_INPUT_LIMIT = 2**24

_WARNED_CODES: set[str] = set()


@dataclasses.dataclass(frozen=True, repr=False)
class UnitTopK:
    """Per-unit top-k values with the example and position each came from.

    Attributes
    ----------
    values:
        ``float64 [units, k]``: each unit's largest values, descending. A unit
        that saw fewer than ``k`` non-NaN values has NaN in its empty slots.
    example_ids:
        ``int64 [units, k]``: the example each value came from (its index in
        the stream of examples, unless ``update(example_ids=...)`` named
        them); ``-1`` in empty slots.
    positions:
        ``int64 [units, k, position_axes]``: the index along each position
        axis (the axes other than the example and unit axes, in order);
        ``position_axes`` is 0 when none remain; ``-1`` in empty slots.
    unit_axis:
        The unit axis the accumulator was built with.
    """

    values: torch.Tensor
    example_ids: torch.Tensor
    positions: torch.Tensor
    unit_axis: int

    def __repr__(self) -> str:
        """Summarize the shape instead of printing every tensor."""

        units, k = (int(size) for size in self.values.shape[:2])
        return (
            f"UnitTopK(units={units}, k={k}, position_axes={int(self.positions.shape[-1])}, "
            f"unit_axis={self.unit_axis})"
        )


def _validate_unit_axis(unit_axis: Any, owner: str) -> int | None:
    """Return ``unit_axis`` after a type check, refusing non-integers."""

    if unit_axis is None:
        return None
    if isinstance(unit_axis, bool) or not isinstance(unit_axis, int):
        raise ArgumentTypeError(
            f"{owner}(unit_axis={unit_axis!r}) needs an integer axis",
            code="stats_unit_axis_type",
            remedy=f"name the unit axis by position, for example {owner}(unit_axis=-1)",
            argument="unit_axis",
        )
    return unit_axis


def _validate_encoder(encoder: Any, owner: str) -> Callable[[Any], Any] | None:
    """Return ``encoder`` after a callability check."""

    if encoder is not None and not callable(encoder):
        raise ArgumentTypeError(
            f"{owner}(encoder={type(encoder).__name__}) is not callable",
            code="stats_encoder_type",
            remedy=f"pass a callable such as {owner}(encoder=sae.encode), or omit encoder=",
            argument="encoder",
        )
    return encoder


def _encode(encoder: Callable[[Any], Any] | None, value: Any) -> Any:
    """Apply ``encoder`` to one batch without building an autograd graph."""

    if encoder is None:
        return value
    with torch.no_grad():
        return encoder(value)


def _resolve_axis(unit_axis: int, ndim: int, owner: str) -> int:
    """Normalize ``unit_axis`` against one batch, refusing the example axis."""

    axis = unit_axis + ndim if unit_axis < 0 else unit_axis
    if ndim < 2 or not 0 <= axis < ndim:
        raise InvalidArgumentError(
            f"{owner}(unit_axis={unit_axis}) is outside a {ndim}-d batch",
            code="stats_unit_axis_out_of_range",
            remedy=(
                "pass batches shaped [examples, ..., units] and an axis inside them, "
                f"for example {owner}(unit_axis=-1)"
            ),
            unit_axis=unit_axis,
            ndim=ndim,
        )
    if axis == 0:
        raise InvalidArgumentError(
            f"{owner}(unit_axis={unit_axis}) names axis 0, the example axis",
            code="stats_unit_axis_is_batch_axis",
            remedy=f"name the unit axis among the other axes, for example {owner}(unit_axis=-1)",
            unit_axis=unit_axis,
        )
    return axis


def _format_bytes(num_bytes: float) -> str:
    """Render a byte count in binary units."""

    for unit, size in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if num_bytes >= size:
            return f"{num_bytes / size:.1f} {unit}"
    return f"{int(num_bytes)} B"


def _warn_state_size(
    label: str, units: int, bytes_per_unit: int, unit_axis: int, cheaper: str
) -> None:
    """Warn once per session when per-unit state passes ``STATE_WARN_BYTES``."""

    code = "stats_per_unit_state_large"
    total = units * bytes_per_unit
    if total <= STATE_WARN_BYTES or code in _WARNED_CODES:
        return
    _WARNED_CODES.add(code)
    fits = max(1, STATE_WARN_BYTES // max(bytes_per_unit, 1))
    warnings.warn(
        TorchLensWarning(
            f"{label} keeps per-unit state for {units} units x {bytes_per_unit} bytes = "
            f"{_format_bytes(total)}, above the {_format_bytes(STATE_WARN_BYTES)} guard; "
            "every unit is kept, nothing is dropped. This warning fires once per session; "
            "see docs/reference/stats.md, Per-unit statistics. "
            f"Remedy: {cheaper}, or keep fewer units with "
            f"encoder=lambda v: v.narrow({unit_axis}, 0, {fits})",
            code=code,
        ),
        stacklevel=3,
    )


def _unit_rows(tensor: torch.Tensor, axis: int) -> tuple[torch.Tensor, int, tuple[int, ...]]:
    """Return ``float64 [units, rows]`` plus the example count and position shape."""

    moved = tensor.movedim(axis, -1)
    position_shape = tuple(int(size) for size in moved.shape[1:-1])
    rows = moved.reshape(-1, moved.shape[-1]).transpose(0, 1).contiguous()
    return rows, int(tensor.shape[0]), position_shape


def _as_batch_tensor(value: Any) -> torch.Tensor:
    """Return one batch as a detached CPU ``float64`` tensor, shape kept."""

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.detach().to(device="cpu", dtype=torch.float64)


class _UnitLayout:
    """The unit count and axis count a per-unit accumulator fixes at its first batch."""

    def __init__(self, owner: str, unit_axis: int) -> None:
        """Record the owner label and the requested unit axis."""

        self.owner = owner
        self.unit_axis = unit_axis
        self.units: int | None = None
        self.ndim: int | None = None

    def check(self, tensor: torch.Tensor) -> int:
        """Return the resolved axis, refusing a batch whose layout changed."""

        axis = _resolve_axis(self.unit_axis, tensor.ndim, self.owner)
        units = int(tensor.shape[axis])
        if self.ndim is not None and tensor.ndim != self.ndim:
            raise InvalidArgumentError(
                f"{self.owner} got a {tensor.ndim}-d batch after {self.ndim}-d batches",
                code="stats_unit_layout_changed",
                remedy="feed batches with the same axes to one accumulator; build one per layout",
                ndim=tensor.ndim,
                expected_ndim=self.ndim,
            )
        if self.units is not None and units != self.units:
            raise InvalidArgumentError(
                f"{self.owner} got {units} units along axis {self.unit_axis} after {self.units}",
                code="stats_unit_count_changed",
                remedy="feed every batch with the same unit count; build one accumulator per site",
                units=units,
                expected_units=self.units,
            )
        self.units, self.ndim = units, tensor.ndim
        return axis


class Quantile:
    """Reservoir-sampling running quantile estimator, pooled or per unit.

    The estimate is exact while at most ``reservoir_size`` values have been
    seen; beyond that it is a uniform subsample, whose quantile standard
    error is ~``sqrt(q * (1 - q) / reservoir_size)`` in RANK space (about
    +/-0.55 percentile points at the median with the default reservoir).

    Sampling uses a PRIVATE stream (``random.Random(seed)`` pooled, a
    ``torch.Generator`` seeded with ``seed`` per unit): it never draws from
    (or perturbs) the process-global engines, which capture replay snapshots
    and restores as protected engines, and two runs with the same seed and
    the same update stream produce identical estimates.

    With ``unit_axis=`` each unit keeps its own reservoir and ``result()``
    maps each quantile to a ``float64 [units]`` tensor. Every unit sees the
    same number of values, so one slot draw per value serves all units: each
    unit's reservoir is a uniform subsample of its own values, with the same
    rank-space error as above. NaN values are kept, as in the pooled
    estimator, so a unit whose reservoir holds a NaN reports NaN. State is
    ``8 * reservoir_size`` bytes per unit (64 KiB at the default 8192), and
    a warning fires once per session when the total passes 256 MiB.
    """

    def __init__(  # noqa: PLR0913 -- the pooled knobs plus the keyword-only per-unit pair
        self,
        quantiles: Iterable[float] = (0.5, 0.95, 0.99),
        name: str | None = None,
        reservoir_size: int = 8192,
        seed: int | None = 0,
        *,
        unit_axis: int | None = None,
        encoder: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize the estimator.

        Parameters
        ----------
        quantiles:
            Quantiles in ``[0, 1]`` to estimate.
        name:
            Optional metric name.
        reservoir_size:
            Maximum sampled values retained in memory (per unit with
            ``unit_axis=``).
        seed:
            Seed for the private sampling stream. The default ``0`` makes
            estimates reproducible run-to-run; pass ``None`` for a
            fresh-entropy stream (still decoupled from the global engine).
        unit_axis:
            Axis of each batch that indexes units (axis 0 is the example
            axis); ``None`` pools every value.
        encoder:
            Optional callable applied to each batch before it is folded, for
            example an SAE's encoder to estimate per-latent quantiles.
        """

        self.name = name
        self.quantiles = tuple(float(q) for q in quantiles)
        self.reservoir_size = int(reservoir_size)
        self.seed = seed
        self.unit_axis = _validate_unit_axis(unit_axis, "tl.stats.Quantile")
        self.encoder = _validate_encoder(encoder, "tl.stats.Quantile")
        self._rng = random.Random(seed)
        self._seen = 0
        self._reservoir: list[float] = []
        self._layout: _UnitLayout | None = None
        self._unit_reservoir: torch.Tensor | None = None
        self._generator: torch.Generator | None = None
        if self.unit_axis is not None:
            self._layout = _UnitLayout("tl.stats.Quantile", self.unit_axis)
            self._generator = torch.Generator()
            if seed is None:
                self._generator.seed()
            else:
                self._generator.manual_seed(int(seed))

    def update(self, value: Any) -> None:
        """Update the reservoir.

        Parameters
        ----------
        value:
            Tensor-like batch value.
        """

        value = _encode(self.encoder, value)
        if self._layout is not None and self._generator is not None:
            self._update_per_unit(_as_batch_tensor(value), self._layout, self._generator)
            return
        items = _as_float_tensor(value).tolist()
        start = 0
        free = self.reservoir_size - len(self._reservoir)
        if free > 0:
            # Fill phase: the first reservoir_size values are all retained,
            # so bulk-extend instead of looping (algorithm-R equivalent).
            start = min(free, len(items))
            self._reservoir.extend(float(item) for item in items[:start])
            self._seen += start
        for item in items[start:]:
            self._seen += 1
            replacement = self._rng.randint(0, self._seen - 1)
            if replacement < self.reservoir_size:
                self._reservoir[replacement] = float(item)

    def _update_per_unit(
        self, tensor: torch.Tensor, layout: _UnitLayout, generator: torch.Generator
    ) -> None:
        """Fold one batch into the per-unit reservoirs (algorithm R, shared slots)."""

        axis = layout.check(tensor)
        rows, _examples, _positions = _unit_rows(tensor, axis)
        units, count = int(rows.shape[0]), int(rows.shape[1])
        size = max(self.reservoir_size, 0)
        if self._unit_reservoir is None:
            _warn_state_size(
                self.name or "tl.stats.Quantile",
                units,
                8 * size,
                layout.unit_axis,
                f"tl.stats.Quantile(unit_axis={layout.unit_axis}, reservoir_size={max(size // 8, 1)})",
            )
            self._unit_reservoir = torch.empty((units, size), dtype=torch.float64)
        reservoir = self._unit_reservoir
        filled = min(self._seen, size)
        take = min(size - filled, count)
        if take > 0:
            reservoir[:, filled : filled + take] = rows[:, :take]
            self._seen += take
        rest = count - take
        if rest <= 0:
            return
        totals = torch.arange(self._seen + 1, self._seen + rest + 1, dtype=torch.float64)
        draws = torch.rand(rest, generator=generator, dtype=torch.float64)
        slots = torch.minimum((draws * totals).floor().long(), totals.long() - 1)
        self._seen += rest
        keep = slots < size
        kept_slots = slots[keep]
        if kept_slots.numel() == 0:
            return
        kept_columns = (torch.arange(rest) + take)[keep]
        # Algorithm R applies replacements in order, so the last draw that
        # lands on a slot wins; resolve that before one vectorized write.
        unique_slots, inverse = torch.unique(kept_slots, return_inverse=True)
        last = torch.full((unique_slots.numel(),), -1, dtype=torch.long)
        last = last.scatter_reduce(0, inverse, torch.arange(kept_slots.numel()), reduce="amax")
        reservoir[:, unique_slots] = rows[:, kept_columns[last]]

    def result(self) -> dict[float, float] | dict[float, torch.Tensor]:
        """Return finalized quantile estimates.

        Returns
        -------
        dict[float, float] | dict[float, torch.Tensor]
            Mapping from requested quantile to estimated value; per unit, to
            a ``float64 [units]`` tensor (empty before the first batch).
        """

        if self._layout is not None:
            return self._result_per_unit()
        if not self._reservoir:
            return dict.fromkeys(self.quantiles, math.nan)
        tensor = torch.tensor(self._reservoir, dtype=torch.float64)
        return {q: float(torch.quantile(tensor, q).item()) for q in self.quantiles}

    def _result_per_unit(self) -> dict[float, torch.Tensor]:
        """Return per-unit quantile vectors from the reservoirs."""

        if self._unit_reservoir is None:
            return {q: torch.empty(0, dtype=torch.float64) for q in self.quantiles}
        units = int(self._unit_reservoir.shape[0])
        filled = min(self._seen, int(self._unit_reservoir.shape[1]))
        if filled == 0 or not self.quantiles:
            return {q: torch.full((units,), math.nan, dtype=torch.float64) for q in self.quantiles}
        data = self._unit_reservoir[:, :filled]
        levels = torch.tensor(self.quantiles, dtype=torch.float64)
        chunk = max(1, _QUANTILE_INPUT_LIMIT // filled)
        estimates = torch.cat(
            [
                torch.quantile(data[start : start + chunk], levels, dim=1)
                for start in range(0, units, chunk)
            ],
            dim=1,
        )
        return {q: estimates[index].clone() for index, q in enumerate(self.quantiles)}


def _first_k_rows(values: torch.Tensor, k: int) -> torch.Tensor:
    """Return, per unit, the row indices of its ``k`` largest non-NaN values.

    Rows must be in (example, position) order: among equal values the
    earliest rows win. Returns ``int64 [units, min(k, rows)]`` in row order,
    ``-1`` where a unit has fewer non-NaN values.
    """

    units, count = int(values.shape[0]), int(values.shape[1])
    width = min(k, count)
    filled = torch.where(values.isnan(), -math.inf, values)
    threshold = torch.topk(filled, width, dim=1).values[:, -1:]
    above = values > threshold
    level = values == threshold
    quota = width - above.sum(dim=1, keepdim=True)
    selected = above | (level & (level.cumsum(dim=1) <= quota))
    slots = torch.where(selected, selected.cumsum(dim=1) - 1, width)
    rows = torch.full((units, width + 1), -1, dtype=torch.long)
    rows.scatter_(1, slots, torch.arange(count).expand(units, count))
    return rows[:, :width]


def _take(source: torch.Tensor, index: torch.Tensor, fill: float | int) -> torch.Tensor:
    """Gather ``source`` rows (shape ``[rows, ...]``) at ``index``, ``fill`` where ``-1``."""

    gathered = source[index.clamp(min=0)]
    mask = (index < 0).reshape(*index.shape, *([1] * (gathered.ndim - index.ndim)))
    return torch.where(mask, torch.as_tensor(fill, dtype=gathered.dtype), gathered)


def _rank_entries(
    values: torch.Tensor, example_ids: torch.Tensor, positions: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep each unit's ``k`` best entries: value descending, then example, then position."""

    valid = example_ids >= 0
    keyed = torch.where(valid, values, -math.inf)
    order = torch.arange(values.shape[1]).expand_as(values)
    keys = [positions[..., axis] for axis in reversed(range(positions.shape[-1]))]
    for key, descending in [*((key, False) for key in keys), (example_ids, False), (keyed, True)]:
        current = torch.gather(key, 1, order)
        order = torch.gather(
            order, 1, torch.sort(current, dim=1, descending=descending, stable=True).indices
        )
    current = torch.gather((~valid).to(torch.int8), 1, order)
    order = torch.gather(order, 1, torch.sort(current, dim=1, stable=True).indices)[:, :k]
    kept_valid = torch.gather(valid, 1, order)
    kept_values = torch.where(kept_valid, torch.gather(values, 1, order), math.nan)
    kept_ids = torch.where(kept_valid, torch.gather(example_ids, 1, order), -1)
    index = order.unsqueeze(-1).expand(*order.shape, positions.shape[-1])
    kept_positions = torch.where(kept_valid.unsqueeze(-1), torch.gather(positions, 1, index), -1)
    return kept_values, kept_ids, kept_positions


def _row_ids(
    example_ids: torch.Tensor, position_shape: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the example id and position of every flattened row."""

    per_example = math.prod(position_shape)
    row_examples = example_ids.repeat_interleave(per_example)
    if position_shape:
        grid = torch.meshgrid(*(torch.arange(size) for size in position_shape), indexing="ij")
        offsets = torch.stack([axis.reshape(-1) for axis in grid], dim=-1)
    else:
        offsets = torch.zeros((1, 0), dtype=torch.long)
    return row_examples, offsets.repeat(int(example_ids.numel()), 1)


class TopK:
    """Streaming top-k value tracker, pooled or per unit with example ids.

    Pooled (the default), ``result()`` is the ``k`` largest scalars seen.
    With ``unit_axis=``, ``result()`` is a :class:`UnitTopK`: per unit, its
    ``k`` largest values with the example id and position each came from.
    The example id defaults to the example's index in the update stream
    (under ``tl.aggregate``, its index in the dataloader's iteration order;
    the dataset index for an unshuffled loader); ``update(example_ids=...)``
    names them instead. Ties break toward the lowest example id, then the
    lowest position, so results do not depend on batch boundaries. NaN
    values are ignored in both modes.

    Per-unit state is ``k * (16 + 8 * position_axes)`` bytes per unit (160 B
    at the default ``k=10`` with no position axis, 240 B with one); each fold
    also allocates transient buffers in unit chunks of at most about four
    million elements. A warning fires once per session when the state total
    passes 256 MiB.
    """

    def __init__(
        self,
        k: int = 10,
        name: str | None = None,
        *,
        unit_axis: int | None = None,
        encoder: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize the tracker.

        Parameters
        ----------
        k:
            Number of largest values to retain (per unit with ``unit_axis=``).
        name:
            Optional metric name.
        unit_axis:
            Axis of each batch that indexes units (axis 0 is the example
            axis); ``None`` pools every value.
        encoder:
            Optional callable applied to each batch before it is folded, for
            example an SAE's encoder to rank examples per latent.
        """

        self.name = name
        self.k = int(k)
        self.unit_axis = _validate_unit_axis(unit_axis, "tl.stats.TopK")
        self.encoder = _validate_encoder(encoder, "tl.stats.TopK")
        self._heap: list[float] = []
        self._layout: _UnitLayout | None = None
        if self.unit_axis is not None:
            self._layout = _UnitLayout("tl.stats.TopK", self.unit_axis)
        self._values: torch.Tensor | None = None
        self._example_ids: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._examples_seen = 0

    def update(self, value: Any, *, example_ids: Any = None) -> None:
        """Update the tracked top-k values.

        NaN values are ignored: they have no order, so they can neither rank
        among the top k nor displace a real value. The batch is reduced with
        ``torch.topk`` and merged with the retained values (identical
        semantics to the former per-element Python loop at a fraction of the
        cost -- the loop was ~50x slower than ``torch.topk`` on real batches).

        Parameters
        ----------
        value:
            Tensor-like batch value.
        example_ids:
            Per-unit mode only: one integer id per example (axis 0) of this
            batch, naming the examples instead of their stream index.
        """

        value = _encode(self.encoder, value)
        if self._layout is not None:
            self._update_per_unit(_as_batch_tensor(value), example_ids, self._layout)
            return
        if example_ids is not None:
            raise InvalidArgumentError(
                "tl.stats.TopK().update(example_ids=...) needs a per-unit tracker",
                code="stats_example_ids_without_unit_axis",
                remedy="build the tracker with tl.stats.TopK(k, unit_axis=-1) to keep example ids",
            )
        if self.k <= 0:
            return
        tensor = _as_float_tensor(value)
        tensor = tensor[~tensor.isnan()]
        if tensor.numel() == 0:
            return
        if tensor.numel() > self.k:
            tensor = torch.topk(tensor, self.k).values
        if self._heap:
            merged = torch.cat([torch.tensor(self._heap, dtype=torch.float64), tensor])
        else:
            merged = tensor
        if merged.numel() > self.k:
            merged = torch.topk(merged, self.k).values
        self._heap = [float(item) for item in merged.tolist()]
        heapq.heapify(self._heap)

    def _batch_example_ids(self, examples: int, example_ids: Any) -> torch.Tensor:
        """Return this batch's example ids, defaulting to the stream index."""

        if example_ids is None:
            return torch.arange(self._examples_seen, self._examples_seen + examples)
        ids = torch.as_tensor(example_ids)
        if ids.shape != (examples,) or ids.dtype.is_floating_point or ids.dtype == torch.bool:
            raise InvalidArgumentError(
                f"tl.stats.TopK update(example_ids=...) has shape {tuple(ids.shape)} "
                f"and dtype {ids.dtype}; the batch has {examples} examples",
                code="stats_example_ids_invalid",
                remedy="pass one integer id per example, for example example_ids=torch.arange(n)",
                examples=examples,
            )
        return ids.to(torch.long)

    def _update_per_unit(self, tensor: torch.Tensor, example_ids: Any, layout: _UnitLayout) -> None:
        """Fold one batch into the per-unit top-k lists."""

        axis = layout.check(tensor)
        rows, examples, position_shape = _unit_rows(tensor, axis)
        ids = self._batch_example_ids(examples, example_ids)
        self._examples_seen += examples
        units = int(rows.shape[0])
        if self._values is None or self._example_ids is None or self._positions is None:
            _warn_state_size(
                self.name or "tl.stats.TopK",
                units,
                max(self.k, 0) * (16 + 8 * len(position_shape)),
                layout.unit_axis,
                f"tl.stats.TopK(k={max(self.k // 2, 1)}, unit_axis={layout.unit_axis})",
            )
            width = max(self.k, 0)
            self._values = torch.full((units, width), math.nan, dtype=torch.float64)
            self._example_ids = torch.full((units, width), -1, dtype=torch.long)
            self._positions = torch.full((units, width, len(position_shape)), -1, dtype=torch.long)
        values, kept_ids, kept_positions = self._values, self._example_ids, self._positions
        if self.k <= 0 or rows.shape[1] == 0:
            return
        row_examples, row_positions = _row_ids(ids, position_shape)
        if bool((row_examples[1:] < row_examples[:-1]).any()):
            order = torch.argsort(row_examples, stable=True)
            rows, row_examples, row_positions = (
                rows[:, order],
                row_examples[order],
                row_positions[order],
            )
        chunk = max(1, _FOLD_CHUNK_ELEMENTS // int(rows.shape[1]))
        for start in range(0, units, chunk):
            stop = min(start + chunk, units)
            picked = _first_k_rows(rows[start:stop], self.k)
            merged = _rank_entries(
                torch.cat(
                    [
                        values[start:stop],
                        torch.gather(rows[start:stop], 1, picked.clamp(min=0)).masked_fill(
                            picked < 0, math.nan
                        ),
                    ],
                    dim=1,
                ),
                torch.cat([kept_ids[start:stop], _take(row_examples, picked, -1)], dim=1),
                torch.cat([kept_positions[start:stop], _take(row_positions, picked, -1)], dim=1),
                self.k,
            )
            values[start:stop], kept_ids[start:stop], kept_positions[start:stop] = merged

    def result(self) -> list[float] | UnitTopK:
        """Return top values in descending order.

        Returns
        -------
        list[float] | UnitTopK
            Retained top-k values; per unit, a :class:`UnitTopK` (zero units
            before the first batch).
        """

        if self._layout is None:
            return sorted(self._heap, reverse=True)
        if self._values is None or self._example_ids is None or self._positions is None:
            width = max(self.k, 0)
            return UnitTopK(
                values=torch.empty((0, width), dtype=torch.float64),
                example_ids=torch.empty((0, width), dtype=torch.long),
                positions=torch.empty((0, width, 0), dtype=torch.long),
                unit_axis=self._layout.unit_axis,
            )
        return UnitTopK(
            values=self._values.clone(),
            example_ids=self._example_ids.clone(),
            positions=self._positions.clone(),
            unit_axis=self._layout.unit_axis,
        )
