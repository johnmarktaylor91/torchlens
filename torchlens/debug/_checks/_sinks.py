"""Attention-sink and outlier-residual-dimension probe over the public phase seam.

Two facts trained transformers share, measured live on the ``capture.hooks`` seam (the
range and liveness probes' seam, which adds no ops to a trace):

- **Attention sinks.** A head whose queries put most of their attention mass on one key
  position, usually the first token. At an attention-probability site (``tl.func("softmax")``
  in eager attention; values shaped ``(..., heads, queries, keys)``) the probe averages each
  head's probabilities over the batch and over the kept queries and flags the head when its
  largest mean mass on a single key reaches ``mass_threshold``.
- **Outlier residual dimensions.** A feature dimension of the residual stream whose mean
  absolute value dwarfs the others. At a residual site (a block's output; values shaped
  ``(..., positions, features)``) the probe divides each dimension's mean absolute value over
  the kept rows by the median of those means and flags dimensions at or above
  ``outlier_factor``.

Conventions, stated rather than guessed:

- **Positions.** The first ``skip_positions`` query positions are left out of the sink
  statistic (under a causal mask query 0 can attend only to itself, so its mass on key 0 is
  1 by construction) and the first ``skip_positions`` positions are left out of the outlier
  statistic (the first token carries the massive activations sinks feed on, which would
  otherwise dominate every mean).
- **Masks.** ``query_mask`` (``batch x queries``, true for real tokens) drops padded query
  rows from the sink statistic. Masked keys already carry no probability.
- **Axes.** Attention values are read as ``(..., heads, queries, keys)``; residual values as
  ``(..., positions, features)`` with ``feature_axis=-1`` by default.

Statistics, not verdicts: sinks and outlier dimensions are normal in trained language
models. Each site keeps its most recent fire's statistics.

This module is the private implementation of the checks-kit sink check (walk card
recipes.8, ruled 3 Nov 2026). Its public spelling, check names and finding codes are
pending, so it emits no ``CheckFinding`` yet; its argument refusals are plain
``ValueError`` (fraction knobs reuse the kit's typed ``check_fraction_invalid``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ... import _state
from ._audit import _validate_fraction
from ._range import _fired_label

__tl_layer__ = "L5"

__all__ = ("OutlierDim", "SinkHead", "SinkProbe")

#: Largest gap between a probability row's sum and 1 that still reads as probabilities.
_ROW_SUM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class SinkHead:
    """One head flagged as an attention sink at one site."""

    label: str
    head: int
    position: int
    mass: float


@dataclass(frozen=True)
class OutlierDim:
    """One residual feature dimension flagged as an outlier at one site."""

    label: str
    dim: int
    factor: float


class SinkProbe:
    """Live attention-sink and outlier-residual-dimension statistics per site.

    Parameters
    ----------
    attention:
        Attention-probability site targets (anything ``hooks=`` accepts).
    residual:
        Residual-stream site targets.
    mass_threshold:
        Smallest mean mass on one key that flags a head, in ``[0, 1]``.
    outlier_factor:
        Smallest ratio of a dimension's mean absolute value to the median dimension's
        that flags it; above 1.
    skip_positions:
        Leading positions left out of both statistics (see the module docstring).
    query_mask:
        Optional ``batch x queries`` boolean mask of real query positions.
    feature_axis:
        Feature axis of residual values.

    Examples
    --------
    >>> probe = SinkProbe(attention=[tl.func("softmax")], residual=[tl.module("transformer.h.5")])
    >>> trace = tl.trace(model, ids, capture=tl.options.CaptureOptions(hooks=probe.hook_plan()))
    >>> sorted({head.position for head in probe.sink_heads})
    [0]
    """

    def __init__(  # noqa: PLR0913 -- two site lists and the four stated conventions
        self,
        attention: Sequence[Any] = (),
        residual: Sequence[Any] = (),
        *,
        mass_threshold: float = 0.5,
        outlier_factor: float = 10.0,
        skip_positions: int = 1,
        query_mask: torch.Tensor | None = None,
        feature_axis: int = -1,
    ) -> None:
        attention, residual = list(attention), list(residual)
        if not attention and not residual:
            raise ValueError(
                "SinkProbe needs at least one attention or residual site, for example "
                "attention=[tl.func('softmax')]."
            )
        _validate_fraction("mass_threshold", mass_threshold)
        if not outlier_factor > 1.0:
            raise ValueError(f"outlier_factor must be above 1, got {outlier_factor!r}.")
        if isinstance(skip_positions, bool) or not isinstance(skip_positions, int):
            raise ValueError(f"skip_positions must be an int, got {skip_positions!r}.")
        if skip_positions < 0:
            raise ValueError(f"skip_positions must be 0 or more, got {skip_positions}.")
        self._attention = attention
        self._residual = residual
        self._mass_threshold = float(mass_threshold)
        self._outlier_factor = float(outlier_factor)
        self._skip = skip_positions
        self._query_mask = None if query_mask is None else query_mask.detach().bool()
        self._feature_axis = feature_axis
        self._facts: dict[str, dict[str, Any]] = {}
        self._fires: dict[str, int] = {}

    def hook_plan(self) -> dict[Any, Any]:
        """Return the ``hooks=`` mapping for ``CaptureOptions`` (the seam)."""

        plan: dict[Any, Any] = {
            site: self._hook(site, self._observe_attention) for site in self._attention
        }
        plan.update({site: self._hook(site, self._observe_residual) for site in self._residual})
        return plan

    def _hook(self, site: Any, observe: Any) -> Any:
        """Build one site's hook (public ``fn(out, *, ctx)`` shape) around ``observe``."""

        def _sink_hook(out: Any, *, ctx: Any) -> Any:
            """Record one fired value's statistics; the value passes through unchanged."""

            if isinstance(out, torch.Tensor) and out.numel():
                label = _fired_label(ctx, str(site))
                with torch.no_grad(), _state.pause_logging():
                    facts = observe(out.detach().float())
                self._fires[label] = self._fires.get(label, 0) + 1
                self._facts[label] = {**facts, "fires": self._fires[label]}
            return out

        return _sink_hook

    def _query_weights(self, batch: int, queries: int) -> torch.Tensor:
        """``batch x queries`` weights of the queries the sink statistic keeps."""

        weights = torch.ones(batch, queries)
        weights[:, : self._skip] = 0.0
        mask = self._query_mask
        if mask is not None and tuple(mask.shape) == (batch, queries):
            weights = weights * mask.float()
        return weights

    def _observe_attention(self, probs: torch.Tensor) -> dict[str, Any]:
        """Per-head largest mean mass on one key, and the heads at or over the threshold."""

        if probs.ndim < 3:
            reason = f"rank {probs.ndim} is not (heads, queries, keys)"
            return {"kind": "attention", "skipped": reason}
        heads, queries, keys = probs.shape[-3:]
        flat = probs.reshape(-1, heads, queries, keys)
        row_sums = flat.sum(dim=-1)
        if bool((flat < 0).any()) or float((row_sums - 1).abs().max()) > _ROW_SUM_TOLERANCE:
            return {"kind": "attention", "skipped": "rows are not probabilities (sum != 1)"}
        weights = self._query_weights(flat.shape[0], queries)
        if float(weights.sum()) == 0.0:
            reason = "no query kept after skip_positions and the mask"
            return {"kind": "attention", "skipped": reason}
        mean = torch.einsum("bhqk,bq->hk", flat, weights) / weights.sum()
        mass, position = mean.max(dim=-1)
        flagged = [h for h in range(heads) if float(mass[h]) >= self._mass_threshold]
        return {
            "kind": "attention",
            "heads": heads,
            "sink_heads": flagged,
            "sink_positions": [int(position[h]) for h in flagged],
            "top_mass": [round(float(value), 6) for value in mass],
            "top_position": [int(value) for value in position],
        }

    def _observe_residual(self, values: torch.Tensor) -> dict[str, Any]:
        """Per-dimension mean absolute value over the median's, and the outliers."""

        if values.ndim < 2:
            return {"kind": "residual", "skipped": f"rank {values.ndim} has no feature axis"}
        moved = values.movedim(self._feature_axis, -1)
        if moved.ndim >= 3:
            moved = moved[..., self._skip :, :]
        rows = moved.reshape(-1, moved.shape[-1])
        if rows.shape[0] == 0:
            return {"kind": "residual", "skipped": "no position kept after skip_positions"}
        magnitude = rows.abs().mean(dim=0)
        median = float(magnitude.median())
        if median == 0.0:
            return {"kind": "residual", "skipped": "the median dimension is all zeros"}
        factor = magnitude / median
        order = torch.argsort(factor, descending=True)
        flagged = [int(d) for d in order if float(factor[d]) >= self._outlier_factor]
        return {
            "kind": "residual",
            "features": int(rows.shape[-1]),
            "outlier_dims": flagged,
            "outlier_factors": [round(float(factor[d]), 3) for d in flagged],
            "median_abs": round(median, 6),
        }

    def facts(self) -> dict[str, dict[str, Any]]:
        """Per-site statistics of each site's most recent fire (``skipped`` says why not)."""

        return {label: dict(facts) for label, facts in self._facts.items()}

    @property
    def sink_heads(self) -> tuple[SinkHead, ...]:
        """Every flagged sink head, by site label then head index."""

        found = []
        for label, facts in self._facts.items():
            for head, position in zip(facts.get("sink_heads", ()), facts.get("sink_positions", ())):
                found.append(SinkHead(label, head, position, facts["top_mass"][head]))
        return tuple(found)

    @property
    def outlier_dims(self) -> tuple[OutlierDim, ...]:
        """Every flagged outlier dimension, by site label then descending factor."""

        found = []
        for label, facts in self._facts.items():
            for dim, factor in zip(facts.get("outlier_dims", ()), facts.get("outlier_factors", ())):
                found.append(OutlierDim(label, dim, factor))
        return tuple(found)

    @property
    def observed_fires(self) -> dict[str, int]:
        """Per-label fire counts (coverage: zero fires is visible, not silent)."""

        return dict(self._fires)
