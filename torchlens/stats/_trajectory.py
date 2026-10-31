"""Logit-lens trajectory statistics in one call (``tl.mechinterp.measure_trajectory``).

A trajectory is the logit-lens prediction path through depth: each block's
hidden state projected through the model's own head (or a user ``lens=``),
one distribution per layer and position. The named statistics are entropy,
KL from the final output, cross-entropy and rank of a target token, and the
top probability. Every batch is traced, its layers are projected ONE at a
time through the shared lens helper of ``torchlens.semantic.logit_lens``,
each projection is reduced to its statistics and dropped, and the values
fold into ``Spine`` accumulators (one per statistic and layer) plus a
per-position running sum, so memory is constant in the number of batches.

Spellings DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .._errors import ArgumentTypeError, InvalidArgumentError
from ..observability._kernels import RunningStats, RunningStatsResult

__all__ = ["TRAJECTORY_STATISTICS", "TrajectoryStats", "measure_trajectory"]

#: Statistics that need only the projections.
_UNTARGETED = ("entropy", "kl_to_final", "max_prob")
#: Statistics that need a target token per position (the batch tail).
_TARGETED = ("cross_entropy", "target_rank")
#: Every statistic name, in result order.
TRAJECTORY_STATISTICS = _UNTARGETED + _TARGETED

#: Rank convention, the same one logit-lens prediction rows record.
_TIE_CONVENTION = "rank = 1 + count(logits strictly greater); tied logits share the smallest rank"


@dataclass(frozen=True)
class TrajectoryStats:
    """Per-layer trajectory statistics folded over every example and position.

    Attributes
    ----------
    layers:
        Block addresses in depth order (one row per layer).
    statistics:
        Statistic names, in the order of ``position_means``.
    position_means:
        ``{statistic: tensor[layers, positions]}``, float64: the mean over
        every example of each layer's value at each position.
    spines:
        ``{statistic: {layer: Spine}}``: the live accumulators, pooled over
        examples and positions. They keep folding and save with
        ``tl.stats.save_fitted``.
    examples:
        Number of examples (batch rows) folded.
    positions:
        Sequence length every batch shared.
    facet:
        Hidden-state facet each layer was read from.
    lens_source:
        ``"user"`` or the default head's kind, as on logit-lens results.
    validated:
        Whether the default lens was numerically validated.
    mask_input:
        Whether each row's own input tokens were banned before reduction.
    tie_convention:
        How ``target_rank`` breaks ties.
    """

    module_addresses: tuple[str, ...]
    statistics: tuple[str, ...]
    position_means: Mapping[str, torch.Tensor]
    running_stats: Mapping[str, Mapping[str, RunningStats]]
    num_examples: int
    num_positions: int
    facet: str
    lens_source: str
    validated: bool
    is_input_masked: bool
    tie_convention: str = _TIE_CONVENTION

    def compute_stats(self, statistic: str) -> dict[str, RunningStatsResult]:
        """Return ``{layer: SpineResult}`` for one statistic (count, min, max, mean, ...)."""

        if statistic not in self.running_stats:
            raise InvalidArgumentError(
                f"TrajectoryStats.compute_stats({statistic!r}) names a statistic this result "
                f"did not measure; it holds {list(self.statistics)}",
                code="stats_trajectory_statistic_unknown",
                remedy=f"use one of them, for example result.compute_stats({self.statistics[0]!r})",
                statistic=statistic,
            )
        return {layer: spine.result() for layer, spine in self.running_stats[statistic].items()}

    def __repr__(self) -> str:
        """Bounded identity card: never dump the tensors."""

        return (
            f"TrajectoryStats({len(self.module_addresses)} layers x {self.num_positions} positions, "
            f"{self.num_examples} examples, statistics={list(self.statistics)}, "
            f"lens_source={self.lens_source!r})"
        )


def measure_trajectory(  # noqa: PLR0913 -- what (statistics), where (facet, layers), how (lens...)
    model: nn.Module,
    data: Iterable[Any] | torch.Tensor,
    *,
    statistics: Sequence[str] | None = None,
    facet: str = "resid_post",
    layers: Sequence[str] | None = None,
    lens: Any = None,
    mask_input: bool = False,
    verify: bool = True,
) -> TrajectoryStats:
    """Trace each batch and fold its logit-lens trajectory statistics.

    Parameters
    ----------
    model:
        Model whose blocks expose ``facet`` and whose head the lens is
        built from (the same facet recipes ``logit_lens`` reads).
    data:
        Iterable of batches, as for ``tl.stats.aggregate``: a batch is the model
        input, or a tuple ``(model_input, targets)`` whose ``targets`` is an
        integer tensor ``[batch, positions]`` of the token each position
        should predict. A bare tensor is one batch.
    statistics:
        Names to measure among ``"entropy"``, ``"kl_to_final"`` (KL from the
        final output distribution to the layer's), ``"max_prob"``,
        ``"cross_entropy"`` and ``"target_rank"`` (the last two need
        targets). Default: every statistic the batches support.
    facet:
        Hidden-state facet to project, as in ``logit_lens``.
    layers:
        Block addresses to project, as in ``logit_lens``; default every block.
    lens:
        Optional user lens (a tuned lens), as in ``logit_lens``.
    mask_input:
        ``True`` bans each row's own input token ids (the model input must be
        an integer tensor) before every reduction, final output included.
    validate:
        As in ``logit_lens``; the lens is checked on the first batch.

    Returns
    -------
    TrajectoryStats
        Per-position means and pooled ``Spine`` accumulators per statistic
        and layer.
    """

    _check_flags(mask_input=mask_input, validate=verify)
    batches = [data] if isinstance(data, torch.Tensor) else data
    if not isinstance(batches, Iterable):
        raise ArgumentTypeError(
            f"tl.mechinterp.measure_trajectory(model, data) got data of type {type(data).__name__}",
            code="stats_trajectory_data_type",
            remedy="pass an iterable of batches, e.g. tl.mechinterp.measure_trajectory(model, [ids])",
            argument="data",
        )
    from ..semantic.logit_lens import _LensRequest

    request = _LensRequest(facet=facet, layers=layers, lens=lens, validate=verify)
    folder = _TrajectoryFolder(statistics)
    for batch in batches:
        folder.fold_batch(model, _split_batch(batch), request, mask_input)
    return folder.result(facet=facet, mask_input=mask_input)


def _check_flags(*, mask_input: Any, validate: Any) -> None:
    """Refuse non-bool switches (a truthy string must not mean True)."""

    for name, value in (("mask_input", mask_input), ("validate", validate)):
        if not isinstance(value, bool):
            raise ArgumentTypeError(
                f"tl.mechinterp.measure_trajectory({name}={value!r}) needs True or False",
                code="stats_trajectory_flag_type",
                remedy=f"spell it tl.mechinterp.measure_trajectory(model, data, {name}=True)",
                argument=name,
            )


def _split_batch(batch: Any) -> tuple[Any, torch.Tensor | None]:
    """Return ``(model_input, targets)``; the tail of a tuple batch is the targets."""

    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        targets = batch[1]
        if not isinstance(targets, torch.Tensor) or targets.is_floating_point():
            raise ArgumentTypeError(
                "tl.mechinterp.measure_trajectory got a (model_input, targets) batch whose targets "
                f"are {type(targets).__name__}, not an integer token-id tensor",
                code="stats_trajectory_targets_type",
                remedy="yield (input_ids, target_ids) with target_ids a [batch, positions] "
                "integer tensor, or yield the model input alone",
                argument="data",
            )
        return batch[0], targets
    return batch, None


class _TrajectoryFolder:
    """Fold per-batch trajectory statistics into Spines and position sums."""

    def __init__(self, statistics: Sequence[str] | None) -> None:
        """Validate the requested names; defaults resolve on the first batch."""

        if statistics is not None:
            names = tuple(statistics)
            unknown = [name for name in names if name not in TRAJECTORY_STATISTICS]
            if unknown or not names or isinstance(statistics, str):
                raise InvalidArgumentError(
                    f"tl.mechinterp.measure_trajectory(statistics={statistics!r}) names "
                    f"{unknown or 'no statistic'}; known: {list(TRAJECTORY_STATISTICS)}",
                    code="stats_trajectory_statistic_unknown",
                    remedy="pass a list of known names, e.g. statistics=['entropy', 'kl_to_final']",
                    statistic=statistics,
                )
            self.requested: tuple[str, ...] | None = names
        else:
            self.requested = None
        self.names: tuple[str, ...] | None = None
        self.module_addresses: tuple[str, ...] | None = None
        self.num_positions: int | None = None
        self.num_examples = 0
        self.sums: dict[str, torch.Tensor] = {}
        self.running_stats: dict[str, dict[str, RunningStats]] = {}
        self.lens_source = ""
        self.validated = False
        self.checked = False

    def _resolve_names(self, has_targets: bool) -> tuple[str, ...]:
        """Fix the statistic list on the first batch; later batches must match it."""

        if self.names is None:
            if self.requested is None:
                self.names = TRAJECTORY_STATISTICS if has_targets else _UNTARGETED
            else:
                self.names = self.requested
        if not has_targets and any(name in _TARGETED for name in self.names):
            raise InvalidArgumentError(
                "tl.mechinterp.measure_trajectory measures cross_entropy/target_rank, but a batch "
                "carries no targets",
                code="stats_trajectory_targets_missing",
                remedy="yield (input_ids, target_ids) batches, or pass "
                "statistics=['entropy', 'kl_to_final', 'max_prob']",
                statistic=list(self.names),
            )
        return self.names

    def fold_batch(
        self,
        model: nn.Module,
        batch: tuple[Any, torch.Tensor | None],
        request: Any,
        mask_input: bool,
    ) -> None:
        """Trace one ``(model_input, targets)`` batch, project layer by layer, reduce and fold."""

        from ..options import CaptureOptions
        from ..semantic.logit_lens import _iter_layer_logits, _mask_input_tokens, _prepare_lens
        from ..user_funcs import trace

        model_input, targets = batch
        names = self._resolve_names(targets is not None)
        banned = _input_ids(model_input) if mask_input else None
        log = trace(
            model,
            model_input,
            capture=CaptureOptions(layers_to_save="all", save_arg_values=True),
        )
        try:
            # The lens is checked once, on the first batch.
            first = dataclasses.replace(request, validate=request.validate and not self.checked)
            setup = _prepare_lens(log, first)
            self.checked = True
            self._start(setup.addresses, setup.lens_source, setup.validated)
            final = _final_log_probs(setup.final_logits, banned, names, _mask_input_tokens)
            for _index, address, logits in _iter_layer_logits(log, setup):
                work = _as_bpv(logits)
                if banned is not None:
                    work = _mask_input_tokens(work, banned)
                values = trajectory_values(work, final, targets, names)
                del work, logits
                self._fold_layer(address, values)
            self.num_examples += int(next(iter(values.values())).shape[0]) if setup.addresses else 0
        finally:
            log.cleanup()

    def _start(self, addresses: tuple[str, ...], lens_source: str, validated: bool) -> None:
        """Record the layer list on the first batch; refuse a different one later."""

        if self.module_addresses is None:
            self.module_addresses = addresses
            self.lens_source, self.validated = lens_source, validated
        elif self.module_addresses != addresses:
            raise InvalidArgumentError(
                "tl.mechinterp.measure_trajectory: a later batch exposes different layers "
                f"({list(addresses)}) than the first ({list(self.module_addresses)})",
                code="stats_trajectory_layers_changed",
                remedy="pass layers= explicitly so every batch projects the same blocks",
                layers=list(addresses),
            )

    def _fold_layer(self, address: str, values: dict[str, torch.Tensor]) -> None:
        """Fold one layer's ``{statistic: [batch, positions]}`` values."""

        row = (self.module_addresses or ()).index(address)
        for name, value in values.items():
            positions = int(value.shape[-1])
            if self.num_positions is None:
                self.num_positions = positions
            elif positions != self.num_positions:
                raise InvalidArgumentError(
                    f"tl.mechinterp.measure_trajectory: a batch has {positions} positions after "
                    f"batches of {self.num_positions}",
                    code="stats_trajectory_positions_changed",
                    remedy="pad or truncate every batch to one sequence length",
                    positions=positions,
                )
            column_sum = value.to(torch.float64).sum(dim=0)
            if name not in self.sums:
                self.sums[name] = torch.zeros(
                    len(self.module_addresses or ()), positions, dtype=torch.float64
                )
            self.sums[name][row] += column_sum.cpu()
            spine = self.running_stats.setdefault(name, {}).setdefault(
                address, RunningStats(name=name)
            )
            spine.update(value)

    def result(self, *, facet: str, mask_input: bool) -> TrajectoryStats:
        """Freeze the folded state into a ``TrajectoryStats``."""

        if self.module_addresses is None or self.names is None or self.num_examples == 0:
            raise InvalidArgumentError(
                "tl.mechinterp.measure_trajectory folded no examples (empty data or no layers)",
                code="stats_trajectory_empty",
                remedy="pass at least one batch, e.g. tl.mechinterp.measure_trajectory(model, [ids])",
                examples=self.num_examples,
            )
        means = {name: self.sums[name] / self.num_examples for name in self.names}
        return TrajectoryStats(
            module_addresses=self.module_addresses,
            statistics=self.names,
            position_means=means,
            running_stats={name: dict(self.running_stats[name]) for name in self.names},
            num_examples=self.num_examples,
            num_positions=int(self.num_positions or 0),
            facet=facet,
            lens_source=self.lens_source,
            validated=self.validated,
            is_input_masked=mask_input,
        )


def _input_ids(model_input: Any) -> torch.Tensor:
    """Return the model input as ``[batch, positions]`` token ids for ``mask_input``."""

    if not isinstance(model_input, torch.Tensor) or model_input.is_floating_point():
        raise InvalidArgumentError(
            "tl.mechinterp.measure_trajectory(mask_input=True) bans the input's own token ids, "
            f"but the model input is {type(model_input).__name__}"
            + (" of floats" if isinstance(model_input, torch.Tensor) else "")
            + ", not an integer token-id tensor",
            code="stats_trajectory_mask_input_unavailable",
            remedy="feed token ids, e.g. tl.mechinterp.measure_trajectory(model, [input_ids], "
            "mask_input=True), or drop mask_input=True",
            argument="mask_input",
        )
    ids = model_input.detach().long()
    return ids.unsqueeze(0) if ids.ndim == 1 else ids.reshape(ids.shape[0], -1)


def _as_bpv(logits: torch.Tensor) -> torch.Tensor:
    """View projected logits as ``[batch, positions, vocab]`` in float32 or float64."""

    work = logits.detach()
    if work.ndim == 2:
        work = work.unsqueeze(-2)
    elif work.ndim > 3:
        work = work.reshape(-1, work.shape[-2], work.shape[-1])
    if work.ndim != 3:
        raise InvalidArgumentError(
            f"tl.mechinterp.measure_trajectory: projected logits have shape {tuple(logits.shape)}",
            code="stats_trajectory_projection_rank",
            remedy="use a lens= that returns [batch, positions, vocab] logits",
            shape=tuple(logits.shape),
        )
    return work if work.dtype == torch.float64 else work.float()


def _final_log_probs(
    final_logits: torch.Tensor | None,
    banned: torch.Tensor | None,
    names: tuple[str, ...],
    mask: Any,
) -> torch.Tensor | None:
    """Return the final output's log-probabilities when ``kl_to_final`` needs them."""

    if "kl_to_final" not in names:
        return None
    if final_logits is None:
        raise InvalidArgumentError(
            "tl.mechinterp.measure_trajectory measures kl_to_final, but the trace has no readable "
            "captured output logits to compare against",
            code="stats_trajectory_final_unavailable",
            remedy="drop it: statistics=['entropy', 'max_prob'], or register the head's "
            "'logits' facet",
            statistic="kl_to_final",
        )
    work = _as_bpv(final_logits)
    if banned is not None:
        work = mask(work, banned)
    return torch.log_softmax(work, dim=-1)


def trajectory_values(
    work: torch.Tensor,
    final_log_probs: torch.Tensor | None,
    targets: torch.Tensor | None,
    names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Reduce one layer's ``[batch, positions, vocab]`` logits to ``{statistic: [batch, positions]}``.

    The formulas (natural log; masked tokens carry ``-inf`` logits and zero
    probability): entropy ``-sum p log p``; ``kl_to_final``
    ``sum q (log q - log p)`` with ``q`` the final output distribution;
    ``max_prob`` ``max p``; ``cross_entropy`` ``-log p[target]``;
    ``target_rank`` ``1 + count(logit > logit[target])``.
    """

    with torch.no_grad():
        log_probs = torch.log_softmax(work, dim=-1)
        probs = log_probs.exp()
        out: dict[str, torch.Tensor] = {}
        if "entropy" in names:
            out["entropy"] = -torch.where(probs > 0, probs * log_probs, 0.0).sum(dim=-1)
        if "kl_to_final" in names and final_log_probs is not None:
            q_log = _align_final(final_log_probs, work)
            q = q_log.exp()
            out["kl_to_final"] = torch.where(q > 0, q * (q_log - log_probs), 0.0).sum(dim=-1)
        if "max_prob" in names:
            out["max_prob"] = probs.amax(dim=-1)
        if targets is not None and any(name in names for name in _TARGETED):
            index = _target_index(targets, work)
            if "cross_entropy" in names:
                out["cross_entropy"] = -log_probs.gather(-1, index).squeeze(-1)
            if "target_rank" in names:
                target_logit = work.gather(-1, index)
                out["target_rank"] = ((work > target_logit).sum(dim=-1) + 1).to(work.dtype)
        return {name: out[name] for name in names}


def _align_final(final_log_probs: torch.Tensor, work: torch.Tensor) -> torch.Tensor:
    """Return final log-probs covering every layer position, or refuse."""

    if tuple(final_log_probs.shape) != tuple(work.shape):
        raise InvalidArgumentError(
            "tl.mechinterp.measure_trajectory measures kl_to_final, but the captured output logits "
            f"{tuple(final_log_probs.shape)} do not cover every projected position "
            f"{tuple(work.shape)} (a logits_to_keep-sliced head)",
            code="stats_trajectory_final_unavailable",
            remedy="run the model with full-sequence logits, or drop kl_to_final: "
            "statistics=['entropy', 'max_prob']",
            statistic="kl_to_final",
        )
    return final_log_probs


def _target_index(targets: torch.Tensor, work: torch.Tensor) -> torch.Tensor:
    """Return ``[batch, positions, 1]`` target ids checked against the vocabulary."""

    ids = targets.detach().long().to(work.device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    vocab = int(work.shape[-1])
    if tuple(ids.shape) != tuple(work.shape[:-1]) or bool(((ids < 0) | (ids >= vocab)).any()):
        raise InvalidArgumentError(
            f"tl.mechinterp.measure_trajectory: targets {tuple(ids.shape)} must be token ids in "
            f"[0, {vocab}) shaped like the projections {tuple(work.shape[:-1])}",
            code="stats_trajectory_targets_type",
            remedy="yield (input_ids, target_ids) with one target id per batch row and position",
            shape=tuple(ids.shape),
        )
    return ids.unsqueeze(-1)
