"""Optimizer adapters for the parameter checks (checks memo item 4).

AdamW / Adam / SGD ship day 1 with decoupled-decay awareness; an UNKNOWN
optimizer never guesses -- its decay band and momentum facts come back
``unavailable`` with a reason, so adding adapters later is purely additive
(memo section 7). Findings carry ``beta1``/``beta2`` when an adapter can
read them, so the momentum lag horizon can be printed later as a pure
formatting change ([UI-SPRINT]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

__tl_layer__ = "L5"

#: Optimizer classes with day-1 adapters and their decoupled-decay truth.
#: AdamW applies weight decay decoupled from the gradient (a rescale applied
#: REGARDLESS of the gradient -- the measured vacuity trap of memo D2); SGD's
#: ``weight_decay`` folds into the gradient instead.
_KNOWN_OPTIMIZERS: dict[str, bool] = {
    "AdamW": True,
    "Adam": False,
    "SGD": False,
}


@dataclass(frozen=True)
class OptimizerFacts:
    """Adapter-read facts about one optimizer (memo item 4).

    ``known`` gates every derived band: an unknown optimizer yields
    ``band_reason`` instead of a guessed decay/momentum model.
    """

    kind: str
    known: bool
    decoupled_decay: bool | None
    band_reason: str | None
    param_group_facts: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def group_fact(self, group_index: int, key: str) -> Any:
        """Return one per-group hyperparameter fact, or ``None``."""

        if 0 <= group_index < len(self.param_group_facts):
            return self.param_group_facts[group_index].get(key)
        return None


def optimizer_facts(optimizer: torch.optim.Optimizer) -> OptimizerFacts:
    """Read adapter facts from an optimizer without touching device state.

    Parameters
    ----------
    optimizer:
        Any torch optimizer. Known kinds (AdamW / Adam / SGD) yield decay
        and momentum facts; unknown kinds yield an honest
        ``band_reason`` and no derived bands (memo item 4: unavailable with
        reason, never a guess).

    Returns
    -------
    OptimizerFacts
        Frozen adapter facts with per-group lr / weight_decay / betas /
        momentum where present.
    """

    kind = type(optimizer).__name__
    known = kind in _KNOWN_OPTIMIZERS
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        facts: dict[str, Any] = {"n_params": len(group.get("params", ()))}
        for key in ("lr", "weight_decay", "betas", "momentum", "eps"):
            if key in group:
                value = group[key]
                facts[key] = tuple(value) if isinstance(value, (list, tuple)) else value
        groups.append(facts)
    return OptimizerFacts(
        kind=kind,
        known=known,
        decoupled_decay=_KNOWN_OPTIMIZERS.get(kind),
        band_reason=(
            None
            if known
            else (
                f"no adapter for optimizer {kind!r}: decay/momentum bands are "
                "unavailable rather than guessed; AdamW, Adam, and SGD ship "
                "adapters day 1"
            )
        ),
        param_group_facts=tuple(groups),
    )


def param_group_lr(optimizer: torch.optim.Optimizer, param: torch.Tensor) -> float | None:
    """Return the learning rate of the group holding ``param``, if any."""

    for group in optimizer.param_groups:
        for candidate in group.get("params", ()):
            if candidate is param:
                lr = group.get("lr")
                return float(lr) if lr is not None else None
    return None


__all__ = ["OptimizerFacts", "optimizer_facts", "param_group_lr"]
