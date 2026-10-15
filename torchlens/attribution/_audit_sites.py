"""The causal audit's plan for claims spread over several sites.

A direct-logit-attribution table and a one-backward read table each score
components that live at different sites of the model, one row per site (or,
for an element-grain read, one row per site with a value per element). The
plan here holds those units, each with its site, the claim's own name for it
and its claimed effect, and measures any set of them through
``trace.fork().do(...)`` on the audit's intervention-ready trace.

A unit is an index into the claim: ``(row,)`` for a whole-site row, or
``(row, *element)`` for one element of an element-grain row (the element
indexed after the batch axis, ablated in every example). Whole-site units are
edited through their site label, element units through
``tl.units(site, mask)``; the recorded pairs are exactly what was executed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from torchlens.attribution._audit_claims import _setting_error


@dataclass(frozen=True)
class _SiteUnit:
    """One audited unit: its site, the element (``()`` = the whole site) and the claim."""

    key: tuple[int, ...]
    site: str
    element: tuple[int, ...]
    component: str
    claimed: float


@dataclass(frozen=True)
class _SitesPlan:
    """What a multi-site audit measures: units at their sites, the edits and the score."""

    trace: Any
    units: tuple[tuple[int, ...], ...]
    rows: dict[tuple[int, ...], _SiteUnit]
    shapes: dict[str, tuple[int, ...]]
    ablation_name: str
    edits: dict[str, Callable[[], Any]]
    score: Callable[[Any], float]
    claim_kind: str
    claim_method: str
    target_repr: str
    extra: dict[str, Any]
    layer: str | None = None
    label: str | None = None

    def claimed_of(self, key: tuple[int, ...]) -> float:
        """Return the claimed effect of one unit."""

        return self.rows[key].claimed

    def site_of(self, key: tuple[int, ...]) -> str:
        """Return the op label one unit lives at."""

        return self.rows[key].site

    def component_of(self, key: tuple[int, ...]) -> str:
        """Return the claim's name for the component one unit belongs to."""

        return self.rows[key].component

    def intervention(self, keys: tuple[tuple[int, ...], ...]) -> tuple[tuple[Any, Any], ...]:
        """Return the ``(selection, edit)`` pairs ablating ``keys``, one pair per site.

        A site ablated whole is selected by its op label; elements of a site
        by ``tl.units(site, mask)`` with the mask selecting them in every
        example.
        """

        from torchlens.selection import units as unit_selection

        by_site: dict[str, list[tuple[int, ...]]] = {}
        for key in keys:
            row = self.rows[key]
            by_site.setdefault(row.site, []).append(row.element)
        pairs: list[tuple[Any, Any]] = []
        for site, elements in by_site.items():
            if any(element == () for element in elements):
                pairs.append((site, self.edits[site]()))
                continue
            mask = torch.zeros(self.shapes[site], dtype=torch.bool)
            for element in elements:
                mask[(slice(None), *element)] = True
            pairs.append((unit_selection(site, mask), self.edits[site]()))
        return tuple(pairs)

    def ablated_score(self, keys: tuple[tuple[int, ...], ...]) -> float:
        """Score a fork of the trace with ``keys`` ablated together."""

        fork = self.trace.fork(name="attribution_causal_audit")
        try:
            with torch.no_grad():
                for selection, edit in self.intervention(keys):
                    fork.do(selection, edit)
                return self.score(fork)
        finally:
            fork.cleanup()

    def unit_scores(self) -> dict[tuple[int, ...], float]:
        """Score every unit ablated alone.

        Whole-site rows under zero ablation run as one
        ``torchlens.experiment.site_sweep`` (one candidate per site, replay
        engine, nothing retained); any other unit, and any candidate the
        sweep did not complete, is measured by its own fork.
        """

        swept = self._swept_scores() if self.ablation_name == "zero" else {}
        return {
            key: swept[key] if key in swept else self.ablated_score((key,)) for key in self.units
        }

    def _swept_scores(self) -> dict[tuple[int, ...], float]:
        """Run the whole-site units through ``site_sweep``; return the completed scores."""

        from torchlens.experiment import site_sweep
        from torchlens.intervention import zero_ablate

        whole = {
            f"u{index}": key for index, key in enumerate(self.units) if not self.rows[key].element
        }
        if not whole:
            return {}
        with torch.no_grad():
            bundle = site_sweep(
                self.trace,
                candidates={name: self.rows[key].site for name, key in whole.items()},
                edit=zero_ablate(),
                metric=self.score,
                retain="none",
                engine="replay",
            )
        return {
            whole[row.candidate_id]: float(row.value)
            for row in bundle.effects().rows
            if row.candidate_id in whole
            and row.status in ("completed", "released")
            and row.value is not None
        }


def _site_edits(
    ablation: Any, sites: dict[str, Tensor], *, remedy: str
) -> tuple[str, dict[str, Callable[[], Any]]]:
    """Return the ablation's name and one edit factory per site.

    ``"zero"`` is ``tl.zero_ablate()``; ``"mean"`` replaces each site with its
    per-element mean over the batch; a number or tensor is a baseline
    activation broadcast to each site.
    """

    from torchlens.intervention import zero_ablate
    from torchlens.intervention.predicates import replace_with

    if isinstance(ablation, str) and ablation == "zero":
        return "zero", dict.fromkeys(sites, zero_ablate)
    edits: dict[str, Callable[[], Any]] = {}
    for site, value in sites.items():
        edits[site] = _replacement(ablation, site, value, remedy=remedy, replace_with=replace_with)
    return ("mean" if isinstance(ablation, str) else "baseline"), edits


def _replacement(
    ablation: Any, site: str, value: Tensor, *, remedy: str, replace_with: Any
) -> Callable[[], Any]:
    """Return the replace-with edit factory of one site."""

    if isinstance(ablation, str):
        fill = value.detach().mean(dim=0, keepdim=True).expand_as(value).clone()
        return lambda: replace_with(fill.clone())
    baseline = torch.as_tensor(ablation, dtype=value.dtype, device=value.device)
    try:
        fill = torch.broadcast_to(baseline, value.shape).clone()
    except RuntimeError as exc:
        raise _setting_error(
            f"the ablation baseline of shape {tuple(baseline.shape)} does not broadcast to "
            f"the output of site {site!r}, of shape {tuple(value.shape)}",
            remedy,
            code="attribution_eval_setting_invalid",
        ) from exc
    return lambda: replace_with(fill.clone())
