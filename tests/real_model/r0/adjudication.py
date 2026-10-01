"""The absence-claim adjudicator (testing MEMO D2 ladder #2, D7 class 2).

Every ``structurally_absent`` claim in a facet-coverage report is a FACTUAL
ASSERTION about the user's model. The adjudicator checks each claim against
the module tree: a classified module of the required kind present in the
tree makes the claim a FAILURE, not information. Two claims were measurably
false at memo time (GPT-2's ``final_norm_kind`` while ``transformer.ln_f``
sits classified in the same report; Llama's ``q`` laundered from a recipe
defect into a model-property claim) -- the sweep pins them through an
enumerated known-false manifest so a NEW false claim fails immediately and
a FIXED one goes stale loudly.

Rules are structural and mechanical -- config/shape/tree-based, never
foreign class names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

FINAL_NORM_FACETS = frozenset(
    {"final_norm_kind", "final_norm_eps", "final_norm_gamma", "final_norm_beta", "final_norm_input"}
)
ATTENTION_PROJECTION_FACETS = frozenset({"q", "k", "v", "attn_out"})
NORM_RECIPES = frozenset({"layer_norm", "rms_norm"})
BLOCK_RECIPES = frozenset({"transformer_residuals"})


@dataclass(frozen=True)
class Falsification:
    """One adjudicated-false ``structurally_absent`` claim."""

    address: str
    facet: str
    rule: str
    evidence: str


def _is_descendant(address: str, ancestor: str) -> bool:
    if ancestor in ("", "self"):
        return address not in ("", "self")
    return address.startswith(ancestor + ".")


def _linear_like_descendants(module: nn.Module) -> int:
    try:
        from transformers.pytorch_utils import Conv1D  # GPT-2's projection class
    except ImportError:  # pragma: no cover - transformers always present at R0
        Conv1D = ()  # type: ignore[assignment]
    count = 0
    for child in module.modules():
        if isinstance(child, nn.Linear) or (Conv1D and isinstance(child, Conv1D)):
            count += 1
    return count


def adjudicate(report: Any, model: nn.Module) -> tuple[Falsification, ...]:
    """Adjudicate every ``structurally_absent`` claim in ``report``.

    Parameters
    ----------
    report:
        A ``FacetCoverageReport`` from ``torchlens.semantic.coverage``.
    model:
        The live model the trace captured (module-tree evidence source).

    Returns
    -------
    tuple[Falsification, ...]
        Every claim the module tree falsifies, in report order.
    """

    rows = report.rows
    block_addresses = [r.address for r in rows if set(r.recipes) & BLOCK_RECIPES]
    top_level_norms = [
        r.address
        for r in rows
        if set(r.recipes) & NORM_RECIPES
        and not any(_is_descendant(r.address, blk) for blk in block_addresses)
    ]
    availability: dict[str, set[str]] = {}
    for r in rows:
        for recipe in r.recipes:
            availability.setdefault(recipe, set()).update(r.available)

    falsified: list[Falsification] = []
    for r in rows:
        for facet, status, _detail in r.missing:
            if status != "structurally_absent":
                continue
            # Rule 1: a final-norm claim while a top-level norm-classified
            # module (outside every block) sits in the same report.
            if facet in FINAL_NORM_FACETS and top_level_norms:
                falsified.append(
                    Falsification(
                        address=r.address,
                        facet=facet,
                        rule="final_norm_present",
                        evidence=f"norm-classified module(s) {top_level_norms} outside"
                        " every block row",
                    )
                )
                continue
            # Rule 2: an attention row denying its own projections while the
            # module owns linear-like children.
            if facet in ATTENTION_PROJECTION_FACETS and any(
                "attention" in recipe for recipe in r.recipes
            ):
                try:
                    module = model.get_submodule(r.address) if r.address != "self" else model
                except AttributeError:
                    module = None
                if module is not None and _linear_like_descendants(module) >= 1:
                    falsified.append(
                        Falsification(
                            address=r.address,
                            facet=facet,
                            rule="attention_projections_present",
                            evidence=f"{_linear_like_descendants(module)} linear-like"
                            " submodules under the claiming attention module",
                        )
                    )
                    continue
            # Rule 3: the same recipe serves this facet on a sibling row of
            # the same trace -- per-row laundering.
            for recipe in r.recipes:
                if facet in availability.get(recipe, ()):
                    falsified.append(
                        Falsification(
                            address=r.address,
                            facet=facet,
                            rule="same_recipe_serves_it_elsewhere",
                            evidence=f"recipe {recipe!r} has {facet!r} available on"
                            " another row of this trace",
                        )
                    )
                    break
    return tuple(falsified)
