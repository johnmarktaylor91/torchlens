"""Grounding a direct-logit-attribution table for the causal audit.

``torchlens.mechinterp.direct_logit_contributions`` splits the model's logit (or
logit difference) at chosen positions into one row per component, plus one
constant row (the final norm's shift and the unembedding bias, projected).
Each row is a claim that removing the component removes that much logit.
The audit tests it the way direct logit attribution is usually checked
against ablation (Elhage et al., "A Mathematical Framework for Transformer
Circuits", 2021; Wang et al., "Interpretability in the Wild", 2023): it
ablates the component's whole site (``Coordinate.op_label``) and measures
the change of the same logit quantity read from the unembedding head's
output on the edited fork.

Before anything is ablated, the head's logits on the audit's own trace must
reproduce ``claim.native`` at the claim's positions and tokens (within the
claim's own identity tolerance); otherwise the claim is about a different run
and the audit refuses. The constant row has no site and is reported apart.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from torchlens.attribution._audit_claims import _AUDIT_DOCS_POINTER, _AuditRequest
from torchlens.attribution._audit_sites import _site_edits, _SitesPlan, _SiteUnit
from torchlens.attribution._result import AttributionError


def _is_contribution_table(claim: Any) -> bool:
    """Return whether ``claim`` is a ``torchlens.mechinterp`` direct-logit table."""

    from torchlens.mechinterp._records import ContributionScores

    return isinstance(claim, ContributionScores)


def _dla_spelling(claim: Any) -> str:
    """Return a runnable ``direct_logit_contributions`` call for the claim's tokens."""

    vs = f", vs={claim.vs_tokens!r}" if claim.vs_tokens is not None else ""
    return (
        "torchlens.mechinterp.direct_logit_contributions(tl.trace(model, x), "
        f"answer={claim.answer_tokens!r}{vs}, positions={claim.positions!r})"
    )


def _dla_remedy(claim: Any) -> str:
    """Return the runnable audit spelling with the claim recomputed on these inputs."""

    return f"tl.attribution.causal_audit(model, x, claim={_dla_spelling(claim)}, k=1)"


def _dla_refusal(claim: Any, why: str, *, code: str) -> AttributionError:
    """Build a coded refusal for a direct-logit claim the audit cannot measure."""

    return AttributionError(
        f"causal_audit cannot audit this direct-logit-attribution table: {why}. "
        f"Remedy: recompute the table from the model on these inputs, e.g. "
        f"{_dla_remedy(claim)}. {_AUDIT_DOCS_POINTER}",
        code=code,
    )


def _logits_label(trace: Any, claim: Any) -> str:
    """Return the op label of the unembedding head's output on the audit trace."""

    from torchlens.mechinterp._anchors import resolve_lm_head

    try:
        head = resolve_lm_head(trace).head
        outputs = list(head.calls[0].output_ops)
    except Exception as exc:  # noqa: BLE001 - any head failure is one refusal
        raise _dla_refusal(
            claim,
            f"the audit's trace of the model has no resolvable unembedding head ({exc})",
            code="attribution_audit_claim_unsupported",
        ) from exc
    if len(outputs) != 1 or head.num_calls != 1:
        raise _dla_refusal(
            claim,
            "the unembedding head does not run once with one output",
            code="attribution_audit_claim_unsupported",
        )
    return str(outputs[0])


def _logit_rows(trace: Any, claim: Any, logits: Tensor) -> tuple[int, ...]:
    """Return the logits rows of the claim's residual positions (or refuse)."""

    from torchlens.mechinterp._anchors import resolve_lm_head
    from torchlens.mechinterp._dla import _map_positions

    view = resolve_lm_head(trace).facets()
    try:
        position_map = view["logits_position_map"]
    except KeyError:
        position_map = None
    residual_len = int(resolve_lm_head(trace).target_op.out.shape[1])
    try:
        _, rows = _map_positions(claim.positions, residual_len, int(logits.shape[-2]), position_map)
    except Exception as exc:  # noqa: BLE001 - a position refusal becomes the audit's own
        raise _dla_refusal(
            claim,
            f"its positions {claim.positions!r} do not map onto the logits ({exc})",
            code="attribution_audit_claim_misaligned",
        ) from exc
    return rows


def _logit_quantity(logits: Tensor, rows: tuple[int, ...], claim: Any) -> Tensor:
    """Return ``[batch, positions, directions]`` logits (or logit diffs) in float64."""

    values = logits.detach().to(torch.float64)
    if values.dim() == 2:
        values = values.unsqueeze(0)
    at_rows = values.index_select(-2, torch.as_tensor(rows, dtype=torch.long))
    quantity = at_rows.index_select(-1, torch.as_tensor(claim.answer_tokens, dtype=torch.long))
    if claim.vs_tokens is not None:
        quantity = quantity - at_rows.index_select(
            -1, torch.as_tensor(claim.vs_tokens, dtype=torch.long)
        )
    return quantity


def _check_native(claim: Any, baseline: Tensor) -> None:
    """Refuse unless the audit trace's logits reproduce ``claim.native``."""

    native = claim.native.detach().to(torch.float64)
    if tuple(native.shape) != tuple(baseline.shape):
        raise _dla_refusal(
            claim,
            f"its native logits have shape {tuple(native.shape)} but the model gives "
            f"{tuple(baseline.shape)} at its positions on these inputs",
            code="attribution_audit_claim_misaligned",
        )
    tolerance = float(claim.identity_receipt.get("tolerance", 0.0))
    gap = float((baseline.to(native.dtype) - native).abs().max())
    if gap > tolerance:
        raise _dla_refusal(
            claim,
            f"the model's logits on these inputs miss the table's native logits by {gap:.3e} "
            f"(the table's identity tolerance is {tolerance:.3e}), so the table describes a "
            "different run (other inputs, weights or mode)",
            code="attribution_audit_claim_misaligned",
        )


def _has_site(trace: Any, label: str) -> bool:
    """Return whether ``label`` names an op with a saved output on ``trace``."""

    try:
        return isinstance(trace[label].out, Tensor)
    except (KeyError, ValueError, IndexError, AttributeError, RuntimeError):
        return False


def _component_units(claim: Any, trace: Any) -> tuple[_SiteUnit, ...]:
    """Return one whole-site unit per component row, or refuse a row with no site here."""

    units = []
    seen: dict[str, str] = {}
    for row, coordinate in enumerate(claim.coordinates):
        site = coordinate.op_label
        if not isinstance(site, str) or not _has_site(trace, site):
            raise _dla_refusal(
                claim,
                f"row {row} ({coordinate.label!r}) has no ablatable site in the audit's trace "
                f"(op label {site!r}); the constant row is reported apart, but every component "
                "row needs a captured site",
                code="attribution_audit_claim_unsupported",
            )
        if site in seen:
            raise _dla_refusal(
                claim,
                f"rows {seen[site]!r} and {coordinate.label!r} share the site {site!r}, so they "
                "cannot be ablated apart",
                code="attribution_audit_claim_unsupported",
            )
        seen[site] = coordinate.label
        claimed = math.fsum(float(v) for v in claim.values[row].detach().flatten().tolist())
        units.append(_SiteUnit((row,), site, (), coordinate.label, claimed))
    return tuple(units)


def _plan_logit_audit(request: _AuditRequest, trace: Any) -> _SitesPlan:
    """Ground a direct-logit-attribution table on the audit's trace."""

    claim = request.claim
    if request.target is not None:
        raise _dla_refusal(
            claim,
            "a direct-logit-attribution table fixes its own target (its answer and vs tokens "
            "at its positions), so target= must be left out",
            code="attribution_audit_claim_misaligned",
        )
    if "partial" in str(claim.identity_receipt.get("check", "")):
        raise _dla_refusal(
            claim,
            "it decomposes part of the residual stream (a partial component stack), so its "
            "native values are a projection of that part, not the model's logits",
            code="attribution_audit_claim_unsupported",
        )
    label = _logits_label(trace, claim)
    rows = _logit_rows(trace, claim, trace[label].out)
    _check_native(claim, _logit_quantity(trace[label].out, rows, claim))
    units = _component_units(claim, trace)
    sites = {unit.site: trace[unit.site].out for unit in units}
    name, edits = _site_edits(
        request.ablation,
        sites,
        remedy=f"tl.attribution.causal_audit(model, x, claim={_dla_spelling(claim)}, edit=0.0)",
    )

    def _score(run: Any) -> float:
        """The claim's logit quantity on a trace or fork, summed over batch, positions, tokens."""

        quantity = _logit_quantity(run[label].out, rows, claim)
        return math.fsum(quantity.flatten().tolist())

    vs = f", vs={claim.vs_tokens!r}" if claim.vs_tokens is not None else ""
    return _SitesPlan(
        trace=trace,
        units=tuple(unit.key for unit in units),
        rows={unit.key: unit for unit in units},
        shapes={site: tuple(int(size) for size in value.shape) for site, value in sites.items()},
        ablation_name=name,
        edits=edits,
        score=_score,
        claim_kind="direct_logit",
        claim_method="direct_logit_contributions",
        target_repr=f"logits(answer={claim.answer_tokens!r}{vs}, positions={claim.positions!r})",
        extra={
            "logits_site": label,
            "constant_row": math.fsum(claim.constant.detach().flatten().tolist()),
            "native_score": math.fsum(claim.native.detach().to(torch.float64).flatten().tolist()),
        },
    )
