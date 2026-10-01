"""Stack composition helpers: norm folding, generic stackers, full grain
(mikit roster items ``apply_norm_scale``, ``stack_facet``,
``full_decomposition``).

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from typing import Any

import torch

from ._anchors import resolve_lm_head
from ._errors import refuse
from ._heads import attention_head_contributions
from ._records import ComponentRow, ComponentStack, Coordinate, StackGrading
from ._residual import residual_decomposition

__all__ = ["apply_norm_scale", "full_decomposition", "stack_facet"]


def apply_norm_scale(
    trace: Any,
    stack: ComponentStack,
    *,
    norm: str = "final",
    scale: str = "frozen",
) -> ComponentStack:
    """Fold one checked norm linearization into a component stack (D8).

    Parameters
    ----------
    trace:
        The trace whose captured norm supplies the operating point.
    stack:
        Any component stack whose rows live in the norm's input space.
    norm:
        ``"final"`` (the dataflow-anchored final norm) or a norm module
        address for intermediate folds.
    scale:
        ``"frozen"`` (default; TLens's own default, verified from source) or
        ``"component"`` -- the per-component RECOMPUTE mode, which is NOT
        additive: the row-sum identity is skipped, the receipt says so, and
        DLA refuses such stacks. TLens ``apply_ln_to_stack``.
    """

    if scale not in ("frozen", "component"):
        refuse(
            code="mi_norm_scale_mode_invalid",
            message=f"Unknown scale mode {scale!r}.",
            remedy='pass scale="frozen" (additive, default) or scale="component" '
            "(non-additive recompute, diagnostic)",
        )
    reconstruction = _norm_reconstruction_for(trace, norm)
    rows = []
    for row in stack.rows:
        value = row.value.to(torch.float32)
        if scale == "frozen":
            folded = reconstruction.apply_frozen(value)
        else:
            centered = (
                value - value.mean(dim=-1, keepdim=True) if reconstruction.centered else value
            )
            own_scale = torch.sqrt(centered.pow(2).mean(dim=-1, keepdim=True) + reconstruction.eps)
            folded = centered / own_scale
            if reconstruction.gamma is not None:
                folded = folded * reconstruction.gamma
        rows.append(
            ComponentRow(
                coordinate=Coordinate(
                    label=row.coordinate.label,
                    op_label=row.coordinate.op_label,
                    site_key=row.coordinate.site_key,
                    pass_index=row.coordinate.pass_index,
                    kind=row.coordinate.kind,
                    module_address=row.coordinate.module_address,
                    provenance=f"{row.coordinate.provenance}; norm_scale={scale}",
                ),
                value=folded,
                coefficient=row.coefficient,
            )
        )
    if scale == "frozen":
        receipt = {
            "check": "frozen_linearization",
            "result": "additive (rows sum to the normalized target minus beta)",
            "norm": reconstruction.module_address,
            "kind": reconstruction.kind,
        }
        diagnostic = stack.diagnostic_only
    else:
        receipt = {
            "check": "per_component_recompute",
            "result": "skipped: scale='component' is NOT additive by construction "
            "(each row renormalized by its own scale); diagnostic only",
            "norm": reconstruction.module_address,
            "kind": reconstruction.kind,
        }
        diagnostic = True
    return ComponentStack(
        tuple(rows),
        grading=stack.grading if scale == "frozen" else "closed_unresolved",
        target_coordinate=stack.target_coordinate,
        target_value=None,
        identity_receipt=receipt,
        diagnostic_only=diagnostic,
    )


def _norm_reconstruction_for(trace: Any, norm: str) -> Any:
    """Resolve a NormReconstruction for ``"final"`` or a module address."""

    if norm == "final":
        return resolve_lm_head(trace).norm_reconstruction()
    from ..semantic._norm_reconstruction import reconstruct_norm
    from ..semantic.facets import AbsenceReason
    from ..semantic.recipes._helpers import config_value

    try:
        record = trace.modules[norm]
    except (KeyError, ValueError, RuntimeError):
        refuse(  # noqa: B904 -- refuse() is NoReturn; exception chaining adds noise
            code="mi_target_unresolvable",
            message=f"norm= module {norm!r} is not in the trace.",
            remedy="pass a norm module address or 'final'",
            norm=norm,
        )
    view = record.facets
    if view is None or "normalized" not in list(view.keys()):
        refuse(
            code="mi_target_unresolvable",
            message=f"Module {norm!r} exposes no norm facets.",
            remedy="pass a module matched by a norm recipe (LayerNorm/RMSNorm family)",
            norm=norm,
        )
    norm_input = view["input"].value
    norm_output = view["normalized"].value
    gamma_param = next((p for p in (record.params or ()) if p.name == "weight"), None)
    beta_param = next((p for p in (record.params or ()) if p.name == "bias"), None)
    eps = config_value(record, "eps", "variance_epsilon")
    return reconstruct_norm(
        input=norm_input,
        output=norm_output,
        gamma=gamma_param.value if gamma_param is not None else None,
        beta=beta_param.value if beta_param is not None else None,
        eps=None if isinstance(eps, AbsenceReason) else float(eps),
        module_address=str(record.address),
        class_name=str(record.class_name),
    )


def stack_facet(trace: Any, name: str) -> ComponentStack:
    """Stack one facet across every module exposing it (the escape hatch).

    Rows read the facet per module in module order; the stack has no summing
    claim (``identity_receipt`` says ``gathered_reads``) and no target.
    """

    from ._heads import _facet_keys_or_none

    rows = []
    for module in trace.modules:
        keys = _facet_keys_or_none(module)
        if keys is None or name not in keys:
            continue
        facet = module.facets[name]
        value = facet.value if hasattr(facet, "value") else facet
        if not isinstance(value, torch.Tensor):
            continue
        rows.append(
            ComponentRow(
                coordinate=Coordinate(
                    label=str(module.address),
                    kind="writer",
                    module_address=str(module.address),
                    provenance=f"facet/{name}",
                ),
                value=value,
            )
        )
    if not rows:
        refuse(
            code="mi_payload_missing",
            message=f"No module exposes a readable {name!r} facet.",
            remedy="check tl.facets.facet_coverage(trace) for the available vocabulary",
            analysis="stack_facet",
            missing_sites=[],
        )
    return ComponentStack(
        tuple(rows),
        grading="complete",
        target_coordinate=Coordinate(label=f"facet:{name}", kind="writer"),
        target_value=None,
        identity_receipt={"check": "gathered_reads", "result": "no summing claim"},
    )


def full_decomposition(
    trace: Any,
    target: Any = "final",
    *,
    positions: Any = None,
) -> ComponentStack:
    """Every writer at HEAD grain: embeddings + per-head rows + MLP rows +
    verified per-layer bias constants (mikit roster ``full_decomposition``).

    Composition of :func:`residual_decomposition` (writer rows) and
    :func:`attention_head_contributions` (head splices): each attention
    writer whose value is payload-identical to its layer's ``attn_out`` is
    replaced by that layer's head rows + verified remainder; writers that
    cannot be spliced stay whole (disclosed). The bitwise gate cannot
    survive the splice (head summation re-associates), so the identity
    receipt carries the tolerance-checked form; DLA's own identity still
    verifies end-to-end.
    """

    base = residual_decomposition(trace, target, positions=None)
    heads = attention_head_contributions(trace, positions=None)
    by_layer: dict[str, list[ComponentRow]] = {}
    for row in heads.rows:
        address = row.coordinate.module_address or ""
        by_layer.setdefault(address, []).append(row)

    spliced: list[ComponentRow] = []
    splice_receipts: list[dict[str, Any]] = []
    for row in base.rows:
        attn_address = _owning_attention_address(row, by_layer)
        if attn_address is None:
            spliced.append(row)
            continue
        layer_rows = by_layer[attn_address]
        head_sum = torch.zeros_like(row.value)
        for head_row in layer_rows:
            head_sum = head_sum + head_row.value
        if torch.allclose(
            head_sum.to(torch.float32), row.value.to(torch.float32), atol=1e-4, rtol=1e-4
        ):
            spliced.extend(layer_rows)
            splice_receipts.append({"writer": row.coordinate.label, "spliced": attn_address})
        else:
            spliced.append(row)
            splice_receipts.append(
                {"writer": row.coordinate.label, "spliced": None, "reason": "value_mismatch"}
            )

    grading: StackGrading = "complete"
    if any(r.coordinate.kind == "unresolved_remainder" for r in spliced):
        grading = "closed_unresolved"
    elif any(r.coordinate.kind == "constant" for r in spliced):
        grading = "closed_constant"

    if positions is not None:
        index = torch.as_tensor(positions, dtype=torch.long)
        spliced = [
            ComponentRow(
                coordinate=r.coordinate,
                value=r.value.index_select(1, index),
                coefficient=r.coefficient,
            )
            for r in spliced
        ]
    return ComponentStack(
        tuple(spliced),
        grading=grading,
        target_coordinate=base.target_coordinate,
        target_value=base.target_value
        if positions is None or base.target_value is None
        else base.target_value.index_select(1, torch.as_tensor(positions, dtype=torch.long)),
        identity_receipt={
            "check": "head_splice_over_bitwise_base",
            "base": base.identity_receipt,
            "splices": splice_receipts,
        },
        diagnostic_only=grading == "closed_unresolved",
    )


def _owning_attention_address(
    row: ComponentRow, by_layer: dict[str, list[ComponentRow]]
) -> str | None:
    """Return the attention address whose head rows can replace this writer."""

    if row.coordinate.kind != "attention":
        return None
    module_address = row.coordinate.module_address or row.coordinate.label
    for address in sorted(by_layer, key=len, reverse=True):
        if module_address == address or module_address.startswith(address + "."):
            return address
    return None
