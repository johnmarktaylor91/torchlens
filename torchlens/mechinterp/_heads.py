"""Per-query-head attention contributions (mikit D10; TLens
``stack_head_results``).

Rows are ``z_h @ W_O_h`` per QUERY head, read from the validated ``result``
facet (a computed view on EVERY implementation -- no model materializes
per-head outputs). The remainder is the ACTUAL DIFFERENCE
``attn_out - sum(head rows)`` -- always right by construction -- and is
labelled ``output_bias`` ONLY when it provably equals the projection bias
and the graph shows no post-projection scaling/dropout effect; otherwise it
stays ``attention_remainder`` with the observed status in provenance, and
the stack grades CLOSED-UNRESOLVED (banned from DLA -- honesty over reach).
Never spread across heads.

``onto=`` contracts each head row against direction vectors BEFORE the
``[layers, heads, batch, seq, d_model]`` stack can exist (452 MB on
gpt2-small at S=1024 without it -- that number is why the argument exists).

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from typing import Any

import torch

from ..semantic.tolerances import within_reconstruction_tolerance
from ._errors import refuse
from ._records import ComponentRow, ComponentStack, Coordinate, StackGrading

__all__ = ["attention_head_contributions"]


def _facet_keys_or_none(module: Any) -> set[str] | None:
    """Return a module's facet-key set, or ``None`` when unreadable.

    Multi-call modules and foreign recipe errors read as ABSENCE here (a
    scan must never die on one exotic module); refusals stay the caller's
    job. Shared by every kit-side facet scan.
    """

    try:
        view = module.facets
    except Exception:  # noqa: BLE001 -- any per-module failure is absence for a scan
        return None
    if view is None:
        return None
    return set(view.keys())


def _attention_modules(trace: Any) -> list[Any]:
    """Return attention module records (those exposing the head vocabulary)."""

    modules = []
    for module in trace.modules:
        keys = _facet_keys_or_none(module)
        if keys is not None and {"result", "attn_out", "n_q_heads", "d_head"} <= keys:
            modules.append(module)
    return modules


def _select_layers(modules: list[Any], layers: Any) -> list[Any]:
    """Filter attention modules by index or address."""

    if layers is None:
        return modules
    selected = []
    wanted = [layers] if isinstance(layers, (int, str)) else list(layers)
    for entry in wanted:
        if isinstance(entry, int):
            if not -len(modules) <= entry < len(modules):
                refuse(
                    code="mi_head_geometry_unavailable",
                    message=f"layers= index {entry} outside the {len(modules)} attention "
                    "modules found.",
                    remedy=f"pass indices inside [-{len(modules)}, {len(modules)})",
                    n_layers=len(modules),
                )
            selected.append(modules[entry])
        else:
            match = [m for m in modules if str(getattr(m, "address", "")) == str(entry)]
            if not match:
                refuse(
                    code="mi_head_geometry_unavailable",
                    message=f"layers= address {entry!r} is not an attention module with "
                    "head facets.",
                    remedy="pass an attention module address exposing result/attn_out",
                    address=str(entry),
                )
            selected.extend(match)
    return selected


def _result_tensor(module: Any) -> torch.Tensor:
    """Read the validated per-head result facet ``[b, pos, head, d_model]``."""

    facet = module.facets["result"]
    value = facet.value if hasattr(facet, "value") else facet
    if not isinstance(value, torch.Tensor):
        refuse(
            code="mi_payload_missing",
            message=f"The per-head result view at "
            f"{getattr(module, 'address', '?')} is not readable "
            f"({type(value).__name__}).",
            remedy="capture with the attention interior saved (retention_plan or "
            "layers_to_save='all'); fused SDPA needs "
            "capture=CaptureOptions(save_arg_values=True)",
            analysis="attention_head_contributions",
        )
    return value


def _remainder_row(
    module: Any, result: torch.Tensor, attn_out: torch.Tensor
) -> tuple[ComponentRow | None, bool]:
    """Build the verified remainder row; return (row_or_None, proven).

    The equality budget's magnitude term is the per-element accumulated
    |addend| bound ``sum_h |result_h| + |attn_out|`` -- the post-sum
    ``|sum_h result_h|`` understates it exactly where heads cancel, which
    made the proof input-dependently flaky (measured before this bound).
    """

    head_sum = result.sum(dim=2)
    remainder = attn_out - head_sum
    bias_param = None
    for child_address in [str(getattr(module, "address", ""))] + list(
        getattr(module, "address_children", ()) or ()
    ):
        try:
            candidate = module.trace.modules[child_address]
        except (KeyError, ValueError, RuntimeError):
            continue
        for param in getattr(candidate, "params", ()) or ():
            if (
                param.name == "bias"
                and len(param.shape) == 1
                and int(param.shape[0]) == int(attn_out.shape[-1])
            ):
                bias_param = param
    magnitude = result.detach().abs().sum(dim=2) + attn_out.detach().abs()
    reduction_length = int(result.shape[2]) * int(result.shape[3])
    if bias_param is not None:
        bias = bias_param.value
        if within_reconstruction_tolerance(
            remainder,
            bias.expand_as(remainder),
            magnitude=magnitude,
            reduction_length=reduction_length,
        ):
            return (
                ComponentRow(
                    coordinate=Coordinate(
                        label=f"{bias_param.module_address}.bias",
                        kind="constant",
                        # The ATTENTION address, so consumers (full_decomposition)
                        # can group a layer's head rows + constant as one splice.
                        module_address=str(getattr(module, "address", "")),
                        provenance="verified_constant",
                    ),
                    value=bias.expand_as(remainder),
                ),
                True,
            )
    if within_reconstruction_tolerance(
        remainder,
        torch.zeros_like(remainder),
        magnitude=magnitude,
        reduction_length=reduction_length,
    ):
        return None, True
    return (
        ComponentRow(
            coordinate=Coordinate(
                label=f"{getattr(module, 'address', '?')}.attention_remainder",
                kind="unresolved_remainder",
                module_address=str(getattr(module, "address", "")),
                provenance="computed_exact_difference; not proven to be the projection "
                "bias (post-projection scaling/dropout or unproven path)",
            ),
            value=remainder,
        ),
        False,
    )


def _shape_row_value(
    value: torch.Tensor, direction: torch.Tensor | None, positions: Any
) -> torch.Tensor:
    """Apply the contract-before-expand contraction and position slicing."""

    if direction is not None:
        value = value.to(torch.float32) @ direction.to(torch.float32)
    if positions is not None:
        index = torch.as_tensor(positions, dtype=torch.long, device=value.device)
        value = value.index_select(1, index)
    return value


def _layer_rows(
    module: Any, direction: torch.Tensor | None, positions: Any
) -> tuple[list[ComponentRow], torch.Tensor, bool, dict[str, str]]:
    """Build one layer's head rows (+ remainder) and its receipt."""

    view = module.facets
    n_q_heads = int(view["n_q_heads"])
    n_kv_heads = int(view["n_kv_heads"])
    group = max(1, n_q_heads // max(1, n_kv_heads))
    address = str(getattr(module, "address", ""))
    result = _result_tensor(module)  # [b, pos, head, d_model]
    attn_out_facet = view["attn_out"]
    attn_out = attn_out_facet.value if hasattr(attn_out_facet, "value") else attn_out_facet
    remainder_row, proven = _remainder_row(module, result, attn_out)
    rows = [
        ComponentRow(
            coordinate=Coordinate(
                label=f"{address}.head{head}",
                op_label=None,
                site_key=None,
                kind="attention",
                module_address=address,
                provenance=f"computed_view/result; kv_group={head // group}",
            ),
            value=_shape_row_value(result[:, :, head, :], direction, positions),
        )
        for head in range(n_q_heads)
    ]
    if remainder_row is not None:
        rows.append(
            ComponentRow(
                coordinate=remainder_row.coordinate,
                value=_shape_row_value(remainder_row.value, direction, positions),
            )
        )
    receipt = {
        "layer": address,
        "remainder": "verified_bias"
        if (proven and remainder_row is not None)
        else ("verified_zero" if proven else "unproven"),
    }
    return rows, attn_out, proven, receipt


def attention_head_contributions(
    trace: Any,
    layers: Any = None,
    *,
    onto: torch.Tensor | None = None,
    positions: Any = None,
) -> ComponentStack:
    """Per-query-head contribution rows + verified remainders (mikit D10).

    Parameters
    ----------
    trace:
        A finished torchlens trace.
    layers:
        Attention-module selector: ``None`` (all), index/address, or a list.
    onto:
        Optional direction vector(s) ``[d_model]`` or ``[d_model, n]``: rows
        are contracted BEFORE any full stack exists (contract-before-expand;
        the memory number lives in the module docstring). Contracted rows
        are score tensors, not residual writers.
    positions:
        Optional position indices; rows are sliced on axis 1 after the
        remainder proof runs on full tensors.

    Returns
    -------
    ComponentStack
        Head rows (+ per-layer verified bias constants) in execution order.
        GQA metadata rides each coordinate's ``expert``-adjacent fields:
        the kv group index is ``head // (n_q_heads // n_kv_heads)``.
    """

    modules = _attention_modules(trace)
    if not modules:
        refuse(
            code="mi_head_geometry_unavailable",
            message="No module in the trace exposes the attention head vocabulary "
            "(result/attn_out/n_q_heads/d_head).",
            remedy="capture with the builtin attention recipes (import torchlens.semantic); "
            "attention-free architectures have no head rows",
        )
    selected = _select_layers(modules, layers)

    direction = None
    if onto is not None:
        direction = onto.reshape(-1, 1) if onto.dim() == 1 else onto

    rows: list[ComponentRow] = []
    grading: StackGrading = "complete"
    receipts = []
    target_total: torch.Tensor | None = None
    for module in selected:
        layer_rows, attn_out, proven, receipt = _layer_rows(module, direction, positions)
        rows.extend(layer_rows)
        receipts.append(receipt)
        if not proven:
            grading = "closed_unresolved"
        elif receipt["remainder"] == "verified_bias" and grading == "complete":
            grading = "closed_constant"
        target_total = attn_out if target_total is None else target_total + attn_out

    if target_total is None:
        refuse(
            code="mi_head_geometry_unavailable",
            message="No attention layer was selected.",
            remedy="pass layers=None (all) or a non-empty selector",
        )
    if direction is not None:
        target_total = target_total.to(torch.float32) @ direction.to(torch.float32)
    if positions is not None:
        index = torch.as_tensor(positions, dtype=torch.long, device=target_total.device)
        target_total = target_total.index_select(1, index)
    return ComponentStack(
        tuple(rows),
        grading=grading,
        target_coordinate=Coordinate(
            label="sum(attn_out over selected layers)", kind="writer", provenance="computed"
        ),
        target_value=target_total,
        identity_receipt={
            "check": "head_rows_plus_remainder_vs_attn_out",
            "result": "exact_by_construction",
            "per_layer": receipts,
            "contracted": direction is not None,
        },
        diagnostic_only=grading == "closed_unresolved",
    )
