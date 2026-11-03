"""Per-neuron MLP contributions (mikit D7 identity gate at neuron grain).

The MLP output projection is linear in its hidden activation, so an MLP's
output decomposes exactly into one row per hidden neuron:
``output = sum_n hidden[..., n] * W_out[n, :] + b_out``. Rows read the
LITERAL operand of the captured output-projection op (the tensor the model
multiplied), never a re-derived activation, and the projection weight comes
from the dataflow-anchored parameter in math orientation (Conv1D ``[in,
out]`` and Linear ``[out, in]`` decided by module class, as in
:func:`head_weight_views`).

The remainder ``output - sum(neuron rows)`` is checked, never assumed: it
is labelled the projection bias ONLY when it provably equals that bias
within the reconstruction tolerance, dropped when it is provably zero, and
otherwise kept as an ``mlp_remainder`` row that grades the stack
CLOSED-UNRESOLVED (diagnostic only; DLA refuses it). The rows plus the
verified constant therefore sum to the captured MLP output under the D7
gate, the same identity receipt head rows carry.

Memory: ``d_mlp`` rows of ``[batch, seq, d_model]`` each (3072 rows per
gpt2-small layer). ``neurons=`` keeps named neurons and folds the rest into
one exact ``other_neurons`` row; ``onto=`` contracts every row against
direction vectors BEFORE any full row exists.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .._semantic.tolerances import within_reconstruction_tolerance
from ._errors import refuse
from ._heads import _facet_keys_or_none, _shape_row_value
from ._params import _math_weight, _orientation, _projection_for, _weight_params
from ._records import ComponentRow, ComponentStack, Coordinate, StackGrading

__all__ = ["mlp_neuron_contributions"]

#: Facets an MLP module must expose for neuron rows.
_MLP_KEYS = frozenset({"down_out", "output"})


@dataclass(frozen=True)
class _NeuronOperands:
    """One MLP's output-projection operands, math-oriented."""

    hidden: torch.Tensor  # [..., d_mlp], the literal projection input
    weight: torch.Tensor  # [d_mlp, d_model]
    bias: torch.Tensor | None  # [d_model]
    output: torch.Tensor  # [..., d_model], the captured MLP output
    projection_address: str
    hidden_source: str


def _mlp_modules(trace: Any) -> list[Any]:
    """Return MLP module records exposing the down-projection vocabulary."""

    modules = []
    for module in trace.modules:
        keys = _facet_keys_or_none(module)
        if keys is not None and keys >= _MLP_KEYS:
            modules.append(module)
    return modules


def _select_mlps(modules: list[Any], layers: Any) -> list[Any]:
    """Filter MLP modules by executed-order index or module address."""

    if layers is None:
        return modules
    selected = []
    wanted = [layers] if isinstance(layers, (int, str)) else list(layers)
    for entry in wanted:
        if isinstance(entry, int):
            if not -len(modules) <= entry < len(modules):
                refuse(
                    code="mi_neuron_geometry_unavailable",
                    message=f"layers= index {entry} outside the {len(modules)} MLP modules found.",
                    remedy=f"pass indices inside [-{len(modules)}, {len(modules)})",
                    n_layers=len(modules),
                )
            selected.append(modules[entry])
            continue
        match = [m for m in modules if str(getattr(m, "address", "")) == str(entry)]
        if not match:
            refuse(
                code="mi_neuron_geometry_unavailable",
                message=f"layers= address {entry!r} is not an MLP module with "
                "down_out/output facets.",
                remedy="pass an MLP module address (e.g. transformer.h.0.mlp)",
                address=str(entry),
            )
        selected.extend(match)
    return selected


def _facet_tensor(module: Any, name: str) -> torch.Tensor | None:
    """Return a facet's tensor payload, or ``None`` when not readable."""

    try:
        facet = module.facets[name]
    except KeyError:
        return None
    value = facet.value if hasattr(facet, "value") else facet
    return value if isinstance(value, torch.Tensor) else None


def _literal_hidden(trace: Any, use_label: str, d_mlp: int, lead: tuple[int, ...]) -> Any:
    """Return the saved projection input reshaped to ``lead + (d_mlp,)``, or None."""

    try:
        use_op = trace.ops[use_label]
    except (KeyError, ValueError):
        return None
    want = 1
    for size in lead:
        want *= int(size)
    for parent in getattr(use_op, "parents", ()) or ():
        try:
            payload = getattr(trace.ops[str(parent)], "out", None)
        except (KeyError, ValueError):
            continue
        if not isinstance(payload, torch.Tensor) or payload.dim() < 2:
            continue
        if int(payload.shape[-1]) == d_mlp and payload.numel() == want * d_mlp:
            return payload.reshape(*lead, d_mlp)
    return None


def _neuron_operands(trace: Any, module: Any) -> _NeuronOperands:
    """Resolve one MLP's hidden operand, output weight, bias and output."""

    address = str(getattr(module, "address", ""))
    output = _facet_tensor(module, "output")
    if output is None:
        refuse(
            code="mi_payload_missing",
            message=f"The MLP output at {address!r} is not readable.",
            remedy="capture with the MLP output saved (layers_to_save='all' or a "
            "retention_plan covering the MLP)",
            analysis="mlp_neuron_contributions",
            missing=[f"{address}.output"],
        )
    weight_params = _weight_params(trace, address)
    if not weight_params:
        refuse(
            code="mi_neuron_geometry_unavailable",
            message=f"No 2D weight parameters found inside {address!r}.",
            remedy="pass the MLP module itself (not a parent container)",
            address=address,
        )
    proj_module, param, _ancestry = _projection_for(
        trace, module.facets["down_out"], address, weight_params
    )
    class_name = str(getattr(proj_module, "class_name", ""))
    weight = _math_weight(param, _orientation(class_name, tuple(param.shape), output.shape[-1]))
    d_mlp = int(weight.shape[0])
    bias_param = next(
        (p for p in (getattr(proj_module, "params", ()) or ()) if p.name == "bias"), None
    )
    use_label = str(param.used_by_ops[0]) if getattr(param, "used_by_ops", None) else ""
    lead = tuple(int(size) for size in output.shape[:-1])
    hidden = _literal_hidden(trace, use_label, d_mlp, lead)
    source = "captured_projection_input"
    if hidden is None:
        candidate = _facet_tensor(module, "intermediate")
        if candidate is not None and tuple(candidate.shape) == lead + (d_mlp,):
            hidden, source = candidate, "intermediate_facet"
    if hidden is None:
        refuse(
            code="mi_payload_missing",
            message=f"The input of the output projection {use_label!r} in {address!r} "
            "was not saved, and no intermediate facet of the right shape is readable.",
            remedy="capture with the MLP interior saved (layers_to_save='all' or a "
            "retention_plan covering the MLP)",
            analysis="mlp_neuron_contributions",
            missing=[use_label],
        )
    return _NeuronOperands(
        hidden=hidden,
        weight=weight,
        bias=bias_param.value if bias_param is not None else None,
        output=output,
        projection_address=str(getattr(proj_module, "address", "")),
        hidden_source=source,
    )


def _check_remainder(ops: _NeuronOperands) -> tuple[str, torch.Tensor | None]:
    """Classify ``output - hidden @ W`` as bias / zero / unresolved.

    Returns ``(status, remainder_or_None)``; the budget's magnitude term is
    the accumulated |addend| bound ``|hidden| @ |W| + |output|``.
    """

    neuron_sum = ops.hidden @ ops.weight
    remainder = ops.output - neuron_sum
    magnitude = ops.hidden.detach().abs() @ ops.weight.detach().abs() + ops.output.detach().abs()
    reduction_length = int(ops.weight.shape[0]) + 1
    if ops.bias is not None and within_reconstruction_tolerance(
        remainder,
        ops.bias.expand_as(remainder),
        magnitude=magnitude,
        reduction_length=reduction_length,
    ):
        return "verified_bias", None
    if within_reconstruction_tolerance(
        remainder,
        torch.zeros_like(remainder),
        magnitude=magnitude,
        reduction_length=reduction_length,
    ):
        return "verified_zero", None
    return "unproven", remainder


def _neuron_index(neurons: Any, d_mlp: int, address: str) -> list[int]:
    """Normalize ``neurons=`` to sorted unique indices in ``[0, d_mlp)``."""

    if neurons is None:
        return list(range(d_mlp))
    wanted = [neurons] if isinstance(neurons, int) else list(neurons)
    normalized = []
    for entry in wanted:
        if not isinstance(entry, int) or isinstance(entry, bool) or not -d_mlp <= entry < d_mlp:
            refuse(
                code="mi_neuron_index_invalid",
                message=f"neurons= entry {entry!r} is not a neuron index of {address!r} "
                f"(d_mlp={d_mlp}).",
                remedy=f"pass integer indices inside [-{d_mlp}, {d_mlp})",
                d_mlp=d_mlp,
            )
        normalized.append(entry % d_mlp)
    return sorted(set(normalized))


def _row(label: str, address: str, provenance: str, kind: str, value: torch.Tensor) -> ComponentRow:
    """Build one MLP-side component row."""

    return ComponentRow(
        coordinate=Coordinate(
            label=label, kind=kind, module_address=address, provenance=provenance
        ),
        value=value,
    )


def _layer_neuron_rows(
    trace: Any, module: Any, neurons: Any, direction: torch.Tensor | None, positions: Any
) -> tuple[list[ComponentRow], torch.Tensor, dict[str, Any]]:
    """Build one MLP's neuron rows (+ folded rest, + constant) and its receipt."""

    address = str(getattr(module, "address", ""))
    ops = _neuron_operands(trace, module)
    status, remainder = _check_remainder(ops)
    d_mlp = int(ops.weight.shape[0])
    chosen = _neuron_index(neurons, d_mlp, address)
    weight = ops.weight if direction is None else ops.weight.float() @ direction.float()
    hidden = ops.hidden if direction is None else ops.hidden.float()
    provenance = f"computed: hidden[n] * W_out[n] ({ops.hidden_source}; {ops.projection_address})"
    rows = [
        _row(
            f"{address}.neuron{n}",
            address,
            provenance,
            "mlp",
            _shape_row_value(hidden[..., n : n + 1] * weight[n], None, positions),
        )
        for n in chosen
    ]
    rest = sorted(set(range(d_mlp)) - set(chosen))
    if rest:
        index = torch.as_tensor(rest, dtype=torch.long, device=hidden.device)
        folded = hidden.index_select(-1, index) @ weight.index_select(0, index)
        rows.append(
            _row(
                f"{address}.other_neurons",
                address,
                f"computed: exact sum of the {len(rest)} unselected neurons",
                "mlp",
                _shape_row_value(folded, None, positions),
            )
        )
    if status == "verified_bias" and ops.bias is not None:
        constant = ops.bias.expand_as(ops.output)
        rows.append(
            _row(
                f"{ops.projection_address}.bias",
                address,
                "verified_constant",
                "constant",
                _shape_row_value(constant, direction, positions),
            )
        )
    elif remainder is not None:
        rows.append(
            _row(
                f"{address}.mlp_remainder",
                address,
                "computed_exact_difference; not proven to be the projection bias",
                "unresolved_remainder",
                _shape_row_value(remainder, direction, positions),
            )
        )
    receipt = {
        "layer": address,
        "remainder": status,
        "hidden_source": ops.hidden_source,
        "d_mlp": d_mlp,
        "neurons": len(chosen),
    }
    return rows, ops.output, receipt


def mlp_neuron_contributions(
    trace: Any,
    layers: Any = None,
    *,
    neurons: Any = None,
    onto: torch.Tensor | None = None,
    positions: Any = None,
) -> ComponentStack:
    """Per-neuron MLP contribution rows + verified constants (mikit D7 gate).

    Parameters
    ----------
    trace:
        A finished torchlens trace with the MLP interior saved.
    layers:
        MLP-module selector: ``None`` (all), executed-order index or address,
        or a list of them.
    neurons:
        Optional neuron indices kept as their own rows in every selected
        layer; the other neurons fold into one exact ``other_neurons`` row
        per layer, so the stack still sums to the MLP output.
    onto:
        Optional direction vector(s) ``[d_model]`` or ``[d_model, n]``; rows
        are contracted BEFORE any full row exists (contracted rows are
        scores, not residual writers).
    positions:
        Optional position indices; rows are sliced on axis 1 after the
        remainder proof runs on full tensors.

    Returns
    -------
    ComponentStack
        Neuron rows (+ folded rest, + per-layer verified bias constants) in
        layer order; ``sum()`` equals the summed captured MLP outputs of the
        selected layers within the reconstruction tolerance. Grading is
        ``complete`` / ``closed_constant`` when every remainder is proven,
        else ``closed_unresolved`` and diagnostic-only.
    """

    modules = _mlp_modules(trace)
    if not modules:
        refuse(
            code="mi_neuron_geometry_unavailable",
            message="No module in the trace exposes the MLP down-projection vocabulary "
            "(down_out/output).",
            remedy="capture with the builtin MLP recipes (import torchlens._semantic); "
            "models without a recognised MLP have no neuron rows",
        )
    selected = _select_mlps(modules, layers)
    direction = None
    if onto is not None:
        direction = onto.reshape(-1, 1) if onto.dim() == 1 else onto

    rows: list[ComponentRow] = []
    grading: StackGrading = "complete"
    receipts = []
    target_total: torch.Tensor | None = None
    for module in selected:
        layer_rows, output, receipt = _layer_neuron_rows(
            trace, module, neurons, direction, positions
        )
        rows.extend(layer_rows)
        receipts.append(receipt)
        if receipt["remainder"] == "unproven":
            grading = "closed_unresolved"
        elif receipt["remainder"] == "verified_bias" and grading == "complete":
            grading = "closed_constant"
        target_total = output if target_total is None else target_total + output

    if target_total is None:
        refuse(
            code="mi_neuron_geometry_unavailable",
            message="No MLP layer was selected.",
            remedy="pass layers=None (all) or a non-empty selector",
        )
    target = _shape_row_value(target_total, direction, positions)
    return ComponentStack(
        tuple(rows),
        grading=grading,
        target_coordinate=Coordinate(
            label="sum(mlp output over selected layers)", kind="writer", provenance="captured"
        ),
        target_value=target,
        identity_receipt={
            "check": "neuron_rows_plus_constant_vs_mlp_output",
            "result": "unproven"
            if grading == "closed_unresolved"
            else "within_reconstruction_tolerance",
            "per_layer": receipts,
            "contracted": direction is not None,
        },
        diagnostic_only=grading == "closed_unresolved",
    )
