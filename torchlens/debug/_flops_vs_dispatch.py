"""PyTorch dispatch-formula FLOP cross-check (torchnative 6.4 / W2.6).

Two independent formula registries disagreeing loudly is the FEATURE:
``torch.utils.flop_counter.FlopCounterMode`` counts by dispatched overload,
TorchLens counts by captured op rule, and this door runs both DETACHED and
reports them side by side -- signed deltas, per-side coverage ledgers,
never one filling the other's unknown cells, never presented as "measured
FLOPs" (both sides are analytic dispatch formulas).

The cross-check is NEVER an oracle: on torch releases through 2.13, FCM
counts ZERO FLOPs for all CPU-default attention
(``aten::_scaled_dot_product_flash_attention_for_cpu`` is absent from its
``flop_registry``; fixed on PyTorch main 2026-09-17, pytorch/pytorch#195801),
so a zero on either side is a COVERAGE fact, not a truth verdict.

Contract: the builder snapshots RNG and module training modes, runs the
TorchLens capture, restores state, runs the RAW model under
``FlopCounterMode(display=False)``, and compares outputs as its execution
witness -- divergence refuses typed (the two sides must describe the same
program before their formulas may be compared).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..errors import CaptureError


@dataclass(frozen=True)
class FlopsCrossCheck:
    """Side-by-side dispatch-formula evidence (never an oracle).

    ``native_total`` and ``torchlens_total`` are both fma=2 analytic
    counts; ``delta`` is signed (native - torchlens). A zero paired with a
    non-empty ``coverage_notes`` row is a registry gap, not a measurement.
    """

    torch_version: str
    convention: str
    torchlens_total: int | None
    native_total: int | None
    delta: int | None
    native_by_overload: dict[str, int] = field(default_factory=dict)
    torchlens_unknown_ops: tuple[str, ...] = ()
    torchlens_counted_ops: int = 0
    witness: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    coverage_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-able report."""

        return {
            "schema": "torchlens.flops_vs_dispatch.v1",
            "torch": self.torch_version,
            "convention": self.convention,
            "torchlens_total": self.torchlens_total,
            "native_total": self.native_total,
            "delta_native_minus_torchlens": self.delta,
            "native_by_overload": dict(self.native_by_overload),
            "torchlens_unknown_ops": list(self.torchlens_unknown_ops),
            "torchlens_counted_ops": self.torchlens_counted_ops,
            "witness": dict(self.witness),
            "provenance": dict(self.provenance),
            "coverage_notes": list(self.coverage_notes),
        }


def _first_tensor(value: Any) -> torch.Tensor | None:
    """Return the first tensor leaf of a model output structure."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def flops_vs_dispatch(
    model: Any,
    input_args: Any = None,
    input_kwargs: dict[str, Any] | None = None,
    *,
    sdpa_backend: str | None = None,
    **trace_kwargs: Any,
) -> FlopsCrossCheck:
    """Run the detached two-way FLOP cross-check on one model + input.

    Parameters
    ----------
    model:
        The model (any ``tl.trace``-able ``nn.Module``).
    input_args, input_kwargs:
        Forwarded to both executions identically.
    sdpa_backend:
        Provenance disclosure of the SELECTED SDPA backend when the caller
        forced one (e.g. ``"math"`` under ``sdpa_kernel(SDPBackend.MATH)``,
        which changes the EXECUTED PROGRAM: formula validation only, never
        a timing comparison). ``None`` means default selection.
    **trace_kwargs:
        Extra ``tl.trace`` kwargs for the TorchLens side (``save=`` defaults
        to the metadata tier).

    Returns
    -------
    FlopsCrossCheck
        Side-by-side totals, per-overload native ledger, TorchLens unknown
        ops, witness facts, and coverage notes.

    Raises
    ------
    CaptureError
        ``flops_crosscheck_witness_failed`` when the two executions diverge
        in output -- the formulas would then describe different programs
        and no comparison is honest.
    """

    from torch.utils.flop_counter import FlopCounterMode

    from .. import user_funcs
    from ..report._compute_truth import compute_aggregation

    rng_state = torch.get_rng_state()
    training_flags = {name: module.training for name, module in model.named_modules()}

    trace_kwargs.setdefault("save", None)
    trace = user_funcs.trace(model, input_args, input_kwargs, **trace_kwargs)
    try:
        aggregation = compute_aggregation(trace)
        torchlens_total = int(aggregation.partition_total)
        unknown_ops = tuple(
            str(row.op_label)
            for row in aggregation.rows
            if row.kind == "op" and row.coverage_class == "unknown"
        )
        counted = sum(
            1 for row in aggregation.rows if row.kind == "op" and row.flops_fma2 is not None
        )
        captured_output = _first_tensor(
            trace.output_ops[0].out if getattr(trace, "output_ops", None) else None
        )
        captured_digest = None if captured_output is None else captured_output.detach().clone()
    finally:
        trace.cleanup()

    # Restore the snapshot so the raw run replays the same program.
    torch.set_rng_state(rng_state)
    for name, module in model.named_modules():
        if name in training_flags:
            module.train(training_flags[name])

    args: tuple[Any, ...]
    if input_args is None:
        args = ()
    elif isinstance(input_args, (list, tuple)):
        args = tuple(input_args)
    else:
        args = (input_args,)
    kwargs = dict(input_kwargs or {})
    with FlopCounterMode(display=False) as counter:
        raw_output = model(*args, **kwargs)
    native_total = int(counter.get_total_flops())
    # FCM's per-module ledger is HIERARCHICAL ("Global" plus one row per
    # module path, each containing its descendants); summing across keys
    # would double-count, so the overload ledger reads the Global row only.
    by_overload: dict[str, int] = {
        str(op_packet): int(value)
        for op_packet, value in counter.get_flop_counts().get("Global", {}).items()
    }

    raw_tensor = _first_tensor(raw_output)
    output_agrees: bool | None = None
    if captured_digest is not None and raw_tensor is not None:
        output_agrees = captured_digest.shape == raw_tensor.shape and bool(
            torch.allclose(
                captured_digest.float(), raw_tensor.detach().float(), rtol=1e-4, atol=1e-5
            )
        )
    if output_agrees is False:
        raise CaptureError(
            "flops_vs_dispatch execution witness failed: the captured and "
            "raw executions produced different outputs, so their dispatch "
            "formulas describe different programs and no comparison is "
            "honest (stochastic layers need a fixed seed or eval mode).",
            code="flops_crosscheck_witness_failed",
            remedy=(
                "Run in eval mode or seed the model so both executions "
                "follow one program, then re-run the cross-check."
            ),
        )

    notes: list[str] = []
    if native_total == 0 and torchlens_total and torchlens_total > 0:
        notes.append(
            "native FCM counted ZERO FLOPs while TorchLens counted "
            f"{torchlens_total}: a flop_registry coverage gap (on CPU-default "
            "attention the missing overload is "
            "aten::_scaled_dot_product_flash_attention_for_cpu, known missing in "
            "torch releases through 2.13 and fixed on PyTorch main "
            "2026-09-17, pytorch/pytorch#195801) -- a coverage fact, never a "
            "truth verdict"
        )
    if unknown_ops:
        notes.append(
            f"TorchLens has no cost rule for {len(unknown_ops)} op(s); native "
            "values NEVER fill TorchLens unknown cells"
        )
    return FlopsCrossCheck(
        torch_version=torch.__version__,
        convention="fma2 (one multiply-accumulate = 2 FLOPs, both sides)",
        torchlens_total=torchlens_total,
        native_total=native_total,
        delta=(
            None
            if torchlens_total is None or native_total is None
            else native_total - torchlens_total
        ),
        native_by_overload=by_overload,
        torchlens_unknown_ops=unknown_ops,
        torchlens_counted_ops=counted,
        witness={
            "output_agrees": output_agrees,
            "rng_restored": True,
            "training_modes_restored": True,
        },
        provenance={
            "sdpa_backend": sdpa_backend or "default",
            "execution_note": (
                "detached: one TorchLens capture, state restore, one raw "
                "FlopCounterMode run; two separately identified executions"
            ),
        },
        coverage_notes=tuple(notes),
    )


__all__ = ["FlopsCrossCheck", "flops_vs_dispatch"]
