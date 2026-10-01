"""Precision bisector: locate where reduced-precision numerics first diverge.

``bisect_precision`` runs the SAME forward twice under identical seeding --
once at the model's native dtypes, once with floating state and inputs cast
to a high-precision reference dtype (fp64 by default) -- then walks the
captured ops in execution order and reports the first op whose native output
separates from the reference beyond tolerance. Every spelling here is
DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from ._common import _compute_ops, _op_label, _safe_out, _source_line, _tensor_unavailable_reason

if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True)
class PrecisionRow:
    """Per-op comparison between the native and reference runs.

    Parameters
    ----------
    label:
        Pass-qualified op label.
    func_name:
        Op function name.
    native_dtype:
        Native output dtype string.
    max_abs_err:
        Maximum elementwise absolute error against the reference output.
    max_rel_err:
        Maximum elementwise relative error against the reference output.
    rtol:
        Relative tolerance this op was judged against.
    atol:
        Absolute tolerance this op was judged against.
    diverged:
        Whether the op crossed tolerance (or mismatched shape/nonfiniteness).
    stochastic:
        Whether the op consumes random draws. Random kernels may consume RNG
        differently across dtypes even under identical seeding, so a
        divergence AT a stochastic op is usually mask mismatch, not precision
        loss; the summary message teaches this rather than excluding the op.
    source_line:
        ``"file:line"`` for the op's call site when available.
    """

    label: str
    func_name: str
    native_dtype: str
    max_abs_err: float
    max_rel_err: float
    rtol: float
    atol: float
    diverged: bool
    stochastic: bool
    source_line: str | None


@dataclass(frozen=True)
class BisectPrecisionResult:
    """Result of :func:`bisect_precision`.

    Parameters
    ----------
    found:
        Whether any compared op diverged beyond tolerance.
    label:
        First divergent op's pass-qualified label, or ``None``.
    func_name:
        First divergent op's function name, or ``None``.
    source_line:
        First divergent op's call site, or ``None``.
    max_abs_err:
        First divergent op's maximum absolute error, or ``None``.
    max_rel_err:
        First divergent op's maximum relative error, or ``None``.
    rows:
        Every compared op in execution order, including agreeing ops.
    skipped:
        ``(label, reason)`` pairs for ops that could not be compared
        (unsaved, non-floating, missing in the reference run, ...). Skips are
        disclosed, never silently dropped: a divergence inside a skipped
        region cannot be excluded.
    message:
        Human-readable summary.
    """

    found: bool
    label: str | None
    func_name: str | None
    source_line: str | None
    max_abs_err: float | None
    max_rel_err: float | None
    rows: tuple[PrecisionRow, ...]
    skipped: tuple[tuple[str, str], ...]
    message: str


def bisect_precision(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[Any, Any] | None = None,
    *,
    reference_dtype: torch.dtype = torch.float64,
    rtol: float | None = None,
    atol: float | None = None,
    seed: int = 0,
) -> BisectPrecisionResult:
    """Locate the first op whose native-precision output drifts from fp64.

    Both runs execute on fresh deep copies through the shared isolated-rerun
    harness (each copy is released from any inherited TorchLens preparation,
    inputs are cloned, and Python/NumPy/torch RNG state is preserved and
    identically seeded), so the caller's model, buffers, and global RNG are
    untouched, an ALREADY-TRACED source model works, and the two forwards are
    directly comparable. Default tolerances derive from each
    op's NATIVE output dtype -- ``rtol = eps ** 0.5`` and ``atol = eps * 10``
    (fp32: ``rtol ~ 3.4e-4``; fp16: ``rtol ~ 3.1e-2``) -- so "diverged" means
    "lost meaningfully more precision than the dtype itself explains", not
    "differs at all" (a native-precision run always differs from fp64
    somewhere). Pass explicit ``rtol=``/``atol=`` to override for every op.

    Parameters
    ----------
    model:
        Source model. It is deep-copied twice and never mutated.
    input_args:
        Forward input value or positional-argument list/tuple, exactly as
        accepted by ``tl.trace``.
    input_kwargs:
        Optional forward keyword arguments.
    reference_dtype:
        High-precision reference dtype for the side-by-side run.
    rtol:
        Fixed relative tolerance; ``None`` derives per-op from native dtype.
    atol:
        Fixed absolute tolerance; ``None`` derives per-op from native dtype.
    seed:
        Seed applied to both runs inside the RNG fork.

    Returns
    -------
    BisectPrecisionResult
        First divergence plus the full per-op comparison table.
    """

    from ..options import CaptureOptions
    from ._rerun import isolated_capture

    _refuse_unsupported_reference_device(model, reference_dtype)
    capture = CaptureOptions(layers_to_save="all")
    native_trace = reference_trace = None
    try:
        native_trace = isolated_capture(
            model,
            input_args,
            input_kwargs,
            seed=seed,
            capture=capture,
        )
        reference_trace = isolated_capture(
            model,
            _cast_tree(input_args, reference_dtype),
            _cast_tree(input_kwargs, reference_dtype),
            seed=seed,
            prepare=lambda run_model: run_model.to(reference_dtype),
            capture=capture,
        )
        return _compare_traces(native_trace, reference_trace, rtol=rtol, atol=atol)
    finally:
        for captured in (native_trace, reference_trace):
            if captured is not None:
                captured.cleanup()


def _compare_traces(
    native_trace: Any, reference_trace: Any, *, rtol: float | None, atol: float | None
) -> BisectPrecisionResult:
    """Walk both traces in execution order and build the comparison result."""

    reference_ops = {_op_label(op): op for op in _compute_ops(reference_trace)}
    rows: list[PrecisionRow] = []
    skipped: list[tuple[str, str]] = []
    first: PrecisionRow | None = None
    for op in _compute_ops(native_trace):
        label = _op_label(op)
        native_out, reason = _safe_out(op)
        reason = reason or _tensor_unavailable_reason(native_out)
        if reason is not None:
            skipped.append((label, reason))
            continue
        reference_op = reference_ops.get(label)
        if reference_op is None:
            skipped.append((label, "missing in reference run (dtype-dependent path?)"))
            continue
        reference_out, reference_reason = _safe_out(reference_op)
        reference_reason = reference_reason or _tensor_unavailable_reason(reference_out)
        if reference_reason is not None:
            skipped.append((label, f"reference {reference_reason}"))
            continue
        # _tensor_unavailable_reason returned None for both payloads, so each is
        # a dense floating tensor; cast() records that narrowing for mypy.
        row = _compare_op(
            op,
            cast(torch.Tensor, native_out),
            cast(torch.Tensor, reference_out),
            rtol=rtol,
            atol=atol,
        )
        if row is None:
            skipped.append((label, "empty tensor"))
            continue
        rows.append(row)
        if row.diverged and first is None:
            first = row
    return _build_result(first, rows, skipped)


def _compare_op(
    op: Any,
    native_out: torch.Tensor,
    reference_out: torch.Tensor,
    *,
    rtol: float | None,
    atol: float | None,
) -> PrecisionRow | None:
    """Compare one op's native output against the reference output."""

    label = _op_label(op)
    eps = float(torch.finfo(native_out.dtype).eps)
    op_rtol = rtol if rtol is not None else eps**0.5
    op_atol = atol if atol is not None else eps * 10.0
    func_name = str(getattr(op, "func_name", ""))

    def _row(max_abs_err: float, max_rel_err: float, diverged: bool) -> PrecisionRow:
        """Build a row with the shared op metadata."""

        return PrecisionRow(
            label=label,
            func_name=func_name,
            native_dtype=str(native_out.dtype),
            max_abs_err=max_abs_err,
            max_rel_err=max_rel_err,
            rtol=op_rtol,
            atol=op_atol,
            diverged=diverged,
            stochastic=func_name.rstrip("_") in _STOCHASTIC_FUNC_NAMES,
            source_line=_source_line(op),
        )

    if native_out.shape != reference_out.shape:
        return _row(float("inf"), float("inf"), True)
    if native_out.numel() == 0:
        return None
    native64 = native_out.detach().to(torch.float64)
    reference64 = reference_out.detach().to(torch.float64)
    native_finite = torch.isfinite(native64)
    reference_finite = torch.isfinite(reference64)
    nonfinite_mismatch = not torch.equal(native_finite, reference_finite)
    both_finite = native_finite & reference_finite
    if both_finite.any():
        diff = (native64 - reference64).abs()
        tiny = torch.finfo(torch.float64).tiny
        rel = diff / reference64.abs().clamp_min(tiny)
        max_abs_err = float(diff[both_finite].max())
        max_rel_err = float(rel[both_finite].max())
        crossed = bool(
            (diff[both_finite] > op_atol + op_rtol * reference64.abs()[both_finite]).any()
        )
    else:
        max_abs_err = max_rel_err = 0.0
        crossed = False
    return _row(max_abs_err, max_rel_err, crossed or nonfinite_mismatch)


def _build_result(
    first: PrecisionRow | None,
    rows: list[PrecisionRow],
    skipped: list[tuple[str, str]],
) -> BisectPrecisionResult:
    """Assemble the result object with an honest summary message."""

    skip_note = (
        f" {len(skipped)} op(s) could not be compared (see result.skipped); a divergence "
        "inside a skipped region cannot be excluded."
        if skipped
        else ""
    )
    if first is not None:
        where = f" ({first.source_line})" if first.source_line else ""
        stochastic_note = (
            " NOTE: this op consumes random draws, and random kernels may consume "
            "RNG differently across dtypes even under identical seeding -- this is "
            "usually mask mismatch, not precision loss. Re-run with the model in "
            "eval mode to bisect the deterministic computation."
            if first.stochastic
            else ""
        )
        message = (
            f"First precision divergence at {first.label}{where}: "
            f"max_abs_err={first.max_abs_err:.3e}, max_rel_err={first.max_rel_err:.3e} "
            f"(rtol={first.rtol:.1e}, atol={first.atol:.1e}). Compared ops before this "
            f"point agree with the high-precision reference.{stochastic_note}{skip_note}"
        )
    elif not rows:
        message = "No ops could be compared against the high-precision reference." + skip_note
    else:
        message = (
            f"No precision divergence beyond tolerance across {len(rows)} compared "
            f"op(s).{skip_note}"
        )
    return BisectPrecisionResult(
        found=first is not None,
        label=None if first is None else first.label,
        func_name=None if first is None else first.func_name,
        source_line=None if first is None else first.source_line,
        max_abs_err=None if first is None else first.max_abs_err,
        max_rel_err=None if first is None else first.max_rel_err,
        rows=tuple(rows),
        skipped=tuple(skipped),
        message=message,
    )


#: Function names (trailing underscores normalized away) whose outputs depend
#: on random draws; cross-dtype RNG consumption is not comparable for these.
_STOCHASTIC_FUNC_NAMES = frozenset(
    {
        "alpha_dropout",
        "bernoulli",
        "binomial",
        "dropout",
        "dropout1d",
        "dropout2d",
        "dropout3d",
        "exponential",
        "feature_alpha_dropout",
        "feature_dropout",
        "geometric",
        "multinomial",
        "normal",
        "poisson",
        "rand",
        "rand_like",
        "randint",
        "randint_like",
        "randn",
        "randn_like",
        "randperm",
        "rrelu",
        "rrelu_with_noise",
        "uniform",
    }
)


def _cast_tree(value: Any, dtype: torch.dtype) -> Any:
    """Return a copy of an input tree with floating tensors cast to a dtype."""

    if isinstance(value, torch.Tensor):
        if torch.is_floating_point(value):
            return value.detach().clone().to(dtype)
        return value
    if isinstance(value, dict):
        return {key: _cast_tree(item, dtype) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cast_tree(item, dtype) for item in value)
    if isinstance(value, list):
        return [_cast_tree(item, dtype) for item in value]
    return value


def _refuse_unsupported_reference_device(model: Any, reference_dtype: torch.dtype) -> None:
    """Refuse device/dtype combinations the reference run cannot execute."""

    if reference_dtype != torch.float64:
        return
    devices = {str(parameter.device.type) for parameter in model.parameters()}
    if "mps" in devices:
        raise RuntimeError(
            "bisect_precision cannot run an fp64 reference on MPS (float64 is "
            "unsupported there). Move the model to CPU for the bisect, or pass a "
            "reference_dtype MPS supports."
        )
