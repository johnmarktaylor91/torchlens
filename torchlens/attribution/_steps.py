"""Shared Integrated-Gradients step machinery: the alpha schedule and step batching.

The midpoint Riemann grid used to live inline at exactly two sites
(``_core.py`` and ``_layer.py``, both spelled ``(step + 0.5) / n_steps``).
This module is the ONE owner of that schedule (attrib memo D19): every
IG-family consumer (input IG, layer IG, layer conductance, the text helper)
reads its path points from :func:`_midpoint_schedule`, so a future quadrature
rule is one function body here, never API churn. Wave 1 deliberately ships
midpoint Riemann semantics only; captum's default Gauss-Legendre differs
(measured 22.5% at n=16 on resnet18), and captum oracle rows must request
``method="riemann_middle"`` (attrib memo D20).

The step runner (:func:`_run_step_batches`) stacks path points on the ordinary
batch axis for throughput (measured 4.25x unguarded, 3.03x with per-chunk
audit on real gpt2 CPU). Batching is a SPEED feature, not a memory feature:
larger chunks trade throughput for live activation memory. Correctness under
batching is guarded by a randomized, seeded, disclosed audit (attrib memo
D18) -- a sampled test, never a proof; the only proof is sequential execution.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from torchlens.attribution._result import AttributionError

_STEP_AUDIT_MODES = ("off", "per_call", "per_chunk")

# Cross-batch-size comparisons use a relative tolerance, never exact equality:
# reduction-order changes under stacking measured 1.3e-07..8.5e-06 relative on
# the funded real-model paths (attrib memo D16). The audit tolerance is
# dtype-aware: float32 transformer gradients show up to ~1e-4 infinity-norm-
# relative kernel noise between batched and unbatched matmuls, while the
# coupling classes the audit exists to catch deviate by >= 2e-2 (measured) --
# 1e-3 keeps a 20x detection margin and a 10x noise margin.
_STEP_AUDIT_RTOL = 1e-4
_STEP_AUDIT_RTOL_LOW_PRECISION = 1e-3
_STEP_AUDIT_ATOL = 1e-7


def _midpoint_alphas(n_steps: int) -> list[float]:
    """Return the midpoint Riemann interpolation coefficients for ``n_steps``.

    Parameters
    ----------
    n_steps
        Number of path points along the straight baseline-to-input path.

    Returns
    -------
    list[float]
        Interpolation coefficients ``(step + 0.5) / n_steps`` in step order.
    """

    return [(step + 0.5) / n_steps for step in range(n_steps)]


def _midpoint_schedule(n_steps: int) -> list[tuple[float, float]]:
    """Return ``(alpha, weight)`` pairs for the midpoint Riemann rule.

    This is the quadrature seam (attrib memo D19): a future rule replaces this
    one function body with rule-specific nodes and weights. Midpoint weights
    are uniform ``1 / n_steps``.

    Parameters
    ----------
    n_steps
        Number of path points.

    Returns
    -------
    list[tuple[float, float]]
        Path-point coefficients and quadrature weights, in step order.
    """

    weight = 1.0 / n_steps
    return [(alpha, weight) for alpha in _midpoint_alphas(n_steps)]


def _validate_step_batch_size(step_batch_size: int | None) -> int:
    """Validate the ``step_batch_size`` knob and normalize ``None`` to 1.

    Parameters
    ----------
    step_batch_size
        Requested number of path points stacked per forward/backward. ``None``
        and ``1`` both mean sequential execution (the general default; numbers
        never change under people's feet -- attrib memo D17).

    Returns
    -------
    int
        Validated chunk size, at least 1.

    Raises
    ------
    AttributionError
        If the value is not ``None`` or a positive integer.
    """

    if step_batch_size is None:
        return 1
    if isinstance(step_batch_size, bool) or not isinstance(step_batch_size, int):
        raise AttributionError(
            "step_batch_size must be None or a positive integer. "
            "Remedy: pass step_batch_size=None for sequential execution or a "
            "small positive integer such as 8.",
            code="step_batch_size_invalid",
        )
    if step_batch_size <= 0:
        raise AttributionError(
            f"step_batch_size must be positive; got {step_batch_size}. "
            "Remedy: pass step_batch_size=None for sequential execution or a "
            "small positive integer such as 8.",
            code="step_batch_size_invalid",
        )
    return step_batch_size


def _validate_step_audit(step_audit: str | None, chunk_size: int) -> str:
    """Validate the audit-ladder mode and resolve the batching-aware default.

    Parameters
    ----------
    step_audit
        Requested audit rung: ``None`` (resolve the default), ``"off"``
        (explicit expert choice, disclosed), ``"per_call"`` (one randomized
        ``(chunk, row)`` recomputed sequentially, the default whenever batching
        is on), or ``"per_chunk"`` (an independent random row per chunk).
    chunk_size
        Validated step chunk size; sequential execution needs no audit.

    Returns
    -------
    str
        Resolved audit mode.

    Raises
    ------
    AttributionError
        If the mode is not in the closed ladder vocabulary.
    """

    if step_audit is None:
        return "per_call" if chunk_size > 1 else "off"
    if step_audit not in _STEP_AUDIT_MODES:
        raise AttributionError(
            f"step_audit must be one of {_STEP_AUDIT_MODES}; got {step_audit!r}. "
            "Remedy: choose 'per_call' (default under batching), 'per_chunk', "
            "or the explicit expert choice 'off'.",
            code="step_audit_invalid",
        )
    return step_audit


def _chunk_steps(alphas: Sequence[float], chunk_size: int) -> list[list[float]]:
    """Split the path-point coefficients into execution chunks.

    Parameters
    ----------
    alphas
        Path-point coefficients in step order.
    chunk_size
        Number of path points stacked per forward/backward.

    Returns
    -------
    list[list[float]]
        Chunks in step order; the final chunk may be shorter.
    """

    return [list(alphas[start : start + chunk_size]) for start in range(0, len(alphas), chunk_size)]


@dataclass(frozen=True)
class _StepAuditRecord:
    """Disclosure record for one step-batching audit run (attrib memo D18).

    Attributes
    ----------
    mode
        Resolved audit rung: ``"off"``, ``"per_call"``, or ``"per_chunk"``.
    seed
        Seed of the audit's own RNG draw (never global torch state).
    audited_pairs
        ``(chunk_index, row_index)`` pairs recomputed sequentially, where
        ``row_index`` indexes the chunk's stacked path points.
    rtol
        Relative comparison tolerance.
    worst_deviation
        Largest observed relative deviation across audited pairs; ``0.0`` when
        nothing was audited.
    """

    mode: str
    seed: int | None
    audited_pairs: tuple[tuple[int, int], ...]
    rtol: float
    worst_deviation: float

    def to_extra(self) -> dict[str, Any]:
        """Return the JSON-friendly audit disclosure for ``result.extra``."""

        return {
            "mode": self.mode,
            "seed": self.seed,
            "audited_pairs": list(self.audited_pairs),
            "rtol": self.rtol,
            "worst_deviation": self.worst_deviation,
            "sampled_test_not_proof": True,
        }


def _relative_deviation(batched: tuple[Tensor, ...], sequential: tuple[Tensor, ...]) -> float:
    """Return the worst SCALE-RELATIVE deviation between two leaf tuples.

    The comparison is infinity-norm relative per leaf:
    ``max|batched - sequential| / (max|sequential| + atol)``. An elementwise
    relative comparison would amplify benign float kernel noise on
    near-zero gradient elements into false batching-audit failures (measured
    ~6e-3 elementwise on real float32 transformer gradients whose normwise
    deviation is ~1e-6), while genuine row COUPLING -- the audit's quarry --
    shifts gradients at the scale of the gradients themselves.

    Parameters
    ----------
    batched
        Per-leaf gradients computed inside a stacked chunk.
    sequential
        Per-leaf gradients recomputed unbatched for the same path point.

    Returns
    -------
    float
        Worst per-leaf infinity-norm-relative deviation.
    """

    worst = 0.0
    for batched_leaf, sequential_leaf in zip(batched, sequential, strict=True):
        scale = float(sequential_leaf.abs().max()) + _STEP_AUDIT_ATOL
        deviation = float((batched_leaf - sequential_leaf).abs().max()) / scale
        worst = max(worst, deviation)
    return worst


class _StepAuditor:
    """Randomized step-batching audit ladder (attrib memo D18).

    A fixed-index audit is defeatable by construction (an adversary coupling
    ``rows[1:]`` leaves row 0 decoupled: silent at 9.7e-08 while the batched
    attribution is 65.3% wrong); randomizing the audited ``(chunk, row)``
    costs nothing. The audit is still a sampled test with detection
    probability ``(k-1)/k`` per call, never a proof.

    Parameters
    ----------
    mode
        Resolved audit rung.
    n_chunks
        Number of execution chunks.
    chunk_sizes
        Number of stacked path points per chunk, in chunk order.
    seed
        Optional deterministic seed for the audit's own draw; when ``None`` a
        fresh seed is drawn from torch's global generator and disclosed.
    """

    def __init__(
        self,
        mode: str,
        n_chunks: int,
        chunk_sizes: Sequence[int],
        seed: int | None,
    ) -> None:
        """Draw the audited (chunk, row) pairs up front from a private RNG."""

        self.mode = mode
        self.rtol = _STEP_AUDIT_RTOL
        self.worst_deviation = 0.0
        self.audited_rows_by_chunk: dict[int, int] = {}
        if mode == "off" or n_chunks == 0 or max(chunk_sizes, default=1) <= 1:
            self.seed = seed
            return
        self.seed = seed if seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if mode == "per_call":
            # One randomized (chunk, row); the FIRST chunk is audited before
            # the rest is committed only when the draw lands on chunk 0, so
            # bias the draw toward chunk 0 by auditing chunk 0 whenever the
            # drawn chunk has size 1 (nothing to audit there).
            candidates = [index for index, size in enumerate(chunk_sizes) if size > 1]
            chunk_index = candidates[
                int(torch.randint(0, len(candidates), (1,), generator=generator).item())
            ]
            row_index = int(
                torch.randint(0, chunk_sizes[chunk_index], (1,), generator=generator).item()
            )
            self.audited_rows_by_chunk[chunk_index] = row_index
        else:
            for chunk_index, size in enumerate(chunk_sizes):
                if size <= 1:
                    continue
                self.audited_rows_by_chunk[chunk_index] = int(
                    torch.randint(0, size, (1,), generator=generator).item()
                )

    def row_for_chunk(self, chunk_index: int) -> int | None:
        """Return the audited row for ``chunk_index``, or ``None``."""

        return self.audited_rows_by_chunk.get(chunk_index)

    def check(
        self,
        chunk_index: int,
        row_index: int,
        batched: tuple[Tensor, ...],
        sequential: tuple[Tensor, ...],
    ) -> None:
        """Compare one batched row against its sequential recomputation.

        Parameters
        ----------
        chunk_index
            Chunk whose stacked gradients are being audited.
        row_index
            Stacked path-point row inside the chunk.
        batched
            Per-leaf gradients extracted from the stacked chunk.
        sequential
            Per-leaf gradients recomputed sequentially for the same alpha.

        Raises
        ------
        AttributionError
            If the deviation exceeds the audit tolerance.
        """

        if any(leaf.dtype != torch.float64 for leaf in sequential):
            self.rtol = max(self.rtol, _STEP_AUDIT_RTOL_LOW_PRECISION)
        deviation = _relative_deviation(batched, sequential)
        self.worst_deviation = max(self.worst_deviation, deviation)
        if deviation > self.rtol:
            raise AttributionError(
                f"step-batching audit failed at chunk {chunk_index}, row {row_index}: "
                f"batched gradients deviate {deviation:.3e} (relative) from the "
                f"sequential recomputation (tolerance {self.rtol:.1e}). The model "
                "couples stacked batch rows (train-mode BatchNorm is the realistic "
                "cause). Remedy: run with step_batch_size=1 (sequential execution "
                "is the only proof).",
                code="step_batch_audit_failed",
            )

    def record(self) -> _StepAuditRecord:
        """Return the frozen disclosure record for this audit run."""

        return _StepAuditRecord(
            mode=self.mode,
            seed=self.seed,
            audited_pairs=tuple(sorted(self.audited_rows_by_chunk.items())),
            rtol=self.rtol,
            worst_deviation=self.worst_deviation,
        )


def _split_stacked_output(output: Any, n_rows: int) -> list[Any]:
    """Split a stacked model-output tree into per-path-point logical outputs.

    Every tensor leaf must carry the stacked axis: its leading dimension must
    be an exact multiple of ``n_rows``. Non-tensor leaves are shared verbatim
    across the split outputs ONLY when they are scalars/None/str (values that
    cannot encode a per-example axis); any other leaf refuses.

    Parameters
    ----------
    output
        Model output produced by one stacked forward.
    n_rows
        Number of stacked path points.

    Returns
    -------
    list[Any]
        Per-path-point output trees, in stacking order.

    Raises
    ------
    AttributionError
        If any tensor leaf's leading dimension does not split evenly.
    """

    if isinstance(output, Tensor):
        if output.ndim == 0 or output.shape[0] % n_rows != 0:
            raise AttributionError(
                "callable targets require the stacked model output to split back "
                f"into {n_rows} logical examples, but a tensor leaf has shape "
                f"{tuple(output.shape)}. Remedy: run with step_batch_size=1, or "
                "use an int target.",
                code="step_batch_target_unsplittable",
            )
        return list(output.chunk(n_rows, dim=0))
    if isinstance(output, tuple) and hasattr(output, "_fields"):
        split_fields = [_split_stacked_output(item, n_rows) for item in output]
        return [type(output)(*(field[row] for field in split_fields)) for row in range(n_rows)]
    if isinstance(output, tuple):
        split_items = [_split_stacked_output(item, n_rows) for item in output]
        return [tuple(item[row] for item in split_items) for row in range(n_rows)]
    if isinstance(output, list):
        split_items = [_split_stacked_output(item, n_rows) for item in output]
        return [[item[row] for item in split_items] for row in range(n_rows)]
    if isinstance(output, dict):
        split_values = {key: _split_stacked_output(value, n_rows) for key, value in output.items()}
        return [{key: value[row] for key, value in split_values.items()} for row in range(n_rows)]
    if output is None or isinstance(output, (bool, int, float, complex, str)):
        return [output] * n_rows
    raise AttributionError(
        "callable targets require the stacked model output tree to split back "
        f"into logical examples, but a leaf of type {type(output).__name__} "
        "cannot be split. Remedy: run with step_batch_size=1, or use an int "
        "target.",
        code="step_batch_target_unsplittable",
    )


def _scalarize_stacked_output(
    output: Any,
    target: Any,
    n_rows: int,
    scalarize: Callable[[Any, Any], Tensor],
) -> Tensor:
    """Scalarize a stacked forward's output as the SUM of per-row targets.

    Int targets scalarize the stacked output directly (``output[..., t].sum()``
    already sums over the stacked axis). Callable targets receive logical
    per-example outputs, so the stacked tree is split first and the per-row
    scalars are summed; gradients per stacked row are unchanged because the
    rows are independent terms of the sum.

    Parameters
    ----------
    output
        Stacked model output.
    target
        Int class index or callable scalarizer.
    n_rows
        Number of stacked path points.
    scalarize
        The kit's single-output scalarizer (``_scalarize_output``).

    Returns
    -------
    Tensor
        One scalar whose gradient carries every stacked row's target gradient.
    """

    if isinstance(target, int) or n_rows == 1:
        return scalarize(output, target)
    per_row_outputs = _split_stacked_output(output, n_rows)
    scalars = [scalarize(row_output, target) for row_output in per_row_outputs]
    return torch.stack(scalars).sum()


__all__ = [
    "_StepAuditRecord",
    "_StepAuditor",
    "_chunk_steps",
    "_midpoint_alphas",
    "_midpoint_schedule",
    "_scalarize_stacked_output",
    "_split_stacked_output",
    "_validate_step_audit",
    "_validate_step_batch_size",
]
