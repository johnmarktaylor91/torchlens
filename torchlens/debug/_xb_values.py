"""Compare the recorded values of one paired module call.

Values are read from each call's recorded output ops (``ModuleCall.output_ops``)
through each trace's payload codec, so a saved and reloaded trace compares the same
as a live one. Shape alignment uses a fixed list, decided from shapes and the module
class only, never from values:

- a leading size-one axis present on one side only is dropped (disclosed);
- channels-last to channels-first (rank 3 to 5) applies only when both module
  classes are convolution, pooling or channel-normalization modules;
- when more than one candidate fits (equal channel and spatial sizes) the pair is
  ``layout_unresolved`` and not evaluated.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .._io.payload_codec import get_payload_codec
from .._utils._stochastic_funcs import _STOCHASTIC_FUNC_NAMES
from ._common import _op_from_label, _op_label, _safe_out
from ._compare_values import _align_for_compare, _compare_unavailable_reason
from ._xb_tolerance import Tolerance, judge, tolerance_for

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace

EQUAL = "equal"
DIFFERENT = "different"
SHAPE_INCOMPATIBLE = "shape-incompatible"
UNAVAILABLE = "unavailable"
NOT_EVALUATED = "not-evaluated"
LAYOUT_UNRESOLVED = "layout_unresolved"

#: Module classes whose outputs may legitimately be channels-last on one backend.
_LAYOUT_CLASSES = re.compile(r"Conv|Pool|BatchNorm|InstanceNorm|GroupNorm")
_CHANNELS_LAST = "channels-last to channels-first"


@dataclass(frozen=True)
class ValueOptions:
    """How values are judged for one comparison.

    Parameters
    ----------
    cross_backend:
        Whether the two runs come from different backends.
    rtol, atol:
        User tolerance overrides, or ``None`` for the policy default.
    scale:
        Reference scale of the cross-backend absolute floor (see :func:`input_scale`).
    """

    cross_backend: bool
    rtol: float | None
    atol: float | None
    scale: float = 1.0


@dataclass(frozen=True)
class ValueResult:
    """What the values of one paired call showed.

    Parameters
    ----------
    outcome:
        ``equal``, ``different``, ``shape-incompatible``, ``unavailable`` or
        ``not-evaluated``.
    max_abs, mean_abs:
        Absolute differences of the compared leaves (``None`` when not compared).
    rtol, atol:
        Effective tolerance (``None`` when not compared).
    transform:
        Applied shape transforms, ``""`` when none.
    reason:
        Row disclosure (dtype widening, half-precision floor, unavailability).
    dtype_match:
        Whether the recorded dtypes agree (``None`` when unread).
    """

    outcome: str
    max_abs: float | None = None
    mean_abs: float | None = None
    rtol: float | None = None
    atol: float | None = None
    transform: str = ""
    reason: str = ""
    dtype_match: bool | None = None


def read_values(trace: Trace, labels: tuple[str, ...]) -> list[tuple[Any, str | None]]:
    """Return ``(tensor or None, unavailable reason)`` for each op label."""

    codec = get_payload_codec(str(trace._provenance_backend))
    values: list[tuple[Any, str | None]] = []
    for label in labels:
        op = _op_from_label(trace, label)
        if op is None:
            values.append((None, "unrecorded"))
            continue
        value, reason = _safe_out(op, codec)
        reason = reason or _compare_unavailable_reason(value)
        values.append((value if reason is None else None, reason))
    return values


def input_scale(trace: Trace, labels: tuple[str, ...]) -> float:
    """The rms of a call's recorded floating inputs, or 1.0 when none is available or all zero."""

    tensors = [
        value.detach().reshape(-1).to("cpu", torch.float64)
        for value, _why in read_values(trace, labels)
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.numel()
    ]
    if not tensors:
        return 1.0
    rms = float(torch.cat(tensors).square().mean().sqrt())
    return rms if rms > 0.0 and rms == rms and rms != float("inf") else 1.0


def drew_random(trace: Trace, op: Any) -> bool:
    """Whether ``op`` is a random producer whose output depended on a draw.

    A dropout-family op that returned its input unchanged (eval mode, or ``p=0``) drew
    nothing; every other random producer counts.
    """

    name = str(getattr(op, "func_name", "") or "").rstrip("_")
    if name not in _STOCHASTIC_FUNC_NAMES:
        return False
    parents = list(getattr(op, "parents", None) or ())
    if "dropout" not in name or not parents:
        return True
    (out, _why_out), (source, _why_source) = read_values(trace, (_op_label(op), str(parents[0])))
    if not isinstance(out, torch.Tensor) or not isinstance(source, torch.Tensor):
        return True
    return out.shape != source.shape or not torch.equal(out, source)


def _squeeze_leading(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Drop a leading size-one axis present on one side only, when that makes shapes equal."""

    if a.dim() == b.dim() + 1 and a.shape[0] == 1 and a.shape[1:] == b.shape:
        return a[0], b, "dropped leading size-one axis (a)"
    if b.dim() == a.dim() + 1 and b.shape[0] == 1 and b.shape[1:] == a.shape:
        return a, b[0], "dropped leading size-one axis (b)"
    return a, b, ""


def _layout_candidates(
    a: torch.Tensor, b: torch.Tensor, justified: bool
) -> list[tuple[str, Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]]:
    """Return every listed transform whose result makes the shapes equal."""

    candidates: list[
        tuple[str, Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]
    ] = []
    if a.shape == b.shape:
        candidates.append(("", lambda x, y: (x, y)))
    if not justified or not 3 <= a.dim() <= 5 or a.dim() != b.dim():
        return candidates
    if b.movedim(-1, 1).shape == a.shape:
        candidates.append((f"{_CHANNELS_LAST} (b)", lambda x, y: (x, y.movedim(-1, 1))))
    if a.movedim(-1, 1).shape == b.shape and a.shape != b.shape:
        candidates.append((f"{_CHANNELS_LAST} (a)", lambda x, y: (x.movedim(-1, 1), y)))
    return candidates


def _align_shapes(
    a: torch.Tensor, b: torch.Tensor, justified: bool
) -> tuple[torch.Tensor, torch.Tensor, str] | ValueResult:
    """Apply the squeeze and at most one layout transform, or say why not."""

    a, b, squeeze = _squeeze_leading(a, b)
    candidates = _layout_candidates(a, b, justified)
    if len(candidates) > 1:
        names = ", ".join(name or "identity" for name, _fn in candidates)
        return ValueResult(NOT_EVALUATED, transform=squeeze, reason=f"{LAYOUT_UNRESOLVED}: {names}")
    if not candidates:
        shapes = f"{tuple(a.shape)} vs {tuple(b.shape)}"
        return ValueResult(SHAPE_INCOMPATIBLE, transform=squeeze, reason=f"shapes {shapes}")
    name, apply = candidates[0]
    a, b = apply(a, b)
    return a, b, "; ".join(part for part in (squeeze, name) if part)


def compare_leaf(
    a: torch.Tensor, b: torch.Tensor, *, justified: bool, options: ValueOptions
) -> ValueResult:
    """Compare one pair of recorded tensors under the tolerance policy."""

    dtype_match = a.dtype == b.dtype
    shaped = _align_shapes(a, b, justified)
    if isinstance(shaped, ValueResult):
        return ValueResult(**{**shaped.__dict__, "dtype_match": dtype_match})
    a_s, b_s, transform = shaped
    tol: Tolerance = tolerance_for(
        a.dtype,
        b.dtype,
        cross_backend=options.cross_backend,
        rtol=options.rtol,
        atol=options.atol,
        scale=options.scale,
    )
    aligned = _align_for_compare(a_s, b_s)
    if aligned is None:  # pragma: no cover - shapes were made equal above
        return ValueResult(SHAPE_INCOMPATIBLE, transform=transform, dtype_match=dtype_match)
    cmp_a, cmp_b, note = aligned
    max_abs, mean_abs, agree = judge(cmp_a, cmp_b, tol)
    reason = "; ".join(part for part in (note, tol.note) if part)
    return ValueResult(
        EQUAL if agree else DIFFERENT,
        max_abs,
        mean_abs,
        tol.rtol,
        tol.atol,
        transform,
        reason,
        dtype_match,
    )


def layout_justified(class_a: str, class_b: str) -> bool:
    """Whether both module classes justify a channels-last transform."""

    return bool(_LAYOUT_CLASSES.search(class_a)) and bool(_LAYOUT_CLASSES.search(class_b))


_SEVERITY = (DIFFERENT, SHAPE_INCOMPATIBLE, NOT_EVALUATED, UNAVAILABLE, EQUAL)


def combine(results: list[ValueResult], disclosure: str) -> ValueResult:
    """Fold per-leaf results into one row result: the most severe outcome wins.

    ``different`` outranks everything, so a difference in any returned leaf is
    never hidden by agreement in another.
    """

    if not results:
        return ValueResult(UNAVAILABLE, reason=disclosure or "no recorded output")
    worst = min(results, key=lambda result: _SEVERITY.index(result.outcome))
    compared = [result for result in results if result.max_abs is not None]
    max_abs = max((r.max_abs for r in compared if r.max_abs is not None), default=None)
    reason = "; ".join(part for part in (disclosure, worst.reason) if part)
    return ValueResult(**{**worst.__dict__, "max_abs": max_abs, "reason": reason})
