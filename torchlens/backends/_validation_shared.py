"""Backend-neutral helpers shared by the preview backends' validation modules.

Validation-side helpers only, no backend imports, safe to import from any
preview backend. The capture-side helper families live in
``backends/_finalize.py``.

The float replay tolerance derivation lives here as the ONE reviewed policy:
every preview backend's replay and validation oracle derives its band from
the dtype's eps (plus a scalar NaN/inf rule) instead of a hardcoded decimal
pair. A band change is a reviewed tolerance change with a rationale; any
future one edits exactly this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np


def ops_by_label(trace: Any) -> dict[str, Any]:
    """Return materialized trace operations keyed by all known labels.

    Parameters
    ----------
    trace:
        Materialized TorchLens trace.

    Returns
    -------
    dict[str, Any]
        Operations keyed by raw, layer, and pass labels.

    Notes
    -----
    Key precedence is load-bearing under recurrence grouping: a group
    leader's RAW label doubles as the shared ``layer_label`` of every later
    pass, so naive last-writer insertion would silently rebind a capture
    record's raw label to the wrong pass. Raw labels are the immutable
    capture identity and always win; pass labels are unique; the ambiguous
    layer label resolves to its first pass.
    """

    result: dict[str, Any] = {}
    for op in trace._layer_list:
        layer_label = getattr(op, "layer_label", None)
        if isinstance(layer_label, str) and layer_label not in result:
            result[layer_label] = op
    for op in trace._layer_list:
        label = getattr(op, "label", None)
        if isinstance(label, str):
            result[label] = op
    for op in trace._layer_list:
        label_raw = getattr(op, "_label_raw", None)
        if isinstance(label_raw, str):
            result[label_raw] = op
    return result


class _FloatInfo(NamedTuple):
    """Minimal finfo shim for float dtypes numpy's ``finfo`` cannot describe."""

    eps: float
    tiny: float


#: bfloat16 keeps fp32's exponent range with a 7-bit mantissa.
_EXTENDED_FLOAT_INFO: dict[str, _FloatInfo] = {
    "bfloat16": _FloatInfo(eps=2.0**-7, tiny=float(np.finfo(np.float32).tiny)),
}


def replay_payload(op: Any, hidden: Mapping[str, Any] | None = None) -> Any:
    """Return the payload a replay check reads for ``op``, or ``None`` when none was kept.

    A filtered or shaped capture hides some public payloads (``save=`` dropped
    them, or ``activation_transform`` without ``save_raw_activations`` cleared
    ``out``) but keeps the raw copies replay-visible in the trace's
    ``_selective_save_hidden_payloads``, keyed by raw and final op labels.

    Parameters
    ----------
    op:
        Materialized operation.
    hidden:
        Hidden payloads by label; ``None`` reads them from the op's owning trace.

    Returns
    -------
    Any
        The public ``op.out`` when saved, else the hidden raw copy, else ``None``
        (callers fail closed on ``None``).
    """

    if bool(getattr(op, "has_saved_activation", False)):
        output = getattr(op, "out", None)
        if output is not None:
            return output
    if hidden is None:
        owner = getattr(op, "_source_trace_or_none", None)
        trace = owner() if callable(owner) else None
        hidden = getattr(trace, "_selective_save_hidden_payloads", None) or {}
    for label in (getattr(op, "_label_raw", None), getattr(op, "label", None)):
        if isinstance(label, str) and label in hidden:
            return hidden[label]
    return None


def float_replay_tolerances(finfo: Any) -> tuple[float, float]:
    """Derive the dtype-honest ``(rtol, atol)`` replay band from a float finfo.

    The ONE reviewed tolerance policy for every preview backend's replay and
    validation oracle, replacing the dtype-blind fp32 decimal
    pair (rtol 1e-5 / atol 1e-6) that was wrong in both directions: fp64
    corruption ~4.5e9 of its own ULPs read as agreement, while a legitimate
    one-ULP fp16 storage-rounding difference false-failed.

    * Accumulating dtypes (eps <= fp32's): the legacy fp32 relative band
      rescaled by the eps ratio, so every dtype gets the SAME strictness
      measured in its own ULPs (fp32 keeps exactly the historical 1e-5).
    * Storage-rounding dtypes (eps > fp32's): values compute in a wider dtype
      and round ONCE to storage, so the legitimate replay difference is a few
      storage ULPs (4-ULP headroom).
    * The absolute term only absorbs jitter at the bottom of the representable
      range (the relative band applied to the smallest normal value); the
      former 1e-6 floor blessed TOTAL corruption of every element below it.

    Parameters
    ----------
    finfo
        ``finfo`` of the payload dtype (``np.finfo`` or ``jnp.finfo``;
        component finfo for complex dtypes).

    Returns
    -------
    tuple[float, float]
        Derived ``(rtol, atol)`` pair.
    """

    eps32 = float(np.finfo(np.float32).eps)
    eps = float(finfo.eps)
    if eps > eps32:
        rtol = 4.0 * eps
    else:
        rtol = 1e-5 * eps / eps32
    return rtol, rtol * float(finfo.tiny)


def float_replay_tolerances_for_dtype_name(dtype_name: str) -> tuple[float, float]:
    """Derive the replay band for a backend dtype STRING (tinygrad spelling).

    Parameters
    ----------
    dtype_name
        Backend dtype string, e.g. ``"dtypes.half"`` / ``"float64"``.

    Returns
    -------
    tuple[float, float]
        Derived ``(rtol, atol)`` pair for the named float family; the fp32
        band when the family is not recognized (the pre-R17 status quo for
        unknown floats, never looser).
    """

    for name, info in _EXTENDED_FLOAT_INFO.items():
        if name in dtype_name:
            return float_replay_tolerances(info)
    if "half" in dtype_name or "float16" in dtype_name:
        return float_replay_tolerances(np.finfo(np.float16))
    if "double" in dtype_name or "float64" in dtype_name:
        return float_replay_tolerances(np.finfo(np.float64))
    return float_replay_tolerances(np.finfo(np.float32))


def scalar_replay_close(left: Any, right: Any, rtol: float, atol: float) -> bool:
    """Compare two scalar float payloads under the derived replay band.

    Carries the array oracles' NaN/inf doctrine to scalar payload streams
    (tinygrad): an identical NaN pattern is agreement, NaN-vs-number fails,
    and equal infinities agree (``abs(inf - inf)`` is NaN, so the naive
    band test false-failed byte-identical non-finite replays).

    Parameters
    ----------
    left
        Left scalar payload.
    right
        Right scalar payload.
    rtol
        Relative tolerance from ``float_replay_tolerances*``.
    atol
        Absolute tolerance from ``float_replay_tolerances*``.

    Returns
    -------
    bool
        True when the scalars agree under the band.
    """

    left_f = float(left)
    right_f = float(right)
    if math.isnan(left_f) or math.isnan(right_f):
        return math.isnan(left_f) and math.isnan(right_f)
    if math.isinf(left_f) or math.isinf(right_f):
        return left_f == right_f
    return abs(left_f - right_f) <= atol + rtol * abs(right_f)
