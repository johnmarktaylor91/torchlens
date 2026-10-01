"""The two-term compute record (A5/A6): true MACs, never flops // 2.

Split from ``capture/flops.py`` (size ratchet); the cost TABLES stay there,
the record layer lives here. Spellings DOCUMENTED-UNSTABLE pending
naming-session ratification; the C02 FactCore substrate consumes this.
"""

from dataclasses import dataclass
from typing import Any

from .flops import (
    _CUSTOM_OP_RULES,
    ELEMENTWISE_FLOPS,
    SPECIALTY_HANDLERS,
    ZERO_ARITHMETIC_CONSTRUCTIONS,
    ZERO_FLOPS_OPS,
    CapturedArgs,
    ParamShapes,
    Shape,
    _numel,
    _prod,
    _safe_shape,
    compute_forward_flops,
)

# ============================================================================
# The two-term compute record (A5/A6; DOCUMENTED-UNSTABLE spellings pending
# naming-session ratification)
# ============================================================================
#
# True MACs are NEVER flops // 2: a biased Linear(8,16) has 256 MACs where
# flops//2 reads 272 (the bias adds are not multiply-accumulates), and a ReLU
# has ZERO MACs (the true answer), never "half its FLOPs". Every counted op
# splits into an FMA term and an everything-else term:
#
#   flops(fma=2) = 2 * fma_macs + other_flops   (the stored convention)
#   flops(fma=1) =     fma_macs + other_flops
#   true MACs    =     fma_macs


@dataclass(frozen=True)
class ComputeRecord:
    """Two-term per-op compute record.

    Parameters
    ----------
    fma_macs:
        Multiply-accumulate count, or ``None`` when the split is unknown.
    other_flops:
        Non-FMA floating-point operations under the per-op cost model, or
        ``None`` when the split is unknown.
    mac_applicability:
        ``"mac"`` (FMA-family op), ``"non_mac"`` (no multiply-accumulate
        structure; ``fma_macs`` is an exact zero), or ``"unknown"``.
    evidence:
        ``"formula_exact"`` (shape-derived exact count), ``"estimated"``
        (documented per-element cost model), or ``"unknown"``.
    reason:
        Optional human-readable provenance note.
    """

    fma_macs: int | None
    other_flops: int | None
    mac_applicability: str
    evidence: str
    reason: str | None = None

    def flops(self, *, fma: int = 2) -> int | None:
        """Collapse the record to a FLOP count under an FMA convention."""

        if self.fma_macs is None or self.other_flops is None:
            return None
        if fma not in (1, 2):
            raise ValueError(f"fma must be 1 or 2, got {fma!r}")
        return fma * self.fma_macs + self.other_flops


#: FMA-family names whose totals are ALL-MAC (no separate bias/other term).
_ALL_MAC_NAMES = frozenset(
    {
        "mm",
        "matmul",
        "__matmul__",
        "__rmatmul__",
        "bmm",
        "multi_dot",
        "chain_matmul",
    }
)

#: FMA-family names whose totals carry an unconditional added-matrix term of
#: numel(output) (addmm-style fused bias).
_ADDMM_NAMES = frozenset({"addmm", "addmm_", "addbmm", "addbmm_", "baddbmm", "baddbmm_"})

#: FMA-family names whose totals carry a bias term of numel(output) exactly
#: when a second parameter shape (the bias) is present.
_BIASED_MAC_NAMES = frozenset(
    {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "convolution",
        "_convolution",
        "_convolution_mode",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    }
)

#: Every FMA-family (MAC-applicable) op name.
MAC_FAMILY_NAMES = frozenset(
    _ALL_MAC_NAMES | _ADDMM_NAMES | _BIASED_MAC_NAMES | {"einsum", "scaled_dot_product_attention"}
)

#: Zero-arithmetic tensor CONSTRUCTIONS (costreport D3): ops that allocate or
#: fill a tensor without floating-point arithmetic. Classified zero-by-rule
#: per op NAME with two-sided witnesses in the test suite; anything ambiguous
#: (rand/randn draw RNG, linspace interpolates) STAYS unknown -- a confident
#: wrong zero is worse than an honest unknown.


def _sdpa_other_flops(q_shape: Shape | None, k_shape: Shape | None) -> int | None:
    """Return the sdpa scale+softmax (non-MAC) term: 6 * batch * S_q * S_k."""

    if not q_shape or not k_shape or len(q_shape) < 2 or len(k_shape) < 2:
        return None
    batch = _prod(q_shape[:-2]) if len(q_shape) > 2 else 1
    return 6 * batch * q_shape[-2] * k_shape[-2]


def _split_all_mac(
    total_flops: int,
    output_shape: Shape | None,
    param_shapes: ParamShapes,
    qk_shapes: tuple[Shape | None, Shape | None] | None,
) -> tuple[int, int] | None:
    """Split a pure-matmul-family total (all-MAC, no other term)."""

    del output_shape, param_shapes, qk_shapes
    if total_flops % 2:
        return None
    return (total_flops // 2, 0)


def _split_addmm(
    total_flops: int,
    output_shape: Shape | None,
    param_shapes: ParamShapes,
    qk_shapes: tuple[Shape | None, Shape | None] | None,
) -> tuple[int, int] | None:
    """Split an addmm-family total (unconditional numel(output) added-matrix term)."""

    del param_shapes, qk_shapes
    other = _numel(output_shape)
    if output_shape is None or total_flops < other or (total_flops - other) % 2:
        return None
    return ((total_flops - other) // 2, other)


def _split_biased(
    total_flops: int,
    output_shape: Shape | None,
    param_shapes: ParamShapes,
    qk_shapes: tuple[Shape | None, Shape | None] | None,
) -> tuple[int, int] | None:
    """Split a linear/conv-family total (numel(output) bias term iff biased)."""

    del qk_shapes
    has_bias = len(param_shapes) > 1
    if has_bias and output_shape is None:
        return None
    other = _numel(output_shape) if has_bias else 0
    if total_flops < other or (total_flops - other) % 2:
        return None
    return ((total_flops - other) // 2, other)


def _split_einsum(
    total_flops: int,
    output_shape: Shape | None,
    param_shapes: ParamShapes,
    qk_shapes: tuple[Shape | None, Shape | None] | None,
) -> tuple[int, int] | None:
    """Split an einsum total.

    A contraction totals ``2*out*contraction`` (all-MAC); a contraction-free
    einsum totals exactly ``numel(output)`` (an elementwise product, no MACs).
    ``2*out*c >= 2*out > out`` for ``c >= 1``, so the cases never collide.
    """

    del param_shapes, qk_shapes
    out_numel = _numel(output_shape)
    if output_shape is not None and total_flops == out_numel:
        return (0, total_flops)
    if total_flops % 2:
        return None
    return (total_flops // 2, 0)


def _split_sdpa(
    total_flops: int,
    output_shape: Shape | None,
    param_shapes: ParamShapes,
    qk_shapes: tuple[Shape | None, Shape | None] | None,
) -> tuple[int, int] | None:
    """Split an sdpa total (matmul MACs + the scale/softmax other term)."""

    del output_shape, param_shapes
    if qk_shapes is None:
        return None
    sdpa_other = _sdpa_other_flops(qk_shapes[0], qk_shapes[1])
    if sdpa_other is None or total_flops < sdpa_other or (total_flops - sdpa_other) % 2:
        return None
    return ((total_flops - sdpa_other) // 2, sdpa_other)


#: Per-name MAC-family splitters. Each mirrors its handler's own formula
#: structure (every handler builds its total as ``2*MACs + other`` in
#: source), so ``2*fma + other == total`` holds by construction; callers
#: still verify it and fail closed to an unknown split on any mismatch.
_MAC_SPLITTERS: dict[str, Any] = {
    **dict.fromkeys(_ALL_MAC_NAMES, _split_all_mac),
    **dict.fromkeys(_ADDMM_NAMES, _split_addmm),
    **dict.fromkeys(_BIASED_MAC_NAMES, _split_biased),
    "einsum": _split_einsum,
    "scaled_dot_product_attention": _split_sdpa,
}

_USER_RULE_RECORD = ComputeRecord(
    fma_macs=None,
    other_flops=None,
    mac_applicability="unknown",
    evidence="estimated",
    reason="user-registered rule (register_op_rule): MAC split not declared",
)


def _split_mac_family(
    func_name: str,
    total_flops: int,
    output_shape: Shape | None,
    param_shapes: ParamShapes,
    qk_shapes: tuple[Shape | None, Shape | None] | None,
) -> tuple[int, int] | None:
    """Split an FMA-family op's total FLOPs into ``(fma_macs, other_flops)``."""

    splitter = _MAC_SPLITTERS.get(func_name)
    if splitter is None:
        return None
    return splitter(total_flops, output_shape, param_shapes, qk_shapes)


def _mac_record(
    func_name: str,
    total_flops: int,
    output_shape: Shape | None,
    param_shapes: ParamShapes,
    qk_shapes: tuple[Shape | None, Shape | None] | None,
) -> ComputeRecord:
    """Package a verified MAC-family split, failing closed on any mismatch."""

    split = _split_mac_family(func_name, total_flops, output_shape, param_shapes, qk_shapes)
    if split is None or 2 * split[0] + split[1] != total_flops:
        return ComputeRecord(
            fma_macs=None,
            other_flops=None,
            mac_applicability="mac",
            evidence="unknown",
            reason=f"{func_name}: exact MAC split unavailable",
        )
    is_sdpa = func_name == "scaled_dot_product_attention"
    return ComputeRecord(
        fma_macs=split[0],
        other_flops=split[1],
        mac_applicability="mac",
        evidence="estimated" if is_sdpa else "formula_exact",
        reason=(
            "matmul terms formula-exact; scale+softmax per-element cost estimated"
            if is_sdpa
            else None
        ),
    )


def _non_mac_record(func_name: str, total_flops: int) -> ComputeRecord | None:
    """Record for the non-MAC tiers (zero tables, elementwise, specialty)."""

    if func_name in ZERO_FLOPS_OPS:
        return ComputeRecord(0, 0, "non_mac", "formula_exact", "memory-layout op")
    if func_name in ZERO_ARITHMETIC_CONSTRUCTIONS:
        return ComputeRecord(0, 0, "non_mac", "formula_exact", "zero-arithmetic construction")
    if func_name in ELEMENTWISE_FLOPS:
        cost = ELEMENTWISE_FLOPS[func_name]
        return ComputeRecord(
            fma_macs=0,
            other_flops=total_flops,
            mac_applicability="non_mac",
            evidence="formula_exact" if cost == 1 else "estimated",
            reason=None if cost == 1 else f"per-element cost model ({cost} FLOPs/element)",
        )
    if func_name in SPECIALTY_HANDLERS:
        return ComputeRecord(
            fma_macs=0,
            other_flops=total_flops,
            mac_applicability="non_mac",
            evidence="estimated",
            reason="shape-derived per-element cost model",
        )
    return None


def compute_forward_compute_record(
    func_name: str,
    output_shape: Shape | None,
    param_shapes: ParamShapes,
    saved_args: CapturedArgs,
    saved_kwargs: dict[str, object],
) -> ComputeRecord | None:
    """Compute the two-term record for one op from capture-time information.

    Same dispatch tiers as :func:`compute_forward_flops`; the collapsed
    ``record.flops(fma=2)`` equals ``compute_forward_flops(...)`` for every
    covered op (pinned by the coverage gate).

    Returns
    -------
    ComputeRecord | None
        The record, or ``None`` for ops outside every tier (truly unknown).
    """

    if func_name is None:
        return None
    if func_name in _CUSTOM_OP_RULES:
        return _USER_RULE_RECORD
    total = compute_forward_flops(func_name, output_shape, param_shapes, saved_args, saved_kwargs)
    if total is None:
        return None
    if func_name in _MAC_SPLITTERS:
        qk_shapes = None
        if func_name == "scaled_dot_product_attention" and len(saved_args) >= 2:
            qk_shapes = (_safe_shape(saved_args[0]), _safe_shape(saved_args[1]))
        return _mac_record(func_name, total, output_shape, param_shapes, qk_shapes)
    return _non_mac_record(func_name, total)


def derive_compute_record_for_op(op: Any) -> ComputeRecord | None:
    """Derive the two-term record for a FINISHED op from its persisted facts.

    Works on live and loaded traces alike: the split reconstructs from
    ``(func_name, flops_forward, shape, param_shapes)`` -- plus the Q/K parent
    shapes for scaled_dot_product_attention -- through the SAME per-name split
    the capture-time record uses, and FAILS CLOSED: any reconstruction that
    cannot reproduce ``2*fma + other == flops_forward`` exactly degrades to an
    unknown split, never a wrong one (the flops//2-on-biased-ops disease this
    replaces).

    Returns
    -------
    ComputeRecord | None
        The record; ``None`` for boundary pseudo-rows (compute not
        applicable) and for ops with unknown FLOPs.
    """

    if getattr(op, "is_input", False) or getattr(op, "is_output", False):
        return None
    stored = getattr(op, "flops_forward", None)
    if stored is not None and int(stored) == 0:
        # Zero recorded compute is an exact zero split for ANY op (synthesized
        # source rows included): 2*0 + 0 == 0.
        return ComputeRecord(0, 0, "non_mac", "formula_exact", "zero recorded compute")
    func_name = getattr(op, "func_name", None)
    if func_name in _CUSTOM_OP_RULES:
        return _USER_RULE_RECORD
    if stored is None or func_name is None or func_name == "none":
        return None
    func_name = str(func_name)
    total = int(stored)
    output_shape = _safe_shape(getattr(op, "shape", None))
    param_shapes = tuple(tuple(shape) for shape in (getattr(op, "param_shapes", None) or ()))
    if func_name in _MAC_SPLITTERS:
        qk_shapes = None
        if func_name == "scaled_dot_product_attention":
            qk_shapes = _resolve_qk_parent_shapes(op)
        return _mac_record(func_name, total, output_shape, param_shapes, qk_shapes)
    return _non_mac_record(func_name, total)


def _resolve_qk_parent_shapes(op: Any) -> tuple[Shape | None, Shape | None] | None:
    """Resolve sdpa Q/K argument shapes from the op's recorded parent edges."""

    positions = getattr(op, "parent_arg_positions", None)
    trace = getattr(op, "source_trace", None)
    if not positions or trace is None:
        return None
    arg_positions = positions.get("args", {})
    shapes: list[Shape | None] = []
    for index in (0, 1):
        label = arg_positions.get(index)
        if label is None:
            return None
        try:
            parent = trace[label]
        except (KeyError, IndexError, ValueError, TypeError, RuntimeError, AttributeError):
            return None
        shapes.append(_safe_shape(getattr(parent, "shape", None)))
    return (shapes[0], shapes[1])
