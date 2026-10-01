"""Position-scaling classes and the padding-waste disclosure (F09; D12).

Costreport item 10: each counted op gets a POSITION-SCALING CLASS --
``linear_in_tokens`` | ``quadratic_in_sequence`` | ``invariant`` |
``unknown`` -- so the padding-waste line uses the class-split estimator
(the naive whole-forward scaler is 0.07% off at S=16 and degrades with
sequence length; the class split matched an independent unpadded-retrace
oracle to ~1.6e-7 relative on the panel's gpt2 batch).

Classification is evidence-based, never guessed: the primary signal is
which dims of the op's RECORDED shapes equal the padded sequence length
(two matches -> quadratic, one -> linear, none -> invariant), with the
op's parameter consumption as a cross-check (a param-consuming GEMM is
linear in tokens; an activation-activation matmul is the quadratic
family). When the two signals conflict the class is ``unknown`` -- a
confident wrong class is worse than an honest unknown (D3).

The disclosure line is three-part (D12): the estimated pad-position
FLOPs with its share of the forward, the largest single contributor
named, and the note that 6ND's D counts pads by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

#: Closed position-scaling vocabulary (D12).
POSITION_SCALING_CLASSES: tuple[str, ...] = (
    "linear_in_tokens",
    "quadratic_in_sequence",
    "invariant",
    "unknown",
)

#: MAC-family func names that consume parameters: linear in tokens (the
#: GEMM rows run once per position).
_PARAM_MAC_LINEAR: frozenset[str] = frozenset(
    {"linear", "addmm", "conv1d", "conv2d", "conv3d", "embedding"}
)

#: Activation-activation matmul family: the attention-scores shape --
#: quadratic in sequence when it carries the token axis twice.
_ACTIVATION_MATMUL: frozenset[str] = frozenset({"matmul", "bmm", "baddbmm", "einsum"})

#: SDPA fuses the two quadratic matmuls and the quadratic softmax.
_SDPA: frozenset[str] = frozenset({"scaled_dot_product_attention"})


def _shape_axis_matches(op: Any, padded_length: int) -> int:
    """Count how many times the padded length appears among output dims."""

    shape = getattr(op, "shape", None) or ()
    return sum(1 for dim in shape if int(dim) == int(padded_length))


def position_scaling_class(op: Any, padded_length: int) -> str:
    """Classify one op's cost scaling in the padded sequence length.

    Two independent signals -- recorded shape occurrence of the padded
    length and the op's family -- must AGREE where both speak; a conflict
    degrades to ``unknown``, never a confident wrong class.
    """

    flops = getattr(op, "flops_forward", None)
    if flops is None or int(flops) == 0:
        return "invariant"
    func_name = str(getattr(op, "func_name", "") or "")
    matches = _shape_axis_matches(op, padded_length)
    if func_name in _SDPA:
        return "quadratic_in_sequence"
    if func_name in _PARAM_MAC_LINEAR:
        return "linear_in_tokens" if matches <= 1 else "unknown"
    # Shared axis-count ladder: the token axis twice is quadratic, once is
    # linear; an axis-absent tensor falls through to the family floor.
    by_matches = {2: "quadratic_in_sequence", 1: "linear_in_tokens"}
    verdict = by_matches.get(min(matches, 2))
    if func_name in _ACTIVATION_MATMUL:
        # Attention scores carry the token axis twice; a projection-style
        # activation matmul carries it once (still linear); axis-absent
        # activation matmuls stay honest as unknown.
        return verdict or "unknown"
    # Elementwise / norm / reduction family: scale with however many
    # token axes the tensor carries.
    return verdict or "invariant"


@dataclass(frozen=True)
class PaddingWaste:
    """The class-split padding-waste estimate (D12): a DISCLOSURE.

    ``waste_flops`` is the estimated forward FLOPs attributable to pad
    positions; ``unclassified_flops`` is the known mass whose scaling
    class could not be determined (excluded from the estimate and
    disclosed, never scaled by a guess).
    """

    waste_flops: int
    forward_flops: int
    waste_by_class: dict[str, int]
    unclassified_flops: int
    largest_contributor: str | None
    largest_contributor_flops: int
    padded_length: int
    real_lengths: tuple[int, ...]

    def __repr__(self) -> str:
        """Bounded identity card (F10/D31): lengths count, never dump.

        ``real_lengths`` scales with the batch, so the card counts it and
        keeps the estimate/disclosure vocabulary intact.
        """

        return (
            f"PaddingWaste(estimated waste_flops={self.waste_flops} of "
            f"forward_flops={self.forward_flops}, padded_length={self.padded_length}, "
            f"{len(self.real_lengths)} real length(s), "
            f"unclassified_flops={self.unclassified_flops}; read .waste_by_class)"
        )

    @property
    def waste_fraction(self) -> float | None:
        """Waste share of the known forward (None when the forward is 0)."""

        if self.forward_flops <= 0:
            return None
        return self.waste_flops / self.forward_flops

    @property
    def disclosure(self) -> str:
        """The three-part padding disclosure line (D12)."""

        fraction = self.waste_fraction
        share = "" if fraction is None else f" ({100.0 * fraction:.1f}% of the known forward)"
        parts = [
            f"padding: ~{self.waste_flops} forward FLOPs attributed to pad positions"
            f"{share} [class-split estimate]",
        ]
        if self.largest_contributor is not None:
            parts.append(
                f"largest single contributor: {self.largest_contributor} "
                f"(~{self.largest_contributor_flops} FLOPs)"
            )
        if self.unclassified_flops:
            parts.append(
                f"{self.unclassified_flops} known FLOPs carried no scaling class and are "
                "excluded from the estimate"
            )
        parts.append("note: the 6ND comparator's D counts pad positions by construction")
        return "; ".join(parts)


def padding_waste(
    trace: Trace,
    *,
    padded_length: int,
    real_lengths: tuple[int, ...] | list[int],
) -> PaddingWaste:
    """Estimate pad-position forward FLOPs via the class-split rule (D12).

    Parameters
    ----------
    trace:
        Finished trace of a PADDED batch forward.
    padded_length:
        The padded sequence length S every row was padded to.
    real_lengths:
        Real (unpadded) token counts per batch row.

    Notes
    -----
    Per class: ``linear_in_tokens`` wastes ``1 - sum(r_i)/(B*S)`` of its
    mass; ``quadratic_in_sequence`` wastes ``1 - sum(r_i^2)/(B*S^2)``;
    ``invariant`` wastes nothing; ``unknown`` is excluded and disclosed.
    """

    if not isinstance(padded_length, int) or isinstance(padded_length, bool) or padded_length < 1:
        raise InvalidArgumentError(
            f"padded_length must be a positive integer; got {padded_length!r}.",
            code="padding_waste_length_invalid",
            remedy="Pass the padded sequence length S of the captured batch.",
        )
    lengths = tuple(int(length) for length in real_lengths)
    if not lengths or any(length < 0 or length > padded_length for length in lengths):
        raise InvalidArgumentError(
            f"real_lengths must be non-empty with every value in [0, padded_length]; "
            f"got {real_lengths!r} against padded_length={padded_length}.",
            code="padding_waste_lengths_invalid",
            remedy="Pass one real token count per batch row, each <= padded_length.",
        )
    batch = len(lengths)
    linear_live = sum(lengths) / (batch * padded_length)
    quadratic_live = sum(length * length for length in lengths) / (batch * padded_length**2)

    from ._compute_truth import classify_row

    waste_by_class = dict.fromkeys(POSITION_SCALING_CLASSES, 0)
    unclassified = 0
    largest: tuple[str | None, int] = (None, 0)
    forward = 0
    for op in getattr(trace, "layer_list", ()) or ():
        if classify_row(op) != "known":
            continue
        flops = int(getattr(op, "flops_forward", 0) or 0)
        if flops == 0:
            continue
        forward += flops
        scaling = position_scaling_class(op, padded_length)
        if scaling == "unknown":
            unclassified += flops
            continue
        if scaling == "linear_in_tokens":
            waste = int(round(flops * (1.0 - linear_live)))
        elif scaling == "quadratic_in_sequence":
            waste = int(round(flops * (1.0 - quadratic_live)))
        else:
            waste = 0
        waste_by_class[scaling] += waste
        if waste > largest[1]:
            largest = (str(getattr(op, "layer_label", "?")), waste)
    total_waste = sum(waste_by_class.values())
    return PaddingWaste(
        waste_flops=total_waste,
        forward_flops=forward,
        waste_by_class={name: value for name, value in waste_by_class.items() if value},
        unclassified_flops=unclassified,
        largest_contributor=largest[0],
        largest_contributor_flops=largest[1],
        padded_length=padded_length,
        real_lengths=lengths,
    )
