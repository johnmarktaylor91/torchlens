"""Validation perturbations: a random wrong value per dtype for the replay tripwire.

The forward validation replays each op from a perturbed parent and requires the
output to move; these helpers produce that perturbation (integer, bool, float
and complex strategies, and sparse parents through their values component).
Leaf module: it imports torch and the sparse component helpers only.
"""

from __future__ import annotations

import torch

from .._utils._sparse_parts import sparse_component_tensors, sparse_with_values

# Maximum number of random perturbation attempts before giving up on finding
# a value different from the original.  Relevant for integer/bool tensors
# where the value space may be small (e.g., a single-element int tensor).
MAX_PERTURB_ATTEMPTS = 100


def _perturb_sparse_outs(parent_outs: torch.Tensor, output_outs: torch.Tensor) -> torch.Tensor:
    """Perturb a sparse parent through its values, keeping layout, indices and shape.

    The dense strategy reads the tensor elementwise (``isfinite``, ``unique``,
    fancy indexing), which the sparse backends do not implement; perturbing
    the values component with that same strategy keeps the tripwire
    non-vacuous while the child op still receives a well-formed sparse input.
    """

    values = sparse_component_tensors(parent_outs)[-1]
    return sparse_with_values(parent_outs, _perturb_layer_outs(values, output_outs))


def _perturb_layer_outs(parent_outs: torch.Tensor, output_outs: torch.Tensor) -> torch.Tensor:
    """Generate a random perturbation of a saved tensor for validation.

    The perturbation strategy varies by dtype to produce meaningful
    "wrong" values while respecting type constraints:

    - **Integer types**: sample uniformly from [min, max+1) of the original
      tensor, with the exclusive high bound clamped by ``torch.iinfo`` so the
      ``max + 1`` cannot overflow the dtype (a saturated range widens to the
      full valid dtype range instead).  If the tensor is all-one-value (single
      unique value), widen the range to [-10, 11) or [0, 11).  Retries up to
      ``MAX_PERTURB_ATTEMPTS`` to avoid accidentally reproducing the original.
    - **Bool**: random 0/1, retried to ensure difference.
    - **Float/complex**: finite range-based random values. Floating tensors
      whose saved values are entirely non-finite are perturbed to finite zeros
      so NaN-aware equality cannot turn the perturbation into a no-op.

    Parameters
    ----------
    parent_outs:
        Original parent tensor to perturb.
    output_outs:
        Child layer output tensor, used to calibrate float perturbation scale.

    Returns
    -------
    torch.Tensor
        New tensor of the same shape and dtype with perturbed values.
    """
    if parent_outs.numel() == 0:
        return parent_outs.detach().clone()
    if parent_outs.layout is not torch.strided:
        return _perturb_sparse_outs(parent_outs, output_outs)

    if parent_outs.dtype in [
        torch.int,
        torch.long,
        torch.short,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ]:
        perturbed_outs = _perturb_integer_outs(parent_outs)
    elif parent_outs.dtype == torch.bool:
        perturbed_outs = _perturb_bool_outs(parent_outs)
    elif parent_outs.is_complex():
        perturbed_outs = _perturb_complex_outs(parent_outs)
    else:
        perturbed_outs = _perturb_float_outs(parent_outs, output_outs)

    return perturbed_outs


def _perturb_integer_outs(parent_outs: torch.Tensor) -> torch.Tensor:
    """Perturb an integer tensor: the strategy :func:`_perturb_layer_outs` documents for integers."""

    device = parent_outs.device
    tensor_unique_vals = torch.unique(parent_outs)
    if len(tensor_unique_vals) > 1:
        # Multiple unique values: sample within the original range.
        #
        # Compute the bounds as Python ints and clamp the exclusive high
        # bound with ``torch.iinfo`` so ``max() + 1`` can never overflow the
        # dtype. A legitimately captured tensor holding ``iinfo.max`` (e.g.
        # PyG sentinel/cluster index tensors carry ``INT64_MAX``) would
        # otherwise wrap ``max() + 1`` to ``INT64_MIN`` and make
        # ``torch.randint`` raise "random_ expects 'from' to be less than
        # 'to'". This mirrors the ``torch.finfo`` clamping the float branch
        # already does; it only bounds the *sampling range*, never skips the
        # perturbation, so the tripwire stays non-vacuous.
        int_info = torch.iinfo(parent_outs.dtype)
        int_lo = int(parent_outs.min().item())
        int_hi = int(parent_outs.max().item())
        int_hi_excl = int_hi + 1 if int_hi < int_info.max else int_info.max
        if int_lo >= int_hi_excl:
            # Saturated range (e.g. the whole tensor sits at ``iinfo.max``):
            # widen to the full valid dtype range so we can still draw a
            # genuinely different value rather than degenerating to a no-op.
            int_lo, int_hi_excl = int_info.min, int_info.max
        perturbed_outs = parent_outs.detach().clone()
        for _ in range(MAX_PERTURB_ATTEMPTS):
            perturbed_outs = torch.randint(
                int_lo,
                int_hi_excl,
                size=parent_outs.shape,
                device=device,
            ).type(parent_outs.dtype)
            if not torch.equal(perturbed_outs, parent_outs):
                break
    else:
        # Single unique value: widen to a fixed range to guarantee a
        # different value can be produced.
        perturbed_outs = parent_outs.detach().clone()
        for _ in range(MAX_PERTURB_ATTEMPTS):
            if torch.min(parent_outs) < 0:
                perturbed_outs = torch.randint(-10, 11, size=parent_outs.shape, device=device).type(
                    parent_outs.dtype
                )
            else:
                perturbed_outs = torch.randint(0, 11, size=parent_outs.shape, device=device).type(
                    parent_outs.dtype
                )
            if not torch.equal(perturbed_outs, parent_outs):
                break
    return perturbed_outs


def _perturb_bool_outs(parent_outs: torch.Tensor) -> torch.Tensor:
    """Perturb a bool tensor: the strategy :func:`_perturb_layer_outs` documents for bools."""

    device = parent_outs.device
    # Random bool, retried until different from original.
    perturbed_outs = parent_outs.detach().clone()
    for _ in range(MAX_PERTURB_ATTEMPTS):
        perturbed_outs = torch.randint(0, 2, size=parent_outs.shape, device=device).bool()
        if not torch.equal(perturbed_outs, parent_outs):
            break
    return perturbed_outs


def _perturb_complex_outs(parent_outs: torch.Tensor) -> torch.Tensor:
    """Perturb a complex tensor uniformly within its real and imaginary ranges."""

    device = parent_outs.device
    # Float/complex: uniform random within the original value range.
    # Using the original range ensures perturbed values stay in the
    # valid domain for range-restricted functions (e.g., bernoulli
    # requires probabilities in [0,1]).  For typical tensors with wide
    # range this produces meaningfully different values.
    real = parent_outs.real.float()
    imag = parent_outs.imag.float()
    r_lo, r_hi = real.min().item(), real.max().item()
    i_lo, i_hi = imag.min().item(), imag.max().item()
    if r_lo == r_hi:
        r_lo, r_hi = r_lo - 1.0, r_hi + 1.0
    if i_lo == i_hi:
        i_lo, i_hi = i_lo - 1.0, i_hi + 1.0
    return torch.complex(
        torch.rand(parent_outs.shape, device=device) * (r_hi - r_lo) + r_lo,
        torch.rand(parent_outs.shape, device=device) * (i_hi - i_lo) + i_lo,
    ).type(parent_outs.dtype)


def _perturb_float_outs(parent_outs: torch.Tensor, output_outs: torch.Tensor) -> torch.Tensor:
    """Perturb a floating tensor: finite range-based values scaled against the child output."""

    device = parent_outs.device
    parent_float = parent_outs.float()
    finite_parent = parent_float[torch.isfinite(parent_float)]
    if finite_parent.numel() == 0:
        return torch.zeros_like(parent_outs)
    lo = finite_parent.min().item()
    hi = finite_parent.max().item()
    finite_output = output_outs.detach().float().abs()
    if finite_output.layout is not torch.strided:
        # A sparse child output (``to_sparse``) calibrates through its dense form.
        finite_output = finite_output.to_dense()
    finite_output = finite_output[torch.isfinite(finite_output)]
    output_scale = finite_output.max().item() if finite_output.numel() else 0.0
    dtype_info = torch.finfo(parent_outs.dtype)
    parent_scale = max(abs(lo), abs(hi), hi - lo, 1.0)
    scaled_expansion: float | None = None
    if hi - lo < max(1e-6, abs(lo) * 1e-6):
        # Near-constant tensor — range is too narrow for meaningful
        # perturbation at float32 precision. Expand by ±10% of the
        # parent magnitude, or by the child output scale when the parent
        # is near zero but the op output is huge. Without the output
        # scale, zero-valued broadcast parents can be perturbed by only
        # ~1 and then disappear under float32 rounding beside 1e38
        # operands.
        expansion = min(
            max(1.0, abs(lo) * 0.1, output_scale * 0.1),
            dtype_info.max * 0.25,
        )
        if output_scale > parent_scale * 1.0e6:
            scaled_expansion = expansion
        else:
            lo, hi = lo - expansion, hi + expansion
    elif output_scale > parent_scale * 1.0e6:
        # A non-constant but tiny parent range can also be invisible
        # when combined additively with enormous operands. Expand only
        # for extreme scale separation so normal range-restricted
        # tensors keep their original valid-domain perturbations.
        scaled_expansion = min(output_scale * 0.1, dtype_info.max * 0.25)
    if scaled_expansion is not None:
        signs = torch.where(
            torch.rand_like(parent_outs.float(), device=device) < 0.5,
            -1.0,
            1.0,
        )
        magnitudes = (
            torch.rand_like(parent_outs.float(), device=device) * 0.5 + 0.5
        ) * scaled_expansion
        perturbed_outs = parent_outs.float() + signs * magnitudes
    else:
        perturbed_outs = torch.rand_like(parent_outs.float(), device=device) * (hi - lo) + lo
    return perturbed_outs.type(parent_outs.dtype)
