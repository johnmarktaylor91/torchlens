"""Shared, validated norm linearization record (mikit D8, build item 3).

ONE record -- :class:`NormReconstruction` -- serves every consumer that folds a
normalization layer: the norm facets, ``logit_lens``, and the mech-interp
kit's DLA. The rules, verbatim from the panel memo:

- The frozen SCALE is computed from the CAPTURED norm input per (batch,
  position): LayerNorm ``sqrt(mean((x - mean(x))^2) + eps)`` (centering
  distributes per component -- it is linear), RMSNorm ``sqrt(mean(x^2) + eps)``
  (no centering). This definition matches TransformerLens's own
  ``ln_final.hook_scale`` bit-exactly on real gpt2 (measured).
- ``kind`` is a VALIDATED three-way classification: ``layernorm_affine``,
  ``rmsnorm_affine``, and ``normalize_only`` (zero affine parameters --
  EVIDENCED by a passing affine-free validation, never defaulted; it occurs
  in the wild: TLens's folded models use LayerNormPre).
- The reconstruction is validated against the captured norm output BEFORE
  anything is handed out (a wrong fold costs 8.7-120 logits while still
  producing a plausible ranking -- the silent-wrongness class this exists to
  kill). Unmatched conventions (a Gemma-style ``(1 + weight)`` without a
  matching recipe convention) and missing ``eps`` REFUSE, never guess.

Every spelling is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NoReturn

import torch

from ..errors._base import TorchLensError
from .tolerances import within_reconstruction_tolerance

__all__ = [
    "NormKind",
    "NormReconstruction",
    "NormReconstructionError",
    "reconstruct_norm",
]

NormKind = Literal["layernorm_affine", "rmsnorm_affine", "normalize_only"]


class NormReconstructionError(TorchLensError, RuntimeError):
    """Raised when a norm cannot be classified, reconstructed, or validated."""


def _refuse(code: str, message: str, remedy: str, **payload: Any) -> NoReturn:
    """Raise a contracted :class:`NormReconstructionError`."""

    raise NormReconstructionError(
        f"{message} Remedy: {remedy}", code=code, remedy=remedy, **payload
    )


@dataclass(frozen=True)
class NormReconstruction:
    """A validated, frozen linearization of one captured norm application.

    Parameters
    ----------
    kind:
        Validated three-way classification.
    eps:
        The norm's epsilon, read from module metadata (never guessed).
    input:
        Captured norm input tensor (the linearization's operating point).
    output:
        Captured norm output tensor (the validation target).
    scale:
        Frozen per-(batch, position) denominator, keepdim on the feature axis.
    mean:
        Frozen per-(batch, position) centering mean (``None`` for RMS kinds).
    gamma:
        Affine weight (``None`` for ``normalize_only``).
    beta:
        Affine bias (``None`` for RMS and ``normalize_only``).
    centered:
        Whether the validated form centers (LayerNorm lineage). Distinguishes
        the two affine-free variants inside ``normalize_only``.
    module_address:
        Address of the norm module the reconstruction linearizes.
    validation_receipt:
        How the reconstruction was checked (check kind + measured residual).
    """

    kind: NormKind
    eps: float
    input: torch.Tensor
    output: torch.Tensor
    scale: torch.Tensor
    mean: torch.Tensor | None
    gamma: torch.Tensor | None
    beta: torch.Tensor | None
    centered: bool
    module_address: str | None
    validation_receipt: dict[str, Any]

    def apply_frozen(self, component: torch.Tensor) -> torch.Tensor:
        """Push one additive component through the frozen linearization.

        Centering distributes per component (it is linear), and the frozen
        scale is a constant, so a stack of components pushed through this map
        sums to the norm's actual output (plus the beta constant, which is
        NOT included here -- constants are the caller's explicit row).
        """

        value = component
        if self.centered:
            value = value - value.mean(dim=-1, keepdim=True)
        value = value / self.scale
        if self.gamma is not None:
            value = value * self.gamma
        return value

    def folded_directions(self, directions: torch.Tensor) -> torch.Tensor:
        """Fold gamma into unembedding-side DIRECTIONS (mikit D9).

        ``directions`` is ``[d_model, n_dirs]``; gamma folds elementwise on
        the feature axis before any vocabulary-sized tensor can exist.
        """

        if self.gamma is None:
            return directions
        return directions * self.gamma.reshape(-1, 1)


def _feature_stats(
    value: torch.Tensor, *, centered: bool, eps: float
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Return the frozen (mean, scale) pair for one captured norm input."""

    if centered:
        mean = value.mean(dim=-1, keepdim=True)
        var = (value - mean).pow(2).mean(dim=-1, keepdim=True)
        return mean, torch.sqrt(var + eps)
    return None, torch.sqrt(value.pow(2).mean(dim=-1, keepdim=True) + eps)


def _reconstruct(
    value: torch.Tensor,
    *,
    centered: bool,
    eps: float,
    gamma: torch.Tensor | None,
    beta: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (mean, scale, reconstruction, |addend| magnitude) for one form.

    The magnitude is the cancellation-aware error basis: a normalized element
    near zero is the difference of same-scale quantities ``x`` and ``mean``,
    so its achievable absolute accuracy is set by ``(|x| + |mean|) / scale``,
    never by the tiny output value (the exact refused-19/36-correct class).
    """

    mean, scale = _feature_stats(value, centered=centered, eps=eps)
    if mean is not None:
        recon = (value - mean) / scale
        magnitude = (value.abs() + mean.abs()) / scale
    else:
        recon = value / scale
        magnitude = value.abs() / scale
    if gamma is not None:
        recon = recon * gamma
        magnitude = magnitude * gamma.abs()
    if beta is not None:
        recon = recon + beta
        magnitude = magnitude + beta.abs()
    return mean, scale, recon, magnitude


def _candidate_forms(
    gamma: torch.Tensor | None, beta: torch.Tensor | None
) -> tuple[tuple[NormKind, bool, torch.Tensor | None, torch.Tensor | None], ...]:
    """Return the classification candidates the evidence admits, in order.

    Each entry is ``(kind, centered, gamma, beta)``. With both affine
    parameters present LayerNorm is the only match; with weight only, RMS
    first (the common case) then centered-affine-without-bias; with neither,
    both affine-free variants (evidenced, never defaulted).
    """

    if gamma is not None and beta is not None:
        return (("layernorm_affine", True, gamma, beta),)
    if gamma is not None:
        return (
            ("rmsnorm_affine", False, gamma, None),
            ("layernorm_affine", True, gamma, None),
        )
    return (
        ("normalize_only", True, None, None),
        ("normalize_only", False, None, None),
    )


def reconstruct_norm(  # noqa: PLR0913 -- the D8 evidence set: every input is real captured evidence
    *,
    input: torch.Tensor,
    output: torch.Tensor,
    gamma: torch.Tensor | None,
    beta: torch.Tensor | None,
    eps: float | None,
    module_address: str | None = None,
    class_name: str | None = None,
) -> NormReconstruction:
    """Classify, reconstruct, and VALIDATE one captured norm application.

    Parameters
    ----------
    input:
        Captured norm input payload.
    output:
        Captured norm output payload (the validation target).
    gamma / beta:
        Affine parameters as the recipe evidenced them (``None`` = absent).
    eps:
        The norm's epsilon; ``None`` refuses (never guessed).
    module_address:
        Norm module address, for coordinates and messages.
    class_name:
        Norm module class name, quoted in refusals.

    Returns
    -------
    NormReconstruction
        The validated record; every candidate form failing validation
        refuses typed instead (unmatched convention).
    """

    if eps is None:
        _refuse(
            code="norm_eps_unavailable",
            message=f"The norm at {module_address or '<unknown>'} carries no epsilon metadata.",
            remedy="capture with a recipe that discloses eps (config_value 'eps'/'variance_epsilon'), "
            "or register a facet recipe for this norm class",
            module_address=module_address,
        )
    if input.shape != output.shape:
        _refuse(
            code="norm_geometry_mismatch",
            message=f"Norm input shape {tuple(input.shape)} does not match output shape "
            f"{tuple(output.shape)} at {module_address or '<unknown>'}.",
            remedy="anchor the norm's input through the dataflow walk before reconstructing",
            module_address=module_address,
        )

    tried: list[str] = []
    eps_value = float(eps)
    for kind, centered, form_gamma, form_beta in _candidate_forms(gamma, beta):
        mean, scale, recon, magnitude = _reconstruct(
            input, centered=centered, eps=eps_value, gamma=form_gamma, beta=form_beta
        )
        if within_reconstruction_tolerance(
            recon, output, magnitude=magnitude, reduction_length=int(input.shape[-1])
        ):
            residual = float((recon.float() - output.float()).abs().max())
            return NormReconstruction(
                kind=kind,
                eps=eps_value,
                input=input,
                output=output,
                scale=scale,
                mean=mean,
                gamma=form_gamma,
                beta=form_beta,
                centered=centered,
                module_address=module_address,
                validation_receipt={
                    "check": "reconstruction_vs_captured_output",
                    "result": "validated",
                    "max_abs_residual": residual,
                    "form": f"{kind}/{'centered' if centered else 'uncentered'}",
                },
            )
        tried.append(f"{kind}/{'centered' if centered else 'uncentered'}")

    _refuse(
        code="norm_convention_unmatched",
        message=f"No known norm convention reproduces the captured output of "
        f"{class_name or 'the norm'} at {module_address or '<unknown>'} "
        f"(tried: {', '.join(tried)}). A nonstandard convention (e.g. Gemma-style "
        f"(1 + weight)) needs its own recipe convention.",
        remedy="register a facet recipe evidencing this norm's convention; the kit never guesses",
        module_address=module_address,
        tried=tried,
    )
    raise AssertionError("unreachable")
