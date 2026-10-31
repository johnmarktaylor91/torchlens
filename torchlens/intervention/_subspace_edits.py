"""k-dimensional subspace ablation and interchange edits (N-IV-6).

``subspace_ablate(basis, mode=...)`` removes, or replaces, the component of a
site's value inside the span of a ``[k, d]`` basis along one feature axis;
``subspace_interchange(basis, source)`` writes the source trace's component in
that span over the target's, keeping the orthogonal complement. With ``P`` the
orthogonal projector onto the span of the basis rows:

* ``mode="zero"``: ``x - P x`` (``project_off`` with a k-dimensional basis);
* ``mode="mean"``: ``x - P x + P m`` for a given mean activation ``m``;
* ``mode="resample"``: ``x - P x + P x[perm]`` with a seeded batch permutation;
* interchange: ``x - P x + P s`` for the source value ``s`` at the same site.

The basis rows need not be orthonormal: ``P`` is built from an orthonormal
basis of their span (reduced QR, rank-checked), so a rank-deficient basis
refuses typed instead of projecting onto a smaller span than the one named.
DOCUMENTED-UNSTABLE spellings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from .._errors import InvalidArgumentError
from .errors import HookValueError
from .hooks import HookContext
from .types import Edit

_MODES = ("zero", "mean", "resample")


def _orthonormal_rows(basis: Any, helper_name: str) -> torch.Tensor:
    """Validate a ``[k, d]`` (or ``[d]``) basis and return an orthonormal ``[k, d]``.

    Parameters
    ----------
    basis:
        Basis rows spanning the subspace.
    helper_name:
        Calling helper, for error text.

    Returns
    -------
    torch.Tensor
        Float64 rows with orthonormal span equal to the basis span.

    Raises
    ------
    InvalidArgumentError
        ``subspace_basis_invalid`` for a non-tensor, non-finite, empty or
        rank-deficient basis.
    """

    if not isinstance(basis, torch.Tensor) or basis.ndim not in (1, 2) or basis.numel() == 0:
        raise InvalidArgumentError(
            f"{helper_name} basis= must be a non-empty [k, d] (or [d]) tensor",
            code="subspace_basis_invalid",
            remedy="pass the basis as a [k, d] tensor of k direction rows over the feature axis",
            argument="basis",
        )
    rows = basis.detach().to(dtype=torch.float64, device="cpu")
    rows = rows.unsqueeze(0) if rows.ndim == 1 else rows
    if not bool(torch.isfinite(rows).all()):
        raise InvalidArgumentError(
            f"{helper_name} basis= has non-finite entries",
            code="subspace_basis_invalid",
            remedy="pass a finite basis",
            argument="basis",
        )
    k, d = int(rows.shape[0]), int(rows.shape[1])
    q, r = torch.linalg.qr(rows.T, mode="reduced")
    diagonal = torch.abs(torch.diagonal(r))
    scale = float(diagonal.max()) if diagonal.numel() else 0.0
    if k > d or scale == 0.0 or bool((diagonal <= scale * 1e-10).any()):
        raise InvalidArgumentError(
            f"{helper_name} basis= has {k} rows but spans fewer dimensions (rank-deficient "
            f"over a {d}-dim feature axis): its span is smaller than the subspace it names",
            code="subspace_basis_invalid",
            remedy="drop linearly dependent rows, or orthogonalize the basis first",
            argument="basis",
        )
    return q.T.contiguous()


def _project(value: torch.Tensor, rows: torch.Tensor, feature_axis: int) -> torch.Tensor:
    """Return the orthogonal projection of ``value`` onto the span of ``rows``."""

    axis = feature_axis % value.ndim
    if int(value.shape[axis]) != int(rows.shape[1]):
        raise HookValueError(
            f"subspace basis dimension {int(rows.shape[1])} does not match the value's "
            f"feature axis {axis} of size {int(value.shape[axis])}",
            code="subspace_basis_dim_mismatch",
        )
    moved = torch.movedim(value, axis, -1)
    basis = rows.to(device=value.device, dtype=value.dtype)
    projected = (moved @ basis.T) @ basis
    return torch.movedim(projected, -1, axis)


def _validate_feature_axis(feature_axis: Any, helper_name: str) -> int:
    """Return ``feature_axis`` as an int, refusing anything else."""

    if isinstance(feature_axis, bool) or not isinstance(feature_axis, int):
        raise InvalidArgumentError(
            f"{helper_name} feature_axis= must be an int axis index; got {feature_axis!r}",
            code="subspace_basis_invalid",
            remedy="name the axis the basis indexes, e.g. feature_axis=-1",
            argument="dim",
        )
    return int(feature_axis)


def subspace_ablate(
    basis: torch.Tensor,
    *,
    mode: str = "zero",
    dim: int = -1,
    mean: torch.Tensor | None = None,
    seed: int | None = None,
) -> Edit:
    """Create an edit that ablates a site's component inside a k-dim subspace.

    Parameters
    ----------
    basis:
        ``[k, d]`` rows spanning the subspace (``[d]`` for one direction).
    mode:
        ``"zero"`` removes the component; ``"mean"`` replaces it with the
        component of ``mean``; ``"resample"`` replaces it with another batch
        row's component under a seeded permutation.
    feature_axis:
        Axis of the site value the basis indexes (default the trailing axis).
    mean:
        Mean activation (broadcastable to the site value) for ``mode="mean"``.
    seed:
        Permutation seed for ``mode="resample"`` (required: a resample edit
        must be reproducible).

    Returns
    -------
    Edit
        The subspace edit.

    Raises
    ------
    InvalidArgumentError
        ``subspace_basis_invalid`` / ``subspace_mode_invalid`` on bad
        arguments.
    """

    rows = _orthonormal_rows(basis, "subspace_ablate")
    axis = _validate_feature_axis(dim, "subspace_ablate")
    if mode not in _MODES:
        raise InvalidArgumentError(
            f"subspace_ablate mode= must be one of {', '.join(_MODES)}; received {mode!r}",
            code="subspace_mode_invalid",
            remedy="pass mode='zero', 'mean' or 'resample'",
            argument="mode",
        )
    if (mode == "mean") != (mean is not None) or (mode == "resample") != (seed is not None):
        raise InvalidArgumentError(
            "subspace_ablate takes mean= exactly when mode='mean' and seed= exactly when "
            f"mode='resample'; received mode={mode!r}, mean given: {mean is not None}, "
            f"seed given: {seed is not None}",
            code="subspace_mode_invalid",
            remedy="pass mean=<tensor> with mode='mean', or seed=<int> with mode='resample'",
            argument="mode",
        )
    mean_value = None if mean is None else mean.detach().clone()

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for this subspace ablation."""

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Replace the in-subspace component of ``out``."""

            del hook
            kept = out - _project(out, rows, axis)
            if mode == "zero":
                return kept
            if mode == "mean":
                reference = torch.broadcast_to(
                    cast(torch.Tensor, mean_value).to(device=out.device, dtype=out.dtype),
                    out.shape,
                )
                return kept + _project(reference, rows, axis)
            generator = torch.Generator().manual_seed(int(seed or 0))
            order = torch.randperm(int(out.shape[0]), generator=generator).to(out.device)
            return kept + _project(out.detach().index_select(0, order), rows, axis)

        return _hook

    from .helpers import _helper_spec

    return _helper_spec(
        "subspace_ablate",
        args=(basis,),
        kwargs={"mode": mode, "dim": axis, "mean": mean, "seed": seed},
        factory=factory,
        portability="opaque_audit",
        batch_independent=mode != "resample",
    )


def subspace_interchange(
    basis: torch.Tensor,
    source: Any,
    *,
    dim: int = -1,
) -> Edit:
    """Create an edit writing the source trace's in-subspace component over the target's.

    The target keeps its component orthogonal to the span; the in-span
    component comes from ``source`` at the same site (same-site donor
    resolution as :func:`torchlens.patch_from`), which is the
    k-dimensional basis interchange used by distributed alignment search
    with a learned rotation the user trained elsewhere.

    Parameters
    ----------
    basis:
        ``[k, d]`` rows spanning the subspace.
    source:
        Source trace supplying the donor value at each edited site.
    feature_axis:
        Axis of the site value the basis indexes.

    Returns
    -------
    Edit
        The interchange edit.
    """

    rows = _orthonormal_rows(basis, "subspace_interchange")
    axis = _validate_feature_axis(dim, "subspace_interchange")
    from .helpers import _helper_spec, patch_from

    donor_spec = patch_from(source)

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for this interchange."""

        donor_hook = donor_spec()

        def _hook(out: torch.Tensor, *, hook: HookContext) -> torch.Tensor:
            """Swap the in-subspace component for the donor's."""

            donor = donor_hook(out, hook=hook).to(dtype=out.dtype)
            return out - _project(out, rows, axis) + _project(donor, rows, axis)

        return _hook

    return _helper_spec(
        "subspace_interchange",
        args=(basis,),
        kwargs={**dict(donor_spec.kwargs), "dim": axis},
        factory=factory,
        portability="opaque_audit",
        batch_independent=True,
    )
