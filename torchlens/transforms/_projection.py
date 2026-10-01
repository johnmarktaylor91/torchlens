"""Frozen linear projection + PCA-apply (memo B8; decision 16).

``project(basis, center)`` is the unconditional primitive: a DIGEST-ADDRESSED
basis (the spec params carry the content digest and geometry, never tensor
values — canonical JSON stays canonical), exact extent checks, and the arrays
staged in a bounded process-level store. A spec rehydrated from an artifact
whose digest is not staged REFUSES with the restaging remedies; artifact
loading never imports or executes code.

``pca_apply(fitted)`` is thin validation over ``project`` consuming the P6
``tl.stats`` fitted payload: centered apply (uncentered apply leaves
Euclidean RDMs exact at ~1e-7 but damages corr-RDMs, r ~ 0.77 measured —
the docs scope the claim instead of moralizing), with the fit scope and
source digest recorded in the spec params.

Fitting DURING writing stays REFUSED at the roster level: a changing basis
gives early and late shards different coordinates — an artifact wrong by
construction. Every spelling here is DOCUMENTED-UNSTABLE pending the naming
sprint.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch

from ._context import TransformContext
from ._errors import TransformContractError
from ._registry import _register_builtin
from ._spec import PlannedStep, TensorSpec, TransformDefinition, TransformSpec, freeze_params

__tl_layer__ = "L4"

__all__ = ["pca_apply", "project"]

_HALF_DTYPES = (torch.float16, torch.bfloat16)

#: Bounded digest-addressed basis store (entries + bytes, LRU-evicted).
_STORE_MAX_ENTRIES = 16
_STORE_MAX_BYTES = 256 * 1024 * 1024

#: digest -> (basis, center) staged arrays.
_BASIS_STORE: OrderedDict[str, tuple[torch.Tensor, torch.Tensor | None]] = OrderedDict()
_STORE_BYTES = 0


def _entry_bytes(entry: tuple[torch.Tensor, torch.Tensor | None]) -> int:
    """Byte size of one staged basis entry."""

    total = 0
    for value in entry:
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total


def _stage_basis(digest: str, basis: torch.Tensor, center: torch.Tensor | None) -> None:
    """Stage basis arrays under their digest (bounded, LRU)."""

    global _STORE_BYTES
    if digest in _BASIS_STORE:
        _BASIS_STORE.move_to_end(digest)
        return
    entry = (basis, center)
    _BASIS_STORE[digest] = entry
    _STORE_BYTES += _entry_bytes(entry)
    while _BASIS_STORE and (
        len(_BASIS_STORE) > _STORE_MAX_ENTRIES or _STORE_BYTES > _STORE_MAX_BYTES
    ):
        _, evicted = _BASIS_STORE.popitem(last=False)
        _STORE_BYTES -= _entry_bytes(evicted)


def _reset_projection_store() -> None:
    """Test door: drop every staged basis."""

    global _STORE_BYTES
    _BASIS_STORE.clear()
    _STORE_BYTES = 0


def _projection_digest(basis: torch.Tensor, center: torch.Tensor | None) -> str:
    """Content digest of the (basis, center) pair in canonical float64 bytes."""

    from ..stats._fitted import fitted_digest

    anchor = (
        torch.zeros((0,), dtype=torch.float64)
        if center is None
        else center.detach().to("cpu", torch.float64)
    )
    empty = torch.zeros((0,), dtype=torch.float64)
    return fitted_digest(basis.detach().to("cpu", torch.float64), anchor, empty)


def _project_normalize(params: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize project params (digest-addressed; values never enter).

    Parameters
    ----------
    params:
        Raw params (``basis_digest``, ``in_extent``, ``out_extent``,
        ``centered``, ``fit_scope``, ``source``).

    Returns
    -------
    dict[str, Any]
        Normalized params.
    """

    digest = params.get("basis_digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise TransformContractError(
            f"project() params need a 'sha256:' basis_digest; got {digest!r}. "
            "Build the spec through tl.transforms.project(basis, center) — "
            "the factory stages the arrays and computes the digest.",
            code="transform_params_invalid",
            remedy="build the spec via project(basis, center) or pca_apply(fitted)",
            basis_digest=digest,
        )
    in_extent = params.get("in_extent")
    out_extent = params.get("out_extent")
    for key, value in (("in_extent", in_extent), ("out_extent", out_extent)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TransformContractError(
                f"project() params need a positive int {key}; got {value!r}.",
                code="transform_params_invalid",
                remedy="build the spec via project(basis, center)",
                param=key,
            )
    centered = params.get("centered", False)
    if not isinstance(centered, bool):
        raise TransformContractError(
            f"project() param 'centered' must be a bool; got {type(centered).__name__}.",
            code="transform_params_invalid",
            remedy="build the spec via project(basis, center)",
        )
    fit_scope = params.get("fit_scope")
    source = params.get("source")
    return {
        "basis_digest": digest,
        "in_extent": int(in_extent),  # type: ignore[arg-type]
        "out_extent": int(out_extent),  # type: ignore[arg-type]
        "centered": centered,
        "fit_scope": None if fit_scope is None else str(fit_scope),
        "source": None if source is None else str(source),
    }


def _project_plan(
    spec: TransformSpec, input_spec: TensorSpec, ctx: TransformContext | None
) -> PlannedStep:
    """Plan project: exact last-axis extent check, output extent replaced.

    Parameters
    ----------
    spec:
        The project spec.
    input_spec:
        Incoming tensor description.
    ctx:
        Unused (the basis is digest-addressed, not context-derived).

    Returns
    -------
    PlannedStep
        Predicted output.
    """

    params = spec.params_dict()
    if input_spec.dtype not in (
        "torch.float16",
        "torch.bfloat16",
        "torch.float32",
        "torch.float64",
    ):
        raise TransformContractError(
            f"project() consumes float tensors; got dtype {input_spec.dtype}.",
            code="transform_plan_invalid",
            remedy="cast to a float dtype first (chain a cast step)",
            dtype=input_spec.dtype,
        )
    rank = len(input_spec.shape)
    if rank < 2:
        raise TransformContractError(
            f"project() needs a feature axis; got rank {rank}.",
            code="transform_plan_invalid",
            remedy="pass tensors of rank >= 2 (the stimulus axis plus features)",
            rank=rank,
        )
    last = input_spec.shape[-1]
    if last is not None and int(last) != int(params["in_extent"]):
        raise TransformContractError(
            f"project() basis expects a feature extent of {params['in_extent']} "
            f"but the input's last axis is {last}; extent checks are exact — "
            "a basis applied to the wrong width is a silent wrong number.",
            code="transform_plan_invalid",
            remedy="match the fitted feature width (pool/flatten to it) or re-fit",
            expected=params["in_extent"],
            observed=last,
        )
    out_dtype = input_spec.dtype
    if input_spec.dtype in ("torch.float16", "torch.bfloat16"):
        out_dtype = "torch.float32"
    return PlannedStep(
        name=spec.name,
        version=spec.version,
        output=TensorSpec(
            shape=(*input_spec.shape[:-1], int(params["out_extent"])), dtype=out_dtype
        ),
        stream_safe=True,
        may_alias=False,
        context_capable=False,
    )


def _project_apply(
    spec: TransformSpec, tensor: torch.Tensor, ctx: TransformContext | None
) -> torch.Tensor:
    """Apply the staged frozen projection: ``(x - center) @ basis.T``.

    Parameters
    ----------
    spec:
        The project spec.
    tensor:
        Batch tensor (feature axis last).
    ctx:
        Unused.

    Returns
    -------
    torch.Tensor
        Projected tensor (fp32 for half inputs, T-C7).

    Raises
    ------
    TransformContractError
        ``transform_basis_unavailable`` when the digest-addressed arrays are
        not staged in this process (rehydrated spec without its sidecar).
    """

    params = spec.params_dict()
    digest = str(params["basis_digest"])
    entry = _BASIS_STORE.get(digest)
    if entry is None:
        raise TransformContractError(
            f"project() basis {digest!r} is not staged in this process; a "
            "rehydrated spec carries the digest ADDRESS, never the arrays "
            "(artifact loading executes nothing).",
            code="transform_basis_unavailable",
            remedy=(
                "rebuild the spec via tl.transforms.project(basis, center) "
                "with the original arrays, or stage them from the fitted "
                "sidecar: pca_apply(tl.stats.load_fitted(path))"
            ),
            basis_digest=digest,
        )
    _BASIS_STORE.move_to_end(digest)
    if not tensor.is_floating_point():
        raise TransformContractError(
            f"project() consumes float tensors; got dtype {tensor.dtype}.",
            code="transform_plan_invalid",
            remedy="cast to a float dtype first (chain a cast step)",
            dtype=str(tensor.dtype),
        )
    if int(tensor.shape[-1]) != int(params["in_extent"]):
        raise TransformContractError(
            f"project() basis expects a feature extent of {params['in_extent']} "
            f"but this batch's last axis is {int(tensor.shape[-1])}.",
            code="transform_plan_invalid",
            remedy="match the fitted feature width (pool/flatten to it) or re-fit",
            expected=params["in_extent"],
            observed=int(tensor.shape[-1]),
        )
    acc = tensor.float() if tensor.dtype in _HALF_DTYPES else tensor
    basis, center = entry
    weights = basis.to(device=acc.device, dtype=acc.dtype)
    if center is not None:
        acc = acc - center.to(device=acc.device, dtype=acc.dtype)
    return acc @ weights.T


def project(basis: torch.Tensor, center: torch.Tensor | None = None) -> TransformSpec:
    """Freeze a linear projection: digest-addressed basis + optional center.

    Parameters
    ----------
    basis:
        ``(out_extent, in_extent)`` float component rows.
    center:
        Optional ``(in_extent,)`` vector subtracted before projecting.
        Uncentered apply leaves Euclidean RDMs exact but damages
        correlation-distance RDMs (r ~ 0.77 measured centered-vs-not).

    Returns
    -------
    TransformSpec
        The frozen project step (params carry digest + geometry, never
        values).
    """

    if not isinstance(basis, torch.Tensor) or basis.dim() != 2 or basis.numel() == 0:
        raise TransformContractError(
            f"project() needs a non-empty (out_extent, in_extent) basis tensor; "
            f"got {type(basis).__name__}"
            + (f" of shape {tuple(basis.shape)}" if isinstance(basis, torch.Tensor) else "")
            + ".",
            code="transform_params_invalid",
            remedy="pass a 2-D float tensor of component rows",
        )
    if not basis.is_floating_point() or not torch.isfinite(basis).all():
        raise TransformContractError(
            "project() basis must be float and fully finite.",
            code="transform_params_invalid",
            remedy="pass a finite float basis",
        )
    if center is not None and (
        not isinstance(center, torch.Tensor)
        or center.dim() != 1
        or int(center.numel()) != int(basis.shape[1])
        or not center.is_floating_point()
        or not torch.isfinite(center).all()
    ):
        raise TransformContractError(
            f"project() center must be a finite float ({int(basis.shape[1])},) "
            "vector matching the basis in_extent.",
            code="transform_params_invalid",
            remedy="pass the fit's feature-mean vector (or None)",
        )
    staged_basis = basis.detach().to("cpu", torch.float64).contiguous()
    staged_center = (
        None if center is None else center.detach().to("cpu", torch.float64).contiguous()
    )
    digest = _projection_digest(staged_basis, staged_center)
    _stage_basis(digest, staged_basis, staged_center)
    params = _project_normalize(
        {
            "basis_digest": digest,
            "in_extent": int(basis.shape[1]),
            "out_extent": int(basis.shape[0]),
            "centered": center is not None,
            "fit_scope": None,
            "source": None,
        }
    )
    return TransformSpec(name="project", version=1, params=freeze_params(params))


def pca_apply(fitted: Any) -> TransformSpec:
    """Apply a fitted PCA basis (thin validation lowering to ``project``).

    Parameters
    ----------
    fitted:
        A :class:`~torchlens.stats.FittedPCA` payload (from
        ``tl.stats.PCA(...).fitted(...)`` or ``tl.stats.load_fitted``).

    Returns
    -------
    TransformSpec
        A centered project step whose params record the fit scope and the
        fitted payload's content digest.
    """

    from ..stats._fitted import FittedPCA, fitted_digest

    if not isinstance(fitted, FittedPCA):
        raise TransformContractError(
            f"pca_apply() needs a tl.stats.FittedPCA payload; got "
            f"{type(fitted).__name__}. Fit-during-write stays REFUSED (a "
            "changing basis gives early and late shards different "
            "coordinates), so the payload comes from a completed fit.",
            code="transform_params_invalid",
            remedy=(
                "fit first (tl.stats.PCA(...).fitted(...)) or load a sidecar "
                "(tl.stats.load_fitted(path)), then pass the payload"
            ),
            value_type=type(fitted).__name__,
        )
    recomputed = fitted_digest(fitted.components, fitted.mean, fitted.explained_variance)
    if recomputed != fitted.digest:
        raise TransformContractError(
            f"pca_apply() payload records digest {fitted.digest!r} but its "
            f"arrays hash to {recomputed!r}; a mutated payload is refused, "
            "never silently applied.",
            code="transform_params_invalid",
            remedy="rebuild the payload from the fit or reload the sidecar",
            recorded=fitted.digest,
            recomputed=recomputed,
        )
    staged_basis = fitted.components.detach().to("cpu", torch.float64).contiguous()
    staged_center = fitted.mean.detach().to("cpu", torch.float64).contiguous()
    digest = _projection_digest(staged_basis, staged_center)
    _stage_basis(digest, staged_basis, staged_center)
    params = _project_normalize(
        {
            "basis_digest": digest,
            "in_extent": int(fitted.n_features),
            "out_extent": int(fitted.components.shape[0]),
            "centered": True,
            "fit_scope": fitted.fit_scope,
            "source": fitted.digest,
        }
    )
    return TransformSpec(name="project", version=1, params=freeze_params(params))


_register_builtin(
    TransformDefinition(
        name="project",
        version=1,
        normalize_params=_project_normalize,
        plan_fn=_project_plan,
        apply_fn=_project_apply,
        context_capable=False,
        stream_safe=True,
        zero_param_preset=False,
    )
)
